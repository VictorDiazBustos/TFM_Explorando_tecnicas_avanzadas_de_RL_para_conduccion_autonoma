import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent
import threading
import time
import copy
import random

# --- A3CNetwork (sin cambios) ---
class A3CNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(A3CNetwork, self).__init__()

        # Capas compartidas
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Actor (policy) - Salida de logits para Categorical
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_logits, state_value

class A3CWorker(threading.Thread):
    def __init__(self, master_agent, worker_id, env_creator, config, global_optimizer, lock, clipping: bool):
        super().__init__(daemon=True)
        self.master_agent = master_agent
        self.worker_id = worker_id
        self.env = env_creator()
        self.config = config
        self.global_optimizer = global_optimizer
        self.lock = lock
        self.clipping = clipping

        # Red local
        self.local_network = A3CNetwork(
            state_dim=config['state_dim'],
            action_dim=config['action_dim']
        ).to(master_agent.device)
        self.local_network.load_state_dict(master_agent.network.state_dict())

        # Buffer de experiencias para n pasos
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        self.n_steps = config.get('n_steps', 20) # Número de pasos antes de actualizar

    def run(self):
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0

        while not self.master_agent.training_done:
            # Sincronizar pesos locales con los globales al inicio de cada rollout
            self.local_network.load_state_dict(self.master_agent.network.state_dict())

            # Limpiar buffer para el nuevo rollout
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.values.clear()
            self.log_probs.clear()
            self.dones.clear()

            # Recolectar n_steps o hasta el final del episodio
            for _ in range(self.n_steps):
                episode_steps += 1
                # Seleccionar acción usando la red local
                action, value, log_prob = self.select_action(state)

                # Ejecutar acción
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Guardar experiencia
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.values.append(value)
                self.log_probs.append(log_prob)
                self.dones.append(done)

                state = next_state

                if done:
                    print(f"Worker {self.worker_id}: Episode finished. Reward: {episode_reward:.2f}, Steps: {episode_steps}")
                    state = self.env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    # Si termina antes de n_steps, salimos para actualizar
                    break

            # Calcular valor del último estado para el bootstrap
            if self.dones[-1]: # Si el último estado fue terminal
                R = torch.tensor([[0.0]], dtype=torch.float32).to(self.master_agent.device)
            else:
                with torch.no_grad():
                    _, last_value = self.local_network(torch.FloatTensor(state).unsqueeze(0).to(self.master_agent.device))
                    R = last_value

            # Calcular ventajas y retornos (targets para el crítico)
            returns = self.compute_returns(R)
            values_tensor = torch.cat(self.values)
            advantages = returns - values_tensor.squeeze() # Ventaja simple A(s,a) = R - V(s)

            # Entrenar y actualizar red global
            self.update_global(advantages, returns)

            # Pequeña pausa para evitar busy-waiting extremo si es necesario
            # time.sleep(0.001)

        self.env.close() # Cerrar el entorno del worker al terminar
        print(f"Worker {self.worker_id} finished.")


    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.master_agent.device)
        # No necesitamos no_grad aquí porque los gradientes se usarán para la actualización
        action_logits, value = self.local_network(state_tensor)

        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Devolvemos tensores para facilitar cálculos posteriores
        return action.item(), value, log_prob

    def compute_returns(self, bootstrap_value: torch.Tensor) -> torch.Tensor:
        returns = torch.zeros(len(self.rewards) + 1, 1).to(self.master_agent.device)
        returns[-1] = bootstrap_value # Valor del último estado (o 0 si fue terminal)

        gamma = self.config['gamma']
        for t in reversed(range(len(self.rewards))):
            returns[t] = self.rewards[t] + gamma * returns[t+1] * (1 - self.dones[t]) # Considerar dones

        return returns[:-1] # Devolver retornos para cada paso, excluyendo el bootstrap

    def update_global(self, advantages: torch.Tensor, returns: torch.Tensor):
        # Normalizar ventajas (opcional pero recomendado)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convertir listas a tensores
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.master_agent.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.master_agent.device)
        # log_probs_tensor = torch.cat(self.log_probs) # Ya son tensores

        # Calcular pérdidas usando la red local actual (antes de la actualización global)
        action_logits, values = self.local_network(states_tensor)
        dist = Categorical(logits=action_logits)

        # Pérdida del actor (policy gradient)
        actor_loss = -(dist.log_prob(actions_tensor) * advantages.detach()).mean() # Usar ventajas calculadas

        # Pérdida del crítico (MSE)
        critic_loss = nn.MSELoss()(values.squeeze(), returns.squeeze().detach()) # Target es el retorno calculado

        # Pérdida de entropía (para fomentar exploración)
        entropy_loss = -dist.entropy().mean()

        # Pérdida total
        c1 = self.config.get('value_loss_coef', 0.5)
        c2 = self.config.get('entropy_coef', 0.01)
        total_loss = actor_loss + c1 * critic_loss + c2 * entropy_loss

        # Calcular gradientes en la red LOCAL
        self.local_network.zero_grad()
        total_loss.backward()
        
        # Opcional: Clipping de gradientes locales antes de transferir
        if self.clipping:
            torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.config.get('max_grad_norm', 40.0))

        # Transferir gradientes y actualizar global
        with self.lock:
            self.global_optimizer.zero_grad()

            # Copiar gradientes de la red local a la red global
            for local_param, global_param in zip(self.local_network.parameters(), self.master_agent.network.parameters()):
                if local_param.grad is not None:
                    # Si el parámetro global no tiene gradiente inicializado (raro después de zero_grad)
                    if global_param.grad is None:
                         global_param.grad = local_param.grad.clone().to(global_param.device)
                    else:
                        # Copiar los datos del gradiente
                        global_param.grad.data.copy_(local_param.grad.data)
                else: # Si el local no tiene gradiente, asegurar que el global tampoco (o es cero)
                    if global_param.grad is not None:
                        global_param.grad.data.zero_()


            # Realizar el paso de optimización en la red GLOBAL usando los gradientes transferidos
            self.global_optimizer.step()

        # Después de la actualización global, sincronizar la red local (opcional pero a menudo hecho al inicio del siguiente rollout)
        self.local_network.load_state_dict(self.master_agent.network.state_dict())


class A3CAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any], clipping: bool, env_creator):
        # env_creator es una función que crea una nueva instancia del entorno, ej: lambda: SUMOEnvironment(...)
        super().__init__(state_dim, action_dim, config)

        # Configuración
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_workers = config.get('n_workers', 4)
        self.config['state_dim'] = state_dim # Asegurar que esté en config para workers
        self.config['action_dim'] = action_dim

        # Red global compartida (asegurar que es compartida entre hilos si es necesario, aunque Adam suele ser robusto)
        self.network = A3CNetwork(state_dim, action_dim).to(self.device)
        # self.network.share_memory() # Si se usan procesos en lugar de hilos
        self.clipping = clipping

        # Optimizador global COMPARTIDO
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.get('lr', 0.0001)
        )

        # Lock para sincronizar actualizaciones del optimizador global
        self.lock = threading.Lock()

        # Función para crear entornos
        self.env_creator = env_creator

        # Estado de entrenamiento
        self.training_done = False

    def start_training(self):
        """Inicia los workers de A3C"""
        self.workers = []
        self.training_done = False
        base_port = 8813
        print(f"Starting {self.n_workers} workers...")
        for i in range(self.n_workers):
            worker_port = base_port + i + random.randint(0, 5) # Añadir offset y algo de aleatoriedad
            print(f"\tAssigning port {worker_port} to worker {i}")

            worker_env_creator = lambda p=worker_port: self.env_creator(gui_flag=False, port=p)

            # Pasamos el optimizador global y el lock a cada worker
            worker = A3CWorker(self, i, worker_env_creator, self.config, self.optimizer, self.lock, self.clipping)
            worker.start()
            self.workers.append(worker)
        print("Workers started.")

    def stop_training(self):
        """Detiene a los workers y espera a que terminen."""
        print("Stopping workers...")
        self.training_done = True
        for worker in self.workers:
            worker.join() # Esperar a que el hilo termine
        print("All workers stopped.")


    def select_action(self, state: np.ndarray) -> int:
        # Esta función es principalmente para evaluación, usa la red global
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, _ = self.network(state)
            action = Categorical(logits=action_logits).sample()
            return action.item()

    def train(self) -> Dict[str, float]:
        # A3C se entrena de forma asíncrona a través de sus workers.
        # Esta función podría usarse para reportar métricas agregadas si se implementa.
        print("A3C training happens in worker threads. Call start_training() / stop_training().")
        return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy_loss': 0.0} # Placeholder

    def save(self, path: str):
        # Guardar solo la red global y el estado del optimizador global
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"A3C model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        # No cargar el estado del optimizador directamente si se reinicia el entrenamiento,
        # pero es útil si se continúa.
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
        except:
            print("Could not load optimizer state, using default.")
        self.network.eval() # Poner en modo evaluación por defecto después de cargar
        print(f"A3C model loaded from {path}")