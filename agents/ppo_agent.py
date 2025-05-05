import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent

class PPONetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(PPONetwork, self).__init__()

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

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evalúa acciones dadas, devolviendo log_probs, valores y entropía."""
        action_logits, state_value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_log_probs, state_value, dist_entropy


class PPOAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any], clipping: bool):
        super().__init__(state_dim, action_dim, config)

        # Configuración
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.vf_coef = config.get('vf_coef', 0.5)  # Coeficiente para la pérdida del valor (value function)
        self.entropy_coef = config.get('entropy_coef', 0.05)  # Coeficiente para la pérdida de entropía
        self.n_steps = config.get('n_steps', 64) # Número de pasos por rollout
        self.batch_size = config.get('batch_size', 32)
        self.n_epochs = config.get('n_epochs', 3) # Número de épocas de optimización por rollout
        self.clipping = clipping

        # Red
        self.network = PPONetwork(state_dim, action_dim).to(self.device)

        # Optimizador
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.get('lr', 0.0003))

        # Memoria para el rollout actual
        self.memory = {'states': [], 'actions': [], 'rewards': [], 'dones': [], 'log_probs': [], 'values': []}
        self.memory_ready = False

    def select_action(self, state: np.ndarray) -> int:
        # Esta función se usa durante la recolección de datos
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, value = self.network(state_tensor)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Almacenar transición (sin next_state, se infiere)
        self.memory['states'].append(state)
        self.memory['actions'].append(action.item())
        self.memory['log_probs'].append(log_prob.item())
        self.memory['values'].append(value.item())
        # 'rewards' y 'dones' se añadirán después del step del entorno

        return action.item()

    def store_outcome(self, reward: float, done: bool):
        """Almacena la recompensa y el estado 'done' después de un paso."""
        if len(self.memory['rewards']) < len(self.memory['states']):
             self.memory['rewards'].append(reward)
             self.memory['dones'].append(done)
        else:
            print("Warning: Trying to store outcome without corresponding state/action.")

        # Comprobar si hemos recolectado suficientes pasos
        if len(self.memory['states']) >= self.n_steps:
            self.memory_ready = True


    def train(self, last_state: np.ndarray, last_done: bool) -> Dict[str, float]:
        if not self.memory_ready:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}

        # Preparación de datos del Rollout
        # Calcular el valor del último estado para GAE
        with torch.no_grad():
            last_value = 0.0 if last_done else self.network(torch.FloatTensor(last_state).unsqueeze(0).to(self.device))[1].item()

        # Calcular Ventajas y Retornos (Targets para Value Function) usando GAE
        advantages = np.zeros(len(self.memory['rewards']), dtype=np.float32)
        returns = np.zeros(len(self.memory['rewards']), dtype=np.float32)
        gae = 0.0
        values_np = np.array(self.memory['values'] + [last_value]) # Añadir el último valor

        for t in reversed(range(len(self.memory['rewards']))):
            delta = self.memory['rewards'][t] + self.gamma * values_np[t + 1] * (1 - self.memory['dones'][t]) - values_np[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.memory['dones'][t]) * gae
            advantages[t] = gae
            returns[t] = gae + values_np[t] # Retorno = Ventaja + Valor

        # Convertir datos a tensores
        states_tensor = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions_tensor = torch.LongTensor(self.memory['actions']).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalizar ventajas (importante)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Optimización en Múltiples Épocas
        num_samples = len(self.memory['states'])
        indices = np.arange(num_samples)
        all_policy_losses = []
        all_value_losses = []
        all_entropy_losses = []

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Extraer mini-batch
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Evaluar acciones con la red actual
                log_probs, values, entropy = self.network.evaluate_actions(batch_states, batch_actions)

                # Calcular ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Calcular pérdidas PPO-Clip
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Pérdida de valor
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)

                # Pérdida total
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy.mean()

                # Optimizar
                self.optimizer.zero_grad()
                loss.backward()
                if self.clipping: # Opcional: Clip gradiente
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy.mean().item())

        # Limpiar memoria para el siguiente rollout
        self.memory = {'states': [], 'actions': [], 'rewards': [], 'dones': [], 'log_probs': [], 'values': []}
        self.memory_ready = False

        return {
            'policy_loss': np.mean(all_policy_losses),
            'value_loss': np.mean(all_value_losses),
            'entropy_loss': np.mean(all_entropy_losses)
        }

    def save(self, path: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"PPO model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
        except:
            print("Could not load optimizer state, using default.")
        self.network.eval()
        print(f"PPO model loaded from {path}")