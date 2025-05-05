import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Any, Tuple
from collections import deque
import random
from .base_agent import BaseAgent

class DiscreteActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DiscreteActor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Salida de Logits para cada acción
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        probs = dist.probs # Probabilidades para el cálculo de la Q objetivo
        return action, log_prob, probs

class DiscreteCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DiscreteCritic, self).__init__()

        # Red Q1
        self.network1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Salida: Q-value para cada acción
        )

        # Red Q2 (Twin Critic)
        self.network2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Salida: Q-value para cada acción
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.network1(state)
        q2 = self.network2(state)
        return q1, q2

    def Q1(self, state: torch.Tensor) -> torch.Tensor:
        return self.network1(state)

class SACAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(state_dim, action_dim, config)

        # Configuración
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha = config.get('alpha', 0.2) # Alpha (temperatura de entropía) - puede ser fija o aprendida
        self.learn_alpha = config.get('learn_alpha', True)
        self.target_update_interval = config.get('target_update_interval', 1)
        self.batch_size = config.get('batch_size', 256)
        self.buffer_size = config.get('buffer_size', 1000000)

        # Redes
        self.actor = DiscreteActor(state_dim, action_dim).to(self.device)
        self.critic = DiscreteCritic(state_dim, action_dim).to(self.device)

        # Redes objetivo
        self.target_critic = DiscreteCritic(state_dim, action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        # Optimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.get('actor_lr', 0.0003))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.get('critic_lr', 0.0003))

        # Alpha aprendido (si se configura)
        if self.learn_alpha:
            # Entropía objetivo
            self.target_entropy = -torch.tensor(action_dim, dtype=torch.float32).to(self.device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.get('alpha_lr', 0.0003))
            self.alpha = self.log_alpha.exp().item()

        # Buffer de experiencias
        self.memory = deque(maxlen=self.buffer_size)

        # Contador de pasos (usado para la actualización suave)
        self.steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Selecciona una acción. Si evaluate=True, toma la más probable."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state)
            if evaluate:
                action = torch.argmax(logits, dim=1) # Acción determinista
            else:
                dist = Categorical(logits=logits)
                action = dist.sample() # Acción estocástica
        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        """Almacena una transición en la memoria."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self) -> Dict[str, float]:
        if len(self.memory) < self.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha_loss': 0.0, 'alpha': self.alpha}

        # Obtener batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).unsqueeze(1).to(self.device)

        # Actualizar Críticos
        with torch.no_grad():
            # Obtener acciones, log_probs y probs para el *siguiente* estado desde el actor *actual*
            next_logits = self.actor(next_states)
            next_dist = Categorical(logits=next_logits)
            next_action_probs = next_dist.probs
            next_action_log_probs = torch.log(next_action_probs + 1e-8)

            # Obtener Q-values del crítico *objetivo* para el siguiente estado
            q1_target_next, q2_target_next = self.target_critic(next_states)
            min_q_target_next = torch.min(q1_target_next, q2_target_next) # [batch_size, action_dim]

            # Calcular el valor V(s') esperado bajo la política actual (con entropía)
            # V(s') = E_{a'~pi}[ Q_target(s', a') - alpha * log pi(a'|s') ]
            # V(s') = sum_a' [ pi(a'|s') * ( Q_target(s', a') - alpha * log pi(a'|s') ) ]
            next_value = torch.sum(next_action_probs * (min_q_target_next - self.alpha * next_action_log_probs), dim=1, keepdim=True)

            # Calcular el target para Q(s, a)
            q_target = rewards + (1.0 - dones) * self.gamma * next_value

        # Obtener Q-values actuales para la acción tomada (s, a)
        current_q1, current_q2 = self.critic(states)
        
        # Seleccionar Q para la acción específica que se tomó
        current_q1 = current_q1.gather(1, actions)
        current_q2 = current_q2.gather(1, actions)

        # Calcular pérdida de los críticos
        critic1_loss = F.mse_loss(current_q1, q_target)
        critic2_loss = F.mse_loss(current_q2, q_target)
        critic_loss = critic1_loss + critic2_loss

        # Optimizar críticos
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actualizar Actor
        # Congelar gradientes de los críticos para la actualización del actor
        for p in self.critic.parameters():
            p.requires_grad = False

        # Calcular log_probs y probs para el estado *actual*
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        action_probs = dist.probs
        action_log_probs = torch.log(action_probs + 1e-8)

        # Obtener Q-values del crítico actual
        q1_actor, q2_actor = self.critic(states)
        min_q_actor = torch.min(q1_actor, q2_actor)

        # Calcular pérdida del actor: maximizar E_{s~D, a~pi}[ alpha * log pi(a|s) - Q(s,a) ]
        # Equivalente a minimizar E_{s~D}[ sum_a [ pi(a|s) * (alpha * log pi(a|s) - Q(s,a)) ] ]
        actor_loss = torch.sum(action_probs * (self.alpha * action_log_probs - min_q_actor.detach()), dim=1).mean()

        # Optimizar actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Descongelar gradientes de los críticos
        for p in self.critic.parameters():
            p.requires_grad = True

        # Actualizar Alpha (si se aprende)
        alpha_loss = torch.tensor(0.0)
        if self.learn_alpha:
            # Usamos los log_probs calculados durante la actualización del actor
            # Alpha loss = E_{s~D, a~pi}[ -log_alpha * (log pi(a|s) + target_entropy) ]
            alpha_loss = -(self.log_alpha * (action_log_probs.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()


        # Actualizar Redes Objetivo (Suave)
        self.steps += 1
        self._soft_update(self.critic, self.target_critic)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss,
            'alpha': self.alpha
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Actualización suave de las redes objetivo"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str):
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'steps': self.steps,
            'alpha': self.alpha
        }
        if self.learn_alpha:
            save_dict['log_alpha_state_dict'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()

        torch.save(save_dict, path)
        print(f"SAC model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.alpha = checkpoint.get('alpha', self.alpha)

        if self.learn_alpha and 'log_alpha_state_dict' in checkpoint:
            self.log_alpha.data = checkpoint['log_alpha_state_dict'].data
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp().item()
            print("Learned alpha state loaded.")
        elif self.learn_alpha:
            print("Warning: Trying to load learned alpha state, but not found in checkpoint.")


        self.target_critic.eval()
        self.actor.eval()
        self.critic.eval()
        print(f"SAC model loaded from {path}. Current alpha: {self.alpha:.4f}")