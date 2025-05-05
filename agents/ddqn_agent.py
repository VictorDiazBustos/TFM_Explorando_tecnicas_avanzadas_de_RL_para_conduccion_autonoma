import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, Any
from .base_agent import BaseAgent

class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DuelingDQN, self).__init__()
        
        # Capas de características
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Capa de valor
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Capa de ventaja
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combinar valor y ventaja
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class DDQNAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any], clipping: bool):
        super().__init__(state_dim, action_dim, config)
        
        # Configuración
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update = config.get('target_update', 10)
        self.batch_size = config.get('batch_size', 64)
        self.memory_size = config.get('memory_size', 10000)
        self.clipping = clipping
        
        # Redes
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizador y memoria
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.get('lr', 0.001))
        self.memory = deque(maxlen=self.memory_size)
        
        # Contador de pasos para actualización del target
        self.steps = 0
        
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        # Almacena una transición en la memoria
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self) -> Dict[str, float]:
        if len(self.memory) < self.batch_size:
            return {'loss': 0.0}
            
        # Obtener batch de memoria
        batch = random.sample(self.memory, self.batch_size)
        # TODO: eliminar comentarios tras validar version
        # states = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        # actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        # rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        # next_states = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        # dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)
        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).unsqueeze(1).to(self.device)
        
        # Calcular Q-values actuales
        # policy_net(states) devuelve Q(s, a) para todas las 'a'
        # .gather(1, actions.unsqueeze(1)) selecciona Q(s, action_tomada)
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Calcular Q-values objetivo usando la red objetivo
        with torch.no_grad():
            # 1. Seleccionar la mejor acción para next_states usando la policy_net
            # policy_net(next_states) -> [batch_size, action_dim]
            # .argmax(1) -> [batch_size] (índice de la mejor acción)
            # .unsqueeze(1) -> [batch_size, 1] (para usar en gather)
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            	
            # 2. Evaluar esas acciones seleccionadas usando la target_net
            # target_net(next_states) -> [batch_size, action_dim]
            # .gather(1, next_actions) -> [batch_size, 1] (Q-value de la acción seleccionada por policy_net)
            # .squeeze() -> [batch_size]
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            
            # Calcular el valor objetivo final
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Calcular pérdida
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        if self.clipping:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Actualizar red objetivo si es necesario
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Actualizar epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def save(self, path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.target_net.eval()
