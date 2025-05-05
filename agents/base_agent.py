from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple

class BaseAgent(ABC):
    """Clase base abstracta para todos los agentes de RL"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Inicializa el agente base
        
        Args:
            state_dim: Dimensión del espacio de estados
            action_dim: Dimensión del espacio de acciones
            config: Diccionario con la configuración del agente
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """
        Selecciona una acción basada en el estado actual
        
        Args:
            state: Estado actual del entorno
            
        Returns:
            int: Acción seleccionada
        """
        pass
    
    @abstractmethod
    def train(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Entrena el agente con un batch de experiencias
        
        Args:
            batch: Diccionario con las experiencias para entrenar
            
        Returns:
            Dict[str, float]: Métricas de entrenamiento
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """
        Guarda el modelo del agente
        
        Args:
            path: Ruta donde guardar el modelo
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        Carga el modelo del agente
        
        Args:
            path: Ruta desde donde cargar el modelo
        """
        pass 