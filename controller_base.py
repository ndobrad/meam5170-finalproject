import numpy as np
from abc import ABC, abstractmethod
from environment import Environment
from acrobot import Acrobot

class Controller(ABC):
    def __init__(self, acrobot:Acrobot) -> None:
        super().__init__()
        self.n_u = acrobot.n_u
        self.n_x = acrobot.n_x
        self.acrobot = acrobot
        
    @abstractmethod
    def compute_feedback(self, current_x) -> np.ndarray:
        pass
    
    @abstractmethod
    def update_target_state(self, x_desired):
        pass
    
    
    
class PathPlanner(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def calculate_path(self, env:Environment) -> np.ndarray:
        pass