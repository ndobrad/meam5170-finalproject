import numpy as np
from abc import ABC, abstractmethod
from environment import Environment
from acrobot import Acrobot
from pydrake.all import PiecewisePolynomial

class Controller(ABC):
    def __init__(self, acrobot:Acrobot) -> None:
        super().__init__()
        self.n_u = acrobot.n_u
        self.n_x = acrobot.n_x
        self.acrobot = acrobot
        self.goal_pos = None
        
    @abstractmethod
    def compute_feedback(self, current_x) -> np.ndarray:
        pass
    
    @abstractmethod
    def update_target_state(self, goal_pos):
        """
        goal_pos is the (x,y) position of the target hold
        """
        self.goal_pos = goal_pos
        
    @abstractmethod
    def update_target_trajectory(self, x_traj:PiecewisePolynomial, u_traj:PiecewisePolynomial):
        """
        goal_pos is the (x,y) position of the target hold
        """
        pass
    
    
    
class PathPlanner(ABC):
    def __init__(self, env:Environment) -> None:
        super().__init__()
        self.env = env
        
    @abstractmethod
    def calculate_path(self) -> np.ndarray:
        """
        Returns an array of indices into the Environment's list of Holds, 
        in the order they should be visited, starting with the Hold the Acrobot
        is initially attached to.
        """
        pass
    
    
class TrajectoryOptimizer(ABC):
    def __init__(self, acrobot:Acrobot) -> None:
        self.acrobot = acrobot
        super().__init__()
        
    @abstractmethod
    def generate_trajectory(self, initial_state, goal_hold_pos) -> np.ndarray:
        pass