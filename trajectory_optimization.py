import numpy as np
from acrobot import Acrobot
from environment import Environment
from controller_base import TrajectoryOptimizer
from pydrake.all import MathematicalProgram, OsqpSolver, DirectCollocation

class DirectCollocationTrajectoryGenerator(TrajectoryOptimizer):
    def __init__(self, acrobot:Acrobot) -> None:
        super().__init__(acrobot)
        
        
    def generate_trajectory(self) -> np.ndarray:
        
        # TODO: initialize acrobot context? create new context and initialize it
        
        collocation_prog = DirectCollocation(
            self.acrobot.plant,
            # self.acrobot.plant.CreateDefaultContext(),
            num_time_samples=n,
            minimum_time_step=,
            maximum_time_step=,
            input_port_index=self.acrobot.input_port
            
        )
        
        
        return super().generate_trajectory()