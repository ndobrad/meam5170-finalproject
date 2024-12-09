import numpy as np
from acrobot import Acrobot
from environment import Environment
from controller_base import TrajectoryOptimizer
from pydrake.all import (MathematicalProgram, OsqpSolver, DirectCollocation, 
                         PiecewisePolynomial, Solve, MathematicalProgramResult)

class DirectCollocationParameters:
    def __init__(self) -> None:
        self.time_samples = None
        self.torque_limit = 10
        self.joint_limits = None
        self.minimum_time_step = 0.1
        self.maximum_time_step = 0.8
        self.time_cost = 0
        self.Q = np.array([[10,  0, 0, 0],
                           [ 0, 10, 0, 0],
                           [ 0,  0, 1, 0],
                           [ 0,  0, 0, 1]])
        self.Qf = self.Q
        self.R = 1



class DirectCollocationTrajectoryGenerator(TrajectoryOptimizer):
    def __init__(self, acrobot:Acrobot, params) -> None:
        super().__init__(acrobot)
        self.collocation_prog = None
        self.params = params
        
    def set_parameters(self, params:DirectCollocationParameters):
        self.params = params
        
    def generate_trajectory(self, initial_state, goal_state) -> MathematicalProgramResult:
        
        # TODO: initialize acrobot context? create new context and initialize it
        context = self.acrobot.plant.CreateDefaultContext()
        self.collocation_prog = DirectCollocation(
            self.acrobot.plant,
            context,
            num_time_samples=self.params.time_samples,
            minimum_time_step=self.params.minimum_time_step,
            maximum_time_step=self.params.maximum_time_step,
            input_port_index=self.acrobot.input_port,
            
        )
        u_init = self.collocation_prog.input(0) # u at t = 0
        u = self.collocation_prog.input() # u at t = indef
        x = self.collocation_prog.state() 
        
        
        self.collocation_prog.AddEqualTimeIntervalsConstraints()
        
        #initial input is 0, initial state is initial_state
        # later, this may need to be changed to be the u when the reset map is hit
        self.collocation_prog.AddConstraintToAllKnotPoints(u_init[0] == 0)
        self.collocation_prog.prog().AddBoundingBoxConstraint(
            initial_state,initial_state, self.collocation_prog.initial_state()
        )
        self.collocation_prog.prog().AddBoundingBoxConstraint(
            goal_state,goal_state, self.collocation_prog.final_state()
        )
                
        #torque limits
        
        self.collocation_prog.AddConstraintToAllKnotPoints(-self.params.torque_limit <= u[0])
        self.collocation_prog.AddConstraintToAllKnotPoints(u[0] <= self.params.torque_limit)
        
        #speed limit?
        
        #costs
        self.collocation_prog.AddRunningCost(u[0] * self.params.R * u[0])
        self.collocation_prog.AddRunningCost(x @ self.params.Q @ x)
        
        self.collocation_prog.AddFinalCost(self.collocation_prog.time() * self.params.time_cost)
        
        
        x_guess = PiecewisePolynomial.FirstOrderHold([0, 1],np.column_stack((initial_state,goal_state)))
        
        self.collocation_prog.SetInitialTrajectory(PiecewisePolynomial(), x_guess)
        
        sol = Solve(self.collocation_prog.prog())
        assert sol.is_success()
        
        
        return sol
        
        
        
        