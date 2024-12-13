import numpy as np
from acrobot import Acrobot
from environment import Environment
from controller_base import TrajectoryOptimizer
from pydrake.all import (MathematicalProgram, OsqpSolver, DirectCollocation, 
                         PiecewisePolynomial, Solve, MathematicalProgramResult)

class DirectCollocationParameters:
    def __init__(self) -> None:
        self.time_samples = 20
        self.torque_limit = 9
        self.joint_limits = None
        self.minimum_time_step = 0.1
        self.maximum_time_step = 0.8
        self.time_cost = 0
        self.Q = np.array([[ 1,  0, 0, 0],
                           [ 0,  1, 0, 0],
                           [ 0,  0, 1, 0],
                           [ 0,  0, 0, 1]])
        self.Qf = self.Q
        self.R = 1
        self.goal_speed_limit = 100



class DirectCollocationTrajectoryGenerator(TrajectoryOptimizer):
    def __init__(self, acrobot:Acrobot, params=DirectCollocationParameters()) -> None:
        super().__init__(acrobot)
        self.collocation_prog = None
        self.params = params
        
    def set_parameters(self, params:DirectCollocationParameters):
        self.params = params
        
    def generate_trajectory(self, initial_state, goal_hold_pos) -> MathematicalProgramResult:
                
        # goal_state = self.acrobot.get_joint_angles(goal_hold_pos,(goal_hold_pos[0] > 0))
        goal_state = np.hstack([self.acrobot.get_joint_angles(goal_hold_pos,(goal_hold_pos[0] > 0)),np.zeros(2)])
        # goal_state = np.hstack([self.acrobot.get_joint_angles(goal_hold_pos,True),np.zeros(2)])
        
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
        
        
        # self.collocation_prog.AddEqualTimeIntervalsConstraints()
        
        #initial input is 0, initial state is initial_state
        # later, this may need to be changed to be the u when the reset map is hit
        self.collocation_prog.AddConstraintToAllKnotPoints(u_init[0] == 0)
        self.collocation_prog.prog().AddBoundingBoxConstraint(
            initial_state,initial_state, self.collocation_prog.initial_state()
        )
        goal_ub = np.copy(goal_state)
        goal_ub[2] = self.params.goal_speed_limit
        goal_ub[3] = self.params.goal_speed_limit
        goal_lb = np.copy(goal_state)
        goal_lb[2] = -self.params.goal_speed_limit
        goal_lb[3] = -self.params.goal_speed_limit
        self.collocation_prog.prog().AddBoundingBoxConstraint(
            goal_lb,goal_ub, self.collocation_prog.final_state()
        )
                
        #torque limits
        
        self.collocation_prog.AddConstraintToAllKnotPoints(-self.params.torque_limit <= u[0])
        self.collocation_prog.AddConstraintToAllKnotPoints(u[0] <= self.params.torque_limit)
        
        #speed limit?
        
        #costs
        self.collocation_prog.AddRunningCost(u[0] * self.params.R * u[0])
        self.collocation_prog.AddRunningCost(x @ self.params.Q @ x)
        
        self.collocation_prog.AddFinalCost(self.collocation_prog.time() * self.params.time_cost)
        
        
        x_guess = PiecewisePolynomial.FirstOrderHold([0, 3],np.column_stack((initial_state,goal_state)))
        
        self.collocation_prog.SetInitialTrajectory(PiecewisePolynomial(), x_guess)
        
        sol = Solve(self.collocation_prog.prog())
        # assert sol.is_success(), sol.GetInfeasibleConstraintNames(self.collocation_prog.prog())
        
        
        return sol
        
        
        
        