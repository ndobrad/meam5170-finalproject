import numpy as np
from acrobot import Acrobot
from environment import Environment
from controller_base import Controller, PathPlanner
from pydrake.all import MathematicalProgram, OsqpSolver

class ZeroController(Controller):
    def compute_feedback(self, current_x) -> np.ndarray:
        return np.zeros((self.n_u,))
    
    def update_target_state(self, x_desired):
        return super().update_target_state(x_desired)

class MPCController(Controller):
    def __init__(self, acrobot: Acrobot, Q, R, Qf) -> None:
        super().__init__(acrobot)
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.x_d = np.zeros(self.n_x)
        
    def update_target_state(self, x_desired):
        self.x_d = x_desired
    
    def compute_feedback(self, x_current) -> np.ndarray:
        # Parameters for the QP
        N = 10
        T = 0.1
        # Initialize mathematical program and decalre decision variables
        prog = MathematicalProgram()
        x = np.zeros((N, 6), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(6, "x_" + str(i))
        u = np.zeros((N-1, 2), dtype="object")
        for i in range(N-1):
            u[i] = prog.NewContinuousVariables(2, "u_" + str(i))
        # Add constraints and cost
        self._add_initial_state_constraint(prog, x, x_current)
        self._add_input_saturation_constraint(prog, x, u, N)
        self._add_dynamics_constraint(prog, x, u, N, T)
        self._add_cost(prog, x, u, N)
        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)
        u_mpc = np.zeros(self.n_u)
        if result.is_success():
            u_mpc = np.array(result.GetSolution(u[0]))
            u_mpc += self.u_d()
        else:
            u_mpc = np.zeros(self.nu)
        return u_mpc
    
    def _add_initial_state_constraint(self, prog:MathematicalProgram, x, x_current):
        lb = x_current
        ub = x_current
        prog.AddBoundingBoxConstraint(lb, ub, x[0])
    
    def _add_input_saturation_constraint(self, prog:MathematicalProgram, x, u, N):
        lb = np.array([self.acrobot.umin]) 
        ub = np.array([self.acrobot.umax])
        for i in range(N-1):
            prog.AddBoundingBoxConstraint(lb, ub, u[i])
            
    def _add_dynamics_constraint(self, prog:MathematicalProgram, x, u, N, T):
        A, B = self.acrobot.discrete_time_linear_dynamics(T)
        for k in range(N-1):
            expr = x[k+1] - (A @ x[k] + B @ u[k])
            prog.AddLinearEqualityConstraint(expr, np.zeros(self.n_x))           

    def _add_cost(self, prog:MathematicalProgram, x, u, N):
        # Need to change this
        Qk = np.block([[self.Q, np.zeros((self.n_x,self.n_u))], [np.zeros((self.n_u,self.n_x)), self.R]])
        # desired_var = np.concatenate([self.x_d(), self.u_d()])
        for i in range(N-1):
            decision_var = np.concatenate([x[i], u[i]])
            prog.AddQuadraticCost((decision_var).T @ Qk @ (decision_var))
        prog.AddQuadraticCost((x[N - 1] - self.x_d).T @ self.Qf @ (x[N - 1] - self.x_d))
        

            
class TrivialPathPlanner(PathPlanner):
    """
    'Plans' a path from the initial position to the first Hold 
    in the environment, which should be within reach
    """
    def calculate_path(self, env: Environment) -> np.ndarray:
        return np.array(env.holds[0])
    
    
    
