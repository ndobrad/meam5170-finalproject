import numpy as np
from acrobot import Acrobot
from environment import Environment
from controller_base import Controller, PathPlanner
from pydrake.all import MathematicalProgram, OsqpSolver
from pydrake.solvers import Solve

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
        self.u_d = np.zeros(self.n_u)
        self.last_u = np.zeros(self.n_u)
    
    def update_target_state(self, goal_pos):
        super().update_target_state(goal_pos)
        self.x_d = np.hstack([self.acrobot.get_joint_angles(goal_pos),np.zeros(2)])
    
    def compute_feedback(self, current_x):
        # Parameters for the QP
        N = 10
        T = 0.1
        # Initialize mathematical program and decalre decision variables
        prog = MathematicalProgram()
        x = np.zeros((N, self.n_x), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(self.n_x, "x_" + str(i))
        u = np.zeros((N-1, self.n_u), dtype="object")
        for i in range(N-1):
            u[i] = prog.NewContinuousVariables(self.n_u, "u_" + str(i))
        # Add constraints and cost
        self._add_initial_state_constraint(prog, x, current_x)
        self._add_input_saturation_constraint(prog, u, N)
        self._add_dynamics_constraint(prog, x, u, N, T, current_x)
        self._add_cost(prog, x, u, N)
        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)
        u_mpc = np.zeros(self.n_u)
        if result.is_success():
            u_mpc = np.array(result.GetSolution(u[0]))
            u_mpc += self.u_d
            self.last_u = u_mpc
        else:
            u_mpc = np.zeros(self.nu)
        return u_mpc

    def _add_initial_state_constraint(self, prog:MathematicalProgram, x, x_current):
        lb = x_current
        ub = x_current
        prog.AddBoundingBoxConstraint(lb, ub, x[0])

    def _add_input_saturation_constraint(self, prog:MathematicalProgram, u, N):
        lb = np.array([self.acrobot.umin]) 
        ub = np.array([self.acrobot.umax])
        for i in range(N-1):
            prog.AddBoundingBoxConstraint(lb, ub, u[i])

    def _add_dynamics_constraint(self, prog:MathematicalProgram, x, u, N, T, x_current):
        #for now, linearize around the current state and last input. Later, update this to 
        #linearize around a trajectory. Maybe the trajectory could be saved to self.trajectory
        # in a class method like update_target_trajectory(self, new_traj) or something.
        #Then, put the next line here inside the for loop and pass it:
        # new_traj.x[k+(some kind of index into trajectory)], new_traj.u[k+(same index)]
                
        for k in range(N-1):
            #inputs to linearization SHOULD be x,u decision variables. How to make this work?
            A, B = self.acrobot.discrete_time_linear_dynamics(T, x_current, self.last_u)
            expr = x[k+1] - (A @ x[k] + B @ u[k])
            prog.AddLinearEqualityConstraint(expr, np.zeros(self.n_x))

    def _add_cost(self, prog:MathematicalProgram, x, u, N):
        for i in range(N-1):
            # prog.AddQuadraticCost((x[i] - self.x_d).T @ self.Q @ (x[i] - self.x_d))
            # prog.AddQuadraticCost((u[i] - self.u_d) * (self.R * (u[i] - self.u_d)))
            prog.AddQuadraticCost(self.Q, np.zeros(self.n_x), x[i])
            prog.AddQuadraticCost(self.R, np.zeros(self.n_u), u[i])
        prog.AddQuadraticCost(self.Qf, np.zeros(self.n_x), x[N-1])


''' TODO: Modification Needed for Non-Linear MOC formulation'''
class NMPCController(Controller):
    def __init__(self, acrobot: Acrobot, Q, R, Qf) -> None:
        super().__init__(acrobot)
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.x_d = np.zeros(self.n_x)
        self.u_d = np.zeros(self.n_u)
        
    def update_target_state(self, x_desired):
        self.x_d = x_desired
    
    def compute_feedback(self, current_x):
        # Parameters for the QP
        N = 10
        T = 0.1
        # Initialize mathematical program and decalre decision variables
        prog = MathematicalProgram()
        x = np.zeros((N, self.n_x), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(self.n_x, "x_" + str(i))
        u = np.zeros((N-1), dtype="object")
        for i in range(N-1):
            u[i] = prog.NewContinuousVariables(self.n_u, "u_" + str(i))
        # Add constraints and cost
        self._add_initial_state_constraint(prog, x, x_current)
        self._add_input_saturation_constraint(prog, x, u, N)
        # self._add_dynamics_constraint(prog, x, u, N, T)
        self._add_nonlinear_dynamics_constraint(prog, x, u, N, T)
        self._add_cost(prog, x, u, N)
        # Solve the QP
        # solver = OsqpSolver()
        # result = solver.Solve(prog)
        result = Solve(prog)
        u_mpc = np.zeros(self.n_u)
        if result.is_success():
            # u_mpc = np.array(result.GetSolution(u[0]))
            # u_mpc += self.u_d()
            u_mpc = result.GetSolution(u[0])
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
        # A, B = self.acrobot.continuous_time_full_dynamics(x, u)
        for k in range(N-1):
            x_operating_point = x[k]
            u_operating_point = u[k]
            A, B = self.acrobot.continuous_time_linear_dynamics(x_operating_point, u_operating_point)
            Ad = np.identity(self.n_x) + A * T
            Bd = B * T
            expr = x[k+1] - (Ad @ x[k] + Bd @ u[k])
            prog.AddLinearEqualityConstraint(expr, np.zeros(self.n_x))  

    def _add_nonlinear_dynamics_constraint(self, prog, x, u, N, T):
        def dynamics_constraint(vars, T, acrobot):
            x_k = vars[:acrobot.n_x]
            u_k = vars[acrobot.n_x:acrobot.n_x+acrobot.n_u]
            x_next = vars[acrobot.n_x+acrobot.n_u:]
            x_dot = acrobot.continuous_time_full_dynamics(x_k, u_k)
            x_next_pred = x_k + T * x_dot
            return x_next_pred - x_next

        for k in range(N-1):
            prog.AddConstraint(
                lambda vars, T=T, acrobot=self.acrobot: dynamics_constraint(vars, T, acrobot),
                lb=np.zeros(self.n_x),
                ub=np.zeros(self.n_x),
                vars=np.concatenate([x[k], u[k], x[k+1]])
            )

    def _add_cost(self, prog:MathematicalProgram, x, u, N):
        # Need to change this
        # Qk = np.block([[self.Q, np.zeros((self.n_x,self.n_u))], [np.zeros((self.n_u,self.n_x)), self.R]])
        # desired_var = np.concatenate([self.x_d(), self.u_d()])
        for k in range(N-1):
            prog.AddQuadraticCost(
            Q=self.Q,
            b=np.zeros(self.n_x),
            c=0,
            vars=x[k],
            is_convex=True
            )
            prog.AddQuadraticCost(
                Q=self.R,
                b=np.zeros(self.n_u),
                c=0,
                vars=u[k],
                is_convex=True
            )
            # decision_var = np.concatenate([x[i], u[i]])
            # prog.AddQuadraticCost((x[k] - self.x_d).T @ self.Q @ (x[k] - self.x_d) + u[k].T * (self.R * u[k]), is_convex=True)
        # prog.AddQuadraticCost((x[N - 1] - self.x_d).T @ self.Qf @ (x[N - 1] - self.x_d), is_convex=True)
        prog.AddQuadraticCost(
            Q=self.Qf,
            b=np.zeros(self.n_x),
            c=0,
            vars=x[N-1],
            is_convex=True
        )

            

    
    
    
