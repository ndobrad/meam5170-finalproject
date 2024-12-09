import numpy as np
from acrobot import Acrobot
from environment import Environment
from controller_base import Controller, PathPlanner
from pydrake.all import MathematicalProgram, OsqpSolver, PiecewisePolynomial
from pydrake.solvers import Solve
from pydrake.autodiffutils import AutoDiffXd

class ZeroController(Controller):
    def compute_feedback(self, current_x) -> np.ndarray:
        return np.zeros((self.n_u,))
    
    def update_target_state(self, x_desired):
        return super().update_target_state(x_desired)



class MPCController(Controller):
    def __init__(self, acrobot: Acrobot, Q, R:np.ndarray, Qf) -> None:
        super().__init__(acrobot)
        self.Q = Q
        self.R = R.reshape(acrobot.n_u,acrobot.n_u)
        self.Qf = Qf
        self.x_d = np.zeros(self.n_x)
        self.u_d = np.zeros(self.n_u)
        # self.last_u = np.zeros(self.n_u)
        self.T = 0.01
    
    def update_target_state(self, goal_pos):
        super().update_target_state(goal_pos)
        self.x_d = np.hstack([self.acrobot.get_joint_angles(goal_pos),np.zeros(2)])
    
    def update_target_trajectory(self, x_traj:PiecewisePolynomial, u_traj:PiecewisePolynomial):
        self.x_traj = x_traj
        self.u_traj = u_traj
    
    def compute_feedback(self, current_x, traj_t):
        # Parameters for the QP
        N = 20
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
        self._add_dynamics_constraint(prog, x, u, N, current_x, traj_t)
        self._add_cost(prog, x, u, N, traj_t)
        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)
        u_mpc = np.zeros(self.n_u)
        if result.is_success():
            u_mpc = np.array(result.GetSolution(u[0]))
            u_mpc += self.u_d
            # self.last_u = u_mpc
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

    def _add_dynamics_constraint(self, prog:MathematicalProgram, x, u, N, x_current, traj_t):
        """
        x, u: decision variables
        N: number of knot points
        T: time between knot points
        x_current: the current state of the system
        traj_t: t where x(t) approx. = x_current
        """
        #for now, linearize around the current state and last input. Later, update this to 
        #linearize around a trajectory. Maybe the trajectory could be saved to self.trajectory
        # in a class method like update_target_trajectory(self, new_traj) or something.
        #Then, put the next line here inside the for loop and pass it:
        # new_traj.x[k+(some kind of index into trajectory)], new_traj.u[k+(same index)]
                
        for k in range(N-1):
            #if we want to do time-varying linear MPC, inputs to linearization 
            # SHOULD be x,u decision variables. How to make this work?
            #Do the below for MPC tracking of predefined trajectory:
            xd = self.x_traj.value(traj_t+(self.T*k)).flatten()
            ud = self.u_traj.value(traj_t+(self.T*k)).flatten()
            A, B = self.acrobot.discrete_time_linear_dynamics(self.T, xd, ud)
            xdot = self.acrobot.continuous_time_full_dynamics(xd, ud)
            expr = x[k+1] - (xdot + A @ (x[k] - xd) + B @ (u[k] - ud))
            prog.AddLinearEqualityConstraint(expr, np.zeros(self.n_x))

    def _add_cost(self, prog:MathematicalProgram, x, u, N, traj_t):
        QR = np.block([[self.Q, np.zeros((np.size(self.Q,0), np.size(self.R,1)))],
                       [np.zeros((np.size(self.R,0), np.size(self.Q,1))), self.R]])
        for i in range(N-1):
            xd = self.x_traj.value(traj_t+(self.T*i)).flatten()
            ud = self.u_traj.value(traj_t+(self.T*i)).flatten()
            # prog.AddQuadraticCost((x[i] - self.x_d).T @ self.Q @ (x[i] - self.x_d))
            # prog.AddQuadraticCost((u[i] - self.u_d) * (self.R * (u[i] - self.u_d)))
            
            # prog.AddQuadraticCost(self.Q, np.zeros(self.n_x), (x[i]-xd))
            # prog.AddQuadraticCost(self.R, np.zeros(self.n_u), (u[i]-ud))
            
            # prog.AddQuadraticCost((x[i]-xd) @ self.Q @ (x[i]-xd))
            # prog.AddQuadraticCost((u[i]-ud) @ self.R @ (u[i]-ud))
            
            xu = np.hstack(((x[i]-xd),(u[i]-ud)))
            prog.AddQuadraticCost(xu.T @ QR @ xu)
            
        val = x[N-1] - self.x_traj.value(traj_t+(self.T*(N-1))).flatten()
        prog.AddQuadraticCost(val @ self.Qf @ val)
        # prog.AddQuadraticCost(self.Qf, np.zeros(self.n_x), x[N-1] - self.x_traj.value(traj_t+(self.T*(N-1))).flatten())


''' TODO: Modification Needed for Non-Linear MOC formulation'''
class NMPCController(Controller):
    def __init__(self, acrobot: Acrobot, Q, R, Qf, N=10) -> None:
        super().__init__(acrobot)
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.x_d = np.zeros(self.n_x)
        self.u_d = np.zeros(self.n_u)
        self.dt = 0.1
        self.N = N
    
    def update_target_state(self, goal_pos):
        self.x_d[:2] = self.acrobot.get_joint_angles(goal_pos)
        self.x_d[2:] = 0  # Zero velocity at the target
    
    def compute_feedback(self, current_x):

        # Initialize mathematical program and decalre decision variables
        prog = MathematicalProgram()
        x = np.zeros((self.N, self.n_x), dtype="object")
        for i in range(self.N):
            x[i] = prog.NewContinuousVariables(self.n_x, "x_" + str(i))
        u = np.zeros((self.N-1), dtype="object")
        for i in range(self.N-1):
            u[i] = prog.NewContinuousVariables(self.n_u, "u_" + str(i))
        
        # Add constraints and cost
        # self._add_initial_state_constraint(prog, x, current_x)
        # self._add_input_saturation_constraint(prog, x, u)
        # self._add_nonlinear_dynamics_constraint(prog, x, u)
        # self._add_cost(prog, x, u)
        self._add_cost_diff_formulation(prog, x, u)

        # Solve the optimization problem
        result = Solve(prog)
        u_mpc = np.zeros(self.n_u)            
        if result.is_success():
            u_mpc = result.GetSolution(u[0])
        else:
            print("Optimization failed!")
            u_mpc = np.zeros(self.n_u)
        return u_mpc
    
    def _add_initial_state_constraint(self, prog:MathematicalProgram, x, x_current):
        lb = x_current
        ub = x_current
        prog.AddBoundingBoxConstraint(lb, ub, x[0])
    
    def _add_input_saturation_constraint(self, prog:MathematicalProgram, x, u):
        lb = np.array([self.acrobot.umin]) 
        ub = np.array([self.acrobot.umax])
        for i in range(self.N-1):
            prog.AddBoundingBoxConstraint(lb, ub, u[i])
            
    def _add_linear_dynamics_constraint(self, prog:MathematicalProgram, x, u, T):
        A, B = self.acrobot.continuous_time_full_dynamics(x, u)
        for k in range(self.N-1):
            x_operating_point = x[k]
            u_operating_point = u[k]
            A, B = self.acrobot.continuous_time_linear_dynamics(x_operating_point, u_operating_point)
            Ad = np.identity(self.n_x) + A * T
            Bd = B * T
            expr = x[k+1] - (Ad @ x[k] + Bd @ u[k])
            prog.AddLinearEqualityConstraint(expr, np.zeros(self.n_x))

    def discrete_dynamics(self, x, u):
        # Use Euler discretization
        x_dot = self.acrobot.continuous_time_full_dynamics(x, u)
        return x + self.dt * x_dot  

    def _add_nonlinear_dynamics_constraint(self, prog:MathematicalProgram, x, u):
        def dynamics(state, input):
            if isinstance(state[0], AutoDiffXd):
                # Create a context with AutoDiffXd
                context = self.acrobot.ad_plant.CreateDefaultContext()
                self.acrobot.ad_plant.SetPositions(context, state[:self.acrobot.n_q])
                self.acrobot.ad_plant.SetVelocities(context, state[self.acrobot.n_q:])
                self.acrobot.ad_plant.get_actuation_input_port().FixValue(context, input)
                derivatives = self.acrobot.ad_plant.AllocateTimeDerivatives()
                self.acrobot.ad_plant.CalcTimeDerivatives(context, derivatives)
                return derivatives.get_vector().CopyToVector()
            else:
                # Use the original continuous_time_full_dynamics for regular floats
                return self.acrobot.continuous_time_full_dynamics(state, input)

        for k in range(self.N - 1):
            prog.AddConstraint(
                lambda vars, dt=self.dt: vars[self.n_x:] - (
                    vars[:self.n_x] + dt * dynamics(vars[:self.n_x], vars[self.n_x:self.n_x+self.n_u])
                ),
                lb=np.zeros(self.n_x),
                ub=np.zeros(self.n_x),
                vars=np.concatenate([x[:, k], u[:, k], x[:, k+1]])
            )

        # Dynamics constraints
        # for k in range(self.N -1):
        #     x_k = x[:, k]
        #     u_k = u[k]
        #     x_next = self.discrete_dynamics(x_k, u_k)
        #     prog.AddConstraint(x[:, k+1] == x_next)

        # for k in range(N-1):
        #     prog.AddConstraint(
        #         lambda vars, T=T, acrobot=self.acrobot: dynamics_constraint(vars, T, acrobot),
        #         lb=np.zeros(self.n_x),
        #         ub=np.zeros(self.n_x),
        #         vars=np.concatenate([x[k], u[k], x[k+1]])
        #     )
    
    def _add_cost(self, prog:MathematicalProgram, x, u):
        # Cost function
        for k in range(self.N-1):
            prog.AddQuadraticCost((x[:, k] - self.x_d).T @ self.Q @ (x[:, k] - self.x_d))
            prog.AddQuadraticCost(u[k].T @ self.R @ u[k])
        
        # Terminal cost
        prog.AddQuadraticCost((x[:, -1] - self.x_d).T @ self.Qf @ (x[:, -1] - self.x_d))

    def _add_cost_diff_formulation(self, prog:MathematicalProgram, x, u):
        # Stage cost
        for k in range(self.N - 1):
            prog.AddQuadraticCost(Q=self.Q, b=-2 * self.Q @ self.x_d, c=(self.x_d.T @ self.Q @ self.x_d), vars=x[k])
            prog.AddQuadraticCost(Q=self.R, b=np.zeros(self.n_u), c=0, vars=u[k])
        # Terminal cost
        prog.AddQuadraticCost(Q=self.Qf, b=-2 * self.Qf @ self.x_d, c=(self.x_d.T @ self.Qf @ self.x_d), vars=x[self.N - 1])