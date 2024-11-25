import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, SceneGraph
from manipulator_dynamics import ManipulatorDynamics

from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver
import pydrake.symbolic as sym
from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables


class Acrobot(object):
    def __init__(self, plant: MultibodyPlant, Q, R, Qf):
        self.l1 = 1.5
        self.l2 = 1.5
        # self.l1c = 2.5
        # self.l2c = 2.5
        # self.m1 = 2
        # self.m2 = 2
        # self.I1 = ???
        # self.I2 = ???
        
        self.plant = plant
        # builder = DiagramBuilder()
        # self.plant, self.scenegraph = AddMultibodyPlantSceneGraph(builder,plant)
        # diagram = builder.Build()
        # sg_context = self.scenegraph.GetMyMutableContextFromRoot(diagram.CreateDefaultContext())
        # query_object = self.scenegraph.get_query_output_port().Eval(sg_context)
        # inspector = query_object.inspector()
                   
        self.n_q = 2
        self.n_x = self.n_q * 2
        self.n_u = 1
        
    def x_d(self):
        # Desired State
        return np.array([0,0,0,0])

    def u_d(self):
        # Desired Input (Required?)
        return np.array([0])

    def continuous_time_full_dynamics(self, x, u):
        q = x[:self.n_q]
        v = x[self.n_q:self.n_x]
        (M, Cv, tauG, B, tauExt) = ManipulatorDynamics(self.plant, q, v)
        M_inv = np.linalg.inv(M)
        v_dot = M_inv @ (B @ u + tauG - Cv)
        return np.hstack((x[-self.n_q:], v_dot))

    
    def continuous_time_linear_dynamics(self):
        #need to linearize around x, u? Use partial feedback linearization?
        pass

    def partial_feedback_linearization(self):
        pass
    
    def discrete_time_linear_dynamics(self, T: float):
        A, B = self.continuous_time_linear_dynamics()
        Ad = np.identity(self.n_x) + A * T
        Bd = B * T
        return Ad, Bd
    
    def add_initial_state_constraint(self, prog, x, x_current):
        lb = x_current
        ub = x_current
        prog.AddBoundingBoxConstraint(lb, ub, x[0])
        # pass

    def add_input_saturation_constraint(self, prog, x, u, N):
        lb = np.array([self.umin - self.u_d()]) 
        ub = np.array([self.umax - self.u_d()])
        for i in range(N-1):
            prog.AddBoundingBoxConstraint(lb,ub, u[i])
        # pass

    def add_dynamics_constraint(self, prog, x, u, N, T):
        A, B = self.discrete_time_linear_dynamics(T)
        for k in range(N-1):
            expr = x[k+1] - (A @ x[k] + B @ u[k])
            prog.AddLinearEqualityConstraint(expr, np.zeros(self.n_x))
        # pass

    def add_cost(self, prog, x, u, N):
        # Need to change this
        Qk = np.block([[self.Q, np.zeros((self.n_x,self.n_u))], [np.zeros((self.n_u,self.n_x)), self.R]])
        # desired_var = np.concatenate([self.x_d(), self.u_d()])
        for i in range(N-1):
            decision_var = np.concatenate([x[i], u[i]])
            prog.AddQuadraticCost((decision_var).T @ Qk @ (decision_var))
        prog.AddQuadraticCost((x[N - 1] - self.x_d()).T @ self.Qf @ (x[N - 1] - self.x_d()))
        # pass

    def compute_mpc_feedback(self, x_current):
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
        self.add_initial_state_constraint(prog, x, x_current)
        self.add_input_saturation_constraint(prog, x, u, N)
        self.add_dynamics_constraint(prog, x, u, N, T)
        self.add_cost(prog, x, u, N)
        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)
        u_mpc = np.zeros(self.n_u)
        if result.is_success():
            u_mpc = np.array(result.GetSolution(u[0]))
            u_mpc += self.u_d()
        else:
            u_mpc = np.zeros(2)
        return u_mpc
        # pass