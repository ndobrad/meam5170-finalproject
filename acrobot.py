import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from manipulator_dynamics import ManipulatorDynamics


class Acrobot(object):
    def __init__(self, plant: MultibodyPlant, Q, R, Qf):
        # self.l1 = 5
        # self.l2 = 5
        # self.l1c = 2.5
        # self.l2c = 2.5
        # self.m1 = 2
        # self.m2 = 2
        # self.I1 = ???
        # self.I2 = ???
        
        self.plant = plant
        
        self.n_q = 2
        self.n_x = self.n_q * 2
        self.n_u = 1
        pass
    
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
    
    def discrete_time_linear_dynamics(self, T: float):
        A, B = self.continuous_time_linear_dynamics()
        Ad = np.identity(self.n_x) + A * T
        Bd = B * T
        return Ad, Bd