import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, SceneGraph
from manipulator_dynamics import ManipulatorDynamics

class Acrobot(object):
    def __init__(self, plant: MultibodyPlant):
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
        
        self.umin = -10000
        self.umax = 10000
        
    def reset_map(self, x):
        """
        Returns a state that represents a system in state x that has just
        switched from being attached at the "shoulder" to being attached 
        at the "hand." 
        """
        ret = np.zeros_like(x)
        
        q_l2 = x[0] + x[1]
        q_corr = q_l2 // (np.pi/4)
        q_std = q_l2 % (np.pi/4)
        ret[0] = -(np.pi/2 - q_std + (q_corr * np.pi/4))
        ret[1] = -x[1]
        #ret[2] = -x[2] - x[3]
        ret[3] = -x[3]
        return ret 

    def continuous_time_full_dynamics(self, x, u):
        q = x[:self.n_q]
        v = x[self.n_q:self.n_x]
        (M, Cv, tauG, B, tauExt) = ManipulatorDynamics(self.plant, q, v)
        # M_inv = np.linalg.inv(M)
        # v_dot = M_inv @ (B @ u + tauG - Cv + tauExt)        
        v_dot = np.linalg.solve(M, (B @ u + tauG - Cv + tauExt))
        return np.hstack((v, v_dot))

    
    def continuous_time_linear_dynamics(self):
        #need to linearize around x, u? Use partial feedback linearization?
        #linearizing around some trajectory? but then how to find traj (MPC)?
        #linearize around current x?
        pass

    def partial_feedback_linearization(self):
        pass
    
    def discrete_time_linear_dynamics(self, T: float):
        A, B = self.continuous_time_linear_dynamics()
        Ad = np.identity(self.n_x) + A * T
        Bd = B * T
        return Ad, Bd

    def get_tip_position(self, x):
        xpos = self.l1 * np.sin(x[0]) + self.l2 * np.sin(x[0] + x[1])
        ypos = -(self.l1 * np.cos(x[0]) + self.l2 * np.cos(x[0] + x[1]))
        return (xpos, ypos)