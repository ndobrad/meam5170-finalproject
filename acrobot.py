import numpy as np
from pydrake.multibody.plant import MultibodyPlant
#from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, SceneGraph
from pydrake.autodiffutils import AutoDiffXd, InitializeAutoDiff, ExtractGradient
from pydrake.systems.framework import BasicVector_
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
        self.ad_plant = self.plant.ToAutoDiffXd()
        self.ad_context = self.ad_plant.CreateDefaultContext()
        self.input_port = self.ad_plant.get_actuation_input_port().get_index()
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
        
        # normalize x[0] and x[1] to [-pi, pi), calculate angle of the distal
        # link relative to the base joint
        q_l2 = np.arctan2(np.sin(x[0]),np.cos(x[0])) + np.arctan2(np.sin(x[1]),np.cos(x[1]))
        q_l2 = np.arctan2(np.sin(q_l2),np.cos(q_l2))
        # calculate offset from smallest multiple of pi/2
        q_std = np.abs(q_l2) % (np.pi/2)
        q_corr = 1 - (np.abs(q_l2) // (np.pi/2))
        # Calculate new base angle
        if q_l2 == 0:
            mult = 1
        else:
            mult = -np.sign(q_l2)
        ret[0] = (np.pi/2 - q_std + (q_corr * np.pi/2)) * mult
        ret[1] = -x[1]
        #ret[2] = -x[2] - x[3]
        ret[3] = -x[3]
        return ret 

    def continuous_time_full_dynamics(self, x, u):
        q = x[:self.n_q]
        v = x[self.n_q:self.n_x]
        (M, Cv, tauG, B, tauExt) = ManipulatorDynamics(self.plant, q, v)      
        v_dot = np.linalg.solve(M, (B @ u + tauG - Cv + tauExt))
        return np.hstack((v, v_dot))

    
    def continuous_time_linear_dynamics(self, xd, ud):
        """
        Linearizes the dynamics around xd, ud, calculates using
        error coordinates
        Returns: A, B
        ref: https://stackoverflow.com/a/64565582
        """
        #need to linearize around x, u
        #linearizing around some trajectory:
        #how to find traj? Direct collocation
        
        xu_val = np.hstack((xd, ud))
        nx = self.ad_context.num_continuous_states()
        
        xu_ad = InitializeAutoDiff(xu_val)
        x_ad = xu_ad[:nx]
        u_ad = xu_ad[nx:]
        self.ad_context.SetContinuousState(x_ad)
        self.ad_plant.get_input_port(self.input_port).FixValue(self.ad_context, BasicVector_[AutoDiffXd](u_ad))  
        derivatives = self.ad_plant.AllocateTimeDerivatives()
        self.ad_plant.CalcTimeDerivatives(self.ad_context, derivatives)
        xdot_ad = derivatives.get_vector().CopyToVector()    

        AB = ExtractGradient(xdot_ad)
        A = AB[:, :nx]
        B = AB[:, nx:]
        
        return A, B
    
    
    def time_varying_linear_dynamics(self, xd, ud):
        """
        Linearizes the dynamics around xd, ud, when 
        xd and ud are Variable objects
        Returns: A, B (are these )
        ref: https://stackoverflow.com/a/64565582,
        https://stackoverflow.com/questions/77687354/substitute-symbolic-variables-with-autodiff-variables
        
        THIS DOES NOT WORK RIGHT NOW
        """
        #need to linearize around Variable x, u
        
        xu_val = np.hstack((xd, ud))

        plant_sym = self.ad_plant.ToSymbolic()
        context_sym = plant_sym.CreateDefaultContext()
        plant_sym.SetPositionsAndVelocities(context_sym, xd)
        # self.ad_plant.get_input_port(self.input_port).FixValue(self.ad_context, BasicVector_[AutoDiffXd](ud))  
        derivatives = plant_sym.AllocateTimeDerivatives()
        plant_sym.CalcTimeDerivatives(context_sym, derivatives)
        xdot_ad = derivatives.get_vector().CopyToVector()    

        AB = ExtractGradient(xdot_ad)
        nx = context_sym.num_continuous_states()
        A = AB[:, :nx]
        B = AB[:, nx:]
        
        return A, B
        
        
    def discrete_time_linear_dynamics(self, T: float, xd, ud):
        """
        Returns x_k+1 via Euler method
        """
        A, B = self.continuous_time_linear_dynamics(xd, ud)
        Ad = np.identity(self.n_x) + A * T
        Bd = B * T
        return Ad, Bd

    def get_tip_position(self, x) -> np.ndarray:
        xpos = self.l1 * np.sin(x[0]) + self.l2 * np.sin(x[0] + x[1])
        ypos = -(self.l1 * np.cos(x[0]) + self.l2 * np.cos(x[0] + x[1]))
        return np.array([xpos, ypos])
    
    def get_joint_angles(self, pos, x1_positive:bool=True) -> np.ndarray:
        """
        inputs:
            pos: x,y coordinates of desired end effector position
            x1_positive: set this to true if you'd like the elbow joint to have a positive angle 
        """
        #https://math.stackexchange.com/a/1033561
        ret = np.zeros(2)
        # d = np.sqrt(pos[0]**2 + pos[1]**2)
        # L = (self.l1**2 - self.l2**2 + d**2)/(2*d)
        # h = np.sqrt(self.l1**2 - L**2)
        # elbow_x = L*pos[0]/d + h*pos[1]/d
        # elbow_y = L*pos[1]/d - h*pos[0]/d
        # ret[0] = np.arctan2(elbow_x, -elbow_y)               
        # ret[1] = np.arctan2((elbow_x*pos[1] - pos[0]*elbow_y),
        #                     np.dot(pos, np.array([elbow_y, elbow_x])))

        ## Alternate method
        # norm_pos = pos / np.linalg.norm(pos)
        # norm_xy = np.array([elbow_y, elbow_x])/ np.linalg.norm(np.array([elbow_y, elbow_x]))
        # ret[1] = np.arccos(np.dot(norm_pos, norm_xy))
        
        x = -pos[1]
        y = pos[0]
        
        if x1_positive:
            ret[1] = np.arccos((x**2+y**2-self.l1**2-self.l2**2)/(2*self.l1*self.l2))
            ret[0] = np.arctan2(y,x) - np.arctan2(self.l2*np.sin(ret[1]),self.l1+self.l2*np.cos(ret[1]))
        else:
            ret[1] = -np.arccos((x**2+y**2-self.l1**2-self.l2**2)/(2*self.l1*self.l2))
            ret[0] = np.arctan2(y,x) - np.arctan2(self.l2*np.sin(ret[1]),self.l1+self.l2*np.cos(ret[1]))
        return ret