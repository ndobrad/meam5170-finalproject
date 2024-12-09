import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from acrobot import Acrobot
from environment import Environment, Hold

class AcrobotVisualizer:
    def __init__(self, acrobot:Acrobot, env:Environment) -> None:
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(env.xmin, env.xmax)
        self.ax.set_ylim(env.ymin, env.ymax)
        self.ax.grid(visible=True)
        self.ax.set_axisbelow(True)
        
        self.env = env
        self.acrobot = acrobot
        
        self.hold_visuals = dict()
        
        self.start_hold = None
        self.goal_hold = None
        

        self.current_hold_color = [0.8, 0, 0.8]
        self.next_hold_color = [0.8, 0.7, 0]
        self.start_hold_color = [0, 0.9, 0.9]
        self.goal_hold_color = [0, 0.8, 0]
        self.default_hold_color = [0.3, 0.3, 0.3]
        self.highlighted_color = [1, 0, 0]
        
        # self.l1 = acrobot.plant.GetRigidBodyByName("red_link")
        
        self.red_link = np.vstack(
            (
                self.acrobot.l1 * np.array([1, 0, 0, 1, 1]),
                0.02 * np.array([1, 1, -1, -1, 1])
            )
        )
        self.blue_link = np.vstack(
            (
                self.acrobot.l2 * np.array([1, 0, 0, 1, 1]),
                0.02 * np.array([1, 1, -1, -1, 1])
            )
        )
        a = np.linspace(0, 2 * np.pi, 50)
        self.hold_points = np.vstack(
            (
                np.cos(a),
                np.sin(a),
            )
        )
        
        
        self.red_link_fill = self.ax.fill(
            self.red_link[0, :], self.red_link[1, :], zorder=2, edgecolor="r",
            facecolor=[1, 0, 0]
        )
        self.blue_link_fill = self.ax.fill(
            self.blue_link[0, :], self.blue_link[1, :], zorder=2, edgecolor="b",
            facecolor=[0, 0, 1]
        )
        
        self._setup_environment(self.env.start_idx, self.env.goal_idx)
        
        
    def draw(self, x, t, origin_offset=None, pose=None, current_hold=None, next_hold=None):
        self._update_environment(current_hold, next_hold)
        R = np.array([
            [np.cos(-np.pi/2 + x[0]), -np.sin(-np.pi/2 + x[0])],
            [np.sin(-np.pi/2 + x[0]), np.cos(-np.pi/2 + x[0])],
        ])
        
        R2 = np.array([
            [np.cos(-np.pi/2 + x[0] + x[1]), -np.sin(-np.pi/2 + x[0] + x[1])],
            [np.sin(-np.pi/2 + x[0] + x[1]), np.cos(-np.pi/2 + x[0] + x[1])],
        ])
        
        if pose == 1:
            p = np.dot(R,self.red_link)
            self.red_link_fill[0].get_path().vertices[:,0] = origin_offset[0] + p[0,:]
            self.red_link_fill[0].get_path().vertices[:,1] = origin_offset[1] + p[1,:]
            
            joint_pos = (self.acrobot.l1 * np.sin(x[0]),
                        -self.acrobot.l1 * np.cos(x[0]))

            p = np.dot(R2,self.blue_link)
            self.blue_link_fill[0].get_path().vertices[:,0] = origin_offset[0] + joint_pos[0] + p[0,:]
            self.blue_link_fill[0].get_path().vertices[:,1] = origin_offset[1] + joint_pos[1] + p[1,:]
        elif pose == -1:
            p = np.dot(R,self.blue_link)
            self.blue_link_fill[0].get_path().vertices[:,0] = origin_offset[0] + p[0,:]
            self.blue_link_fill[0].get_path().vertices[:,1] = origin_offset[1] + p[1,:]
            
            joint_pos = (self.acrobot.l2 * np.sin(x[0]),
                        -self.acrobot.l2 * np.cos(x[0]))

            p = np.dot(R2,self.red_link)
            self.red_link_fill[0].get_path().vertices[:,0] = origin_offset[0] + joint_pos[0] + p[0,:]
            self.red_link_fill[0].get_path().vertices[:,1] = origin_offset[1] + joint_pos[1] + p[1,:]
        
        self.ax.set_title("t = {:.1f}".format(t))
        
    def _setup_environment(self, origin_hold=None, goal_hold=None):
        # for each hold in self.env
        #   self._draw_hold(hold)
        for hi in range(len(self.env.holds)):
        # for h in self.env.holds:
            if hi == origin_hold:
                color = self.start_hold_color
                self.start_hold = hi
            elif hi == goal_hold:
                color = self.goal_hold_color
                self.goal_hold = hi
            elif self.env.holds[hi].is_highlighted:
                color = self.highlighted_color
            else:
                color = self.default_hold_color
            hold_fill = self.ax.fill(
                self.env.holds[hi].size * self.hold_points[0, :], 
                self.env.holds[hi].size * self.hold_points[0, :], 
                zorder=1, edgecolor="k",
                facecolor=color)
            
            hold_fill[0].get_path().vertices[:, 0] = (self.env.holds[hi].size * self.hold_points[0,:] 
                                                      + self.env.holds[hi].position[0])
            hold_fill[0].get_path().vertices[:, 1] = (self.env.holds[hi].size * self.hold_points[1,:] 
                                                      + self.env.holds[hi].position[1])
            self.hold_visuals[hi] = hold_fill
            
            
    def _update_environment(self, current_hold=None, next_hold=None):
        for hi in range(len(self.env.holds)):
            if hi == current_hold:
                color = self.current_hold_color
            elif hi == next_hold:
                color = self.next_hold_color
            elif hi == self.start_hold:
                color = self.start_hold_color
            elif hi == self.goal_hold:
                color = self.goal_hold_color
            elif self.env.holds[hi].is_highlighted:
                color = self.highlighted_color
            else:
                color = self.default_hold_color
            
            self.hold_visuals[hi][0].set_facecolor(color)
            # Unnecessary redraw of holds:
            # self.hold_visuals[hi][0].get_path().vertices[:, 0] = self.hold_points[0,:]*self.env.holds[hi].size + self.env.holds[hi].position[0]
            # self.hold_visuals[hi][0].get_path().vertices[:, 1] = self.hold_points[1,:]*self.env.holds[hi].size + self.env.holds[hi].position[1]
            
    


def create_animation(bot_vis:AcrobotVisualizer, x_traj, t, origin_offsets, poses, current_holds, next_holds):
    def update(i):
        bot_vis.draw(x_traj[i,:], t[i], origin_offsets[i,:], poses[i], current_holds[i], next_holds[i])

    ani = animation.FuncAnimation(
        bot_vis.fig, update, x_traj.shape[0], interval=(t[1]-t[0]) * 1000, 
    )
    return ani