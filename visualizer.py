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
        
        self.env = env
        self.acrobot = acrobot
        
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
        
        
        self.red_link_fill = self.ax.fill(
            self.red_link[0, :], self.red_link[1, :], zorder=1, edgecolor="r",
            facecolor=[1, 0, 0]
        )
        self.blue_link_fill = self.ax.fill(
            self.blue_link[0, :], self.blue_link[1, :], zorder=1, edgecolor="b",
            facecolor=[0, 0, 1]
        )
        
        
        
    def draw(self, x, t):
        self._draw_environment()
        
        R = np.array([
            [np.cos(-np.pi/2 + x[0]), -np.sin(-np.pi/2 + x[0])],
            [np.sin(-np.pi/2 + x[0]), np.cos(-np.pi/2 + x[0])],
        ])
        
        p = np.dot(R,self.red_link)
        self.red_link_fill[0].get_path().vertices[:,0] = p[0,:]
        self.red_link_fill[0].get_path().vertices[:,1] = p[1,:]
        
        joint_pos = (self.acrobot.l1 * np.sin(x[0]),
                     -self.acrobot.l1 * np.cos(x[0]))
        
        R = np.array([
            [np.cos(-np.pi/2 + x[0] + x[1]), -np.sin(-np.pi/2 + x[0] + x[1])],
            [np.sin(-np.pi/2 + x[0] + x[1]), np.cos(-np.pi/2 + x[0] + x[1])],
        ])
        
        p = np.dot(R,self.red_link)
        self.blue_link_fill[0].get_path().vertices[:,0] = joint_pos[0] + p[0,:]
        self.blue_link_fill[0].get_path().vertices[:,1] = joint_pos[1] + p[1,:]
        
        self.ax.set_title("t = {:.1f}".format(t))
        
        
    def _draw_hold(self, hold:Hold):
        pass
    
    def _draw_environment(self):
        # for each hold in self.env
        #   self._draw_hold(hold)
        pass
    


def create_animation(bot_vis, x_traj, t):
    def update(i):
        bot_vis.draw(x_traj[i,:], t[i])

    ani = animation.FuncAnimation(
        bot_vis.fig, update, x_traj.shape[0], interval=1e-2 * 1000
    )
    return ani