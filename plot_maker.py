import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
from pydrake.all import PiecewisePolynomial, Polynomial

def add_trajectory_to_plot(ax:Axes, traj:PiecewisePolynomial, series_name:str):
    t = np.linspace(0,traj.end_time(),500)
    n_x = len(traj.value(0).flatten())
    x = np.zeros((len(t),n_x))
    for i in range(len(t)):
        x[i,:] = traj.value(t[i]).flatten()
    for k in range(n_x):
        ax.plot(t, x[:,k], label='{}_{}'.format(series_name,k))
     


def make_trajectory_plot(trajectories):
    fig, ax = plt.subplots()
    add_trajectory_to_plot(ax, trajectories[1], 'x')
    ax.set_title("x(t)")
    ax.legend()
    
    fig, ax = plt.subplots()
    add_trajectory_to_plot(ax, trajectories[2], 'u')
    ax.set_title("u(t)")
    
        
