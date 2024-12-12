import numpy as np
from pydrake.all import PiecewisePolynomial, Polynomial

def get_total_effort(u_traj:PiecewisePolynomial, end_time:float=None):
    if end_time is None:
        t = np.linspace(0,u_traj.end_time(),1000)
    else:
        t = np.linspace(0,end_time,1000)
    dt = t[1] - t[0]
    u_tot = 0
    for i in range(len(t)):
        u_tot += np.abs(u_traj.value(t[i]).flatten()) * dt
    return u_tot @ np.ones_like(u_tot)