import numpy as np
from acrobot import Acrobot
from scipy.integrate import solve_ivp
from controller_base import Controller, PathPlanner

def simulate_acrobot(x0, tf, acrobot:Acrobot, controller:Controller, path_planner:PathPlanner=None):
    t0 = 0.0
    dt = 1e-2

    x = [x0]
    u = [np.zeros((acrobot.n_u,))]
    t = [t0]

    while t[-1] < tf:
        current_time = t[-1]
        current_x = x[-1]
        current_u_command = np.zeros(2)

        current_u_command = controller.compute_feedback(current_x)
        # TODO: uncomment below if we need to constrain input
        # current_u_real = np.clip(current_u_command, acrobot.umin, acrobot.umax)
        current_u_real = current_u_command
        # Autonomous ODE for constant inputs to work with solve_ivp
        def f(t, x):
            return acrobot.continuous_time_full_dynamics(current_x, current_u_real)
        # Integrate one step
        sol = solve_ivp(f, (0, dt), current_x, first_step=dt)#, method='DOP853')

        # Record time, state, and inputs
        t.append(t[-1] + dt)
        x.append(sol.y[:, -1])
        u.append(current_u_command)

    x = np.array(x)
    u = np.array(u)
    t = np.array(t)
    return x, u, t