from collections import namedtuple
import numpy as np
from acrobot import Acrobot
from scipy.integrate import solve_ivp
from controller_base import Controller, PathPlanner, TrajectoryOptimizer
from environment import Environment, Hold

SimResult = namedtuple('SimResult', ['t','x','u','origin_offsets','pose','current_holds','next_holds','trajectories','active_trajectories'])

def simulate_acrobot(x0, tf, acrobot:Acrobot, controller:Controller, path_planner:PathPlanner, traj_opt:TrajectoryOptimizer):
    t0 = 0.0
    dt = 1e-2

    x = [x0]
    u = [np.zeros((acrobot.n_u,))]
    t = [t0]
    
    plan = path_planner.calculate_path()
    active_plan_steps = [0] #list of PLAN indices
    next_plan_steps = [1] #list of PLAN indices
    pose = [1] #indicate which link (red/blue) is latched to hold. Used for vis only.
    
    origin_offsets = [path_planner.env.holds[plan[0]].position]
    
    # once the arm attaches to final goal position, set target to nonexistant Hold outside env
    # to avoid triggering guard
    terminal_hold = Hold((path_planner.env.xmax + 1, 0), 0) 
    
    #set attributes of guard fun
    guard.terminal = True #end solve_ivp if guard triggered
    
    #initial conditions
    current_origin_offset = origin_offsets[-1]
    current_next_plan_step = next_plan_steps[-1]
    goal_hold = path_planner.env.holds[plan[current_next_plan_step]]
    goal_location = goal_hold.position - current_origin_offset #next hold location relative to robot base
    controller.update_target_state(goal_location)
    traj_sol = traj_opt.generate_trajectory(x[0],goal_location)
    x_traj = traj_opt.collocation_prog.ReconstructStateTrajectory(traj_sol)
    u_traj = traj_opt.collocation_prog.ReconstructInputTrajectory(traj_sol)
    x_traj_collection = dict()
    x_traj_collection[0] = x_traj
    x_traj_col_idx = [0]
    controller.update_target_trajectory(x_traj,u_traj)
    traj_time = 0
    
    while t[-1] < tf:
        current_time = t[-1]
        current_x = x[-1]
        current_u_command = np.zeros(1)
        current_pose = pose[-1]
        current_plan_step = active_plan_steps[-1]
        current_next_plan_step = next_plan_steps[-1]
        current_origin_offset = origin_offsets[-1]
        current_x_traj_col_idx = x_traj_col_idx[-1]

        # If current_next_hold is None, then the robot is attached to the final goal and we can
        # stop computing inputs (but continue simulating until tf)
        if current_next_plan_step is not None:

            current_u_command = controller.compute_feedback(current_x, traj_time)
            # TODO: uncomment below if we need to constrain input
            # current_u_command = np.clip(current_u_command, acrobot.umin, acrobot.umax)
        else:
            goal_hold = terminal_hold
            goal_location = goal_hold.position
            #don't generate an input once the final goal state is reached
            
        current_u_real = current_u_command
        target_time = dt
        
        # Autonomous ODE for constant inputs to work with solve_ivp
        # extra inputs necessary for event fun (solve_ivp needs all callables to
        # have the same number of inputs)
        def f(t, x, bot, goal, size):
            return acrobot.continuous_time_full_dynamics(x, current_u_real)
            # return acrobot.continuous_time_full_dynamics(current_x, current_u_real)
            
        timestep_finished = False
        while not timestep_finished:
        # Integrate one step
            sol = solve_ivp(f, (0, target_time), current_x, first_step=target_time, 
                            events=guard, args=(acrobot, goal_location, goal_hold.size))
            if sol.status == 1:
                current_origin_offset = current_origin_offset + acrobot.get_tip_position(sol.y[:,-1])
                current_x = acrobot.reset_map(sol.y[:,-1])
                target_time = target_time - sol.t_events[0][0]
                current_pose *= -1
                current_plan_step = current_next_plan_step
                if current_next_plan_step + 1 < len(plan):
                    current_next_plan_step += 1
                    goal_hold = path_planner.env.holds[plan[current_next_plan_step]]
                    goal_location = goal_hold.position - current_origin_offset
                    traj_sol = traj_opt.generate_trajectory(current_x, goal_location)
                    x_traj = traj_opt.collocation_prog.ReconstructStateTrajectory(traj_sol)
                    u_traj = traj_opt.collocation_prog.ReconstructInputTrajectory(traj_sol)
                    current_x_traj_col_idx += 1
                    x_traj_collection[current_x_traj_col_idx] = x_traj
                    controller.update_target_trajectory(x_traj,u_traj)
                    traj_time = 0
                    # controller.update_target_state(goal_location)
                else:
                    current_next_plan_step = None
                    current_x_traj_col_idx = None
                    goal_hold = terminal_hold
                    goal_location = goal_hold.position
                
            else:
                timestep_finished = True

            traj_time += sol.t[-1] #advance the trajectory timestep by the amount of time simulated this loop
            
        # Record time, state, and inputs
        t.append(t[-1] + dt)
        x.append(sol.y[:, -1])
        u.append(current_u_command)
        pose.append(current_pose)
        active_plan_steps.append(current_plan_step)
        next_plan_steps.append(current_next_plan_step)
        origin_offsets.append(current_origin_offset)
        x_traj_col_idx.append(current_x_traj_col_idx)

    #translate the lists of PathPlanner indices into lists of Environment.holds indices
    active_holds = np.array([plan[q] for q in active_plan_steps])
    target_holds = np.array([plan[q] if q is not None else None for q in next_plan_steps])
    
    ret = SimResult(t=np.array(t), x=np.array(x), u=np.array(u), 
                    origin_offsets=np.array(origin_offsets),
                    pose=np.array(pose), 
                    current_holds=active_holds, 
                    next_holds=target_holds,
                    trajectories=x_traj_collection,
                    active_trajectories=x_traj_col_idx)
   
    return ret



def guard(t, x, acrobot:Acrobot, goal_state:tuple, size):
    bot_pos = acrobot.get_tip_position(x)
    return (np.sqrt((bot_pos[0]-goal_state[0])**2 
                    + (bot_pos[1]-goal_state[1])**2) 
            - size)
