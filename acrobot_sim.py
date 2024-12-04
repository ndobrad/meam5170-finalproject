from collections import namedtuple
import numpy as np
from acrobot import Acrobot
from scipy.integrate import solve_ivp
from controller_base import Controller, PathPlanner
from environment import Environment, Hold

SimResult = namedtuple('SimResult', ['t','x','u','origin_offsets','pose','current_holds','next_holds'])

def simulate_acrobot(x0, tf, acrobot:Acrobot, controller:Controller, path_planner:PathPlanner, ):
    t0 = 0.0
    dt = 1e-2

    x = [x0]
    u = [np.zeros((acrobot.n_u,))]
    t = [t0]
    
    plan = path_planner.calculate_path(0,1)
    active_plan_steps = [0] #list of PLAN indices
    next_plan_steps = [1] #list of PLAN indices
    pose = [1] #indicate which link (red/blue) is latched to hold. Used for vis only.
    
    origin_offsets = [path_planner.env.holds[plan[0]].position]
    
    # once the arm attaches to final goal position, set target to nonexistant Hold outside env
    # to avoid triggering guard
    terminal_hold = Hold((path_planner.env.xmax + 1, 0), 0) 
    
    #set attributes of guard fun
    guard.terminal = True #end solve_ivp if guard triggered
    
    while t[-1] < tf:
        current_time = t[-1]
        current_x = x[-1]
        current_u_command = np.zeros(1)
        current_pose = pose[-1]
        current_plan_step = active_plan_steps[-1]
        current_next_plan_step = next_plan_steps[-1]
        current_origin_offset = origin_offsets[-1]

        # If current_next_hold is None, then the robot is attached to the final goal and we can
        # stop computing inputs (but continue simulating until tf)
        if current_next_plan_step is not None:
            goal_hold = path_planner.env.holds[plan[current_next_plan_step]]
            goal_location = goal_hold.position - current_origin_offset #next hold location relative to robot base
            controller.update_target_state(goal_location)
            current_u_command = controller.compute_feedback(current_x)
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
                else:
                    current_next_plan_step = None
                    goal_hold = terminal_hold
                    goal_location = goal_hold.position
                
            else:
                timestep_finished = True

        # Record time, state, and inputs
        t.append(t[-1] + dt)
        x.append(sol.y[:, -1])
        u.append(current_u_command)
        pose.append(current_pose)
        active_plan_steps.append(current_plan_step)
        next_plan_steps.append(current_next_plan_step)
        origin_offsets.append(current_origin_offset)

    #translate the lists of PathPlanner indices into lists of Environment.holds indices
    active_holds = np.array([plan[q] for q in active_plan_steps])
    target_holds = np.array([plan[q] if q is not None else None for q in next_plan_steps])
    
    ret = SimResult(t=np.array(t), x=np.array(x), u=np.array(u), 
                    origin_offsets=np.array(origin_offsets),
                    pose=np.array(pose), 
                    current_holds=active_holds, 
                    next_holds=target_holds)
   
    return ret



def guard(t, x, acrobot:Acrobot, goal_state:tuple, size):
    bot_pos = acrobot.get_tip_position(x)
    return (np.sqrt((bot_pos[0]-goal_state[0])**2 
                    + (bot_pos[1]-goal_state[1])**2) 
            - size)
