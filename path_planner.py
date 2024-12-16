"""
The path planner, with multiple options for heuristics.
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt
from environment import Environment
from controller_base import PathPlanner

class AStarPathPlanner(PathPlanner):
    def __init__(self, env:Environment):
        super().__init__(env)
        self.bounds = (env.xmin, env.xmax, env.ymin, env.ymax)
        self.holds = env.holds
        # self.range = env.spacing + env.grasp_radius
        self.range = 3 # reach of the Acrobot
        assert env.start_idx is not None
        assert env.goal_idx is not None
        self.start_idx = env.start_idx
        self.goal_idx = env.goal_idx

    def l2_heuristic(self, curr_idx):
        curr_pos = self.holds[curr_idx].position
        goal_pos = self.holds[self.goal_idx].position
        return np.linalg.norm(np.array(curr_pos) - np.array(goal_pos))
    
    def edge_cost(self, prev_idx, curr_idx, next_idx):

        curr_pos = np.array(self.holds[curr_idx].position)
        
        # if prev_idx is -1, then curr_idx is the initial hold 
        # of the environment and the previous position should be the 
        # "at rest" state of the arm, straight down from the current hold
        if prev_idx is None:
            prev_pos = curr_pos - np.array([0,self.range])
        else:
            prev_pos = np.array(self.holds[prev_idx].position)
        next_pos = np.array(self.holds[next_idx].position)
        
        L_next = np.linalg.norm(next_pos - curr_pos)

        return L_next
    
    def get_neighbors(self, curr_idx):
        curr_pos = np.array(self.holds[curr_idx].position)
        neighbors = []
        for idx, hold in enumerate(self.holds):
            if idx != curr_idx:
                pos = np.array(hold.position)
                dist = np.linalg.norm(pos - curr_pos)
                if dist < self.range:
                    neighbors.append(idx)
        return neighbors

    def calculate_path(self):
        """
        Calculate path using A*
        """
        open_set = []   # priority queue with (cost, index)
        heapq.heappush(open_set, (0, self.start_idx))
        closed_set = set()
        g_costs = {self.start_idx: 0}
        parent = {self.start_idx: None}
        while open_set:
            # if len(closed_set) % 5 == 0:
            #     print("closed set contains {} nodes".format(len(closed_set)))
            _, curr_idx = heapq.heappop(open_set)
            if curr_idx in closed_set:
                continue
            closed_set.add(curr_idx)
            if curr_idx == self.goal_idx:
                path = [curr_idx]
                while curr_idx in parent:
                    curr_idx = parent[curr_idx]
                    if curr_idx is not None:
                        path.append(curr_idx)
                path.reverse()
                return path
            for neighbor_idx in self.get_neighbors(curr_idx):
                if neighbor_idx in closed_set:
                    continue
                new_g = g_costs[curr_idx] + self.edge_cost(parent[curr_idx], curr_idx, neighbor_idx)
                if neighbor_idx not in g_costs or new_g < g_costs[neighbor_idx]:
                    g_costs[neighbor_idx] = new_g
                    f = self.l2_heuristic(neighbor_idx)
                    total_cost = new_g + f
                    heapq.heappush(open_set, (total_cost, neighbor_idx))
                    parent[neighbor_idx] = curr_idx
        return []
    
    def plot(self, env, paths, file_path):
        # TODO
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, hold in enumerate(self.holds):
            x, y = hold.position
            if hold.is_latched:
                cm = 'mo'
                label = None
            elif i == self.start_idx:
                cm = 'cs'
                label = "Start"
            elif i == self.goal_idx:
                cm = 'gs'
                label = "Goal"
            else:
                 cm = 'ko'
                 label = None
            ax.plot(x, y, cm, label=label)
        colors = ['r', 'y', 'b']
        for idx, (label, path) in enumerate(paths.items()):
            if path:
                path_positions = [env.holds[idx].position for idx in path]
                x_coords, y_coords = zip(*path_positions)
                ax.plot(x_coords, y_coords, linestyle='-', linewidth=2, color=colors[idx % len(colors)], label=label)
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_title("A* Paths")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.grid(True)
        plt.savefig(file_path)
        pass
    
class TrajectoryMatchPathPlanner(AStarPathPlanner):
    def __init__(self, env:Environment, trajectory_inputs):
        super().__init__(env)
        self.trajectory_inputs = trajectory_inputs[trajectory_inputs[:, 7] >= 0]
        self.chosen_trajectories:dict[tuple,any] = {}
               
    def edge_cost(self, prev_idx, curr_idx, next_idx):
        curr_pos = self.holds[curr_idx].position
        
        # if prev_idx is -1, then curr_idx is the initial hold 
        # of the environment and the previous position should be the 
        # "at rest" state of the arm, straight down from the current hold
        if prev_idx is None:
            prev_pos = curr_pos - np.array([0,self.range])
        else:
            prev_pos = self.holds[prev_idx].position
        next_pos = self.holds[next_idx].position

        # get prev/next positions relative to current position
        prev_pos_relative = prev_pos - curr_pos
        next_pos_relative = next_pos - curr_pos
        
        diff = self.trajectory_inputs[:,0:4] - np.hstack((prev_pos_relative, next_pos_relative))

        diff_norm = [np.dot(r,r) for r in diff]
        # diff_norm = 

        closest_pairs = self.trajectory_inputs[np.min(diff_norm) == diff_norm,:]
       
        # minimum positive input
        valid_closest_pairs = closest_pairs[closest_pairs[:, 7] >= 0]
        
        if valid_closest_pairs.size == 0:
            return None
        chosen_trajectory = valid_closest_pairs[np.argmin(valid_closest_pairs[:, 7])]
        self.chosen_trajectories[(prev_idx, curr_idx, next_idx)] = chosen_trajectory
        return np.min(valid_closest_pairs[:, 7])

class EnergyPathPlanner(AStarPathPlanner):
    def __init__(self, env):
        super().__init__(env)
    
    def edge_cost(self, prev_idx, curr_idx, next_idx):
        # simplified torque-based model of simple pendulum
        g = 9.81
        m = 10.0
        alpha = 0.0
        beta = 0.0
        
        curr_pos = np.array(self.holds[curr_idx].position)
        # if prev_idx is -1, then curr_idx is the initial hold 
        # of the environment and the previous position should be the 
        # "at rest" state of the arm, straight down from the current hold
        if prev_idx is None:
            prev_pos = curr_pos - np.array([0,self.range])
        else:
            prev_pos = np.array(self.holds[prev_idx].position)
        next_pos = np.array(self.holds[next_idx].position)
        
        # change pendulum length halfway through arc
        L_prev = np.linalg.norm(prev_pos - curr_pos)
        L_next = np.linalg.norm(next_pos - curr_pos)
        if L_prev <= 0:
            L_prev = 0.05
        if L_next <= 0:
            L_next = 0.05
        delta_x_prev = curr_pos[0] - prev_pos[0]
        delta_x_next = next_pos[0] - curr_pos[0]
        sin_theta_prev = delta_x_prev / L_prev
        sin_theta_next = delta_x_next / L_next
        # gravitational torque
        tau_prev = m * g * L_prev * sin_theta_prev
        tau_next = m * g * L_next * sin_theta_next
        # gravitational potential energy
        delta_y = next_pos[1] - prev_pos[1]
        delta_E_g = m * g * delta_y
        return tau_prev + tau_next + beta * delta_E_g + alpha * L_next



class TrivialPathPlanner(PathPlanner):
    """
    'Plans' a path from the initial position to the first Hold 
    in the environment, which should be within reach (assume that the
    environment only contains one hold besides the initial hold)
    """
    def calculate_path(self) -> np.ndarray:
        return np.array([self.env.start_idx, self.env.goal_idx])

if __name__ == "__main__":
    # plot A* with different edge costs
    env = Environment()
    env.generate_static_random((0, 10, 0, 10), (1, 1), (9, 9), 500, 1, 0.99)
    trajectory_inputs = np.genfromtxt("collocation_sim_results.csv", delimiter=',')
    planner = AStarPathPlanner(env, trajectory_inputs)
    path_energy = planner.calculate_path()
    path_trajectory = planner.calculate_path(trajectories=True)
    path_distance = planner.calculate_path(False)
    paths = {
        "Energy-based": path_energy,
        "Trajectory-based": path_trajectory,
        "Distance-based": path_distance
    }
    file_path = "./env_plot/A_star_test.png"
    planner.plot(env, paths, file_path)
