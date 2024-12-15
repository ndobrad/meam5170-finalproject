"""
The path planner, with multiple options for heuristics.
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt
from environment import Environment
from controller_base import PathPlanner


class AStarPathPlanner(PathPlanner):
    def __init__(self, env:Environment, trajectory_inputs=None):
        super().__init__(env)
        self.bounds = (env.xmin, env.xmax, env.ymin, env.ymax)
        self.holds = env.holds
        self.range = env.spacing + env.grasp_radius
        self.start_idx = env.start_idx
        self.goal_idx = env.goal_idx
        self.trajectory_inputs = trajectory_inputs
        # normalize trajectory optimization grid
        traj_x_min, traj_x_max, traj_y_min, traj_y_max = [0.0, 0.0, 2.0, 2.0]
        trajectory_inputs[:, 0:2] = (trajectory_inputs[:, 0:2] - [traj_x_min, traj_y_min]) / [
            (traj_x_max - traj_x_min),
            (traj_y_max - traj_y_min)
        ]
        trajectory_inputs[:, 2:4] = (trajectory_inputs[:, 2:4] - [traj_x_min, traj_y_min]) / [
            (traj_x_max - traj_x_min),
            (traj_y_max - traj_y_min)
        ]

    def l2_heuristic(self, curr_idx):
        curr_pos = self.holds[curr_idx].position
        goal_pos = self.holds[self.goal_idx].position
        return np.linalg.norm(np.array(curr_pos) - np.array(goal_pos))
    
    def get_closest_trajectory_input(self, prev_idx, next_idx):
        # normalize hold pos within env
        prev_pos = np.array(self.holds[prev_idx].position)
        prev_pos_scaled = (prev_pos[0] - [self.bounds[0], self.bounds[2]]) / [
            (self.bounds[1] - self.bounds[0]),
            (self.bounds[3] - self.bounds[2])            
        ]
        next_pos = np.array(self.holds[next_idx].position)
        next_pos_scaled = (next_pos[0] - [self.bounds[0], self.bounds[2]]) / [
            (self.bounds[1] - self.bounds[0]),
            (self.bounds[3] - self.bounds[2])            
        ]
        prev_next_vec = np.array(next_pos_scaled) - np.array(prev_pos_scaled)
        prev_next_mag = np.linalg.norm(prev_next_vec)
        prev_next_unit = prev_next_vec / prev_next_mag if prev_next_mag > 0 else np.zeros_like(prev_next_vec)
        start_end_units = self.trajectory_inputs[:, 2:4] - self.trajectory_inputs[:, 0:2]
        start_end_mags = np.linalg.norm(start_end_units, axis=1)
        # orientation similarity
        cos_similarities = np.dot(start_end_units, prev_next_unit)
        max_cos_similarity = np.max(cos_similarities)
        closest_orientation_rows = self.trajectory_inputs[cos_similarities == max_cos_similarity]
        # distance similarity
        start_end_mags_closest = start_end_mags[cos_similarities == max_cos_similarity]
        closest_distances = np.abs(start_end_mags_closest - prev_next_mag)
        closest_pairs = closest_orientation_rows[np.argmin(closest_distances)]
        # minimum positive input
        valid_closest_pairs = closest_pairs[closest_pairs[:, 7] >= 0]
        if valid_closest_pairs.size == 0:
            return None
        return np.min(valid_closest_pairs[:, 7])

    def edge_cost(self, prev_idx, curr_idx, next_idx, energy=True, trajectories=False):
        # simplified torque-based model of simple pendulum
        g = 9.81
        m = 10.0
        alpha = 0.0
        beta = 0.0
        prev_pos = np.array(self.holds[prev_idx].position)
        curr_pos = np.array(self.holds[curr_idx].position)
        next_pos = np.array(self.holds[next_idx].position)
        # change pendulum length halfway through arc
        L_prev = np.linalg.norm(prev_pos - curr_pos)
        L_next = np.linalg.norm(next_pos - curr_pos)
        if not energy:
            return L_next
        if trajectories:
            return self.get_closest_trajectory_input(prev_idx, next_idx)
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

    def calculate_path(self, energy=True, trajectories=False):
        """
        Calculate path using A*
        """
        open_set = []   # priority queue with (cost, index)
        heapq.heappush(open_set, (0, self.start_idx))
        closed_set = set()
        g_costs = {self.start_idx: 0}
        parent = {}
        while open_set:
            _, curr_idx = heapq.heappop(open_set)
            if curr_idx in closed_set:
                continue
            closed_set.add(curr_idx)
            if curr_idx == self.goal_idx:
                path = [curr_idx]
                while curr_idx in parent:
                    curr_idx = parent[curr_idx]
                    path.append(curr_idx)
                path.reverse()
                return path
            for neighbor_idx in self.get_neighbors(curr_idx):
                if neighbor_idx in closed_set:
                    continue
                new_g = g_costs[curr_idx] + self.edge_cost(curr_idx, neighbor_idx, energy, trajectories)
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
    trajectory_inputs = np.genfromtxt("placeholder_input.csv", delimiter=',')
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
