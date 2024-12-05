"""
The path planner, with multiple options for heuristics.
"""

import numpy as np
import heapq
from environment import Environment
import matplotlib as plt

class PathPlanner:
    def __init__(self, env:Environment):
        self.holds = env.holds
        self.range = env.spacing + env.grasp_radius
        self.start_idx = env.start_idx
        self.goal_idx = env.goal_idx
        self.heuristics = []

    def l2_heuristic(self, curr_idx):
        curr_pos = self.holds[curr_idx].position
        goal_pos = self.holds[self.goal_idx].position
        return np.linalg.norm(np.array(curr_pos) - np.array(goal_pos))

    def edge_cost(self, prev_idx, curr_idx, next_idx, energy=True):
        # simplified torque-based model of simple pendulum
        g = 9.81
        m = 1.0
        prev_pos = np.array(self.holds[prev_idx].position)
        curr_pos = np.array(self.holds[curr_idx].position)
        next_pos = np.array(self.holds[next_idx].position)
        # change pendulum length halfway through arc
        L_prev = np.linalg.norm(prev_pos - curr_pos)
        L_next = np.linalg.norm(next_pos - curr_pos)
        if not energy:
            return L_next
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
        return tau_prev + tau_next + delta_E_g
    
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

    def A_star(self, energy=True):
        open_set = []   # priority queue with (cost, index)
        heapq.heappush(open_set, (0, self.start_idx))
        g_costs = {self.start_idx: 0}
        parent = {}
        while open_set:
            _, curr_idx = heapq.heappop(open_set)
            if curr_idx == self.goal_idx:
                path = [curr_idx]
                while curr_idx in parent:
                    curr_idx = parent[curr_idx]
                    path.append(curr_idx)
                path.reverse()
                return path
            for neighbor_idx in self.get_neighbors(curr_idx):
                new_g = g_costs[curr_idx] + self.edge_cost(curr_idx, neighbor_idx, energy)
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
        colors = ['y', 'r']
        for idx, (label, path) in enumerate(paths.items()):
            if path:
                path_positions = [env.holds[idx].position for idx in path]
                x_coords, y_coords = zip(*path_positions)
                ax.plot(x_coords, y_coords, linestyle='-', linewidth=2, color=colors[idx % len(colors)], label=label)
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_title("A* Paths")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.grid(True)
        plt.savefig(file_path)
        pass

if __name__ == "__main__":
    # plot A* with different edge costs
    env = Environment()
    env.generate_static_random((0, 5, 0, 5), (1, 1), (4, 4), 500, 1, 0.99)
    planner = PathPlanner(env)
    path_energy = planner.A_star()
    path_distance = planner.A_star(False)
    paths = {
        "Energy-based": path_energy,
        "Distance-based": path_distance
    }
    file_path = "./env_plot/A_star_test.png"
    planner.plot(env, paths, file_path)
