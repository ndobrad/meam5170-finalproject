"""
Defines the 2D environment for the brachiating robot, as well as functions to generate simple environment types.
"""

import matplotlib.pyplot as plt
import random
import numpy as np

class Hold:
    def __init__(self, position, is_latched=False, is_dynamic=False, movement_type=None, movement_params=None):
        """
        Args:
            position (tuple): the (x, y) position of the hold
            is_latched (bool): whether the robot is latched onto the hold
            is_dynamic (bool, optional): whether the hold is dynamic, defaults to False
            movement_type (str, optional): how the hold moves on contact if dynamic, e.g. linear, oscillating, defaults to None
            movement_params (list, optional): parameters for movement type, e.g. direction, speed, defaults to None
        """
        self.position = position
        self.is_latched = is_latched
        self.is_dynamic = is_dynamic
        self.movement_type = movement_type
        self.movement_params = movement_params
    
    def update_position(self, timestep):
        """
        Args:
            timestep (float): dt (sec), used to compute the new position of the hold
        Returns:
            None, hold is updated in place
        """
        # TODO
        pass


class Environment:
    def __init__(self, xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0, grasp_radius=None):
        """
        Args:
            xmin (float): grid boundary
            xmax (float): grid boundary
            ymin (float): grid boundary
            ymax (float): grid boundary
            grasp_radius: radius around hold within which end effector can latch
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.grasp_radius = grasp_radius
        self.holds = []
        self.start_idx = None
        self.goal_idx = None
    
    def add_hold(self, x, y, is_dynamic=False, movement_type=None, movement_params=None):
        """
        helper function to add a hold to an environment
        """
        self.holds.append(Hold((x, y), False, is_dynamic, movement_type, movement_params))

    
    def update_hold(self, timestep, velocity):
        """
        Args:
            timestep (float): dt (sec) for update
            velocity (tuple): (x, y) float values of robot velocity
        Returns:
            None, holds are updated in place
        """
        # TODO: update hold which is latched
        pass

    def generate_static_monkey_bars(self, num_holds, spacing):
            """
            Generates an environment of equidistant static holds in a line, from start to goal.

            Args:
                num_holds (int): number of holds
                spacing (float): space (m) between holds
            Returns:
                None, environment is generated in place
            """
            buffer = 3 * spacing
            self.xmin = 0
            self.xmax = num_holds * spacing + (2 * buffer)
            self.ymin = 0
            self.ymax = 2 * buffer
            for i in range(num_holds):
                x = buffer + (i * spacing)
                y = buffer
                self.add_hold(x, y)
            self.start_idx = 0
            self.goal_idx = num_holds - 1

    def generate_static_random(self, grid_bounds, start, goal, num_holds, spacing):
            """
            Args:
                grid_bounds (tuple): (xmin, xmax, ymin, ymax) float values for grid boundary
                start (tuple): (x, y) float values of starting hold position
                goal (tuple): (x, y) float values of target hold position
                num_holds (int): number of holds
                spacing (float): maximum space (m) between any two holds
            Returns:
                None, environment is generated in place
            """
            # TODO: generate environment
            random.seed(17)
            goal_reachable = False
            self.holds.append(Hold(start))
            self.start_idx = 0

            def weighted_sample():
                #  TODO: grid based sampling to enforce balanced hold density
                grid = {}   # (cell bottom left corner coords, num holds in cell)
                for hold in self.holds:
                    cell = (hold.position[0] // spacing, hold.position[1] // spacing)
                    grid[cell] = grid.get(cell, 0) + 1
                min_density_cell = min(grid, key=lambda cell: grid[cell])
                x = random.uniform(min_density_cell[0] * spacing, (min_density_cell[0] + 1) * spacing)
                y = random.uniform(min_density_cell[1] * spacing, (min_density_cell[1] + 1) * spacing)
                return (x, y)
            
            def l2_dist(p1, p2):
                return np.linalg.norm(np.array(p1) - np.array(p2))
            
            #  TODO: RRT
            while len(self.holds) < num_holds:
                if random.random() < 0.95:
                    sample = weighted_sample()
                else:
                    sample = goal
                nearest_node = min(self.holds, key=lambda h: l2_dist(sample, h.position))
                nearest_pos = nearest_node.position
                dist = l2_dist(sample, nearest_pos)
                rand_scale = random.uniform(0.5, 1.0)
                new_x = nearest_pos[0] + (sample[0] - nearest_pos[0]) / dist * spacing * rand_scale
                new_y = nearest_pos[1] + (sample[1] - nearest_pos[1]) / dist * spacing * rand_scale
                new_node = (new_x, new_y)
                self.holds.append(Hold(new_node))
                if l2_dist(new_node, goal) <= spacing:
                    goal_reachable = True
            self.holds.append(Hold(goal))
            self.goal_idx = len(self.holds) - 1

            return goal_reachable
    
    def plot(self, path):
        #  TODO: plot simple graph of environment
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
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_title("Grid")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.grid(True)
        plt.savefig(path)
        pass

# test plotting
env = Environment()
# env.generate_static_monkey_bars(10, 1)
# env.holds[3].is_latched = True
# env.holds[4].is_latched = True
# path = "./env_plot/monkey_bar_test.png"
reachable = env.generate_static_random((0, 0, 10, 10), (1, 1), (9, 9), 1000, 1)
print(reachable)
path = "./env_plot/static_random_test.png"
env.plot(path)
