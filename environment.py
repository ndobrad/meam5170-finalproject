"""
Defines the 2D environment for the brachiating robot, as well as functions to generate simple environment types.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class Hold:
    def __init__(self, position, size, is_latched=False, is_dynamic=False, movement_type=None, movement_params=None):
        """
        Args:
            position (tuple): the (x, y) position of the hold
            size (double): the radius of the graspable area of the hold
            is_latched (bool): whether the robot is latched onto the hold
            is_dynamic (bool, optional): whether the hold is dynamic, defaults to False
            movement_type (str, optional): how the hold moves on contact if dynamic, e.g. linear, oscillating, defaults to None
            movement_params (list, optional): parameters for movement type, e.g. direction, speed, defaults to None
        """
        self.position = np.array(position)
        self.size = size
        self.is_latched = is_latched
        self.is_dynamic = is_dynamic
        self.movement_type = movement_type
        self.movement_params = movement_params
        self.is_highlighted = False
    
    def update_position(self, timestep):
        """
        Args:
            timestep (float): dt (sec), used to compute the new position of a dynamic hold
        Returns:
            None, hold is updated in place
        """
        # TODO
        pass


class Environment:
    def __init__(self, xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0, grasp_radius=0.15, initial_hold=None):
        """
        Args:
            xmin (float): grid boundary
            xmax (float): grid boundary
            ymin (float): grid boundary
            ymax (float): grid boundary
            grasp_radius: default radius around each hold within which end effector can latch
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.grasp_radius = grasp_radius
        self.spacing = None

        self.holds = []
        self.start_idx = None
        self.goal_idx = None
        
        if initial_hold is not None:
            self.holds.append(Hold(initial_hold, self.grasp_radius))
            self.start_idx = len(self.holds) - 1

    
    def add_hold(self, position, size, is_dynamic=False, movement_type=None, movement_params=None):
        """
        helper function to add a hold to an environment
        """
        self.holds.append(Hold(position, size, False, is_dynamic, movement_type, movement_params))

    
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
    
    def generate_single_hold(self, pos:np.ndarray):
        """
        Add a single hold to the environment and set it as the goal
        Returns:
            The index of the newly-added hold
        """
        new_hold = Hold(pos, self.grasp_radius, is_dynamic=False)
        self.holds.append(new_hold)
        self.goal_idx = len(self.holds) - 1
        return len(self.holds) - 1

    def generate_static_monkey_bars(self, num_holds, spacing):
        """
        Generates an environment of equidistant static holds in a line, from start to goal.

        Args:
            num_holds (int): number of holds
            spacing (float): space (m) between holds
        Returns:
            None, environment is generated in place
        """
        self.spacing = spacing
        buffer = 3 * spacing
        self.xmin = 0
        self.xmax = num_holds * spacing + (2 * buffer)
        self.ymin = 0
        self.ymax = 2 * buffer
        for i in range(num_holds):
            x = buffer + (i * spacing)
            y = buffer
            self.add_hold((x, y), self.grasp_radius)
        self.start_idx = 0
        self.goal_idx = num_holds - 1
            
            
    def generate_vertical_monkey_bars(self, num_holds, spacing):
        """
        Generates an environment of equidistant static holds in a line, from start to goal.

        Args:
            num_holds (int): number of holds
            spacing (float): space (m) between holds
        Returns:
            None, environment is generated in place
        """
        self.spacing = spacing
        buffer = 3 
        self.xmin = 0
        self.xmax = (2 * buffer)
        self.ymin = 0
        self.ymax = (num_holds-1) * spacing + (2 * buffer)
        for i in range(num_holds):
            y = buffer + ((num_holds-(i+1)) * spacing)
            x = buffer
            self.add_hold((x, y), self.grasp_radius)
        self.start_idx = 0
        self.goal_idx = num_holds - 1

    def generate_static_random(self, grid_bounds, start, goal, num_holds, spacing, goal_bias=0.95, seed=17):
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
        self.spacing = spacing
        self.xmin, self.xmax, self.ymin, self.ymax = grid_bounds
        random.seed(seed)
        goal_reachable = False
        self.add_hold(start, self.grasp_radius)
        self.start_idx = 0

        def weighted_sample():
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
        
        # RRT
        while len(self.holds) < num_holds:
            if random.random() < goal_bias:
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
            self.add_hold(new_node, self.grasp_radius)
            if l2_dist(new_node, goal) <= spacing:
                goal_reachable = True
        self.add_hold(goal, self.grasp_radius)
        self.goal_idx = len(self.holds) - 1
        
        # self.xmin -= spacing + 50
        # self.xmax += spacing+ 50
        # self.ymin -= spacing+ 50
        # self.ymax += spacing+ 50
        
        return goal_reachable
        
        
    def generate_random_uniform(self, bounds, start, goal, num_holds, seed=17):
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        random.seed(seed)
        
        self.add_hold(start, self.grasp_radius)
        self.add_hold(goal, self.grasp_radius)
        self.start_idx = 0
        self.goal_idx = 1
        while len(self.holds) < num_holds:
            x = random.uniform(self.xmin, self.xmax)
            y = random.uniform(self.ymin, self.ymax)
            self.add_hold((x,y),self.grasp_radius)
        
        
    def get_relative_position(self, base_hold_index, other_hold_index):
        assert base_hold_index is not None
        assert other_hold_index is not None
        return self.holds[int(other_hold_index)].position - self.holds[int(base_hold_index)].position
    
    
    def plot(self, path):
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

if __name__ == "__main__":
    # test plotting
    env = Environment()
    # env.generate_static_monkey_bars(10, 1)
    # env.holds[3].is_latched = True
    # env.holds[4].is_latched = True
    # path = "./env_plot/monkey_bar_test.png"
    reachable = env.generate_static_random((0, 10, 0, 10), (1, 1), (9, 9), 500, 1, 0.99)
    # reachable = env.generate_static_random((0, 5, 0, 5), (1, 1), (4, 4), 500, 1, 0.99)
    print(reachable)
    path = "./env_plot/static_random_test.png"
    env.plot(path)
