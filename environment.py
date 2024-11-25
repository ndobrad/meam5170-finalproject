"""
Defines the 2D environment for the brachiating robot, as well as functions to generate simple environment types.
"""
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
        # TODO: how to handle weight transfer? 
        # pass in separate vector to be used if dynamic? 
        # concurrent updates of both holds or global state?
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
    def __init__(self, xmin, xmax, ymin, ymax, grasp_radius):
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
    
    def update_holds(self, timestep):
        """
        Args:
            timestep (float): dt (sec) for update
        Returns:
            None, holds are updated in place
        """
        # TODO: update holds which are latched
        pass
    
    def generate_single_hold(self, acrobot_full_length):
        new_hold = Hold(np.array([acrobot_full_length/2, acrobot_full_length/2]),is_dynamic=False)
        self.holds.append(new_hold)

    def generate_static_monkey_bars(self, num_holds, spacing):
            """
            Generates an environment of equidistant static holds in a line, from start to goal.

            Args:
                num_holds (int): number of holds
                spacing (float): space (m) between holds
            Returns:
                None, environment is generated in place
            """
            # TODO: generate environment
            pass

    def generate_static_random(self, grid_bounds, start, goal, num_holds, spacing):
            """
            Args:
                grid_bounds (tuple): (xmin, xmax, ymin, ymax) float values for grid boundary
                start (tuple): (x, y) float values of starting hold
                goal (tuple): (x, y) float values of target hold
                num_holds (int): number of holds
                spacing (float): maximum space (m) between any two holds
            Returns:
                None, environment is generated in place
            """
            # TODO: generate environment with seed
            pass
    
    def visualize(self):
        #  TODO: plot simple graph of environment
        pass
