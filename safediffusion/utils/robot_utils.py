"""
Utility functions for simulation and robot interaction.

Author: Wonsuhk Jung
"""

from robosuite.robots import Manipulator
from zonopy.contset.zonotope import zonotope

def check_robot_collision(env, object_names_to_grasp=set(), verbose=False):
    """
    Returns True if the robot is in collision with the environment or gripper is in collision with non-graspable object.

    Args
        env: (MujocoEnv) environment
        object_ignore: (set) set of object geoms to ignore for collision checking
    """
    assert isinstance(env.robots[0], Manipulator)
    assert isinstance(object_names_to_grasp, set)

    robot_model = env.robots[0].robot_model
    gripper_model = env.robots[0].gripper

    # Extract contact geoms
    # (1) Check robot collision with objects: the non-gripper should not touch anything
    contacts_with_robot = env.get_contacts(robot_model)
    if contacts_with_robot: 
        if verbose:
            print("Robot collision with: ", contacts_with_robot)
        return True
        
    # (2) Check gripper collision with objects
    contacts_with_gripper = env.get_contacts(gripper_model)
    if contacts_with_gripper and not contacts_with_gripper.issubset(object_names_to_grasp):
        if verbose:
            contacts_with_gripper = contacts_with_gripper.difference(object_names_to_grasp)
            print("Gripper collision with: ", contacts_with_gripper)
        return True

    return False