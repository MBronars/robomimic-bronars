"""
Utility functions for simulation and robot interaction.

Author: Wonsuhk Jung
"""
import os
import math
import numpy as np
import xml.etree.ElementTree as ET
import torch
from robosuite.robots import Manipulator
# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope

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

def quaternion_to_rotation_matrix(quat):
    """
    Convert a quaternion into a 3x3 rotation matrix.
    
    Parameters:
    quat (list or torch.Tensor): Quaternion in the form [w, x, y, z]

    Returns:
    torch.Tensor: 3x3 rotation matrix
    """
    w, x, y, z = quat

    # Calculate the rotation matrix elements
    R = torch.tensor([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], dtype=torch.float)

    return R

def parse_body(body, params, Tj, K, actuator_map):
    """ Recursively parse a body and its children, collecting parameters """
    # Parse the Kinematic properties
    pos = list(map(float, body.get('pos').split()) if body.get('pos') else [0, 0, 0])
    quat = list(map(float, body.get('quat').split()) if body.get('quat') else [1, 0, 0, 0])
    
    # Parsing inertial properties
    inertial = body.find('inertial')
    if inertial is not None:
        mass = float(inertial.get('mass'))
        inertia = list(map(float, inertial.get('diaginertia').split()))

        I = torch.tensor([
            [inertia[0], 0, 0],
            [0, inertia[1], 0],
            [0, 0, inertia[2]]
        ], dtype=torch.float)

        G = torch.block_diag(I, mass * torch.eye(3))

        # Store parameters
        params['mass'].append(mass)
        params['I'].append(I)
        params['G'].append(G)
        params['com'].append(torch.tensor(pos, dtype=torch.float))
        params['com_rot'].append(torch.tensor(quat, dtype=torch.float))

    # Parsing joint information
    joint = body.find('joint')
    if joint is not None:
        joint_name = joint.get('name')
        axis = list(map(float, joint.get('axis').split()))
        pos_lim = None
        vel_lim = None
        tor_lim = None
        lim_flag = False

        # Handle position limits
        if joint.get('limited') == 'true':
            pos_lim = list(map(float, joint.get('range').split()))
            lim_flag = True

        # Fetching from actuator
        if joint_name in actuator_map:
            actuator = actuator_map[joint_name]
            tor_lim = list(map(float, actuator.get('ctrlrange').split()))

        # Store joint-related data
        params['joint_axes'].append(torch.tensor(axis, dtype=torch.float))
        params['pos_lim'].append(pos_lim)
        params['vel_lim'].append(vel_lim)
        params['tor_lim'].append(tor_lim)
        params['lim_flag'].append(lim_flag)

        # Update transforms
        transform = torch.eye(4, dtype=torch.float)
        transform[:3, :3] = quaternion_to_rotation_matrix(quat)  # Assuming identity for simplicity
        transform[:3, 3] = torch.tensor(pos, dtype=torch.float)
        params['H'].append(transform)
        params['R'].append(transform[:3, :3])
        params['P'].append(transform[:3, 3])

        Tj = Tj @ transform
        K_prev = K
        K = torch.eye(4)
        K[:3, :3], K[:3, 3] = torch.eye(3), torch.tensor(pos, dtype=torch.float)
        params['M'].append(torch.inverse(K_prev) @ transform @ K)

        w = Tj[:3, :3] @ torch.tensor(axis, dtype=torch.float)
        v = torch.cross(-w, Tj[:3, 3])
        params['screw'].append(torch.hstack((w, v)))

    # Recursively parse child bodies
    for child in body.findall('body'):
        parse_body(child, params, Tj, K, actuator_map)

def import_mujoco_robot(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root

def load_single_mujoco_robot_arm_params(xml_file, base_name = "base"):
    # TODO: add the zonotope parsing from the mesh file
    
    root = import_mujoco_robot(xml_file)

    params = {
        'mass': [],             # mass
        'I': [],                # moment of inertia
        'G': [],                # spatial inertia
        'com': [],              # CoM position
        'com_rot': [],          # CoM orientation (rotation)
        'joint_axes': [],       # joint axes
        'H': [],                # transform of ith joint in prev. joint in home config.
        'R': [],                # rotation of the joint transform
        'P': [],                # translation of the joint transform
        'M': [],                # transform of ith CoM in prev. CoM in home config.
        'screw': [],            # screw axes in base
        'pos_lim': [],          # joint position limit
        'vel_lim': [],          # joint velocity limit
        'tor_lim': [],          # joint torque limit
        'lim_flag': [],         # False for continuous, True for everything else
        'link_zonos': [],       # link zonotopes (not applicable in this case, but kept for consistency)
    }

    Tj = torch.eye(4, dtype=torch.float)  # transform of ith joint in base
    K = torch.eye(4, dtype=torch.float)   # transform of ith CoM in ith joint

    # Map actuators to their joints
    actuators = root.findall('.//actuator/motor')
    actuator_map = {actuator.get('joint'): actuator for actuator in actuators}

    # Start parsing from the base body
    base_body = root.find(f'.//worldbody/body[@name="{base_name}"]')
    parse_body(base_body, params, Tj, K, actuator_map)

    params['n_bodies'] = len(params['mass'])
    params['n_joints'] = len(params['joint_axes'])

    return params
