"""
Utility functions for simulation and robot interaction.

Author: Wonsuhk Jung
"""
import os

import torch
import numpy as np
import xml.etree.ElementTree as ET

from collections import deque

from robosuite.robots import Manipulator
# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope

import safediffusion.utils.reachability_utils as ReachUtils

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
    
    Args:
        quat (list or torch.Tensor): Quaternion in the form [w, x, y, z]

    Returns:
        R (torch.Tensor): 3x3 rotation matrix
    """
    w, x, y, z = quat

    # Calculate the rotation matrix elements
    R = torch.tensor([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], dtype=torch.float)

    return R

# ------------------------------------------- #
# XML-file Parser
# ------------------------------------------- #
def is_mesh_name_used_for_collision_check(mesh_name):
    """
    Returns if the mesh_name follows the naming convention that is used for collision check

    Args:
        mesh_name (str): the name of the mesh
    
    Returns:
        (bool): True if the mesh name is for collsion check
    """
    # Ambiguous naming convention, but some xml does not specify the name of the mesh.
    if mesh_name is None:
        return True
    
    elif mesh_name.lower().endswith("vis") or mesh_name.lower().endswith("visual"):
        return False
    
    elif mesh_name.lower().endswith("col") or mesh_name.lower().endswith("collision"):
        return True
    
    else:
        raise NotImplementedError
    

def parse_body(body, params, Tj, K, actuator_map, meshes_map, xml_file):
    """ 
    Recursively parse a body and its children, collecting parameters
    
    Args:
        body: (xml_object), a 
        params: (dict), a params that is shared within all bodies
        Tj (torch.tensor): 4x4 transform matrix
        K

    NOTE: Tips for the xml file
        1) `get(tag)`  returns the property that has a name of tag inside the head <>
        2) `find(tag)` returns the child xml object that has a name of tag
    """
    # Parse position (pos), quat (quat) of the body relative to the parent body
    pos  = list(map(float, body.get('pos').split()) if body.get('pos') else [0, 0, 0])
    quat = list(map(float, body.get('quat').split()) if body.get('quat') else [1, 0, 0, 0])
    
    # Parse inertial properties
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
    
    # Parsing the geom information
    geoms = body.findall('geom')
    if geoms is not None:
        link_zonotope = []
        for geom in geoms:
            geom_type = geom.get('type')
            
            assert geom_type in ["mesh", "box"], f"Unsupported geometry type {geom_type} has been queried."

            if geom_type == "mesh":
                mesh_name = geom.get('name')
                if is_mesh_name_used_for_collision_check(mesh_name):
                    mesh_key = geom.get('mesh')
                    mesh_file = os.path.join(os.path.dirname(xml_file), meshes_map[mesh_key].get('file'))
                    zonotope = ReachUtils.get_zonotope_from_stl_file(stl_file = mesh_file)
                else:
                    continue
            
            elif geom_type == "box":
                geom_size = geom.get('size')
                zonotope = ReachUtils.get_zonotope_from_box_geom(pos  = torch.zeros(3,), 
                                                                    rot  = torch.eye(3), 
                                                                    size = geom_size)
            else:
                raise NotImplementedError
            
            link_zonotope.append(zonotope)

        params['link_zonos'].append(link_zonotope)


    # Recursively parse child bodies
    for child in body.findall('body'):
        parse_body(child, params, Tj, K, actuator_map, meshes_map, xml_file)

def import_mujoco_robot(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root

def load_single_mujoco_robot_arm_params(xml_file, base_name = "base"):
    """
    Load the robot arm params that has the single arm (hence tree is the list)

    Args:
        xml_file (str): the file path for the robot
        base_name (str): the body name that we consider a base
    
    Returns:

    NOTE: check if this body includes "right hand" which is the point we attach the gripper
    """
    root = import_mujoco_robot(xml_file)

    params = {
        'pos_lim': [],          # joint position limit
        'vel_lim': [],          # joint velocity limit
        'joint_axes': [],       # joint axes
        'R': [],                # rotation of the joint transform
        'P': [],                # translation of the joint transform
        'lim_flag': [],         # False for continuous, True for everything else
        'mass': [],             # mass
        'tor_lim': [],          # joint torque limit


        'I': [],                # moment of inertia
        'G': [],                # spatial inertia
        'com': [],              # CoM position
        'com_rot': [],          # CoM orientation (rotation)
        'H': [],                # transform of ith joint in prev. joint in home config.
        'M': [],                # transform of ith CoM in prev. CoM in home config.
        'screw': [],            # screw axes in base
        'link_zonos': [],       # link zonotopes (not applicable in this case, but kept for consistency)
    }

    Tj = torch.eye(4, dtype=torch.float)  # transform of ith joint in base
    K = torch.eye(4, dtype=torch.float)   # transform of ith CoM in ith joint

    # Map actuators to their joints
    actuators = root.findall('.//actuator/motor')
    actuator_map = {actuator.get('joint'): actuator for actuator in actuators}

    # Map mesh to their files
    meshes = root.findall('.//asset/mesh')
    meshes_map = {mesh.get('name') : mesh for mesh in meshes}

    # Start parsing from the base body
    base_body = root.find(f'.//worldbody/body[@name="{base_name}"]')

    parse_body(base_body, params, Tj, K, actuator_map, meshes_map, xml_file)

    params['n_bodies'] = len(params['mass'])
    params['n_joints'] = len(params['joint_axes'])

    return params

class TreeNode:
    def __init__(self, name, pos=None, quat=None, parent=None):
        self.name = name
        self.pos = pos if pos is not None else [0, 0, 0]
        self.quat = quat if quat is not None else [1, 0, 0, 0]
        self.rot  = self._quaternion_to_rotation_matrix(self.quat)
        self.parent = parent
        self.children = []
        self.mass = None
        self.inertia = None
        self.joint_info = {}
        self.geom_info = []

    def add_child(self, child):
        self.children.append(child)

    def _quaternion_to_rotation_matrix(self, quat):
        """
        Convert a quaternion to a rotation matrix.

        Args:
            quat (list): Quaternion as [w, x, y, z].

        Returns:
            torch.tensor: 3x3 rotation matrix.
        """
        w, x, y, z = quat
        return torch.tensor([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ], dtype=torch.float)


class RobotXMLParser:
    def __init__(self, 
                 xml_file, 
                 base_name = "base",
                 dtype     = torch.float64,
                 device    = torch.device):
        """
        Initialize the parser with the XML file path and set up the root of the tree.

        Args:
            xml_file (str): Path to the XML file to parse.
        """
        self.xml_file = xml_file
        self.tree = ET.parse(xml_file)
        self.root_element = self.tree.getroot()
        self.root_node = None  # This will be the root of our body tree
        self.dtype = dtype
        self.device = device

        # Maps for actuators and meshes
        self.actuator_map = self._create_actuator_map()
        self.meshes_map = self._create_meshes_map()

        # Initialize the parsing process starting from the base body
        base_body = self.root_element.find(f'.//worldbody/body[@name="{base_name}"]')
        if base_body is not None:
            self.root_node = self._parse_body(base_body)
        
        self.n_joint      = self.get_n_joint()
        self.joint_axes   = self.get_joint_axes()
        self.rot_skew_sym = self.get_rot_skew_sym()

    def to_tensor(self, x):
        if type(x) == torch.Tensor:
            return x.to(dtype=self.dtype, device=self.device)
        else:
            return torch.tensor(x, dtype=self.dtype, device=self.device)
        

    def _create_actuator_map(self):
        """
        Create a mapping of actuators to their corresponding joints.

        Returns:
            dict: A dictionary mapping joint names to actuator elements.
        """
        actuators = self.root_element.findall('.//actuator/motor')
        return {actuator.get('joint'): actuator for actuator in actuators}

    def _create_meshes_map(self):
        """
        Create a mapping of mesh names to their corresponding mesh file details.

        Returns:
            dict: A dictionary mapping mesh names to mesh elements.
        """
        meshes = self.root_element.findall('.//asset/mesh')
        return {mesh.get('name'): mesh for mesh in meshes}
    
    def _get_zonotope_from_geom_data(self, geom_data):
        # Parse geom information
        geom_type = geom_data["type"]

        if geom_type == "mesh":
            mesh_key  = geom_data["mesh"]
            mesh_scale = self.meshes_map[mesh_key].get("scale")
            mesh_scale = self.to_tensor(list(map(float, mesh_scale.split()))) if mesh_scale is not None else None
            mesh_file = os.path.join(os.path.dirname(self.xml_file), self.meshes_map[mesh_key].get('file'))
            zonotope = ReachUtils.get_zonotope_from_stl_file(stl_file = mesh_file)
            if mesh_scale is not None:
                zonotope = ReachUtils.scale_zonotope(zonotope, scale = mesh_scale)

        elif geom_type == "box":
            geom_size = geom_data["size"]
            zonotope = ReachUtils.get_zonotope_from_box_geom(pos  = torch.zeros(3,), 
                                                             rot  = torch.eye(3), 
                                                             size = geom_size)
        
        else:
            raise NotImplementedError
        
        return zonotope.to(dtype = self.dtype, device=self.device)
    
    def is_geom_name_used_for_collision_check(self, geom_name):
        """
        Returns if the mesh_name follows the naming convention that is used for collision check

        Args:
            mesh_name (str): the name of the mesh
        
        Returns:
            (bool): True if the mesh name is for collsion check
        """
        # Ambiguous naming convention, but some xml does not specify the name of the mesh.
        if geom_name is None:
            return False
        
        elif geom_name.lower().endswith("vis") or geom_name.lower().endswith("visual"):
            return False
        
        elif geom_name.lower().endswith("col") or geom_name.lower().endswith("collision"):
            return True
        
        else:
            raise NotImplementedError

    def _parse_body(self, body, parent_node=None):
        """
        Recursively parse a body and its children, building a tree structure.

        Args:
            body: (xml_object), the current body in the XML tree.
            parent_node: (TreeNode), the parent node in the tree (None for the root body).

        Returns:
            TreeNode: The root of the subtree starting from this body.
        """
        body_name = body.get('name')
        pos = list(map(float, body.get('pos').split())) if body.get('pos') else [0, 0, 0]
        quat = list(map(float, body.get('quat').split())) if body.get('quat') else [1, 0, 0, 0]

        # Create a new tree node for this body
        node = TreeNode(name=body_name,
                         pos=self.to_tensor(pos), 
                         quat=self.to_tensor(quat), 
                         parent=parent_node)

        # If this node has a parent, add it to the parent's children list
        if parent_node:
            parent_node.add_child(node)

        # Parse inertial properties
        inertial = body.find('inertial')
        if inertial is not None:
            mass = float(inertial.get('mass'))
            inertia = list(map(float, inertial.get('diaginertia').split()))
            I = torch.tensor([
                [inertia[0], 0, 0],
                [0, inertia[1], 0],
                [0, 0, inertia[2]]
            ], dtype=torch.float)
            node.mass = mass
            node.inertia = I

        # Parse joint information
        joint = body.find('joint')
        if joint is not None:
            joint_name = joint.get('name')
            axis = list(map(float, joint.get('axis').split())) if joint.get('axis') else [1, 0, 0]
            pos_lim = None
            tor_lim = None
            if joint.get('limited') == 'true':
                pos_lim = list(map(float, joint.get('range').split()))
            if joint_name in self.actuator_map:
                actuator = self.actuator_map[joint_name]
                tor_lim = list(map(float, actuator.get('ctrlrange').split()))
            node.joint_info = {
                'name': joint_name,
                'axis': axis,
                'pos_lim': pos_lim,
                'tor_lim': tor_lim
            }
        
        geoms = body.findall('geom')
        if geoms:
            for geom in geoms:
                geom_type = geom.get('type')
                geom_name = geom.get('name')

                if self.is_geom_name_used_for_collision_check(geom_name):
                    assert geom_type in ["mesh", "box"], f"Unsupported geometry type {geom_type}."
                    geom_data = {
                        'type': geom_type,
                        'name': geom.get('name'),
                        'mesh': geom.get('mesh') if geom_type == "mesh" else None,
                        'size': list(map(float, geom.get('size').split())) if geom_type == "box" else None
                    }
                    geom_data['zonotope'] = self._get_zonotope_from_geom_data(geom_data)
                    node.geom_info.append(geom_data)

        # Recursively parse all child bodies
        for child in body.findall('body'):
            self._parse_body(child, parent_node=node)

        return node

    def get_root_node(self):
        """
        Get the root node of the body tree.

        Returns:
            TreeNode: The root node of the parsed body tree.
        """
        return self.root_node
    
    def get_joint_axes(self):
        """
        Get the joint axes using BFS
        """
        joint_axes = []
        
        queue = deque([self.root_node])

        while queue:
            current_node = queue.popleft()

            # insert action here
            if current_node.joint_info:
                joint_axes.append(current_node.joint_info["axis"])

            # Enqueue all children of the current node
            for child in current_node.children:
                queue.append(child)
        
        joint_axes = torch.tensor(joint_axes)

        return joint_axes
    
    def get_rot_skew_sym(self):
        w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],
                            [[0,0,-1],[0,0,0],[1,0,0]],
                            [[0,1,0],[-1,0,0],[0,0,0.0]]])
        rot_skew_sym = (w@self.joint_axes.T).transpose(0,-1)

        return rot_skew_sym
    
    def get_n_joint(self):
        joint_axes = self.get_joint_axes()
        return joint_axes.shape[0]
    
    def rot(self, q):
        q = q.reshape(q.shape+(1,1))
        W = self.rot_skew_sym
        I = torch.eye(3)

        R = I + torch.sin(q) * W + (1-torch.cos(q)) * W@W

        return R
    
    def forward_kinematics_zono(self, q):
        """
        Perform forward kinematics on the robot using the provided joint angles q.

        Args:
            q (torch.tensor): A tensor of joint angles.

        Returns:
            list: A list of zonotopes representing the forward occupancy of the robot links.
        """
        def _recursive_fk_zono(node, Ri, Pi, joint_index):
            """
            Recursively compute forward kinematics for each node in the tree.

            Args:
                node (TreeNode): The current node in the tree.
                Ri (torch.tensor): The current rotation matrix.
                Pi (torch.tensor): The current position vector.
                joint_index (int): The index of the current joint in the chain.

            Returns:
                int: Updated joint index after processing the current node and its children.
            """

            # Compute zonotope for the current node
            Pi = Ri @ node.pos + Pi
            Ri = Ri @ node.rot
            
            if node.joint_info:
                # Update the rotation and position based on the current joint
                Ri = Ri @ self.rot(q)[[joint_index]].squeeze(0)  # Apply joint rotation
                joint_index += 1
            
            if node.geom_info:
                for geom in node.geom_info:
                    zonotope = Ri @ geom['zonotope'] + Pi
                    self.arm_zonos.append(zonotope)

            # Recursively process each child node
            for child in node.children:
                joint_index = _recursive_fk_zono(child, Ri, Pi, joint_index)

            return joint_index

        # Initialize rotation and position based on the root node
        Ri = self.to_tensor(torch.eye(3))
        Pi = self.to_tensor(self.root_node.pos)
        self.arm_zonos = []

        # Start recursive forward kinematics computation
        _recursive_fk_zono(self.root_node, Ri, Pi, 0)

        return self.arm_zonos
    
    def forward_kinematics(self, q, name):
        """
        Perform forward kinematics on the robot using the provided joint angles q.

        Args:
            q (torch.tensor): A tensor of joint angles.

        Returns:
            list: A list of zonotopes representing the forward occupancy of the robot links.
        """
        self.fk_done = False
        def _recursive_fk(node, Ri, Pi, joint_index):
            """
            Recursively compute forward kinematics for each node in the tree.

            Args:
                node (TreeNode): The current node in the tree.
                Ri (torch.tensor): The current rotation matrix.
                Pi (torch.tensor): The current position vector.
                joint_index (int): The index of the current joint in the chain.

            Returns:
                int: Updated joint index after processing the current node and its children.
            """

            # Compute zonotope for the current node
            Pi = Ri @ node.pos + Pi
            Ri = Ri @ node.rot
            
            if node.joint_info:
                # Update the rotation and position based on the current joint
                Ri = Ri @ self.rot(q)[[joint_index]].squeeze(0)  # Apply joint rotation
                joint_index += 1

            if node.name == name:
                self.fk_done = True
                self.R = Ri
                self.P = Pi

            # Recursively process each child node
            for child in node.children:
                if not self.fk_done:
                    joint_index = _recursive_fk(child, Ri, Pi, joint_index)

            return joint_index

        # Initialize rotation and position based on the root node
        Ri = self.to_tensor(torch.eye(3))
        Pi = self.to_tensor(self.root_node.pos)

        # Start recursive forward kinematics computation
        _recursive_fk(self.root_node, Ri, Pi, 0)

        if self.fk_done:
            return self.R, self.P
        
        else:
            raise NotImplementedError