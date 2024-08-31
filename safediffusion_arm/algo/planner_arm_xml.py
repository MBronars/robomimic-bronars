import os

import numpy as np
import torch

import robosuite

from safediffusion.algo.helper import ReferenceTrajectory
import safediffusion.utils.reachability_utils as ReachUtils
import safediffusion.utils.robot_utils as RobotUtils
import safediffusion.utils.transform_utils as TransformUtils
import safediffusion.utils.math_utils as MathUtils

from safediffusion_arm.algo.planner_arm import ArmtdPlanner


# GLOBAL ARM REGISTRATION
ROBOT_JOINT_VEL_LIMIT         = dict(kinova3 = [1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218],
                                     panda   = [2.6180, 2.6180, 2.6180, 2.6180, 3.1416, 3.1416, 3.1416])
ROBOT_EEF_BODY_NAME           = dict(kinova3 = "Bracelet_Link",
                                     panda   = "link7")
ROBOT_GRIPPER_SITE_BODY_NAME  = dict(kinova3 = "right_hand",
                                     panda   = "right_hand")

# GLOBAL GRIPPER REGISTRATION
GRIPPER_INIT_QPOS               = dict(robotiq_gripper_85 = [-0.026, -0.267, -0.200, -0.026, -0.267, -0.200],
                                       panda_gripper      = [0.020833, -0.020833])
GRIPPER_LEFT_FINGER_BODY_NAME   = dict(robotiq_gripper_85 = "left_inner_finger",
                                       panda_gripper      = "leftfinger")
GRIPPER_RIGHT_FINGER_BODY_NAME  = dict(robotiq_gripper_85 = "right_inner_finger",
                                       panda_gripper      = "rightfinger")
GRIPPER_BASE_BODY_NAME          = dict(robotiq_gripper_85 = "robotiq_85_adapter_link",
                                       panda_gripper      = "right_gripper")

class ArmtdPlannerXML(ArmtdPlanner):
    """
    Armtd Planner where the robot model is loaded using xml file.
    """
    def __init__(self,
                 action_config, 
                 robot_name = "Kinova3",
                 gripper_name = None,
                 **kwargs):
        
        self.robot_xml_file          = os.path.join(robosuite.__path__[0], 
                                                 f"models/assets/robots/{robot_name.lower()}/robot.xml")
        
        if gripper_name is not None:
            self.gripper_xml_file    = os.path.join(robosuite.__path__[0], 
                                                    f"models/assets/grippers/{gripper_name.lower()}.xml")
                
        assert robot_name.lower() in ROBOT_JOINT_VEL_LIMIT.keys(), f"The joint velocity limit of {robot_name} has not been registered."
        assert os.path.exists(self.robot_xml_file)

        super().__init__(action_config, 
                         robot_name   = robot_name, 
                         gripper_name = gripper_name, 
                         **kwargs)

    def get_robot_n_links(self):
        params = RobotUtils.load_single_mujoco_robot_arm_params(self.robot_xml_file)
        n_links = params['n_joints']

        return n_links

    def load_robot_config_and_joint_limits(self):
        """
        Read the params (n_joints, P, R, joint_axes) from the robosuite xml file

        TODO: remove dependency on load_single_mujoco_robot_arm_params
        """
        # parse parameters from the xml file
        
        params = RobotUtils.load_single_mujoco_robot_arm_params(self.robot_xml_file)

        self.params = {'n_joints': params['n_joints'],
                       'P'       : [self.to_tensor(p) for p in params['P']],
                       'R'       : [self.to_tensor(r) for r in params['R']]}
        
        # joint axes
        self.joint_axes   = self.to_tensor(torch.stack(params['joint_axes']))
        w = self.to_tensor([[[0,0,0],[0,0,1],[0,-1,0]],
                            [[0,0,-1],[0,0,0],[1,0,0]],
                            [[0,1,0],[-1,0,0],[0,0,0.0]]])
        self.rot_skew_sym = (w@self.joint_axes.T).transpose(0,-1)
        
        # joint position limit
        self.pos_lim        = self._process_pos_lim(params['pos_lim'])
        self.lim_flag       = np.array(params['lim_flag'])
        self.actual_pos_lim = self.pos_lim[self.lim_flag]
        self.n_pos_lim      = int(sum(self.lim_flag))

        # joint velocity limit
        self.vel_lim        = self.to_tensor(ROBOT_JOINT_VEL_LIMIT[self.robot_name.lower()])

        self.robot_eef_name          = ROBOT_EEF_BODY_NAME[self.robot_name.lower()]
        self.robot_gripper_site_name = ROBOT_GRIPPER_SITE_BODY_NAME[self.robot_name.lower()]

        # This directs to the gripper site
        self.arm_parser             = RobotUtils.RobotXMLParser(xml_file = self.robot_xml_file, 
                                                                dtype    = self.dtype, 
                                                                device   = self.device)
        self.T_last_link_to_gripper_base = TransformUtils.make_pose(rot = self.arm_parser.rot_params[-1], 
                                                                    pos = self.arm_parser.pos_params[-1])

    def _process_pos_lim(self, pos_lim):
        """
        Process the joint position limit compatible to ArmtdPlanner

        Args:
            pos_lim (list): list length of n_joint, each list has 2D array of joint limit [min, max].
                            If the joint limit does not exist, it says None.
        
        Returns:
            pos_lim_processed (tensor): an array size of (n_joint, 2), 
                                        1st column indicates max, 2nd column indicates min.
                                        None has been replaced to [pi, -pi]
        """
        PI = torch.pi
        
        pos_lim_processed = [(torch.tensor([PI, -PI]) 
                              if x is None else torch.tensor([max(x), min(x)])) 
                              for x in pos_lim]
        
        pos_lim_processed = self.to_tensor(torch.stack(pos_lim_processed))

        return pos_lim_processed
    
    def load_robot_link_zonotopes(self):
        """
        Load robot link zonotopes by parsing it from the xml file
        """
        self.arm_parser          = RobotUtils.RobotXMLParser(self.robot_xml_file, 
                                                             dtype  = self.dtype, 
                                                             device = self.device)
        arm_link_zonos_stl       = self.arm_parser.link_zonos_stl

        # TODO: Exclude the base link and the end-effector link(?)
        arm_link_zonos_stl       = arm_link_zonos_stl[1:]
        assert len(arm_link_zonos_stl) == self.n_links

        # for single-arm robot, each body has one geom
        self._link_zonos_stl     = [link[0] for link in arm_link_zonos_stl]
        self._link_polyzonos_stl = [link_zono.to_polyZonotope().to(dtype = self.dtype, device = self.device) 
                                    for link_zono in self._link_zonos_stl]
    
    def load_gripper_config(self, gripper_name):
        """
        Load the gripper model using RobotXMLParser

        Assumes there exists only binary status for the gripper: closed & open

        TODO: closed_gripper_qpos, open_gripper_qpos, gripper_base_name should be
              automatically parsed
        """
        self.gripper_base_name          = GRIPPER_BASE_BODY_NAME[gripper_name.lower()]
        self.gripper_left_finger_name   = GRIPPER_LEFT_FINGER_BODY_NAME[gripper_name.lower()]
        self.gripper_right_finger_name  = GRIPPER_RIGHT_FINGER_BODY_NAME[gripper_name.lower()]
        self.gripper_init_qpos          = self.to_tensor(GRIPPER_INIT_QPOS[gripper_name.lower()])

        self.gripper_parser = RobotUtils.RobotXMLParser(self.gripper_xml_file, 
                                                        base_name = self.gripper_base_name,
                                                        dtype     = self.dtype,
                                                        device    = self.device)
        
        # relative base from gripper_base to the grasping site
        qpos_init                    = self.gripper_init_qpos
        T_left_finger                = TransformUtils.make_pose(*self.gripper_parser.forward_kinematics(qpos_init, self.gripper_left_finger_name))
        T_right_finger               = TransformUtils.make_pose(*self.gripper_parser.forward_kinematics(qpos_init, self.gripper_right_finger_name))
        self.T_gripper_base_to_grasp = TransformUtils.pose_interp(T_left_finger, T_right_finger, ratio = 0.5)

    def get_arm_eef_pos_at_q(self, q):
        """
        Return the position of the robot end-effector in world frame at given joint angle
        It does by doing the forward kinematics of the robot.

        Args
            qpos: (n_links,) torch tensor

        Output
            P: position of the end-effector relative to 
        """ 
        T_world_to_eef   = self.get_arm_link_pose_at_q(self.robot_eef_name, q)

        _, p = TransformUtils.pose_to_rot_pos(T_world_to_eef)

        return p

    def get_arm_zonotopes_at_q(self, q):
        """
        Get zonotope representation of the arm at the given joint angle q

        Args:
            q (torch.tensor) (n_joint,) the query joint angle

        Returns:
            arm_zonos (list <zonotope>)
        """

        arm_zonos = self.arm_parser.forward_kinematics_zono(q)

        rot, pos = TransformUtils.pose_to_rot_pos(self.T_world_to_arm_base)

        arm_zonos = [ReachUtils.transform_zonotope(zono, pos=pos, rot=rot)
                     for zono in arm_zonos]

        return arm_zonos
    
    def get_arm_link_pose_at_q(self, link_name, q):
        """
        Get the relative pose of the link of given name at joint angle `q` from the world

        Args:
            q (torch.tensor)       : the query joint angle
            name (str)             : the name of the link of focus
            return_gradient (bool) : True if returns the gradient of the pose

        Returns:
            T (torch.tensor)       : (4, 4) transformation matrix
            grad_T (torch.tensor)  : (n_joint, 4, 4) gradient of the transformation matrix

        TODO: maybe check if link_name is valid in this func?
        """
        q                  = self.to_tensor(q)
        R, p               = self.arm_parser.forward_kinematics(q = q, name = link_name)
        T_arm_base_to_link = TransformUtils.make_pose(rot = R, pos = p)
        T_world_to_link    = self.T_world_to_arm_base @ T_arm_base_to_link
        T_world_to_link    = self.to_tensor(T_world_to_link)
        
        return T_world_to_link
    
    def get_arm_jacobian_at_q(self, q):
        """
        Get the Jacobian matrix of the arm at the given joint angle q

        Args:
            q (torch.tensor) (n_joint,) the query joint angle

        Returns:
            J (torch.tensor) (6, n_joint) the Jacobian matrix
        """
        J = self.arm_parser.compute_space_jacobian(q)

        return self.to_tensor(J)
    
    def get_grasping_pos_at_q(self, q, return_gradient = False):
        """
        Get the grasping position relative to the world

        Args:
            q (np.darray) (n_joint,) arm joint angle
        
        Returns:
            pos_world_to_grasp (np.darray) (3,) the grasping position relative to the world
            grad_pos_world_to_grasp (np.darray) (7, 3)

        TODO: unify the naming convention as base, last_link, gripper_base, grasping_site
        """
        assert self.gripper_name is not None

        T_world_to_gripper_base = self.get_arm_link_pose_at_q(q = q, link_name = self.robot_gripper_site_name)
        T_world_to_grasp        = T_world_to_gripper_base @ self.T_gripper_base_to_grasp
        _, pos_world_to_grasp   = TransformUtils.pose_to_rot_pos(T_world_to_grasp)
        pos_world_to_grasp      = self.to_tensor(pos_world_to_grasp)
        

        if return_gradient:
            # TODO: cannot further proceed: do not know how to get the manipulator jacobian of the gripper
            J_eef          = self.arm_parser.compute_space_jacobian(q = q, name = self.robot_gripper_site_name)

            # transform jacobian to the world frame
            R_world_to_arm_base, _ = TransformUtils.pose_to_rot_pos(self.T_world_to_arm_base)
            PHI = torch.zeros((6, 6))
            PHI[:3, :3] = R_world_to_arm_base
            PHI[3:, 3:] = R_world_to_arm_base
            J_world_to_eef = PHI @ J_eef

            # TODO: need to transform the jacobian to the grasping frame
            grad_pos_world_to_grasp = self.to_tensor(J_world_to_eef[3:, :])

            return pos_world_to_grasp, grad_pos_world_to_grasp
        
        else:
            return pos_world_to_grasp
    
    def get_gripper_zonotopes_at_q(self, q, T_frame_to_base, use_approximation = False):
        """
        Get the gripper zonotopes at the given joint angle q
        [frame: world]
        
        Args:
            q (torch.tensor) the gripper joint angle

        Returns:
            gripper_zonos (list <zonotope>)

        NOTE: [0: hand_collision (mesh), 
               1: left_outer_knuckle_collision (mesh), 
               2: left_outer_finger_collision (mesh),
               3: left_inner_finger_collision (mesh),
               4: left_fingertip_collision (box),
               5: left_fingerpad_collision (box),
               6: left_inner_knuckle_collision (mesh),
               7-12: right **]
        """
        gripper_zonos = self.gripper_parser.forward_kinematics_zono(q)

        if use_approximation:
            gripper_zonos = ReachUtils.get_bounding_box_of_zonotope_lists(gripper_zonos)
            gripper_zonos = [gripper_zonos]
            
        rot, pos = TransformUtils.pose_to_rot_pos(T_frame_to_base)

        gripper_zonos = [ReachUtils.transform_zonotope(zono, pos=pos, rot=rot)
                            for zono in gripper_zonos]

        return gripper_zonos
    
    def get_forward_occupancy_from_plan(self, plan, only_end_effector = False, vis_gripper = False, gripper_init_qpos = None):
        """
        Get series of forward occupancy zonotopes from the given plan

        Args:
            plan (ReferenceTrajectory), the plan that contains the joint angle, joint velocity
            only_end_effector (bool), get forward occupancy of only end effector

        Returns:
            FO_link (zonotopes), the zonotope list of plan length
        """
        assert isinstance(plan, ReferenceTrajectory)
        assert not (vis_gripper and gripper_init_qpos is None)

        FO_link = []

        for i in range(len(plan)):
            FO_i = self.get_arm_zonotopes_at_q(plan.x_des[i])
            if only_end_effector:
                FO_i = [FO_i[-1]]
            
            if vis_gripper:
                T_world_to_gripper_base = self.get_arm_link_pose_at_q(self.robot_gripper_site_name, plan.x_des[i]) 
                FO_i_gripper = self.get_gripper_zonotopes_at_q(gripper_init_qpos, 
                                                               T_frame_to_base   = T_world_to_gripper_base,
                                                               use_approximation = self.use_gripper_approximation)

                FO_i.extend(FO_i_gripper)

            FO_link.append(FO_i)

        return FO_link