import torch
import numpy as np

from armtd.environments.arm_3d import Arm_3D
from robosuite.environments.base import MujocoEnv
from robosuite.environments.robot_env import RobotEnv
from robosuite.models.objects import MujocoObject
from zonopy.contset.zonotope.zono import zonotope

class ZonotopeMuJoCoEnv(Arm_3D):
    """
    Environment for the 3D arm with zonotopes as obstacles.

    Philosophy: NEVER MODIFY THE MUJOCOENV
    """
    def __init__(self, mujoco_env, kwargs):
        """
        """
        assert isinstance(mujoco_env, MujocoEnv)

        self.env = mujoco_env
        self.robot_name  = mujoco_env.robots[0].name
        self.policy_freq = mujoco_env.control_freq # (Hz)
        
        super().__init__(robot = self.robot_name, 
                         T_len = self.policy_freq,
                         **kwargs)
    
    def zonotope_from_mujoco_object(self, object):
        """
        Create zonotopic representation of bounding box of MuJoCo Object

        Args
            env: Robosuite Environment
            object: MujocoObject

        Output
            zonotope

        NOTE: Need to check if the center of the bounding box is the center of the root body.
        -- objects/MuJoCoXMLObject: bbox[2] = max(obj.bottom_offset, obj.bbox_top_offset) - obj.bottom_offset
        -- The z-axis is not aligned -- why?
        """
        assert isinstance(object, MujocoObject)

        c = self.env.sim.data.get_body_xpos(object.root_body)
        R = self.env.sim.data.get_body_xmat(object.root_body)

        G = np.diag(object.get_bounding_box_half_size())
        G = R@G

        Z = np.vstack([c, G])

        return zonotope(Z)
    
    ###############################
    ###### GETTER FUNCTIONS  ######
    ###############################
    def get_qpos_from_sim(self, robot_idx=0):
        qpos = self.env.sim.data.qpos[self.env.robots[robot_idx].controller.qpos_index]
        return qpos
    
    def get_qvel_from_sim(self, robot_idx=0):
        qvel = self.env.sim.data.qvel[self.env.robots[robot_idx].controller.qpos_index]
        return qvel
    
    ###############################
    #### KINEMATICS ###############
    ###############################
    def zono_FO(self, qpos):
        """
        Return the forward occupancy of the robot links with the zonotopes.
        It does by doing the forward kinematics of the robot.

        Args
            qpos: (n_links,) torch tensor

        Output
            link_zonos: list of zonotopes
        """
        R_qi = self.rot(qpos)
        Ri = torch.eye(3,dtype=self.dtype,device=self.device)
        Pi = torch.zeros(3,dtype=self.dtype,device=self.device)

        link_zonos = []
        for i in range(self.n_links):
            Pi = Ri@self.P0[i] + Pi
            Ri = Ri@self.R0[i]@R_qi[i]
            link_zono = Ri@self.__link_zonos[i] + Pi
            link_zonos.append(link_zono)
        
        return link_zonos
    
    ################################
    # Env Template Functions
    ################################
    def reset(self):
        # TODO 1. self.qgoal = update the position of the goal according to the mode (pick, place)
        
        # sync the robot state with the mujoco env
        self.qpos = self.get_qpos_from_sim()
        self.qvel = self.get_qvel_from_sim()

        # sync the obstacles with the mujoco env
        self.obs_zonos = [self.zonotope_from_mujoco_object(obj) for obj in self.env.objects]
        
        # reset the internal status
        self.done = False
        self.collision = False

        return self.get_observations()
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def render(self, show=True):
        pass

    def close(self):
        super().close()