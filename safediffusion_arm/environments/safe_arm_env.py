import os
import numpy as np

from robosuite.controllers.controller_factory import SwitchingController

from safediffusion.envs.env_safety import Geom
from safediffusion.envs.env_zonotope import ZonotopeEnv
import safediffusion.utils.transform_utils as TransformUtils

# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope

class SafeArmEnv(ZonotopeEnv):
    def __init__(self, env, **kwargs):
        # TODO: write assertion statement here

        super().__init__(env, **kwargs)

        self.done = False
        self.success = False

    # -------------------------------------------------------- #
    # Helper Functions for SafeArmEnv
    # -------------------------------------------------------- #
    def switch_controller(self, controller_name):
        """
        Switch the action representation & low-level controller of the environment

        Args:
            controller_name
        """
        # cache the pointer
        controller = self.unwrapped_env.robots[0].controller

        assert(controller, SwitchingController, "The controller should be an instance of SwitchingController.")
        assert(controller_name in controller.controller_dict.keys(), "The controller name should be registered in the SwitchingController's controller name")

        # update the environment action 
        if controller.active_controller is not controller_name:
            # update the controller
            controller.switch_to(controller_name)
            
            # update internal status of env about action representation
            arm_ac_dim     = controller.control_dim
            gripper_ac_dim = 1
            self.unwrapped_env._action_dim = arm_ac_dim + gripper_ac_dim

    def is_geom_id_visual(self, geom_id):
        """
        Check if the geom_id is visual
        """
        geom_name = self.env.sim.model.geom_id2name(geom_id)
        return self.is_geom_name_visual(geom_name)
    
    def is_geom_name_visual(self, geom_name):
        """
        Check if the geom_id is visual
        """
        if geom_name is None:
            return False
        
        elif geom_name.endswith("visual") or \
             geom_name.endswith("vis") or \
             geom_name.startswith("visual"):
            return True
        
        return False

    def is_geom_visual(self, geom):
        """
        Check if the query geom is visual

        Geometry is visual if its name ends/starts with `visual`, `vis`
        if name is None, we consider it not visual for this environment.

        Args:
            geom: Geom object

        Returns:
            Return true if the geom is visual

        TODO: Can we check other convention to check visuality of the geom?
        """
        assert isinstance(geom, Geom)

        return self.is_geom_name_visual(geom.name)


    # -------------------------------------------------------- #
    # Abstract functions to override (SafetyEnv)
    # -------------------------------------------------------- #

    def is_safe(self):
        """ 
        For Arm, the safety is defined as robot geometry not colliding with non-grasping objects

        TODO: Implement this checking
        """
        # Logic: if grasping, ignore grasping object and use collision
        # TODO: how do we check the grasping status?
        
        return not self.collision()
    
    def register_robot_geoms(self):
        """ Register the collision geometry that belongs to robot
        """
        # get geometry names that belong to robot, gripper
        geoms = self.get_geom_that_body_name_starts_with(["robot", "gripper"])

        # filter out the `visual` geometry
        geoms = [geom for geom in geoms if not self.is_geom_visual(geom)]

        return geoms

    def register_static_obstacle_geoms(self):
        """ 
        Register the collision geometry that is considered dangerous and static
        """
        # get geometry that belong to mount & bin
        geoms = self.get_geom_that_body_name_starts_with(["mount", "bin"])
        # filter out the `visual` geometry
        geoms = [geom for geom in geoms if not self.is_geom_visual(geom)]

        return geoms

    def register_dynamic_obstacle_geoms(self):
        """ 
        Register the geometry that is considered dangerous and dynamic
        """
        geoms = []
        
        object_names = self.env.env.obj_names

        for object_name in object_names:
            geoms.extend(self.get_geom_that_body_name_starts_with(object_name))
        
        geoms = [geom for geom in geoms if not self.is_geom_visual(geom)]

        return geoms

    # -------------------------------------------------------- #
    # Abstract functions to override (EnvBase)
    # -------------------------------------------------------- #
    def set_goal(self, qpos = None):
        """
        Override the `set_goal` method from `EnvBase`.

        This method sets the goal observation based on an external specification.
        Specifically, it defines the target joint configuration (qpos) for the robot arm.

        Args:
            qpos (np.ndarray, optional): Desired goal joint configuration.
                If provided, this will set the `_target_qpos` attribute.
                Defaults to None.

        Returns:
            dict: The current goal as a dictionary, retrieved from `get_goal()`.
        """
        if qpos is not None:
            self._target_qpos = qpos

        return self.get_goal()
    
    def get_goal(self):
        """
        Override the `get_goal` method from `EnvBase`.

        Returns:
            dict: A dictionary representing the goal configuration.
            If `_target_qpos` is defined, it includes the goal joint configuration.
        """
        goal_dict = dict()

        if hasattr(self, '_target_qpos'):
            goal_dict["qpos"] = self._target_qpos
        
        return goal_dict

    def get_observation(self, obs=None):
        """ 
        Postprocess observation from SafeZonotopeEnv
        """
        obs = super().get_observation(obs)

        # add the transform between the world to the robot base
        mjdata = self.unwrapped_env.sim.data

        T_world_to_arm_base = TransformUtils.make_pose(
                                 rot = mjdata.get_body_xmat("robot0_base"),
                                 pos = mjdata.get_body_xpos("robot0_base")
                                 )

        obs["T_world_to_arm_base"] = T_world_to_arm_base 

        return obs
    
    def reset(self):
        """ 
        Reset the environment

        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        obs = super().reset()

        self.done    = False
        self.success = False

        return obs

    def reset_to(self, state_dict):
        """
        Reset the environment to the given state

        Args:
            state_dict (dict): dictionary containing the states to reset

        Returns:
            obs (np.ndarray): observation
        """
        obs = super().reset_to(state_dict)

        self.done = False
        self.success = False
        
        return obs

    def step(self, action):
        """
        Step the environment with the given action

        Args:
            action (np.ndarray): action to take

        Returns:
            obs (np.ndarray): observation
            reward (float): reward
            done (bool): done flag
            info (dict): info dictionary
        """
        obs, reward, done, info = super().step(action)

        # TODO: maybe update the `done` information according to the criteria
        return obs, reward, done, info