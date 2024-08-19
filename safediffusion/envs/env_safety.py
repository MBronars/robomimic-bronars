"""
This file contains the robomimic environment wrapper that is used
to provide a standardized environment API for safety-filter.
"""
import abc

import torch
import numpy as np
import pandas as pd
import gym
import robomimic.envs.env_base as EB
from robomimic.envs.wrappers import EnvWrapper

from safediffusion.utils.env_utils import get_innermost_env

class Geom():
    def __init__(self, name, id):
        self.name = name
        self.id   = id
    
    def __repr__(self):
        return self.name
        
class SafetyEnv(EB.EnvBase, abc.ABC):
    """
    Wrapper of the environment that employs set-representations for world model

    Supports wrapping Envwrapper and EnvBase
    
    TODO: make it as an abstract class
    """
    def __init__(self, env: EB.EnvBase,
                 dtype = torch.float,
                 device = torch.device("cpu"),
                 **kwargs):
        
        self.env = env
        self.unwrapped_env = get_innermost_env(self.env)

        # safety-related geoms
        self.robot_geoms                 = self.register_robot_geoms()
        self.static_obstacle_geoms       = self.register_static_obstacle_geoms()
        self.dynamic_obstacle_geoms      = self.register_dynamic_obstacle_geoms()
        
        # additional settings
        self.dtype = dtype
        self.device = device

        # hashing table
        self.geom_table                  = self.create_geom_table()

        # NOTE: gym.Env is not influenced by the global seeding, so we need to seed the environment manually
        self.seed_env()

    # ---------------------------------------------------------------------------- #
    # ------------------------------- Seed-related ----------------------------- #

    def seed_env(self):
        """
        Seed the environment

        Args:
            seed (int): seed to set
        """
        assert hasattr(self, "unwrapped_env"), "The environment should have unwrapped_env attribute"
        seed = np.random.randint(0, 2**32 - 1)
        if isinstance(self.unwrapped_env, gym.Env):
            self.unwrapped_env._np_random, seed = gym.utils.seeding.np_random(seed)
        
    # ---------------------------------------------------------------------------- #
    # ------------------------------- Safety-related ----------------------------- #
    # ---------------------------------------------------------------------------- #
    @abc.abstractmethod
    def register_robot_geoms(self):
        """
        Register the robot's geometries for safety-checking.

        Args:
            geom_names (list): list of geometry names that are related to the agent
        """
        raise NotImplementedError

    @abc.abstractmethod
    def register_static_obstacle_geoms(self):
        """
        Register the danger's geometries for safety-checking.

        Args:
            geom_names (list): list of geometry names that are related to the agent
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def register_dynamic_obstacle_geoms(self):
        """
        Register the danger's geometries for safety-checking.

        Args:
            geom_names (list): list of geometry names that are related to the agent
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_safe(self) -> bool:
        """
        Retrieve safety information from the MuJoCo simulation environment.

        Every environment has its own safety specification. Override this method to reflect the safety specification.
        
        Returns:
            is_safe (bool): The safety specification is specified
        """
        raise NotImplementedError
    
    def collision(self):
        obstacle_geom_id = {geom.id for geom in self.static_obstacle_geoms}.union(
                            {geom.id for geom in self.dynamic_obstacle_geoms}
                            )
        
        robot_geom_id = {geom.id for geom in self.robot_geoms}
        
        for i in range(self.unwrapped_env.sim.data.ncon):
            contact = self.unwrapped_env.sim.data.contact[i]

            if (contact.geom1 in robot_geom_id and contact.geom2 in obstacle_geom_id) or \
               (contact.geom2 in robot_geom_id and contact.geom1 in obstacle_geom_id):
                return True
            
        return False
    
    def img_danger_filter(self, img):
        """
        Danger Filter for the image, makes the image red-ish.

        Args:
            img (np.ndarray): image to filter
        
        Returns:
            filtered_img (np.ndarray): filtered image
        """
        red_filter = np.zeros_like(img)
        red_filter[:, :, 0] = img[:, :, 0]

        alpha = 0.5
        filtered_img = (1-alpha)*img + alpha * red_filter
        filtered_img = filtered_img.astype(np.uint8)

        return filtered_img
    
    def img_intervene_filter(self, img):
        """
        Danger Filter for the image, makes the image red-ish.

        Args:
            img (np.ndarray): image to filter
        
        Returns:
            filtered_img (np.ndarray): filtered image
        """
        blue_filter = np.zeros_like(img)
        blue_filter[:, :, 2] = img[:, :, 2]

        alpha = 0.5
        filtered_img = (1-alpha)*img + alpha * blue_filter
        filtered_img = filtered_img.astype(np.uint8)

        return filtered_img

    
    # ---------------------------------------------------------------------------------- #
    # ------------------------------ Implement EnvBase Methods ------------------------- #
    # ---------------------------------------------------------------------------------- #
    
    # ---------------------------------------------------------------------------------- #
    # ----------- This changes the internal state of the simulation -------------------- #

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(obs)

        return obs, reward, done, info

    def reset(self):
        """
        Reset and sync with the environment
        """
        obs = self.env.reset()
        return self.get_observation(obs)
    
    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains:
                - states (np.ndarray): initial state of the mujoco environment
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        assert isinstance(state, dict) and "states" in state, "The state should contain the states key"
        assert state["states"].shape == self.get_state()["states"].shape, "The state shape is not compatible"

        obs = self.env.reset_to(state)
        return self.get_observation(obs)

    def set_goal(self, **kwargs):
        return self.env.set_goal(**kwargs)
    
    # ---------------------------------------------------------------------------------- #
    # ----------- This does not change the internal state of the simulation ------------ #

    def get_observation(self, obs=None):
        if obs is None:
            obs = self.env.get_observation()
        obs["safe"] = self.is_safe()

        return obs

    def get_state(self):
        return self.env.get_state()

    def get_goal(self):
        return self.env.get_goal()

    def get_reward(self):
        return self.env.get_reward()

    def is_done(self):
        return self.env.is_done()

    def is_success(self):
        return self.env.is_success()
    
    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """Renders the environment.
        This adds the safety information to the rendering.

        Args:
            mode (str): the mode to render with
        """
        if mode in ["human", "rgb_array"]:
            # use the default rendering method
            img = self.env.render(mode, height, width, camera_name, **kwargs)
        else:
            img = self.custom_render(mode, height, width, camera_name, **kwargs)

        if not self.is_safe():
            img = self.img_danger_filter(img)

        return img
    
    def custom_render(self, mode, height, width, camera_name, **kwargs):
        raise NotImplementedError
    
    def serialize(self):
        pass

    @classmethod
    def create_for_data_processing(
        cls,
        camera_names,
        camera_height,
        camera_width,
        reward_shaping,
        render=None,
        render_offscreen=None,
        use_image_obs=None,
        use_depth_obs=None,
        **kwargs,
    ):
        return cls.env.create_for_data_processing(camera_names, camera_height, camera_width, reward_shaping,
                                                  render, render_offscreen, use_image_obs, use_depth_obs,
                                                  **kwargs)

    @property
    def action_dimension(self):
        return self.env.action_dimension
    
    @property
    def name(self):
        return "Safe" + self.env.name

    @property
    def rollout_exceptions(self):
        return self.env.rollout_exceptions

    @property
    def base_env(self):
        return self.env.base_env

    @property
    def type(self):
        return self.env.type


    # ---------------------------------------------------------------------------- #
    # ----------------------------- Helper Functions ----------------------------- #
    # ---------------------------------------------------------------------------- #
    def create_geom_table(self):
        """
        Get the table that stores meta-data of the geometries.

        This assumes the MuJoCo environment.
        """
        n_geoms = self.unwrapped_env.sim.model.ngeom
        geom_info_dict = dict()
        
        robot_geom_ids = [geom.id for geom in self.robot_geoms]
        static_obs_geom_ids = [geom.id for geom in self.static_obstacle_geoms]
        dynamic_obs_geom_ids = [geom.id for geom in self.dynamic_obstacle_geoms]
        
        for geom_id in range(n_geoms):
            geom_info_dict[geom_id] = dict()
            geom_name = self.unwrapped_env.sim.model.geom_id2name(geom_id)
            geom_info_dict[geom_id]["name"]        = geom_name
            geom_info_dict[geom_id]["type"]        = self.unwrapped_env.sim.model.geom_type[geom_id]
            geom_info_dict[geom_id]["body_id"]     = self.unwrapped_env.sim.model.geom_bodyid[geom_id]
            geom_info_dict[geom_id]["robot"]       = geom_id in robot_geom_ids
            geom_info_dict[geom_id]["static_obs"]  = geom_id in static_obs_geom_ids
            geom_info_dict[geom_id]["dynamic_obs"] = geom_id in dynamic_obs_geom_ids
        
        df = pd.DataFrame.from_dict(geom_info_dict, orient='index')

        return df
    
    def get_geom_that_body_name_is_in(self, body_names):
        """
        Get the list of geometry of which body has name in the given list of body names.

        Args:
            body_names (list): list of body names
        
        Returns:
            geoms (list): list of geometry names that are related to the body names
        """
        # Argument handling
        if isinstance(body_names, str):
            body_names = [body_names]

        mjmodel = self.unwrapped_env.sim.model
        
        geoms = [Geom(id = gid, name = mjmodel.geom_id2name(gid))
                for (gid, bid) in enumerate(mjmodel.geom_bodyid)
                for body_name in body_names 
                if bid == mjmodel.body_name2id(body_name)]
        
        return geoms

    def get_geom_that_body_name_starts_with(self, prefix_list):
        """
        Get the geometry that body name start with the given prefix.
        
        Args:
            prefix_list (list): list of prefixes
        Returns:
            geoms (list): list of geometries whose body name start with the given prefix
        """
        if isinstance(prefix_list, str):
            prefix_list = [prefix_list]

        mjmodel = self.unwrapped_env.sim.model
        body_names = [body_name
                      for body_name in mjmodel.body_names
                      for prefix in prefix_list
                      if body_name.startswith(prefix)]
        
        return self.get_geom_that_body_name_is_in(body_names)
    
    def get_geom_that_body_name_ends_with(self, postfix_list):
        """
        Get the geometry that body name ends with the given postfix.
        
        Args:
            postfix_list (list): list of postfixes

        Returns:
            geoms (list): list of geometry names that start with the given postfix
        """
        if isinstance(postfix_list, str):
            postfix_list = [postfix_list]

        mjmodel = self.unwrapped_env.sim.model

        body_names = [body_name
                      for body_name in mjmodel.body_names
                      for postfix in postfix_list
                      if body_name.endswith(postfix)]
        
        return self.get_geom_that_body_name_is_in(body_names)
    
    def get_geom_that_name_starts_with(self, prefix_list):
        """
        Get the list of geometry of which name that start with the given prefix.
        
        Args:
            prefix_list (list): list of prefixes
        Returns:
            geom (list): list of geometry (Geom) of which name starts with the given prefix
        """

        if isinstance(prefix_list, str):
            prefix_list = [prefix_list]
        
        # pointer to the MuJoCo model
        mjmodel = self.unwrapped_env.sim.model

        geoms = [
            Geom(id=id, name=name)
            for id in range(mjmodel.ngeom)
            if (name := mjmodel.geom_id2name(id)) is not None and name.startswith(tuple(prefix_list))
        ]
        
        return geoms
    
    def get_geom_name_ends_with(self, postfix_list):
        """
        Get the list of geometry of which name ends with the given postfix.
        
        Args:
            postfix (list): list of postfixes
        Returns:
            geom (list): list of geometry (Geom) of which name ends with the given postfix
        """

        if isinstance(postfix_list, str):
            postfix_list = [postfix_list]
        
        # pointer to the MuJoCo model
        mjmodel = self.unwrapped_env.sim.model

        geoms = [
            Geom(id=id, name=name)
            for id in range(mjmodel.ngeom)
            if (name := mjmodel.geom_id2name(id)) is not None and name.endswith(tuple(postfix_list))
        ]
        
        return geoms