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

        # get mujoco sim object, supports Envwrapper and Env
        if isinstance(env, EnvWrapper):
            self.sim = env.env.env.sim
        elif isinstance(env, EB.EnvBase):
            self.sim = env.env.sim

        # safety-related geoms
        self.robot_geom_names            = self.register_robot_geoms()
        self.static_obstacle_geom_names  = self.register_static_obstacle_geoms()
        self.dynamic_obstacle_geom_names = self.register_dynamic_obstacle_geoms()
        
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
        seed = np.random.randint(0, 2**32 - 1)
        env_innermost = get_innermost_env(self.env)
        if isinstance(env_innermost, gym.Env):
            env_innermost._np_random, seed = gym.utils.seeding.np_random(seed)
        
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
        obstacles = []
        obstacles.extend(self.static_obstacle_geom_names)
        obstacles.extend(self.dynamic_obstacle_geom_names)

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = self.sim.model.geom_id2name(contact.geom1)
            geom2 = self.sim.model.geom_id2name(contact.geom2)

            if (geom1 in self.robot_geom_names and geom2 in obstacles) or \
               (geom2 in self.robot_geom_names and geom1 in obstacles):
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
        n_geoms = self.sim.model.ngeom
        geom_info_dict = dict()
        
        for geom_id in range(n_geoms):
            geom_info_dict[geom_id] = dict()
            geom_name = self.sim.model.geom_id2name(geom_id)
            geom_info_dict[geom_id]["name"]        = geom_name
            geom_info_dict[geom_id]["type"]        = self.sim.model.geom_type[geom_id]
            geom_info_dict[geom_id]["body_id"]     = self.sim.model.geom_bodyid[geom_id]
            geom_info_dict[geom_id]["robot"]       = geom_name in self.robot_geom_names
            geom_info_dict[geom_id]["static_obs"]  = geom_name in self.static_obstacle_geom_names
            geom_info_dict[geom_id]["dynamic_obs"] = geom_name in self.dynamic_obstacle_geom_names
        
        df = pd.DataFrame.from_dict(geom_info_dict, orient='index')

        return df
    
    def get_geom_name_of_body_name(self, body_names):
        """
        Get the geometry names of the list of given body names.

        Args:
            body_names (list): list of body names
        
        Returns:
            geom_names_of_body (list): list of geometry names that are related to the body names
        """
        # Argument handling
        if isinstance(body_names, str):
            body_names = [body_names]
        
        geom_names = []

        for body_name in body_names:
            body_id = self.sim.model.body_name2id(body_name)

            geom_names_iter = [self.sim.model.geom_id2name(gid) 
                                for (gid, bid) in enumerate(self.sim.model.geom_bodyid) 
                                if bid == body_id]
            
            geom_names.extend(geom_names_iter)

        return geom_names
    
    def get_geom_name_starts_with(self, prefix_list):
        """
        Get the geometry names that start with the given prefix.
        
        Args:
            prefix_list (list): list of prefixes
        Returns:
            geom_names (list): list of geometry names that start with the given prefix
        """

        if isinstance(prefix_list, str):
            prefix_list = [prefix_list]
  
        geom_names = []

        for prefix in prefix_list:
            geom_names_iter = [name 
                               for name in self.sim.model.geom_names 
                               if name.startswith(prefix)]
            geom_names.extend(geom_names_iter)
        
        return geom_names
    
    def get_geom_name_ends_with(self, postfix_list):
        """
        Get the geometry names that ends with the given postfix.
        
        Args:
            postfix (list): list of postfixes
        Returns:
            geom_names (list): list of geometry names that ends with the given postfix
        """

        if isinstance(postfix_list, str):
            postfix_list = [postfix_list]
        
        geom_names = []

        for postfix in postfix_list:
            geom_names_iter = [name 
                               for name in self.sim.model.geom_names 
                               if name.endswith(postfix)]
            geom_names.extend(geom_names_iter)
        
        return geom_names