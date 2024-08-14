import os
import numpy as np

from safediffusion.envs.env_zonotope import ZonotopeEnv
import torch
import matplotlib.pyplot as plt

# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope

class SafeMazeEnv(ZonotopeEnv):
    """
    Safety Zonotope Wrapper for Maze2D environments.

    The safety specifications are defined as not colliding with the walls.

    NOTE: get_observations() returns the robot qpos, while all the renderer functions are based on the world position.
    """
    def __init__(self, env, **kwargs):
        assert env.name.startswith('maze2d')

        super().__init__(env, **kwargs)
        
        # Set the third-party renderer
        self.janner_renderer = None

        # Transformation between the worlds
        self.d4rlmazepos_to_worldpos = np.array([1, 1])
        self.robotqpos_to_worldpos = self.get_robot_to_world_transformation()

        self.success_radius = 0.2

        self.success        = False
        self.done           = False
        

    # --------------------- Helper Function ------------------------------ #
    def set_state(self, pos, vel):
        """ 
        Set the state of the MuJoCo simulation using position and velocity in world frame
        
        Args:
            pos (np.ndarray): position of the agent in the world frame
            vel (np.ndarray): velocity of the agent in the world frame
        
        NOTE: this pos and vel is different from the qpos and qvel
        """
        t_dummy = np.array([0])
        qpos = pos - self.robotqpos_to_worldpos
        qvel = vel

        state_dict = {"states": np.concatenate([t_dummy, qpos, qvel])}

        return self.reset_to(state_dict)
    
    def get_robot_to_world_transformation(self):
        """
        Get the transformation matrix that transforms the robot qpos to the world position (x, y)

        In the Maze 2D environment, it is simple transformation
        """
        robot_zonotope = self.geom_table["zonotope"][self.geom_table["robot"]].values[0]
        world_robotpos = np.array(robot_zonotope.center[:2])
        body_robotpos  = self.unwrapped_env.sim.data.qpos

        offset = world_robotpos - body_robotpos

        return offset
    
    def is_success(self):
        """
        Maze 2D environment is successful when the agent reaches the target

        Pointmaze environment does not return success info, so overriden.
        """
        dict = {"task": self.success}

        return dict

    # ------------------- SafetyEnv related functions -------------------#
    def is_safe(self):
        """ For Maze 2D, the safety is defined as not colliding with the walls
        """
        return not self.collision()

    def register_robot_geoms(self):
        """ Register the robot geometry that has body name of particle
        """
        robot_body_names = ["particle"]
        
        return self.get_geom_that_body_name_is_in(robot_body_names)
    
    def register_static_obstacle_geoms(self):
        """ Register the geometry that is considered dangerous
        """
        static_obs_prefix = ["wall"]

        return self.get_geom_that_name_starts_with(static_obs_prefix)
    
    def register_dynamic_obstacle_geoms(self):
        """ Register the geometry that is considered dangerous
        """
        return []
    
    def get_observation(self, obs=None):
        obs = super().get_observation(obs)
        return obs
    
    def reset(self):
        """
        Reset the environment
        """
        obs = super().reset()
        self.success = False
        self.done = False

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
        self.success = False
        self.done = False

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
        
        # Update the success flag
        goal_qpos = self.get_goal()["flat"][:2]

        # Environment is frame stacking wrapper. Retrieve the most recent one.
        if obs["flat"][:2].ndim == 2:
            qpos = obs["flat"][-1, :2]
        else:
            qpos = obs["flat"][:2]

        if np.linalg.norm(goal_qpos - qpos) <= self.success_radius:
            self.success = True
            done = True
    
        return obs, reward, done, info
    
    def get_goal(self):
        """
        Get the goal qpos, qvel configuration for the robot
        """
        # Get goal qpos
        goal_pos     = self.unwrapped_env.get_target()          # goal d4rl pos
        goal_pos     = goal_pos + self.d4rlmazepos_to_worldpos  # goal safemazeenv pos
        goal_qpos    = goal_pos - self.robotqpos_to_worldpos    # goal robot qpos
        
        # Get goal qvel
        goal_qvel     = np.zeros(2,)
        goal_state   = np.concatenate((goal_qpos, goal_qvel))

        goal_dict = {}
        goal_dict["flat"] = goal_state
        
        return goal_dict
    
    def set_goal(self, pos):
        """ 
        Set the goal position of the agent

        Args:
            pos (np.ndarray): goal position in the world frame
        """
        
        pos = pos - self.d4rlmazepos_to_worldpos # goal d4rl pos
        self.unwrapped_env.set_target(target_location=pos)

        pos_world = pos + self.d4rlmazepos_to_worldpos
        c = torch.hstack([torch.tensor(pos_world), torch.tensor(0)])
        G = torch.eye(3) * 0.1
        Z = torch.vstack([c, G])
        goal_zonotope = zonotope(Z)

        self.goal_zonotope = goal_zonotope

        return self.get_goal()


    # ------------------- Render Helper ------------------------------ #        
    def draw_zonotopes(self, zonotopes, color, alpha, linewidth):
        """
        Draw the 2D zonotope on the plot

        Args:
            zonotopes (list): list of zonotopes
            color (str): color of the zonotope
            alpha (float): transparency of the zonotope
            linewidth (float): width of the line

        Returns:
            zonotope_patches (list): list of patches that are drawn
            zonotope_patch_data (list): list of vertices of the zonotopes
        """        
        zonotope_patch_data = torch.vstack([zonotope.polyhedron_patch() for zonotope in zonotopes])

        zonotope_patches = []
        for zonotope_patch in zonotope_patch_data:
            zonotope_patches.append(self.ax.fill(zonotope_patch[:, 0], zonotope_patch[:, 1],
                                             edgecolor=color, facecolor=color, alpha=alpha, linewidth=linewidth))
        
        return (zonotope_patches, zonotope_patch_data)

    def initialize_renderer(self):
        """
        Initialize the renderer for the zonotope visualization

        This function makes the plot to be 2D
        """
        self.fig = plt.figure()
        self.fig.set_size_inches(self.render_setting["width"], self.render_setting["height"])
        self.ax = self.fig.add_subplot(111)
        
        # static zonotopes
        # This assumes that static_obs has the outer walls of the environment
        patches_data = self.draw_zonotopes_in_geom_table("static_obs")
        vertices = torch.vstack([patches_data.reshape(-1, 3)])
        max_V = vertices.cpu().numpy().max(axis=0)
        min_V = vertices.cpu().numpy().min(axis=0)

        self.ax.set_xlim([min_V[0] - 0.1, max_V[0] + 0.1])
        self.ax.set_ylim([min_V[1] - 0.1, max_V[1] + 0.1])

    def custom_render(self, mode=None, height=None, width=None, camera_name="agentview", **kwargs):
        """Renders the environment.

        Args:
            mode (str): the mode to render with
        """
        if mode == "zonotope":
            img = super().custom_render(mode, height, width, camera_name, **kwargs)
            
        elif mode == "janner":
            if self.janner_renderer is None:
                from safediffusion.utils.render_utils import Maze2dRenderer
                self.janner_renderer = Maze2dRenderer(self.env.name)
            
            img = self.janner_renderer.render(self.get_observation()["flat"], **kwargs)
        
        return img
    
    def adjust_camera(self, camera_name):
        """
        For Maze2D env, ignore this
        """
        pass

