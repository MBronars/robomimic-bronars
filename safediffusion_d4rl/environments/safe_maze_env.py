import numpy as np

from safediffusion.envs.env_zonotope import ZonotopeEnv
import torch
import matplotlib.pyplot as plt

class SafeMazeEnv(ZonotopeEnv):
    """
    Safety Wrapper for Maze2D environments.

    The safety specifications are defined as not colliding with the walls.
    """
    def __init__(self, env, **kwargs):
        assert env.name.startswith('maze2d')
        super().__init__(env, **kwargs)
        self.janner_renderer = None

    # ------------------- Helper Function ------------------------------ #
    def set_state(self, qpos, qvel):
        """ Set the state of the environment
        """
        t_dummy = np.array([0])
        state_dict = {"states": np.concatenate([t_dummy, qpos, qvel])}
        return self.reset_to(state_dict)

    # ------------------- SafetyEnv related functions -------------------#
    def is_safe(self):
        """ For Maze 2D, the safety is defined as not colliding with the walls
        """
        return not self.collision()

    def register_robot_geoms(self):
        """ Register the robot geometry that has body name of particle
        """
        robot_body_names = ["particle"]
        
        return self.get_geom_name_of_body_name(robot_body_names)
    
    def register_static_obstacle_geoms(self):
        """ Register the geometry that is considered dangerous
        """
        static_obs_prefix = ["wall"]

        return self.get_geom_name_starts_with(static_obs_prefix)
    
    def register_dynamic_obstacle_geoms(self):
        """ Register the geometry that is considered dangerous
        """
        return []
    
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

    def custom_render(self, mode=None, height=None, width=None, camera_name=None, **kwargs):
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

