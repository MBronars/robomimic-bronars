import abc
import os
from enum import Enum

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw

# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope
import safediffusion.utils.reachability_utils as reach_utils
from safediffusion.envs.env_safety import SafetyEnv

class GeomType(Enum):
    PLANE = 0
    HFIELD = 1
    SPHERE = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CYLINDER = 5
    BOX = 6
    MESH = 7
    SDF = 8

class ZonotopeEnv(SafetyEnv):
    """
    Zonotope Environment Class
    """
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

        # visualization setting
        self.render_setting = kwargs["render"]["zonotope"]
        self.fig     = None
        self.ax      = None
        self.patches = {
            "robot": None,
            "goal": None,
            "dynamic_obs": None,
            "static_obs": None,
            "FRS": None
        }

        self.plots = {
            "plan": None,
            "backup_plan": None
        }

    def get_observation(self, obs=None):
        """
        Add zonotope information to the observation dictionary

        Args:
            obs (dict): observation dictionary
        
        Returns:
            obs_dict (dict): updated observation dictionary with zonotope information
        
        Example:
            obs_dict["zonotope"]["obstacle"] = list of zonotopes of the obstacles
            obs_dict["zonotope"]["robot"] = list of zonotopes of the robot
        """
        obs_dict = super().get_observation(obs)

        # make sure to synchronize before getting observation
        self._sync()

        zonotope_dict = dict()
        zonotope_dict["obstacle"] = []
        zonotope_dict["obstacle"].extend(list(self.geom_table["zonotope"][self.geom_table["static_obs"]]))
        zonotope_dict["obstacle"].extend(list(self.geom_table["zonotope"][self.geom_table["dynamic_obs"]]))
        zonotope_dict["robot"] = []
        zonotope_dict["robot"].extend(list(self.geom_table["zonotope"][self.geom_table["robot"]]))

        obs_dict["zonotope"] = zonotope_dict

        return obs_dict

    def set_goal(self, **kwargs):
        obs = super().set_goal(**kwargs)
        self._sync()
        return obs
    
    @property
    def name(self):
        return "Zono" + super().name
    
    def create_geom_table(self):
        """
        Add the zonotope column that stores the zonotopic representation of the geometry
        """
        geom_table             = super().create_geom_table()
        geom_table["zonotope"] = [self.get_zonotope_from_geom_id(geom_id) 
                                       for geom_id in geom_table.index]
        
        return geom_table
    
    # ------------------------------------------------------------ #
    # ----------------- Zonotope-related functions --------------- #
    # ------------------------------------------------------------ #
    def _sync(self):
        """
        Update the zonotope of the robot and dynamic obstacles
        """
        robot_mask = self.geom_table["robot"]
        dynamic_obs_mask = self.geom_table["dynamic_obs"]
        update_mask = robot_mask | dynamic_obs_mask
        self.geom_table.loc[update_mask, "zonotope"] = [self.get_zonotope_from_geom_id(geom_id) 
                                                        for geom_id in self.geom_table.loc[update_mask].index]

    def get_zonotope_from_geom_name(self, geom_name):
        """
        Get the zonotope from the geom_name

        Args:
            geom_name (str): name of the geom

        Returns:
            ZP (zonotope): zonotope of the geom
        """
        geom_id = self.sim.model.geom_name2id(geom_name)
        return self.get_zonotope_from_geom_id(geom_id)

    def get_zonotope_from_geom_id(self, geom_id):
        """
        Get the zonotope from the geom_id.

        This assumes the sim.forward is called before calling this function so that geom_xpos, geom_xmat, geom_size are updated.

        Args:
            geom_id (int): id of the geom
        
        Returns:
            ZP (zonotope): zonotope of the geom

        NOTE: Currently supports only Plane, Sphere, Cylinder, Box
        TODO: Support more geom types: 1) MESH, 2) CAPSULE, 3) ELLIPSOID, 4) HFIELD, 5) SDF
        """
        geom_type = self.sim.model.geom_type[geom_id]
        geom_pos = self.sim.data.geom_xpos[geom_id]
        geom_rot = self.sim.data.geom_xmat[geom_id].reshape(3, 3)
        geom_size = self.sim.model.geom_size[geom_id]

        # Get Zonotope Primitive (ZP)
        if geom_type == GeomType.PLANE.value:
            ZP = reach_utils.get_zonotope_from_plane_geom(pos=geom_pos, rot=geom_rot, size=geom_size)
        elif geom_type == GeomType.SPHERE.value:
            ZP = reach_utils.get_zonotope_from_sphere_geom(pos=geom_pos, rot=geom_rot, size=geom_size)
        elif geom_type == GeomType.CYLINDER.value:
            ZP = reach_utils.get_zonotope_from_cylinder_geom(pos=geom_pos, rot=geom_rot, size=geom_size)
        elif geom_type == GeomType.BOX.value:
            ZP = reach_utils.get_zonotope_from_box_geom(pos=geom_pos, rot=geom_rot, size=geom_size)
        elif geom_type == GeomType.MESH.value:
            ZP = None
        else:
            print(f"Geom type {geom_type} is not supported yet.")
            raise NotImplementedError("Not Implemented")
        
        if ZP is not None:
            ZP = ZP.to(device=self.device, dtype=self.dtype)

        return ZP

    # ------------------------------------------------------------ #
    # ----------------- Renderer-related functions --------------- #
    # ------------------------------------------------------------ #
    def clear_patches(self, patches):
        """
        Clear the patches from the figure.

        This uses recursive removing of the patches in order to support both 3d and 2d plot.

        Args:
            patches: list of patches or a single patch to remove
        """
        if type(patches) == list:
            for patch in patches:
                self.clear_patches(patch)
        else:
            patches.remove()
    
    def draw_zonotopes_in_geom_table(self, key, remove_if_exists=True):
        """
        Draw the zonotope group in the geom_table

        Args:
            key (str): key of the zonotope group
            remove_if_exists (bool): remove the existing patches if True
        
        Returns:
            patch_data (list): list of vertices of the zonotopes
        
        Example:
            draw_zonotopes_in_geom_table("static_obs")
        """
        # Input validation
        assert key in self.patches and key in self.geom_table.columns, f"Key {key} is not in the patches dictionary"
        
        # Clear patches if exists
        if self.patches[key] is not None and remove_if_exists:
            self.clear_patches(self.patches[key])
        # Get the zonotopes series data
        zonotopes = self.geom_table["zonotope"][self.geom_table[key]]

        if zonotopes.empty:
            return None
        
        self.patches[key], patch_data = self.draw_zonotopes(zonotopes, 
                                                self.render_setting[key]["color"],
                                                self.render_setting[key]["alpha"],
                                                self.render_setting[key]["linewidth"])
        return patch_data
    
    def draw_zonotopes(self, zonotopes, color, alpha, linewidth):
        """
        Draw the 3D zonotope on the plot

        Args:
            zonotopes (list): list of zonotopes
            color (str): color of the zonotope
            alpha (float): transparency of the zonotope
            linewidth (float): width of the line

        Returns:
            patches (list): list of patches that are drawn
            patch_data (list): list of vertices of the zonotopes
        """
        patch_data = torch.vstack([zonotope.polyhedron_patch() for zonotope in zonotopes])

        patches = self.ax.add_collection3d(Poly3DCollection(patch_data,
                                                             edgecolor=color,
                                                             facecolor=color,
                                                             alpha=alpha,
                                                             linewidth=linewidth))
        
        return patches, patch_data

    def initialize_renderer(self):
        """
        Initialize the renderer for the zonotope visualization
        """
        self.fig = plt.figure()
        self.fig.set_size_inches(self.render_setting["width"], self.render_setting["height"])
        self.ax = plt.axes(projection='3d')

        # static zonotopes
        static_obs_zono_patches_data = self.draw_zonotopes_in_geom_table("static_obs")

        vertices = torch.vstack([static_obs_zono_patches_data.reshape(-1, 3)])
        max_V = vertices.cpu().numpy().max(axis=0)
        min_V = vertices.cpu().numpy().min(axis=0)

        self.ax.set_xlim([min_V[0] - 0.1, max_V[0] + 0.1])
        self.ax.set_ylim([min_V[1] - 0.1, max_V[1] + 0.1])
        self.ax.set_zlim([min_V[2] - 0.1, max_V[2] + 0.5]) # robot height

    def draw_box(self, img, box_width, box_height, box_color):
        # Convert the NumPy array to a PIL Image
        pil_img = Image.fromarray(img)

        # Create a drawing context
        draw = ImageDraw.Draw(pil_img)

        # Define the position for the box (bottom left corner)
        image_width, image_height = pil_img.size
        box_position = (0, image_height - box_height)

        # Draw the rectangle (box)
        draw.rectangle([box_position, (box_position[0] + box_width, box_position[1] + box_height)], fill=box_color)

        # Convert the PIL Image back to a NumPy array
        img_with_box = np.array(pil_img)
        return img_with_box



    def custom_render(self, mode=None, height=None, width=None, camera_name=None, **kwargs):
        """Renders the environment as 3D zonotope world.

        Args:
            mode (str): the mode to render with
        """
        if self.fig is None:
            self.initialize_renderer()

        # dynamic zonotopes
        self.draw_zonotopes_in_geom_table("dynamic_obs", remove_if_exists=True)
        self.draw_zonotopes_in_geom_table("robot", remove_if_exists=True)

        # FRS
        if "FRS" in kwargs.keys() and kwargs["FRS"] is not None:
            if self.patches["FRS"] is not None:
                self.clear_patches(self.patches["FRS"])
            self.patches["FRS"], _ = self.draw_zonotopes(kwargs["FRS"], "green", 0.5, 0.1)

        if "plan" in kwargs.keys():
            if self.plots["plan"] is not None:
                self.plots["plan"][0].remove()
            self.plots["plan"] = self.ax.plot(kwargs["plan"][:, 0], kwargs["plan"][:, 1], color='r', linewidth=2)            

        if "backup_plan" in kwargs.keys():
            if self.plots["backup_plan"] is not None:
                self.plots["backup_plan"][0].remove()
            self.plots["backup_plan"] = self.ax.plot(kwargs["backup_plan"][:, 0], kwargs["backup_plan"][:, 1], color='b', linewidth=2)

        # goal zonotopes: TODO
        # self.goal_patches.remove()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if "intervened" in kwargs.keys() and kwargs["intervened"] and self.is_safe():
            img = self.img_intervene_filter(img)
            # img = self.draw_box(img, box_width=100, box_height=100, box_color=(0, 255, 0))

        return img