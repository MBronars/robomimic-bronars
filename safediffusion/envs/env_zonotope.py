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
import safediffusion.utils.list_utils as ListUtils
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

        # zonotope objects to draw
        self.patches = {
            "robot"       : None,
            "dynamic_obs" : None,
            "static_obs"  : None,
            "FRS"         : None
        }

        self.plots = {
            "plan": None,
            "backup_plan": None,
            "plans": None
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
        geom_table                       = super().create_geom_table()
        geom_table["zonotope_primitive"] = [self.get_zonotope_primitive_from_geom_id(geom_id) 
                                                for geom_id in geom_table.index]
        geom_table["zonotope"]           = [self.get_zonotope_from_geom_id(geom_id) 
                                                for geom_id in geom_table.index]
        
        return geom_table
    
    # ------------------------------------------------------------ #
    # ----------------- Zonotope-related functions --------------- #
    # ------------------------------------------------------------ #
    def _sync(self):
        """
        Update the zonotope of the robot and dynamic obstacles
        """
        mjdata = self.unwrapped_env.sim.data
        robot_mask = self.geom_table["robot"]
        dynamic_obs_mask = self.geom_table["dynamic_obs"]
        update_mask = robot_mask | dynamic_obs_mask
        self.geom_table.loc[update_mask, "zonotope"] = [reach_utils.transform_zonotope(
                                                            zono = self.geom_table["zonotope_primitive"][geom_id],
                                                            pos  = mjdata.geom_xpos[geom_id],
                                                            rot  = mjdata.geom_xmat[geom_id].reshape(3, 3)
                                                        ) 
                                                        for geom_id in self.geom_table.loc[update_mask].index]

    def get_zonotope_from_geom_name(self, geom_name):
        """
        Get the zonotope from the geom_name

        Args:
            geom_name (str): name of the geom

        Returns:
            ZP (zonotope): zonotope of the geom
        """
        geom_id = self.unwrapped_env.sim.model.geom_name2id(geom_name)
        return self.get_zonotope_from_geom_id(geom_id)

    def get_zonotope_primitive_from_geom_id(self, geom_id):
        """
        Get the zonotope primitive of the geometry given `geom_id`

        This zonotope does not depend on the current pos/rot of the geometry

        Args
            geom_id (int): id of the geometry
        
        Returns
            zonotope_primitive (zonotope)
        """

        mjmodel = self.unwrapped_env.sim.model

        geom_type = mjmodel.geom_type[geom_id]
        geom_size = mjmodel.geom_size[geom_id]

        zero = torch.zeros(3,)
        eye  = torch.eye(3)

        # Get Zonotope Primitive (ZP)
        if geom_type == GeomType.PLANE.value:
            ZP = reach_utils.get_zonotope_from_plane_geom(pos=zero, rot=eye, size=geom_size)
        elif geom_type == GeomType.SPHERE.value:
            ZP = reach_utils.get_zonotope_from_sphere_geom(pos=zero, rot=eye, size=geom_size)
        elif geom_type == GeomType.CYLINDER.value:
            ZP = reach_utils.get_zonotope_from_cylinder_geom(pos=zero, rot=eye, size=geom_size)
        elif geom_type == GeomType.BOX.value:
            ZP = reach_utils.get_zonotope_from_box_geom(pos=zero, rot=eye, size=geom_size)
        elif geom_type == GeomType.MESH.value:
            mesh_id = mjmodel.geom_dataid[geom_id]
            vert_start = mjmodel.mesh_vertadr[mesh_id]
            vert_count = mjmodel.mesh_vertnum[mesh_id]
            vertices = mjmodel.mesh_vert[vert_start:vert_start + vert_count].reshape(-1, 3)
            ZP = reach_utils.get_zonotope_from_mesh_vertices(vertices=vertices)
        else:
            print(f"Geom type {geom_type} is not supported yet.")
            raise NotImplementedError("Not Implemented")

        if ZP is not None:
            ZP = ZP.to(device=self.device, dtype=self.dtype)
        
        return ZP
    
    def get_zonotope_from_geom_id(self, geom_id):
        """
        Get the zonotope given the simulation data
        """

        mjdata = self.unwrapped_env.sim.data
        
        ZP = self.get_zonotope_primitive_from_geom_id(geom_id)

        geom_pos = mjdata.geom_xpos[geom_id]
        geom_rot = mjdata.geom_xmat[geom_id].reshape(3, 3)

        Z = reach_utils.transform_zonotope(ZP, pos = geom_pos, rot = geom_rot)

        return Z

    # TODO: start from here. replace get_zonotope_from_geom_id, refactor reach_utils


    
    # def get_zonotope_from_geom_id(self, geom_id):
    #     """
    #     Get the zonotope from the geom_id.

    #     This assumes the sim.forward is called before calling this function so that geom_xpos, geom_xmat, geom_size are updated.

    #     Args:
    #         geom_id (int): id of the geom
        
    #     Returns:
    #         ZP (zonotope): zonotope of the geom

    #     NOTE: Currently supports only Plane, Sphere, Cylinder, Box
    #     TODO: Support more geom types: 1) MESH, 2) CAPSULE, 3) ELLIPSOID, 4) HFIELD, 5) SDF
    #     """
    #     mjmodel = self.unwrapped_env.sim.model
    #     mjdata  = self.unwrapped_env.sim.data
        
    #     geom_type = mjmodel.geom_type[geom_id]
    #     geom_size = mjmodel.geom_size[geom_id]

    #     geom_pos = mjdata.geom_xpos[geom_id]
    #     geom_rot = mjdata.geom_xmat[geom_id].reshape(3, 3)
        
    #     # Get Zonotope Primitive (ZP)
    #     if geom_type == GeomType.PLANE.value:
    #         ZP = reach_utils.get_zonotope_from_plane_geom(pos=geom_pos, rot=geom_rot, size=geom_size)
    #     elif geom_type == GeomType.SPHERE.value:
    #         ZP = reach_utils.get_zonotope_from_sphere_geom(pos=geom_pos, rot=geom_rot, size=geom_size)
    #     elif geom_type == GeomType.CYLINDER.value:
    #         ZP = reach_utils.get_zonotope_from_cylinder_geom(pos=geom_pos, rot=geom_rot, size=geom_size)
    #     elif geom_type == GeomType.BOX.value:
    #         ZP = reach_utils.get_zonotope_from_box_geom(pos=geom_pos, rot=geom_rot, size=geom_size)
    #     elif geom_type == GeomType.MESH.value:
    #         mesh_id = mjmodel.geom_dataid[geom_id]
    #         vert_start = mjmodel.mesh_vertadr[mesh_id]
    #         vert_count = mjmodel.mesh_vertnum[mesh_id]
    #         vertices = mjmodel.mesh_vert[vert_start:vert_start + vert_count].reshape(-1, 3)
    #         ZP = reach_utils.get_zonotope_from_mesh_vertices(vertices=vertices)

    #         # TODO: register ZP only and transform to get the representation
    #         # ZP = reach_utils.transform_zonotope(ZP, pos = geom_pos, rot = geom_rot)
    #     else:
    #         print(f"Geom type {geom_type} is not supported yet.")
    #         raise NotImplementedError("Not Implemented")
        
    #     if ZP is not None:
    #         ZP = ZP.to(device=self.device, dtype=self.dtype)

    #     return ZP

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
    
    def add_circle(self, img, color):
        """
        Add a circle to the image

        Args:
            img (np.array): image to add the circle
            center (tuple): center of the circle
            radius (int): radius of the circle
        
        Returns:
            img (np.array): updated image with the circle
        """
        radius = 16
        center = [64, 64]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], fill=color)
        return np.array(img)
    
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
        zonotopes = ListUtils.maybe_flatten_the_list(zonotopes)

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

        # initialize the domain
        self.ax.set_xlim([min_V[0] - 0.1, max_V[0] + 0.1])
        self.ax.set_ylim([min_V[1] - 0.1, max_V[1] + 0.1])
        self.ax.set_zlim([min_V[2] - 0.1, max_V[2] + 0.5]) # robot height

        # zoom setting
        self.zoom(zoom_factor = self.render_setting["zoom_factor"])

        # ticks setting
        if not self.render_setting["ticks"]:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            if hasattr(self.ax, 'set_zticks'):
                self.ax.set_zticks([])
        
        # grid setting
        self.ax.grid(self.render_setting["grid"])

    def zoom(self, zoom_factor):
        # Apply zoom by adjusting axis limits
        if zoom_factor != 1.0:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            zlim = self.ax.get_zlim() if hasattr(self.ax, 'get_zlim') else None

            # Calculate the new limits based on the zoom factor
            new_xlim = [(xlim[0] + (xlim[1] - xlim[0]) / 2) - (xlim[1] - xlim[0]) / (2 * zoom_factor), 
                        (xlim[0] + (xlim[1] - xlim[0]) / 2) + (xlim[1] - xlim[0]) / (2 * zoom_factor)]
            new_ylim = [(ylim[0] + (ylim[1] - ylim[0]) / 2) - (ylim[1] - ylim[0]) / (2 * zoom_factor), 
                        (ylim[0] + (ylim[1] - ylim[0]) / 2) + (ylim[1] - ylim[0]) / (2 * zoom_factor)]

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)

            if zlim:
                new_zlim = [(zlim[0] + (zlim[1] - zlim[0]) / 2) - (zlim[1] - zlim[0]) / (2 * zoom_factor), 
                            (zlim[0] + (zlim[1] - zlim[0]) / 2) + (zlim[1] - zlim[0]) / (2 * zoom_factor)]
                self.ax.set_zlim(new_zlim)
    
    def draw_zonotopes_in_kwargs_if_exists(self, kwargs, name):
        """
        Draw the zonotope category of name `name` at the figure canvas

        Args:
            kwargs (dict), the dictionary that has a zonotope kwargs[name]
            name (str), the name of the zonotope category
        """
        if name in kwargs.keys() and kwargs[name] is not None:
            if self.patches[name] is not None:
                self.clear_patches(self.patches[name])
            
            self.patches[name], _ = self.draw_zonotopes(kwargs[name],
                                                        self.render_setting[name]["color"],
                                                        self.render_setting[name]["alpha"],
                                                        self.render_setting[name]["linewidth"])

    def draw_traj3D_in_kwargs_if_exists(self, kwargs, name):
        """
        Draw the trajectory of name `name` at the figure canvas
        
        """
        if name in kwargs.keys() and kwargs[name] is not None:
            if self.plots[name] is not None:
                self.plots[name][0].remove()
            self.plots[name] = self.ax.plot(
                kwargs[name][:, 0], 
                kwargs[name][:, 1], 
                kwargs[name][:, 2], 
                color=self.render_setting[name]["color"], 
                linewidth=self.render_setting[name]["linewidth"],
                linestyle=self.render_setting[name]["linestyle"]
            )

    def custom_render(self, mode=None, height=None, width=None, camera_name="agentview", **kwargs):
        """Renders the environment as 3D zonotope world.

        Args:
            mode (str): the mode to render with
        """
        if self.fig is None:
            self.initialize_renderer()

        # dynamic zonotopes
        self.draw_zonotopes_in_geom_table("dynamic_obs", remove_if_exists=True)
        self.draw_zonotopes_in_geom_table("robot", remove_if_exists=True)

        if "plan" in kwargs.keys():
            if self.plots["plan"] is not None:
                self.plots["plan"][0].remove()
            self.plots["plan"] = self.ax.plot(kwargs["plan"][:, 0], kwargs["plan"][:, 1], 
                                              color     = self.render_setting["plan"]["color"], 
                                              linewidth = self.render_setting["plan"]["linewidth"],
                                              linestyle = self.render_setting["plan"]["linestyle"]
                                            )            

        if "backup_plan" in kwargs.keys():
            if self.plots["backup_plan"] is not None:
                self.plots["backup_plan"][0].remove()
            self.plots["backup_plan"] = self.ax.plot(kwargs["backup_plan"][:, 0], kwargs["backup_plan"][:, 1], 
                                                     color     = self.render_setting["backup_plan"]["color"], 
                                                     linewidth = self.render_setting["backup_plan"]["linewidth"],
                                                     linestyle = self.render_setting["backup_plan"]["linestyle"]
                                                    )
        
        if "plans" in kwargs.keys():
            # Remove existing plot if any
            if self.plots["plans"] is not None:
                for line in self.plots["plans"]:
                    line.remove()
            
            # Initialize list to hold plot objects
            self.plots["plans"] = []
            
            # Generate a colormap
            colormap = plt.cm.get_cmap('tab10', len(kwargs["plans"]))
            
            # Plot each trajectory with a different color
            for i in range(len(kwargs["plans"])):
                plan = kwargs["plans"][i]
                color = colormap(i / len(kwargs["plans"]))
                line, = self.ax.plot(plan[:, 0], plan[:, 1], 
                                    color     = color, 
                                    linewidth = self.render_setting["plans"]["linewidth"],
                                    linestyle = self.render_setting["plans"]["linestyle"],
                                    alpha     = 0.6)
                self.plots["plans"].append(line)
        
        if hasattr(self, "goal_zonotope"):
            # goal zonotopes: TODO
            if self.patches["goal"] is not None:
                self.clear_patches(self.patches["goal"])
            self.patches["goal"], _ = self.draw_zonotopes([self.goal_zonotope], 
                                                         self.render_setting["goal"]["color"], 
                                                         self.render_setting["goal"]["alpha"], 
                                                         self.render_setting["goal"]["linewidth"])
        
        self.adjust_camera(camera_name)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if "intervened" in kwargs.keys() and kwargs["intervened"]:
            img = self.add_circle(img, color = (0, 255, 0)) # if intervened, add green circle
        else:
            img = self.add_circle(img, color = (0, 0, 255)) # if not intervened, add blue circle

        return img

    def adjust_camera(self, camera_name):
        # camera view
        if camera_name == "agentview":
            self.ax.view_init(elev=30, azim=30)
        elif camera_name == "frontview":
            self.ax.view_init(elev=5, azim=0)
        elif camera_name == "topview":
            self.ax.view_init(elev=90, azim=0)
        else:
            raise NotImplementedError