import os
from enum import Enum

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh

from robosuite.environments.base import MujocoEnv
from robosuite.environments.robot_env import RobotEnv
from robosuite.models.objects import MujocoObject
import robosuite.utils.transform_utils as T


use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy:
    from zonopy.contset import zonotope
else:
    from safediffusion.armtdpy.reachability.conSet import zonotope

from safediffusion.armtdpy.environments.arm_3d import Arm_3D
import safediffusion.utils.reachability_utils as reach_utils


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

class ZonotopeMuJoCoEnv(Arm_3D):
    """
    Zontopic Twin of the MuJoCo RobotEnv, tested on PickPlace only yet.
    (NEVER override, modify the MuJoCoEnv)

    NOTE
        1. Represents the visual zonotopes: World, VisualObject
        2. Represents the static, collision zonotopes: Arena, Mount
        3. Represents the dynamic, collision zonotopes: Robot, Gripper, Object
    
    TODO:
        1. Represent the gripper object
        2. Re-hash all the data structure to be more efficient
    """
    def __init__(self, mujoco_env, render_online = False, render_kwargs = None, **kwargs):
        """
        """
        assert isinstance(mujoco_env, MujocoEnv)

        # Allocate variables to initialize parent class
        self.env = mujoco_env
        self.robot_name  = mujoco_env.robots[0].name
        self.policy_freq = mujoco_env.control_freq # (Hz)
        self.render_online = render_online
        self.render_kwargs = render_kwargs

        # Initialize the parent class
        super().__init__(robot = self.robot_name, 
                         T_len = self.policy_freq,
                         reset = False,
                         **kwargs)
        
        # Hash geometry ids
        self.hash_geom_ids()

        # construct zonotopes that is unchanged throughout the simulation
        self._link_zonos_stl = self._Arm_3D__link_zonos # hack: to get the zonotopes from the parent class
        self._link_polyzonos_stl = self.link_zonos # hack: to get the zonotopes from the parent class
        self._object_zonos_stl = [self.zonotope_from_object_mesh_file(obj) for obj in self.env.objects]
        self._visual_object_zonos_stl = [self.zonotope_from_object_mesh_file(obj) for obj in self.env.visual_objects]
        # self._gripper_zonos_stl = [self.zonotope_from_object_mesh_file(obj) for obj in self.env.robots[0].gripper]

        # construct zonotopes with respect to the world frame
        self.arena_zonos = self.get_arena_zonotopes()
        self.mount_zonos = self.get_mount_zonotopes()
        self.arm_zonos = []
        self.object_zonos = []
        self.gripper_zonos = []
        self.visual_object_zonos = []

        # Construct the transformation matrix from the world to the base frame
        self.T_world_to_base = self.get_transform_world_to_base()

        # simulation objects to query the information: should be used only for getter functions
        self.helper_robot = self.env.robots[0]
        self.helper_controller = self.helper_robot.controller

        self.initialize_renderer()

    def hash_geom_ids(self):
        n_geoms = self.get_num_geom()

        self.robot_geom_ids = [geom_id for geom_id in range(n_geoms) if self.get_body_name_from_geom_id(geom_id).startswith("robot")]
        self.gripper_geom_ids = [geom_id for geom_id in range(n_geoms) if self.get_body_name_from_geom_id(geom_id).startswith("gripper")]
        self.mount_geom_ids = [geom_id for geom_id in range(n_geoms) if self.get_body_name_from_geom_id(geom_id).startswith("mount")]
        self.arena_geom_ids = [geom_id for geom_id in range(n_geoms) if self.get_body_name_from_geom_id(geom_id) in ["bin1", "bin2"]]
        self.object_geom_ids = [value[0] for value in self.env.obj_geom_id.values() if len(value) != 0]
    
    ##########################################
    # Construct zonotopes from MuJoCo Models
    ##########################################

    def get_zonotope_from_geom_id(self, geom_id):
        """
        Get the zonotope from the geom_id
        """
        geom_type = self.env.sim.model.geom_type[geom_id]
        geom_pos = self.env.sim.data.geom_xpos[geom_id]
        geom_rot = self.env.sim.data.geom_xmat[geom_id].reshape(3, 3)
        geom_size = self.env.sim.model.geom_size[geom_id]

        # Get Zonotope Primitive
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
            raise NotImplementedError("Not Implemented")
        
        if ZP is not None:
            ZP = ZP.to(device=self.device, dtype=self.dtype)
        return ZP
    
    def zonotope_from_object_mesh_file(self, object):
        """
        Construct the zonotope primitive from the mesh file

        Calls the numpy-stl to compute the vertices and bounding box
        """
        assert isinstance(object, MujocoObject)
        mesh_asset = [asset for asset in object.asset if asset.tag == 'mesh']
        mesh_file  = mesh_asset[0].attrib["file"]
        mesh_ext   = mesh_file.split(".")[-1]
        mesh_file  = mesh_file.replace(mesh_ext, "stl")

        mesh_obj = mesh.Mesh.from_file(mesh_file)
        mesh_V = mesh_obj.vectors.reshape(-1, 3)
        
        # construct bbox zonotope here
        max = mesh_V.max(axis=0)
        min = mesh_V.min(axis=0)

        c = (max+min)/2
        G = np.diag((max-min)/2)
        Z = np.vstack([c, G])

        Z = torch.tensor(Z, dtype=self.dtype, device=self.device)

        return zonotope(Z)
    
    def get_arm_zonotopes(self, frame='world'):
        return self.get_arm_zonotopes_at_q(self.qpos, frame=frame)

    def get_arm_zonotopes_at_q(self, q, frame='world'):
        """
        Return the forward occupancy of the robot arms with the zonotopes.
        It does by doing the forward kinematics of the robot.

        Args
            qpos: (n_links,) torch tensor

        Output
            arm_zonos: list of zonotopes
        """
        if frame == 'world':
            Ri = torch.asarray(self.T_world_to_base[0:3, 0:3], dtype=self.dtype, device=self.device)
            Pi = torch.asarray(self.T_world_to_base[0:3, 3], dtype=self.dtype, device=self.device)

        elif frame == 'base':
            Ri = torch.eye(3,dtype=self.dtype,device=self.device)
            Pi = torch.zeros(3,dtype=self.dtype,device=self.device)
        else:
            raise NotImplementedError("Not Implemented")
        
        R_qi = self.rot(q)
        arm_zonos = []
        for i in range(self.n_links):
            Pi = Ri@self.P0[i] + Pi
            Ri = Ri@self.R0[i]@R_qi[i]
            arm_zono = Ri@self._link_zonos_stl[i] + Pi
            arm_zonos.append(arm_zono)
        
        return arm_zonos
    
    def get_arena_zonotopes(self):
        """
        Get the zonotopes of the arena
        """
        zonotopes = []
        for geom_id in self.arena_geom_ids:
            if self.is_visual_geom(geom_id):
                Z = None
            else:
                Z = self.get_zonotope_from_geom_id(geom_id)
            zonotopes.append(Z)

        zonotopes = [zono for zono in zonotopes if zono is not None]

        return zonotopes
    
    def get_object_zonotopes(self):
        """
        Get the zonotope of the objects
        We assume that all objects are created with mesh file
        """
        zonotopes = []
        for obj_idx, obj in enumerate(self.env.objects):
            obj_pos = self.env.sim.data.get_body_xpos(obj.root_body)
            obj_rot = self.env.sim.data.get_body_xmat(obj.root_body)
            obj_stl = self._object_zonos_stl[obj_idx]
            object_zono = reach_utils.transform_zonotope(zono=obj_stl, pos=obj_pos, rot=obj_rot)
            zonotopes.append(object_zono)

        return zonotopes
    
    def get_visual_object_zonotopes(self):
        """
        Get the zonotope of the objects
        We assume that all objects are created with mesh file
        """
        zonotopes = []
        for obj_idx, obj in enumerate(self.env.visual_objects):
            obj_pos = self.env.sim.data.get_body_xpos(obj.root_body)
            obj_rot = self.env.sim.data.get_body_xmat(obj.root_body)
            obj_stl = self._visual_object_zonos_stl[obj_idx]
            object_zono = reach_utils.transform_zonotope(zono=obj_stl, pos=obj_pos, rot=obj_rot)
            zonotopes.append(object_zono)

        return zonotopes
    
    def get_gripper_zonotopes(self):
        """
        Get the zonotopes of the robot gripper
        """
        zonotopes = []
        for geom_id in self.gripper_geom_ids:
            if self.is_visual_geom(geom_id):
                Z = None
            else:
                Z = self.get_zonotope_from_geom_id(geom_id)
            zonotopes.append(Z)

        zonotopes = [zono for zono in zonotopes if zono is not None]
        
        return zonotopes
    
    def get_mount_zonotopes(self):
        """
        Get the zonotopes of the mount
        """

        zonotopes = []
        for geom_id in self.mount_geom_ids:
            if self.is_visual_geom(geom_id):
                Z = None
            else:
                Z = self.get_zonotope_from_geom_id(geom_id)
            zonotopes.append(Z)

        zonotopes = [zono for zono in zonotopes if zono is not None]
        
        return zonotopes
    
    def is_visual_geom(self, geom_id):
        """
        Check if the geom_id is in collision with the robot or objects
        """
        geom_name = self.env.sim.model.geom_id2name(geom_id)
        if geom_name is not None and (geom_name.endswith("visual") or geom_name.endswith("vis")):
            return True
        return False
    
    ##########################################
    # Transforms
    ##########################################
    def get_transform_world_to_base(self):
        T_world_to_base = np.zeros((4, 4))
        T_world_to_base[0:3, 0:3] = self.env.sim.data.get_body_xmat("robot0_base")
        T_world_to_base[0:3, 3]   = self.env.sim.data.get_body_xpos("robot0_base")
        T_world_to_base[3, 3] = 1

        return T_world_to_base
    
    ###############################
    ###### GETTER FUNCTIONS  ######
    ###############################
    def get_qpos_from_sim(self, robot_idx=0):
        qpos = self.env.sim.data.qpos[self.env.robots[robot_idx].controller.qpos_index]
        qpos = torch.tensor(qpos, dtype=self.dtype, device=self.device)
        return qpos
    
    def get_qvel_from_sim(self, robot_idx=0):
        qvel = self.env.sim.data.qvel[self.env.robots[robot_idx].controller.qpos_index]
        qvel = torch.tensor(qvel, dtype=self.dtype, device=self.device)
        return qvel
    
    def get_num_geom(self):
        return self.env.sim.model.ngeom
    
    def get_body_name_from_geom_id(self, geom_id):
        body_id = self.env.sim.model.geom_bodyid[geom_id]
        return self.env.sim.model.body_id2name(body_id)
    
    def get_geom_info_table(self):
        geom_info = dict()
        for geom_id in range(self.get_num_geom()):
            geom_info[geom_id] = dict()
            geom_info[geom_id]["geom_name"] = self.env.sim.model.geom_id2name(geom_id)
            geom_info[geom_id]["geom_type"] = self.env.sim.model.geom_type[geom_id]
            geom_info[geom_id]["body_name"] = self.get_body_name_from_geom_id(geom_id)

        return geom_info
    
    ################################
    # Env Template Functions
    ################################
    def reset(self):
        raise NotImplementedError
    
    def sync(self):
        """
        Reset - synchronize with the MuJoCo environment
        """
        # TODO 1. self.qgoal = update the position of the goal according to the mode (pick, place)
        # sync the robot state with the mujoco env
        self.qpos = self.get_qpos_from_sim()
        self.qvel = self.get_qvel_from_sim()
        self.qgoal = np.zeros(self.n_links,) # TODO: change this to the object to place & pick using inverse kinematics

        # sync the zonotopes for visualization (all frame is respect to the world frame)
        self.arm_zonos = self.get_arm_zonotopes(frame='world')
        self.gripper_zonos = self.get_gripper_zonotopes()
        self.object_zonos = self.get_object_zonotopes()
        self.visual_object_zonos = self.get_visual_object_zonotopes()

        # TODO: construct obstacle flag and add obstacles here
        obs_zonos = []
        obs_zonos.extend(self.arena_zonos)
        obs_zonos.extend(self.mount_zonos)
        # obs_zonos.extend(self.object_zonos)
        self.obs_zonos = obs_zonos
        
        # reset the internal status
        self.done = False
        self.collision = False

        return self.get_observations()
    
    def step(self, action):
        """
        Step function is not available for the zonotope environment
        """
        raise NotImplementedError("Not Implemented")
    
    def close(self):
        super().close()

    def get_observations(self):
        observation = {'qpos':self.qpos,'qvel':self.qvel}        
        return observation

    ###################################
    # Visualization Code              #
    ###################################
    def initialize_renderer(self):
        render_width = 20
        render_height = 20
        arena_color = 'red'
        arena_alpha = 0.1
        arena_linewidth = 0.1

        mount_color = 'red'
        mount_alpha = 0.5
        mount_linewidth = 0.05
        
        if self.fig is None:
            if self.render_online:
                plt.ion()
            
            self.fig = plt.figure()
            self.fig.set_size_inches(render_width, render_height)
            self.ax = plt.axes(projection='3d')

            # TODO: check this
            if not self.ticks:
                plt.tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            
            if self.render_kwargs is not None and "save_dir" in self.render_kwargs:
                os.makedirs(self.render_kwargs["save_dir"], exist_ok=True)

        # Initialize the patches for the robots, objects, arenas (fixed throughout the simulation)
        self.arm_patches = self.ax.add_collection3d(Poly3DCollection([]))
        self.gripper_patches = self.ax.add_collection3d(Poly3DCollection([]))
        self.object_patches = self.ax.add_collection3d(Poly3DCollection([]))
        self.visual_object_patches = self.ax.add_collection3d(Poly3DCollection([]))
        self.FRS_patches = self.ax.add_collection3d(Poly3DCollection([])) # FRS patches of the backup plan
        self.FO_desired_patches = self.ax.add_collection3d(Poly3DCollection([])) # FO patches of the desired plan
        self.FO_backup_patches = self.ax.add_collection3d(Poly3DCollection([])) # FO patches of the backup plan

        arena_patches_data = torch.vstack([arena_zono.polyhedron_patch() for arena_zono in self.arena_zonos])
        self.arena_patches = self.ax.add_collection3d(Poly3DCollection(arena_patches_data, 
                                                                        edgecolor='black', 
                                                                        facecolor=arena_color, 
                                                                        alpha=arena_alpha, 
                                                                        linewidths=arena_linewidth))
        
        mount_patches_data = torch.vstack([mount_zono.polyhedron_patch() for mount_zono in self.mount_zonos])
        self.mount_patches = self.ax.add_collection3d(Poly3DCollection(mount_patches_data, 
                                                                        edgecolor='black', 
                                                                        facecolor=mount_color, 
                                                                        alpha=mount_alpha, 
                                                                        linewidths=mount_linewidth))

        vertices = torch.vstack([mount_patches_data.reshape(-1, 3),
                                 arena_patches_data.reshape(-1, 3)])
        max_V = vertices.cpu().numpy().max(axis=0)
        min_V = vertices.cpu().numpy().min(axis=0)

        self.ax.set_xlim([min_V[0] - 0.1, max_V[0] + 0.1])
        self.ax.set_ylim([min_V[1] - 0.1, max_V[1] + 0.1])
        self.ax.set_zlim([min_V[2] - 0.1, max_V[2] + 0.5]) # robot height
        
    def render(self, FO_desired_zonos=None, FO_backup_zonos=None):
        # Vis settings here
        robot_color = 'black'
        robot_alpha = 0.5
        robot_linewidth = 1

        gripper_color = 'black'
        gripper_alpha = 1
        gripper_linewidth = 1

        object_color = 'blue'
        object_alpha = 0.8
        object_linewidth = 1

        visual_object_color = 'green'
        visual_object_alpha = 0.2
        visual_object_linewidth = 1

        FO_desired_color = 'cyan'
        FO_desired_alpha = 0.3
        FO_desired_linewidth = 0.05

        FO_backup_color = 'green'
        FO_backup_alpha = 0.05
        FO_backup_linewidth = 0.05

        # clear all the previous patches
        self.arm_patches.remove()
        self.gripper_patches.remove()
        self.object_patches.remove()
        self.visual_object_patches.remove()
        
        # Render the robot links and gripper
        arm_patches_data = torch.vstack([arm_zono.polyhedron_patch() for arm_zono in self.arm_zonos])
        self.arm_patches = self.ax.add_collection3d(Poly3DCollection(arm_patches_data, 
                                                                      edgecolor=robot_color, 
                                                                      facecolor=robot_color, 
                                                                      alpha=robot_alpha, 
                                                                      linewidths=robot_linewidth))
        
        gripper_patches_data = torch.vstack([gripper_zono.polyhedron_patch() for gripper_zono in self.gripper_zonos])
        self.gripper_patches = self.ax.add_collection3d(Poly3DCollection(gripper_patches_data, 
                                                                edgecolor=gripper_color, 
                                                                facecolor=gripper_color, 
                                                                alpha=gripper_alpha, 
                                                                linewidths=gripper_linewidth))
        
        # Render the objects
        object_patches_data = torch.vstack([obj.polyhedron_patch() for obj in self.object_zonos])
        self.object_patches = self.ax.add_collection3d(Poly3DCollection(object_patches_data, 
                                                                        edgecolor=object_color, 
                                                                        facecolor=object_color, 
                                                                        alpha=object_alpha, 
                                                                        linewidths=object_linewidth))
        
        # Render the visual objects
        visual_object_patches_data = torch.vstack([obj.polyhedron_patch() for obj in self.visual_object_zonos])
        self.visual_object_patches = self.ax.add_collection3d(Poly3DCollection(visual_object_patches_data, 
                                                                        edgecolor=visual_object_color, 
                                                                        facecolor=visual_object_color, 
                                                                        alpha=visual_object_alpha, 
                                                                        linewidths=visual_object_linewidth))
            
        if FO_backup_zonos is not None:
            self.FO_backup_patches.remove()
            FO_backup_to_vis = []
            for FO_backup_zonos_at_t in FO_backup_zonos:
                FO_backup_to_vis.extend(FO_backup_zonos_at_t)
            FO_backup_patches_data = torch.vstack([obj.polyhedron_patch() for obj in FO_backup_to_vis])
            self.FO_backup_patches = self.ax.add_collection3d(Poly3DCollection(FO_backup_patches_data,
                                                                        edgecolor=FO_backup_color, 
                                                                        facecolor=FO_backup_color, 
                                                                        alpha=FO_backup_alpha, 
                                                                        linewidths=FO_backup_linewidth))
        
        if FO_desired_zonos is not None:
            self.FO_desired_patches.remove()
            FO_desired_to_vis = []
            for FO_desired_zonos_at_t in FO_desired_zonos:
                FO_desired_to_vis.extend(FO_desired_zonos_at_t)
            FO_desired_patches_data = torch.vstack([obj.polyhedron_patch() for obj in FO_desired_to_vis])
            self.FO_desired_patches = self.ax.add_collection3d(Poly3DCollection(FO_desired_patches_data, 
                                                                        edgecolor=FO_desired_color, 
                                                                        facecolor=FO_desired_color, 
                                                                        alpha=FO_desired_alpha, 
                                                                        linewidths=FO_desired_linewidth))

        # Flush and re-draw      
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Save the figure
        if self.render_kwargs is not None and "save_dir" in self.render_kwargs:
            plt.savefig(os.path.join(self.render_kwargs["save_dir"], f"task.png"))
        

        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return img


#########
# DEPRECATED
#########
    # def zonotope_from_mujoco_object(self, object):
    #     """
    #     Create zonotopic representation of bounding box of MuJoCo Object

    #     Args
    #         env: Robosuite Environment
    #         object: MujocoObject

    #     Output
    #         zonotope

    #     NOTE: Need to check if the center of the bounding box is the center of the root body.
    #     -- objects/MuJoCoXMLObject: bbox[2] = max(obj.bottom_offset, obj.bbox_top_offset) - obj.bottom_offset
    #     -- The z-axis is not aligned -- why?
    #     """
    #     assert isinstance(object, MujocoObject)

    #     # env.objects[0].xml

    #     c = self.env.sim.data.get_body_xpos(object.root_body)
    #     R = self.env.sim.data.get_body_xmat(object.root_body)

    #     G = np.diag(object.get_bounding_box_half_size())
    #     G = R@G

    #     Z = np.vstack([c, G])

    #     return zonotope(Z)