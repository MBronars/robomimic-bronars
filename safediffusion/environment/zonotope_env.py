import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from armtd.environments.arm_3d import Arm_3D
from robosuite.environments.base import MujocoEnv
from robosuite.environments.robot_env import RobotEnv
from robosuite.models.objects import MujocoObject
import robosuite.utils.transform_utils as T
from zonopy.contset.zonotope.zono import zonotope
from enum import Enum, auto, unique

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
    Environment for the 3D arm with zonotopes as obstacles.

    Philosophy: NEVER MODIFY THE MUJOCOENV
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
        self._link_zonos_stl = self._Arm_3D__link_zonos
        self.arena_zonotopes = self.get_arena_zonotopes()
        self.mount_zonotopes = self.get_mount_zonotopes()
        self.initialize_renderer()

    def hash_geom_ids(self):
        n_geoms = self.get_num_geom()

        self.robot_geom_ids = [geom_id for geom_id in range(n_geoms) if self.get_body_name_from_geom_id(geom_id).startswith("robot")]
        self.gripper_geom_ids = [geom_id for geom_id in range(n_geoms) if self.get_body_name_from_geom_id(geom_id).startswith("gripper")]
        self.mount_geom_ids = [geom_id for geom_id in range(n_geoms) if self.get_body_name_from_geom_id(geom_id).startswith("mount")]
        self.arena_geom_ids = [geom_id for geom_id in range(n_geoms) if self.get_body_name_from_geom_id(geom_id) in ["bin1", "bin2"]]
        self.object_geom_ids = [value[0] for value in self.env.obj_geom_id.values() if len(value) != 0]
    
    #######################################
    # Construct zonotopes from geometry
    #######################################

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

        return ZP
    
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

        # env.objects[0].xml

        c = self.env.sim.data.get_body_xpos(object.root_body)
        R = self.env.sim.data.get_body_xmat(object.root_body)

        G = np.diag(object.get_bounding_box_half_size())
        G = R@G

        Z = np.vstack([c, G])

        return zonotope(Z)
    
    def is_visual_geom(self, geom_id):
        """
        Check if the geom_id is in collision with the robot or objects
        """
        geom_name = self.env.sim.model.geom_id2name(geom_id)
        if geom_name is not None and (geom_name.endswith("visual") or geom_name.endswith("vis")):
            return True
        return False
    
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
        return self.env.sim.model.geom_type.shape[0]
    
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

        # TODO: START HERE
        Ri = torch.asarray(self.env.sim.data.get_body_xmat('robot0_base'), dtype=self.dtype,device=self.device)
        Pi = torch.asarray(self.env.sim.data.get_body_xpos('robot0_base'), dtype=self.dtype,device=self.device)

        link_zonos = []
        for i in range(self.n_links):
            Pi = Ri@self.P0[i] + Pi
            Ri = Ri@self.R0[i]@R_qi[i]
            link_zono = Ri@self._link_zonos_stl[i] + Pi
            link_zonos.append(link_zono)
        
        return link_zonos
    
    ################################
    # Env Template Functions
    ################################
    def reset(self):
        """
        Reset - synchronize with the MuJoCo environment
        """
        # TODO 1. self.qgoal = update the position of the goal according to the mode (pick, place)
        
        # sync the robot state with the mujoco env
        self.qpos = self.get_qpos_from_sim()
        self.qvel = self.get_qvel_from_sim()
        self.link_zonos = self.zono_FO(self.qpos)
        self.gripper_zonotopes = self.get_gripper_zonotopes()

        # sync the obstacles with the mujoco env
        self.object_zonotopes = [self.zonotope_from_mujoco_object(obj) for obj in self.env.objects]
        
        
        # reset the internal status
        self.done = False
        self.collision = False


        return self.get_observations()
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
    
    def close(self):
        super().close()

    def get_observations(self):
        observation = {'qpos':self.qpos,'qvel':self.qvel}        
        return observation

    ###################################
    # Visualization Code              #
    ###################################
    def initialize_renderer(self):
        arena_color = 'red'
        arena_alpha = 0.1
        arena_linewidth = 0.5

        mount_color = 'red'
        mount_alpha = 0.1
        mount_linewidth = 0.5

        
        if self.fig is None:
            if self.render_online:
                plt.ion()
            
            self.fig = plt.figure()
            self.ax = plt.axes(projection='3d')
            self.ax.set_xlim(self.render_kwargs["xlim"])
            self.ax.set_ylim(self.render_kwargs["ylim"])
            self.ax.set_zlim(self.render_kwargs["zlim"])

            # TODO: check this
            if not self.ticks:
                plt.tick_params(which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            
            if self.render_kwargs is not None and "save_dir" in self.render_kwargs:
                os.makedirs(self.render_kwargs["save_dir"], exist_ok=True)

        # Initialize the patches for the robots, objects, arenas (fixed throughout the simulation)
        self.link_patches = self.ax.add_collection3d(Poly3DCollection([]))
        self.gripper_patches = self.ax.add_collection3d(Poly3DCollection([]))
        self.object_patches = self.ax.add_collection3d(Poly3DCollection([]))

        arena_patches_data = torch.vstack([arena_zono.polyhedron_patch() for arena_zono in self.arena_zonotopes])
        self.arena_patches = self.ax.add_collection3d(Poly3DCollection(arena_patches_data, 
                                                                        edgecolor='black', 
                                                                        facecolor=arena_color, 
                                                                        alpha=arena_alpha, 
                                                                        linewidths=arena_linewidth))
        
        mount_patches_data = torch.vstack([mount_zono.polyhedron_patch() for mount_zono in self.mount_zonotopes])
        self.mount_patches = self.ax.add_collection3d(Poly3DCollection(mount_patches_data, 
                                                                        edgecolor='black', 
                                                                        facecolor=mount_color, 
                                                                        alpha=mount_alpha, 
                                                                        linewidths=mount_linewidth))
        
        # self.env.sim.data.get_geom_xpos
        
        # self.env.sim.model.geom_bodyid
        
    def render(self):
        # Vis settings here
        robot_color = 'blue'
        robot_alpha = 0.5
        robot_linewidth = 1

        gripper_color = 'black'
        gripper_alpha = 1
        gripper_linewidth = 1

        object_color = 'green'
        object_alpha = 0.8
        object_linewidth = 1


        # Render the robot
        link_patches_data = torch.vstack([link_zono.polyhedron_patch() for link_zono in self.link_zonos])
        self.link_patches = self.ax.add_collection3d(Poly3DCollection(link_patches_data, 
                                                                      edgecolor=robot_color, 
                                                                      facecolor=robot_color, 
                                                                      alpha=robot_alpha, 
                                                                      linewidths=robot_linewidth))
        
        gripper_patches_data = torch.vstack([gripper_zono.polyhedron_patch() for gripper_zono in self.gripper_zonotopes])
        self.gripper_patches = self.ax.add_collection3d(Poly3DCollection(gripper_patches_data, 
                                                                edgecolor=gripper_color, 
                                                                facecolor=gripper_color, 
                                                                alpha=gripper_alpha, 
                                                                linewidths=gripper_linewidth))
        
        # Render the objects
        object_patches_data = torch.vstack([obj.polyhedron_patch() for obj in self.object_zonotopes])
        self.object_patches = self.ax.add_collection3d(Poly3DCollection(object_patches_data, 
                                                                        edgecolor=object_color, 
                                                                        facecolor=object_color, 
                                                                        alpha=object_alpha, 
                                                                        linewidths=object_linewidth))
        
        # Render the objects
        object_patches_data = torch.vstack([obj.polyhedron_patch() for obj in self.object_zonotopes])
        self.object_patches = self.ax.add_collection3d(Poly3DCollection(object_patches_data, 
                                                                        edgecolor=object_color, 
                                                                        facecolor=object_color, 
                                                                        alpha=object_alpha, 
                                                                        linewidths=object_linewidth))
        
        # Render the arena        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Save the figure
        if self.render_kwargs is not None and "save_dir" in self.render_kwargs:
            plt.savefig(os.path.join(self.render_kwargs["save_dir"], f"task.png"))