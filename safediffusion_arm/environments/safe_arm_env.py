import os
import numpy as np

from safediffusion.envs.env_safety import Geom
from safediffusion.envs.env_zonotope import ZonotopeEnv
import matplotlib.pyplot as plt

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
        """ Register the collision geometry that is considered dangerous and static
        """
        # get geometry that belong to mount & bin
        geoms = self.get_geom_that_body_name_starts_with(["mount", "bin"])
        # filter out the `visual` geometry
        geoms = [geom for geom in geoms if not self.is_geom_visual(geom)]

        return geoms


    def register_dynamic_obstacle_geoms(self):
        """ Register the geometry that is considered dangerous and dynamic
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

    def get_observation(self, obs=None):
        """ Postprocess observation        
        """
        obs = super().get_observation(obs)

        return obs
    
    def reset(self):
        """ Reset the environment

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


class SafePickPlaceEnv(SafeArmEnv):
    """
    Assumes SingleArmEnv
    """
    def __init__(self, env, init_active_object = "milk", **kwargs):
        """
        init_active_object_id: the id of initial object to grasp
        """

        self.active_object    = init_active_object
        
        super().__init__(env, **kwargs)

        self.gripper_innerpad_geoms = []
        self.gripper_innerpad_geoms.extend(self.unwrapped_env.robots[0].gripper.important_geoms["left_fingerpad"])
        self.gripper_innerpad_geoms.extend(self.unwrapped_env.robots[0].gripper.important_geoms["right_fingerpad"]) 
        self.gripper_innerpad_geoms.extend(["gripper0_right_inner_knuckle_collision",
                                            "gripper0_right_inner_finger_collision",
                                            "gripper0_right_fingertip_collision",
                                            "gripper0_left_inner_knuckle_collision",
                                            "gripper0_left_inner_finger_collision",
                                            "gripper0_left_fingertip_collision"
                                        ])
        
        # Render settings
        self.patches["active_object"] = None

    def get_observation(self, obs=None):
        """
        PickPlace 
        """
        obs = super().get_observation(obs)

        # add the transform between the world to the robot base
        mjdata = self.unwrapped_env.sim.data
        T_world_to_base           = np.zeros((4, 4))
        T_world_to_base[0:3, 0:3] = mjdata.get_body_xmat("robot0_base")
        T_world_to_base[0:3, 3]   = mjdata.get_body_xpos("robot0_base")
        T_world_to_base[3, 3]     = 1

        obs["T_world_to_base"]    = T_world_to_base 

        return obs

    
    def is_safe(self):
        """
        Definition of the safety at pick & place task. The robot has a few phases

        1) Reach 
            At "reach phase" (robot reaches toward the active object),
            the safety specifications are:
            - robot not colliding with the non-active object / mount / arena
            - robot (EXCEPT inner fingerpad) not colliding with the active object
                - this encourages contact between "inner side" of gripper and active object
                - restricting inner fingerpad to the fingerpad only emphasizes the stable grasp

        2) Grasp
            At "grasp phase" (robot grasping the active object),
            the safety specifications are: 
            - robot not colliding with the non-active object / mount / arena

        3) Lift / Hover
            At "lift phase" (robot grasping & lifting the active object)
            the safety specifications are:
            - robot + active object not colliding with the non-active object / mount / arena
        
        NOTE: This safety specification is imagining that `place` phase does not require
        smooth placing. User can add extra safety specifications
        """
        
        # pointer to mjmodel
        mjmodel = self.unwrapped_env.sim.model
        mjdata  = self.unwrapped_env.sim.data

        # geometry ids
        active_obj = self.unwrapped_env.objects[self.unwrapped_env.object_to_id[self.active_object.lower()]]
        active_object_geom_ids = {mjmodel.geom_name2id(name) 
                                  for name in active_obj.contact_geoms}

        obstacle_geom_ids = {geom.id for geom in self.static_obstacle_geoms}.union(
                            {geom.id for geom in self.dynamic_obstacle_geoms})

        robot_geom_ids = {geom.id for geom in self.robot_geoms}

        gripper_innerpad_geom_ids = {mjmodel.geom_name2id(name) for name in self.gripper_innerpad_geoms}

        # decide the phase
        # dist_from_gripper_to_active_obj = self.dist_from_gripper_to_obj(active_obj)
        dist_from_plane_to_active_obj   = self.dist_from_plane_to_obj(active_obj)
        grasping = self.is_grasping()

        is_safe_now = True

        if not grasping:
            # Reach
            for i in range(mjdata.ncon):
                contact = mjdata.contact[i]

                if self.check_contact(contact, robot_geom_ids, obstacle_geom_ids) and \
                   not self.check_contact(contact, gripper_innerpad_geom_ids, active_object_geom_ids):
                    is_safe_now = False
                    break
        
        elif dist_from_plane_to_active_obj < 0.05:
            # Grasp
            # If robot collides with non-active objects / mount / arena, it is unsafe.
            # NOTE: robot colliding with the active object is safe
            obstacle_without_active_object  = obstacle_geom_ids.difference(active_object_geom_ids)
            for i in range(mjdata.ncon):
                contact = mjdata.contact[i]

                if self.check_contact(contact, robot_geom_ids, obstacle_without_active_object):
                    is_safe_now = False
                    break

        else:
            # Lift & Hover
            obstacle_without_active_object  = obstacle_geom_ids.difference(active_object_geom_ids)
            robot_with_active_object        = robot_geom_ids.union(active_object_geom_ids)
            
            for i in range(self.unwrapped_env.sim.data.ncon):
                contact = self.unwrapped_env.sim.data.contact[i]

                if self.check_contact(contact, obstacle_without_active_object, robot_with_active_object):
                    is_safe_now = False
                    break
                
        return is_safe_now
        
    def dist_from_gripper_to_obj(self, obj):
        """
        Return the distance from the gripper to the object

        Args:
            obj (MjModel): MuJoCo model of the object

        Returns:
            dist (int): the distance from the object to the gripper
        """
        dist = self.unwrapped_env._gripper_to_target(
            gripper = self.unwrapped_env.robots[0].gripper,
            target  = obj.root_body,
            target_type = "body",
            return_distance = True
        )

        return dist

    def dist_from_plane_to_obj(self, obj):
        """
        Return the distance from the plane to the bottom of the object

        Args:
            obj (MjModel): MuJoCo model of the object

        Returns:
            dist (float): the distance from the bottom of the object to the floor of the bin.
        """
        z_floor = self.unwrapped_env.bin2_pos[2]
        object_z_locs = self.unwrapped_env.sim.data.body_xpos[[self.unwrapped_env.obj_body_id[obj.name]]][:, 2]
        object_z_locs_bottom = object_z_locs + obj.bottom_offset[2]

        dist = np.maximum(z_floor - object_z_locs_bottom, 0)

        return dist


    def check_contact(self, contact, geom1_group, geom2_group):
        """
        Check if the contact is between the geometry gruop 1 and group 2.

        Args
            contact: MjSim.contact
            geom1_group: list (int)
            geom2_group: list (int)
        
        Returns
            Returns true if the contact is made
        """
        return (contact.geom1 in geom1_group and contact.geom2 in geom2_group) or \
               (contact.geom2 in geom1_group and contact.geom1 in geom2_group)

    def is_grasping(self):
        """
        Check if the gripper is grasping the active object

        Returns:
            bool
        """
        gripper       = self.unwrapped_env.robots[0].gripper
        active_object = self.unwrapped_env.objects[self.unwrapped_env.object_to_id[self.active_object.lower()]]

        is_grasping   = self.unwrapped_env._check_grasp(
                                                gripper=gripper,
                                                object_geoms=active_object.contact_geoms
                                            )
        return is_grasping
    
    def create_geom_table(self):
        """
        Create useful geometry table

        For SafePickPlaceEnv, we add the "active_object" column that identifies the geometry
        that belongs to the active object
        """
        geom_table = super().create_geom_table()

        active_object_id   = self.unwrapped_env.object_to_id[self.active_object.lower()]
        active_object_name = self.unwrapped_env.obj_names[active_object_id]
        active_geom_id     = self.unwrapped_env.obj_geom_id[active_object_name]
        
        geom_table['active_object'] = geom_table.index.isin(active_geom_id)

        return geom_table

    def custom_render(self, mode=None, height=None, width=None, camera_name=None, **kwargs):
        if self.fig is None:
            self.initialize_renderer()

        # dynamic zonotopes
        self.draw_zonotopes_in_geom_table("dynamic_obs", remove_if_exists=True)
        self.draw_zonotopes_in_geom_table("robot", remove_if_exists=True)
        self.draw_zonotopes_in_geom_table("active_object", remove_if_exists=True)

        # camera view
        if camera_name == "agentview":
            self.ax.view_init(elev=30, azim=30)
        elif camera_name == "frontview":
            self.ax.view_init(elev=5, azim=0)
        elif camera_name == "topview":
            self.ax.view_init(elev=90, azim=0)
        else:
            raise NotImplementedError

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if "intervened" in kwargs.keys() and kwargs["intervened"]:
            img = self.add_circle(img, color = (0, 255, 0)) # if intervened, add green circle
        else:
            img = self.add_circle(img, color = (0, 0, 255)) # if not intervened, add blue circle

        return img


class SafePickPlaceBreadEnv(SafePickPlaceEnv):
    """
    This environment is created only to override the notion of success.
    """
    def __init__(self, env, **kwargs):
        super().__init__(env, init_active_object="bread", **kwargs)
    
    def is_success(self):
        """
        The task is success if the bread is in the bin
        """
        active_object_id = self.unwrapped_env.object_to_id[self.active_object.lower()]
        succ = bool(self.unwrapped_env.objects_in_bins[active_object_id])

        return {"task": succ}

