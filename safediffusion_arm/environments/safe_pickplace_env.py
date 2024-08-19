import numpy as np

from safediffusion_arm.environments.safe_arm_env import SafeArmEnv

class SafePickPlaceEnv(SafeArmEnv):
    """
    Assumes SingleArmEnv

    Add observation "T_world_to_base", "state_dict" to the SafeArmEnv
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
        self.patches.update({
            "active_object" : None,
            "plan"          : None,
            "backup_plan"   : None,
            "goal"          : None,
        })
    
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
        # HACK
        obs["state_dict"]         = self.get_state()

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
        """
        Custom renderer function for PickPlaceEnv
        """
        if self.fig is None:
            self.initialize_renderer()

        # dynamic zonotopes
        self.draw_zonotopes_in_geom_table("dynamic_obs", remove_if_exists=True)
        self.draw_zonotopes_in_geom_table("robot", remove_if_exists=True)
        self.draw_zonotopes_in_geom_table("active_object", remove_if_exists=True)

        # extra argument zonotopes
        self.draw_zonotopes_in_kwargs_if_exists(kwargs, "FRS")
        self.draw_zonotopes_in_kwargs_if_exists(kwargs, "goal")

        # extra argument trajectory
        self.draw_traj3D_in_kwargs_if_exists(kwargs, "plan")
        self.draw_traj3D_in_kwargs_if_exists(kwargs, "backup_plan")
        
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