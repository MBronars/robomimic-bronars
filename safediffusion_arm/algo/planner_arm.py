import os
import math
from copy import deepcopy
from collections import deque

import torch
import numpy as np
import einops

import robosuite

from safediffusion.armtdpy.environments.arm_3d import Arm_3D
from safediffusion.armtdpy.reachability.forward_occupancy.FO import forward_occupancy
from safediffusion.armtdpy.reachability.joint_reachable_set.load_jrs_trig import preload_batch_JRS_trig
from safediffusion.armtdpy.reachability.joint_reachable_set.process_jrs_trig import process_batch_JRS_trig

from safediffusion.algo.helper import traj_uniform_acc, ReferenceTrajectory
from safediffusion.algo.planner_base import ParameterizedPlanner
from safediffusion.utils.npy_utils import scale_array_from_A_to_B
import safediffusion.utils.transform_utils as TransformUtils
import safediffusion.utils.math_utils as MathUtils

# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope, batchZonotope, polynomial_zonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope, batchZonotope, polyZonotope

class ArmtdPlanner(ParameterizedPlanner):
    def __init__(self, 
                 action_config,
                 robot_name   = "Kinova3",
                 gripper_name = None,
                 dt           = 0.01,
                 t_f          = 1,
                 time_pieces  = [0.5],
                 **kwargs):
        """
        Armtd-style planner for the arm robot

        Args:
            action_config (dict): the configuration for the action to the environment

        TODO: inputting action_config is awkward. Refactor later.
        TODO: tensor, numpy is awkward right now (tensor = inside computation, numpy = observation)
        """
        # prepare the useful variables for robot configuration, joint limits, reachable sets
        self.dimension    = 3
        self.robot_name   = robot_name
        self.gripper_name = gripper_name
        self.time_pieces  = time_pieces
        
        # TODO: n_joints is more useful than n_joints. change it to n_joints
        self.n_links = self.get_robot_n_links()

        # State and Parameter Definition
        state_dict = {f"q{joint_id}": joint_id for joint_id in range(self.n_links)}
        param_dict = {f"kv{joint_id}": joint_id for joint_id in range(self.n_links)}
        param_dict.update({f"ka{joint_id}": joint_id + self.n_links for joint_id in range(self.n_links)})

        ParameterizedPlanner.__init__(self, state_dict, param_dict, dt, t_f, **kwargs)

        # robot configuration
        self.load_robot_config_and_joint_limits()
        self.load_robot_link_zonotopes()
        self.load_joint_reachable_set()
        if self.gripper_name is not None:
            self.load_gripper_config(self.gripper_name)

        # load zonotope configuration
        self.zono_order = kwargs["zonotope"]["order"]
        self.max_combs  = kwargs["zonotope"]["max_comb"]
        self.combs      = self.generate_combinations_upto(self.max_combs)

        # trajectory optimization
        self.opt_dim     = range(self.n_links, 2*self.n_links)
        self.weight_dict = dict(joint_pos_goal       = 0.0, 
                                joint_pos_projection = 0.0,
                                grasp_pos_goal       = 0.0)

        # Environment action configuration
        self.t_cur         = 0
        self.action_config = action_config
        
        # Internal status
        self.calibrated    = False

        # FUTUREPROOFING
        # TODO: for each constraint tag, implement prepare_data, compute_constraint
        # TODO: for each objective tag, implement prepare_data, compute_objective
        # self.constraints = [JointConstraint(), ArmConstraint(), GripperConstraint()]
        # self.objectives  = [JointGoalObjective(), GripperGoalObjective(), ProjectionObjective()]
        # self.register_constraints(["arm_collision", "joint_position", "joint_velocity", "gripper_collision"])
        # self.register_objectives(["joint_pos_goal", "joint_pos_projection", "eef_goal"])

    # ---------------------------------------------- #
    # Loading the robot configurations
    # ---------------------------------------------- #
    def calibrate(self, env):
        """
        Set the transformation matrix from the world coordinate to the arm base coordinate
        """
        self.disp("Calibrating...")
        obs                      = env.get_observation()
        self.T_world_to_arm_base = self.to_tensor(obs["T_world_to_arm_base"])
        self.calibrated          = True

    def load_joint_reachable_set(self):
        """
        Load offline-computed joint reachable set

        TODO: vel_lim should be parsed from the offline JRS config
        """
        # load pre-computed joint reachable set
        self.JRS_tensor = self.to_tensor(preload_batch_JRS_trig())
        
        # load the parameter bound used for computing JRS
        delta_kv        = self.vel_lim
        delta_ka        = torch.pi/24 * torch.ones(self.n_links,)
        self.FRS_info   = {"delta_k": torch.hstack((delta_kv, delta_ka))}

    
    def load_robot_link_zonotopes(self):
        """
        Load the zonotope representations of the given `robot_name`

        TODO: unify link_zono and link_polyzono later
        """
        # load the robot arm zonotopes
        robot_model              = Arm_3D(self.robot_name)

        self._link_zonos_stl     = robot_model._link_zonos
        self._link_polyzonos_stl = robot_model.link_zonos
        
    
    def get_robot_n_links(self, robot_name):
        """
        Load the number of the links given the robot name

        Args:
            robot_name (str): the name of the robot
        
        Returns:
            n_links (int): the number of the links
        """
        robot_model = Arm_3D(robot_name)
        n_links     = robot_model.n_links # number of links (without gripper)     

        return n_links   
    
    def load_gripper_config(self, gripper_name):
        raise NotImplementedError
    
    def load_robot_config_and_joint_limits(self):
        """
        Cache the robot information, mainly referencing Arm_3D library (armtdpy/environments/robots)

        This code set-up the robot configuration, safety specifications.

        TODO: Find TODO notes below
        """
        robot_model    = Arm_3D(self.robot_name)

        # robot configuration specifications
        self.params  = {'n_joints' : self.n_links,
                       'P'        : [self.to_tensor(p) for p in robot_model.P0], 
                       'R'        : [self.to_tensor(r) for r in robot_model.R0]}
        
        self.joint_axes = self.to_tensor(robot_model.joint_axes)
        w = self.to_tensor([[[0,0,0],[0,0,1],[0,-1,0]],
                            [[0,0,-1],[0,0,0],[1,0,0]],
                            [[0,1,0],[-1,0,0],[0,0,0.0]]])
        self.rot_skew_sym = (w@self.joint_axes.T).transpose(0,-1)

        # safety specifications 1: joint angle limit
        self.pos_lim        = robot_model.pos_lim.cpu()
        self.actual_pos_lim = robot_model.pos_lim[robot_model.lim_flag].cpu()
        self.n_pos_lim      = int(robot_model.lim_flag.sum().cpu())
        self.lim_flag       = robot_model.lim_flag.cpu()

        # safety specifications 2: joint velocity limit
        self.vel_lim        = robot_model.vel_lim.cpu()
        
        # TODO: we should find ohter way to retrieve control_freq, output_max, maybe config.safety
        # self.dt_plan  = 1.0/robot_model.env.control_freq
        # vel_lim_robosuite = torch.tensor(robot_model.helper_controller.output_max/self.dt_plan, dtype=self.dtype)


    # ---------------------------------------------- #
    # Abstract fucntions for the ParameterizedPlanner
    # ---------------------------------------------- #
    def __call__(self, obs_dict, goal_dict=None, random_initialization=False):
        assert(self.calibrated, "The planner should be calibrated before using.")

        plan, info = super().__call__(obs_dict, goal_dict, random_initialization)

        return plan, info

    def model(self, t, p0, param, return_gradient=False):
        """
        Get x(t; p0, k), given the initial state p0, trajectory parameter k.

        Args:
            t :    (N,),   time
            p0:    (B, 7), initial planning state, (p0_0, p0_1, ..., p0_6)
            param: (B, 14), initial planning param, (kv_0, ..., kv_6, ka_0, ..., ka_6)
        
        Returns:
            x:  (B, N, 7), desired trajectory
            dx: (B, N, 7), desired velocity

        NOTE: This model should be consistent with the JRS model defined at MATLAB CORA
        """
        assert max(t) <= self.t_f

        if p0.ndim == 1: p0 = p0[None, :]
        if param.ndim == 1: param = param[None, :]

        B = p0.shape[0]
        N = t.shape[0]

        x  = torch.zeros((B, N, self.n_state))
        dx = torch.zeros((B, N, self.n_state))

        # parse parameter
        k_v = param[:, :self.n_links]
        k_a = param[:, self.n_links:]

        t1 = self.time_pieces[0]

        x1, dx1 = traj_uniform_acc(t[t<=t1], p0, k_v, k_a)

        p1 = p0 + k_v*t1 + 0.5*k_a*t1**2
        v1 = k_v + k_a*t1
        a1 = -v1/(self.t_f-t1)
        x2, dx2 = traj_uniform_acc(t[t>t1]-t1,  p1, v1, a1)

        x[:, t<=t1, :], dx[:, t<=t1, :] = (x1, dx1)
        x[:, t >t1, :], dx[:, t >t1, :] = (x2, dx2)

        if return_gradient:
            grad_to_optvar = torch.zeros((B, N, self.n_state, len(self.opt_dim)))
            mask = (t <= t1)
            grad_first_piece  = 0.5*(t[mask])**2
            grad_second_piece = 0.5*t1**2 + t1*(t[~mask]-t1) + 0.5 * (-t1/(self.t_f-t1)) * (t[~mask]-t1)**2
            grad_to_optvar[:, t<=t1, :, :] = grad_first_piece[:, None, None]*torch.eye(self.n_state)
            grad_to_optvar[:, t >t1, :, :] = grad_second_piece[:, None, None]*torch.eye(self.n_state)

            return x, dx, grad_to_optvar
        
        else:
            return x, dx

    def get_pstate_from_obs(self, obs_dict):
        """
        Get the planning state from the observation dictionary
        """
        return self.to_tensor(obs_dict['robot0_joint_pos'])


    def get_pparam_from_obs_and_optvar(self, obs_dict, k_opt):
        """
        Get the trajectory parameter from the observation dictionary
        """
        return self.to_tensor(np.concatenate([obs_dict['robot0_joint_vel'], k_opt]))

    def process_observation_and_goal_for_TO(self, obs_dict, goal_dict=None):
        """
        Process the observation and goal for the trajectory optimization
        Args:
            obs_dict (dict): observation dictionary
            goal_dict (dict): goal dictionary
        
        Returns:
            obs_dict (dict): processed observation dictionary
            goal_dict (dict): processed goal dictionary

        NOTE: Make sure to deepcopy the obs_dict and goal_dict
        """
        # we do not want for obs_dict, goal_dict to change
        obs_dict_processed  = deepcopy(obs_dict)
        goal_dict_processed = deepcopy(goal_dict)
        
        for key in goal_dict.keys():
            if key == "qpos":
                qgoal = self.to_tensor(goal_dict["qpos"])
                goal_dict_processed["qpos"] = qgoal
                self.goal_zonotope = self.get_arm_zonotopes_at_q(q = qgoal)
            
            elif key == "reference_trajectory":
                pass

            elif key == "grasp_pos":
                grasp_pos_goal = self.to_tensor(goal_dict["grasp_pos"])
                goal_dict_processed["grasp_pos"] = grasp_pos_goal
            
            else:
                raise NotImplementedError

        return obs_dict_processed, goal_dict_processed
    
    def _prepare_problem_data(self, obs_dict, goal_dict = None, random_initialization = False):
        """
        Prepare the meta-data for the trajectory optimization and the data for
        constraints, objectives to avoid redundant computations.

        Args:
            obs_dict:
            goal_dict:
            random_initialization: True if the optimization starts at random value
        
        Returns:
            data (dict), key = "meta", "constraint", "objective"
        """
        # construct constraint/objective data
        problem_data = dict(
            constraint = self._prepare_constraint_data(obs_dict),
            objective  = self._prepare_objective_data(obs_dict, goal_dict)
        )

        # also add the meta data
        if random_initialization:
            optvar_0 = self.to_tensor((torch.rand(len(self.opt_dim))-0.5)*2) # initial point
        else:
            optvar_0 = self.to_tensor(torch.zeros(len(self.opt_dim)))

        problem_data["meta"] = {
            "n_optvar"     : len(self.opt_dim),
            "n_constraint" : sum([data["n_con"] for data in problem_data["constraint"].values()]),
            "optvar_0"     : optvar_0,
            "state"        : torch.hstack((self.to_tensor(obs_dict["robot0_joint_pos"]), 
                                           self.to_tensor(obs_dict["robot0_joint_vel"])))
        }

        return problem_data

    # ---------------------------------------------- #
    # Objective
    # ---------------------------------------------- #
    def _prepare_objective_data(self, obs_dict, goal_dict):
        """
        Prepare the target information from the obs_dict and goal_dict

        Returns:
            data (dict):
                [key] objective_name 
                [value] data (dict)
                        [key] argument_to_objective_function
        
        """
        data = {}
        
        for key in goal_dict.keys():
            if key == "qpos":
                data["joint_pos_goal"] = dict(qpos_goal = goal_dict["qpos"])
            
            elif key == "grasp_pos":
                data["grasp_pos_goal"] = dict(grasp_pos_goal = goal_dict["grasp_pos"])

            elif key == "reference_trajectory":
                data["joint_pos_projection"] = dict(reference_trajectory = goal_dict["reference_trajectory"])
            
            else:
                raise NotImplementedError

        return data

    def _compute_objective(self, x, problem_data):
        """
        Prepare the cost function for the trajectory optimization

        x: (n_optvar,), flattened trajectory
        """
        # parse the meta data: used throughout the function
        meta_data = problem_data["meta"]
        qpos      = meta_data["state"][:self.n_links]
        qvel      = meta_data["state"][self.n_links:]
        
        objective_data = problem_data["objective"]
        
        ka = self.to_tensor(x) * self.FRS_info["delta_k"][self.opt_dim]

        Obj = {}
        Grad = {}
        
        # qpos goal (verified)
        if "joint_pos_goal" in objective_data.keys():
            Obj["joint_pos_goal"], Grad["joint_pos_goal"] = self._compute_objective_joint_pos_goal(
                                                            ka    = ka, 
                                                            qpos  = qpos, 
                                                            qvel  = qvel, 
                                                            **objective_data["joint_pos_goal"])
        
        # joint_pos_projection (verified)
        if "joint_pos_projection" in objective_data.keys():
            Obj["joint_pos_projection"], Grad["joint_pos_projection"] = self._compute_objective_joint_pos_projection(
                                                                    ka = ka,
                                                                    qpos = qpos,
                                                                    qvel = qvel,
                                                                    **objective_data["joint_pos_projection"])
        
        # # grasping pos goal
        # if "grasp_pos_goal" in objective_data.keys():
        #     Obj["grasp_pos_goal"], Grad["grasp_pos_goal"] = self._compute_objective_grasp_pos_goal(
        #                                                             ka = ka,
        #                                                             qpos = qpos,
        #                                                             qvel = qvel,
        #                                                             **objective_data["grasp_pos_goal"])
            
        # weighting
        total_cost = self.to_tensor(0)
        total_grad = self.to_tensor(torch.zeros_like(ka))
        for key in Obj.keys():
            total_cost += Obj[key]  * self.weight_dict[key]
            total_grad += Grad[key] * self.weight_dict[key]
        total_grad = total_grad * self.FRS_info["delta_k"][self.opt_dim]
        
        # detaching
        Obj = total_cost.cpu().numpy()
        Grad = total_grad.cpu().numpy()
        
        return Obj, Grad
    
    # def test_func(self, ka, qpos, qvel, objective_data):
    #     o, _ =  self._compute_objective_joint_pos_projection(
    #                                                             ka = ka,
    #                                                             qpos = qpos,
    #                                                             qvel = qvel,
    #                                                             **objective_data["joint_pos_projection"])
    #     return o
    
    # jac = MathUtils.compute_jacobian(function=test_func, args = dict(self=self, ka=ka, qpos=qpos, qvel=qvel, objective_data = objective_data), var_name = 'ka')
    # _, g =  self._compute_objective_joint_pos_projection(
    #                                                     ka = ka,
    #                                                     qpos = qpos,
    #                                                     qvel = qvel,
    #                                                     **objective_data["joint_pos_projection"])
    
    def _compute_objective_joint_pos_goal(self, ka, qpos, qvel, qpos_goal):
        """
        c(ka) = ||q(t_f) - qgoal||

        Args:
            ka:   (n_optvar,) joint acceleration - unnormalized
            qpos: (n_link,) current joint position
            qvel: (n_link,) current joint velocity
            data: {qgoal}: goal joint position
        
        Returns:
            Obj: c(ka)
            Grad: the gradient of c(ka) w.r.t. ka
        """
        # joint-angle trajectory design
        t_last = self.to_tensor([self.t_f])
        param  = torch.hstack([qvel, ka])
        qpos_last, _, grad_qpos_last_to_ka = self.model(t_last, qpos, param, return_gradient = True)
        
        # eliminating batch-dim and horizon-dim
        qpos_last            = qpos_last[0][0]
        grad_qpos_last_to_ka = grad_qpos_last_to_ka[0][0]

        # the gradient of wrap joint should be identity mapping
        dq = self.wrap_cont_joint_to_pi(qpos_last - qpos_goal).squeeze()
        grad_dq_to_ka = grad_qpos_last_to_ka
        
        # cost function
        Obj = torch.sum(dq ** 2)
        Grad_to_dq = 2*dq

        # chain rule
        Grad = grad_dq_to_ka @ Grad_to_dq

        return Obj, Grad
    
    def _compute_objective_joint_pos_projection(self, ka, qpos, qvel, reference_trajectory):
        """
        c(ka) = \sum_{t=0}^{T}||q(t) - qd(t)||

        Args:
            ka:   (n_optvar,) joint acceleration - unnormalized
            qpos: (n_link,) current joint position
            qvel: (n_link,) current joint velocity
            reference_trajectory: (ReferenceTrajectory), the nominal trajectory to project
        
        Returns:
            Obj: c(ka)
            Grad: the gradient of c(ka) w.r.t. ka
        """
        assert(isinstance(reference_trajectory, ReferenceTrajectory))

        t_eval = reference_trajectory.t_des[reference_trajectory.t_des <= self.t_f]
        q_eval = reference_trajectory.x_des[reference_trajectory.t_des <= self.t_f]
        
        param  = torch.hstack([qvel, ka])
        qs, _, grad_qs_to_ka = self.model(t_eval, qpos, param, return_gradient = True)

        # eliminate batch dimension
        qs            = qs[0]                     # (N_T, n_joint)
        grad_qs_to_ka = grad_qs_to_ka[0]          # (N_T, n_joint, n_joint)

        dqs = self.wrap_cont_joint_to_pi(qs - q_eval)
        grad_dqs_to_ka = grad_qs_to_ka            # (N_T, n_joint, n_joint)

        # TODO: start here
        Obj = torch.mean(torch.sum(dqs**2, axis=0))
        Grad_to_dq = 2 * dqs / dqs.shape[1]
        Grad = torch.einsum('ijk,ij->k', grad_dqs_to_ka, Grad_to_dq)  # shape (n_joint)
        
        # Grad_to_dq = 2 * dqs / dqs.shape[0]  # shape (N_T, n_joint)
        # Grad_to_ka = torch.einsum('ij,ijk->k', Grad_to_dq, grad_dqs_to_ka)  # shape (n_joint)
        # Grad = Grad_to_ka

        # Obj.backward()
        
        # Grad_to_dq = dqs.grad.squeeze(0)                                    # (1, N_T, n_joint)
        # dqs.detach()
        
        # # Apply the chain rule
        # Grad_to_ka = torch.einsum('tij,tj->ij', grad_dqs_to_ka, Grad_to_dq)  # (N_T, n_joint, n_joint) @ (N_T, n_joint) -> (n_joint, n_joint)

        # # Sum over the time dimension
        # Grad = torch.sum(Grad_to_ka, dim=0)  # (n_joint, n_joint) -> (n_joint,)

        return Obj, Grad
    
    def _compute_objective_grasp_pos_goal(self, ka, qpos, qvel, grasp_pos_goal):
        """
        c(ka) = ||p_eef(q(t; k)) - p_eef_des||_2^2

        Args:
            ka:   (n_optvar,) joint accleration - unnormalized
            qpos: (n_joint, ) current joint position
            qvel: (n_joint, ) current joint velocity
            grasp_pos_goal (torch.tensor)
        """
        # q(tf; ka)
        t_last = self.to_tensor([self.t_f])
        param  = torch.hstack([qvel, ka])
        qpos_last, _, grad_qpos_last_to_ka = self.model(t_last, qpos, param, return_gradient = True)
        
        # eliminate batch_size, time_horizon dimension, which is first two dimension
        qpos_last = qpos_last[0, 0]
        grad_qpos_last_to_ka = grad_qpos_last_to_ka[0, 0 ]

        # p_grasp(q(tf; ka))
        grasp_pos, grad_grasp_pos_to_qpos = self.get_grasping_pos_at_q(qpos_last, return_gradient=True)

        # p_grasp(q(tf; ka)) - p_grasp_goal
        delta_grasp_pos           = grasp_pos - grasp_pos_goal
        grad_delta_grasp_pos_to_q = grad_grasp_pos_to_qpos

        # cost = ||p_grasp(tf; ka) - p_grasp_goal||_2^2
        cost                         = torch.sum((delta_grasp_pos) ** 2)
        grad_cost_to_delta_grasp_pos = 2*delta_grasp_pos

        # chain rule
        grad_cost_to_ka = grad_qpos_last_to_ka @ grad_delta_grasp_pos_to_q @ grad_cost_to_delta_grasp_pos

        return cost, grad_cost_to_ka

    # ---------------------------------------------- #
    # Constraints
    # ---------------------------------------------- #
    def _prepare_constraint_data(self, obs_dict):
        """
        Per planning horizon, prepare the data used for computing constraints, jacobian
        that is not influenced by optvar, hence could be pre-computed.

        Args:
            obs_dict
        
        Returns:
            data (dict (key = "constraint_name", value = data))
        """
        data = {}

        qpos             = self.to_tensor(obs_dict["robot0_joint_pos"])
        qvel             = self.to_tensor(obs_dict["robot0_joint_vel"])
        qpos_gripper     = self.to_tensor(obs_dict["robot0_gripper_qpos"])
        
        # Constraint 1: arm collision constraint
        data["arm_collision"] = self._prepare_constraint_arm_collision(
                                            qpos                = qpos, 
                                            qvel                = qvel, 
                                            obstacles           = obs_dict["zonotope"]["obstacle"])
        
        # Constraint 2: joint position/velocity constraint
        data["joint"]           = self._prepare_constraint_joint()

        # Constraint 3: collision b/w robot gripper - {non-active object, arena, mount}
        # TODO: .polytope operation takes too long.
        if self.gripper_name is not None:
            obstacles = []
            obstacles.extend(obs_dict["zonotope"]["static_obs"])        # mount, arena
            obstacles.extend(obs_dict["zonotope"]["non_active_object"]) # object not to grasp
            data["gripper_collision"] = self._prepare_constraint_gripper_collision(
                                                qpos                = qpos, 
                                                qvel                = qvel, 
                                                qpos_gripper        = qpos_gripper,
                                                obstacles           = obstacles)
        
        # Constraint 4: collision b/w robot gripper outside - {active objet}

        # Constraint 5: self-collision

        return data
    
    def _prepare_constraint_joint(self):
        """
        Prepare the constraints data for joint velocity and joint position limit
        """
        data = dict(n_con = 2*(3*self.n_pos_lim + self.n_links))

        return data
    
    def _prepare_constraint_arm_collision(self, qpos, qvel, obstacles):
        """
        Prepare the constraints data for collision-check of robot arms (i.e., w/o gripper) and 
        objects / mount / arena.

        Args:
            qpos: (n_link, 1), joint position of the robot
            qvel: (n_link, 1), joint velocity of the robot
            obstacles: list <zonotope>: the list of zonotope representation for the objects, arena, mounts
            use_gripper: (bool), use gripper for collision check

        Returns:
            data: dict,
                    (A, b) Ax <= b H-polytopee representation
                    FO_links: (n_link, 1) list of polyzonotope representation of the all possible volumes of each link
        """
        # robot forward occupancy
        FO_links = self.FRS_all_traj_params(qpos, qvel, return_transforms = False)

        # TODO: remove this later
        self.arm_FRS_links = FO_links

        # obstacle zonotope
        n_obs     = len(obstacles)

        # constraint data
        # the robot link should not intersect the obstacles for all time horizon
        n_interval = self.n_timestep - 1

        # TODO: polytope is the bottleneck, this could be 
        A = np.zeros((self.n_links,n_obs),dtype=object)
        b = np.zeros((self.n_links,n_obs),dtype=object)
        for (j, FO_link) in enumerate(FO_links):
            for (i, obstacle) in enumerate(obstacles):
                obs_Z    = einops.repeat(obstacle.Z, 'm n -> repeat m n', repeat=n_interval)
                A_o, b_o = batchZonotope(torch.cat((obs_Z, FO_link.Grest),-2)).polytope(self.combs) # A: n_timesteps,*,dimension  
                A[j, i]  = A_o.cpu()
                b[j, i]  = b_o.cpu()
        
        data = dict(
            A_obs    = A,
            b_obs    = b,
            FO_links = FO_links,
            n_con    = n_obs * self.n_links * n_interval
        )

        return data
    
    def _prepare_constraint_gripper_collision(self, qpos, qvel, qpos_gripper, obstacles):
        """
        Prepare the constraints data for collision-check of the gripper and non_activeobjects / mount / arena.

        Args:
            qpos: (n_link, 1), joint position of the robot
            qvel: (n_link, 1), joint velocity of the robot
            obstacles: list <zonotope>: the list of zonotope representation for the non-active objects, arena, mounts
            closed_gripper: (bool), whether the gripper is closed or not

        Returns:
            data: dict,
                    (A, b) Ax <= b H-polytopee representation
                    FO_links: (n_link, 1) list of polyzonotope representation of the all possible volumes of each link
        
        TODO
        """
        # robot forward occupancy
        _, R_transform, P_transform = self.FRS_all_traj_params(qpos, qvel, return_transforms=True)

        R_last_link_to_gripper = self.to_tensor(self.T_last_link_to_gripper_base[:3, :3])
        P_last_link_to_gripper = self.to_tensor(self.T_last_link_to_gripper_base[:3,  3])
        
        # primitive gripper stl (forward kinematics inside)
        gripper_links_stl      = self.get_gripper_zonotopes_at_q(qpos_gripper, 
                                                                 T_frame_to_base = self.to_tensor(torch.eye(4)))
        gripper_links_stl      = [link.to_polyZonotope() for link in gripper_links_stl]

        # static transform (that does not depend on the joint angle) to the gripper-attachment site
        gripper_links          = [R_last_link_to_gripper@link + P_last_link_to_gripper for link in gripper_links_stl]

        # get the forward occupancy of the grippers
        FO_gripper_links       = [R_transform[-1]@link + P_transform[-1] for link in gripper_links]
        FO_gripper_links       = [link.reduce_indep(self.zono_order) for link in FO_gripper_links]

        # TODO: remove this later
        self.gripper_FRS_links = FO_gripper_links

        # obstacle zonotope
        n_obs           = len(obstacles)
        n_gripper_links = len(FO_gripper_links)

        # constraint data
        # the robot link should not intersect the obstacles for all time horizon
        n_interval = self.n_timestep - 1

        A = np.zeros((n_gripper_links,n_obs), dtype=object)
        b = np.zeros((n_gripper_links,n_obs), dtype=object)
        for (j, FO_gripper_link) in enumerate(FO_gripper_links):
            for (i, obstacle) in enumerate(obstacles):
                obs_Z    = einops.repeat(obstacle.Z, 'm n -> repeat m n', repeat=n_interval)
                A_o, b_o = batchZonotope(torch.cat((obs_Z, FO_gripper_link.Grest),-2)).polytope(self.combs) # A: n_timesteps,*,dimension  
                A[j, i]  = A_o.cpu()
                b[j, i]  = b_o.cpu()
        
        data = dict(
            A_obs            = A,
            b_obs            = b,
            FO_gripper_links = FO_gripper_links,
            n_con            = n_obs * n_gripper_links * n_interval
        )

        return data

    def _compute_constraints(self, x, problem_data):
        """
        Prepare the cost function for the trajectory optimization

        x: (n_optvar,), normalized between [-1, 1]
        """
        n_interval = self.n_timestep - 1

        # parse the meta data: used throughout the function
        meta_data = problem_data["meta"]
        qpos      = meta_data["state"][:self.n_links]
        qvel      = meta_data["state"][self.n_links:]

        cdata = problem_data["constraint"]
        
        # Constraint - Initialization
        ka   = einops.repeat(self.to_tensor(x), 'n -> repeat n', repeat=n_interval) * self.FRS_info["delta_k"][self.opt_dim]
        Cons_list = []
        Jac_list = []

        # C1 - joint constraint
        Cons_joint, Jac_joint_to_ka = self._compute_joint_constraints(ka=ka[0], qpos=qpos, qvel=qvel)
        Cons_list.append(Cons_joint); Jac_list.append(Jac_joint_to_ka)

        # C2 - robot arm collision constraint
        Cons_rc, Jac_rc_to_ka = self._compute_arm_collision_constraints(ka = ka, data = cdata["arm_collision"])
        Cons_list.append(Cons_rc); Jac_list.append(Jac_rc_to_ka)
        
        # # C3 - gripper collision constraint
        if self.gripper_name is not None:
            Cons_gc, Jac_gc_to_ka = self._compute_gripper_collision_constraints(ka = ka, data = cdata["gripper_collision"])
            Cons_list.append(Cons_gc); Jac_list.append(Jac_gc_to_ka)
        
        # Vectorizing
        Cons      = torch.hstack(Cons_list).cpu().numpy()
        Jac_to_ka = torch.vstack(Jac_list)
        Jac_to_x  = Jac_to_ka * self.FRS_info["delta_k"][self.opt_dim].cpu().numpy()

        return Cons, Jac_to_x
    
    def _compute_gripper_collision_constraints(self, ka, data):
        """
        Gripper collision constraint
        """
        """
        Robot arm collision constraint
        """
        T = self.n_timestep - 1
        M = data["n_con"] # number of constraints
        N = len(self.opt_dim)
        
        FO_links = data["FO_gripper_links"]
        A_obs    = data["A_obs"]
        b_obs    = data["b_obs"]
        n_links  = len(FO_links)
        n_obs    = b_obs.shape[1]

        Cons     = torch.zeros(M, dtype = self.dtype)
        Jac      = torch.zeros(M, N, dtype = self.dtype)

        beta = ka/self.FRS_info["delta_k"][self.opt_dim]
        for (l, FO_link) in enumerate(FO_links):
            c_k      = FO_link.center_slice_all_dep(beta)
            grad_c_k = FO_link.grad_center_slice_all_dep(beta)
            for o in range(n_obs):
                # constraint: max(Ax - b) <= 0
                h_obs = (A_obs[l][o]@c_k.unsqueeze(-1)).squeeze(-1) - b_obs[l][o]
                h_obs = h_obs.nan_to_num(-torch.inf)
                cons, ind = torch.max(h_obs, -1) # shape: n_timsteps, SAFE if >=1e-6 

                # gradient of the constraint
                A_max     = A_obs[l][o].gather(-2, ind.reshape(T, 1, 1).repeat(1, 1, self.dimension)) 
                grad_cons = (A_max@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6

                Cons[(l + n_links*o)*T:(l + n_links*o + 1)*T] = - cons
                Jac[(l + n_links*o)*T:(l + n_links*o + 1)*T]  = - grad_cons
        
        return Cons, Jac

    def _compute_arm_collision_constraints(self, ka, data):
        """
        Robot arm collision constraint
        """
        T = self.n_timestep - 1
        N = self.n_links
        M = data["n_con"] # number of constraints
        
        FO_links = data["FO_links"]
        A_obs    = data["A_obs"]
        b_obs    = data["b_obs"]
        n_obs    = b_obs.shape[1]

        Cons     = torch.zeros(M, dtype = self.dtype)
        Jac      = torch.zeros(M, N, dtype = self.dtype)

        beta = ka/self.FRS_info["delta_k"][self.opt_dim]
        for (l, FO_link) in enumerate(FO_links):
            c_k      = FO_link.center_slice_all_dep(beta)
            grad_c_k = FO_link.grad_center_slice_all_dep(beta)
            for o in range(n_obs):
                # constraint: max(Ax - b) <= 0
                h_obs = (A_obs[l][o]@c_k.unsqueeze(-1)).squeeze(-1) - b_obs[l][o]
                h_obs = h_obs.nan_to_num(-torch.inf)
                cons, ind = torch.max(h_obs, -1) # shape: n_timsteps, SAFE if >=1e-6 

                # gradient of the constraint
                A_max     = A_obs[l][o].gather(-2, ind.reshape(T, 1, 1).repeat(1, 1, self.dimension)) 
                grad_cons = (A_max@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6

                Cons[(l + N*o)*T:(l + N*o + 1)*T] = - cons
                Jac[(l + N*o)*T:(l + N*o + 1)*T]  = - grad_cons
        
        return Cons, Jac

    def _compute_joint_constraints(self, ka, qpos, qvel):
        """
        Compute the joint velocity / position constraints

        qmin <= q(t; ka) <= qmax
        dqmin <= dq(t; ka) <= dqmax

        Returns:
            Cons: (2*3*self.n_pos_lim + 2*self.n_link)
            Jac : (", self.n_link)
        """
        # timing
        T_PEAK         = self.time_pieces[0]
        T_FULL         = self.t_f
        T_PEAK_OPTIMUM = -qvel/ka # time to optimum of first half traj.
        
        # qpos(T_PEAK_OPTIMUM)
        qpos_peak_optimum      = (T_PEAK_OPTIMUM > 0)*(T_PEAK_OPTIMUM < T_PEAK)*\
                                (qpos + qvel*T_PEAK_OPTIMUM + 0.5*ka*T_PEAK_OPTIMUM**2).nan_to_num()
        
        grad_qpos_peak_optimum = torch.diag((T_PEAK_OPTIMUM>0)*(T_PEAK_OPTIMUM<T_PEAK)*\
                                            (0.5*qvel**2/(ka**2)).nan_to_num())

        # qpos(T_PEAK)
        qpos_peak      = qpos + qvel * T_PEAK + 0.5 * ka * T_PEAK**2
        grad_qpos_peak = 0.5 * T_PEAK**2 * torch.eye(self.n_links, dtype=self.dtype)
        
        # qvel(T_PEAK)
        qvel_peak      = qvel + ka * T_PEAK
        grad_qvel_peak = T_PEAK * torch.eye(self.n_links, dtype=self.dtype)

        # qpos(T_FULL)
        braking_accel   = (0 - qvel_peak)/(T_FULL - T_PEAK)
        qpos_brake      = qpos_peak + qvel_peak*(T_FULL - T_PEAK) + 0.5*braking_accel*(T_FULL-T_PEAK)**2
        grad_qpos_brake = 0.5 * T_PEAK * T_FULL * torch.eye(self.n_links, dtype=self.dtype)

        # joint position constraints
        qpos_possible_max_min = torch.vstack((qpos_peak_optimum, qpos_peak, qpos_brake))[:,self.lim_flag] 
        qpos_ub               = (qpos_possible_max_min - self.actual_pos_lim[:,0]).flatten()
        qpos_lb               = (self.actual_pos_lim[:,1] - qpos_possible_max_min).flatten()
        grad_qpos_ub          = torch.vstack((grad_qpos_peak_optimum[self.lim_flag],
                                              grad_qpos_peak[self.lim_flag],
                                              grad_qpos_brake[self.lim_flag]))
        grad_qpos_lb          = - grad_qpos_ub

        # joint velocity constraints
        qvel_ub      =  qvel_peak - self.vel_lim
        qvel_lb      = -qvel_peak - self.vel_lim
        grad_qvel_ub = grad_qvel_peak
        grad_qvel_lb = -grad_qvel_peak

        Cons = torch.hstack((qvel_ub, qvel_lb, qpos_ub, qpos_lb))
        Jac  = torch.vstack((grad_qvel_ub, grad_qvel_lb, grad_qpos_ub, grad_qpos_lb))

        return Cons, Jac

    def collision_check(self, plan, obs):
        assert isinstance(plan, ReferenceTrajectory), "The plan should be the instance of ReferenceTrajectory"
        assert "zonotope" in obs.keys() and "obstacle" in obs["zonotope"].keys()

        qs = plan.x_des
        R_q = self.rot(qs)

        obs_zonos           = obs["zonotope"]["obstacle"]
        
        n_obs = len(obs_zonos)

        R, P = TransformUtils.pose_to_rot_pos(self.T_world_to_arm_base)
        R = self.to_tensor(R)
        P = self.to_tensor(P)
        
        if len(R_q.shape) == 4:
            time_steps = len(R_q)
            for j in range(self.n_links):
                P = R@self.params["P"][j] + P
                R = R@self.params["R"][j]@R_q[:,j]
                link = batchZonotope(self._link_polyzonos_stl[j].Z.unsqueeze(0).repeat(time_steps,1,1))
                link = R@link+P
                for o in range(n_obs):
                    buff = link - obs_zonos[o]
                    _,b = buff.polytope(self.combs)
                    unsafe = b.min(dim=-1)[0]>1e-6
                    if any(unsafe):
                        self.qpos_collision = qs[unsafe]
                        return True

        else:
            time_steps = 1
            for j in range(self.n_links):
                P = R@self.params["P"][j] + P
                R = R@self.params["R"][j]@R_q[j]
                link = R@self._link_zonos_stl[j]+P
                for o in range(n_obs):
                    buff = link - obs_zonos[o]
                    _,b = buff.polytope(self.combs)
                    if min(b) > 1e-6:
                        self.qpos_collision = qs
                        return True
        
        return False
    
    # ---------------------------------------------- #
    # Action configuration
    # ---------------------------------------------- #
    def get_action(self, obs_dict, goal_dict = None):
        """
        Get action that can be sent to the environment
        """
        # if there is no plan, generate the plan
        # TODO: change `self.time_pieces[0]`
        if self.active_plan is None or self.t_cur > self.time_pieces[0]:
            _, _ = self.__call__(obs_dict, goal_dict, random_initialization=False)
            self.t_cur = 0

        self.t_cur += self.action_config["dt"]

        return self.get_action_from_plan_at_t(self.t_cur, self.active_plan, obs_dict)
        
    def get_action_from_plan_at_t(self, t_des, plan, obs_dict):
        """
        Plan to the action representation

        The action configuration is the normalized delta joint angle.

        Args:
            t_des (float): the query time to retrieve the plan, desired joint angle
            plan (ReferenceTrajectory): the plan to track
            obs_dict (dict): this dictionary includes the current state information
        
        Returns:
            action (float): the normalized delta joint angle
        """
        qpos = np.array(self.get_pstate_from_obs(obs_dict = obs_dict))
        qdes = plan.x_des[min(math.ceil(t_des/self.dt), len(plan) - 1)]

        action = (qdes - qpos).cpu().numpy()
        action  = scale_array_from_A_to_B(action, 
                                          A=[self.action_config["output_min"], self.action_config["output_max"]], 
                                          B=[self.action_config["input_min"] , self.action_config["input_max"]]) # (T, D) array
        action  = np.clip(action, self.action_config["input_min"], self.action_config["input_max"])

        return action

    # ---------------------------------------------- #
    # Helper Functions
    # ---------------------------------------------- #
    def wrap_cont_joint_to_pi(self, phases):
        if len(phases.shape) == 1:
            phases = phases.unsqueeze(0)
        phases_new = torch.clone(phases)
        phases_new[:,~self.lim_flag] = (phases[:,~self.lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi

        return phases_new

    def rot(self, q):
        """
        Rodrigues' formula

        Args:
            q: the joint angle
        
        Returns:
            R: rotation matrix
        """
        q = q.reshape(q.shape+(1,1))

        R = self.to_tensor(torch.eye(3)) + \
            torch.sin(q) * self.rot_skew_sym + \
            (1-torch.cos(q))*self.rot_skew_sym@self.rot_skew_sym

        return R
    
    def get_arm_zonotopes_at_q(self, q):
        """
        Return the forward occupancy of the robot arms with the zonotopes.
        It does by doing the forward kinematics of the robot.

        Args
            qpos: (n_links,) torch tensor

        Output
            arm_zonos: list of zonotopes
        """
        R_qi = self.rot(q)
        arm_zonos = []
        for i in range(self.n_links):
            Pi = Ri@self.params["P"][i] + Pi
            Ri = Ri@self.params["R"][i]@R_qi[i]
            arm_zono = Ri@self._link_zonos_stl[i] + Pi
            arm_zonos.append(arm_zono)
        
        return arm_zonos

    def get_gripper_zonotopes_at_q(self, q, T_frame_to_base):
        """
        Return the forward occupancy of the gripper with the zonotopes

        It does by doing the forward kinematics of the gripper
        """
        raise NotImplementedError
    
    def get_arm_eef_pos_at_q(self, q):
        """
        Return the position of the robot end-effector in world frame at given joint angle
        It does by doing the forward kinematics of the robot.

        Args
            qpos: (n_links,) torch tensor

        Output
            P: position of the end-effector relative to 
        """
        R_qi = self.rot(q)
        for i in range(self.n_links):
            Pi = Ri@self.params["P"][i] + Pi
            Ri = Ri@self.params["R"][i]@R_qi[i]
        
        return Pi.cpu().numpy()
    
    def get_arm_link_pose_at_q(self, q, return_gradient = False):
        raise NotImplementedError
    
    def get_grasping_pos_at_q(self, q, return_gradient = False):
        """
        Get the grasping position relative to the world

        Args:
            q (numpy array) (n_joint,) arm joint angle
        
        Returns:
            pos_world_to_grasp (numpy array) (3,) the grasping position relative to the world

        TODO: unify the naming convention as base, last_link, gripper_base, grasping_site
        """
        raise NotImplementedError
    
    def get_arm_eef_pos_from_plan(self, plan):
        assert isinstance(plan, ReferenceTrajectory)

        eef_pos = [self.get_arm_eef_pos_at_q(plan.x_des[i])
                   for i in range(len(plan))]
        eef_pos = np.stack(eef_pos, axis=0)

        return eef_pos
    
    def get_grasping_pos_from_plan(self, plan):
        assert isinstance(plan, ReferenceTrajectory)

        grasping_pos = [self.get_grasping_pos_at_q(plan.x_des[i])[0]
                        for i in range(len(plan))]
        grasping_pos = np.stack(grasping_pos, axis=0)

        return grasping_pos

    def get_forward_occupancy_from_plan(self, plan, only_end_effector = False, vis_gripper = False, gripper_init_qpos = None):
        """
        Get series of forward occupancy zonotopes from the given plan

        Args:
            plan (ReferenceTrajectory), the plan that contains the joint angle, joint velocity
            only_end_effector (bool), get forward occupancy of only end effector

        Returns:
            FO_link (zonotopes), the zonotope list of plan length
        """
        raise NotImplementedError
    
    def FRS_all_traj_params(self, qpos, qvel, gripper = False, return_transforms=False):
        """
        Get the forward reachable set (FRS) of the robot arm links under
        all possible trajectory parameters

        Args
            qpos (torch.tensor): (n_joint, 1) joint angle
            qvel (torch.tensor): (n_joint, 1) joint velocity

        Returns
            TODO
        """
        _, R_trig        = process_batch_JRS_trig(jrs_tensor = self.JRS_tensor, 
                                                  q_0        = qpos, 
                                                  qd_0       = qvel, 
                                                  joint_axes = self.joint_axes)
         
        FRS_links, R, P  = forward_occupancy(rotatos         = R_trig, 
                                             link_zonos      = self._link_polyzonos_stl, 
                                             robot_params    = self.params, 
                                             T_world_to_base = self.T_world_to_arm_base)

        if return_transforms:
            return FRS_links, R, P
        else:
            return FRS_links
    
    def get_FRS_from_obs_and_optvar(self, obs_dict, k_opt):
        """
        Get the FRS given the optimized trajectory parameter

        This is a getter function, and should not be used for any computation
        """
        FRS_all           = []

        qpos                = self.to_tensor(obs_dict["robot0_joint_pos"])
        qpos_gripper        = self.to_tensor(obs_dict["robot0_gripper_qpos"])
        qvel                = self.to_tensor(obs_dict["robot0_joint_vel"])
        
        arm_FRS_all, R, P = self.FRS_all_traj_params(qpos, qvel, return_transforms=True)
        FRS_all.extend(arm_FRS_all)

        R_last_link_to_gripper = self.to_tensor(self.T_last_link_to_gripper_base[:3, :3])
        P_last_link_to_gripper = self.to_tensor(self.T_last_link_to_gripper_base[:3,  3])
        gripper_links     = self.get_gripper_zonotopes_at_q(qpos_gripper, 
                                                            T_frame_to_base = self.to_tensor(torch.eye(4)))
        gripper_links     = [link.to_polyZonotope() for link in gripper_links]
        gripper_links     = [R_last_link_to_gripper@link + P_last_link_to_gripper for link in gripper_links]
        gripper_FRS_all   = [R[-1]@link + P[-1] for link in gripper_links]
        gripper_FRS_all   = [link.reduce_indep(self.zono_order) for link in gripper_FRS_all]
        FRS_all.extend(gripper_FRS_all)

        beta              = k_opt/self.FRS_info["delta_k"][self.opt_dim]
        beta              = einops.repeat(beta, 'n -> repeat n', repeat=self.n_timestep-1)
        FRS               = [FRS.slice_all_dep(beta) for FRS in FRS_all]

        return FRS