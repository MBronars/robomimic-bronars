import os
import math
from copy import deepcopy

import torch
import numpy as np
import einops

import robosuite

from safediffusion.armtdpy.environments.arm_3d import Arm_3D
from safediffusion.armtdpy.planning.armtd_3d import wrap_to_pi
from safediffusion.armtdpy.reachability.forward_occupancy.FO import forward_occupancy
from safediffusion.armtdpy.reachability.joint_reachable_set.load_jrs_trig import preload_batch_JRS_trig
from safediffusion.armtdpy.reachability.joint_reachable_set.process_jrs_trig import process_batch_JRS_trig

from safediffusion.algo.helper import traj_uniform_acc, ReferenceTrajectory
from safediffusion.algo.planner_base import ParameterizedPlanner
from safediffusion.utils.npy_utils import scale_array_from_A_to_B
import safediffusion.utils.robot_utils as RobotUtils

# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope, batchZonotope, polynomial_zonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope, batchZonotope, polyZonotope

class ArmtdPlanner(ParameterizedPlanner):
    """
    Armtd-style of planner for the manipulator robot
    """
    def __init__(self, 
                 action_config, 
                 robot_name   = "Kinova3",
                 **kwargs):
        
        # prepare the useful variables for robot configuration, joint limits, reachable sets
        self.dimension  = 3

        # load robot configuration
        self.robot_name = robot_name
        self.load_robot_n_links(self.robot_name)

        # Trajectory design
        state_dict = {f"q{joint_id}": joint_id for joint_id in range(self.n_links)}
        param_dict = {f"kv{joint_id}": joint_id for joint_id in range(self.n_links)}
        param_dict.update({f"ka{joint_id}": joint_id + self.n_links for joint_id in range(self.n_links)})

        # Timing
        dt         = 0.01 # planning time interval
        t_f        = 1.0  # planning horizon

        ParameterizedPlanner.__init__(self, state_dict, param_dict, dt, t_f, **kwargs)

        # robot configuration
        self.load_robot_config_and_joint_limits(self.robot_name)
        self.load_robot_link_zonotopes(self.robot_name)
        self.load_joint_reachable_set()

        # load zonotope configuration
        self.zono_order = kwargs["zonotope"]["order"]
        self.max_combs  = kwargs["zonotope"]["max_comb"]
        self.combs      = self.generate_combinations_upto(self.max_combs)

        # piecewise trajectory design
        self.time_pieces = [0.5]

        # trajectory optimization
        self.opt_dim     = range(self.n_links, 2*self.n_links)
        self.weight_dict = dict(qpos_goal = 0.0, qpos_projection = 0.0)

        # Environment configuration
        self.t_cur = 0
        self.action_config = action_config

    # ---------------------------------------------- #
    # Loading the robot configurations
    # ---------------------------------------------- #

    def load_joint_reachable_set(self):
        """
        Load offline-computed joint reachable set
        """
        # load pre-computed joint reachable set
        self.JRS_tensor = self.to_tensor(preload_batch_JRS_trig())
        
        # load the parameter bound used for computing JRS
        delta_kv        = self.vel_lim
        delta_ka        = torch.pi/24 * torch.ones(self.n_links,)
        self.FRS_info   = {"delta_k": torch.hstack((delta_kv, delta_ka))}

    
    def load_robot_link_zonotopes(self, robot_name):
        """
        Load the zonotope representations of the given `robot_name`

        TODO: unify link_zono and link_polyzono later
        """
        # load the robot arm zonotopes
        robot_model              = Arm_3D(robot_name)
        self._link_zonos_stl     = robot_model._link_zonos
        self._link_polyzonos_stl = robot_model.link_zonos
        
    
    def load_robot_n_links(self, robot_name):
        """
        Load the number of the links given the robot name
        """
        robot_model    = Arm_3D(robot_name)
        self.n_links = robot_model.n_links # number of links (without gripper)        
    
    def load_robot_config_and_joint_limits(self, robot_name):
        """
        Cache the robot information, mainly referencing Arm_3D library (armtdpy/environments/robots)

        This code set-up the robot configuration, safety specifications.

        TODO: Find TODO notes below
        """
        robot_model    = Arm_3D(robot_name)

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
        
        # Objective 1: qpos_goal
        if "qpos_goal" in goal_dict.keys():
            qgoal = self.to_tensor(goal_dict["qpos_goal"]["qgoal"])
            goal_dict_processed["qpos_goal"]["qgoal"] = qgoal
            self.goal_zonotope = self.get_arm_zonotopes_at_q(q              = qgoal,
                                                            T_frame_to_base = obs_dict["T_world_to_base"])

        # Objective 2: qpos_projection
        if "qpos_projection" in goal_dict.keys():
            pass

        return obs_dict_processed, goal_dict_processed
    
    def _prepare_problem_data(self, obs_dict, goal_dict = None, random_initialization = True):
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
            data: (dict: key = objective_name, value = data)
        """
        data = {}

        # prepare data for the eef_goal objective
        active_obj_id   = 1
        active_obj_pose = obs_dict["object"][7*active_obj_id:7*(active_obj_id+1)]
        data["eef_goal"] = {}
        data["eef_goal"]["pose"] = active_obj_pose
        
        # prepare data for the qpos_goal
        if "qpos_goal" in goal_dict.keys():
            data["qpos_goal"] = {}
            data["qpos_goal"]["qgoal"] = goal_dict["qpos_goal"]["qgoal"]
        
        if "qpos_projection" in goal_dict.keys():
            data["qpos_projection"] = {}
            data["qpos_projection"]["plan"] = goal_dict["qpos_projection"]["plan"]

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
        
        # qpos goal
        if "qpos_goal" in objective_data.keys():
            Obj["qpos_goal"], Grad["qpos_goal"] = self._compute_objective_qpos_goal(
                                                            ka    = ka, 
                                                            qpos  = qpos, 
                                                            qvel  = qvel, 
                                                            **objective_data["qpos_goal"])
        # qpos_projection
        if "qpos_projection" in objective_data.keys():
            Obj["qpos_projection"], Grad["qpos_projection"] = self._compute_objective_qpos_projection(
                                                                    ka = ka,
                                                                    qpos = qpos,
                                                                    qvel = qvel,
                                                                    **objective_data["qpos_projection"])

        # eef goal: TODO
        
        
        # weighting
        total_cost = self.to_tensor(0)
        total_grad = self.to_tensor(torch.zeros_like(ka))
        for key in Obj.keys():
            total_cost += Obj[key] * self.weight_dict[key]
            total_grad += Grad[key] *self.weight_dict[key]
        total_grad = total_grad * self.FRS_info["delta_k"][self.opt_dim]
        
        # detaching
        Obj = total_cost.cpu().numpy()
        Grad = total_grad.cpu().numpy()
        
        return Obj, Grad
    
    def _compute_objective_qpos_goal(self, ka, qpos, qvel, qgoal):
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
        T_PEAK         = self.time_pieces[0]
        T_FULL         = self.t_f
        
        # qpos(T_PEAK)
        qpos_peak      = qpos + qvel * T_PEAK + 0.5 * ka * T_PEAK**2
        qvel_peak      = qvel + ka * T_PEAK

        # qpos(T_FULL)
        braking_accel  = (0 - qvel_peak)/(T_FULL - T_PEAK)
        qf             = qpos_peak + qvel_peak*(T_FULL - T_PEAK) + 0.5*braking_accel*(T_FULL-T_PEAK)**2
        grad_qf        = 0.5 * T_PEAK * T_FULL

        dq = self.wrap_cont_joint_to_pi(qf - qgoal).squeeze()

        Obj  = torch.sum(dq**2)
        Grad = 2*grad_qf*dq

        return Obj, Grad
    
    def _compute_objective_qpos_projection(self, ka, qpos, qvel, plan):
        """
        c(ka) = \sum_{t=0}^{T}||q(t) - qd(t)||

        Args:
            ka:   (n_optvar,) joint acceleration - unnormalized
            qpos: (n_link,) current joint position
            qvel: (n_link,) current joint velocity
            plan: (ReferenceTrajectory), the nominal trajectory to project
        
        Returns:
            Obj: c(ka)
            Grad: the gradient of c(ka) w.r.t. ka
        """
        assert(isinstance(plan, ReferenceTrajectory))

        T_FULL         = self.t_f

        t_eval = plan.t_des[plan.t_des <= T_FULL]
        q_eval = plan.x_des[plan.t_des <= T_FULL]

        q       = (qpos + torch.outer(t_eval, qvel)+ 0.5*torch.outer(t_eval**2, ka))
        dq      = self.wrap_cont_joint_to_pi(q-q_eval)
        grad_dq = 0.5*t_eval**2

        Obj     = torch.sum(torch.mean(dq**2, dim=0))
        Grad    = 2*grad_dq@dq

        return Obj, Grad

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

        # Constraint 1: collision b/w robot arm - {objects, arena, mount}
        data["robot_collision"] = self._prepare_constraint_robot_collision(
                                            qpos            = qpos, 
                                            qvel            = qvel, 
                                            obstacles       = obs_dict["zonotope"]["obstacle"],
                                            T_world_to_base = obs_dict["T_world_to_base"])
        
        # Constraint 2: joint position/velocity constraint
        data["joint"]  = dict(n_con = 2*(3*self.n_pos_lim + self.n_links))

        # Constraint 4: collision b/w robot gripper - {non-active object, arena, mount}
        
        # Constraint 5: collision b/w robot gripper outside - {active objet}

        # Constraint 6: self-collision

        return data
    
    def _prepare_constraint_robot_collision(self, qpos, qvel, obstacles, T_world_to_base):
        """
        Prepare the constraints data for collision-check of robot arms (w/o gripper) and 
        objects / mount / arena.

        Args:
            qpos: (n_link, 1), joint position of the robot
            qvel: (n_link, 1), joint velocity of the robot
            obstacles: list <zonotope>: the list of zonotope representation for the objects, arena, mounts
            T_world_to_base: (4, 4), the transformation matrix from world to the robot_base

        Returns:
            data: dict,
                    (A, b) Ax <= b H-polytopee representation
                    FO_links: (n_link, 1) list of polyzonotope representation of the all possible volumes of each link
        """
        # robot forward occupancy
        _, R_trig         = process_batch_JRS_trig(self.JRS_tensor, qpos, qvel, self.joint_axes)
        link_polyzonos    = [pz.to(dtype=self.dtype, device=self.device) for pz in self._link_polyzonos_stl]
        FO_links,_, _     = forward_occupancy(R_trig, link_polyzonos, self.params, T_world_to_base=T_world_to_base)

        # obstacle zonotope
        n_obs     = len(obstacles)

        # constraint data
        # the robot link should not intersect the obstacles for all time horizon
        n_interval = self.n_timestep - 1

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
        
        # Constraint - Initialization
        ka   = einops.repeat(self.to_tensor(x), 'n -> repeat n', repeat=n_interval) * self.FRS_info["delta_k"][self.opt_dim]

        # C1 - joint constraint
        Cons_joint, Jac_joint_to_ka = self._compute_joint_constraints(ka=ka[0], qpos=qpos, qvel=qvel)

        # C2 - robot arm collision constraint
        Cons_rc, Jac_rc_to_ka = self._compute_robot_collision_constraints(ka   = ka, 
                                                                          data = problem_data["constraint"]["robot_collision"])

        # transform it back with respect to x
        Cons       = torch.hstack((Cons_joint, Cons_rc))
        Jac_to_ka  = torch.vstack((Jac_joint_to_ka, Jac_rc_to_ka))
        Jac_to_x   = Jac_to_ka * self.FRS_info["delta_k"][self.opt_dim]

        return Cons, Jac_to_x

    def _compute_robot_collision_constraints(self, ka, data):
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
        """
        Discrete collision checking algorithm.

        It checks the collision by intersecting the robot zonotopes at each planning time steps with
        the obstacles zonotopes.

        Args
            plan: ReferenceTrajectory torch array (qpos)
            obs: observation dictionary

        Returns
            is_collision: bool

        NOTE: This could be accelerated using batch zonotope
        """
        assert isinstance(plan, ReferenceTrajectory), "The plan should be the instance of ReferenceTrajectory"
        assert "zonotope" in obs.keys() and "robot" in obs["zonotope"].keys() and "obstacle" in obs["zonotope"].keys()
        robot_zonotope = obs["zonotope"]["robot"][0]
        assert(torch.allclose(robot_zonotope.center, plan[0][1]+self.offset, 1e-6), 
               "The plan is not consistent with the current position")

    def collision_check(self, plan, obs):
        assert isinstance(plan, ReferenceTrajectory), "The plan should be the instance of ReferenceTrajectory"
        assert "zonotope" in obs.keys() and "obstacle" in obs["zonotope"].keys()

        qs = plan.x_des
        R_q = self.rot(qs)

        obs_zonos = obs["zonotope"]["obstacle"]
        n_obs = len(obs_zonos)

        if len(R_q.shape) == 4:
            time_steps = len(R_q)
            R, P = self.to_tensor(torch.eye(3)), self.to_tensor(torch.zeros(3))
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
            R, P = self.to_tensor(torch.eye(3)), self.to_tensor(torch.zeros(3))
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
    # Helper
    # ---------------------------------------------- #
    def wrap_cont_joint_to_pi(self, phases):
        if len(phases.shape) == 1:
            phases = phases.unsqueeze(0)
        phases_new = torch.clone(phases)
        phases_new[:,~self.lim_flag] = (phases[:,~self.lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi
        return phases_new

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
        qdes = plan.x_des[math.ceil(t_des/self.dt)]

        action = (qdes - qpos).cpu().numpy()
        action  = scale_array_from_A_to_B(action, 
                                          A=[self.action_config["output_min"], self.action_config["output_max"]], 
                                          B=[self.action_config["input_min"] , self.action_config["input_max"]]) # (T, D) array
        action  = np.clip(action, self.action_config["input_min"], self.action_config["input_max"])

        return action



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
    
    def get_arm_zonotopes_at_q(self, q, T_frame_to_base):
        """
        Return the forward occupancy of the robot arms with the zonotopes.
        It does by doing the forward kinematics of the robot.

        Args
            qpos: (n_links,) torch tensor

        Output
            arm_zonos: list of zonotopes
        """
        Ri = self.to_tensor(T_frame_to_base[0:3, 0:3])
        Pi = self.to_tensor(T_frame_to_base[0:3, 3])
        
        R_qi = self.rot(q)
        arm_zonos = []
        for i in range(self.n_links):
            Pi = Ri@self.params["P"][i] + Pi
            Ri = Ri@self.params["R"][i]@R_qi[i]
            arm_zono = Ri@self._link_zonos_stl[i] + Pi
            arm_zonos.append(arm_zono)
        
        return arm_zonos
    
    def get_eef_pos_at_q(self, q, T_frame_to_base):
        """
        Return the pose of the robot ende-effector at given joint angle
        It does by doing the forward kinematics of the robot.

        Args
            qpos: (n_links,) torch tensor

        Output
            P: position of the end-effector
        """
        Ri = self.to_tensor(T_frame_to_base[0:3, 0:3])
        Pi = self.to_tensor(T_frame_to_base[0:3, 3])
        
        R_qi = self.rot(q)
        for i in range(self.n_links):
            Pi = Ri@self.params["P"][i] + Pi
            Ri = Ri@self.params["R"][i]@R_qi[i]
        
        return Pi.cpu().numpy()
    
    def get_eef_pos_from_plan(self, plan, T_world_to_base):
        assert isinstance(plan, ReferenceTrajectory)

        eef_pos = [self.get_eef_pos_at_q(plan.x_des[i], T_world_to_base)
                   for i in range(len(plan))]
        eef_pos = np.stack(eef_pos, axis=0)

        return eef_pos


    def get_forward_occupancy_from_plan(self, plan, T_world_to_base, only_end_effector = False):
        """
        Get series of forward occupancy zonotopes from the given plan

        Args:
            plan (ReferenceTrajectory), the plan that contains the joint angle, joint velocity
            T_world_to_base (4 by 4 matrix), the transformation matrix from the world to the base
            only_end_effector (bool), get forward occupancy of only end effector

        Returns:
            FO_link (zonotopes), the zonotope list of plan length
        """
        assert isinstance(plan, ReferenceTrajectory)

        FO_link = []

        for i in range(len(plan)):
            FO_i = self.get_arm_zonotopes_at_q(plan.x_des[i], T_world_to_base)
            if only_end_effector:
                FO_i = [FO_i[-1]]
            FO_link.append(FO_i)

        return FO_link

class ArmtdPlannerXML(ArmtdPlanner):
    def __init__(self,
                 action_config, 
                 robot_name,
                 **kwargs):
        
        self.robot_xml_file       = os.path.join(robosuite.__path__[0], 
                                                 f"models/assets/robots/{robot_name.lower()}/robot.xml")
        self.joint_vel_limit_dict = dict(
            kinova3 = [1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218]
        )
        
        assert os.path.exists(self.robot_xml_file)

        super().__init__(action_config, robot_name, **kwargs)

    def load_robot_n_links(self, robot_name):
        params = RobotUtils.load_single_mujoco_robot_arm_params(self.robot_xml_file)
        self.n_links = params['n_joints']

    def load_robot_config_and_joint_limits(self, robot_name):
        """
        Read the params (n_joints, P, R, joint_axes) from the robosuite xml file
        """
        assert robot_name.lower() in self.joint_vel_limit_dict.keys(), \
            f"The joint velocity limit of {robot_name} has not been registered."
        
        # parse parameters from the xml file
        
        params = RobotUtils.load_single_mujoco_robot_arm_params(self.robot_xml_file)

        self.params = {'n_joints': params['n_joints'],
                       'P'       : [self.to_tensor(p) for p in params['P']],
                       'R'       : [self.to_tensor(r) for r in params['R']]}
        
        # joint axes
        self.joint_axes   = self.to_tensor(torch.stack(params['joint_axes']))
        w = self.to_tensor([[[0,0,0],[0,0,1],[0,-1,0]],
                            [[0,0,-1],[0,0,0],[1,0,0]],
                            [[0,1,0],[-1,0,0],[0,0,0.0]]])
        self.rot_skew_sym = (w@self.joint_axes.T).transpose(0,-1)
        
        # joint position limit
        self.pos_lim        = self._process_pos_lim(params['pos_lim'])
        self.lim_flag       = np.array(params['lim_flag'])
        self.actual_pos_lim = self.pos_lim[self.lim_flag]
        self.n_pos_lim      = int(sum(self.lim_flag))

        # joint velocity limit
        self.vel_lim        = self.to_tensor(self.joint_vel_limit_dict[robot_name.lower()])

    def _process_pos_lim(self, pos_lim):
        """
        Process the joint position limit compatible to ArmtdPlanner

        Args:
            pos_lim (list): list length of n_joint, each list has 2D array of joint limit [min, max].
                            If the joint limit does not exist, it says None.
        
        Returns:
            pos_lim_processed (tensor): an array size of (n_joint, 2), 
                                        1st column indicates max, 2nd column indicates min.
                                        None has been replaced to [pi, -pi]
        """
        PI = torch.pi
        
        pos_lim_processed = [(torch.tensor([PI, -PI]) 
                              if x is None else torch.tensor([max(x), min(x)])) 
                              for x in pos_lim]
        
        pos_lim_processed = self.to_tensor(torch.stack(pos_lim_processed))

        return pos_lim_processed

        
