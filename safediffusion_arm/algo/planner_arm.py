import os
from copy import deepcopy

import torch
import numpy as np

import robomimic
import einops


from safediffusion.armtdpy.environments.arm_3d import Arm_3D
from safediffusion.armtdpy.environments.arm_3d import Arm_3D
from safediffusion.armtdpy.planning.armtd_3d import wrap_to_pi
from safediffusion.armtdpy.reachability.forward_occupancy.FO import forward_occupancy
from safediffusion.armtdpy.reachability.joint_reachable_set.load_jrs_trig import preload_batch_JRS_trig
from safediffusion.armtdpy.reachability.joint_reachable_set.process_jrs_trig import process_batch_JRS_trig

from safediffusion.algo.helper import traj_uniform_acc
from safediffusion.algo.planner_base import ParameterizedPlanner

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
    def __init__(self, LLC, robot_name="Kinova3", **kwargs):
        # prepare the useful variables for robot configuration, joint limits, reachable sets
        self.dimension  = 3
        self.LLC        = LLC # low-level controller

        # load robot configuration
        self.robot_name = robot_name
        self.load_robot_config_and_joint_limits(self.robot_name)
        self.load_robot_link_zonotopes(self.robot_name)
        self.load_joint_reachable_set()

        # load zonotope configuration
        self.zono_order = kwargs["zonotope"]["order"]
        self.max_combs  = kwargs["zonotope"]["max_comb"]
        self.combs      = self.generate_combinations_upto(self.max_combs)

        # Trajectory design
        state_dict = {f"q{joint_id}": joint_id for joint_id in range(self.n_links)}
        param_dict = {f"kv{joint_id}": joint_id for joint_id in range(self.n_links)}
        param_dict.update({f"ka{joint_id}": joint_id + self.n_links for joint_id in range(self.n_links)})

        # Timing
        dt         = 0.01
        t_f        = 1.0

        ParameterizedPlanner.__init__(self, state_dict, param_dict, dt, t_f, **kwargs)

        # piecewise trajectory design
        self.time_pieces = [0.5]
        assert min(self.time_pieces) > 0
        assert max(self.time_pieces) < t_f

        # trajectory optimization
        self.opt_dim     = range(self.n_links, 2*self.n_links)
        self.weight_dict = dict(qpos_goal = 0.0, qpos_projection = 0.0)


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
        robot_model = Arm_3D(robot_name)
        self._link_zonos_stl = robot_model._link_zonos
        self._link_polyzonos_stl = robot_model.link_zonos


    def load_robot_config_and_joint_limits(self, robot_name):
        """
        Cache the robot information, mainly referencing Arm_3D library (armtdpy/environments/robots)

        This code set-up the robot configuration, safety specifications.

        TODO: Find TODO notes below
        """
        robot_model    = Arm_3D(robot_name)

        # robot configuration specifications
        self.n_links = robot_model.n_links # number of links (without gripper)
        self.params  = {'n_joints' : self.n_links,
                       'P'        : [self.to_tensor(p) for p in robot_model.P0], 
                       'R'        : [self.to_tensor(r) for r in robot_model.R0]}
        
        self.joint_axes = self.to_tensor(robot_model.joint_axes)

        # safety specifications 1: joint angle limit
        self.pos_lim        = robot_model.pos_lim.cpu()
        self.actual_pos_lim = robot_model.pos_lim[robot_model.lim_flag].cpu()
        self.n_pos_lim      = int(robot_model.lim_flag.sum().cpu())
        self.lim_flag       = robot_model.lim_flag.cpu()

        # safety specifications 2: joint velocity limit
        self.vel_lim = robot_model.vel_lim.cpu()
        
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
        # NOTE: makeshift
        goal_dict = {}
        goal_dict["qgoal"] = self.to_tensor(torch.zeros(7, ))

        return deepcopy(obs_dict), deepcopy(goal_dict)
    
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
            data: (dict: key = objective_name, value = data)
        """
        data = {}

        active_obj_id   = 1
        active_obj_pose = obs_dict["object"][7*active_obj_id:7*(active_obj_id+1)]

        data = dict(
            eef_pose  = dict(eefgoal = active_obj_pose),
            qpos_goal = dict(qgoal = goal_dict["qgoal"])
        )  

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
        Obj["qpos_goal"], Grad["qpos_goal"] = self._compute_objective_qpos_goal(
                                                        ka    = ka, 
                                                        qpos  = qpos, 
                                                        qvel  = qvel, 
                                                        data  = objective_data["qpos_goal"])
        # eef goal

        # qpos project
        
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
    
    def _compute_objective_qpos_goal(self, ka, qpos, qvel, data):
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
        assert "qgoal" in data.keys()

        qgoal          = data["qgoal"]

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
    
    # ---------------------------------------------- #
    # Helper
    # ---------------------------------------------- #

    def wrap_cont_joint_to_pi(self, phases):
        if len(phases.shape) == 1:
            phases = phases.unsqueeze(0)
        phases_new = torch.clone(phases)
        phases_new[:,~self.lim_flag] = (phases[:,~self.lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi
        return phases_new 
    
    def track_reference_traj(self, obs_dict, reference_traj):
        """
        Return the actions that can be sent to the environment.

        Action representation for the robomimic joint angle controller is the delta joint angle.

        Args:
            ob  : (dict), observation dictionary from the environment
            plan: (ReferenceTrajectory): joint angle reference trajectories
        """
        
        qpos_curr      = np.array(self.get_pstate_from_obs(obs_dict = obs_dict))

        # compensate for the initial lag
        delta_qpos     = np.array(reference_traj.x_des[1:] - reference_traj.x_des[:-1])
        delta_qpos[0] += reference_traj.x_des[0] - qpos_curr
        actions        = delta_qpos
        actions        = np.clip(actions, self.LLC.output_min, self.LLC.output_max)










        offset  = np.array(reference_traj.x_des[0] - qpos) # if initial planning state is not the same as the current state
        actions = np.array(reference_traj.x_des[1:] - reference_traj.x_des[:-1]) + offset
        if ((actions.max(0) > self.LLC.output_max) | (actions.min(0) < self.LLC.output_min)).any():
            self.disp("Actions are out of range: joint goal position gets different with the plan", self.verbose)

        actions = np.clip(actions, self.LLC.output_min, self.LLC.output_max)
        actions = scale_array_from_A_to_B(actions, A=[self.LLC.output_min, self.LLC.output_max], B=[self.LLC.input_min, self.LLC.input_max]) # (T, D) array
        actions = np.clip(actions, self.LLC.input_min, self.LLC.input_max)
        assert(np.all(actions >= self.LLC.input_min) and np.all(actions <= self.LLC.input_max)), "Actions are out of range"

        return actions
