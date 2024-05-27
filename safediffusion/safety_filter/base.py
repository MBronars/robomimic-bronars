import os
import sys

import torch
import numpy as np
import cyipopt

use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy:
    from zonopy.contset import batchZonotope
else:
    from safediffusion.armtdpy.reachability.conSet import batchZonotope

from safediffusion.armtdpy.reachability.forward_occupancy.FO import forward_occupancy
from safediffusion.armtdpy.reachability.joint_reachable_set.load_jrs_trig import preload_batch_JRS_trig
from safediffusion.armtdpy.reachability.joint_reachable_set.process_jrs_trig import process_batch_JRS_trig
from safediffusion.armtdpy.planning.armtd_3d import wrap_to_pi

from safediffusion.environment.zonotope_env import ZonotopeMuJoCoEnv

from safediffusion.utils.npy_utils import scale_array_from_A_to_B

# Change this along with the n_timestepss
T_PLAN = 0.5
T_FULL = 1.0

class ReferenceTrajectory:
    # TODO: change the dtype and device
    # TODO: change the code accordingly for the safety filter
    def __init__(self, t_des, x_des, dx_des=None):
        assert t_des.shape[0] == x_des.shape[0]

        if dx_des is None:
            # If dx_des is not provided, compute it from x_des using forward difference
            dt_des = (t_des[1:] - t_des[:-1]).unsqueeze(-1)
            self.dx_des = (x_des[1:] - x_des[:-1])/dt_des
            self.x_des = x_des[:-1]
            self.t_des = t_des[:-1]
        else:
            assert t_des.shape[0] == dx_des.shape[0]
            self.x_des = x_des
            self.t_des = t_des
            self.dx_des = dx_des
        
        self._created_from_traj_param = False

    def __getitem__(self, key):
        """ 
        Returns another ReferenceTrajectory object with the sliced data
        """
        if isinstance(key, slice):
            plan = ReferenceTrajectory(self.t_des[key], self.x_des[key], self.dx_des[key])
            if self._created_from_traj_param:
                plan.stamp_trajectory_parameter(self._traj_param)
            return plan
        
        elif isinstance(key, int):
            if key < 0:
                key = len(self) + key
            if key >= self.__len__():
                raise IndexError("Index out of range")
            return (self.t_des[key], self.x_des[key], self.dx_des[key])
        
        elif isinstance(key, (list, torch.Tensor)) and key.dtype == torch.bool:
            # Handle boolean array masking
            if len(key) != len(self.t_des):
                raise ValueError("Boolean mask must have the same length as the reference trajectory")
            
            plan = ReferenceTrajectory(self.t_des[key], self.x_des[key], self.dx_des[key])
            if self._created_from_traj_param:
                plan.stamp_trajectory_parameter(self._traj_param)
            return plan

        else:
            raise TypeError("Invalid argument type")
        
    def __len__(self):
        return self.t_des.shape[0]
    
    def set_start_time(self, t_start = 0):
        """
        Shift the time vector to start from t_start
        """
        self.t_des = self.t_des - self.t_des[0] + t_start

    def stamp_trajectory_parameter(self, traj_param):
        """
        Stamp the parameter k to the reference trajectory
        """
        self._created_from_traj_param = True
        self._traj_param = traj_param
    
    def get_trajectory_parameter(self):
        """
        Get the stamped parameter to the reference trajectory
        """
        assert self._created_from_traj_param
        return self._traj_param

class SafetyFilter:
    """
    TODO: convert back-and-forth between parameter, referencetrajectory, action
    TODO: this safety filter assumes all equally-spaced time steps
    """
    def __init__(self, zono_env,
                 zono_order=40,
                 max_combs=200,
                 n_head=1,
                 dtype=torch.float,
                 device=torch.device('cpu'),
                 nlp_time_limit=0.5,
                 verbose=False):
        """
        Receding-horizon implementation of the safety filter

        Args:
            zono_env: zonotope twin of the simulation environment, worlds are zonofied
                      all zonotopes are with respect to the world frame
        """
        assert isinstance(zono_env, ZonotopeMuJoCoEnv)
        self.dtype  = dtype
        self.device = device
        self.env = zono_env
        self.PI = torch.tensor(torch.pi,dtype=self.dtype,device=self.device)
        self.n_timesteps = 100 # Number of timesteps for CORA reachability
        self.eps = 1e-5
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.nlp_time_limit = nlp_time_limit

        self.cache_robot_info_from_zono_env(zono_env)
        self.JRS_tensor = preload_batch_JRS_trig(dtype=self.dtype, device=self.device)

        # Safety layer configuration
        self.w_goal   = 0.0
        self.w_proj   = 1.0
        self.n_head   = n_head

        # Initialize settings related to the trajectory optimization
        self.generate_combinations_upto()

        self.verbose = verbose

    def actions_to_reference_traj(self, actions):
        """
        Map the actions from the performance policy to the time-parameterized trajectory for safety filter.
        Some dimension of actions might include gripper actions which is not filtered by safety filter.

        Args
            actions: tensor shape of (N_horizon, action_dim), joint angle changes

        Output
            reference_traj: ReferenceTrajectory object that has t_des, x_des, dx_des

        TODO: Few Questions to think about:
            1) Does the horizon of the performance policy influences optimization?
            2) What should be the appropriate dtype and device?

        Author: Wonsuhk Jung
        """
        n_actions = actions.shape[0]

        # Step 1. Parse actions related to the joint space
        ctrl = self.env.helper_controller
        actions = actions[:, ctrl.qpos_index]
        actions = np.clip(actions, ctrl.input_min, ctrl.input_max)
        actions = scale_array_from_A_to_B(actions, A=[ctrl.input_min, ctrl.input_max], B=[ctrl.output_min, ctrl.output_max]) # (T, D) array

        # x_des: (T, D) array, last joint angle is dropped
        x_cur = self.env.qpos
        x_des = x_cur + np.cumsum(actions, 0)
        x_des = np.vstack([x_cur, x_des])
        x_des = torch.asarray(x_des, dtype=self.dtype, device=self.device)

        # t_des
        t_des = torch.arange(0, (n_actions+1)*self.dt_plan, self.dt_plan, dtype=self.dtype, device=self.device)

        reference_traj = ReferenceTrajectory(t_des, x_des)
        
        return reference_traj
    
    def reference_traj_to_actions(self, traj):
        """
        Map the reference trajectory back to the actions for robosuite environment
        This does by computing the difference between the joint angles and scaling it back to the action space

        Args
            traj: ReferenceTrajectory object
        Output
            actions: numpy array shape of (N_horizon, action_dim), joint angle changes
        """
        assert isinstance(traj, ReferenceTrajectory)

        # TODO: change 0.05 part later
        ctrl = self.env.helper_controller
        actions = np.array(traj.x_des[1:] - traj.x_des[:-1])
        actions = scale_array_from_A_to_B(actions, A=[ctrl.output_min, ctrl.output_max], B=[ctrl.input_min, ctrl.input_max]) # (T, D) array
        actions = np.clip(actions, ctrl.input_min, ctrl.input_max)
        assert(np.all(actions >= ctrl.input_min) and np.all(actions <= ctrl.input_max)), "Actions are out of range"

        return actions

    def __call__(self, actions):
        """
        Assume diffusion model gives joint position as actions

        Args
            actions: tensor shape of (N_horizon, action_dim), joint angle changes. last dim includes gripper actions.
        """
        # Translate actions to joint angle reference trajectory
        plan_ref    = self.actions_to_reference_traj(actions)
        plan_backup_new = self.monitor_and_compute_backup_plan(plan_ref)

        if plan_backup_new is not None:
            assert np.allclose(plan_backup_new[0][1], plan_ref[self.n_head][1], atol=self.eps), "Backup plan is not starting from the end of the head plan"
            self.clear_plan_backup()
            self.set_plan_backup(plan_backup_new)
            actions_safe = actions[:self.n_head]

        else:
            plan_backup_prev = self.pop_plan_backup(self.n_head)
            actions_joint = self.reference_traj_to_actions(plan_backup_prev)
            actions_gripper = self.get_gripper_actions(actions)[:self.n_head]
            actions_safe = np.hstack([actions_joint, actions_gripper])

            self.disp(f"{len(self._plan_backup)} backup plans left")

        # ctrl = self.env.helper_controller
        # assert(np.all(actions >= ctrl.input_min) and np.all(actions <= ctrl.input_max))
        return actions_safe
    
    def get_gripper_actions(self, actions):
        """
        Extract the gripper actions from the actions
        """
        all_indices_set = set(range(actions.shape[1]))
        joint_indices_set = set(self.env.helper_controller.qpos_index)
        gripper_indices_set = all_indices_set - joint_indices_set

        gripper_indices = sorted(list(gripper_indices_set))

        return actions[:, gripper_indices]
    
    def initialize_backup_plan(self):
        """
        Initialize the backup plan
        """
        q0 = self.env.qpos
        kv = torch.zeros_like(q0)
        kv += torch.randn_like(kv)*0.1
        ka = torch.zeros_like(q0)
        plan = self.rollout_param_to_reference_trajectory(q0, kv, ka)

        assert(self.check_head_plan_safety(plan)), "Initial plan is not safe"

        self.set_plan_backup(plan)
    
    def clear_plan_backup(self):
        """
        Clear the backup plan
        """
        self._plan_backup = None
    
    def set_plan_backup(self, plan):
        """
        Set the backup plan
        """
        self._plan_backup = plan
    
    def pop_plan_backup(self, n):
        """
        Pop the backup plan of n length
        """
        plan = self._plan_backup[:n+1]
        self._plan_backup = self._plan_backup[n:]

        return plan
    
    def get_plan_backup_trajparam(self):
        """
        Get the backup plan
        """
        return self._plan_backup.get_trajectory_parameter()
    
    def forward_occupancy_from_reference_traj(self, traj, only_end_effector=False):
        """
        Compute the forward occupancy from the reference trajectory

        TODO: change this to the batch zonotope version
        """
        assert isinstance(traj, ReferenceTrajectory)

        FO_link = []

        for i in range(len(traj)):
            FO_i = self.env.get_arm_zonotopes_at_q(traj.x_des[i])
            if only_end_effector:
                FO_i = [FO_i[-1]]
            FO_link.append(FO_i)

        return FO_link
    
    def start_episode(self):
        self.env.sync()
        self.initialize_backup_plan()

    def generate_combinations_upto(self):
        self.combs = [torch.combinations(torch.arange(i), 2) for i in range(self.max_combs+1)]

    def cache_robot_info_from_zono_env(self, zono_env):
        """ Load robot-relevant information to the class variable
        
        Args:
            robot: An environment object to parse robot information
        """
        assert zono_env.dimension == 3
        self.dimension = 3
        self.n_links   = len(zono_env.env.robots[0].robot_joints) # number of links (without gripper)
        
        P, R = [], []
        for p,r in zip(zono_env.P0, zono_env.R0):
            P.append(p.to(dtype=self.dtype, device=self.device))
            R.append(r.to(dtype=self.dtype, device=self.device))
        
        self.params = {'n_joints':self.n_links,
                       'P':P, 
                       'R':R}
        
        self.joint_axes = zono_env.joint_axes.to(dtype=self.dtype, device=self.device)
        self.pos_lim = zono_env.pos_lim.cpu()
        self.actual_pos_lim = zono_env.pos_lim[zono_env.lim_flag].cpu()
        self.n_pos_lim = int(zono_env.lim_flag.sum().cpu())
        self.lim_flag = zono_env.lim_flag.cpu()
        self.dt_plan  = 1.0/zono_env.env.control_freq

        # set the velocity limit: should be minimum of the robosuite output max and the real robot velocity limit
        vel_lim_robot = zono_env.vel_lim.cpu()
        vel_lim_robosuite = torch.tensor(zono_env.helper_controller.output_max/self.dt_plan, dtype=self.dtype)
        self.vel_lim = torch.min(vel_lim_robot, vel_lim_robosuite)


    def check_head_plan_safety(self, plan):
        """ Check if the plan is safe
        
        Args:
            t: The time vector of the head plan
            x: The position vector of the head plan
        
        Returns:
            bool: True if the head plan is safe, False otherwise

        NOTE: In this base safety filter, it uses discrete-time forward occupancy to check the safety
        """
        is_colliding = self.env.collision_check(qs=plan.x_des)
        is_exceeding_joint_limit = self.env.joint_limit_check_with_explicit_plans(t_des=plan.t_des, 
                                                                                  x_des=plan.x_des, 
                                                                                  dx_des=plan.dx_des)
        self.disp("Head plan is not safe", when=is_colliding)
        self.disp("Head plan is exceeding joint limit", when=is_exceeding_joint_limit)

        return not is_colliding and not is_exceeding_joint_limit
    
    def compute_backup_plan(self, plan):
        """ Compute the backup plan from the initial state of the tail plan.
        This does by projecting the plan to the safe parameterized trajectory.
        It returns backup plan in (ReferenceTrajectory) if exists. If not, it returns None.
        
        Args:
            plan: (ReferenceTrajectory) The reference plan to compute the backup plan
        
        Returns:
            backup_plan
        """
        # prepare constraints for NLP
        (t_0, x_0, dx_0) = plan[0]

        self.prepare_constraints(x_0, dx_0, self.env.obs_zonos)

        # trajectory optimization
        # TODO: sometimes, for a same optimization problem, it returns flag = -5. Need to investigate
        ka_backup, flag = self.trajopt(t_des=plan.t_des,
                                    x_des=plan.x_des,
                                    dx_des=plan.dx_des,
                                    ka_0=torch.zeros(self.n_links),
                                    qgoal=self.env.qgoal
                                )
        
        is_backup_plan_feasible = (flag == 0)

        if is_backup_plan_feasible:
            backup_plan = self.rollout_param_to_reference_trajectory(q0=x_0, kv=dx_0, ka=ka_backup)
        else:
            backup_plan = None
            self.disp(f"Backup plan is not feasible with flag {flag}")

        return backup_plan
    
    def rollout_param_to_reference_trajectory(self, q0, kv, ka):
        """ Rollout the trajectory parameterized by the parameter q0, kv, ka
        
        Args:
            q0: The initial planning state vector
                of shape (n,) where n is the number of the joints
            kv: The initial velocity trajectory parameter vector
                of shape (n,) where n is the number of the joints
            ka: The acceleration trajectory parameter vector
                of shape (n,) where n is the number of the joints
        
        Output:
            The plan trajectory in the form of ReferenceTrajectory
                t_des: The time vector of shape (n_t,)
                x_des: The desired joint angle vector of shape (n_t, n_joint)
                dx_des: The desired joint velocity vector of shape (n_t, n_joint)

        TODO: Remove T_FULL, T_PLAN from the global variable
        """
        q0 = q0.to(self.device, self.dtype)
        kv = kv.to(self.device, self.dtype)
        ka = ka.to(self.device, self.dtype)

        t_des = torch.arange(0, T_FULL+self.dt_plan, self.dt_plan, dtype=self.dtype, device=self.device)
        idx_peak = int(T_PLAN/self.dt_plan)

        t_to_peak  = t_des[0:idx_peak+1]
        t_to_brake  = t_des[idx_peak+1:] - t_to_peak[-1]
        
        # First piece trajectory (accelerate)
        q_to_peak = wrap_to_pi(q0 + 
                            torch.outer(t_to_peak,kv) + 
                            0.5*torch.outer(t_to_peak**2,ka))
        dq_to_peak = kv + torch.outer(t_to_peak, ka)
        
        # Second piece trajector (decelerate)
        braking_accel = (0-dq_to_peak[-1])/(T_FULL-T_PLAN)
        q_to_brake = wrap_to_pi(q_to_peak[-1] +
                                torch.outer(t_to_brake, dq_to_peak[-1])+
                                0.5*torch.outer(t_to_brake**2,braking_accel))        
        dq_to_brake = dq_to_peak[-1] + torch.outer(t_to_brake, braking_accel)

        q_des = torch.vstack([q_to_peak, q_to_brake])
        dq_des = torch.vstack([dq_to_peak, dq_to_brake])

        traj_des = ReferenceTrajectory(t_des, q_des, dq_des)
        traj_des.stamp_trajectory_parameter(traj_param=ka)

        return traj_des

    def monitor_and_compute_backup_plan(self, traj_des):
        """ Given a performance policy, check if performance policy has safe
        backup plan.

        Args:
            t_des: The desired time vector for the performance policy
            x_des: The desired position vector for the performance policy
            dx_des: The desired velocity vector for the performance policy
            env: The environment object

        Returns:
            bool: True if the performance policy has a safe backup plan, False otherwise
        """
        assert isinstance(traj_des, ReferenceTrajectory)

        backup_plan = None

        # Checking first criterion: head plan safety
        head_plan = traj_des[:self.n_head]
        is_head_plan_safe = self.check_head_plan_safety(head_plan)

        if is_head_plan_safe:
            # Checking second criterion: backup plan feasibility
            tail_plan = traj_des[self.n_head:]
            tail_plan.set_start_time(0)
            backup_plan = self.compute_backup_plan(tail_plan)

        return backup_plan
        
    
    ###################################################
    #########   Trajectory Optimization ###############
    ###################################################

    def prepare_constraints(self,qpos,qvel,obstacles):
        """ Prepare the constraints for NLP. After this, you have N*M constraints
        where N is the number of links, M is the number of the obstacles.
        
        Args
            qpos: The initial position vector for the plan shape (N,)
            qvel: The initial velocity vector for the plan shape (N,)
            obstacles: The list of zonotope representation of obstacle (M,)
        """
        _, R_trig = process_batch_JRS_trig(self.JRS_tensor,
                                           qpos.to(dtype=self.dtype,device=self.device),
                                           qvel.to(dtype=self.dtype,device=self.device),
                                           self.joint_axes)
        # TODO: This should be carefully done: coordinate transformation
        link_polyzonos = [pz.to(dtype=self.dtype, device=self.device) for pz in self.env._link_polyzonos_stl]
        self.FO_link,_, _ = forward_occupancy(R_trig, link_polyzonos, self.params, T_world_to_base=self.env.T_world_to_base) # NOTE: zono_order

        n_obs = len(obstacles)

        self.A = np.zeros((self.n_links,n_obs),dtype=object)
        self.b = np.zeros((self.n_links,n_obs),dtype=object)
        self.g_ka = torch.pi/24 #torch.maximum(self.PI/24,abs(qvel/3))
        for j in range(self.n_links):
            self.FO_link[j] = self.FO_link[j].cpu()
            for o in range(n_obs):                
                obs_Z = obstacles[o].Z.unsqueeze(0).repeat(self.n_timesteps,1,1)
                A, b = batchZonotope(torch.cat((obs_Z,self.FO_link[j].Grest),-2)).polytope(self.combs) # A: n_timesteps,*,dimension  
                self.A[j,o] = A.cpu()
                self.b[j,o] = b.cpu()
        
        self.qpos = qpos.to(dtype=self.dtype).cpu()
        self.qvel = qvel.to(dtype=self.dtype).cpu()

    def trajopt(self, qgoal, t_des, x_des, dx_des, ka_0):
        """ Nonlinear optimization that finds the optimal joint acceleration
        to reach the goal configuration and avoid obstacles. The cost function
        weight balances theh cost projected distance and the cost to reach the goal.
        It tries to compute the safe backup plan of T_FULL seconds from the state
        (self.qpos, self.qvel).
        
        Args
            qgoal: goal configuration
            t_ref: 
            x_ref
        """ 
        n_obs = self.A.shape[1]
        M_obs = self.n_links*self.n_timesteps*n_obs
        M = M_obs+2*self.n_links+6*self.n_pos_lim # number of constraints

        t_des = t_des.to(device=self.device)
        x_des = x_des.to(device=self.device)   

        class nlp_setup():
            x_prev = np.zeros(self.n_links)*np.nan

            def objective(p,x):
                p.compute_objective(x)
                return p.Obj
            
            def gradient(p,x):
                p.compute_objective(x)
                return p.Grad
            
            def constraints(p,x): 
                p.compute_constraints(x)
                return p.Cons

            def jacobian(p,x):
                p.compute_constraints(x)
                return p.Jac
            
            def compute_objective(p,x):
                try:
                    ka = torch.tensor(x, dtype=self.dtype, device=self.device)

                    t_eval = t_des[t_des<T_FULL]
                    q_eval = x_des[t_des<T_FULL]

                    q = (self.qpos + torch.outer(t_eval, self.qvel)+
                        0.5*self.g_ka*torch.outer(t_eval**2,ka))
                    grad_q = 0.5*self.g_ka*t_eval**2

                    qf = self.qpos + T_PLAN*self.qvel + 0.5*self.g_ka*T_PLAN**2*ka
                    grad_qf = 0.5*self.g_ka*T_PLAN**2                    

                    dq = self.wrap_cont_joint_to_pi(q-q_eval)
                    dq_goal = self.wrap_cont_joint_to_pi(qf-qgoal)

                    cost_projected_dist = torch.sum(torch.mean(dq**2, dim=0))
                    grad_cost_projected_dist = 2*grad_q@dq

                    cost_goal = torch.sum(dq_goal**2)
                    grad_cost_goal = 2*grad_qf*dq_goal

                    cost = self.w_proj*cost_projected_dist + self.w_goal*cost_goal
                    grad_cost = self.w_proj*grad_cost_projected_dist + self.w_goal*grad_cost_goal

                    p.Obj = cost.cpu().numpy()
                    p.Grad = grad_cost.cpu().numpy()

                except Exception as e:
                    raise e

            def compute_constraints(p,x):
                # try:
                    if (p.x_prev!=x).any():                
                        ka = torch.tensor(x,dtype=self.dtype).unsqueeze(0).repeat(self.n_timesteps,1)
                        Cons = torch.zeros(M,dtype=self.dtype)
                        Jac = torch.zeros(M,self.n_links,dtype=self.dtype)
                        
                        # position and velocity constraints
                        t_peak_optimum = -self.qvel/(self.g_ka*ka[0]) # time to optimum of first half traj.
                        qpos_peak_optimum = (t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(self.qpos+self.qvel*t_peak_optimum+0.5*(self.g_ka*ka[0])*t_peak_optimum**2).nan_to_num()
                        grad_qpos_peak_optimum = torch.diag((t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(0.5*self.qvel**2/(self.g_ka*ka[0]**2)).nan_to_num())

                        qpos_peak = self.qpos + self.qvel * T_PLAN + 0.5 * (self.g_ka * ka[0]) * T_PLAN**2
                        grad_qpos_peak = 0.5 * self.g_ka * T_PLAN**2 * torch.eye(self.n_links,dtype=self.dtype)
                        qvel_peak = self.qvel + self.g_ka * ka[0] * T_PLAN
                        grad_qvel_peak = self.g_ka * T_PLAN * torch.eye(self.n_links,dtype=self.dtype)

                        bracking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
                        qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2
                        # can be also, qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL+T_PLAN) + 0.5 * (self.g_ka * ka[0]) * T_PLAN * T_FULL
                        grad_qpos_brake = 0.5 * self.g_ka * T_PLAN * T_FULL * torch.eye(self.n_links,dtype=self.dtype) # NOTE: need to verify equation

                        qpos_possible_max_min = torch.vstack((qpos_peak_optimum,qpos_peak,qpos_brake))[:,self.lim_flag] 
                        qpos_ub = (qpos_possible_max_min - self.actual_pos_lim[:,0]).flatten()
                        qpos_lb = (self.actual_pos_lim[:,1] - qpos_possible_max_min).flatten()
                        
                        grad_qpos_ub = torch.vstack((grad_qpos_peak_optimum[self.lim_flag],grad_qpos_peak[self.lim_flag],grad_qpos_brake[self.lim_flag]))
                        grad_qpos_lb = - grad_qpos_ub

                        Cons[M_obs:] = torch.hstack((qvel_peak-self.vel_lim, -self.vel_lim-qvel_peak,qpos_ub,qpos_lb))
                        Jac[M_obs:] = torch.vstack((grad_qvel_peak, -grad_qvel_peak, grad_qpos_ub, grad_qpos_lb))

                        for j in range(self.n_links):
                            c_k = self.FO_link[j].center_slice_all_dep(ka)
                            grad_c_k = self.FO_link[j].grad_center_slice_all_dep(ka)
                            for o in range(n_obs):
                                h_obs = (self.A[j][o]@c_k.unsqueeze(-1)).squeeze(-1) - self.b[j][o]
                                cons, ind = torch.max(h_obs.nan_to_num(-torch.inf),-1) # shape: n_timsteps, SAFE if >=1e-6 
                                A_max = self.A[j][o].gather(-2,ind.reshape(self.n_timesteps,1,1).repeat(1,1,self.dimension)) 
                                grad_cons = (A_max@grad_c_k).squeeze(-2) # shape: n_timsteps, n_links safe if >=1e-6
                                Cons[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = - cons
                                Jac[(j+self.n_links*o)*self.n_timesteps:(j+self.n_links*o+1)*self.n_timesteps] = - grad_cons                            
                        
                        p.Cons = Cons.numpy()
                        p.Jac = Jac.numpy()
                        p.x_prev = np.copy(x) 
                
            def intermediate(p, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
                """Prints information at every Ipopt iteration."""
                print(f"Iteration {iter_count}: {obj_value}, Primal Feas {inf_pr}, Dual Feas {inf_du}")

        nlp = cyipopt.Problem(
            n = self.n_links,
            m = M,
            problem_obj=nlp_setup(),
            lb = [-1]*self.n_links,
            ub = [1]*self.n_links,
            cl = [-1e20]*M,
            cu = [-1e-6]*M,
        )

        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)
        nlp.add_option('max_wall_time', self.nlp_time_limit)

        k_opt, self.info = nlp.solve(ka_0.cpu().numpy())

        k_opt = torch.tensor(self.g_ka * k_opt, dtype=self.dtype, device=self.device)
        flag  = self.info['status']

        return k_opt, flag
    
    ###################################################
    # Zonotope Environment Biniding uitls
    ###################################################
    def sync_env(self):
        """
        Synchronize the zonotope environment with the simulation environment

        Returns the observation from the zonotope world
        """
        return self.env.sync()

    ###################################################
    # Utils
    ###################################################
    def wrap_cont_joint_to_pi(self,phases):
        if len(phases.shape) == 1:
            phases = phases.unsqueeze(0)
        phases_new = torch.clone(phases)
        phases_new[:,~self.lim_flag] = (phases[:,~self.lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi
        return phases_new
    
    def FO_zonos_sliced_at_param(self, k):
        zonos = []
        for j in range(self.n_links): 
            FO_link_slc = self.FO_link[j].slice_all_dep((k/self.g_ka).unsqueeze(0).repeat(100,1)).reduce(4)
            zonos.append(FO_link_slc)
        
        return zonos
    
    def disp(self, msg, when=None):
        """
        Display the message when the condition is met and verbose is True
        """
        if self.verbose and (when is None or when):
            msg_format = "[INFO]     " + msg
            print(msg_format)