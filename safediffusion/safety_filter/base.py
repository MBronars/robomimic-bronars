import sys
import time

import torch
import numpy as np
import cyipopt

sys.path.append('../')

from armtd.reachability.conSet import batchZonotope
from armtd.reachability.forward_occupancy.FO import forward_occupancy
from armtd.reachability.joint_reachable_set.load_jrs_trig import preload_batch_JRS_trig
from armtd.reachability.joint_reachable_set.process_jrs_trig import process_batch_JRS_trig
from armtd.planning.armtd_3d import wrap_to_pi

from safediffusion.environment.zonotope_env import ZonotopeMuJoCoEnv

# Change this along with the n_timestepss
T_PLAN = 0.5
T_FULL = 1.0

class ReferenceTrajectory:
    # TODO: overload the indexing operator, slicing operator and change the dtype and device
    # TODO: change the code accordingly for the safety filter
    def __init__(self, t_des, x_des, dx_des):
        assert t_des.shape[0] == x_des.shape[0] == dx_des.shape[0]

        self.t_des = t_des
        self.x_des = x_des
        self.dx_des = dx_des

    def __len__(self):
        return self.t_des.shape[0]

class SafetyFilter:
    """
    TODO: convert back-and-forth between parameter, referencetrajectory, action
    """
    def __init__(self, zono_env,
                 zono_order=40,
                 max_combs=200,
                 n_head=1,
                 dtype=torch.float,
                 device=torch.device('cpu'),
                 nlp_time_limit=0.5):
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
        self.eps = 1e-6
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.nlp_time_limit = nlp_time_limit

        self.cache_robot_info_from_zono_env(zono_env)
        self.JRS_tensor = preload_batch_JRS_trig(dtype=self.dtype, device=self.device)

        # Initialize settings related to the trajectory optimization
        self.generate_combinations_upto()

        # Safety layer configuration
        self.w_goal   = 0.0
        self.w_proj   = 1.0
        self.n_head   = n_head
        # self.t_head   = t_head # head plan horizon
        self.dt_plan  = 1.0/zono_env.env.control_freq # time step for the plan

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
        actions = actions[:, self.env.helper_controller.qpos_index]

        # Step 2. Compute dx_des from actions
        # TODO: Scaling factor 0.05 is hardcoded for now, remove this later
        # scale actions to velocity
        actions = actions*0.05
        # actions = np.clip(actions, LLC.input_min, LLC.input_max)
        # actions = (actions - LLC.action_input_transform) * LLC.action_scale + LLC.action_output_transform

        dx_des = actions/self.dt_plan
        dx_des = torch.asarray(dx_des, dtype=self.dtype, device=self.device)

        # x_des
        x_cur = self.env.qpos
        x_des = x_cur + np.cumsum(actions[:-1], 0)
        x_des = np.vstack([x_cur, x_des])
        x_des = torch.asarray(x_des, dtype=self.dtype, device=self.device)

        # t_des
        t_des = torch.arange(0, n_actions*self.dt_plan, self.dt_plan, dtype=self.dtype, device=self.device)

        reference_plan = ReferenceTrajectory(t_des, x_des, dx_des)
        
        return reference_plan
    
    def reference_traj_to_actions(self, traj):
        """
        Map the reference trajectory back to the actions for robosuite environment
        This does by computing the difference between the joint angles and scaling it back to the action space
        """
        assert isinstance(traj, ReferenceTrajectory)

        # TODO: change 0.05 part later
        actions = traj.x_des[1:] - traj.x_des[:-1]
        actions = actions/0.05

        # LLC = self.env.helper_controller
        # actions = (actions-LLC.action_output_transform)/LLC.action_scale + LLC.action_input_transform
        # actions = np.clip(actions, LLC.input_min, LLC.input_max)
        actions = torch.asarray(actions, dtype=self.dtype, device=self.device)

        return actions

    def __call__(self, actions):
        """
        Assume diffusion model gives joint position as actions
        """
        traj_des = self.actions_to_reference_traj(actions)
        (is_safe, backup_plan) = self.monitor(traj_des)

        if is_safe:
            actions_safe = actions[:self.n_head]
            self.backup_actions = self.reference_traj_to_actions(backup_plan)

        else:
            actions_safe = self.backup_actions[:self.n_head]
            self.backup_actions = self.backup_actions[self.n_head:]

        return actions_safe
    
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
        self.vel_lim = zono_env.vel_lim.cpu()
        self.pos_lim = zono_env.pos_lim.cpu()
        self.actual_pos_lim = zono_env.pos_lim[zono_env.lim_flag].cpu()
        self.n_pos_lim = int(zono_env.lim_flag.sum().cpu())
        self.lim_flag = zono_env.lim_flag.cpu()

    def check_head_plan_safety(self, t, x, dx):
        """ Check if the head plan is safe
        
        Args:
            t: The time vector of the head plan
            x: The position vector of the head plan
        
        Returns:
            bool: True if the head plan is safe, False otherwise
        """

        is_colliding = self.env.collision_check(x)
        is_exceeding_joint_limit = self.env.joint_limit_check_with_explicit_plans(t, x, dx)

        return not is_colliding and not is_exceeding_joint_limit
    
    def check_backup_plan_feasibility(self, t, x, dx):
        """ Check if the safe backup plan is feasible
        
        Args:
            t: The time vector of the tail plan
            x: The position vector of the tail plan
            dx: The velocity vector of the tail plan
        
        Returns:
            bool: True if the tail plan is safe, False otherwise
        """
        # prepare constraints for NLP
        self.prepare_constraints(x[0], dx[0], self.env.obs_zonos)

        # trajectory optimization
        ka_backup, flag = self.trajopt(t_des=t,
                                        x_des=x,
                                        dx_des=dx,
                                        ka_0=torch.zeros(self.n_links),
                                        qgoal=self.env.qgoal
                                        )
        
        is_tail_plan_safe = flag == 0

        backup_plan = self.rollout_param(q0=x[0], kv=dx[0], ka=ka_backup)
        backup_plan = ReferenceTrajectory(*backup_plan)
        
        # TODO: bad coding
        self.ka_backup = ka_backup

        if not is_tail_plan_safe:
            print("[INFO] Backup plan is not safe")

        return (is_tail_plan_safe, backup_plan)
    
    def rollout_param(self, q0, kv, ka):
        """ Rollout the trajectory parameterized by the parameter ka
        
        Args:
            q0: The initial planning state vector
                of shape (n,) where n is the number of the joints
            kv: The initial velocity trajectory parameter vector
                of shape (n,) where n is the number of the joints
            ka: The acceleration trajectory parameter vector
                of shape (n,) where n is the number of the joints
        
        Output:
            t_des: The time vector of shape (n_t,)
            q_des: The desired position vector of shape (n_t, n_joint)
            dq_des: The desired velocity vector of shape (n_t, n_joint)
        """

        q0 = q0.to(self.device)
        kv = kv.to(self.device)
        ka = ka.to(self.device)

        t_des = torch.arange(0, T_FULL+self.eps, self.dt_plan, dtype=self.dtype, device=self.device)
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

        return (t_des, q_des, dq_des)

    def monitor(self, traj_des):
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

        # TODO: Implement checking first criterion: head plan feasibility
        # Naive checker for now: collision checking
        head_mask = traj_des.t_des <= (self.dt_plan * self.n_head)
        head_time = traj_des.t_des[head_mask].cpu()
        head_vel = traj_des.dx_des[head_mask].cpu()
        head_pos = traj_des.x_des[head_mask].cpu()

        is_head_plan_safe = self.check_head_plan_safety(t=head_time, x=head_pos, dx=head_vel)

        if not is_head_plan_safe:
            print("[INFO] Head plan is not safe")

        # TODO: Implement second criterion: tail backup plan feasibility
        tail_mask = traj_des.t_des >= (self.dt_plan * self.n_head)
        tail_pos = traj_des.x_des[tail_mask].cpu()
        tail_time = traj_des.t_des[tail_mask].cpu() - traj_des.t_des[tail_mask][0].cpu()
        tail_vel = traj_des.dx_des[tail_mask].cpu()

        (is_backup_plan_feasible, backup_plan) = self.check_backup_plan_feasibility(t=tail_time, x=tail_pos, dx=tail_vel)

        is_safe = is_head_plan_safe and is_backup_plan_feasible

        return (is_safe, backup_plan)
        
    
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
                
                # except Exception as e:
                #     raise e


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
    ###########      Utils                     ########
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