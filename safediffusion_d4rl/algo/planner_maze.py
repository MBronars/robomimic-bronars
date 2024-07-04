import os
from copy import deepcopy

import numpy as np
import torch
import einops

from scipy.io import loadmat

import robomimic
import matplotlib.pyplot as plt

from safediffusion.algo.plan import ReferenceTrajectory
from safediffusion.algo.planner_base import ParameterizedPlanner
from safediffusion.algo.helper import traj_uniform_acc
from safediffusion.utils.reachability_utils import get_zonotope_from_segment

# decide which zonotope library to use
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy: 
    from zonopy.contset import zonotope, batchZonotope, polynomial_zonotope
else: 
    from safediffusion.armtdpy.reachability.conSet import zonotope, batchZonotope, polyZonotope

class Simple2DPlanner(ParameterizedPlanner):
    def __init__(self, **kwargs):
        """
        Tests out the ARMTD-style planner
        """
        # TODO: Make this as an abstract function
        state_dict = {"p_x": 0 , "p_y": 1}
        param_dict = {"k_vx": 0, "k_vy": 1, "k_ax": 2, "k_ay": 3}
        
        # timing
        dt = 0.01
        t_f = 1.0
        
        ParameterizedPlanner.__init__(self, state_dict, param_dict, dt, t_f, **kwargs)

        # Piecewise trajectory design
        self.time_pieces = [0.5]
        self.offset = self.to_tensor([1, 1])
        assert min(self.time_pieces) > 0
        assert max(self.time_pieces) < t_f

        # loading reachable sets
        self.FRS_tensor_path = os.path.join(robomimic.__path__[0], "../safediffusion_d4rl/pointmaze/reachability/frs_tensor_saved/frs_tensor_mat.mat")
        self.FRS_tensor, self.FRS_info = self.preload_frs(dtype = self.dtype, device=self.device)
        self.combs = self.generate_combinations_upto(200)

        # For optimization
        self.opt_dim = [2, 3]

        # useful var
        self.state_key = "flat"

    # ------------------------------------------------------------------------------------------- #
    # --------------------- Define the Planning Model ------------------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    def get_pstate_from_obs(self, obs_dict):
        """
        Define a map from the observation dictionary to the planning state

        For Maze2D, the planning state is the position of the robot
        """
        p = obs_dict[self.state_key][:2]
        
        return p
    
    def get_pparam_from_obs_and_optvar(self, obs_dict, k_opt):
        """
        Define a map from the observation dictionary and the optimized trajectory parameter to the trajectory parameter

        For Maze2D, the trajectory parameter is the initial velocity and acceleration of the robot
        """
        v0 = obs_dict[self.state_key][2:]
        param = torch.hstack([v0, k_opt])
        
        return param
    
    def model(self, t, p0, param):
        """
        Get x(t; x0, k), given the initial state x0, trajectory parameter k.

        Args:
            t :    (N,),   time
            p0:    (B, 2), initial planning state, (p_x, p_y)
            param: (B, 4), initial planning param, (k_vx, k_vy, k_ax, k_ay)
        
        Returns:
            x:  (B, N, 2), desired trajectory
            dx: (B, N, 2), desired velocity

        NOTE: This model should be consistent with the model defined at MATLAB CORA
        """
        assert max(t) <= self.t_f

        if p0.ndim == 1: p0 = p0[None, :]
        if param.ndim == 1: param = param[None, :]

        B = p0.shape[0]
        N = t.shape[0]

        x  = torch.zeros((B, N, self.n_state))
        dx = torch.zeros((B, N, self.n_state))

        # parse parameter
        k_v = param[:, [self.param_dict["k_vx"], self.param_dict["k_vy"]]]
        k_a = param[:, [self.param_dict["k_ax"], self.param_dict["k_ay"]]]

        t1 = self.time_pieces[0]

        x1, dx1 = traj_uniform_acc(t[t<=t1], p0, k_v, k_a)
        x2, dx2 = traj_uniform_acc(t[t>t1]-t1,  x1[:, -1, :], dx1[:, -1, :], -dx1[:, -1, :]/(self.t_f-t1))

        x[:, t<=t1, :], dx[:, t<=t1, :] = (x1, dx1)
        x[:, t >t1, :], dx[:, t >t1, :] = (x2, dx2)

        return x, dx
    
    # ------------------------------------------------------------------------------------------- #
    # --------------------------------- Trajectory Optimization --------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    def _prepare_problem_data(self, obs_dict, goal_dict):
        """
        TODO: Setup the useful variables for the trajectory optimization
        """
        num_robot_zonotope    = len(obs_dict["zonotope"]["robot"])
        num_obstacle_zonotope = len(obs_dict["zonotope"]["obstacle"])

        # prepare the problem data
        problem_data = {}

        # prepare the meta-data
        problem_data["meta"] = {}
        problem_data["constraint"] = {}
        problem_data["objective"]  = {}

        problem_data["meta"]["n_optvar"]      = len(self.opt_dim)
        problem_data["meta"]["n_constraint"]  = 0
        problem_data["meta"]["optvar_0"]      = self.to_tensor(0*torch.ones(problem_data["meta"]["n_optvar"])) # initial point
        problem_data["meta"]["state"]         = self.to_tensor(obs_dict[self.state_key])


        # prepare the data for constraints
        
        # --------------------------------------------------------------------------------- #
        # ------------------(C-1) Collision Constraints ----------------------------------- #
        # --------------------------------------------------------------------------------- #
        obstacles  = obs_dict["zonotope"]["obstacle"]
        n_interval = self.FRS_info["n_reachable_set"]

        # (C-1-1) Get the forward occupancy of the robot given the current position and velocity
        # NOTE: position is already included in robot zonotope)
        # NOTE: test this function using test_forward_occupancy(self, FO, pos_0, vel_0, acc_0)
        FO = self.forward_occupancy(robot_zonotope=obs_dict["zonotope"]["robot"], vel=obs_dict[self.state_key][2:])
        A  = np.zeros((num_robot_zonotope, num_obstacle_zonotope), dtype=object)
        b  = np.zeros((num_robot_zonotope, num_obstacle_zonotope), dtype=object)

        # (C-1-2) Buffer the obstacle zonotope with the non-optvar generators
        # NOTE: deleteZerosGenerators() is used but not sure if it actually works
        for (i, obstacle) in enumerate(obstacles):
            obstacle = obstacle.project().deleteZerosGenerators()
            obs_Z    = einops.repeat(obstacle.Z, 'm n -> repeat m n', repeat=n_interval)
            A_o, b_o = batchZonotope(torch.cat((obs_Z, FO.Grest),-2)).polytope(self.combs) # A: n_timesteps,*,dimension  
            A[0, i]  = A_o.cpu()
            b[0, i]  = b_o.cpu()

        problem_data["constraint"]["collision"]       = {}
        problem_data["constraint"]["collision"]["A"]  = A
        problem_data["constraint"]["collision"]["b"]  = b
        problem_data["constraint"]["collision"]["FO"] = [FO]

        problem_data["meta"]["n_constraint"] += num_robot_zonotope*num_obstacle_zonotope*n_interval
        # --------------------------------------------------------------------------------- #

        # --------------------------------------------------------------------------------- #
        # ------------------(C-2) Velocity Constraints ------------------------------------ #
        # --------------------------------------------------------------------------------- #
        v_dim = [self.param_dict["k_vx"], self.param_dict["k_vy"]]
        problem_data["constraint"]["velocity"]          = {}
        problem_data["constraint"]["velocity"]["v_max"] = self.FRS_info["delta_k"][v_dim]
        problem_data["constraint"]["velocity"]["v_min"] = -self.FRS_info["delta_k"][v_dim]
            
        problem_data["meta"]["n_constraint"] += 2 * len(v_dim)


        # -------------------------------------------------------------------------------- #
        # ------------------(J-1) Objective ---------------------------------------------- #
        #--------------------------------------------------------------------------------- #
        # TODO: Implement the set-goal objective function
        problem_data["objective"]["goal"] = {}
        problem_data["objective"]["goal"]["target"] = self.to_tensor(goal_dict[self.state_key][0:2])

        # problem_data["objective"]["projection"] = {}
        # problem_data["objective"]["projection"]["desired_plan"] = obs_dict["plan"]

        return problem_data

    def _compute_constraints(self, k, problem_data):
        """
        Get C(k; problem_data) <= 0

        Args
            k: (n_opt_var, ), trajectory parameter, numpy array

        NOTE: the k is the normalized one
        """
        assert k.shape[0] == len(self.opt_dim) and k.ndim == 1

        meta_data = problem_data["meta"]
        M  = meta_data["n_constraint"]
        n  = meta_data["n_optvar"]
        x0 = meta_data["state"]
        T  = self.FRS_info["n_reachable_set"]
        

        # ------------------(C-0) Initial Constraints --------------------- #
        Cons = torch.zeros(M, dtype=self.dtype)
        Jac  = torch.zeros(M, n, dtype=self.dtype)

        k    = k*self.FRS_info["delta_k"][self.opt_dim]+self.FRS_info["c_k"][self.opt_dim]
        beta = (k-self.FRS_info["c_k"][self.opt_dim])/self.FRS_info["delta_k"][self.opt_dim]
        beta = einops.repeat(beta, 'n -> repeat n', repeat=T)

        # ------------------(C-1) Collision Constraints --------------------- #
        collision_data = problem_data["constraint"]["collision"]

        # collision constraints
        A  = collision_data["A"]                                        # (200, n_constraint, n_workspace)
        b  = collision_data["b"]                                        # (200, n_constraint, 1)
        FO = collision_data["FO"]
        ndim = FO[0].dimension                                          # dimension of the workspace

        (n_robot, n_obstacle) = A.shape
        for j in range(n_robot):
            c_k      = FO[j].center_slice_all_dep(beta).unsqueeze(-1)   # (200, n_workspace, 1)
            grad_c_k = FO[j].grad_center_slice_all_dep(beta)            # (200, n_workspace, n_optvar)
            for o in range(n_obstacle):
                # max(Ax - b) <= 0
                h_obs     = (A[j][o]@c_k).squeeze(-1) - b[j][o]         # (200, n_constraint)
                h_obs     = h_obs.nan_to_num(-torch.inf)                # (we need to handle nan values to be meaningless)
                cons, ind = torch.max(h_obs, -1)                        # (200, 1)
                
                # gradient of the constraint
                A_max     = A[j][o].gather(-2, ind.reshape(T, 1, 1).repeat(1, 1, ndim))
                grad_cons = (A_max@grad_c_k).squeeze(-2)

                Cons[(j+n_robot*o)*T:(j+n_robot*o+1)*T] = -cons
                Jac[(j+n_robot*o)*T: (j+n_robot*o+1)*T] = -grad_cons

        # ------------------(C-2) Velocity Constraints --------------------- #
        # x0 + v0*t <= v_ub, x0 + v0*t >= v_lb
        data = problem_data["constraint"]["velocity"]

        v0    = torch.tensor(x0[2:], dtype=self.dtype, device=self.device)
        v_max = data["v_max"]
        v_min = data["v_min"]
        t_pk  = self.time_pieces[0]

        vpk              = v0 + t_pk * k
        jac_vpk_to_k    = t_pk * torch.eye(2)
        jac_k_to_beta   = torch.diag(self.FRS_info["delta_k"][self.opt_dim])
        jac_vpk_to_beta = jac_vpk_to_k @ jac_k_to_beta

        cons = torch.hstack((vpk - v_max, v_min - vpk))
        jac  = torch.vstack((jac_vpk_to_beta, -jac_vpk_to_beta))

        Cons[(n_robot*n_obstacle)*T:] = cons
        Jac[(n_robot*n_obstacle)*T:]  = jac
        
        return (Cons, Jac)
    
    def process_observation_and_goal_for_TO(self, obs_dict, goal_dict=None):
        """
        Process the observation and goal for the trajectory optimization
        """
        obs_dict = deepcopy(obs_dict)
        obs_dict[self.state_key] = self.to_tensor(obs_dict[self.state_key])
        obs_dict[self.state_key][:2] += self.offset

        if goal_dict is not None:
            goal_dict = deepcopy(goal_dict)
            goal_dict[self.state_key] = self.to_tensor(goal_dict[self.state_key])
            goal_dict[self.state_key][:2] += self.offset

        return obs_dict, goal_dict
    
    def postprocess_plan(self, plan):
        """
        The reference trajectory model is defined in the workspace.
        This function shifts the plan to the robot joint space
        """
        plan.x_des = plan.x_des - self.offset

        return plan

    def _compute_objective(self, k, problem_data):
        """
        Get J(k; problem_data)

        Returns:
            Obj: objective function value
            Grad: gradient of the objective function
        """
        assert k.shape[0] == len(self.opt_dim) and k.ndim == 1
        k    = k*self.FRS_info["delta_k"][self.opt_dim]+self.FRS_info["c_k"][self.opt_dim]

        meta_data = problem_data["meta"]
        x0 = meta_data["state"]
        pos0 = x0[:2]
        v0   = x0[2:]
        t_pk = self.time_pieces[0]
        k = self.to_tensor(k)
        

        data = problem_data["objective"]["goal"]
        goal = data["target"]


        # goal objective
        pos              = pos0 + v0*t_pk + 1/2*k*t_pk**2            # accelerating
        pos              = pos  + (v0 + k*t_pk)*(self.t_f-t_pk)/2    # braking

        grad_pos_to_k    = (1/2*t_pk**2 + t_pk*(self.t_f-t_pk)/2) * torch.eye(2)
        grad_k_to_beta   = torch.diag(self.FRS_info["delta_k"][self.opt_dim])
        grad_pos_to_beta = grad_pos_to_k @ grad_k_to_beta 

        cost_goal              = torch.sum((pos - goal)**2)
        grad_cost_goal_to_beta = 2*(pos - goal) @ grad_pos_to_beta      # (1, n_optvar)


        # weighting
        cost = cost_goal
        grad = grad_cost_goal_to_beta

        # detaching
        Obj = cost.cpu().numpy()
        Grad = grad.cpu().numpy()

        return (Obj, Grad)

    # --------------------- Reachability Helper ----------------------- #
    def preload_frs(self, dtype=torch.float, device=torch.device('cpu')):
        """
        Load the forward-reachable-set pre-computed using CORA toolbox.

        It also checks if the FRS is consistent with the planning model specification.

        Returns
            FRS_tensor: (n_reachable_set, n_generators, n_dim), n_dim = n_state + n_param + n_time(1)
            FRS_info: Assumes that the trajectory parameter resides in the `box`
                dt: time step of the reachability analysis
                c_k: center of the box
                delta_k: half size of the box
        """
        FRS_tensor = []
        FRS_tensor_load = loadmat(self.FRS_tensor_path)

        # assert if the reachability setting is consistent with the planning model specification
        assert(FRS_tensor_load["t_total"].squeeze() == self.t_f)
        assert(FRS_tensor_load["t_plan"].squeeze() == self.time_pieces[0])

        # This assumes that the domain of trajectory parameter is `box`-zonotope
        c_k      = torch.zeros(self.n_param,)
        c_k[[self.param_dict["k_vx"], self.param_dict["k_vy"]]] = torch.tensor(FRS_tensor_load["c_kv"].squeeze(), 
                                                                               dtype=dtype, device=device)
        c_k[[self.param_dict["k_ax"], self.param_dict["k_ay"]]] = torch.tensor(FRS_tensor_load["c_ka"].squeeze(), 
                                                                               dtype=dtype, device=device)

        delta_k  = torch.zeros(self.n_param,)
        delta_k[[self.param_dict["k_vx"], self.param_dict["k_vy"]]] = torch.tensor(FRS_tensor_load["delta_kv"].squeeze(),
                                                                                    device=device, dtype=dtype)
        delta_k[[self.param_dict["k_ax"], self.param_dict["k_ay"]]] = torch.tensor(FRS_tensor_load["delta_ka"].squeeze(),
                                                                                    device=device, dtype=dtype)
    
        FRS_tensor = torch.tensor(FRS_tensor_load["FRS_tensor"], dtype=dtype, device=device)

        # save the configuration for reachability computation
        FRS_info = {}
        FRS_info["dt"]      = torch.tensor(FRS_tensor_load["dt"].squeeze(), device=device, dtype=dtype)
        FRS_info["c_k"]     = c_k.to(device, dtype)
        FRS_info["delta_k"] = delta_k.to(device, dtype)
        FRS_info["n_reachable_set"] = FRS_tensor.shape[0]

        return FRS_tensor, FRS_info
    
    def forward_occupancy(self, robot_zonotope, vel):
        """
        Compute the forward occupancy at R^2 given the current position and velocity

        Args
            robot_zonotope: (n_body,) zonotopes of R^2
            vel: (2,) initial velocity

        Returns
            FRS: (n_body, n_timestep, n_c + n_g, 2) polynomial zonotope, that is k_ax, k_ay-sliceable
        """
        # Shift the FRS to the current position (pos)
        FRS_batch_zono = batchZonotope(self.FRS_tensor)

        # Slice with respect to the current velocity (vel)
        slice_dim = [self.n_state + self.param_dict["k_vx"], self.n_state + self.param_dict["k_vy"]]
        slice_pt  = vel.tolist()
        FRS_batch_zono = FRS_batch_zono.slice(slice_dim, slice_pt)

        # Make it to the polyzonotope, just to manage the id
        id_dim = [self.n_state + self.param_dict["k_ax"], self.n_state + self.param_dict["k_ay"]]
        PZ_FRS = FRS_batch_zono.deleteZerosGenerators(sorted=True).to_polyZonotope_multi_dim(id_dim, "k_a")

        # now we can drop the dimension we do not need
        PZ_FRS_2D         = PZ_FRS.project([0, 1])
        PZ_FRS_2D         = PZ_FRS_2D.deleteZerosGenerators()
        robot_zonotope_2D = robot_zonotope[0].project([0, 1])
        robot_zonotope_2D = robot_zonotope_2D.deleteZerosGenerators()
        
        PZ_FRS_2D         = PZ_FRS_2D + robot_zonotope_2D
        PZ_FRS_2D         = PZ_FRS_2D.deleteZerosGenerators()

        return PZ_FRS_2D
    
    # ------------------------------------- Helper Functions ------------------------------------- #
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
        robot_zonotope = obs["zonotope"]["robot"][0].project([0, 1])
        assert(torch.allclose(robot_zonotope.center, plan[0][1]+self.offset), "The plan is not consistent with the current position")

        obstacle_zonos = [zono.project([0, 1]) for zono in obs["zonotope"]["obstacle"]]

        # checks if the robot at each time step is in collision with the obstacles
        for i in range(len(plan)):
            robot_zonotope_i = robot_zonotope + plan[i][1] - plan[0][1]
            for obstacle_zono in obstacle_zonos:
                buff = obstacle_zono - robot_zonotope_i
                _, b = buff.polytope(self.combs)
                unsafe = b.min(dim=-1)[0] > 1e-6
                if unsafe:
                    return True

        return False

    def get_FRS_from_obs_and_optvar(self, obs_dict, k_opt):
        """
        Get the FRS given the optimized trajectory parameter
        """
        FRS = self.forward_occupancy(obs_dict["zonotope"]["robot"], obs_dict[self.state_key][2:])
        k_opt = k_opt/self.FRS_info["delta_k"][self.opt_dim]
        k_opt = einops.repeat(k_opt, 'n -> repeat n', repeat=FRS.Z.shape[0])
        FRS = FRS.slice_all_dep(k_opt)
        
        return FRS
        
    def generate_combinations_upto(self, max_combs):
        return [torch.combinations(torch.arange(i), 2) for i in range(max_combs+1)]
    
    def test_forward_occupancy(self, FO, pos_0, vel_0, acc_0):
        """
        Args
            FO: (B, n_g, 2)
        """
        init_state = torch.tensor(pos_0, dtype=self.dtype, device=self.device)
        param      = torch.cat([torch.tensor(vel_0), torch.tensor(acc_0)]).to(dtype=self.dtype, device=self.device)
        (x_des, dx_des) = self.model(self.t_des, init_state, param)

        ka = torch.tensor(acc_0, dtype=self.dtype, device=self.device)
        ka = ka/self.FRS_info["delta_k"][3:4]
        ka = einops.repeat(ka, 'n -> repeat n', repeat=FO.Z.shape[0])
        FO_k = FO.slice_all_dep(ka)

        self.render(x_des[0], FO_k)
    
    def render(self, plan, FO=None):
        """
        Render the plan

        Args:
            plan: (N_t, 2) torch array
            FO: (N_t, n_g, 2) zonotopes
        """

        # Plot t-p_t diagrams
        fig, axs = plt.subplots(self.n_state, 1)
        axs = axs.flatten()  # Flatten in case of more than one row
        
        for i, (k, v) in enumerate(self.state_dict.items()):
            ax = axs[i]
            ax.plot(self.t_des, plan[:, v], label=k)
            ax.legend()
            ax.grid(True)
            ax.set_title(k)

        if self.render_kwargs["render_offline"]:
            os.makedirs(self.render_kwargs["save_dir"], exist_ok=True)
            save_path = os.path.join(self.render_kwargs["save_dir"], 
                                     f"time_vs_plan.png")
            plt.savefig(save_path, format='png')
            plt.close(fig)
        

        # trajectory plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(plan[:, 0], plan[:, 1])
        ax.grid(True)
        
        if FO is not None:
            # static zonotopes
            # This assumes that static_obs has the outer walls of the environment
            patches_data = torch.vstack([zono.polyhedron_patch() for zono in FO])
            vertices = torch.vstack([patches_data.reshape(-1, 2)])
            max_V = vertices.cpu().numpy().max(axis=0)
            min_V = vertices.cpu().numpy().min(axis=0)

            ax.set_xlim([min_V[0] - 0.1, max_V[0] + 0.1])
            ax.set_ylim([min_V[1] - 0.1, max_V[1] + 0.1])

            patches = []
            for patch in patches_data:
                patches.append(ax.fill(patch[:, 0], patch[:, 1],
                                    edgecolor='blue', facecolor='blue', alpha=0.5, linewidth=0.1))
                
            fig.canvas.draw()

        if self.render_kwargs["render_offline"]:
            os.makedirs(self.render_kwargs["save_dir"], exist_ok=True)
            save_path = os.path.join(self.render_kwargs["save_dir"], 
                                     f"traj.png")
            plt.savefig(save_path, format='png')
            plt.close(fig)

        


        


