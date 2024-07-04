import os
import abc

from copy import deepcopy
import torch
import cyipopt
import matplotlib.pyplot as plt

from safediffusion.algo.plan import ReferenceTrajectory

import einops

class TrajectoryOptimization(object):
    """
    Wrapper class of planner to render the optimization
    """
    def __init__(self, planner, problem_data, verbose=False):
        assert isinstance(planner, ParameterizedPlanner)
        self.planner = planner
        self.problem_data = problem_data
        self.verbose = verbose

        required_keys = ["n_optvar", "n_constraint", "optvar_0"]

        assert all([key in problem_data["meta"].keys() for key in required_keys])
    
    def objective(self, x):
        """
        Objective function for the trajectory optimization
        """
        Obj, _ = self.planner.compute_objective(x, self.problem_data)
        return Obj
    
    def gradient(self, x):
        """
        Gradient of the objective function
        """
        _, Grad = self.planner.compute_objective(x, self.problem_data)
        return Grad
    
    def constraints(self, x):
        """
        Constraints for the trajectory optimization
        """
        Cons, _ = self.planner.compute_constraints(x, self.problem_data)
        return Cons    
    
    def jacobian (self, x):
        """
        Jacobian of the constraints
        """
        _, Jac = self.planner.compute_constraints(x, self.problem_data)
        return Jac

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """
        Intermediate callback function
        """
        if self.verbose:
            print(f"Iteration {iter_count}: {obj_value}, Primal Feas {inf_pr}, Dual Feas {inf_du}")

    @property
    def n_optvar(self):
        return self.problem_data["meta"]["n_optvar"]
    
    @property
    def n_constraint(self):
        return self.problem_data["meta"]["n_constraint"]
    
    @property
    def x_0(self):
        return self.problem_data["meta"]["optvar_0"]

class ParameterizedPlanner(abc.ABC):
    """
    The planner that plans the trajectory for the agent in the environment
    The plan should be time-parameterized.

    NOTE: this planner only uses torch, not numpy

    WORLD REPRESENTATION: (x, y, vx, vy) where (x, y) is NOT the robot qpos
    """
    def __init__(self, 
                 state_dict, 
                 param_dict, 
                 dt, 
                 t_f, 
                 device = torch.device('cpu'),
                 dtype  = torch.float,
                 nlp_time_limit = 1.0,
                 **kwargs):

        self.state_dict = state_dict                             # state dictionary that maps the name to index
        self.param_dict = param_dict                             # param dictionary that maps the name to index
        
        # Timing
        self.dt             = dt                                          # planning time step (sec)
        self.t_f            = t_f                                         # time horizon (sec)
        self.t_des          = torch.arange(0, self.t_f+self.dt, self.dt)  # planning time vector
        self.nlp_time_limit = nlp_time_limit        

        self.device   = device
        self.dtype    = dtype

        # print optimization log
        self.verbose = False

        # directory to save the plan
        if "render" in kwargs.keys():
            self.render_kwargs = kwargs["render"]

    @abc.abstractmethod
    def model(self, t, init_state, param):
        """
        Get x(t; p0, k), given the initial planning state p0, trajectory parameter k.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_pstate_from_obs(self, obs_dict):
        """
        Get the planning state from the observation dictionary
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_pparam_from_obs_and_optvar(self, obs_dict, k_opt):
        """
        Get the trajectory parameter from the observation dictionary
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _compute_constraints(self, x, problem_data):
        """
        Prepare the constraints for the trajectory optimization
        
        x: (n_optvar,), flattened trajectory
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_objective(self, x, problem_data):
        """
        Prepare the cost function for the trajectory optimization

        x: (n_optvar,), flattened trajectory
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _prepare_problem_data(self, obs_dict, goal_dict):
        """
        Setup the useful variables for the trajectory optimization
        """
        raise NotImplementedError
    
    @abc.abstractmethod
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
        raise NotImplementedError
    
    def postprocess_plan(self, plan):
        """
        Postprocess the plan (Optional)
        """
        return plan
    
    def __call__(self, obs_dict, goal_dict=None):
        """
        Get the reference trajectory

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            plan (ReferenceTrajectory): the reference trajectory in qpos, qvel
        """
        # Proces the observation dictionary and goal dictionary compatible to the backup policy
        obs_dict, goal_dict = self.process_observation_and_goal_for_TO(obs_dict, goal_dict)
        
        # Trajectory optimization
        k_opt, info = self.trajopt(obs_dict, goal_dict)
        
        # Wrap it to the reference trajectory
        plan = self.make_plan(obs_dict, k_opt)

        plan = self.postprocess_plan(plan)

        return plan, info

    def make_plan(self, obs_dict, k_opt):
        """
        Make the plan from the current observationa and optimized trajectory parameter

        NOTE: should be using self.model to get the plan
        """
        p0    = self.get_pstate_from_obs(obs_dict)
        param = self.get_pparam_from_obs_and_optvar(obs_dict, k_opt)
        (x_des, dx_des) = self.model(self.t_des, p0, param)

        plan = ReferenceTrajectory(self.t_des, x_des[0], dx_des[0], self.dtype, self.device)
        plan.stamp_trajectory_parameter(param)
        
        return plan
    
    def trajopt(self, obs_dict, goal_dict):
        """
        Based on the observation and goal, solve the trajectory optimization problem

        1) prepare the problem data
        2) construct the wrapper
        3) solve the optimization
            k* = max J(k) s.t. C(k) < 0

        Args:
            status (dict): status of the optimization problem
                flag: 
                0 -> Algorithm terminated successfully at a point satisfying the convergence tolerances

                2 -> Algorithm converged to a point of local infeasibility. Problem may be infeasible
        
        Returns:
            k_opt (torch.tensor): optimized trajectory parameter
        """
        problem_data  = self._prepare_problem_data(obs_dict, goal_dict)
        problem       = TrajectoryOptimization(self, problem_data, verbose=self.verbose)
        n_optvar      = problem.n_optvar
        n_constraint  = problem.n_constraint
        k_0           = problem.x_0

        nlp = cyipopt.Problem(
            n = n_optvar,
            m = n_constraint,
            problem_obj = problem,
            lb = [-1] * n_optvar,
            ub = [1] * n_optvar,
            cl = [-1e20] * n_constraint,
            cu = [-1e-6] * n_constraint
        )

        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)
        nlp.add_option('max_wall_time', self.nlp_time_limit)
        
        k_opt, info = nlp.solve(k_0.cpu().numpy())

        k_opt = torch.tensor(k_opt, device=self.device, dtype=self.dtype)
        k_opt = k_opt * self.FRS_info["delta_k"][self.opt_dim]

        if info["status"] != 0:
            self.disp(info["status_msg"])

        return k_opt, info

    def compute_objective(self, x, problem_data):
        """
        Prepare the cost function for the trajectory optimization

        NOTE: we need to implement try and except block since cyipopt does not handle exceptions

        x: (n_optvar,), flattened trajectory
        """
        try:
            x = self.to_tensor(x)
            return self._compute_objective(x, problem_data)
        except Exception as e:
            print("[Trajopt / compute_objective]: Error! ", e)
            raise NotImplementedError

    def compute_constraints(self, x, problem_data):
        """
        Prepare the constraints for the trajectory optimization

        NOTE: we need to implement try and except block since cyipopt does not handle exceptions
        
        x: (n_optvar,), flattened trajectory
        """
        try:
            x = self.to_tensor(x)
            return self._compute_constraints(x, problem_data)
        except Exception as e:
            print("[Trajopt / compute_constraints]: Error! ", e)
            raise Exception    

    def __repr__(self):
        print(f"Planner: {self.__class__.__name__}")
        for k, v in self.state_dict.items():
            print(f"State: {k} -> {v}")
        for k, v in self.param_dict.items():
            print(f"Param: {k} -> {v}")

    def to_tensor(self, x):
        if type(x) == torch.Tensor:
            return x
        else:
            return torch.tensor(x, dtype=self.dtype, device=self.device)

    def render(self, init_state, param):
        """
        Render the plan
        """
        t_des, x_des = self(init_state, param)
        
        fig, axs = plt.subplots(self.n_state, 1)
        axs = axs.flatten()  # Flatten in case of more than one row
        
        # Plot each item in its own subplot
        for i, (k, v) in enumerate(self.state_dict.items()):
            ax = axs[i]
            ax.plot(t_des, x_des[v, :], label=k)
            ax.legend()
            ax.grid(True)
            ax.set_title(k)

        if self.render_kwargs["render_online"]:
            plt.show()
        
        if self.render_kwargs["render_offline"]:
            os.makedirs(self.render_kwargs["save_dir"], exist_ok=True)
            save_path = os.path.join(self.render_kwargs["save_dir"], 
                                     f"plan_s{init_state}_k{param}.png")
            plt.savefig(save_path, format='png')
            plt.close(fig)

    def disp(self, msg):
        print("========================================")
        print(f"[{self.__class__.__name__}]: {msg}")
        print("========================================")

    @property
    def n_state(self):
        return len(self.state_dict.keys())
    
    @property
    def n_param(self):
        return len(self.param_dict.keys())

    @property
    def n_timestep(self):
        return int(self.t_f / self.dt) + 1