import os
import abc

import numpy as np
import cyipopt
import matplotlib.pyplot as plt

from safediffusion.algo.plan import ReferenceTrajectory

import torch

class TrajectoryOptimization(object):
    """
    Wrapper class of planner to render the optimization
    """
    def __init__(self, planner):
        assert isinstance(planner, ParameterizedPlanner)
        self.planner = planner
    
    def objective(self, x):
        """
        Objective function for the trajectory optimization
        """
        Obj, _ = self.planner.compute_objective(x)
        return Obj
    
    def gradient(self, x):
        """
        Gradient of the objective function
        """
        _, Grad = self.planner.compute_objective(x)
        return Grad
    
    def constraints(self, x):
        """
        Constraints for the trajectory optimization
        """
        Cons, _ = self.planner.compute_constraints(x)
        return Cons    
    
    def jacobian (self, x):
        """
        Jacobian of the constraints
        """
        _, Jac = self.planner.compute_constraints(x)
        return Jac

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """
        Intermediate callback function
        """
        print(f"Iteration {iter_count}: {obj_value}, Primal Feas {inf_pr}, Dual Feas {inf_du}")

class ParameterizedPlanner(abc.ABC):
    """
    The planner that plans the trajectory for the agent in the environment
    The plan should be time-parameterized.
    """
    def __init__(self, 
                 state_dict, 
                 param_dict, 
                 dt, 
                 t_f, 
                 device = "cpu",
                 dtype  = np.float64,
                 **kwargs):

        self.state_dict = state_dict                             # state dictionary that maps the name to index
        self.param_dict = param_dict                             # param dictionary that maps the name to index
        
        # Timing
        self.dt       = dt                                       # planning time step (sec)
        self.t_f      = t_f                                      # time horizon (sec)
        self.t_des    = np.arange(0, self.t_f+self.dt, self.dt)  # planning time vector

        # Trajectory Optimization

        # directory to save the plan
        if "render_kwargs" in kwargs.keys():
            self.render_kwargs = kwargs["render_kwargs"]

    @abc.abstractmethod
    def model(self, t, init_state, param):
        """
        Get x(t; x0, k), given the initial state x0, trajectory parameter k.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def compute_constraints(self, x):
        """
        Prepare the constraints for the trajectory optimization
        
        x: (n_optvar,), flattened trajectory
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_objective(self, x):
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

    def __call__(self, obs_dict, goal_dict=None):
        """
        Get the reference trajectory

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            plan (ReferenceTrajectory): the reference trajectory
        """
        # Solve the trajectory optimization
        x0 = obs_dict["flat"]
        k_opt, flag = self.trajopt(obs_dict, goal_dict)

        # Wrap it to the reference trajectory
        (x_des, dx_des) = self.model(self.t_des, x0, k_opt)
        plan = ReferenceTrajectory(self.t_des, x_des, dx_des)

        return plan
    
    def trajopt(self, obs_dict, goal_dict):
        """
        Based on the observation and goal, solve the trajectory optimization problem

        1) prepare the problem data
        2) construct the wrapper
        3) solve the optimization
            k* = max J(k) s.t. C(k) < 0
        """
        problem_data  = self._prepare_problem_data(obs_dict, goal_dict)
        problem       = TrajectoryOptimization(self, problem_data)
        n_optvar      = problem.n_optvar
        n_constraints = problem.n_constraints

        nlp = cyipopt.Problem(
            n = n_optvar,
            m = n_constraints,
            problem_obj = problem,
            lb = [-1] * n_optvar,
            ub = [-1] * n_optvar,
            cl = [-1e20] * n_constraints,
            cu = [-1e-6] * n_constraints
        )

        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)
        nlp.add_option('max_wall_time', self.nlp_time_limit)

        k_opt, info = nlp.solve()
        # nlp.solve(k_init)

        k_opt = torch.tensor(k_opt, device=self.device, dtype=self.dtype)
        flag  = info["status"]

        return k_opt, flag

    def __repr__(self):
        print(f"Planner: {self.__class__.__name__}")
        for k, v in self.state_dict.items():
            print(f"State: {k} -> {v}")
        for k, v in self.param_dict.items():
            print(f"Param: {k} -> {v}")

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

    @property
    def n_state(self):
        return len(self.state_dict.keys())
    
    @property
    def n_param(self):
        return len(self.param_dict.keys())

    @property
    def n_timestep(self):
        return int(self.t_f / self.dt) + 1