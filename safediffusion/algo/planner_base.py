import os

import numpy as np
import matplotlib.pyplot as plt

from safediffusion.algo.plan import ReferenceTrajectory


class ParameterizedPlanner(object):
    def __init__(self):
        pass

    def model(self, x, k):


class ParameterizedPlanner(object):
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

        # directory to save the plan
        if "render_kwargs" in kwargs.keys():
            self.render_kwargs = kwargs["render_kwargs"]

    def __call__(self, obs_dict, goal_dict=None):
        """
        Get the reference trajectory

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        """
        pass


    def model(self, init_state, param, return_derivative=False):
        """
        Get x(t; x0, k), given the initial state x0, trajectory parameter k.
        """
        assert(init_state.shape[-1] == self.n_state)
        assert(param.shape[-1] == self.n_param)

        if return_derivative:
            return (self.t_des, self.x_des(init_state, param), self.dx_des(init_state, param))
        else:
            return (self.t_des, self.x_des(init_state, param))

    
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