import numpy as np

from safediffusion.algo.planner_base import ParameterizedPlanner
from safediffusion.algo.helper import traj_uniform_acc

class Simple2DPlanner(ParameterizedPlanner):
    def __init__(self, **kwargs):
        """
        Tests out the ARMTD-style planner
        """
        # TODO: Make this as an abstract function
        state_dict = {"x": 0, "y": 1}
        param_dict = {"k_vx": 0, "k_ax": 1, "k_vy": 2, "k_ay": 3}
        dt = 0.1
        t_f = 2.0
        
        ParameterizedPlanner.__init__(self, state_dict, param_dict, dt, t_f, **kwargs)

        # Piecewise trajectory design
        self.time_pieces = [1.0]
        assert min(self.time_pieces) > 0
        assert max(self.time_pieces) < t_f

    def _prepare_problem_data(self, obs_dict, goal_dict):
        """
        TODO: Setup the useful variables for the trajectory optimization
        """
        pass

    def compute_constraints(self, k):
        """
        Get C(k; problem_data)
        """
        pass

    def compute_objective(self, k):
        """
        Get J(k; problem_data)
        """
        pass
    
    def model(self, t, x_0, param):
        """
        Get x(t; x0, k), given the initial state x0, trajectory parameter k.

        Args:
            t :    (N,),   time
            x0:    (B, 2), initial state, (p_x, p_y)
            param: (B, 4), param, (k_vx, k_ax, k_vy, k_ay)
        
        Returns:
            x:  (B, N, 2), desired trajectory
            dx: (B, N, 2), desired velocity
        """
        assert max(t) <= self.t_f

        B = x_0.shape[0]
        N = t.shape[0]

        x  = np.zeros((B, N, self.n_state))
        dx = np.zeros((B, N, self.n_state))

        # parse parameter
        k_v = param[:, [self.param_dict["k_vx"], self.param_dict["k_vy"]]]
        k_a = param[:, [self.param_dict["k_ax"], self.param_dict["k_ay"]]]

        t1 = self.time_pieces[0]

        x1, dx1 = traj_uniform_acc(t[t<=t1], x_0, k_v, k_a)
        x2, dx2 = traj_uniform_acc(t[t>t1]-t1,  x1[:, -1, :], dx1[:, -1, :], -dx1[:, -1, :]/(self.t_f-t1))

        x[:, t<=t1, :], dx[:, t<=t1, :] = (x1, dx1)
        x[:, t >t1, :], dx[:, t >t1, :] = (x2, dx2)

        return x, dx