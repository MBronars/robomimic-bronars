import numpy as np

from safediffusion_d4rl.planner.planner_base import ParameterizedPlanner
from safediffusion_d4rl.planner.helper import traj_uniform_acc

class Simple2DPlanner(ParameterizedPlanner):
    def __init__(self, **kwargs):
        """
        Tests out the ARMTD-style planner
        """
        state_dict = {"x": 0, "y": 1}
        param_dict = {"k_vx": 0, "k_ax": 1, "k_vy": 2, "k_ay": 3}
        dt = 0.1
        t_f = 2.0
        
        ParameterizedPlanner.__init__(self, state_dict, param_dict, dt, t_f, **kwargs)

        # Piecewise trajectory design
        self.time_pieces = [1.0]
        assert min(self.time_pieces) > 0
        assert max(self.time_pieces) < t_f
    
    def x_des(self, x0, param):
        # parse parameter
        k_v = param[[self.param_dict["k_vx"], self.param_dict["k_vy"]]]
        k_a = param[[self.param_dict["k_ax"], self.param_dict["k_ay"]]]

        # piecewise C1 trajectory
        t1 = self.time_pieces[0]
        t_des = np.arange(0, self.t_f+self.dt, self.dt)

        # Piece 1
        t_des_1 = t_des[t_des <= t1]
        x_des_1 = traj_uniform_acc(t_des_1, x0, k_v, k_a)

        # Piece 2
        t_des_2 = t_des[t_des > t1] - t1
        x_des_2 = traj_uniform_acc(t_des_2, x_des_1[:, -1], k_v+k_a*t1, -(k_v+k_a*t1)/(self.t_f-t1))

        return np.hstack([x_des_1, x_des_2])
    
    def dx_des(self, x0, param):
        # parse parameter
        k_v = param[[self.param_dict["k_vx"], self.param_dict["k_vy"]]]
        k_a = param[[self.param_dict["k_ax"], self.param_dict["k_ay"]]]

        # piecewise C1 trajectory
        t1 = self.time_pieces[0]
        t_des = np.arange(0, self.t_f+self.dt, self.dt)

        # Piece 1
        t_des_1 = t_des[t_des <= t1]
        dx_des_1 = np.expand_dims(k_v, axis=-1) + np.outer(k_a, t_des_1)

        # Piece 2
        t_des_2 = t_des[t_des > t1] - t1
        dx_des_2 = np.expand_dims(dx_des_1[:, -1], axis=-1) + np.outer(dx_des_1[:, -1]/(self.t_f-t1), t_des_2)

        return np.hstack([dx_des_1, dx_des_2])