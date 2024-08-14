import numpy as np
import torch

from robomimic.algo import RolloutPolicy
from robomimic.algo.diffusers import DiffusionPolicyUNet
from robomimic.utils.obs_utils import unnormalize_dict

from safediffusion.algo.predictor import StatePredictor
from safediffusion.algo.plan import ReferenceTrajectory

class ArmEnvSimStatePredictor(StatePredictor):
    """
    This state predictor predicts state by actually interacting with the sim environment.
    """
    def __init__(self, env):
        self.env = env
        self.env.reset()

        state_dim  = self.env.get_state()["states"].shape[0]-1
        action_dim = self.env.action_dimension
        dt  = self.env.unwrapped_env.sim.model.opt.timestep

        super().__init__(state_dim = state_dim, action_dim = action_dim, dt = dt)

    def __call__(self, x0, actions):
        assert(actions.ndim == 3)
        assert(actions.shape[-1] == self.action_dim)
        assert(x0.shape[-1] == self.state_dim)

        B = actions.shape[0]
        T = actions.shape[1]

        plans = []

        for idx in range(B):
            x0_iter      = x0[idx]
            actions_iter = actions[idx]

            _ = self.env.reset_to({"states": np.concatenate([[0], x0_iter])})
            
            states_iter = [x0_iter]

            for action in actions_iter:
                obs, _, _, _ = self.env.step(action)
                states_iter.append(obs["flat"][-1, :])

            states_iter = np.stack(states_iter, 0)

            t_des  = torch.linspace(0, self.dt*T, T+1)
            x_des  = states_iter[:, :2]
            dx_des = states_iter[:, 2:]

            plan = ReferenceTrajectory(t_des=t_des, x_des=x_des, dx_des=dx_des, dtype=self.dtype, device=self.device)

            plans.append(plan)
        
        return plans