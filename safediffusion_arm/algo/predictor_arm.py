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

        state_dim  = self.env.get_state()["states"].shape[0] # this includes the time dimension
        action_dim = self.env.action_dimension
        dt         = 1/self.env.unwrapped_env.control_freq

        super().__init__(state_dim = state_dim, action_dim = action_dim, dt = dt)

    def __call__(self, x0, actions):
        """
        state_dict \in \R^79 = 1 + 2*13 + 28
            [time                  \R^1, 
             robot_joint_pos       \R^7,
             robot_gripper_qpos    \R^6
             object1_(qpos, qquat) \R^7
             object2_(qpos, qquat) \R^7
             object3_(qpos, qquat) \R^7
             object4_(qpos, qquat) \R^7
             robot_joint_vel       \R^7,
             robot_gripper_qvel    \R^6,
             ?                     \R^24
        ]

        obs
            object1_qpos \R^3
            object1_qquat(diff. convention) \R^4

        NOTE: the prediction is different from the realized value. The predicted value has about 0.01 (rad) error
        """
        assert(actions.ndim == 3)
        assert(actions.shape[-1] == self.action_dim)
        assert(x0.shape[-1] == self.state_dim)

        B = actions.shape[0]
        T = actions.shape[1]

        plans = []

        for idx in range(B):
            x0_iter      = x0[idx]
            actions_iter = actions[idx]

            # reset to the initial state
            obs = self.env.reset_to({"states": x0_iter})
            
            states_iter = [np.concatenate([obs["robot0_joint_pos"], obs["robot0_joint_vel"]])]
            n_link = obs["robot0_joint_pos"].shape[0]

            # simulate the environment
            for action in actions_iter:
                obs, _, _, _ = self.env.step(action)
                states_iter.append(np.concatenate([obs["robot0_joint_pos"], obs["robot0_joint_vel"]]))

            states_iter = np.stack(states_iter, 0)

            t_des  = torch.linspace(0, self.dt*T, T+1)
            x_des  = states_iter[:, :n_link]
            dx_des = states_iter[:, n_link:]

            plan = ReferenceTrajectory(t_des=t_des, x_des=x_des, dx_des=dx_des, dtype=self.dtype, device=self.device)

            plans.append(plan)
        
        return plans