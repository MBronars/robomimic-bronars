"""
Run and evaluate the diffusion policy on the PointMaze environment.
"""
import os

import numpy as np

import robomimic

import safediffusion_d4rl.pointmaze.utils as MazeUtils

from d4rl.pointmaze.maze_model import MEDIUM_MAZE_UNSAFE

# largemaze + diffusion policy
POLICY_PATH = os.path.join(robomimic.__path__[0],
                           "../diffusion_policy_trained_models/maze2d/large_diffpolicy_H32/models/model_epoch_1900.pth") # policy checkpoint

# medium_maze + diffusion policy
POLICY_PATH = os.path.join(robomimic.__path__[0], 
                           "../diffusion_policy_trained_models/maze2d_H128/20240704141057/models/model_epoch_700.pth") # policy checkpoint

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 
                           "../exps/safety_filter_large/safe_diffusion_maze.json")

if __name__ == "__main__":
    # ----------------------------------------- #
    ckpt_path        = POLICY_PATH
    config_json_path = CONFIG_PATH
    rollout_horizon  = 2000
    render_mode      = "rgb_array"
    seeds            = [32, 34, 41]
    x0               = np.array([3.2, 5.5, 0.1, 0.0])
    target_pos       = np.array([2.0, 7.0])
    # ----------------------------------------- #
    policy, env, config = MazeUtils.policy_and_env_from_checkpoint_and_config(
                                        ckpt_path, 
                                        config_json_path, 
                                        "diffusion",
                                        env_kwargs = {"maze_spec": MEDIUM_MAZE_UNSAFE},
                                        # policy_kwargs = {"horizon": {"action_horizon": 16}}
                                    )

    # rollout the policy in the environment using fixed initial state and the target position
    stats = MazeUtils.rollout(
                        policy      = policy,           # policy
                        env         = env,              # environment
                        horizon     = rollout_horizon,  # rollout horizon
                        render_mode = render_mode,      # render mode 
                        x0          = x0,               # initial state
                        target_pos  = target_pos,       # target position
                        save_dir    = config.safety.render.save_dir
    )

    # rollout the policy in the environment using random initial states
    stats = MazeUtils.rollout_random_seed(
                        policy      = policy,           # policy and environment
                        env         = env,              # experiment configuration
                        horizon     = rollout_horizon,  # rollout horizon
                        render_mode = render_mode,      # render mode
                        seeds       = seeds,            # random seeds
                        save_dir    = config.safety.render.save_dir
    )