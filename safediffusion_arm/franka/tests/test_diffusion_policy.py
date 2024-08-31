"""
Run and evaluate the safety filter on the PickPlace-Kinova environment.
"""
import os

import numpy as np

import robomimic

import safediffusion_arm.franka.utils as FrankaUtils
from safediffusion.utils.rand_utils import set_random_seed

POLICY_PATH = os.path.join(robomimic.__path__[0], 
                           "../diffusion_policy_trained_models/franka/model_epoch_400.pth") # delta-end-effector
CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           "../exps/base/config.json")

if __name__ == "__main__":
    # ----------------------------------------- #
    ckpt_path        = POLICY_PATH
    config_json_path = CONFIG_PATH
    rollout_horizon  = 1000
    render_mode      = "rgb_array"
    render_mode      = "zonotope"
    seed             = 35
    # ----------------------------------------- #
    # This seed is used to set the random seed for the environment and the policy, 
    # regardless of the random seed used for the rollout
    set_random_seed(42)

    policy, env, config = FrankaUtils.policy_and_env_from_checkpoint_and_config(
                                        ckpt_path, 
                                        config_json_path, 
                                        "diffusion",
                                    )

    # rollout the policy in the environment using random initial states
    stats = FrankaUtils.rollout_with_seed(
                        policy       = policy,           # policy and environment
                        env          = env,              # experiment configuration
                        horizon      = rollout_horizon,  # rollout horizon
                        render_mode  = render_mode,      # render mode
                        seed         = seed,
                        save_dir     = config.safety.render.save_dir,
                        camera_names = ["agentview", "frontview"]
    )