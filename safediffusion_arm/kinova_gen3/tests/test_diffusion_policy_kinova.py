"""
Run and evaluate the safety filter on the PickPlace-Kinova environment.
"""
import os

import numpy as np

import robomimic

import safediffusion_arm.kinova_gen3.utils as KinovaUtils
from safediffusion.utils.rand_utils import set_random_seed

POLICY_PATH = os.path.join(robomimic.__path__[0], 
                           "../diffusion_policy_trained_models/kinova/model_epoch_600_joint.pth") # delta-end-effector
CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           "../exps/diffusion_policy/safediffusion_arm.json")

if __name__ == "__main__":
    # ----------------------------------------- #
    # Test-seed: [11, 35, 74]
    ckpt_path        = POLICY_PATH
    config_json_path = CONFIG_PATH
    rollout_horizon  = 1000
    render_mode      = "zonotope"
    seed             = 35
    # ----------------------------------------- #
    # This seed is used to set the random seed for the environment and the policy, 
    # regardless of the random seed used for the rollout
    set_random_seed(42)

    policy, env, config = KinovaUtils.policy_and_env_from_checkpoint_and_config(
                                        ckpt_path, 
                                        config_json_path, 
                                        "diffusion",
                                    )

    # rollout the policy in the environment using random initial states
    stats = KinovaUtils.rollout_with_seed(
                        policy       = policy,           # policy and environment
                        env          = env,              # experiment configuration
                        horizon      = rollout_horizon,  # rollout horizon
                        render_mode  = render_mode,      # render mode
                        seed         = seed,
                        save_dir     = config.safety.render.save_dir,
                        camera_names = ["agentview", "frontview"]
    )