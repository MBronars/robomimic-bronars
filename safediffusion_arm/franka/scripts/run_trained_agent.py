"""
Train diffusion policy on pointmaze environment.

Code mainly adapted from diffusers.
"""
import os
import argparse

import robomimic
import robomimic.scripts.run_trained_agent as rmrun


if __name__ == "__main__":
    """
    Args:
        config: path to base json config, otherwise use default config
        algo: if provided, override the algorithm name defined in the config
        name: if provided, override the experiment name defined in the config
        dataset: if provided, override the dataset path defined in the config
        output: if provided, override the output folder path defined in the config
        auto-remove-exp: force delete the experiment folder if it exists
        debug: set this flag to run a quick training run for debugging purposes
    """
    agent_path = os.path.join(robomimic.__path__[0], "../diffusion_policy_trained_models/breadpickplace_act/first_trial/20240831133640/models/model_epoch_475.pth")

    args = argparse.Namespace(agent = agent_path,
                              n_rollouts = 27,
                              horizon = None,
                              env = None,
                              render = False,
                              video_path = None,
                              video_skip = 5,
                              camera_names = None,
                              dataset_path = None,
                              dataset_obs = False,
                              seed = None,
                              json_path = None,
                              error_path = None,
                              hz = None,
                              dp_eval_steps = None)

    rmrun.run_trained_agent(args)