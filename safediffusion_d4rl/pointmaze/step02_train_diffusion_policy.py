"""
Train diffusion policy on pointmaze environment.

D4RL dataset: maze2d_umaze_v1s
"""
import os
import argparse

import robomimic
import robomimic.scripts.train as rmtrain


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
    config_path = os.path.join(robomimic.__path__[0], "exps/safediffusion/diffusion_policy_maze.json")
    # dataset_path = os.path.join(robomimic.__path__[0], "../datasets/d4rl/converted/maze2d_medium_dense_v1.hdf5")
    dataset_path = os.path.join(robomimic.__path__[0], "../datasets/d4rl/converted/antmaze_large_diverse_v2.hdf5")
    config_path = os.path.join(robomimic.__path__[0], "exps/safediffusion/diffusion_policy_antmaze.json")

    args = argparse.Namespace(config = config_path,
                              algo = None,
                              name = None,
                              dataset = dataset_path,
                              output = None,
                              debug = True,
                              auto_remove_exp = False)

    rmtrain.main(args)