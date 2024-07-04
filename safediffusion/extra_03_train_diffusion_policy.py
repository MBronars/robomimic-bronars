"""
Train diffusion policy on pointmaze environment.

Code mainly adapted from diffusers.
"""
import os
import argparse

import robomimic
import robomimic.scripts.train as rmtrain


def train_diffusers():
    """
    Train diffusers-style policy on pointmaze environment.
    """
    config_path = os.path.join(robomimic.__path__[0], "exps/safediffusion/diffusers_maze_norm.json")

    args = argparse.Namespace(config = config_path,
                              algo = None,
                              name = None,
                              output = None,
                              dataset = None,
                              debug = True,
                              auto_remove_exp = False)

    rmtrain.main(args)

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
    train_diffusers()

    # config_path = os.path.join(robomimic.__path__[0], "exps/safediffusion/diffusers_maze.json")
    # dataset_path = os.path.join(robomimic.__path__[0], "../datasets/can/ph/low_dim_v141.hdf5")


    # args = argparse.Namespace(config = config_path,
    #                           algo = None,
    #                           name = None,
    #                           dataset = dataset_path,
    #                           output = None,
    #                           debug = True,
    #                           auto_remove_exp = False)

    # rmtrain.main(args)