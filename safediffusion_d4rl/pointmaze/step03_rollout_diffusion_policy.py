import argparse
import json
import h5py
import imageio
import numpy as np
import os
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.utils.file_utils import RESULT_DIR

from d4rl.pointmaze.maze_model import MEDIUM_MAZE_UNSAFE


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, camera_names=None):

    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()

    # env.reset_to_location
    # state_dict = env.get_state()

    # obs = env.reset_to(state_dict)

    goal = env.get_goal()

    # for i in range(200):
    #     action = np.random.randn(env.action_dimension) * 0.1
    #     obs, reward, done, info = env.step(action)
    #     env.render()

    results = {}
    video_count = 0
    total_reward = 0
    rollout = []
    try:
        for step_i in range(horizon):
            act = policy(ob=obs, goal=goal)
            next_obs, r, done, _ = env.step(act)
            total_reward += r
            success = env.is_success()["task"]

            rollout.append(next_obs["flat"])
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1
            
            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()
    
    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    rollout = np.array(rollout)

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    return stats


if __name__ == "__main__":
    ############# User Parameter ##############
    # rand_seeds = np.array(np.random.random(50)*100,dtype=int)
    rand_seeds = [42]
    ckpt_path = os.path.join(robomimic.__path__[0], "../diffusion_policy_trained_models/maze2d/20240620205734/models/model_epoch_1200.pth") # policy checkpoint
    rollout_horizon = 400
    model_timestep = 1e-3
    ###########################################

    # TODO: SEED does not work :(
    for rand_seed in rand_seeds:
        result_dir = os.path.join(os.path.dirname(__file__), f"eval/{rand_seed}")
        os.makedirs(result_dir, exist_ok=True)
        # Set random seed
        set_random_seed(rand_seed)
        # Set up device
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        # restore policy and environment from checkpoint
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

        # restore environment
        ckpt_dict["env_metadata"]["env_kwargs"]["maze_spec"] = MEDIUM_MAZE_UNSAFE
        env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=True, render_offscreen=False, verbose=True)
        env    = env.unwrapped

        video_writer = imageio.get_writer(f"{result_dir}/DPmujoco.mp4", fps=20)

        stats = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=False,
            video_writer=video_writer,
            video_skip=5,
            camera_names=["agentview"]
        )