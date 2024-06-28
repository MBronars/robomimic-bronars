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
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.utils.io_utils import RESULT_DIR

def overwrite_controller_to_joint_position(ckpt_dict):
    ckpt_dict['env_metadata']['env_kwargs']['controller_configs'] = {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": 150,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2
        }
    return ckpt_dict


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, camera_names=None):

    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    obs = env.reset_to(state_dict)

    # for i in range(200):
    #     action = np.random.randn(env.action_dimension) * 0.1
    #     obs, reward, done, info = env.step(action)
    #     env.render()

    results = {}
    video_count = 0
    total_reward = 0
    try:
        for step_i in range(horizon):
            act = policy(ob=obs)
            next_obs, r, done, _ = env.step(act)
            total_reward += r
            success = env.is_success()["task"]

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


    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    return stats


if __name__ == "__main__":
    ############# User Parameter ##############
    rand_seeds = np.array(np.random.random(50)*500,dtype=int)
    rand_seeds = [11, 63, 307, 363, 366, 408, 413]
    # ckpt_path = os.path.join(os.path.dirname(__file__), "assets/model_epoch_300.pth") # policy checkpoint
    # ckpt_path = os.path.join(os.path.dirname(__file__), "assets/model_epoch_600_joint.pth") # policy checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), "assets/model_epoch_300_joint_actions.pth") # policy checkpoint
    rollout_horizon = 200
    model_timestep = 1e-3
    ###########################################

    for rand_seed in rand_seeds:
        result_dir = f"{RESULT_DIR}/scenario_{rand_seed}"
        os.makedirs(result_dir, exist_ok=True)
        # Set random seed
        set_random_seed(rand_seed)
        # Set up device
        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        # restore policy and environment from checkpoint
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
        ckpt_dict = overwrite_controller_to_joint_position(ckpt_dict)
        ############ Change Environment ##############
        # change the PickPlace environment setting here (Refer to robosuite/PickPlace.py for more details)
        # ckpt_dict["env_metadata"]["env_kwargs"]["single_object_mode"] = 1
        # ckpt_dict["env_metadata"]["env_kwargs"]["render_camera"] = "robot0_eye_in_hand"
        ##############################################
        env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=True, render_offscreen=False, verbose=True)
        env.env.model_timestep=model_timestep

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


