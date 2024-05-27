"""
Test Receding Horizon Safety Filter for Blind ARMTD Policy

Author: Wonsuhk Jung
"""
import os
import time
from copy import deepcopy

import imageio
import numpy as np
import torch

from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as AcUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.environment.zonotope_env import ZonotopeMuJoCoEnv
from safediffusion.safety_filter.base import SafetyFilter
from safediffusion.utils.io_utils import RESULT_DIR

def overwrite_controller_to_joint_position(ckpt_dict):
    ckpt_dict['env_metadata']['env_kwargs']['controller_configs'] = {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": 50,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2
        }
    return ckpt_dict

def overwrite_policy_action_horizon(policy, n_head):
    assert isinstance(policy, RolloutPolicy)
    policy.policy.algo_config.horizon.unlock()
    policy.policy.algo_config.horizon.action_horizon = n_head
    policy.policy.algo_config.horizon.lock()

    return policy

def unnormalize(policy, ac):
    """
    Unnormalize the action tensor using the training normalization stats
        policy: (RolloutPolicy) policy
        ac: (1, D) tensor
    """
    ac = TensorUtils.to_numpy(ac)
    if policy.action_normalization_stats is not None:
        action_keys = policy.policy.global_config.train.action_keys
        action_shapes = {k: policy.action_normalization_stats[k]["offset"].shape[1:] for k in policy.action_normalization_stats}
        ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes, action_keys=action_keys)
        ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=policy.action_normalization_stats)
        action_config = policy.policy.global_config.train.action_config
        for key, value in ac_dict.items():
            this_format = action_config[key].get('format', None)
            if this_format == 'rot_6d':
                rot_6d = torch.from_numpy(value).unsqueeze(0)
                rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
                ac_dict[key] = rot
        ac = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)

    return ac


# TODO: 1) Analyze & imitate run_trained_agent.py
def rollout_with_safety_filter(policy, safety_filter, env, horizon, 
                               render=False, video_writer=None, zono_video_writer=None, 
                               video_skip=5, camera_names=None):
    """
    Args:
        policy: performance policy
        safety_filter: safety filter
        env: (robosuite environment) environment to interact with MuJoCo
        horizon: (int)
        render: (bool)
        video_writer: VideoWriter
        video_skip: int
        camera_names: list
    """
    assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(safety_filter, SafetyFilter)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state() # what does this return?
    obs = env.reset_to(state_dict)

    if safety_filter is not None:
        safety_filter.start_episode()
        safety_filter.env.render()

    results = {}
    results["t_perf"] = []
    results["t_safe"] = []
    video_count = 0
    debug_video_count = 0
    total_reward = 0
    try:
        for step_i in range(horizon):
            # The performance policy should return a sequence of the joint positions
            # Shape of (T, D)
            t_perf     = time.time()
            act = policy(ob=obs)
            actions    = np.vstack([unnormalize(policy, ac) for ac in policy.policy.get_predicted_action_sequence()]) # (B, Tp-To, D)
            t_perf     = time.time() - t_perf
            
            if safety_filter is not None:
                t_safe = time.time()
                # NOTE: START HERE
                act    = safety_filter(actions)
                act    = act.squeeze(0)
                t_safe = time.time() - t_safe

            # Visualize the zonotope world
            if zono_video_writer is not None:
                if debug_video_count % video_skip == 0:
                    FO_backup = safety_filter.forward_occupancy_from_reference_traj(safety_filter._plan_backup, only_end_effector=True)
                    FO_desired = safety_filter.forward_occupancy_from_reference_traj(safety_filter.actions_to_reference_traj(actions), only_end_effector=True)
                    video_img = safety_filter.env.render(FO_desired_zonos=FO_desired, FO_backup_zonos=FO_backup)
                    zono_video_writer.append_data(video_img)
                debug_video_count += 1

            next_obs, r, done, _ = env.step(act)
            total_reward += r
            success = env.is_success()["task"]

            # synchronize zonotope twin world with the environment
            safety_filter.sync_env()
            
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
            
            results["t_perf"].append(t_perf)
            results["t_safe"].append(t_safe)
            
            if done or success:
                break
            
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
    
    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
    
    return stats


if __name__ == "__main__":
    """
    User Arguments
        rand_seeds: scenario lists
        ckpt_path : path to the checkpoint of diffusion policy
        rollout_horizon: int
    """
    rand_seeds = [11, 63, 307, 363, 366, 408, 413]
    ckpt_path = os.path.join(os.path.dirname(__file__), "assets/model_epoch_300_joint_actions.pth")
    rollout_horizon = 500
    n_head = 1
    ##########################################################

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    for rand_seed in rand_seeds:
        set_random_seed(rand_seed)
        result_dir = f"{RESULT_DIR}/scenario_{rand_seed}"
        os.makedirs(result_dir, exist_ok=True)
        
        # diffusion policy configuration
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
        # HACK: overwrite the policy to use the joint position controller
        policy = overwrite_policy_action_horizon(policy, n_head)
        ckpt_dict = overwrite_controller_to_joint_position(ckpt_dict)

        # environment configuration
        env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=True, render_offscreen=False, verbose=True)
        mujoco_video_writer = imageio.get_writer(f"{result_dir}/SFmujoco.mp4", fps=20)
        
        # safety filter configuration
        zonotope_twin_env = ZonotopeMuJoCoEnv(env.env, render_online=True, ticks=True)
        safety_filter = SafetyFilter(zono_env=zonotope_twin_env, n_head=1, verbose=True)
        zono_video_writer = imageio.get_writer(f"{result_dir}/SFzonotwin.mp4", fps=20)

        # run simulation
        stats = rollout_with_safety_filter(
            policy=policy,
            safety_filter=safety_filter,
            env=env,
            horizon=rollout_horizon,
            render=False,
            video_writer=mujoco_video_writer,
            zono_video_writer = zono_video_writer,
            video_skip=5,
            camera_names=["agentview"]
        )

        print(stats)

