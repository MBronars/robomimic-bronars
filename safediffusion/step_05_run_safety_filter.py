"""
Test Receding Horizon Safety Filter for Blind ARMTD Policy

Author: Wonsuhk Jung
"""
import os
import time
from copy import deepcopy

import imageio
import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.environment.zonotope_env import ZonotopeMuJoCoEnv
from safediffusion.safety_filter.base import SafetyFilter
from safediffusion.utils.io_utils import RESULT_DIR

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
            act_unsafe = policy(ob=obs)
            actions    = policy.policy.action_sequence_ref.squeeze().detach().cpu().numpy()
            t_perf     = time.time() - t_perf
            
            if safety_filter is not None:
                t_safe = time.time()
                act    = safety_filter(actions)
                act    = act[0]
                t_safe = time.time() - t_safe
            
            # TODO: Hack: now we are using the unsafe action just to roll out the environment
            act = act_unsafe

            # Visualize the zonotope world
            if zono_video_writer is not None:
                if debug_video_count % video_skip == 0:
                    FRS_zonos = safety_filter.FO_zonos_sliced_at_param(safety_filter.ka_backup)
                    FO_desired = safety_filter.forward_occupancy_from_reference_traj(safety_filter.actions_to_reference_traj(actions), only_end_effector=True)
                    video_img = safety_filter.env.render(FRS_zonos=FRS_zonos, FO_desired_zonos=FO_desired)
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
    rand_seeds = [10, 13, 14, 38, 42, 68, 71, 126, 318]
    ckpt_path = os.path.join(os.path.dirname(__file__), "assets/model_epoch_600_joint.pth")
    rollout_horizon = 500
    ########################################################

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    for rand_seed in rand_seeds:
        set_random_seed(rand_seed)
        result_dir = f"{RESULT_DIR}/scenario_{rand_seed}"
        os.makedirs(result_dir, exist_ok=True)
        
        # diffusion policy configuration
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
        env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=True, render_offscreen=False, verbose=True)
        mujoco_video_writer = imageio.get_writer(f"{result_dir}/mujoco.mp4", fps=20)
        
        # safety filter configuration
        zonotope_twin_env = ZonotopeMuJoCoEnv(env.env, render_online=True, ticks=True)
        safety_filter = SafetyFilter(zono_env=zonotope_twin_env, n_head=1)
        zono_video_writer = imageio.get_writer(f"{result_dir}/zonotwin.mp4", fps=20)

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

