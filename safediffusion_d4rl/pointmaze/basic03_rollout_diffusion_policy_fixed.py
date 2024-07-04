"""
Test the functionality of safety filter in the robomimic environment
"""
import os
import imageio
import numpy as np

import robomimic
from robomimic.algo import RolloutPolicy
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from d4rl.pointmaze.maze_model import MEDIUM_MAZE_UNSAFE

from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.utils.file_utils import load_config_from_json
from safediffusion.envs.env_safety import SafetyEnv
from safediffusion_d4rl.environments.safe_maze_env import SafeMazeEnv
from safediffusion.algo.safety_filter import SafetyFilter, SafeDiffusionPolicy

"""
NOTE: Using Diffusion Policy now
"""
POLICY_PATH = os.path.join(robomimic.__path__[0], "../diffusion_policy_trained_models/maze2d/20240701101051/models/model_epoch_1900.pth") # policy checkpoint
# POLICY_PATH = os.path.join(robomimic.__path__[0], "../diffusion_policy_trained_models/maze2d/20240701101051/models/model_epoch_500.pth") # diffuser checkpoint
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "exps/safety_filter/safe_diffusion_maze.json")

def rollout(policy, env, horizon, state, target_pos, **render_kwargs):
    assert isinstance(env, SafetyEnv)
    assert isinstance(policy, RolloutPolicy) or isinstance(policy, SafetyFilter)

    video_skip   = render_kwargs["video_skip"]
    render_mode  = render_kwargs["render_mode"]
    result_dir   = render_kwargs["result_dir"]

    # set fixed initial state
    obs   = env.reset()
    state = np.asarray(state)
    obs   = env.set_state(pos = state[:2], vel = state[2:])
    obs.pop("safe"); obs.pop("zonotope")

    # set fixed goal
    goal  = env.get_goal()
    goal['flat'][:2] = np.array(target_pos)
    goal['flat'][:2] = goal['flat'][:2] - np.array([1, 1]) # HACK
    
    # Initialize
    policy.start_episode()
    
    total_reward = 0

    video_path = os.path.join(result_dir, f"rollout_{env.name}_{render_mode}.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)
    try:
        for step_i in range(horizon):
            # obs["flat"][:, :2] = obs["flat"][:, :2] - np.array([1, 1]) # HACK
            act = policy(ob=obs, goal=goal)
            next_obs, r, done, _ = env.step(act)
            total_reward += r
            success = env.is_success()["task"]
            is_safe = next_obs.pop("safe")
            zonotope = next_obs.pop("zonotope")

            if step_i % video_skip == 0:
                img = env.render(mode=render_mode, height=512, width=512)
                video_writer.append_data(img)

            if done or success:
                break

            if not is_safe:
                print("Not Safe!")

            obs = next_obs
    
    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    return stats

if __name__ == "__main__":
    """
    Given the robomimic environment and the configuration file, 
    the safety wrapper is implemented and tested-out.
    """
    #-------------------------------------------------#
    #                   USER ARGUMENTS                #
    #------------------------------------------------ #
    ckpt_path        = POLICY_PATH
    config_json_path = CONFIG_PATH
    rollout_horizon  = 500
    Ta               = 8
    render_mode      = "zonotope"
    x0               = np.array([3, 6, -1, 1])
    target           = np.array([7, 6])
    #-------------------------------------------------#
    set_random_seed(42)
        
    device            = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    policy.policy.algo_config.horizon.unlock()
    policy.policy.algo_config.horizon.action_horizon = Ta
    policy.policy.algo_config.horizon.lock()


    # HACK
    ckpt_dict["env_metadata"]["env_kwargs"]["maze_spec"] = MEDIUM_MAZE_UNSAFE

    # Load the Safe Environment
    config = load_config_from_json(config_json_path)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=False, render_offscreen=True, verbose=True)
    env_safety = SafeMazeEnv(env, **config.safety)

    result_dir = os.path.join(os.path.dirname(__file__), f"eval/T{Ta}_x{x0}_g{target}")
    os.makedirs(result_dir, exist_ok=True)

    stats = rollout(
        policy=policy,
        env=env_safety,
        horizon=rollout_horizon,
        state = x0,
        target_pos = target,
        video_skip=5,
        render_mode = render_mode,
        result_dir = result_dir
    )