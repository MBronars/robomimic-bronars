"""
Testing-out the functionality of the backup policy
"""
import os
import json
import numpy as np
import imageio

from robomimic.config import config_factory
from robomimic.envs.env_gym import EnvGym


from safediffusion.utils.file_utils import load_config_from_json
from safediffusion_d4rl.environments.safe_maze_env import SafeMazeEnv
from safediffusion_d4rl.algo.planner_maze import Simple2DPlanner

def test(env, policy, x0, render_mode = None):
    obs = env.reset()
    state = np.array(x0)
    obs = env.set_state(qpos = state[:2], qvel = state[2:])
    goal = env.get_goal()

    video_path = os.path.join(os.path.dirname(__file__), f"exps/backup_test/")
    os.makedirs(video_path, exist_ok=True)
    video_path = os.path.join(video_path, f"rollout_{env.name}_{render_mode}.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    for iter in range(60):
        action = policy(obs, goal)
        next_obs, _, _, _ = env.step(action)
        if render_mode is not None:
            img = env.render(mode=render_mode, height=512, width=512)
            video_writer.append_data(img)

        obs = next_obs
        
    video_writer.close()


if __name__ == "__main__":
    """
    Given the robomimic environment and the configuration file, 
    the safety wrapper is implemented and tested-out.
    """
    #-------------------------------------------------#
    #                   USER ARGUMENTS                #
    #------------------------------------------------ #
    env_name = "maze2d-medium-dense-v1"
    config_json_path = os.path.join(os.path.dirname(__file__), "exps/render_test/safe_diffusion_maze.json")
    #-------------------------------------------------#

    config = load_config_from_json(config_json_path)

    env_robomimic = EnvGym(env_name)
    env_safety    = SafeMazeEnv(env_robomimic, **config.safety)
    policy        = Simple2DPlanner()

    test(env_safety, policy, x0 = [1, 2, -1, -1], render_mode = "janner")