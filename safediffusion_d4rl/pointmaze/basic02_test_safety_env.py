"""
Test-out the functionality of the safety wrapper of the robomimic environment
"""
import os
import json
import numpy as np
import imageio

from robomimic.config import config_factory
from robomimic.envs.env_gym import EnvGym

from safediffusion.utils.file_utils import load_config_from_json
from safediffusion_d4rl.environments.safe_maze_env import SafeMazeEnv


def test_move(env, x0, action, render_mode=None):
    if type(env) == EnvGym:
        obs = env.reset()
        state = dict()
        state["states"] = np.concatenate(([0], x0))
        obs = env.reset_to(state)
    else:
        obs = env.reset()
        state = np.array(x0)
        obs = env.set_state(pos=state[:2], vel=state[2:])

    video_path = os.path.join(os.path.dirname(__file__), f"exps/render_test2/")
    os.makedirs(video_path, exist_ok=True)
    video_path = os.path.join(video_path, f"rollout_{env.name}_{render_mode}.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    for iter in range(60):
        next_obs, _, _, _ = env.step(action)
        if render_mode is not None:
            img = env.render(mode=render_mode, height=512, width=512)
            video_writer.append_data(img)
        
    video_writer.close()

def test_safety_env(env_safety):    
    test_move(env_safety, x0 = [2, 2, -1, -1], action = [-0.2, 1], render_mode = "janner")
    test_move(env_safety, x0 = [2, 2, -1, -1], action = [-0.2, 1], render_mode = "zonotope")
    test_move(env_safety, x0 = [2, 2, -1, -1], action = [-0.2, 1], render_mode = "rgb_array")


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
    env_safety = SafeMazeEnv(env_robomimic, **config.safety)

    test_safety_env(env_safety)
    test_move(env_robomimic, x0 = [2, 2, -1, -1], action = [-0.2, 1], render_mode = "rgb_array")
    



