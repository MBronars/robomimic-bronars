"""
Test-out the functionality of the zonotope world
"""
import os
import json
import numpy as np
import imageio

from robomimic.config import config_factory
from robomimic.envs.env_gym import EnvGym

from safediffusion_d4rl.environments.safe_maze_env import SafeMazeEnv


def test_move(env, action, render_mode=None):
    obs = env.reset()

    state = np.array([1, 2, -1, -1])
    obs = env.set_state(qpos=state[:2], qvel=state[2:])

    video_path = os.path.join(os.path.dirname(__file__), "exps/render_test/")
    os.makedirs(video_path, exist_ok=True)
    video_path = os.path.join(video_path, f"rollout_{render_mode}.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    for iter in range(60):
        next_obs, _, _, _ = env.step(action)
        if render_mode is not None:
            img = env.render(mode=render_mode, height=512, width=512)
            video_writer.append_data(img)
        
    video_writer.close()

def test_safety_env(env_safety):    
    print(env_safety.name)
    obs = env_safety.reset()
    print(obs)

    # img = env_safety.render(mode='rgb_array', height=512, width=512)

    test_move(env_safety, action = [-0.2, 1], render_mode = "janner")


if __name__ == "__main__":
    env_name = "maze2d-large-dense-v1"
    
    config_name = os.path.join(os.path.dirname(__file__), "exps/render_test/safe_diffusion_maze.json")
    ext_cfg = json.load(open(config_name, 'r'))
    config = config_factory("safediffusion")
    with config.values_unlocked():
        config.update(ext_cfg)


    env_robomimic = EnvGym(env_name)
    env_safety = SafeMazeEnv(env_robomimic, **config.safety)

    test_safety_env(env_safety)
    



