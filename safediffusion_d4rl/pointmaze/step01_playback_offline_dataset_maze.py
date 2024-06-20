"""
Playback an offline dataset for the maze2d-point environment and render the environment

The possible options are listed in `d4rl.pointmaze.__init__.py` that contains:
- maze2d-umaze-v1
- maze2d-medium-v1
- maze2d-large-v1
- maze2d-umaze-dense-v1
- maze2d-medium-dense-v1
- maze2d-large-dense-v1
"""
import os
import argparse
import imageio
import numpy as np

import gym
import d4rl # required for registering the environments
from d4rl.pointmaze import maze_model

def collision_check(env):
    """
    Check if the agent collides with the obstacles in the environment
    """
    obstacle_prefix = "wall"
    agent_geom_name = "particle_geom"

    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        geom1 = env.sim.model.geom_id2name(contact.geom1)
        geom2 = env.sim.model.geom_id2name(contact.geom2)

        if (geom1.startswith(obstacle_prefix) and geom2 == agent_geom_name) or \
           (geom2.startswith(obstacle_prefix) and geom1 == agent_geom_name):
           return True
    
    return False

def playback_d4rl_maze(env_name, render_onscreen=False, video_skip_frame=5):
    """
    Playback a offline dataset for the maze2d-point environment and render the environment

    Args
        env_name: str, the name of the environment
        render_onscreen: bool, whether to render the environment on screen
        video_skip_frame: int, the number of frames to skip when rendering the video
    """
    # create the environment
    env = gym.make(env_name)
    isinstance(env, maze_model.MazeEnv)

    # dummy reset
    env.reset()
    env.set_target()

    # get the offline dataset that is associated to the task
    # `robomimic` retrieves this dataset from the d4rl
    dataset = env.get_dataset() 

    n_timestep = dataset['observations'].shape[0]

    # initialize state and the goal
    init_state = dataset['observations'][0]
    env.set_state(init_state[0:2], init_state[2:])
    env.set_target(dataset['infos/goal'][0])

    # initialize the stats
    stats = dict(num_collision = 0)

    # initialize the renderer
    if not render_onscreen:
        base_folder = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(base_folder, exist_ok=True)
        video_path = os.path.join(base_folder, f"{env_name}_playback.mp4")
        video_writer = imageio.get_writer(video_path, fps=20)

    # playback the dataset
    for step in range(n_timestep):
        action = dataset['actions'][step]        
        env.step(action)

        if collision_check(env):
            stats['num_collision'] += 1

        if render_onscreen:
            env.render(mode="human")
        else:
            if step % video_skip_frame == 0:
                video_img = env.render(mode='rgb_array', height=512, width=512)
                video_writer.append_data(video_img)

        timeout = dataset['timeouts'][step]
        if timeout:
            env.set_target(dataset['infos/goal'][step+1])
            print("Next Episode, resetting the goal")
    
    video_writer.close()

    return stats

if __name__ == "__main__":
    """
    Example usage:
        env_name = "maze2d-large-dense-v1"
        render_onscreen = False    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_onscreen', action='store_true', help='Render trajectories')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    args = parser.parse_args()

    playback_d4rl_maze(args.env_name, render_onscreen=args.render_onscreen)