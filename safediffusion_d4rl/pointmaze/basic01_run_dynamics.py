"""
Test-out the tracking performance of the Maze2D environment.

NOTE: Please note that original d4rl.pointmaze has offset of [0.3, 0.3], making not consistent with the world model.
      The offset is reflected
"""
import os
import numpy as np

import imageio

import gym
import matplotlib.pyplot as plt

from safediffusion.utils.render_utils import Maze2dRenderer, plot_history_maze
from safediffusion_d4rl.planner.planner_base import ParameterizedPlanner
from safediffusion_d4rl.planner.helper import match_trajectories
from safediffusion_d4rl.planner.planner_maze import Simple2DPlanner

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

def test_plan(planner, x0, param, render=True):
    """
    Test the planner
    """
    plan = planner(x0, param, return_derivative=False)
    if render:
        planner.render(x0, param)

    return plan

def test_track(env, planner, x0, ka, render_kwargs):
    """
    Render the tracking of env.agent with the planner.
    The planner plans the trajectory and the agent tracks it.
    """
    # TODO: Implement the new renderer and evaluate tracking ability
    # TODO: Compare with diffusion policy timings
    assert isinstance(env, gym.Env)
    assert isinstance(planner, ParameterizedPlanner)
    assert x0.shape == env.observation_space.shape
    assert ka.shape[0] == planner.n_param - 2

    # Tracking 
    Kp = 30
    Kd = 0.5
    
    # Reset the environment
    env.reset()
    env.set_state(x0[:2], x0[2:])

    # Set the planning parameter
    param = np.zeros(planner.n_param,)
    param[planner.param_dict["k_vx"]] = x0[2]
    param[planner.param_dict["k_vy"]] = x0[3]
    param[planner.param_dict["k_ax"]] = ka[0]
    param[planner.param_dict["k_ay"]] = ka[1]

    (t_des, x_des) = planner(x0[:2], param)

    # set-up the savepath if the renderer mode is in ["human", "rgb_array"]
    planner.render(x0[:2], param)
    render_mode = render_kwargs["render_mode"]
    render_offline = render_kwargs["render_mode"]
    if render_offline:
        os.makedirs(render_kwargs["save_dir"], exist_ok=True)
        video_path = os.path.join(render_kwargs["save_dir"], f"rollout_{render_mode}_Kp{Kp}_Kd{Kd}.mp4")
        video_writer = imageio.get_writer(video_path, fps=20)

    # logging
    stat = {}
    stamps  = []
    rollout = [env.state_vector().copy()]
    actions = []
    terrors = []

    plan    = [x_des.copy()]
    total_reward = 0

    num_total_steps = int(max(t_des)/env.model.opt.timestep)
    # Execute the plan
    for step in range(num_total_steps):
        # Per simulation step
        
        # observe time 
        t = env.sim.data.time
        state = env.state_vector().copy()

        next_waypoint = match_trajectories(t, t_des, x_des)
        next_waypoint = next_waypoint[0].squeeze()

        # control policy to track the waypoint
        action = Kp*(next_waypoint[:2] - state[:2]) - Kd*state[2:4]
        next_observation, reward, terminal, _ = env.step(action)
        total_reward = total_reward + reward

        # log
        rollout.append(next_observation)
        stamps.append(t)
        actions.append(action)
        terrors.append(next_waypoint[:2] - state[:2])

        if collision_check(env):
            print(f"collision at time {t}")
            return

        if render_offline:
            if step % render_kwargs["video_skip_frame"] == 0:
                if render_mode == "rgb_array":
                    video_img = env.render(mode='rgb_array', height=512, width=512)
                if render_mode == "janner":
                    video_img = env.custom_renderer.renders(np.array(rollout), plans = np.array(plan).squeeze())
                video_writer.append_data(video_img)    

    if render_offline:
        video_writer.close()
    
    stat["rollout"] = np.array(rollout)
    stat["total_reward"] = total_reward
    stat["actions"] = np.array(actions)
    stat["stamps"] = np.array(stamps)
    stat["tracking_errors"] = np.array(terrors)

    return stat

if __name__ == "__main__":
    """
    Tests out the proper ka bound, tracking control policy
    """
    env_name = "maze2d-medium-v1"
    # x0    = np.array([1, 2, 5, 0.1])   # (px, py, vx, vy)
    # param = np.array([-7.5, -0.1])     # (ka_x, ka_y)

    x0    = np.array([1, 2, 2, 0.1])   # (px, py, vx, vy)
    param = np.array([-2.2, -0.5])     # (ka_x, ka_y)

    save_dir = os.path.join(os.path.dirname(__file__), "results", f"{env_name}_x{x0}_k{param}")

    render_kwargs = {
        "planner":{
            "save_dir": save_dir,
            "render_online": False,
            "render_offline": True
        },
        "env":{
            "save_dir": save_dir,
            "render_mode": "janner",                                         # rgb_array, janner
            "render_online": False,
            "render_offline": True,
            "video_skip_frame": 5
        }
    }
    #########################################################################################################

    # automated from here
    planner  = Simple2DPlanner(render_kwargs=render_kwargs["planner"])
    renderer = Maze2dRenderer(env_name)
    env = renderer.env

    if render_kwargs["env"]["render_mode"] == "janner":
        env.custom_renderer = renderer

    stat = test_track(env, planner, x0, param, render_kwargs["env"])

    plot_history_maze(stat, save_dir = save_dir)