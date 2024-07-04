"""
Testing-out the functionality of the backup policy
"""
import os
import json
import numpy as np
import imageio

from robomimic.config import config_factory
from robomimic.envs.env_gym import EnvGym

from safediffusion.envs.env_safety import SafetyEnv
from safediffusion.utils.file_utils import load_config_from_json
from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.algo.helper import match_trajectories

from safediffusion_d4rl.environments.safe_maze_env import SafeMazeEnv
from safediffusion_d4rl.algo.planner_maze import Simple2DPlanner

from d4rl.pointmaze.maze_model import MEDIUM_MAZE_UNSAFE

"""
NOTE: Seems like the Unknown Status Code error is attributed to the infeasibility of the problem.
It is analytically impossible to find the solution in this case.
"""

def test(env, policy, x0, target_pos, save_dir = None, render_mode = None):
    """
    A simple test function to test the tracking of the planner.    
    """

    assert(isinstance(env, SafetyEnv) and isinstance(policy, Simple2DPlanner))

    obs   = env.reset()
    state = np.array(x0)
    obs   = env.set_state(pos = state[:2], vel = state[2:])
    goal  = env.set_goal(pos = target_pos)

    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"rollout_x0{x0}_goal{target_pos}_{env.name}_{render_mode}.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    # Tracking
    (Kp, Kd) = (30, 0.5)
    n_step_per_plan = 30

    # 60 is planning timestep
    for _ in range(60):
        plan, _ = policy(obs, goal)

        t_plan_start = env.sim.data.time
        # HACK
        for i in range(n_step_per_plan):
            t = env.sim.data.time - t_plan_start
            state = obs["flat"]
            state[:2] = state[:2] + env.robotqpos_to_worldpos

            next_waypoint = match_trajectories(t, policy.t_des, np.transpose(plan.x_des))
            next_waypoint = next_waypoint[0].squeeze()

            # control policy to track the waypoint
            action = Kp*(next_waypoint[:2] - state[:2]) - Kd*state[2:4]
        
            next_obs, _, _, _ = env.step(action)
        
            if render_mode is not None:
                if t == 0:
                    pparam = plan.get_trajectory_parameter()
                    optvar = pparam[policy.opt_dim]
                    FRS = policy.get_FRS_from_obs_and_optvar(obs, optvar)
                    img = env.render(mode=render_mode, height=512, width=512, FRS=FRS, plan=plan.x_des)
                else:
                    img = env.render(mode=render_mode, height=512, width=512, plan=plan.x_des)
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
    env_name         = "maze2d-medium-dense-v1"
    config_json_path = os.path.join(os.path.dirname(__file__), "exps/backup_policy/safe_diffusion_maze.json")
    seed             = 42
    #-------------------------------------------------#
    set_random_seed(seed)

    config = load_config_from_json(config_json_path)

    env_robomimic = EnvGym(env_name, maze_spec=MEDIUM_MAZE_UNSAFE)
    env_safety    = SafeMazeEnv(env_robomimic, **config.safety)
    policy        = Simple2DPlanner(**config.safety)

    # SCENARIO 1
    test(env_safety, policy, 
         x0          = [3, 6, -1, 1], 
         target_pos  = [2, 7],
         save_dir    = config.safety.render.save_dir,
         render_mode = "zonotope")