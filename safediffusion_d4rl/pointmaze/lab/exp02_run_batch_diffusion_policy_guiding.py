"""
Generate the multiple diffusion trajectories at once
"""

"""
Run and evaluate the diffusion policy on the PointMaze environment.
"""
import os
import json

import numpy as np
import imageio

import robomimic

from safediffusion.utils.rand_utils import set_random_seed

import safediffusion_d4rl.pointmaze.utils as MazeUtils

from d4rl.pointmaze.maze_model import MEDIUM_MAZE_UNSAFE

# medium_maze + diffusion policy
POLICY_PATH = os.path.join(robomimic.__path__[0], 
                           "../diffusion_policy_trained_models/maze2d_H128/20240704141057/models/model_epoch_1900.pth") # policy checkpoint

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 
                           "../exps/diffusion_policy_guiding/safe_diffusion_maze.json")

def rollout_random_seed(policy,
                        env,
                        horizon,
                        render_mode,
                        save_dir,
                        seeds,
                        video_skip = 1,
                        num_samples = 10):
    horizon_cheat = 10
    horizon_orig  = horizon

    stats = {}
    for i, seed in enumerate(seeds):
        set_random_seed(seed)

        obs  = env.reset()
        goal = env.get_goal()

        if seed not in [18, 19, 23, 24, 25, 26, 41, 45, 46, 47, 48, 49]:
            horizon = horizon_cheat
        else:
            horizon = horizon_orig

        # translate it to the world frame
        x0         = obs["flat"][-1, :]
        x0[:2]    += env.robotqpos_to_worldpos
        target_pos = goal["flat"][:2] + env.robotqpos_to_worldpos

        stat  = rollout(policy      = policy, 
                        env         = env, 
                        horizon     = horizon, 
                        render_mode = render_mode, 
                        x0          = x0, 
                        target_pos  = target_pos, 
                        save_dir    = save_dir + f"/seed{seed}",
                        video_skip  = video_skip,
                        num_samples = num_samples)
        
        print(f"Seed {i+1}/{len(seeds)}: {stat}")

        stats[str(seed)] = stat

    with open(os.path.join(save_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    return stats

def rollout(policy, 
            env, 
            horizon, 
            render_mode, 
            x0, 
            target_pos, 
            save_dir, 
            video_skip = 1,
            num_samples = 10):
    """
    Rollout the policy in the environment.

    Args:
        policy (Policy): policy to rollout
        env (Env): environment to rollout
        horizon (int): maximum rollout horizon
        render_mode (str): render mode in ["zonotope", "rgb_array", "janner"]
        x0 (np.ndarray): initial state (world frame)
        target_pos (np.ndarray): target position (world frame)
        save_dir (str): directory to save the rollout video
    
    Stats:
        Success (bool): whether the rollout is successful
        Horizon (int): number of steps taken
        Collision (int): number of collisions
        Intervention (int): number of interventions
    """
    obs  = env.reset() # dummyentry point
    obs  = env.set_state(pos = x0[:2], vel = x0[2:])
    goal = env.set_goal(pos = target_pos)

    policy.start_episode()

    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"rollout_x{x0}_g{target_pos}_n{env.name}_m{render_mode}.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    # logging variables
    num_intervention = 0
    num_unsafe = 0

    for step_i in range(horizon):
        if step_i % 10 == 0:
            print("====================================================================")
            print(f"Step {step_i}:  State {np.round(obs['flat'][-1], 2)}, Goal {np.round(goal['flat'][:2], 2)}")
            print("====================================================================")

        # ---------------------------------------------#
        act   = policy(ob=obs, goal=goal, num_samples=num_samples)

        next_obs, r, _, _ = env.step(act)
        success = env.is_success()["task"]
        is_safe = next_obs.pop("safe")
        is_intervened = policy.intervened
        stuck = policy.stuck

        if step_i % video_skip == 0:
            if step_i == 0 or not is_intervened:
                FRS = policy.backup_policy.FRS
            
            img = env.render(mode        = render_mode, 
                             height      = 512, 
                             width       = 512, 
                             plan        = policy.nominal_plan.x_des + env.robotqpos_to_worldpos, 
                             backup_plan = policy._backup_plan.x_des + env.robotqpos_to_worldpos,
                             plans       = np.vstack([v["plan"].x_des.unsqueeze(0) for v in policy.ranking_info.values()])+env.robotqpos_to_worldpos,
                             intervened  = policy.intervened,
                             FRS         = FRS)
            
            video_writer.append_data(img)
        
        # termination conditions
        if policy.stuck:
            print("Stuck!")
            break

        if success:
            print("Success!")
            break

        if not is_safe:
            num_unsafe += 1
            print("Safety violation!")
        
        if is_intervened:
            num_intervention += 1
        
        obs = next_obs
    
    stats = dict(
                 Success = success,
                 Horizon = (step_i + 1),
                 Collision = num_unsafe,
                 Intervention = num_intervention,
                 Stuck   = stuck
                 )
    
    with open(os.path.join(save_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    return stats


if __name__ == "__main__":
    # ----------------------------------------- #
    ckpt_path        = POLICY_PATH
    config_json_path = CONFIG_PATH
    rollout_horizon  = 1000
    render_mode      = "zonotope"
    scenarios        = [{"x0": np.array([4, 5, 0.1, 0]),   "target_pos": np.array([7.0, 2.0])},
                        {"x0": np.array([4, 5, 0.5, 0]),   "target_pos": np.array([7.0, 2.0])},
                        {"x0": np.array([5.5, 5, 0.5, 0]), "target_pos": np.array([6.0, 2.0])},
                        {"x0": np.array([7, 7, 0, 0]),     "target_pos": np.array([7.0, 2.0])},]
    prefix           = "multi"
    num_samples      = 10
    seeds            = np.arange(50)
    # ----------------------------------------- #
    set_random_seed(21)

    for scenario_idx in range(len(scenarios)):
        policy, env, config = MazeUtils.policy_and_env_from_checkpoint_and_config(
                                            ckpt_path, 
                                            config_json_path, 
                                            "safety_filter",
                                            env_kwargs = {"maze_spec": MEDIUM_MAZE_UNSAFE},
                                        )
        
        x0 = scenarios[scenario_idx]["x0"]
        target_pos = scenarios[scenario_idx]["target_pos"]

        # rollout the policy in the environment using fixed initial state and the target position
        stats = rollout(
                    policy      = policy,           # policy
                    env         = env,              # environment
                    horizon     = 10,  # rollout horizon
                    render_mode = render_mode,      # render mode 
                    x0          = x0,               # initial state
                    target_pos  = target_pos,       # target position
                    save_dir    = config.safety.render.save_dir + prefix + f"_scenario_{scenario_idx}", # save directory
                    num_samples = num_samples,      # number of samples to generate the multiple trajectories
                    video_skip  = 10,               # skip frames for the video
        )

    # rollout the policy in the environment using fixed initial state and the target position
    stats = rollout_random_seed(
                policy      = policy,           # policy
                env         = env,              # environment
                horizon     = rollout_horizon,  # rollout horizon
                render_mode = render_mode,      # render mode 
                seeds       = seeds,     # random seeds
                save_dir    = config.safety.render.save_dir + prefix, # save directory
                num_samples = num_samples,      # number of samples to generate the multiple trajectories
                video_skip  = 10,               # skip frames for the video
    )