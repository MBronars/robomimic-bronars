import os
import imageio
from copy import deepcopy

import numpy as np
import torch
import json

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo.algo import RolloutPolicy

from safediffusion.utils.file_utils import load_config_from_json
from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.envs.env_safety import SafetyEnv

from safediffusion_d4rl.environments.safe_maze_env import SafeMazeEnv
from safediffusion_d4rl.algo.safety_filter_maze import SafeDiffusionPolicyMaze, IdentityDiffusionPolicyMaze, SafeDiffuserMaze
from safediffusion_d4rl.algo.planner_maze import Simple2DPlanner
from safediffusion_d4rl.algo.predictor_maze import MazeEnvSimStatePredictor, DiffuserStatePredictor


from d4rl.pointmaze.maze_model import MEDIUM_MAZE_OPEN, LARGE_MAZE_OPEN, MEDIUM_MAZE

OPEN_MAZE_DICT = {
    "large" : LARGE_MAZE_OPEN,
    "medium": MEDIUM_MAZE
    # "medium": MEDIUM_MAZE_OPEN
}

# ------------------------------------------------ #
# --- Helper functions for running the policy --- #
# ------------------------------------------------ #
def get_FRS_from_backup_policy(policy, env, obs):
    """
    Get the FRS from the backup policy
    """
    obs_frs  = deepcopy(obs)

    backup_plan = policy._backup_plan

    # modify obs_frs["flat"]
    obs_frs["flat"] = torch.hstack([backup_plan.x_des[0], backup_plan.dx_des[0]])

    # modify
    # world_pos = backup_plan.x_des[0] + env.robotqpos_to_worldpos
    world_pos = backup_plan.x_des[0] + policy.backup_policy.offset
    obs_frs["zonotope"]["robot"][0].center = torch.hstack([world_pos, torch.tensor(0)])

    trajparam    = backup_plan.get_trajectory_parameter()
    optvar       = trajparam[policy.backup_policy.opt_dim]
    FRS          = policy.backup_policy.get_FRS_from_obs_and_optvar(obs_frs, optvar)
    
    return FRS


def rollout(policy, 
            env, 
            horizon, 
            render_mode, 
            x0, 
            target_pos, 
            save_dir, 
            video_skip = 5):
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
    assert isinstance(env, SafetyEnv)
    assert isinstance(policy, RolloutPolicy)

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

        act = policy(ob=obs, goal=goal)
        next_obs, r, _, _ = env.step(act)
        success = env.is_success()["task"]
        is_safe = next_obs.pop("safe")
        is_intervened = policy.intervened
        stuck = policy.stuck

        if step_i % video_skip == 0:
            if step_i == 0 or not is_intervened:
                # FRS = get_FRS_from_backup_policy(policy, env, obs)
                FRS = policy.backup_policy.FRS
            
            img = env.render(mode        = render_mode, 
                             height      = 512, 
                             width       = 512, 
                             plan        = policy.nominal_plan.x_des + env.robotqpos_to_worldpos, 
                             backup_plan = policy._backup_plan.x_des + env.robotqpos_to_worldpos,
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

def rollout_random_seed(policy,
                        env,
                        horizon,
                        render_mode,
                        seeds,
                        save_dir):
    """
    Simulate the policy in environment with the random seeds

    Args:
        policy (Policy): policy to rollout
        env (Env): environment to rollout
        horizon (int): maximum rollout horizon
        render_mode (str): render mode in ["zonotope", "rgb_array", "janner"]
        seeds (list): list of random seeds
        save_dir (str): directory to save the rollout video
    
    Returns:
        stats (list): list of rollout statistics
    """
    stats = []

    for i, seed in enumerate(seeds):
        set_random_seed(seed)

        obs  = env.reset()
        goal = env.get_goal()

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
                        save_dir    = save_dir + f"/seed{seed}")
        
        print(f"Seed {i+1}/{len(seeds)}: {stat}")

        stats.append(stat)
    
    return stats


def policy_and_env_from_checkpoint_and_config(ckpt_path, config_path, policy_type, env_kwargs=None, policy_kwargs=None):
    """
    Load the diffusion policy and the environment from the checkpoint path and the configuration.

    Args:
        ckpt_path (str): path to the policy checkpoint
        config_path (str): path to the configuration file
        policy_type (str): type of the policy, must be in ["diffusion", "safety_filter", "backup"]
        env_kwargs (dict): additional environment arguments
        policy_kwargs (dict): additional policy arguments
    """
    assert policy_type in ["diffusion", "safety_filter", "backup", "safe_diffuser"]

    config = load_config_from_json(config_path)

    # robomimic-style loading policy
    device            = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path = ckpt_path, 
                                                         device    = device, 
                                                         verbose   = True)
    
    # override the action horizon
    policy.policy.algo_config.horizon.unlock_keys()
    policy.policy.algo_config.horizon["action_horizon"] = config.safety.filter.n_head
    policy.policy.algo_config.horizon.lock_keys()

    # maybe further override the policy setting
    if policy_kwargs is not None:
        for k, v in policy_kwargs.items():
            policy.policy.algo_config[k].unlock_keys()
            for vk, vv in v.items():
                policy.policy.algo_config[k][vk] = vv
            policy.policy.algo_config[k].lock_keys()

    # maybe override the environment setting
    if env_kwargs is not None:
        for k, v in env_kwargs.items():
            ckpt_dict["env_metadata"]["env_kwargs"][k] = v
    
    # robomimic-style loading environment
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict        = ckpt_dict, 
                                           render           = False, 
                                           render_offscreen = True, 
                                           verbose          = True)
    
    # wrap the environment with the safety wrapper
    env_safe  = SafeMazeEnv(env, **config.safety)
    dt_action = env_safe.unwrapped_env.sim.model.opt.timestep

    
    # wrap the policy that needs 1) state predictor and 2) backup_policy
    ckpt_dictCopy = deepcopy(ckpt_dict)
    
    maze_size_str = env.unwrapped.name.split("-")[1]
    ckpt_dictCopy["env_metadata"]["env_kwargs"]["maze_spec"] = OPEN_MAZE_DICT[maze_size_str]
    envCopy, _   = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dictCopy, render=False, render_offscreen=True, verbose=True)
    envCopy_safe = SafeMazeEnv(envCopy, **config.safety)
    predictor = MazeEnvSimStatePredictor(envCopy_safe)

    # predictor = DiffuserStatePredictor(rollout_policy = policy, dt = dt_action)
    # load the wrapped policy
    if policy_type == "diffusion":
        policy_wrapped  = IdentityDiffusionPolicyMaze(
                                        rollout_policy = policy, 
                                        backup_policy  = Simple2DPlanner(verbose=config.safety.trajopt.verbose),
                                        dt_action      = dt_action,
                                        predictor      = predictor,
                                        **config.safety)
    elif policy_type == "safe_diffuser":
        policy_wrapped  = SafeDiffuserMaze(
                                        rollout_policy = policy, 
                                        backup_policy  = Simple2DPlanner(verbose=config.safety.trajopt.verbose),
                                        dt_action      = dt_action,
                                        predictor      = predictor,
                                        **config.safety)

    elif policy_type == "safety_filter":
        policy_wrapped  = SafeDiffusionPolicyMaze(
                                        rollout_policy = policy, 
                                        backup_policy  = Simple2DPlanner(verbose=config.safety.trajopt.verbose),
                                        dt_action      = dt_action,
                                        predictor      = predictor,
                                        **config.safety)
    elif policy_type == "backup":
        policy_wrapped  = Simple2DPlanner(verbose=config.safety.trajopt.verbose)
        raise NotImplementedError("Backup policy is not implemented yet.")
    
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


    return policy_wrapped, env_safe, config