"""
Test the functionality of safety filter in the robomimic environment
"""
import os
from copy import deepcopy

import imageio
import numpy as np
import torch

import robomimic
from robomimic.algo import RolloutPolicy
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

# safe diffusion specific imports
from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.utils.file_utils import load_config_from_json
from safediffusion.envs.env_safety import SafetyEnv

# maze2d specific imports
from d4rl.pointmaze.maze_model import LARGE_MAZE, LARGE_MAZE_OPEN
from safediffusion_d4rl.environments.safe_maze_env import SafeMazeEnv
from safediffusion_d4rl.algo.safety_filter_maze import SafeDiffusionPolicyMaze, IdentityDiffusionPolicyMaze
from safediffusion_d4rl.algo.planner_maze import Simple2DPlanner

POLICY_PATH = os.path.join(robomimic.__path__[0], "../diffusion_policy_trained_models/maze2d/large_diffpolicy_H32/models/model_epoch_1900.pth") # policy checkpoint
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "exps/safety_filter_large/safe_diffusion_maze.json")

def zonotope_intersects(Z1, Z2, combs):
    """
    Check if two zonotopes intersect
    """
    assert isinstance(Z2, list)

    for zono in Z2:
        zono = zono.project().deleteZerosGenerators()

        Z1_ = Z1 - zono
        _,b = Z1_.polytope(combs)

        unsafe = b.min(dim=-1)[0]>1e-6

        if any(unsafe):
            return True
    
    return False


def get_FRS_from_backup_policy(policy, env, obs):
    """
    Get the FRS from the backup policy
    """
    obs_frs  = deepcopy(obs)

    backup_plan = policy._backup_plan

    # modify obs_frs["flat"]
    obs_frs["flat"] = torch.hstack([backup_plan.x_des[0], backup_plan.dx_des[0]])

    # modify
    world_pos = backup_plan.x_des[0] + env.robotqpos_to_worldpos
    obs_frs["zonotope"]["robot"][0].center = torch.hstack([world_pos, torch.tensor(0)])

    trajparam    = backup_plan.get_trajectory_parameter()
    optvar       = trajparam[policy.backup_policy.opt_dim]
    FRS          = policy.backup_policy.get_FRS_from_obs_and_optvar(obs_frs, optvar)
    
    return FRS

def test_state_predictions(policy, env, x0, target_pos, video_skip, save_dir, render_mode):
    """
    Test out the quality of the state predictions using dummy environment
    """
    assert isinstance(env, SafetyEnv)
    assert isinstance(policy, RolloutPolicy)

    # initialize the environment
    obs  = env.reset()
    obs  = env.set_state(pos = x0[:2], vel = x0[2:])
    goal = env.set_goal(pos = target_pos)

    policy.start_episode()

    video_path = os.path.join(save_dir, f"state_prediction.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    rollout = [obs["flat"][-1, :]]
    for i in range(10):
        # _ = policy(ob=obs, goal=goal)
        obs.pop("safe"); obs.pop("zonotope")
        (plan, actions) = policy.get_plan_from_nominal_policy(obs, goal)
        plan = plan + np.array([1, 1, 0, 0])
        
        naction = actions.shape[0]

        for j in range(naction):
            act = actions[j]
            obs, _, _, _ = env.step(act)
            rollout.append(obs["flat"][-1, :])
            
            img = env.render(mode=render_mode, height=512, width=512, plan=plan)
            video_writer.append_data(img)
        
    video_writer.close()
        

def rollout(policy, env, horizon, x0, target_pos,
            video_skip = 5, save_dir = None, render_mode = None):
    assert isinstance(env, SafetyEnv)
    assert isinstance(policy, RolloutPolicy)

    # initialize the environment
    obs  = env.set_state(pos = x0[:2], vel = x0[2:])
    goal = env.set_goal(pos = target_pos)
    policy.start_episode()
        
    total_reward = 0

    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"rollout_x{x0}_g{target_pos}_n{env.name}_m{render_mode}.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    try:
        for step_i in range(horizon):
            act = policy(ob=obs, goal=goal)
            next_obs, r, done, _ = env.step(act)
            total_reward += r
            success = env.is_success()["task"]
            is_safe = next_obs.pop("safe")

            if step_i % video_skip == 0:
                if not policy.intervened:
                    # new FRS is computed
                    FRS = get_FRS_from_backup_policy(policy, env, obs)
                    # TODO: Error is reproduced here
                    # assert(not zonotope_intersects(FRS, obs["zonotope"]["obstacle"], policy.backup_policy.combs))

                img = env.render(mode=render_mode, height=512, width=512, 
                                 plan        = policy.nominal_plan.x_des + env.robotqpos_to_worldpos, 
                                 backup_plan = policy._backup_plan.x_des + env.robotqpos_to_worldpos,
                                 intervened  = policy.intervened,
                                 FRS         = FRS)
                
                video_writer.append_data(img)

            if len(policy._backup_plan) == 0:
                # TODO: Ideally, this would end with stopping. Just replan.
                print("No backup plan available!")
                break
            
            if done or success:
                print("Task Completed!")
                if success:
                    print("Success!")
                break

            if not is_safe:
                print("MuJoCo Collision Detected!")

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
    rollout_horizon  = 1000
    rand_seeds       = np.linspace(60, 101, 101-60+1).astype(int)
    render_mode      = "zonotope"
    #-------------------------------------------------#
    for rand_seed in rand_seeds:
        try:
            set_random_seed(rand_seed)
            config = load_config_from_json(config_json_path)

            # robomimic-style loading policy
            device            = TorchUtils.get_torch_device(try_to_use_cuda=True)
            policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
            
            policy.policy.algo_config.horizon.unlock()
            To = policy.policy.algo_config.horizon.observation_horizon
            policy.policy.algo_config.horizon.action_horizon = config.safety.filter.n_head
            
            # robomimic-style loading environment
            ckpt_dict["env_metadata"]["env_kwargs"]["maze_spec"] = LARGE_MAZE
            env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=False, render_offscreen=True, verbose=True)
            
            # wrap the environment with safety wrapper
            env_safe     = SafeMazeEnv(env, **config.safety)

            # copy of the environment for state prediction purpose
            ckpt_dictCopy = deepcopy(ckpt_dict)
            ckpt_dictCopy["env_metadata"]["env_kwargs"]["maze_spec"] = LARGE_MAZE_OPEN
            envCopy, _   = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dictCopy, render=False, render_offscreen=True, verbose=True)
            envCopy_safe = SafeMazeEnv(envCopy, **config.safety)

            # wrap the policy with the safety filter
            policy_unsafe = IdentityDiffusionPolicyMaze(rollout_policy = policy, 
                                                backup_policy  = Simple2DPlanner(verbose=True),
                                                dt_action      = env_safe.sim.model.opt.timestep,
                                                predictor      = envCopy_safe,
                                                **config.safety)

            policy_safe   = SafeDiffusionPolicyMaze(rollout_policy = policy, 
                                                backup_policy  = Simple2DPlanner(verbose=True),
                                                dt_action      = env_safe.sim.model.opt.timestep,
                                                predictor      = envCopy_safe,
                                                **config.safety)

            # policy_safe = policy

            obs  = env_safe.reset()
            goal = env_safe.get_goal()

            x0 = obs["flat"][-1, :]
            x0[:2] += env_safe.robotqpos_to_worldpos
            target_pos = goal["flat"][:2] + env_safe.robotqpos_to_worldpos
            
            print(f"====================== Seed {rand_seed} Diffusion =========================")
            # rollout the diffusion
            # stats = rollout(
            #     policy=policy_unsafe,
            #     env=env_safe,
            #     x0 = x0,
            #     target_pos = target_pos,
            #     horizon=rollout_horizon,
            #     video_skip=10,
            #     save_dir= config.safety.render.save_dir+f"seed_{rand_seed}"+"_diff",
            #     render_mode=render_mode
            # )

            print(f"====================== Seed {rand_seed} Safety =========================")
            # rollout the diffusion with safety filter
            stats = rollout(
                policy=policy_safe,
                env=env_safe,
                x0 = x0,
                target_pos = target_pos,
                horizon=rollout_horizon,
                video_skip=10,
                save_dir= config.safety.render.save_dir+f"seed_{rand_seed}"+"_safe",
                render_mode=render_mode
            )

        except Exception as e:
            print(f"Got exception: {e}")
            continue