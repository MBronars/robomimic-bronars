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
from safediffusion.algo.planner_base import ParameterizedPlanner
from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.envs.env_safety import SafetyEnv

from safediffusion_arm.environments.safe_pickplace_env import SafePickPlaceBreadEnv
from safediffusion_arm.algo.planner_arm_xml import ArmtdPlannerXML
from safediffusion_arm.algo.predictor_arm import ArmEnvSimStatePredictor
from safediffusion_arm.algo.safety_filter_arm import SafeDiffusionPolicyArm, IdentityDiffusionPolicyArm, IdentityBackupPolicyArm

def load_joint_position_controller_config():
    """
    Joint position controller for the Kinova Environment
    """
    return {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": 2e3,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2
        }


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
    assert policy_type in ["diffusion", "safety_filter", "backup"]

    config = load_config_from_json(config_path)

    # robomimic-style loading policy
    device            = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path = ckpt_path, 
                                                         device    = device, 
                                                         verbose   = True)
    
    # maybe override the policy setting
    if policy_kwargs is not None:
        for k, v in policy_kwargs.items():
            policy.policy.algo_config[k].unlock_keys()
            for vk, vv in v.items():
                policy.policy.algo_config[k][vk] = vv
            policy.policy.algo_config[k].lock_keys()
    
    # override the action horizon
    policy.policy.algo_config.horizon.unlock_keys()
    policy.policy.algo_config.horizon["action_horizon"] = config.safety.filter.n_head
    policy.policy.algo_config.horizon.lock_keys()

    # maybe override the environment setting
    if env_kwargs is not None:
        for k, v in env_kwargs.items():
            ckpt_dict["env_metadata"]["env_kwargs"][k] = v

    # switching controller
    perf_controller_config    = ckpt_dict["env_metadata"]["env_kwargs"]["controller_configs"]
    backup_controller_config  = load_joint_position_controller_config()
    ckpt_dict["env_metadata"]["env_kwargs"]["controller_configs"] = {
        'type': 'SWITCHING',
        "interpolation": None,
        'controllers': [perf_controller_config, backup_controller_config]
    }
    
    # robomimic-style loading environment
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict        = ckpt_dict, 
                                           render           = False, 
                                           render_offscreen = True, 
                                           verbose          = True)
    
    # wrap the environment with the safety wrapper
    env_safe  = SafePickPlaceBreadEnv(env, **config.safety)
    dt_action = 1/env_safe.unwrapped_env.control_freq

    # wrap the policy that needs 1) state predictor and 2) backup_policy
    ckpt_dictCopy = deepcopy(ckpt_dict)
    ckpt_dictCopy["env_metadata"]["env_kwargs"]["controller_configs"] = perf_controller_config
    envCopy, _   = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dictCopy, 
                                                 render=False, 
                                                 render_offscreen=True, 
                                                 verbose=True)
    
    envCopy_safe = SafePickPlaceBreadEnv(envCopy, **config.safety)
    predictor    = ArmEnvSimStatePredictor(envCopy_safe)

    # predictor = DiffuserStatePredictor(rollout_policy = policy, dt = dt_action)
    # load the wrapped policy
    backup_controller_config["dt"] = env_safe.unwrapped_env.control_timestep
    # backup_policy = ArmtdPlanner(action_config = backup_controller_config, 
    #                              **config.safety)
    backup_policy = ArmtdPlannerXML(action_config = backup_controller_config,
                                    robot_name = "panda",
                                    gripper_name = "panda_gripper",
                                    **config.safety)
    backup_policy.calibrate(env_safe)
    
    if policy_type == "diffusion":
        policy_wrapped  = IdentityDiffusionPolicyArm(
                                        rollout_policy = policy, 
                                        backup_policy  = backup_policy,
                                        dt_action      = dt_action,
                                        predictor      = predictor,
                                        rollout_controller_name = "OSC_POSE",
                                        backup_controller_name  = "JOINT_POSITION",
                                        **config.safety)

    elif policy_type == "safety_filter":
        policy_wrapped  = SafeDiffusionPolicyArm(
                                        rollout_policy = policy,
                                        backup_policy  = backup_policy,
                                        dt_action      = dt_action,
                                        predictor      = predictor,
                                        rollout_controller_name = "OSC_POSE",
                                        backup_controller_name  = "JOINT_POSITION",
                                        **config.safety)
        
    elif policy_type == "backup":
        policy_wrapped  = IdentityBackupPolicyArm(
                                rollout_policy = policy,
                                backup_policy  = backup_policy,
                                dt_action      = dt_action,
                                predictor      = predictor,
                                rollout_controller_name = "OSC_POSE",
                                backup_controller_name  = "JOINT_POSITION",
                                **config.safety)
    
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


    return policy_wrapped, env_safe, config

def rollout_planner_with_seed(planner,
                              env,
                              horizon,
                              seed,
                              save_dir,
                              render_mode  = "rgb_array",
                              camera_names = ["agentview"],
                              video_skip   = 5,
                              video_fps    = 20):
    """
    Rollout the backup planner

    Environment has its own action representation, low-level controller

    Planner shoots out the plan
    """
    assert isinstance(env, SafetyEnv)
    assert isinstance(planner, ParameterizedPlanner)

    SUCCESS_FLAG = 1

    set_random_seed(seed)

    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    planner.start_episode()

    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"rollout_seed{seed}_n{env.name}_m{render_mode}.mp4")
    video_writer = imageio.get_writer(video_path, fps=video_fps)
    
    for step_i in range(horizon):
        if step_i % 10 == 0:
            print("====================================================================")
            print(f"Step {step_i}: State ")
            print("====================================================================")

        action_arm = planner.get_action(obs_dict = obs)
        action     = np.concatenate([action_arm, [0]])
        next_obs, _, done, _ = env.step(action)
        success = env.is_success()["task"]

        # Online rendering
        if render_mode == "human":
            env.render(mode="human", camera_name=camera_names[0])
        
        # Offline rendering
        else:
            if step_i % video_skip == 0:
                if render_mode == "rgb_array":
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                
                elif render_mode == "zonotope":
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(
                            env.render(mode = render_mode, height = 512, width = 512, camera_name = cam_name,
                                       goal = planner.goal_zonotope
                                        # plan        = policy.nominal_plan.x_des + env.robotqpos_to_worldpos, 
                                        # backup_plan = policy._backup_plan.x_des + env.robotqpos_to_worldpos,
                                        # intervened  = policy.intervened,
                                        # FRS         = FRS
                                    )
                        )
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)

                else:
                    raise NotImplementedError
        
        # termination conditions
        # if policy.stuck:
        #     print("Stuck!")
        #     break

        if done or success:
            print("Success!")
            break

        # if not is_safe:
        #     num_unsafe += 1
        #     print("Safety violation!")
        
        # if is_intervened:
        #     num_intervention += 1
        
        obs = deepcopy(next_obs)
        # TODO: what does this line do?
        state_dict = env.get_state()
    
    stats = dict(
                 Success = bool(success),
                 Horizon = (step_i + 1),
                #  Collision = num_unsafe,
                #  Intervention = num_intervention,
                )
    
    with open(os.path.join(save_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    return stats


def rollout_with_seed(policy,
                      env,
                      horizon,
                      seed,
                      save_dir,
                      render_mode  = "rgb_array",
                      camera_names = ["agentview"],
                      video_skip   = 5,
                      video_fps    = 20):
    """
    """
    assert isinstance(env, SafetyEnv)
    assert isinstance(policy, RolloutPolicy)

    set_random_seed(seed)

    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    policy.start_episode()

    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"rollout_seed{seed}_n{env.name}_m{render_mode}.mp4")
    video_writer = imageio.get_writer(video_path, fps=video_fps)
    
    # logging variables
    # num_intervention = 0
    # num_unsafe = 0

    for step_i in range(horizon):
        if step_i % 10 == 0:
            print("====================================================================")
            # TODO: write the summary of each step 
            print(f"Step {step_i}: State ")
            print("====================================================================")
        
        act, controller_to_use = policy(ob = obs)

        env.switch_controller(controller_to_use)
        next_obs, _, done, _ = env.step(act)

        success = env.is_success()["task"]

        # TODO: implement Safety Wrapper
        # is_safe = next_obs.pop("safe")

        # TODO: implement Safety Filter policy
        # is_intervened = policy.intervened
        # stuck = policy.stuck

        # Online rendering
        if render_mode == "human":
            env.render(mode="human", camera_name=camera_names[0])
        
        # Offline rendering
        else:
            if step_i % video_skip == 0:
                if render_mode == "rgb_array":
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                
                elif render_mode == "zonotope":
                    video_img = []

                    # prepare visualization
                    plan = policy.backup_policy.get_eef_pos_from_plan(
                            plan              = policy.nominal_plan,
                            T_world_to_base   = obs["T_world_to_base"],
                    )

                    backup_plan = policy.backup_policy.get_eef_pos_from_plan(
                            plan              = policy._backup_plan,
                            T_world_to_base   = obs["T_world_to_base"],
                    )

                    FRS  = policy.backup_policy.get_forward_occupancy_from_plan(
                            plan              = policy._backup_plan,
                            T_world_to_base   = obs["T_world_to_base"]
                    )

                    for cam_name in camera_names:
                        video_img.append(
                            env.render(mode        = render_mode, 
                                       height      = 512, 
                                       width       = 512, 
                                       camera_name = cam_name,
                                       # zonotope
                                       FRS         = FRS,
                                       goal        = policy.backup_policy.goal_zonotope,
                                       # trajectory
                                       plan        = plan,
                                       backup_plan = backup_plan,
                                       intervened  = policy.intervened,
                                    )
                        )
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)

                else:
                    raise NotImplementedError
        
        # termination conditions
        # if policy.stuck:
        #     print("Stuck!")
        #     break

        if done or success:
            print("Success!")
            break

        # if not is_safe:
        #     num_unsafe += 1
        #     print("Safety violation!")
        
        # if is_intervened:
        #     num_intervention += 1
        
        obs = deepcopy(next_obs)
        # TODO: what does this line do?
        state_dict = env.get_state()
    
    stats = dict(
                 Success = bool(success),
                 Horizon = (step_i + 1),
                #  Collision = num_unsafe,
                #  Intervention = num_intervention,
                )
    
    with open(os.path.join(save_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    return stats