import os
import imageio
from copy import deepcopy

import numpy as np
import torch
import json

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo.algo import RolloutPolicy

from safediffusion.algo.planner_base import ParameterizedPlanner
from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.envs.env_safety import SafetyEnv

def overwrite_controller_to_joint_position(ckpt_dict):
    ckpt_dict['env_metadata']['env_kwargs']['controller_configs'] = {
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
    return ckpt_dict

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
    """

    assert isinstance(env, SafetyEnv)
    assert isinstance(planner, ParameterizedPlanner)

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
            # TODO: write the summary of each step 
            print(f"Step {step_i}: State ")
            # print(f"Step {step_i}:  State {np.round(obs['flat'][-1], 2)}, Goal {np.round(goal['flat'][:2], 2)}")
            print("====================================================================")

        # TODO: pop zonotope and safe here?
        plan, info = planner(obs_dict=obs)

        # TODO: make plan to the joint action repr.
        # ctrl = self.env.helper_controller
        # offset = np.array(traj.x_des[0] - self.env.qpos) # if initial planning state is not the same as the current state
        # actions = np.array(traj.x_des[1:] - traj.x_des[:-1]) + offset
        # if ((actions.max(0) > ctrl.output_max) | (actions.min(0) < ctrl.output_min)).any():
        #     self.disp("Actions are out of range: joint goal position gets different with the plan", self.verbose)
        # actions = np.clip(actions, ctrl.output_min, ctrl.output_max)
        # actions = scale_array_from_A_to_B(actions, A=[ctrl.output_min, ctrl.output_max], B=[ctrl.input_min, ctrl.input_max]) # (T, D) array
        # actions = np.clip(actions, ctrl.input_min, ctrl.input_max)

        # TODO: what is `done`` here?
        next_obs, _, done, _ = env.step(act)
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
            # print(f"Step {step_i}:  State {np.round(obs['flat'][-1], 2)}, Goal {np.round(goal['flat'][:2], 2)}")
            print("====================================================================")

        # TODO: pop zonotope and safe here?
        obs = {k: obs[k] for k in policy.policy.global_config.all_obs_keys}
        
        act = policy(obs_dict=obs)
        # TODO: what is `done`` here?
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
                    for cam_name in camera_names:
                        video_img.append(
                            env.render(mode = render_mode, height = 512, width = 512, camera_name = cam_name,
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