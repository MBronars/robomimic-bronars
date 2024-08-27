"""
Add the gripper constraint

TODO 1: Gripper FO vis
TODO 2: Collision check @ planner
TODO 3: Gripper constraint @ planner
"""
import os
import json
from copy import deepcopy

import imageio
import numpy as np
import einops

import robomimic
from robomimic.algo.algo import RolloutPolicy

from safediffusion.envs.env_safety import SafetyEnv
from safediffusion.utils.rand_utils import set_random_seed
import safediffusion_arm.kinova_gen3.utils as KinovaUtils

POLICY_PATH = os.path.join(robomimic.__path__[0], 
                           "../diffusion_policy_trained_models/kinova/model_epoch_600_joint.pth") # delta-end-effector
CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           "../exps/add_gripper_objective/safediffusion_arm.json")

def test(policy, env, horizon, render_mode, seed, save_dir, camera_names,
         video_fps = 20, video_skip = 5):
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

    for step_i in range(horizon):
        if step_i % 10 == 0:
            print("====================================================================")
            # TODO: write the summary of each step 
            print(f"Step {step_i}: State ")
            print("====================================================================")
        
        act, controller_to_use = policy(ob = obs)

        if policy.stuck:
            print("Unable to compute the backup plan.")
            return

        env.switch_controller(controller_to_use)
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

                    plan = policy.backup_policy.get_grasping_pos_from_plan(
                        plan                = policy.nominal_plan,
                        T_world_to_arm_base = obs["T_world_to_arm_base"]
                    )

                    # prepare visualization
                    plan = policy.backup_policy.get_eef_pos_from_plan(
                            plan                  = policy.nominal_plan,
                            T_world_to_arm_base   = obs["T_world_to_arm_base"],
                    )

                    # if len(policy._backup_plan) > 0:
                    #     backup_plan = policy.backup_policy.get_eef_pos_from_plan(
                    #             plan              = policy._backup_plan,
                    #             T_world_to_base   = obs["T_world_to_base"],
                    #     )
                    # else:
                    #     backup_plan = None

                    # Check the zonotope of the interest for visualization
                    
                    # This check if the env_zonotope (directly reading the robot geometry form sim)
                    # collides with the forward_occupancy result using link_zonotopes
                    # FO_arm  = policy.backup_policy.get_arm_zonotopes_at_q(
                    #             q                 = policy.backup_policy.to_tensor(obs["robot0_joint_pos"]),
                    #             T_frame_to_base   = obs["T_world_to_base"]
                    #          )
                    
                    # T_world_to_arm_base = policy.backup_policy.to_tensor(obs["T_world_to_base"])
                    # T_arm_base_to_gripper_base = policy.backup_policy.get_link_pose_at_q("right_hand", obs["robot0_joint_pos"]) 
                    # T_world_to_gripper_base = T_world_to_arm_base @ T_arm_base_to_gripper_base

                    # FO_gripper = policy.backup_policy.get_gripper_zonotopes_at_q(
                    #     q               = policy.backup_policy.to_tensor(obs["robot0_gripper_qpos"]),
                    #     T_frame_to_base = T_world_to_gripper_base
                    # )

                    # FRS = []
                    # FRS.extend(FO_arm)
                    # FRS.extend(FO_gripper)

                    # Visualize the forward occupancy

                    n_skip = 25

                    FRS_links = []
                    FRS_links.extend(policy.backup_policy.gripper_FRS_links)
                    FRS_links.extend(policy.backup_policy.arm_FRS_links)

                    param  = policy._backup_plan.get_trajectory_parameter()
                    optvar = param[policy.backup_policy.opt_dim]
                    beta   = optvar/policy.backup_policy.FRS_info["delta_k"][policy.backup_policy.opt_dim]
                    beta   = einops.repeat(beta, 'n -> repeat n', repeat=policy.backup_policy.n_timestep-1)
                    FRS = [FRS_link.slice_all_dep(beta) for FRS_link in FRS_links]
                    FRS = [FRS_link[0::n_skip] for FRS_link in FRS]

                    # n_skip = 20
                    # param  = policy._backup_plan.get_trajectory_parameter()
                    # optvar = param[policy.backup_policy.opt_dim]
                    # FRS = policy.backup_policy.get_FRS_from_obs_and_optvar(obs, optvar)
                    # FRS = [FRS_link[0::n_skip] for FRS_link in FRS]

                    for cam_name in camera_names:
                        video_img.append(
                            env.render(mode        = render_mode, 
                                       height      = 512, 
                                       width       = 512, 
                                       camera_name = cam_name,
                                       # zonotope
                                       FRS         = FRS,
                                    #    goal        = policy.backup_policy.goal_zonotope,
                                       # trajectory
                                       plan        = plan,
                                    #    backup_plan = backup_plan,
                                       intervened  = policy.intervened,
                                    )
                        )
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)

                else:
                    raise NotImplementedError

        if done or success:
            print("Success!")
            break

        obs = deepcopy(next_obs)
        # TODO: what does this line do?
        state_dict = env.get_state()
    
    stats = dict(
                 Success = bool(success),
                 Horizon = (step_i + 1),
                )
    
    with open(os.path.join(save_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    return stats

if __name__ == "__main__":
    # ----------------------------------------- #
    # Test-seed: [11, 35, 74]
    ckpt_path        = POLICY_PATH
    config_json_path = CONFIG_PATH
    rollout_horizon  = 1000
    render_mode      = "zonotope"
    # seeds            = [5, 8, 10, 12, 13, 19, 27]
    seeds            = [28, 30]
    # ----------------------------------------- #
    # This seed is used to set the random seed for the environment and the policy, 
    # regardless of the random seed used for the rollout
    set_random_seed(42)

    policy, env, config = KinovaUtils.policy_and_env_from_checkpoint_and_config(
                                        ckpt_path, 
                                        config_json_path, 
                                        "safety_filter",
                                        policy_kwargs = {"horizon": {"prediction_horizon": 32}}
    )

    # rollout the policy in the environment using random initial states
    for seed in seeds:
        stats = test(
                    policy       = policy,           # policy and environment
                    env          = env,              # experiment configuration
                    horizon      = rollout_horizon,  # rollout horizon
                    render_mode  = render_mode,      # render mode
                    seed         = seed,
                    save_dir     = config.safety.render.save_dir,
                    # camera_names = ["agentview", "frontview"]
                    camera_names = ["agentview"]
        )