"""
Visualize the desired trajectory and test the tracking controller

Author: Wonsuhk Jung
"""
import os
import time
from copy import deepcopy

import imageio
import numpy as np
import torch

import robosuite as suite
from robosuite.controllers import load_controller_config

from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as AcUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.environment.zonotope_env import ZonotopeMuJoCoEnv
from safediffusion.safety_filter.base import SafetyFilter
from safediffusion.utils.io_utils import RESULT_DIR

def overwrite_controller_to_joint_position(ckpt_dict):
    ckpt_dict['env_metadata']['env_kwargs']['controller_configs'] = {
        "type": "JOINT_POSITION",
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.2,
        "output_min": -0.2,
        "kp": 200,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2
        }
    return ckpt_dict

def overwrite_policy_action_horizon(policy, n_head):
    assert isinstance(policy, RolloutPolicy)
    policy.policy.algo_config.horizon.unlock()
    policy.policy.algo_config.horizon.action_horizon = n_head
    policy.policy.algo_config.horizon.lock()

    return policy

def unnormalize(policy, ac):
    """
    Unnormalize the action tensor using the training normalization stats
        policy: (RolloutPolicy) policy
        ac: (1, D) tensor
    """
    ac = TensorUtils.to_numpy(ac)
    if policy.action_normalization_stats is not None:
        action_keys = policy.policy.global_config.train.action_keys
        action_shapes = {k: policy.action_normalization_stats[k]["offset"].shape[1:] for k in policy.action_normalization_stats}
        ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes, action_keys=action_keys)
        ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=policy.action_normalization_stats)
        action_config = policy.policy.global_config.train.action_config
        for key, value in ac_dict.items():
            this_format = action_config[key].get('format', None)
            if this_format == 'rot_6d':
                rot_6d = torch.from_numpy(value).unsqueeze(0)
                rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
                ac_dict[key] = rot
        ac = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)

    return ac


# TODO: 1) Analyze & imitate run_trained_agent.py
def track_safe_policy(q0, dq0, ddq0, safety_filter, env, render, video_writer, zono_video_writer, video_skip, camera_names):
    """
    Track the safe policy

    Args:
        safety_filter: (SafetyFilter) safety filter | make desired trajectory
        env: (EnvBase) environment
        render: (bool) render the environment
        video_writer: (imageio) video writer
        zono_video_writer: (imageio) zonotope video writer
        video_skip: (int) video skip
        camera_names: (list) camera names
    """
    assert isinstance(env, EnvBase)
    assert isinstance(safety_filter, SafetyFilter)
    assert not (render and (video_writer is not None))

    obs = env.reset()
    state_dict = env.get_state() # what does this return?
    state_dict["states"][1:8] = q0
    state_dict["states"][42:49] = dq0
    obs = env.reset_to(state_dict)
    env.env.model_timestep = 1e-3
    env.env.robots[0].controller.new_update = True # HACK: to update the joint position of the controller

    safety_filter.start_episode()
    safety_filter.env.render()
    
    assert(np.allclose(q0, obs["robot0_joint_pos"]) and np.allclose(dq0, obs["robot0_joint_vel"]))

    plan = safety_filter.rollout_param_to_reference_trajectory(q0, dq0, ddq0)
    n_horizon = len(plan)
    FO_plan = safety_filter.forward_occupancy_from_reference_traj(plan, only_end_effector=True)
    # actions = safety_filter.reference_traj_to_actions(plan)
    # actions = np.hstack([actions, np.zeros((actions.shape[0], 1))]) # add gripper actions

    video_count = 0
    stats = {}
    stats["tracking_error"] = []
    stats["joint_goal_error"] = []
    for idx in range(n_horizon-1):
        action = safety_filter.reference_traj_to_actions(plan[idx:idx+2]).squeeze()
        action = np.hstack([action, 0])
        obs, reward, done, _ = env.step(action)
        safety_filter.sync_env()

        # Visualize the zonotope world
        if zono_video_writer is not None:
            if video_count % video_skip == 0:
                video_img = safety_filter.env.render(FO_desired_zonos=FO_plan)
                zono_video_writer.append_data(video_img)
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
            video_count += 1
        
        stats["joint_goal_error"].append(env.env.robots[0].controller.goal_qpos - plan[idx+1][1].cpu().numpy()) # action has no problem
        stats["tracking_error"].append(obs["robot0_joint_pos"] - plan[idx+1][1].cpu().numpy())
    
    print(abs(np.vstack(stats["tracking_error"])).max(0))
    return stats




if __name__ == "__main__":
    
    ############## User Arguments ##########################
    rand_seed = 11                                      # environment seed
    load_environment_from_ckpt = True                   # load environment from checkpoint
    ckpt_name = "assets/model_epoch_300_joint_actions.pth"
    q0 = np.array([0, 0.4, 0, -np.pi+1.4, 0, -1, np.pi/2]) # initial joint position
    dq0 = np.zeros(7)                                          # initial joint velocity
    ddq0 = np.array([1, 1, -1, -1, 0, 0, 0])                     # initial joint acceleration
    ########################################################

    
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    set_random_seed(rand_seed)
    result_dir = f"{RESULT_DIR}/scenario_{rand_seed}"
    os.makedirs(result_dir, exist_ok=True)
    mujoco_video_writer = imageio.get_writer(f"{result_dir}/track_mujoco.mp4", fps=20)
    zono_video_writer = imageio.get_writer(f"{result_dir}/track_zonotope.mp4", fps=20)

    # make environment
    if load_environment_from_ckpt:
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_name)
        _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
        ckpt_dict = overwrite_controller_to_joint_position(ckpt_dict)

        env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=True, render_offscreen=False, verbose=True)
    
    else:
        controller_name  = 'JOINT_POSITION'

        options = {}
        options["env_name"] = 'PickPlace'
        options["robots"] = 'Kinova3'
        options["controller_configs"] = load_controller_config(default_controller=controller_name)

        env = suite.make(
            **options,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=20,
        )
    
    # safety filter configuration
    zonotope_twin_env = ZonotopeMuJoCoEnv(env.env, render_online=True, ticks=True)
    safety_filter = SafetyFilter(zono_env=zonotope_twin_env, n_head=1, verbose=True)

    # q0 = torch.tensor(q0, dtype=safety_filter.dtype, device=safety_filter.device)
    obs = env.reset()
    q0 = torch.tensor(obs["robot0_joint_pos"], dtype=safety_filter.dtype, device=safety_filter.device)
    dq0 = torch.tensor(dq0, dtype=safety_filter.dtype, device=safety_filter.device)
    ddq0 = torch.tensor(ddq0, dtype=safety_filter.dtype, device=safety_filter.device)

    stats = track_safe_policy(q0 = q0, dq0 = dq0, ddq0 = ddq0,
                              safety_filter=safety_filter,
                              env=env,
                              render=False,
                              video_writer=mujoco_video_writer,
                              zono_video_writer=zono_video_writer,
                              video_skip=1,
                              camera_names=["agentview"])
    
    


