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
        act = policy(ob=obs)
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