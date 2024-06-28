"""
Test and Visualize the D4RL dataset
"""


import os
import json
import h5py
import numpy as np
import imageio

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

def view_raw_d4rl_dataset(env_name):
    base_folder = os.path.join(robomimic.__path__[0], "../datasets", "d4rl")
    dataset_path = os.path.join(base_folder, f"{env_name}.hdf5")

    # summarize the dataset
    f = h5py.File(dataset_path, "r")

    print("hi")

def summarize_task(env_name):
    base_folder = os.path.join(robomimic.__path__[0], "../datasets", "d4rl", "converted")
    dataset_path = os.path.join(base_folder, f"{env_name}.hdf5")

    # summarize the dataset
    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())
    num_demos = len(demos)

    action_dim = f["data"][demos[0]]["actions"].shape[1]
    obs_dim = f["data"][demos[0]]["obs"]["flat"].shape[1]

    print(f"Dataset: {env_name} has {num_demos} demos")
    print(f"Action dim: {action_dim}")
    print(f"Obs dim: {obs_dim}")

def playback_trajectory(env_name, demo_idx_list):
    base_folder = os.path.join(robomimic.__path__[0], "../datasets", "d4rl", "converted")
    dataset_path = os.path.join(base_folder, f"{env_name}.hdf5")

    # summarize the dataset
    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())

    env_meta = json.loads(f["data"].attrs["env_args"])
    print("==== Env Meta ====")
    print(json.dumps(env_meta, indent=4))
    print("")

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, 
        render=False,            # no on-screen rendering
        render_offscreen=True,   # off-screen rendering to support rendering video frames
    )

    video_path = os.path.join(base_folder, f"{env_name}_playback.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    
    for demo_idx in demo_idx_list:
        # NOTE: the target is always fixed, this was verified by checking the reward in 
        # `maze_model` to be consistent with the observed reward
        demo_grp = f["data"][demos[demo_idx]]
        num_samples = demo_grp.attrs["num_samples"]

        assert(np.allclose(np.exp(-np.linalg.norm(demo_grp["obs"]["flat"][:, 0:2]-env.env.get_target(), 2, 1)),demo_grp["rewards"]))

        if any(demo_grp["rewards"][0:]):
            print("Successful Demo")
            print(f"{num_samples} samples in demo {demo_idx}")
        else:
            continue
        
        init_state = demo_grp["obs"]["flat"][0]
        init_state = np.hstack([0, init_state]) # add a dummy state (time) to match the env's state format
        init_state_dict = dict(states = init_state)
        env.reset()
        env.reset_to(init_state_dict)
        actions = demo_grp["actions"]
        for t in range(actions.shape[0]):
            env.step(actions[t])
            video_img = env.render(mode="rgb_array", height=512, width=512)
            video_writer.append_data(video_img)
        
    video_writer.close()

if __name__ == "__main__":
    """
    TODO: Integrate this with the "Janner - Diffusers" codebase
    """
    env2raw = {
               "maze2d_umaze_v1": "maze2d-umaze-sparse-v1",
               "maze2d_medium_dense_v1": "maze2d-medium-dense-v1",
               "maze2d_large_dense_v1": "maze2d-large-dense-v1"
            }
    
    # d4rl_env_name = "maze2d_umaze_v1"
    d4rl_env_name = "maze2d_medium_dense_v1"
    # d4rl_env_name = "maze2d_large_dense_v1"

    d4rl_env_raw = env2raw[d4rl_env_name]
    view_raw_d4rl_dataset(d4rl_env_raw)

    num_demos = 100
    playback_trajectory(d4rl_env_name, np.linspace(0, num_demos-1, num_demos, dtype=int).tolist())

