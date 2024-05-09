"""
Create a zontope twin world from a Robosuite environment

Author: Wonsuhk Jung
"""
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config
from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.environment.zonotope_env import ZonotopeMuJoCoEnv

if __name__ == "__main__":
    ############ User Parameters #####################
    # controller_name: JOINT_POSITION, JOINT_VELOCITY, JOINT_TORQUE, OSC_POSITION, OSC_POSE, IK_POSE
    controller_name  = 'JOINT_POSITION'
    # controller_name  = 'OSC'
    rand_seed = 42
    # rand_seed = 126
    visualize = False
    ##################################################

    set_random_seed(rand_seed)

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
    
    env.reset()
    env.viewer.set_camera(camera_id=1)

    render_kwargs = {
        "save_dir": f"figures/twin_world_{rand_seed}"}
    
    zono_env = ZonotopeMuJoCoEnv(env, render_online=True, ticks=True, render_kwargs=render_kwargs)
    zono_env.reset()
    zono_env.render()

    for i in range(5000):        
        # if i % 1000 == 0:
        #     env.reset()
        #     env.viewer.set_camera(camera_id=2)
        #     zono_env.reset()
        #     zono_env.render()

        env.step(np.zeros(env.action_dim))
        obs = env.render()
        