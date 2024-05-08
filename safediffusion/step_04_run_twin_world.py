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
    rand_seed = 42
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
    env.viewer.set_camera(camera_id=0)

    zono_env = ZonotopeMuJoCoEnv(env)
    zono_env.reset()
    zono_env.visualize_task()

    for i in range(500):
        env.step(np.zeros(env.action_dim))
        env.render()

    print("Done!")