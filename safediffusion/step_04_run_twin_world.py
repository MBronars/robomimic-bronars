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

    render_kwargs = {
        "xlim": [-0.3, 0.3],
        "ylim": [-0.3, 0.3],
        "zlim": [0, 1.0],
        "save_dir": "figures_twin_world"}
    
    zono_env = ZonotopeMuJoCoEnv(env, render_online=True, ticks=True, render_kwargs=render_kwargs)
    zono_env.reset()
    zono_env.render()

    for i in range(500):
        if i % 50 == 0:
            # env.reset()
            print(env.sim.data.get_body_xpos('robot0_base'))
            print(env.sim.data.get_body_xpos('robot0_shoulder_link'))
            print(env.sim.data.get_body_xpos('robot0_forearm_link'))
        # env.step(np.random.rand(env.action_dim))
        env.step(np.zeros(env.action_dim))

        env.render()