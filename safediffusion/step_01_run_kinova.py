"""
Run Kinova robot in PickPlace environment with various controllers.

Author: Wonsuhk Jung
"""
import numpy as np

import robosuite as suite
from robosuite.controllers import load_controller_config
from safediffusion.utils.rand_utils import set_random_seed
from safediffusion.utils.reachability_utils import zonotope_from_robosuite_env


def tracking_error(env):
    tracking_error = env.robots[0].controller.goal_qpos - env.robots[0].controller.joint_pos
    
    return tracking_error

def get_zonotopes_from_objects(objects):
    """
    TODO 1: Define zonotope for the object
    TODO 2: Define zonotope for the robot
    TODO 3: check collisions
    """
    
    pass

if __name__ == "__main__":
    # TODO 1: Change camera angle
    # TODO 2: Define zonotope for the object
    # TODO 3: Define zonotope for the robot
    # TODO 4: check collisions

    ############ User Parameters ############
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

    zono_env = zonotope_from_robosuite_env(env)

    low, high = env.action_spec

    test_idx = 5

    for i in range(500):
        # action = np.random.uniform(low, high)
        action = np.zeros((env.action_dim))
        action[test_idx] = np.cos(i / 50)
        obs, reward, done, _ = env.step(action)
        if visualize: env.render()

        # log interesting properties
        # print(f"Joint tracking error: {np.max(abs(tracking_error(env)))}")
        # zonotopes = zonotope_from_robosuite_env(env)
        # plot_zonos(zonotopes)

        