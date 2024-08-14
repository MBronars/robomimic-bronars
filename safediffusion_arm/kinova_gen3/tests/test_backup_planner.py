import os

# robomimic
import robomimic

import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.file_utils as FileUtils
from safediffusion.utils.file_utils import load_config_from_json

from safediffusion_arm.environments.safe_arm_env import SafePickPlaceBreadEnv
from safediffusion_arm.algo.planner_arm import ArmtdPlanner
import safediffusion_arm.kinova_gen3.utils as KinovaUtils

POLICY_PATH = os.path.join(robomimic.__path__[0], 
                           "../diffusion_policy_trained_models/kinova/model_epoch_600_joint.pth") # delta-end-effector
CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           "../exps/basic/safediffusion_arm.json")

def test(planner, env, horizon, seed, save_dir):
    """ 
    Write the testing script
    """
    stats = KinovaUtils.rollout_planner_with_seed(
                                        planner      = planner,
                                        env          = env,
                                        horizon      = horizon,
                                        seed         = seed,
                                        save_dir     = save_dir,
                                        render_mode  = "zonotope",
                                        video_skip   = 5,
                                        camera_names = ["frontview", "agentview"]
                                    )

if __name__ == "__main__":
    """
    The functionality of backup planner is tested out on the SafetyEnv

    horizon = 0.01 second per step
    """
    # ----------------------------------- #
    #           USER ARGUMENTS            #
    # ----------------------------------- #
    ckpt_path            = POLICY_PATH
    config_json_filename = CONFIG_PATH
    horizon              = 500
    seed                 = 35
    # ----------------------------------- #

    # robomimic-style loading environment
    device            = TorchUtils.get_torch_device(try_to_use_cuda=True)
    _, ckpt_dict      = FileUtils.policy_from_checkpoint(ckpt_path = ckpt_path, 
                                                         device    = device, 
                                                         verbose   = True)
    ckpt_dict         = KinovaUtils.overwrite_controller_to_joint_position(ckpt_dict)
    
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict        = ckpt_dict, 
                                           render           = False, 
                                           render_offscreen = True, 
                                           verbose          = True)
    
    # load configuration object
    config_json_filepath = os.path.join(os.path.dirname(__file__), config_json_filename)
    config = load_config_from_json(config_json_filepath)
    
    env    = SafePickPlaceBreadEnv(env, **config.safety)
    
    policy = ArmtdPlanner(**config.safety)
    policy.update_weight(dict(qpos_goal=1.0, qpos_projection=0.0))

    test(planner   = policy,
        env      = env,
        horizon  = horizon,
        seed     = seed,
        save_dir = config.safety.render.save_dir)

