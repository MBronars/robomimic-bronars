import os

# robomimic
import robomimic

from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.file_utils as FileUtils
from safediffusion.utils.file_utils import load_config_from_json

from safediffusion_arm.environments.safe_arm_env import SafePickPlaceBreadEnv
import safediffusion_arm.kinova_gen3.utils as KinovaUtils

# POLICY_PATH = os.path.join(robomimic.__path__[0], 
#                            "../diffusion_policy_trained_models/kinova/model_epoch_300_joint_actions.pth") # delta-joint action
POLICY_PATH = os.path.join(robomimic.__path__[0], 
                           "../diffusion_policy_trained_models/kinova/model_epoch_600_joint.pth") # delta-end-effector
CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           "../exps/basic/safediffusion_arm.json")

def test_safety_env(policy, env, horizon, seed, save_dir):
    """ Write the testing script
    """
    stats = KinovaUtils.rollout_with_seed(policy   = policy,
                                        env      = env,
                                        horizon  = horizon,
                                        seed     = seed,
                                        save_dir = save_dir,
                                        video_skip = 3,
                                        )

def test_zonotope_env(policy, env, horizon, seed, save_dir):
    """ Write the testing script
    """
    stats = KinovaUtils.rollout_with_seed(policy   = policy,
                                        env      = env,
                                        horizon  = horizon,
                                        seed     = seed,
                                        save_dir = save_dir,
                                        render_mode = "zonotope",
                                        video_skip = 5
                                        )

def test_rollout(env, policy):
    pass

if __name__ == "__main__":
    """
    Given the robomimic environment and the configuration file,
    the safety wrapper is implemented and tested-out.
    Testing without renderer

    horizon = 0.01 second per step
    """
    # ----------------------------------- #
    #           USER ARGUMENTS            #
    # ----------------------------------- #
    ckpt_path            = POLICY_PATH
    config_json_filename = CONFIG_PATH
    horizon              = 500
    seed                 = 63
    # ----------------------------------- #

    # robomimic-style loading policy
    device            = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path = ckpt_path, 
                                                         device    = device, 
                                                         verbose   = True)
    
    # robomimic-style loading environment
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict        = ckpt_dict, 
                                           render           = False, 
                                           render_offscreen = True, 
                                           verbose          = True)
    
    # load configuration object
    config_json_filepath = os.path.join(os.path.dirname(__file__), config_json_filename)
    config = load_config_from_json(config_json_filepath)
    
    # maybe override the policy and environment here, see pointmaze/utils.py

    # TODO: make the 
    env   = SafePickPlaceBreadEnv(env, init_active_object_id=1, **config.safety)

    test_safety_env(policy   = policy,
                    env      = env,
                    horizon  = horizon,
                    seed     = seed,
                    save_dir = config.safety.render.save_dir)

    # test_safety_env(env_safearm)
