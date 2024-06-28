from robomimic.config.base_config import BaseConfig
from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig

class SafeDiffusionConfig(DiffusionPolicyConfig):
    ALGO_NAME = "safediffusion"

    def __init__(self, dict_to_load=None):
        super().__init__(dict_to_load=dict_to_load)
        self.unlock_keys()
        self.safety_config()
        self.lock_keys()

    def safety_config(self):
        # Render configuration for the zonotope renderers
        self.safety.render.zonotope.width = 20
        self.safety.render.zonotope.height = 20
        self.safety.render.zonotope.ticks  = True
        self.safety.render.zonotope.robot.color = "black"
        self.safety.render.zonotope.robot.alpha = 0.1
        self.safety.render.zonotope.robot.linewidth = 0.5
        self.safety.render.zonotope.static_obs.color = "red"
        self.safety.render.zonotope.static_obs.alpha = 0.1
        self.safety.render.zonotope.static_obs.linewidth = 0.5
        self.safety.render.zonotope.dynamic_obs.color = "blue"
        self.safety.render.zonotope.dynamic_obs.alpha = 0.1
        self.safety.render.zonotope.dynamic_obs.linewidth = 0.5
        
