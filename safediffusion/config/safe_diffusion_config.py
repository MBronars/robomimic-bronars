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
        # safety-filter configuration
        self.safety.filter.n_head = 1
        self.safety.filter.verbose = True
        self.safety.filter.max_init_attempts = 10

        # trajectory optimization
        self.safety.trajopt.verbose = False
        
        # render configuration
        self.safety.render.save_dir = None
        
        self.safety.render.zonotope.width = 20
        self.safety.render.zonotope.height = 20
        self.safety.render.zonotope.ticks  = True
        
        self.safety.render.zonotope.robot.color = "black"
        self.safety.render.zonotope.robot.alpha = 0.1
        self.safety.render.zonotope.robot.linewidth = 0.5

        self.safety.render.zonotope.goal.color = "purple"
        self.safety.render.zonotope.goal.alpha = 0.3
        self.safety.render.zonotope.goal.linewidth = 0.5
        
        self.safety.render.zonotope.static_obs.color = "red"
        self.safety.render.zonotope.static_obs.alpha = 0.1
        self.safety.render.zonotope.static_obs.linewidth = 0.5
        
        self.safety.render.zonotope.dynamic_obs.color = "blue"
        self.safety.render.zonotope.dynamic_obs.alpha = 0.1
        self.safety.render.zonotope.dynamic_obs.linewidth = 0.5
        
        # FRS (optional)
        self.safety.render.zonotope.frs.color = "green"
        self.safety.render.zonotope.frs.alpha = 0.1
        self.safety.render.zonotope.frs.linewidth = 0.5

        # Plan (optional)
        self.safety.render.zonotope.plan.color = "blue"
        self.safety.render.zonotope.plan.linewidth = 5.0
        self.safety.render.zonotope.plan.linestyle = "-"

        self.safety.render.zonotope.backup_plan.color = "green"
        self.safety.render.zonotope.backup_plan.linewidth = 1.0
        self.safety.render.zonotope.backup_plan.linestyle = "--"

        
