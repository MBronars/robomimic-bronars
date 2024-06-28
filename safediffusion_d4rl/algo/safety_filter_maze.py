from safediffusion.algo.safety_filter import SafeDiffusionPolicy


class SafeDiffusionPolicyMaze(SafeDiffusionPolicy):
    def preprocess_to_reference_traj(self, obs, actions):
        """
        TODO: Implement the Map: rollout_policy.action_dim -> backup_policy.state_dim
        """
        raise NotImplementedError
    
    def postprocess_to_actions(self, reference_traj):
        """
        TODO: Implement the Map: backup_policy.state_dim -> rollout_policy.action_dim
        """
        raise NotImplementedError
    
    def check_head_plan_safety(self, ob, plan):
        """
        TODO: Implement the safety verifier, maybe this better suit as a method of the Planner class
        """
        raise NotImplementedError