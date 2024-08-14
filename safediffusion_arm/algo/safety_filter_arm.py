from copy import deepcopy

import torch
import numpy as np
import einops

from robomimic.algo.diffusers import DiffusionPolicyUNet as DiffusersUNet
from robomimic.utils.obs_utils import unnormalize_dict
from safediffusion.algo.plan import ReferenceTrajectory
from safediffusion.algo.safety_filter import SafeDiffusionPolicy

from safediffusion.utils.npy_utils import scale_array_from_A_to_B


class SafeDiffusionPolicyArm(SafeDiffusionPolicy):
    def __init__(self, 
                 rollout_policy, 
                 backup_policy, 
                 dt_action, 
                 predictor, 
                 **config
                 ):
        """
        The reference trajectory is defined in the 2D world space (x, y)
            x_des  = (x, y)
            dx_des = (v_x, v_y)
        """
        super().__init__(rollout_policy, backup_policy, dt_action=dt_action, **config)

        # object for state prediction
        # TODO: Make this as a third-party object whose `predict(init_state, actions)` returns the state predictions
        self.predictor = predictor

    # --------------------------------------------- #
    # Abstract functions for SafetyFilter
    # --------------------------------------------- #
    @property
    def safety_critical_state_keys(self):
        return ["flat"]
    
    def update_backup_policy_weight(self):
        """
        Update the weight dictionary of the backup policy according to the internal status.
        This function reflects the strategy of the objective of the backup planner.

        NOTE: If the policy is stucked, change to the goal mode
        TODO: Check if all keys of the weight dict is in the backup planner
        """
        weight_dict = dict(qpos_goal       = 1.0, 
                           qpos_projection = 0.0)
        
        self.backup_policy.update_weight(weight_dict)
    
    def get_plan_from_nominal_policy(self, obs, goal=None, num_samples=1):
        """
        Get the plan from the nominal policy

        Args
            ob   : (B, 1) Observation array
            goal : (B, 1) Goal array
        
        Returns
            plan   : (B, 1) ReferenceTrajectory array
            actions: (B, T_p, D_a) np.ndarray
        """
        # Preprocess the observation dictionary compatible to the rollout policy
        obs_perf = {k: obs[k] for k in self.rollout_policy_obs_keys}
        obs_perf = self.rollout_policy._prepare_observation(obs_perf)
        
        if goal is not None:
            goal_perf = {k: goal[k] for k in self.rollout_policy_obs_keys}
            goal_perf = self.rollout_policy._prepare_observation(goal_perf)

        # Compute the multiple trajectories with the policy
        actions_candidates = self.rollout_policy.policy.get_multi_actions(obs_perf, goal_perf, num_trajectories=num_samples)
        actions_candidates = np.stack([actions.cpu().numpy() for actions in actions_candidates], 0) # (B, Tp, Da)
        
        # State prediction model
        init_states = einops.repeat(obs["flat"][-1, :], 'd -> repeat d', repeat=num_samples) # (B, Dx)
        plans_candidates = self.predictor(init_states, actions_candidates)

        # Ranking the plans
        best_plan, best_plan_idx, ranking_info = self.rank_plans(plans_candidates, obs, goal)
        best_actions = actions_candidates[best_plan_idx]
        self.ranking_info = ranking_info
        
        return best_plan, best_actions
        
    def postprocess_to_actions(self, reference_traj, ob):
        """
        Given the reference trajectory, postprocess to executable action that can be sent to the environment

        Implementation of prediction function (s_{t..t+T_p}) -> (a_{t..t+P})

        In SafeDiffusionArm, this function converts the joint angle trajectories into the action representation
        for joint-position controller

        Args
            reference_traj : (B, 1) ReferenceTrajectory array
            ob             : dict, observation dictionary from the environment
        
        Returns
            actions : (B, T_p, D_a) np.ndarray
        """
        assert isinstance(reference_traj, ReferenceTrajectory)

        actions = self.backup_policy.execute_plan(reference_traj, ob)

        return actions
    
    def check_head_plan_safety(self, plan, ob):
        """
        TODO: Implement the safety verifier, maybe this better suit as a method of the Planner class
        """
        collision = self.backup_policy.collision_check(plan, ob)
        return not collision
    
    def process_dicts_for_backup_policy(self, ob, goal, plan=None):
        """
        Given the plan, observation dictionary, and goal dictionary from the environment,
        preprocess the data to be compatible with the backup policy

        Args
            ob   : observation dictionary
            goal : goal dictionary
            plan : ReferenceTrajectory object (optional)
        
        Returns
            obs_dict : observation dictionary
            goal_dict: goal dictionary

        # TODO: Since the safety filter is the wrapper of the backup policy, it would be good if we can parse the
        #       available objective function keys from the backup policy
        """
        assert plan is None or isinstance(plan, ReferenceTrajectory)

        obs_dict  = deepcopy(ob)
        goal_dict = deepcopy(goal)

        initialize_mode = (plan is None)

        if initialize_mode:
            # When there is no nominal plan available, compute the backup plan starting at current state
            for key in self.safety_critical_state_keys:
                if obs_dict[key].ndim == 2:
                    obs_dict[key] = ob[key][-1, :]

        else:
            # When nominal plan is available, compute the backup plan at the initial state of the nominal plan
            
            # update observation
            # The obs["flat"] and obs["zonotope"]["robot"] is updated accordingly
            if ob["flat"].ndim == 2:
                current_qposvel = ob["flat"][-1, :]
            else:
                current_qposvel = ob["flat"]
            
            backup_qposvel = torch.hstack([plan[0][1], plan[0][2]]) 
            center_shift = torch.hstack([(backup_qposvel[:2] - current_qposvel[:2]), torch.tensor(0)])

            obs_dict["flat"] = backup_qposvel 
            obs_dict["zonotope"]["robot"][0] += center_shift

            # update goal dictionary to provide info. about the objective function
            goal_dict["reference_trajectory"] = plan

        return obs_dict, goal_dict

    def rank_plans(self, plans, obs, goal):
        OBJ_VAL_INIT = 1e4

        best_obj_val = OBJ_VAL_INIT

        ranking_dict = dict()
        for idx, plan in enumerate(plans):
            tail_plan         = plan[self.n_head:]
            backup_plan, info = self.compute_backup_plan(tail_plan, obs, goal)

            is_safe = info["status"] == 0
            obj_val = info["obj_val"]

            if is_safe and obj_val < best_obj_val:
                best_obj_val = obj_val
                best_plan = plan
                best_plan_idx = idx
            
            # log info into ranking_dict
            # metrics
            ranking_dict[idx] = dict()
            ranking_dict[idx]["plan"]        = plan
            ranking_dict[idx]["safe"]        = is_safe
            ranking_dict[idx]["obj_val"]     = obj_val

            # optional
            ranking_dict[idx]["backup_plan"] = backup_plan
            ranking_dict[idx]["backup_FRS"]  = self.backup_policy.FRS

        if best_obj_val == OBJ_VAL_INIT:
            # This plan would not be used anyway
            best_plan = plans[0]
            best_plan_idx = 0
        
        self.pretty_print_ranking(ranking_dict)

            # NOTE: maybe change the mode of the backup plan to target mode?
        
        return best_plan, best_plan_idx, ranking_dict
    
    def pretty_print_ranking(self, ranking_dict):
        print("====================================================================")
        print("Ranking of the Plans")
        for idx in range(len(ranking_dict)):
            print(f'{idx}: Plan {ranking_dict[idx]["plan"].x_des[-1]} | ObjVal {ranking_dict[idx]["obj_val"]} | Safe {ranking_dict[idx]["safe"]}')
        print("====================================================================")

class IdentityDiffusionPolicyArm(SafeDiffusionPolicyArm):
    def monitor_and_compute_backup_plan(self, plan, ob, goal):
        raise NotImplementedError

class IdentityBackupPolicyArm(SafeDiffusionPolicyArm):
    def monitor_and_compute_backup_plan(self, plan, ob, goal=None):
        raise NotImplementedError