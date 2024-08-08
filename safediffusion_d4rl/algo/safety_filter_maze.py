from copy import deepcopy

import torch
import numpy as np
import einops


from robomimic.algo.diffusers import DiffusionPolicyUNet as DiffusersUNet
from robomimic.utils.obs_utils import unnormalize_dict
from safediffusion.algo.plan import ReferenceTrajectory
from safediffusion.algo.safety_filter import SafeDiffusionPolicy


class SafeDiffusionPolicyMaze(SafeDiffusionPolicy):
    def __init__(self, rollout_policy, backup_policy, dt_action, predictor, **config):
        """
        The reference trajectory is defined in the 2D world space (x, y)
            x_des  = (x, y)
            dx_des = (v_x, v_y)
        """
        super().__init__(rollout_policy, backup_policy, dt_action=dt_action, **config)

        # object for state prediction
        # TODO: Make this as a third-party object whose `predict(init_state, actions)` returns the state predictions
        self.predictor = predictor

    @property
    def safety_critical_state_keys(self):
        return ["flat"]
    
    def update_backup_policy_weight(self):
        """
        Update the weight of the backup policy
        """
        weight_dict = dict(goal = 0.0, projection = 1.0)
        self.backup_policy.update_weight(weight_dict)
    
    # def get_plan_from_nominal_policy(self, obs, goal):
    #     """
    #     Get the plan from the nominal policy

    #     Args:
    #         obs: observation dictionary
    #         goal: goal dictionary

    #     Returns:
    #         plan   : ReferenceTrajectory object length of Tp-To+2,
    #         actions: (Tp-To+1, Da) np.ndarray

    #     NOTE: The plan dimension lies in the qpos, not world pos
    #     TODO: Remove the :2, 2: make it as a parameter
    #     """
    #     # preprocess the observation dictionary compatible to the rollout policy
    #     obs_perf = {k: obs[k] for k in self.rollout_policy_obs_keys}

    #     # clear the action queue so that diffusion policy does not previously computed actions
    #     self.rollout_policy.policy.action_queue.clear()

    #     # call this function to compute the internal (action, states) predictions
    #     _ = self.rollout_policy(obs_perf, goal)

    #     # initial state and actions
    #     init_state = obs["flat"][-1, :] # The last row is the current observation
    #     actions    = self.rollout_policy.policy._action_sequence_ref[0].cpu().numpy()

    #     # state predictions
    #     states     = self.predict_states_from_actions(init_state, actions)

    #     # wrap it to the reference trajectory
    #     n_actions = actions.shape[0]
    #     t_des     = torch.linspace(0, self.dt_action*n_actions, n_actions+1)
    #     x_des     = states[:, :2]
    #     dx_des    = states[:, 2:]

    #     plan = ReferenceTrajectory(t_des=t_des, x_des=x_des, dx_des=dx_des, dtype=self.dtype, device=self.device)

    #     return plan, actions
    
    def get_plan_from_nominal_policy(self, obs, goal=None, num_samples=1):
        """
        Coupling diffusion policy with guiding of the safety filter

        Returns:
            plan
            actions
            info: dict containing head_plan and backup_plan
        
        NOTE: Technically, the backup plan is computed along with this procedure.
        However, we compute again since we are going to replace the trajopt with some
        other neural network later.
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
        Given the reference trajectory and observation dictionary, 
        postprocess to actions for the robomimic environment.
        TODO: Implement the Map: backup_policy.state_dim -> rollout_policy.action_dim
        """
        Kp = 30
        Kd = 0.5

        x = ob["flat"][-1, :2]
        dx = ob["flat"][-1, 2:]

        state = np.concatenate([x, dx])

        actions = []
        states  = [state]
        # TODO: change to track the next waypoint
        for i in range(len(reference_traj)-1):
            x_des = reference_traj[i+1][1]
            action = Kp*(x_des - state[:2]) - Kd*state[2:]
            action = np.expand_dims(action, 0)
            
            plan = self.predictor(np.expand_dims(state, 0), 
                                  np.expand_dims(action, 0))[0]
            
            next_state = np.concatenate([plan.x_des[-1], plan.dx_des[-1]])
            state = next_state
            
            actions.append(action[0])
            states.append(state)

        actions = np.stack(actions, 0)
        
        states  = np.stack(states, 0)
        error = states[:, :2] - reference_traj.x_des.cpu().numpy()
        error = np.linalg.norm(error, axis=1)

        # self.disp(f"Backup Plan Tracking Error: {error.max()}")

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

class IdentityDiffusionPolicyMaze(SafeDiffusionPolicyMaze):
    def monitor_and_compute_backup_plan(self, plan, ob, goal):
        info = dict()
        info["head_plan"] = True
        
        # This would be never used. We just need placeholder
        info["backup_plan"] = ReferenceTrajectory(t_des = self.backup_policy.t_des,
                                                  x_des = torch.zeros(self.backup_policy.t_des.shape[0], 2))
        
        info["backup_plan"].stamp_trajectory_parameter(torch.zeros(4,))
        

        return info
    
class SafeDiffuserMaze(SafeDiffusionPolicyMaze):
    def __init__(self, rollout_policy, backup_policy, dt_action, predictor, **config):
        """
        The reference trajectory is defined in the 2D world space (x, y)
            x_des  = (x, y)
            dx_des = (v_x, v_y)
        """
        assert isinstance(rollout_policy.policy, DiffusersUNet)
        super().__init__(rollout_policy, backup_policy, dt_action=dt_action, predictor=predictor, **config)

    def get_plan_from_nominal_policy(self, obs, goal=None, num_samples=1):
        """
        Coupling diffusion policy with guiding of the safety filter

        Returns:
            plan
            actions
            info: dict containing head_plan and backup_plan
        
        NOTE: Technically, the backup plan is computed along with this procedure.
        However, we compute again since we are going to replace the trajopt with some
        other neural network later.
        """
        # Preprocess the observation dictionary compatible to the rollout policy
        obs_perf = {k: obs[k] for k in self.rollout_policy_obs_keys}
        obs_perf = self.rollout_policy._prepare_observation(obs_perf)
        if goal is not None:
            goal_perf = {k: goal[k] for k in self.rollout_policy_obs_keys}
            goal_perf = self.rollout_policy._prepare_observation(goal_perf)

        # Compute the multiple trajectories with the policy
        # NOTE: no need to clear the action queue anymore
        actions_candidates, prediction_candidates = self.rollout_policy.policy.get_multi_actions(
                                                        obs_perf, goal_perf, num_trajectories=num_samples)
        
        prediction_dict = {"flat": prediction_candidates.cpu().numpy()}
        prediction_dict = unnormalize_dict(prediction_dict, self.rollout_policy.obs_normalization_stats)
        prediction      = prediction_dict["flat"]

        T = actions_candidates.shape[1]

        plans_candidates = []
        for idx in range(num_samples):
            t_des = torch.linspace(0, self.dt_action*(T-1), T)
            x_des = prediction[idx][:, :2]
            dx_des = prediction[idx][:, 2:]

            plan = ReferenceTrajectory(t_des=t_des, x_des=x_des, dx_des=dx_des, dtype=self.dtype, device=self.device)
            plans_candidates.append(plan)

        actions_candidates = np.stack([actions.cpu().numpy() for actions in actions_candidates], 0) # (B, Tp, Da)
        
        # # State prediction model
        
        init_states = einops.repeat(obs["flat"][-1, :], 'd -> repeat d', repeat=num_samples) # (B, Dx)
        self.plan_true = self.predictor(init_states, actions_candidates)

        # Ranking the plans
        best_plan, best_plan_idx, ranking_info = self.rank_plans(plans_candidates, obs, goal)
        best_actions = actions_candidates[best_plan_idx]
        self.ranking_info = ranking_info
        
        return best_plan, best_actions

        

        
