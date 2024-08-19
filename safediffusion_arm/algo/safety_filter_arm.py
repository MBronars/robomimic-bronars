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
                 rollout_controller_name,
                 backup_controller_name,
                 **config
                 ):
        """
            rollout_policy (RolloutPolicy), the nominal performance policy
            dt_action (float): time interval for calling the 
        
        """
        super().__init__(rollout_policy, backup_policy, dt_action=dt_action, **config)

        # object for state prediction
        # TODO: Make this as a third-party object whose `predict(init_state, actions)` returns the state predictions
        self.predictor = predictor
        self.rollout_controller_name = rollout_controller_name
        self.backup_controller_name = backup_controller_name

        # internal status
        self.action_idx = 0

    # --------------------------------------------- #
    # Abstract functions for SafetyFilter
    # --------------------------------------------- #
    @property
    def safety_critical_state_keys(self):
        return ["robot0_joint_pos", "robot0_joint_vel"]
    
    def __call__(self, ob, goal=None, **kwargs):
        """
        Main entry point of our receding-horizon safety filter.

        If the action queue is empty, we generate a new plan and check the safety.

        NOTE: In this version, the nominal policy generates the plan based on the o_{t}.
              Ideally, the policy should generate the plan based on the o_{t+T_a}. 
              This version is equivalent to assuming the perfect state prediction.

        TODO: Rewrite this function -- when we use nominal policy, action queue is fine.
        However, when we use backup policy, action should be decided every time.
        """
        # if there is no backup plan, the agent is static, initialize the backup plan.
        if self.has_no_backup_plan():
            self.initialize_backup_plan(ob, goal)
            self.action_idx = 0

        # for every period (`self.n_head`), update the nominal plan and backup plan
        # at time t, compute new plan t..t+Ta, and decide which strategy to take for next Ta horizon
        if self.action_idx % self.n_head == 0:    
            self.update(ob, goal, **kwargs)
            self.action_idx = 0

        # get action that can be sent to robomimic env.
        ac, controller_to_use = self.get_action(ob)
        self.action_idx  += 1

        return ac, controller_to_use
    
    def pop_backup_plan(self, horizon):
        """
        Pop the backup plan for the amount of the horizon
        """
        # retrieve the plan
        plan = self._backup_plan[self._backup_plan.t_des <= horizon]
        
        # update the remaining backup plan
        self._backup_plan = self._backup_plan[self._backup_plan.t_des >= horizon]
        self._backup_plan.set_start_time(0)
        
        return plan
    
    def update(self, ob, goal, **kwargs):
        (plan, actions) = self.get_plan_from_nominal_policy(ob, goal, **kwargs)
        info            = self.monitor_and_compute_backup_plan(plan, ob, goal)

        # Definition of safety in this framework
        is_safe     = info["head_plan"] and info["backup_plan"] is not None

        if is_safe:
            self.clear_backup_plan()
            self.set_backup_plan(info["backup_plan"])
        else:
            horizon = self.n_head * self.dt_action
            self.backup_plan_cur = self.pop_backup_plan(horizon)

        # update internal status
        self.nominal_actions = actions
        self.nominal_plan    = plan
        self.intervened      = not is_safe

        # summarize the update
        k_backup = self._backup_plan.get_trajectory_parameter().cpu().numpy()
        msg = f'Using Diffusion: {not self.intervened} | ' + \
                f'Head Plan Safety: {info["head_plan"]}  | ' + \
                f'{len(self._backup_plan)} backup actions left: (k={np.round(k_backup, 2)})'
        self.disp(msg)
    

    def get_action(self, ob):
        if self.intervened:
            # using backup planner
            action_gripper = self.nominal_actions[self.action_idx, -1]
            action_arm     = self.backup_policy.get_action_from_plan_at_t(
                t_des    = (self.action_idx+1) * self.dt_action,
                plan     = self.backup_plan_cur,
                obs_dict = ob
            )
            action = np.concatenate([action_arm, [action_gripper]])
            controller_to_use = self.backup_controller_name
        
        else:
            action = self.nominal_actions[self.action_idx]
            controller_to_use = self.rollout_controller_name
        
        return action, controller_to_use
    
    def update_backup_policy_weight(self):
        """
        Update the weight dictionary of the backup policy according to the internal status.
        This function reflects the strategy of the objective of the backup planner.

        NOTE: If the policy is stucked, change to the goal mode
        TODO: Check if all keys of the weight dict is in the backup planner
        """
        weight_dict = dict(qpos_goal       = 1.0, 
                           qpos_projection = 0.0)

        # check that we configured all the objectives of backup policy
        for key in self.backup_policy_weight_keys:
            assert key in weight_dict.keys()
        
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
        else:
            goal_perf = None

        # Compute the multiple trajectories with the policy
        actions_candidates = self.rollout_policy.policy.get_multi_actions(obs_perf, goal_perf, num_trajectories=num_samples)
        actions_candidates = np.stack([actions.cpu().numpy() for actions in actions_candidates], 0) # (B, Tp, Da)
        
        # State prediction model
        init_states = einops.repeat(obs["state_dict"]["states"], 'd -> repeat d', repeat=num_samples) # (B, Dx)
        plans_candidates = self.predictor(init_states, actions_candidates)

        # TODO: Ranking the plans
        # best_plan, best_plan_idx, ranking_info = self.rank_plans(plans_candidates, obs, goal)
        # best_actions = actions_candidates[best_plan_idx]
        # self.ranking_info = ranking_info
        
        return plans_candidates[0], actions_candidates[0]
        
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
        raise NotImplementedError
    
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
            goal_dict: goal dictionary where goal_dict[backup_planner.objective_tag] = dict()

        # TODO: Since the safety filter is the wrapper of the backup policy, it would be good if we can parse the
        #       available objective function keys from the backup policy
        """
        assert plan is None or isinstance(plan, ReferenceTrajectory)

        obs_dict  = deepcopy(ob)
        goal_dict = deepcopy(goal)

        if goal_dict is None:
            goal_dict = {}

        initialize_mode = (plan is None)

        if initialize_mode:
            # When there is no nominal plan available, compute the backup plan starting at current state
            for key in self.safety_critical_state_keys:
                if obs_dict[key].ndim == 2:
                    obs_dict[key] = ob[key][-1, :]

        else:
            # When nominal plan is available, compute the backup plan at the initial state of the nominal plan
            # TODO
            # 1) robot_zonotope is not used, hence not updated in this code. Refer to safemaze if we need
            # 2) does not consider envstackwrapper
            obs_dict["robot0_joint_pos"] = plan[0][1]
            obs_dict["robot0_joint_vel"] = plan[0][2]

            # update goal dictionary to provide info. about the objective function
            # HACK
            # goal_dict["qpos_goal"]       = dict(qgoal=torch.zeros(7,))
            goal_dict["qpos_goal"]       = dict(qgoal=torch.tensor([ torch.pi/2, -torch.pi/2, torch.pi/4, 0, 0, 0, 0]))
            goal_dict["qpos_goal"]       = dict(qgoal=torch.tensor([-torch.pi/2, -torch.pi/2, torch.pi/4, 0, 0, 0, 0]))
            goal_dict["qpos_projection"] = dict(plan=plan)

            for key in self.backup_policy_weight_keys:
                assert key in goal_dict.keys(), \
                f"The objective [{key}] for backup policy should be specified in goal_dict."

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
    """
    Implementation of the wrapper of the diffusion policy that always returns the actions
    from the diffusion policy
    """
    def monitor_and_compute_backup_plan(self, plan, ob, goal):
        # fill the dummy info
        info = dict()

        info["head_plan"] = True
        
        # This would be never used. We just need placeholder
        info["backup_plan"] = ReferenceTrajectory(t_des = self.backup_policy.t_des,
                                                  x_des = torch.zeros(self.backup_policy.n_timestep, self.backup_policy.n_state))
        info["backup_plan"].stamp_trajectory_parameter(torch.zeros(self.backup_policy.n_param,))
        
        return info

class IdentityBackupPolicyArm(SafeDiffusionPolicyArm):
    def monitor_and_compute_backup_plan(self, plan, ob, goal=None):
        raise NotImplementedError