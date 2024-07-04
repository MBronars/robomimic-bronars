"""
Implementation of the safety filter algorithm.
"""
import abc
from copy import deepcopy
from typing import Any
from collections import deque

import torch

from robomimic.algo import RolloutPolicy
from safediffusion.algo.plan import ReferenceTrajectory
from safediffusion.algo.planner_base import ParameterizedPlanner


class SafetyFilter(RolloutPolicy, abc.ABC):
    """
    Wrapper of the policy (or rolloutpolicy) to make the policy safe

    TODO: Think of how to load dtype and device wisely
    """
    def __init__(self, rollout_policy, backup_policy, dt_action,
                 dtype  = torch.float32,
                 device = torch.device("cpu"),
                **config):
        
        assert isinstance(rollout_policy, RolloutPolicy)
        assert isinstance(backup_policy, ParameterizedPlanner)

        self.rollout_policy = rollout_policy
        self.backup_policy  = backup_policy
        self.dt_action      = dt_action                     # action time step
        self.dtype          = dtype
        self.device         = device
        
        # performance policy-related
        self.rollout_policy_obs_keys = self.rollout_policy.policy.global_config.all_obs_keys
        
        # backup policy-related
        # TODO: fill-in the backup policy observation keys
        
        # Safety filter parameter
        self.n_head         = config["filter"]["n_head"]    # length of the head actions (each unit corresponds to action applied to the environment)

        # Safety filter data structure
        self._backup_plan   = None                          # backup plan (reference trajectory)
        self._actions_queue = deque()                       # queue of safe actions
    
    def __call__(self, ob, goal=None):
        """
        Main entry point of our receding-horizon safety filter.

        If the action queue is empty, we generate a new plan and check the safety.
        """
        if len(self._actions_queue) == 0:
            (plan, actions) = self.get_plan_from_nominal_policy(ob, goal)
            info            = self.monitor_and_compute_backup_plan(plan, ob, goal)

            # Definition of safety in this framework
            is_safe     = info["head_plan"] and info["backup_plan"] is not None

            if is_safe:
                self.clear_backup_plan()
                self.set_backup_plan(info["backup_plan"])
                action_safe = actions[:self.n_head]

            else:
                backup_plan = self.pop_backup_plan(self.n_head)
                action_safe = self.postprocess_to_actions(backup_plan, ob)
            
            self._actions_queue.extend(action_safe)

            # Log useful variables for rendering purposes
            self.nominal_plan = plan
            self.intervened   = not is_safe

        return self._actions_queue.popleft()
    
    def check():
        """
        Check the compatibility of the performance policy, backup policy, observation, goal dictionary

        TODO: PLEASE IMPLEMENT THIS FUNCTION
        """
        pass
    
    def monitor_and_compute_backup_plan(self, plan, ob, goal=None):
        """
        Monitor the safety of the head plan and compute the backup plan if necessary

        Args
            plan : (ReferenceTrajectory) The reference plan to monitor and compute the backup plan
        
        Returns
            info : dict containing the safety information
        """
        assert isinstance(plan, ReferenceTrajectory)
        
        info = dict()

        # Checking 1st criteria
        head_plan = plan[:self.n_head+1]
        is_head_plan_safe = self.check_head_plan_safety(head_plan, ob)
        info["head_plan"] = is_head_plan_safe

        # Checking 2nd criteria
        if is_head_plan_safe:
            tail_plan = plan[self.n_head:]
            tail_plan.set_start_time(0)
            backup_plan = self.compute_backup_plan(tail_plan, ob, goal)
            info["backup_plan"] = backup_plan

        return info
    
    def compute_backup_plan(self, plan, ob, goal):
        """ 
        Compute the backup plan from the initial state of the tail plan.
        This does by projecting the plan to the safe parameterized trajectory.
        It returns backup plan in (ReferenceTrajectory) if exists. If not, it returns None.
        
        Args:
            plan: (ReferenceTrajectory) The reference plan to compute the backup plan
        
        Returns:
            backup_plan
        """

        ob_backup, goal_backup = self.process_dicts_for_backup_policy(plan, ob, goal)

        backup_plan, info = self.backup_policy(obs_dict=ob_backup, goal_dict=goal_backup)

        if info["status"] == 0:
            return backup_plan
        else:
            return None

    # ------------------------------------------------------------------------------------ #
    # ------------------------- Abstract class to override ------------------------------  #
    # ------------------------------------------------------------------------------------ #
    @abc.abstractmethod
    def process_dicts_for_backup_policy(self, plan, ob, goal):
        """
        Given the plan, observation dictionary, and goal dictionary from the environment (Safety Wrapper)
        preprocess the data to be compatible with the backup policy

        Args
            plan : ReferenceTrajectory object
            ob   : observation dictionary
            goal : goal dictionary
        
        Returns
            obs_dict : observation dictionary
            goal_dict: goal dictionary

        NOTE: This function should be written based on 1) backup policy's input keys and 2) ReferenceTrajectory object
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_plan_from_nominal_policy(self, ob, goal):
        """
        Get the plan from the nominal policy

        Args
            ob   : (B, 1) Observation array
            goal : (B, 1) Goal array
        
        Returns
            plan   : (B, 1) ReferenceTrajectory array
            actions: (B, T_p, D_a) np.ndarray
        """
        raise NotImplementedError
    
    # @abc.abstractmethod
    # def preprocess_to_reference_traj(self, obs, actions):
    #     """
    #     Given the observation and action sequences, preprocess to reference trajectory.

    #     Implementation of prediction function (o_{t}, a_{t..t+P}) -> (s_{t..t+T_p})

    #     Args
    #         obs     : (B, T_o, D_o) np.ndarray
    #         actions : (B, T_p, D_a) np.ndarray
        
    #     Returns
    #         reference_traj: (B, 1) ReferenceTrajectory array
    #     """
    #     raise NotImplementedError
    
    @abc.abstractmethod
    def postprocess_to_actions(self, reference_traj, ob):
        """
        Given the reference trajectory, postprocess to executable action that can be sent to the environment

        Implementation of prediction function (s_{t..t+T_p}) -> (a_{t..t+P})

        Args
            reference_traj : (B, 1) ReferenceTrajectory array
        
        Returns
            actions : (B, T_p, D_a) np.ndarray
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def check_head_plan_safety(self, ob, plan):
        """
        Check the safety of the head plan

        Args
            plan : (B, 1) ReferenceTrajectory array
        
        Returns
            is_safe : (B, 1) np.ndarray
        """
        raise NotImplementedError

    # ------------------------------------------------------------------------------------ #
    # ------------------------- Backup Plan Management ----------------------------------  #
    # ------------------------------------------------------------------------------------ #
    def clear_backup_plan(self):
        """
        Clear the backup plan
        """
        self._backup_plan = None
    
    def set_backup_plan(self, plan):
        """
        Set the backup plan
        """
        self._backup_plan = plan
    
    def pop_backup_plan(self, n):
        """
        Pop the backup plan of n length
        """
        plan = self._backup_plan[:n+1]
        self._backup_plan = self._backup_plan[n:]

        return plan
    
    def get_plan_backup_trajparam(self):
        """
        Get the trajectory parameter of the backup plan
        """
        return self._backup_plan.get_trajectory_parameter()

    # ------------------------------------------------------------------------------------ #
    # ------------------------- Compatibility with RolloutPolicy ------------------------- #
    # ------------------------------------------------------------------------------------ #
    def start_episode(self):
        """
        Prepare the policy to start a new rollout.
        """
        self.rollout_policy.start_episode()
        self.clear_backup_plan()

    def _prepare_observation(self, ob):
        """
        Prepare raw observation dict from environment for policy.

        Args:
            ob (dict): single observation dictionary from environment (no batch dimension, 
                and np.array values for each key)
        """
        return self.rollout_policy._prepare_observation(ob)

    def __repr__(self):
        """Pretty print network description"""
        return self.rollout_policy.__repr__()


class SafeDiffusionPolicy(SafetyFilter):
    """
    Diffusion-specific algorithms (e.g., guiding)
    """
    def __call__(self, ob, goal=None):
        # TODO: Implement guiding here
        return super().__call__(ob, goal)
