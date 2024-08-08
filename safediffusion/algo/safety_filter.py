"""
Implementation of the safety filter algorithm.
"""
import abc
import time
from copy import deepcopy
from typing import Any
from collections import deque

import torch
import numpy as np

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
        self.backup_policy_weight_keys = self.backup_policy.weight_dict.keys()
        
        # safety filter parameter
        self.n_head            = config["filter"]["n_head"]    # length of the head actions (each unit corresponds to action applied to the environment)
        self.max_init_attempts = config["filter"]["max_init_attempts"] # maximum number of attempts to initialize the backup plan
        self.verbose           = config["filter"]["verbose"]

        # internal data structure & data
        self._backup_plan   = None                          # backup plan (reference trajectory)
        self._actions_queue = deque()                       # queue of safe actions
        self.nominal_plan   = None
        self.intervened     = False
        self.stuck          = False                         # robot is stuck - no feasible backup plan could be found

    def __call__(self, ob, goal=None, **kwargs):
        """
        Main entry point of our receding-horizon safety filter.

        If the action queue is empty, we generate a new plan and check the safety.

        NOTE: In this version, the nominal policy generates the plan based on the o_{t}.
              Ideally, the policy should generate the plan based on the o_{t+T_a}. 
              This version is equivalent to assuming the perfect state prediction.
        """
        if len(self._actions_queue) == 0:
            if self.has_no_backup_plan():
                self.initialize_backup_plan(ob, goal)

            (plan, actions) = self.get_plan_from_nominal_policy(ob, goal, **kwargs)
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

            # summarize
            k_backup = self._backup_plan.get_trajectory_parameter().cpu().numpy()
            msg = f'Using Diffusion: {not self.intervened} | ' + \
                  f'Head Plan Safety: {info["head_plan"]}  | ' + \
                  f'{len(self._backup_plan)} backup actions left: (k={np.round(k_backup, 2)})'
            self.disp(msg)

        return self._actions_queue.popleft()
    
    def has_no_backup_plan(self):
        """
        Check whether the backup plan is empty
        """
        return self._backup_plan is None or len(self._backup_plan) <= 1
    
    @abc.abstractmethod
    def safety_critical_state_keys(self):
        """
        Return the keys of the safety-critical states
        """
        raise NotImplementedError
    
    def check():
        """
        Check the compatibility of the performance policy, backup policy, observation, goal dictionary

        TODO: PLEASE IMPLEMENT THIS FUNCTION
        """
        pass

    def initialize_backup_plan(self, ob, goal):
        """
        Initialize the backup plan from the initial state of the environment

        Args
            ob   : observation dictionary
            goal : goal dictionary
        """
        ob_init, goal_init = self.process_dicts_for_backup_policy(ob=ob, goal=goal, plan=None)

        # set backup policy objective function null, trajectory optimization finds the feasible solution only
        weight_dict = {k: 0.0 for k in self.backup_policy_weight_keys}
        self.backup_policy.update_weight(weight_dict)
            
        count = 0
        while count < self.max_init_attempts:
            backup_plan, info = self.backup_policy(obs_dict              = ob_init, 
                                                   goal_dict             = goal_init, 
                                                   random_initialization = True)
            if info["status"] == 0:
                self.set_backup_plan(backup_plan)
                break

            count += 1
        
        if count == self.max_init_attempts:
            self.stuck = True
    
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
            backup_plan, _ = self.compute_backup_plan(tail_plan, ob, goal)
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

        ob_backup, goal_backup = self.process_dicts_for_backup_policy(ob=ob, goal=goal, plan=plan)

        self.update_backup_policy_weight()
        backup_plan, info = self.backup_policy(obs_dict=ob_backup, goal_dict=goal_backup)

        if info["status"] == 0:
            return backup_plan, info
        else:
            return None, info

    def disp(self, msg):
        print(f"[{self.__class__.__name__}]:      {msg}")

    # ------------------------------------------------------------------------------------ #
    # ------------------------- Abstract class to override ------------------------------  #
    # ------------------------------------------------------------------------------------ #
    @abc.abstractmethod
    def update_backup_policy_weight(self):
        """
        Update the weight dictionary of the backup policy according to the internal status.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def process_dicts_for_backup_policy(self, ob, goal, plan):
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
        # initialize the rollout policy
        self.rollout_policy.start_episode()

        # initialize the backup policy
        self.backup_policy.start_episode()

        # initialize the safety filter internal variables
        self.clear_backup_plan()
        self._actions_queue.clear()
        self.nominal_plan   = None
        self.intervened     = False
        self.stuck          = False
        self.ranking_info   = None

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
    def rank_plans(plans, obs, goal):
        """
        Rank the plans based on the safety and performance criteria

        This is optional
        """
        raise NotImplementedError