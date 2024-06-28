"""
Implementation of the safety filter algorithm.
"""
import abc
from typing import Any
from robomimic.algo import RolloutPolicy
from safediffusion.algo.plan import ReferenceTrajectory
from safediffusion.algo.planner_base import ParameterizedPlanner
from collections import deque

class SafetyFilter(RolloutPolicy, abc.ABC):
    """
    Wrapper of the policy (or rolloutpolicy) to make the policy safe
    """
    def __init__(self, rollout_policy, backup_policy, **config):
        assert isinstance(rollout_policy, RolloutPolicy)
        assert isinstance(backup_policy, ParameterizedPlanner)

        self.rollout_policy = rollout_policy
        self.backup_policy  = backup_policy
        
        # Safety filter parameter
        self.n_head         = config["filter"]["n_head"]    # length of the head plan

        # Safety filter data structure
        self._backup_plan   = None                          # backup plan (reference trajectory)
        self._actions_queue = deque()                       # queue of safe actions
    
    def __call__(self, ob, goal=None):
        """
        Main entry point of our receding-horizon safety filter.

        If the action queue is empty, we generate a new plan and check the safety.
        """
        if len(self._actions_queue) == 0:
            actions     = self.rollout_policy(ob, goal)
            plan        = self.preprocess_to_reference_traj(ob, actions)
            info        = self.monitor_and_compute_backup_plan(plan, ob, goal)

            # Definition of safety in this framework
            is_safe     = info["head_plan"] and info["backup_plan"] is not None

            if is_safe:
                self.clear_backup_plan()
                self.set_backup_plan(info["backup_plan"])
                action_safe = actions[:self.n_head]

            else:
                backup_plan = self.pop_backup_plan(self.n_head)
                action_safe = self.postprocess_to_actions(backup_plan)
            
            self._actions_queue.extend(action_safe)

        return self._actions_queue.popleft()
    
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
        head_plan = plan[:self.n_head]
        is_head_plan_safe = self.check_head_plan_safety(ob, head_plan)
        info["head_plan"] = is_head_plan_safe

        # Checking 2nd criteria
        if is_head_plan_safe:
            tail_plan   = plan[self.n_head:].set_start_time(0)
            backup_plan = self.compute_backup_plan(tail_plan)
            info["backup_plan"] = backup_plan

        return info
    
    def compute_backup_plan(self, ob, plan, goal):
        """ 
        Compute the backup plan from the initial state of the tail plan.
        This does by projecting the plan to the safe parameterized trajectory.
        It returns backup plan in (ReferenceTrajectory) if exists. If not, it returns None.
        
        Args:
            plan: (ReferenceTrajectory) The reference plan to compute the backup plan
        
        Returns:
            backup_plan
        """
        # TODO: wraps the (ob, plan, goal) into the (ob, goal) format for backup_policy
        backup_plan = self.backup_policy(ob, goal=goal)

        return backup_plan

    # ------------------------------------------------------------------------------------ #
    # ------------------------- Abstract class to override ------------------------------  #
    # ------------------------------------------------------------------------------------ #
    @abc.abstractmethod
    def preprocess_to_reference_traj(self, obs, actions):
        """
        Given the observation and action sequences, preprocess to reference trajectory.

        Implementation of prediction function (o_{t}, a_{t..t+P}) -> (s_{t..t+T_p})

        Args
            obs     : (B, T_o, D_o) np.ndarray
            actions : (B, T_p, D_a) np.ndarray
        
        Returns
            reference_traj: (B, 1) ReferenceTrajectory array
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def postprocess_to_actions(self, reference_traj):
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
