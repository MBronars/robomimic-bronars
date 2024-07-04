from copy import deepcopy

import torch
import numpy as np

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
        self.predictor.reset()
    
    def get_plan_from_nominal_policy(self, obs, goal):
        """
        Get the plan from the nominal policy

        Args:
            obs: observation dictionary
            goal: goal dictionary

        Returns:
            plan   : ReferenceTrajectory object length of Tp-To+2,
            actions: (Tp-To+1, Da) np.ndarray

        NOTE: The plan dimension lies in the qpos, not world pos
        TODO: Remove the :2, 2: make it as a parameter
        """
        # preprocess the observation dictionary compatible to the rollout policy
        obs_perf = {k: obs[k] for k in self.rollout_policy_obs_keys}

        # clear the action queue so that diffusion policy does not previously computed actions
        self.rollout_policy.policy.action_queue.clear()

        # call this function to compute the internal (action, states) predictions
        _ = self.rollout_policy(obs_perf, goal)

        # initial state and actions
        init_state = obs["flat"][-1, :] # The last row is the current observation
        actions    = self.rollout_policy.policy._action_sequence_ref[0].cpu().numpy()

        # state predictions
        states     = self.predict_states_from_actions(init_state, actions)

        # wrap it to the reference trajectory
        n_actions = actions.shape[0]
        t_des     = torch.linspace(0, self.dt_action*n_actions, n_actions+1)
        x_des     = states[:, :2]
        dx_des    = states[:, 2:]

        plan = ReferenceTrajectory(t_des=t_des, x_des=x_des, dx_des=dx_des, dtype=self.dtype, device=self.device)

        return plan, actions
        
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
            action = action.unsqueeze(0).cpu().numpy()
            
            next_state = self.predict_states_from_actions(state, action)

            state = next_state[-1]
            
            actions.append(action[0])
            states.append(state)

        actions = np.stack(actions, 0)
        
        states  = np.stack(states, 0)
        error = states[:, :2] - reference_traj.x_des.cpu().numpy()
        error = np.linalg.norm(error, axis=1)

        print("Error: ", error)

        return actions
    
    def check_head_plan_safety(self, plan, ob):
        """
        TODO: Implement the safety verifier, maybe this better suit as a method of the Planner class
        """
        collision = self.backup_policy.collision_check(plan, ob)
        return not collision
    
    def predict_states_from_actions(self, init_state, actions):
        """
        HACK: Roll-out the dummy environments to get the state predictions

        Args
            init_state: (D_s, ) np.ndarray
            actions   : (T_p, D_a) np.ndarray
        """
        # Initialize your dummy env with the initial state
        _ = self.predictor.reset_to({"states": np.concatenate([[0], init_state])})

        states = [init_state]
        
        # rollout the actions
        for action in actions:
            obs, _, _ , _ = self.predictor.step(action)
            states.append(obs["flat"][-1, :])

        states = np.stack(states, 0)

        return states
    
    def process_dicts_for_backup_policy(self, plan, ob, goal):
        """
        Given the plan, observation dictionary, and goal dictionary from the environment,
        preprocess the data to be compatible with the backup policy

        Args
            plan : ReferenceTrajectory object
            ob   : observation dictionary
            goal : goal dictionary
        
        Returns
            obs_dict : observation dictionary
            goal_dict: goal dictionary
        """
        obs_dict = deepcopy(ob)
        goal_dict = deepcopy(goal)

        # HACK: This code requires knowledge of what ReferenceTrajectory looks like: (t_des, x_des, dx_des)
        obs_dict["flat"] = torch.hstack([plan[0][1], plan[0][2]]) 
        goal_dict["reference_trajectory"] = plan

        return obs_dict, goal_dict
        

        
