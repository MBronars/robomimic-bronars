import abc

import torch
import numpy as np

from safediffusion.algo.plan import ReferenceTrajectory

class StatePredictor:
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 dt,
                 device = torch.device('cpu'),
                 dtype  = torch.float):
        
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.dt         = dt
        self.dtype      = dtype
        self.device     = device

    @abc.abstractmethod
    def __call__(self, x0, actions):
        """
        Args:
            x0:      the array of initial states (B, state_dim)
            actions: the array of actions        (B, T, action_dim)

        Returns:
            predictions: the array of states     (B, T+1, state_dim)
        """
        raise NotImplementedError