"""
Implementation of pieces of trajectory
"""

import numpy as np
from scipy.interpolate import interp1d

def match_trajectories(T_des, *args):
    """
    Given desired sample times T_des and any number of time vectors (1D arrays)
    and associated trajectories (2D arrays), reinterpolate the given trajectories
    linearly at the desired times.

    If T_des[0] < T_i[0], then the output Z_i is pre-padded with Z_i[:,0],
    and similarly if T_des[-1] > T_i[-1].

    Parameters:
        T_des (1D array): Desired sample times.
        *args: Variable length argument list where pairs of time vectors and 
               trajectories are expected, followed optionally by the interpolation type.

    Returns:
        List of interpolated trajectories.
    """
    if isinstance(args[-1], str):
        interp_type = args[-1]
        args = args[:-1]
    else:
        interp_type = 'linear'
        
    if np.isscalar(T_des):
        T_des = np.array([T_des])
    
    results = []
    for i in range(0, len(args), 2):
        T = np.array(args[i])
        Z = np.array(args[i+1])
        
        # If T_des exceeds the bounds of T, pad T and Z accordingly
        if T_des[0] < T[0]:
            T = np.insert(T, 0, T_des[0])
            Z = np.insert(Z, 0, Z[:, 0], axis=1)
        
        if T_des[-1] > T[-1]:
            T = np.append(T, T_des[-1])
            Z = np.append(Z, Z[:, -1].reshape(-1, 1), axis=1)
        
        if len(T) == 1 and np.array_equal(T_des, T):
            results.append(Z)
        else:
            interpolator = interp1d(T, Z, kind=interp_type, axis=1, fill_value="extrapolate")
            Z_des = interpolator(T_des)
            results.append(Z_des)
    
    return results

def traj_uniform_acc(t, x0, v0, a):
    """
    Implementation of trajectory with uniform acceleration

    Args:
        t : (B, 1), time
        x0: (B, n_state), initial state
        v0: (B, n_state), initial velocity
        a : (B, n_state), acceleration
    """
    return np.expand_dims(x0, axis=-1) + np.outer(v0, t) + np.outer(a, t**2) / 2
