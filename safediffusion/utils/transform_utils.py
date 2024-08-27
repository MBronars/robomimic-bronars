"""
Useful transformation-related (rotation, translation) functions

All functions assume the input of numpy array
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def make_pose(rot, pos):
    """
    Args:
        rot (np.array) 3x3 rotation matrix
        pos (np.array) 3,  position vector

    Returns
        T (np.array) 4x4 transformation matrix
    """
    T = np.eye(4)

    T[:3, :3] = rot
    T[:3, 3]  = pos

    return T

def pose_to_rot_pos(T):
    rot = T[:3, :3]
    pos = T[:3,  3]
    
    return rot, pos

def quaternion_to_rotation_matrix(quat):
    """
    Convert a quaternion into a 3x3 rotation matrix.
    
    Args:
        quat (list or torch.Tensor): Quaternion in the form [w, x, y, z]

    Returns:
        R (torch.Tensor): 3x3 rotation matrix
    """
    w, x, y, z = quat

    # Calculate the rotation matrix elements
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

    return R

def rotation_matrix_to_quaternion(rot):
    """
    Convert a 3x3 rotation matrix into a quaternion.
    
    Args:
        rot (np.ndarray or torch.Tensor): 3x3 rotation matrix

    Returns:
        quat (np.ndarray): Quaternion in the form [w, x, y, z]
    """
    q = R.from_matrix(rot).as_quat()

    return q


def rot_interp(R1, R2, ratio):
    # Convert rotation matrices to quaternions
    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()

    # Interpolate quaternions
    q_interp = (1 - ratio) * q1 + ratio * q2
    q_interp = q_interp / np.linalg.norm(q_interp)  # Normalize the quaternion

    # Convert the interpolated quaternion back to a rotation matrix
    R_interp = R.from_quat(q_interp).as_matrix()

    return R_interp

def pos_interp(p1, p2, ratio):
    p_interp = (1 - ratio) * p1 + ratio * p2  # t is the interpolation parameter

    return p_interp


def pose_interp(T1, T2, ratio):
    R1, p1 = pose_to_rot_pos(T1)
    R2, p2 = pose_to_rot_pos(T2)

    R_interp = rot_interp(R1, R2, ratio)
    p_interp = pos_interp(p1, p2, ratio)

    T_interp = make_pose(R_interp, p_interp)

    return T_interp