"""
Useful transformation-related (rotation, translation) functions
"""
import torch
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
    if isinstance(rot, np.ndarray) and isinstance(pos, np.ndarray):
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3]  = pos
        return T
    
    elif isinstance(rot, torch.Tensor) and isinstance(pos, torch.Tensor):
        T = torch.eye(4, dtype=rot.dtype, device=rot.device)
        T[:3, :3] = rot
        T[:3, 3]  = pos
        return T
    
    else:
        raise TypeError("Input arguments must both be either np.ndarray or torch.Tensor.")

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

    if isinstance(R1, torch.Tensor):
        R_interp = torch.tensor(R_interp, dtype=R1.dtype, device=R1.device)
    elif isinstance(R1, np.ndarray):
        R_interp = np.array(R_interp)

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

def skew_symmetric(angular_axis):
    """
    Skew matrix representation of the angular axis

    [a b c] -> [ 0 -c  b;
                 c  0 -a;
                -b  a  0]                

    Args:
        angular_axis (np.array) (B, 3)
    
    Returns:
        skew_symmetric_matrix (np.array) (B, 3, 3)
    """
    assert angular_axis.shape[-1] == 3

    ndim = angular_axis.ndim
    if ndim == 1:
        angular_axis = angular_axis[np.newaxis, :]

    # Define the skew-symmetric template matrix
    w = np.array([[[0,  0,  0],
                   [0,  0,  -1],
                   [0,  1,  0]],

                  [[0,  0,  1],
                   [0,  0,  0],
                   [-1,  0,  0]],

                  [[ 0, -1,  0],
                   [ 1,  0,  0],
                   [ 0,  0,  0]]])  # Shape (3, 3, 3)

    # Reshape angular_axis to shape (B, 3, 1) to allow broadcasting with w
    angular_axis = angular_axis[:, np.newaxis, :]  # Shape (B, 1, 3)

    # Perform matrix multiplication using broadcasting
    skew_matrices = np.einsum('bij,jik->bik', angular_axis, w)  # Shape (B, 3, 3)

    if ndim == 1:
        skew_matrices = skew_matrices[0]

    return skew_matrices

def bracket_screw_axis(screw_axis):
    """
    Bracket operation of the screw axis

    Args:
        screw_axis (np.array): (B, 6) [v, w]
    
    Returns:
        bracket_screw_axis (np.array): (B, 4, 4) matrix [[w] v; 0 0]
    """
    assert screw_axis.shape[-1] == 6, "The last dimension should be 6."
    assert np.allclose(np.linalg.norm(screw_axis[:, 3:], axis = 1), 1)
    
    if screw_axis.ndim == 1:
        screw_axis = screw_axis[np.newaxis, :]

    B = screw_axis.shape[0]

    v = screw_axis[:, :3]
    w = screw_axis[:, 3:]

    out = np.concatenate([skew_symmetric(w), v[:, :, np.newaxis]], axis = -1) # (B, 3, 4)
    out = np.concatenate([out, np.zeros((B, 1, 4))]        , axis = -2)

    return out

def adjoint_T(T):
    """
    Adjoint transformation matrix

    Args:
        T (np.array): (4, 4) transformation matrix
    
    Returns:
        adjoint_T (np.array): (6, 6) adjoint transformation matrix
    """
    R, p = pose_to_rot_pos(T)

    adjoint_T = np.zeros((6, 6))

    adjoint_T[:3, :3] = R
    adjoint_T[3:, 3:] = R

    adjoint_T[3:, :3] = skew_symmetric(p)@R

    return adjoint_T