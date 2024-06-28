"""
Forward Occupancy Algorithm
Author: Yongseok Kwon
"""
import os
import sys
sys.path.append('../..')
import torch
use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy:
    from zonopy.contset import polyZonotope, matPolyZonotope
else:
    from safediffusion.armtdpy.reachability.conSet import polyZonotope, matPolyZonotope

def forward_occupancy(rotatos,link_zonos,robot_params,zono_order=2, T_world_to_base=None):
    '''
    Based on the rotatopes (=the set of possible rotation matrixes) for each joint i, calculate
    the forward occupancy. The link_zonos assume the zonotope represnetation of the links with 
    respect to the robot base frame.

    Input
        rotatos    : (N_joint, N_T, N_gen, SO(3))
        link_zonos : zonotope representations of links

    Output
        FO_link    : (N_joint, N_T, N_gen, 3)
        R_motor    : The set of rotation matrixes from the fixed frame (N_joint, N_T, N_gen, SO(3))
        p_motor    : The set of translation vectors from the fixed frame(N_joint, N_T, N_gen, SO(3))
    '''    
    dtype, device = rotatos[0].dtype, rotatos[0].device
    n_joints = robot_params['n_joints']
    P = robot_params['P']
    R = robot_params['R']

    if T_world_to_base is None:
        P_motor = [polyZonotope(torch.zeros(3,dtype=dtype,device=device).unsqueeze(0))]
        R_motor = [matPolyZonotope(torch.eye(3,dtype=dtype,device=device).unsqueeze(0))]
    else:
        p_world_to_base = torch.asarray(T_world_to_base[0:3, 3], dtype=dtype, device=device).unsqueeze(0)
        R_world_to_base = torch.asarray(T_world_to_base[0:3, 0:3], dtype=dtype, device=device).unsqueeze(0)
        P_motor = [polyZonotope(p_world_to_base)]
        R_motor = [matPolyZonotope(R_world_to_base)]

    FO_link = []
    for i in range(n_joints):
        P_motor_temp = R_motor[-1]@P[i] + P_motor[-1]
        P_motor.append(P_motor_temp.reduce_indep(zono_order))
        R_motor_temp = R_motor[-1]@R[i]@rotatos[i]
        R_motor.append(R_motor_temp.reduce_indep(zono_order))
        FO_link_temp = R_motor[-1]@link_zonos[i] + P_motor[-1]
        FO_link.append(FO_link_temp.reduce_indep(zono_order))
    return FO_link, R_motor[1:], P_motor[1:]

