import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from robosuite.environments.robot_env import RobotEnv
from robosuite.models.objects import MujocoObject
import robosuite.utils.transform_utils as T


use_zonopy = os.getenv('USE_ZONOPY', 'false').lower() == 'true'
if use_zonopy:
    from zonopy.contset import zonotope
else:
    from safediffusion.armtdpy.reachability.conSet import zonotope


def transform_zonotope(zono, pos, rot):
    """
    Transform 3D zonotope
    """
    assert isinstance(zono, zonotope) and zono.dimension == 3
    assert pos.shape[0] == 3

    rot = torch.asarray(rot, dtype = zono.dtype, device =zono.device)
    pos = torch.asarray(pos, dtype = zono.dtype, device =zono.device)

    c = zono.center
    G = zono.generators

    c_new = rot@c + pos
    G_new = G@rot.T
    Z_new = np.vstack([c_new, G_new])
    Z_new = torch.asarray(Z_new)

    return zonotope(Z_new)

def plot_zonos(zonos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for zono in zonos:
        zono.plot3d(ax, facecolor='red', edgecolor='red')

    ax.set_xlim(-0.5, 0.5)  # set x-axis limits from 0 to 6
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 1.1)
    ax.view_init(elev=0, azim=0)
    
    fig.savefig('saved_figure1.png')
    

def zonotope_from_robosuite_env(env):
    # TODO: bins_arena
    zonos = []
    for obj in env.objects:
        zono = zonotope_from_mujoco_object(env, obj)
        zonos.append(zono)

    return zonos
    

def zonotope_from_mujoco_object(env, object):
    """
    Create zonotopic representation of bounding box of MuJoCo Object

    Args
        env: Robosuite Environment
        object: MujocoObject

    Output
        zonotope

    NOTE: Need to check if the center of the bounding box is the center of the root body.
    -- objects/MuJoCoXMLObject: bbox[2] = max(obj.bottom_offset, obj.bbox_top_offset) - obj.bottom_offset
    -- The z-axis is not aligned -- why?

    Author: Wonsuhk Jung
    """
    assert isinstance(env, RobotEnv)
    assert isinstance(object, MujocoObject)

    c = env.sim.data.get_body_xpos(object.root_body)
    R = env.sim.data.get_body_xmat(object.root_body)

    G = np.diag(object.get_bounding_box_half_size())
    G = R@G

    Z = np.vstack([c, G])

    return zonotope(Z)


def get_zonotope_from_plane_geom(pos, rot, size):
    """
    Create zonotope from plane geometry

    The size means (half_width, half_height, spacing)

    Args
        geom: MuJoCo geom

    Output
        zonotope
    """
    size[2] = 0.05

    return get_zonotope_from_box_geom(pos, rot, size)

def get_zonotope_from_box_geom(pos, rot, size):
    """
    Create zonotope from box geometry

    The size means (half_dx, half_dy, half_dz)

    Args
        geom: MuJoCo geom

    Output
        zonotope
    """
    c = pos
    G = rot@np.diag(size)

    Z = np.vstack([c, G])
    Z = torch.asarray(Z)

    return zonotope(Z)


def get_zonotope_from_cylinder_geom(pos, rot, size):
    """
    Create zonotope from cylinder geometry

    The size means (radius, half_height)

    Args
        geom: MuJoCo geom

    Output
        zonotope
    """
    c = pos
    G = rot@np.diag([size[0], size[0], size[1]])
    Z = np.vstack([c, G])
    Z = torch.asarray(Z)

    return zonotope(Z)

def get_zonotope_from_sphere_geom(pos, rot, size):
    """
    Create zonotope from sphere geometry

    The size means (radius)

    Args
        geom: MuJoCo geom

    Output
        zonotope
    """
    c = pos
    G = rot@np.diag([size[0], size[0], size[0]])
    Z = np.vstack([c, G])
    Z = torch.asarray(Z)

    return zonotope(Z)

if __name__ == "__main__":
    c = np.array([1, 1, 1])
    G = np.eye(3)
    Z = zonotope(np.vstack([c, G]))
    Z = torch.asarray(Z)

    Z_new = transform_zonotope(Z, pos=np.array([1, 2, 3]), rot=G)