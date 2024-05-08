import numpy as np
import matplotlib.pyplot as plt

from robosuite.environments.robot_env import RobotEnv
from robosuite.models.objects import MujocoObject
from zonopy.contset.zonotope.zono import zonotope

from armtd.environments.arm_3d import Arm_3D

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