import einops

from safediffusion.algo.helper import ReferenceTrajectory
from safediffusion.algo.safety_filter import SafetyFilter
from safediffusion_arm.algo.planner_arm import ArmtdPlanner
from safediffusion_arm.algo.planner_arm_xml import ArmtdPlannerXML

def vis_plan_as_traj_of_site(policy, plan_to_vis, site_name):
    """
    Args:
        policy (SafeDiffusionPolicy): policy object
        plan_to_vis (ReferenceTrajectory): plan to visualize
        site_name (str): site name
        
    Returns:
        traj (np.ndarray): trajectory (N, 3)
    """    
    assert isinstance(policy, SafetyFilter), "The policy is not a SafetyFilter."
    assert isinstance(policy.backup_policy, ArmtdPlanner), "The backup policy is not an ArmtdPlanner."
    assert isinstance(plan_to_vis, ReferenceTrajectory), "The plan to visualize is not a ReferenceTrajectory."
    assert plan_to_vis is not None, "The plan to visualize is None."
    
    if site_name == "eef":
        traj = policy.backup_policy.get_arm_eef_pos_from_plan(plan = plan_to_vis)
        
    elif site_name == "grasp":
        traj = policy.backup_policy.get_grasping_pos_from_plan(plan = plan_to_vis)
    
    else:
        raise ValueError(f"Invalid site_name: {site_name}")
    
    return traj

# TODO: check if head plan is considering gripper changes
def vis_plan_as_forward_occupancy(policy, plan_to_vis, show_arm = True, show_gripper = True, gripper_qpos = None):
    """
    Visualizes the forward occupancy of the plan with non-changing gripper shape
    
    Args:
        policy (_type_): _description_
        plan_to_vis (_type_): _description_
        show_arm (bool, optional): _description_. Defaults to True.
        show_gripper (bool, optional): _description_. Defaults to True.
        gripper_qpos (_type_, optional): _description_. Defaults to None.
    """
    assert isinstance(policy, SafetyFilter), "The policy is not a SafetyFilter."
    assert isinstance(policy.backup_policy, ArmtdPlanner), "The backup policy is not an ArmtdPlanner."
    
    if gripper_qpos is None:
        gripper_qpos = policy.backup_policy.gripper_init_qpos
    
    FO = policy.backup_policy.get_forward_occupancy_from_plan(
                                                         plan               = plan_to_vis, 
                                                         vis_gripper        = show_gripper, 
                                                         only_end_effector  = (not show_arm), 
                                                         gripper_init_qpos  = gripper_qpos
                                                        )
    
    return FO

def vis_forward_occupancy_at_q(policy, qpos_arm, qpos_gripper = None, show_arm = False, show_gripper = False):
    """
    Visualizes the forward occupancy of the robot at q
    
    Args:
        policy (_type_): _description_
        q (_type_): _description_
        show_arm (bool, optional): _description_. Defaults to True.
        show_gripper (bool, optional): _description_. Defaults to True.
        gripper_qpos (_type_, optional): _description_. Defaults to None.
    """
    assert isinstance(policy, SafetyFilter), "The policy is not a SafetyFilter."
    assert isinstance(policy.backup_policy, ArmtdPlannerXML), "The backup policy is not an ArmtdPlannerXML."
    assert not (qpos_gripper is None and show_gripper), "show_gripper is True but qpos_gripper is None."
    
    FO = []
    
    if show_arm:
        arm_zonos = policy.backup_policy.get_arm_zonotopes_at_q(qpos_arm)
        FO.extend(arm_zonos)
        
    if show_gripper:
        T_world_to_gripper_base = policy.backup_policy.get_arm_link_pose_at_q(link_name = policy.backup_policy.robot_gripper_site_name,
                                                                              q         = qpos_arm)
        
        gripper_zonos           = policy.backup_policy.get_gripper_zonotopes_at_q(q               = qpos_gripper, 
                                                                                  T_frame_to_base = T_world_to_gripper_base)
        
        FO.extend(gripper_zonos)
    
    return FO

def vis_FRS_backup(policy, show_arm = False, show_gripper = True, n_skip = 25):
    """
    Visualize the slice of the FRS of the backup policy.
    We fetch the FRS computed at the Trajectory Optimization Step.
    """
    assert isinstance(policy, SafetyFilter), "The policy is not a SafetyFilter."
    assert isinstance(policy.backup_policy, ArmtdPlannerXML), "The backup policy is not an ArmtdPlannerXML."
    
    backup_policy = policy.backup_policy
    backup_plan = policy._backup_plan
    
    # fetch the FRS links pre-computed when we did trajectory optimization for the backup policy
    FRS_links = []
    if show_arm:
        FRS_links.extend(backup_policy.arm_FRS_links)
    if show_gripper:
        FRS_links.extend(backup_policy.gripper_FRS_links)
    
    
    # slice with the plan parameter
    if not FRS_links:
        # get the current plan parameter
        param  = backup_plan.get_trajectory_parameter()
        optvar = param[backup_policy.opt_dim]
        beta   = optvar/backup_policy.FRS_info["delta_k"][policy.backup_policy.opt_dim]
        beta   = einops.repeat(beta, 'n -> repeat n', repeat=backup_policy.n_timestep-1)
        
        # slice with the current plan
        FRS    = [FRS_link.slice_all_dep(beta) for FRS_link in FRS_links]
        
        # downsampling
        FRS    = [FRS_link[0::n_skip] for FRS_link in FRS]
        
    else:
        FRS    = []
    
    return FRS