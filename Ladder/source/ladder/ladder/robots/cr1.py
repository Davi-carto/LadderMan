from pathlib import Path
from xml.etree import ElementTree

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


REPO_ROOT = Path(__file__).resolve().parents[5]
CR1_URDF = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)

CR1_ACTUATED_JOINT_NAMES = (
    "waist_z_joint",
    "waist_x_joint",
    "waist_y_joint",
    "left_shoulder_y_joint",
    "left_shoulder_x_joint",
    "left_shoulder_z_joint",
    "left_elbow_joint",
    "right_shoulder_y_joint",
    "right_shoulder_x_joint",
    "right_shoulder_z_joint",
    "right_elbow_joint",
    "neck_z_joint",
    "neck_y_joint",
    "left_hip_y_joint",
    "left_hip_x_joint",
    "left_hip_z_joint",
    "left_knee_joint",
    "left_ankle_y_joint",
    "left_ankle_x_joint",
    "right_hip_y_joint",
    "right_hip_x_joint",
    "right_hip_z_joint",
    "right_knee_joint",
    "right_ankle_y_joint",
    "right_ankle_x_joint",
)


def _joint_limits_from_urdf() -> tuple[dict[str, float], dict[str, float]]:
    root = ElementTree.parse(CR1_URDF).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    effort_limits = {}
    velocity_limits = {}
    for name in CR1_ACTUATED_JOINT_NAMES:
        limit = joints[name].find("limit")
        effort_limits[name] = float(limit.attrib["effort"])
        velocity_limits[name] = float(limit.attrib["velocity"])
    return effort_limits, velocity_limits


CR1_EFFORT_LIMITS, CR1_VELOCITY_LIMITS = _joint_limits_from_urdf()
CR1_COLLISION_BODY_NAMES = tuple(
    link.attrib["name"]
    for link in ElementTree.parse(CR1_URDF).getroot().findall("link")
    if link.find("collision") is not None
)
CR1_STIFFNESS = 80.0
CR1_DAMPING = 4.0

CR1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(CR1_URDF),
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.86),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=list(CR1_ACTUATED_JOINT_NAMES),
            effort_limit_sim=CR1_EFFORT_LIMITS,
            velocity_limit_sim=CR1_VELOCITY_LIMITS,
            stiffness=CR1_STIFFNESS,
            damping=CR1_DAMPING,
        )
    },
)

CR1_ACTION_SCALE = {
    name: 0.25 * effort_limit / CR1_STIFFNESS for name, effort_limit in CR1_EFFORT_LIMITS.items()
}


__all__ = [
    "CR1_ACTION_SCALE",
    "CR1_ACTUATED_JOINT_NAMES",
    "CR1_CFG",
    "CR1_COLLISION_BODY_NAMES",
    "CR1_URDF",
]
