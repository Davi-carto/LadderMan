from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb.ladder_geometry import LADDER_A, LadderGeometry, get_rung_pose

import ladder.tasks.ladder_climb.mdp as mdp


def _ladder_material(static_friction: float = 1.2, dynamic_friction: float = 1.0) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
    )


def _rung_cfg(index: int, geometry: LadderGeometry = LADDER_A) -> AssetBaseCfg:
    pos, rot = get_rung_pose(index, geometry)
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Ladder/rung_{index:02d}",
        spawn=sim_utils.CuboidCfg(
            size=(geometry.rung_width, geometry.rung_length, geometry.rung_width),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            physics_material=_ladder_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.46, 0.42, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        collision_group=0,
    )


def _rail_cfg(side: str, lateral_sign: float, geometry: LadderGeometry = LADDER_A) -> AssetBaseCfg:
    direction = geometry.climb_direction
    origin = geometry.origin
    center_distance = 0.5 * (geometry.num_rungs - 1) * geometry.rung_spacing
    pos = (
        origin[0] + center_distance * direction[0],
        origin[1] + lateral_sign * 0.5 * geometry.rung_length,
        origin[2] + center_distance * direction[2],
    )
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Ladder/rail_{side}",
        spawn=sim_utils.CuboidCfg(
            size=(geometry.rail_depth, geometry.rail_width, geometry.rail_length),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            physics_material=_ladder_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.30, 0.28)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=geometry.rail_quat_wxyz),
        collision_group=0,
    )


@configclass
class LadderSceneCfg(InteractiveSceneCfg):
    """G1 scene with a fixed procedural Ladder A."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    rung_00 = _rung_cfg(0)
    rung_01 = _rung_cfg(1)
    rung_02 = _rung_cfg(2)
    rung_03 = _rung_cfg(3)
    rung_04 = _rung_cfg(4)
    rung_05 = _rung_cfg(5)
    rung_06 = _rung_cfg(6)
    rung_07 = _rung_cfg(7)
    rung_08 = _rung_cfg(8)
    rung_09 = _rung_cfg(9)
    rung_10 = _rung_cfg(10)
    rung_11 = _rung_cfg(11)
    rail_left = _rail_cfg("left", 1.0)
    rail_right = _rail_cfg("right", -1.0)

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=6,
        track_air_time=True,
        force_threshold=5.0,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class CommandsCfg:
    motion = mdp.LadderMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        ladder_geometry=LADDER_A,
        pose_range={
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        },
        velocity_range={name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")},
        joint_position_range=(-0.05, 0.05),
    )


@configclass
class ExpertObservationCfg(ObsGroup):
    reference_body_pos_b = ObsTerm(func=mdp.reference_body_pos_b, params={"command_name": "motion"})
    reference_joint_pos = ObsTerm(func=mdp.reference_joint_pos, params={"command_name": "motion"})
    motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
    motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
    motion_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})
    contact_indicator = ObsTerm(func=mdp.contact_indicator, params={"command_name": "motion"})
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    root_orientation = ObsTerm(func=mdp.root_orientation_6d)
    joint_pos_history = ObsTerm(func=mdp.joint_pos, history_length=9)
    action_history = ObsTerm(func=mdp.last_action, history_length=3)
    ladder_relative_root_pos = ObsTerm(func=mdp.ladder_relative_root_pos, params={"command_name": "motion"})
    body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
    body_lin_vel = ObsTerm(func=mdp.robot_body_lin_vel_b, params={"command_name": "motion"})
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    body_pos_error = ObsTerm(func=mdp.body_pos_error_b, params={"command_name": "motion"})
    body_lin_vel_error = ObsTerm(func=mdp.body_lin_vel_error_b, params={"command_name": "motion"})
    joint_pos_error = ObsTerm(func=mdp.joint_pos_error, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CriticObservationCfg(ExpertObservationCfg):
    contact_error_b = ObsTerm(func=mdp.contact_error_b, params={"command_name": "motion"})


@configclass
class ObservationsCfg:
    policy: ExpertObservationCfg = ExpertObservationCfg()
    critic: CriticObservationCfg = CriticObservationCfg()


LOWER_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
]

UPPER_BODY_NAMES = [
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

LOWER_BODY_JOINT_PATTERN = r".*_(?:hip_.*|knee|ankle_.*)_joint"
UPPER_BODY_JOINT_PATTERN = r"(?:waist_.*|neck_.*|.*_shoulder_.*|.*_elbow|.*_wrist_.*)_joint"


@configclass
class RewardsCfg:
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.8,
        params={"command_name": "motion", "std": 0.25},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.4,
        params={"command_name": "motion", "std": 0.35},
    )
    motion_body_pos_asym = RewTerm(
        func=mdp.asymmetric_motion_relative_body_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "lower_body_std": 0.18,
            "upper_body_std": 0.35,
            "lower_body_names": LOWER_BODY_NAMES,
            "upper_body_names": UPPER_BODY_NAMES,
        },
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.6,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_joint_pos_upper = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "temperature": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[UPPER_BODY_JOINT_PATTERN]),
        },
    )
    motion_joint_pos_lower = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "temperature": 0.25,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[LOWER_BODY_JOINT_PATTERN]),
        },
    )
    left_hand_contact_position = RewTerm(
        func=mdp.ladder_contact_tracking_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "limb_names": ["left_hand"],
            "active_weight": 0.3,
            "temperature": 0.03,
        },
    )
    right_hand_contact_position = RewTerm(
        func=mdp.ladder_contact_tracking_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "limb_names": ["right_hand"],
            "active_weight": 0.3,
            "temperature": 0.03,
        },
    )
    foot_contact_position = RewTerm(
        func=mdp.ladder_contact_tracking_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "limb_names": ["left_foot", "right_foot"],
            "active_weight": 0.3,
            "temperature": 0.3,
        },
    )
    left_hand_swing_contact = RewTerm(
        func=mdp.limb_swing_contact_penalty,
        weight=-0.1,
        params={
            "command_name": "motion",
            "limb_names": ["left_hand"],
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_wrist_yaw_link"]
            ),
            "force_threshold": 5.0,
        },
    )
    right_hand_swing_contact = RewTerm(
        func=mdp.limb_swing_contact_penalty,
        weight=-0.1,
        params={
            "command_name": "motion",
            "limb_names": ["right_hand"],
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["right_wrist_yaw_link"]
            ),
            "force_threshold": 5.0,
        },
    )
    left_hand_downward_contact = RewTerm(
        func=mdp.limb_downward_contact_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_wrist_yaw_link"]
            ),
            "force_threshold": 5.0,
        },
    )
    right_hand_downward_contact = RewTerm(
        func=mdp.limb_downward_contact_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["right_wrist_yaw_link"]
            ),
            "force_threshold": 5.0,
        },
    )
    left_hand_upward_contact_force = RewTerm(
        func=mdp.limb_upward_contact_force_reward,
        weight=0.3,
        params={
            "command_name": "motion",
            "limb_names": ["left_hand"],
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_wrist_yaw_link"]
            ),
            "force_threshold": 5.0,
            "force_saturation": 50.0,
            "normalization_limb_names": ["left_hand", "right_hand"],
        },
    )
    right_hand_upward_contact_force = RewTerm(
        func=mdp.limb_upward_contact_force_reward,
        weight=0.3,
        params={
            "command_name": "motion",
            "limb_names": ["right_hand"],
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["right_wrist_yaw_link"]
            ),
            "force_threshold": 5.0,
            "force_saturation": 50.0,
            "normalization_limb_names": ["left_hand", "right_hand"],
        },
    )
    left_hand_step_surface_contact = RewTerm(
        func=mdp.ladder_surface_contact_penalty,
        weight=0.0,
        params={
            "command_name": "motion",
            "limb_name": "left_hand",
            "sensor_cfg": SceneEntityCfg(
                "left_hand_step_contacts", body_names=["left_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
            "allow_target_contact": True,
            "target_distance_threshold": 0.08,
        },
    )
    right_hand_step_surface_contact = RewTerm(
        func=mdp.ladder_surface_contact_penalty,
        weight=0.0,
        params={
            "command_name": "motion",
            "limb_name": "right_hand",
            "sensor_cfg": SceneEntityCfg(
                "right_hand_step_contacts", body_names=["right_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
            "allow_target_contact": True,
            "target_distance_threshold": 0.08,
        },
    )
    left_hand_side_board_contact = RewTerm(
        func=mdp.ladder_surface_contact_penalty,
        weight=0.0,
        params={
            "command_name": "motion",
            "limb_name": "left_hand",
            "sensor_cfg": SceneEntityCfg(
                "left_hand_side_board_contacts", body_names=["left_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
            "allow_target_contact": False,
            "target_distance_threshold": 0.08,
        },
    )
    right_hand_side_board_contact = RewTerm(
        func=mdp.ladder_surface_contact_penalty,
        weight=0.0,
        params={
            "command_name": "motion",
            "limb_name": "right_hand",
            "sensor_cfg": SceneEntityCfg(
                "right_hand_side_board_contacts", body_names=["right_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
            "allow_target_contact": False,
            "target_distance_threshold": 0.08,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_torque_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 5.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_complete = DoneTerm(
        func=mdp.motion_complete,
        time_out=True,
        params={"command_name": "motion"},
    )
    root_pose = DoneTerm(
        func=mdp.bad_root_pose_consecutive,
        params={
            "command_name": "motion",
            "position_threshold": 0.5,
            "orientation_threshold": 1.2,
            "consecutive_steps": 25,
        },
    )
    body_local_pose = DoneTerm(
        func=mdp.bad_body_local_pose_consecutive,
        params={
            "command_name": "motion",
            "position_threshold": 0.5,
            "orientation_threshold": 1.2,
            "consecutive_steps": 25,
        },
    )
    required_contact = DoneTerm(
        func=mdp.bad_required_contact_consecutive,
        params={"command_name": "motion", "position_threshold": 0.45, "consecutive_steps": 30},
    )


@configclass
class EventCfg:
    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 7.2),
            "dynamic_friction_range": (0.3, 3.6),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    joint_position_offset = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {axis: (-0.02, 0.02) for axis in ("x", "y", "z")},
        },
    )
    root_velocity_push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 9.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.08, 0.08),
                "y": (-0.08, 0.08),
                "z": (-0.03, 0.03),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.10, 0.10),
            },
        },
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class LadderEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based LadderMan expert environment."""

    scene: LadderSceneCfg = LadderSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (2.0, -2.5, 1.8)
        self.viewer.lookat = (0.9, 0.0, 1.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
