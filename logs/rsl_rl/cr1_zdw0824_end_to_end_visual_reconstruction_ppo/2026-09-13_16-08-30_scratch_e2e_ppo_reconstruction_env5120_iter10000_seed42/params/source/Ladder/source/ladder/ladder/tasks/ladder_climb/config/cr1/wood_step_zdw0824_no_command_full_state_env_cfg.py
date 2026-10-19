"""Nonvisual full-state PPO task for the accepted ZDW0824 reference."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp
from ladder.tasks.ladder_climb.ladder_env_cfg import (
    LOWER_BODY_JOINT_PATTERN,
    UPPER_BODY_JOINT_PATTERN,
)
from ladder.tasks.ladder_climb.ladder_geometry import generate_ladder
from ladder.utils.contact_schema import CR1_CONTACT_DISTANCE_THRESHOLDS

from .wood_step_env_cfg import REPO_ROOT
from .wood_step_geometry_dr_env_cfg import (
    STOP_COMMAND_GROUNDED_THIN_RAIL_BOX_SPECS,
    _wood_step_ladder_collection_cfg,
)
from .wood_step_reference_rails_protected_per_body_thin_rails_safe_frame0_env_cfg import (
    CR1RailsProtectedPerBodyThinRailsSceneCfg,
)
from .wood_step_reference_rails_protected_per_body_safe_frame0_env_cfg import (
    CR1RailsProtectedPerBodyRewardsCfg,
)
from .wood_step_reference_force_limited_env_cfg import _force_excess_term
from .wood_step_transformer_vision_thin_rails_ppo_env_cfg import (
    CR1ThinRailsTransformerVisionCriticCfg,
    CR1ThinRailsTransformerVisionPPOEnvCfg,
)


ZDW0824_MIDDLE_STAND_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "zdw0824_capture1"
    / "pitchminus3_yawminus4_xminus030"
    / "36_middle_default_standing_100_zplus5mm"
    / "zdw0824_capture1_stage36_train_ready_middle_default100_zplus5mm_reset5mm_50hz_motion.npz"
)
ZDW0824_LADDER_BOX_SPEC_SET = "thin_rails_grounded"
FOOT_WRONG_RUNG_FORCE_SOFT_N = 20.0
FOOT_WRONG_RUNG_FORCE_SCALE_N = 100.0
FOOT_WRONG_RUNG_FORCE_WEIGHT = -0.10
FOOT_WRONG_RUNG_MAX_EXCESS = 1.5
MOTION_BODY_LIN_VEL_STD = 0.75
MOTION_BODY_POS_LOWER_STD = 0.18
MOTION_BODY_POS_UPPER_STD = 0.25
UPPER_JOINT_BROAD_TEMPERATURE = 1.0
LOWER_JOINT_BROAD_TEMPERATURE = 0.8
HAND_CONTACT_WEIGHT = 0.32
FOOT_CONTACT_WEIGHT = 0.30
HAND_TARGET_TEMPERATURE = 0.01
HAND_REFERENCE_TEMPERATURE = 0.03
FOOT_TARGET_TEMPERATURE = 0.02
FOOT_REFERENCE_TEMPERATURE = 0.02
LOWER_JOINT_VEL_TRACKING_WEIGHT = -0.05
LOWER_JOINT_VEL_TRACKING_DEADBAND = 0.15
LOWER_JOINT_VEL_TRACKING_HUBER_DELTA = 0.5
LOWER_ACTION_SECOND_DIFF_WEIGHT = -0.05
ZDW0824_LOWER_BODY_NAMES = [
    "base_link",
    "left_hip_x_link",
    "left_knee_link",
    "left_ankle_x_link",
    "right_hip_x_link",
    "right_knee_link",
    "right_ankle_x_link",
]
ZDW0824_UPPER_BODY_NAMES = [
    "body",
    "left_shoulder_x_link",
    "left_elbow_link",
    "left_wrist_x_link",
    "left_sphere_hand_link",
    "right_shoulder_x_link",
    "right_elbow_link",
    "right_wrist_x_link",
    "right_sphere_hand_link",
]
ZDW0824_REFERENCE_LIMB_BODY_NAMES = {
    "left_hand": "left_sphere_hand_link",
    "right_hand": "right_sphere_hand_link",
    "left_foot": "left_ankle_x_link",
    "right_foot": "right_ankle_x_link",
}
ZDW0824_LADDER_GEOMETRY = generate_ladder(
    phi=1.0401464700698853,
    rung_spacing=0.26650500297546387,
    rung_length=0.6399999856948853,
    rung_radius_or_width=0.12999999523162842,
    num_rungs=7,
    origin=(0.4449999928474426, 0.03999999910593033, 0.1599999964237213),
)
ZDW0824_CONTACT_TARGET_OFFSETS = {
    "left_hand": (0.0, 0.23, 0.045),
    "right_hand": (0.0, -0.23, 0.045),
    "left_foot": (-0.04, 0.11, 0.0),
    "right_foot": (-0.05, -0.16, 0.0),
}


@configclass
class CR1ZDW0824NoCommandFullStateRewardsCfg(
    CR1RailsProtectedPerBodyRewardsCfg
):
    """ZDW0824 tracking rewards shared by nonvisual and visual tasks."""

    motion_body_pos_asym = RewTerm(
        func=mdp.asymmetric_motion_relative_body_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "lower_body_std": MOTION_BODY_POS_LOWER_STD,
            "upper_body_std": MOTION_BODY_POS_UPPER_STD,
            "lower_body_names": ZDW0824_LOWER_BODY_NAMES,
            "upper_body_names": ZDW0824_UPPER_BODY_NAMES,
        },
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.6,
        params={"command_name": "motion", "std": MOTION_BODY_LIN_VEL_STD},
    )
    motion_joint_pos_upper = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=0.25,
        params={
            "command_name": "motion",
            "temperature": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[UPPER_BODY_JOINT_PATTERN]
            ),
        },
    )
    motion_joint_pos_upper_broad = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=0.25,
        params={
            "command_name": "motion",
            "temperature": UPPER_JOINT_BROAD_TEMPERATURE,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[UPPER_BODY_JOINT_PATTERN]
            ),
        },
    )
    motion_joint_pos_lower = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "temperature": 0.25,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[LOWER_BODY_JOINT_PATTERN]
            ),
        },
    )
    motion_joint_pos_lower_broad = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "temperature": LOWER_JOINT_BROAD_TEMPERATURE,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[LOWER_BODY_JOINT_PATTERN]
            ),
        },
    )
    left_hand_contact_position = RewTerm(
        func=mdp.ladder_phase_conditioned_limb_tracking_exp,
        weight=HAND_CONTACT_WEIGHT,
        params={
            "command_name": "motion",
            "limb_names": ["left_hand"],
            "reference_body_names": [
                ZDW0824_REFERENCE_LIMB_BODY_NAMES["left_hand"]
            ],
            "target_temperature": HAND_TARGET_TEMPERATURE,
            "reference_temperature": HAND_REFERENCE_TEMPERATURE,
        },
    )
    right_hand_contact_position = RewTerm(
        func=mdp.ladder_phase_conditioned_limb_tracking_exp,
        weight=HAND_CONTACT_WEIGHT,
        params={
            "command_name": "motion",
            "limb_names": ["right_hand"],
            "reference_body_names": [
                ZDW0824_REFERENCE_LIMB_BODY_NAMES["right_hand"]
            ],
            "target_temperature": HAND_TARGET_TEMPERATURE,
            "reference_temperature": HAND_REFERENCE_TEMPERATURE,
        },
    )
    foot_contact_position = RewTerm(
        func=mdp.ladder_phase_conditioned_limb_tracking_exp,
        weight=FOOT_CONTACT_WEIGHT,
        params={
            "command_name": "motion",
            "limb_names": ["left_foot", "right_foot"],
            "reference_body_names": [
                ZDW0824_REFERENCE_LIMB_BODY_NAMES["left_foot"],
                ZDW0824_REFERENCE_LIMB_BODY_NAMES["right_foot"],
            ],
            "target_temperature": FOOT_TARGET_TEMPERATURE,
            "reference_temperature": FOOT_REFERENCE_TEMPERATURE,
        },
    )

    left_foot_wrong_rung_force_excess = _force_excess_term(
        "left_foot_step_contacts",
        "left_ankle_x_link",
        force_soft=FOOT_WRONG_RUNG_FORCE_SOFT_N,
        force_scale=FOOT_WRONG_RUNG_FORCE_SCALE_N,
        surface="all",
        weight=FOOT_WRONG_RUNG_FORCE_WEIGHT,
        limb_name="left_foot",
        rung_scope="non_target",
        max_excess=FOOT_WRONG_RUNG_MAX_EXCESS,
    )
    right_foot_wrong_rung_force_excess = _force_excess_term(
        "right_foot_step_contacts",
        "right_ankle_x_link",
        force_soft=FOOT_WRONG_RUNG_FORCE_SOFT_N,
        force_scale=FOOT_WRONG_RUNG_FORCE_SCALE_N,
        surface="all",
        weight=FOOT_WRONG_RUNG_FORCE_WEIGHT,
        limb_name="right_foot",
        rung_scope="non_target",
        max_excess=FOOT_WRONG_RUNG_MAX_EXCESS,
    )


@configclass
class CR1ZDW0824SmoothOutputRewardsCfg(
    CR1ZDW0824NoCommandFullStateRewardsCfg
):
    """Stage-36 rewards with stronger lower-body acceleration regularization."""

    # Keep the existing raw-action rate penalty and additionally discourage
    # high-frequency realized joint motion under observation/actuator noise.
    # The lower body receives stronger regularization because the measured
    # residual jitter is concentrated in the hip, knee, and ankle joints.
    joint_acc_l2_upper = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[UPPER_BODY_JOINT_PATTERN]
            )
        },
    )
    joint_acc_l2_lower = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[LOWER_BODY_JOINT_PATTERN]
            )
        },
    )
    action_second_difference_l2_lower = RewTerm(
        func=mdp.action_second_difference_l2,
        weight=LOWER_ACTION_SECOND_DIFF_WEIGHT,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[LOWER_BODY_JOINT_PATTERN]
            )
        },
    )
    motion_joint_vel_lower_robust = RewTerm(
        func=mdp.motion_joint_velocity_error_deadband_huber,
        weight=LOWER_JOINT_VEL_TRACKING_WEIGHT,
        params={
            "command_name": "motion",
            "deadband": LOWER_JOINT_VEL_TRACKING_DEADBAND,
            "delta": LOWER_JOINT_VEL_TRACKING_HUBER_DELTA,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[LOWER_BODY_JOINT_PATTERN]
            ),
        },
    )


@configclass
class CR1ZDW0824ThinRailsSceneCfg(CR1RailsProtectedPerBodyThinRailsSceneCfg):
    """Thin-rail collision scene shifted down with the accepted reference."""

    ladder = _wood_step_ladder_collection_cfg(
        STOP_COMMAND_GROUNDED_THIN_RAIL_BOX_SPECS
    )


@configclass
class CR1ZDW0824FullStateObservationCfg(
    CR1ThinRailsTransformerVisionCriticCfg
):
    """Shared 780-D actor/critic state without camera observations."""

    def __post_init__(self):
        super().__post_init__()
        self.limb_heightmaps.params = dict(self.limb_heightmaps.params)
        self.limb_heightmaps.params[
            "ladder_box_spec_set"
        ] = ZDW0824_LADDER_BOX_SPEC_SET


@configclass
class CR1ZDW0824NoCommandFullStateObservationsCfg:
    policy: CR1ZDW0824FullStateObservationCfg = (
        CR1ZDW0824FullStateObservationCfg()
    )
    critic: CR1ZDW0824FullStateObservationCfg = (
        CR1ZDW0824FullStateObservationCfg()
    )


@configclass
class CR1ZDW0824NoCommandFullStateMiddleStandEnvCfg(
    CR1ThinRailsTransformerVisionPPOEnvCfg
):
    """Stage-36 reference with identical full-state inputs and a top stand."""

    scene: CR1ZDW0824ThinRailsSceneCfg = CR1ZDW0824ThinRailsSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
    )
    observations: CR1ZDW0824NoCommandFullStateObservationsCfg = (
        CR1ZDW0824NoCommandFullStateObservationsCfg()
    )
    rewards: CR1ZDW0824NoCommandFullStateRewardsCfg = (
        CR1ZDW0824NoCommandFullStateRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 90.0
        self.commands.motion.motion_file = str(ZDW0824_MIDDLE_STAND_MOTION)
        self.commands.motion.ladder_geometry = ZDW0824_LADDER_GEOMETRY
        self.commands.motion.limb_target_offsets = dict(
            ZDW0824_CONTACT_TARGET_OFFSETS
        )
        self.commands.motion.limb_target_distance_thresholds = dict(
            CR1_CONTACT_DISTANCE_THRESHOLDS
        )
        self.commands.motion.fixed_start_pose_range = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (0.0, 0.005),
            "roll": (-0.01, 0.01),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        }
        self.commands.motion.use_reset_valid_mask = False
        self.commands.motion.stop_command_training_enabled = False
        self.commands.motion.stop_hold_reset_probability = 0.0
        self.commands.motion.stop_seek_episode_probability = 0.0
        self.commands.motion.fixed_start_probability = 0.10
        self.commands.motion.fixed_start_pose_randomization_probability = 0.70
        self.commands.motion.secondary_fixed_start_frame = 1500
        self.commands.motion.secondary_fixed_start_probability = 0.10
        self.commands.motion.secondary_fixed_start_requires_nominal_ladder = True
        self.commands.motion.uniform_start_probability = 0.40
        self.terminations.root_pose.params = dict(
            self.terminations.root_pose.params
        )
        self.terminations.root_pose.params["orientation_threshold"] = 0.8


@configclass
class CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg(
    CR1ZDW0824NoCommandFullStateMiddleStandEnvCfg
):
    """Separate Stage-36 task that learns smooth motion under full DR."""

    rewards: CR1ZDW0824SmoothOutputRewardsCfg = (
        CR1ZDW0824SmoothOutputRewardsCfg()
    )
