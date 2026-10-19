"""Policy-only end-to-end visual PPO on the retained blind-task MDP."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp
from ladder.tasks.ladder_climb.ladder_env_cfg import TerminationsCfg

from .wood_step_transformer_vision_stop_command_env_cfg import (
    CR1StopCommandTransformerVisionDepthCfg,
    CR1StopCommandTransformerVisionInputsCfg,
)
from .wood_step_transformer_vision_stop_command_validated_task_env_cfg import (
    ZDW0824_VISUAL_STOP_COMMAND_MOTION,
)
from .wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824FullStateObservationCfg,
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg,
    CR1ZDW0824SmoothOutputRewardsCfg,
    CR1ZDW0824ThinRailsSceneCfg,
    FOOT_CONTACT_WEIGHT,
    FOOT_REFERENCE_TEMPERATURE,
    FOOT_TARGET_TEMPERATURE,
    HAND_CONTACT_WEIGHT,
    HAND_REFERENCE_TEMPERATURE,
    HAND_TARGET_TEMPERATURE,
    ZDW0824_REFERENCE_LIMB_BODY_NAMES,
)
from .wood_step_reference_rails_protected_per_body_safe_frame0_env_cfg import (
    RAIL_CONTACT_SENSOR_NAMES,
)


def _foot_ground_contact_sensor_cfg(side: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_ankle_x_link",
        filter_prim_paths_expr=[
            "/World/ground",
            "{ENV_REGEX_NS}/Ladder_base_platform",
        ],
        history_length=6,
        force_threshold=5.0,
        debug_vis=False,
    )


@configclass
class CR1EndToEndVisualOracleSceneCfg(CR1ZDW0824ThinRailsSceneCfg):
    """Oracle scene with pair-filtered ground sensing for both feet."""

    left_foot_ground_contacts = _foot_ground_contact_sensor_cfg("left")
    right_foot_ground_contacts = _foot_ground_contact_sensor_cfg("right")


@configclass
class CR1EndToEndVisualInputsCfg(CR1StopCommandTransformerVisionInputsCfg):
    """Deployable command-aware, time-sampled history and 30 Hz camera exposures."""

    depth_updated = ObsTerm(func=mdp.cr1_transformer_vision_depth_updated)

    def __post_init__(self):
        super().__post_init__()
        self.proprio_history.params = dict(self.proprio_history.params)
        self.proprio_history.params["num_proprio_frames"] = 10
        self.proprio_history.params["num_depth_frames"] = 3
        self.proprio_history.params["proprio_interval_s"] = 0.1
        self.proprio_history.params["depth_history_offsets_s"] = (0.0, 0.2, 0.4)
        self.proprio_history.params["history_buffer_capacity"] = 64
        self.proprio_history.params["include_climb_command"] = True
        # The wrapper obtains current proprioception from history[:, -1],
        # retaining the exact same sensor-noise realization as the encoder.


@configclass
class CR1EndToEndVisualOracleContactGoalCfg(ObsGroup):
    """Oracle support state and next stable intent for the diagnostic actor."""

    next_stable_goal_b = ObsTerm(
        func=mdp.motion_semantic_target_contact_pos_b,
        params={"command_name": "motion"},
    )
    actual_support_relation = ObsTerm(
        func=mdp.actual_support_relation,
        params={"command_name": "motion"},
    )
    current_expected_support_relation = ObsTerm(
        func=mdp.current_expected_support_relation,
        params={"command_name": "motion"},
    )
    next_stable_support_relation = ObsTerm(
        func=mdp.next_stable_support_relation,
        params={"command_name": "motion"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class CR1AMPStyleObservationCfg(ObsGroup):
    """Training-only, phase-free robot state consumed by the AMP critic."""

    state = ObsTerm(
        func=mdp.amp_style_observation,
        params={"command_name": "motion"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CR1EndToEndVisualObservationsCfg:
    """Deployable visual inputs plus the privileged PPO critic."""

    critic: CR1ZDW0824FullStateObservationCfg = CR1ZDW0824FullStateObservationCfg()
    transformer_vision_depth: CR1StopCommandTransformerVisionDepthCfg = (
        CR1StopCommandTransformerVisionDepthCfg()
    )
    transformer_vision_inputs: CR1EndToEndVisualInputsCfg = CR1EndToEndVisualInputsCfg()


@configclass
class CR1EndToEndVisualOracleContactGoalObservationsCfg(
    CR1EndToEndVisualObservationsCfg
):
    """Base end-to-end observations plus diagnostic-only oracle actor intent."""

    oracle_contact_goal: CR1EndToEndVisualOracleContactGoalCfg = (
        CR1EndToEndVisualOracleContactGoalCfg()
    )


@configclass
class CR1EndToEndVisualOracleContactGoalAMPObservationsCfg(
    CR1EndToEndVisualOracleContactGoalObservationsCfg
):
    """Oracle actor inputs plus a separate training-only AMP state."""

    amp_style: CR1AMPStyleObservationCfg = CR1AMPStyleObservationCfg()


@configclass
class CR1EndToEndVisualOracleTerminationsCfg(TerminationsCfg):
    """Goal-gated task failures; dense reference-deviation failures are disabled."""

    body_local_pose = None
    required_contact = None
    stable_anchor_timeout = DoneTerm(
        func=mdp.stable_anchor_timeout,
        params={"command_name": "motion"},
    )


@configclass
class CR1EndToEndVisualOracleRewardsCfg(CR1ZDW0824SmoothOutputRewardsCfg):
    """Stable-support contact shaping for the gated Oracle visual task."""

    left_hand_contact_position = RewTerm(
        func=mdp.stable_support_phase_conditioned_limb_tracking_exp,
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
        func=mdp.stable_support_phase_conditioned_limb_tracking_exp,
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
        func=mdp.stable_support_phase_conditioned_limb_tracking_exp,
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
    next_stable_target_distance_progress = RewTerm(
        func=mdp.next_stable_target_distance_progress,
        weight=2.0,
        params={"command_name": "motion"},
    )
    next_stable_support_completion = RewTerm(
        func=mdp.stable_anchor_completion_bonus,
        weight=1.0,
        params={"command_name": "motion"},
    )


@configclass
class CR1EndToEndVisualPPOEnvCfg(
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg
):
    """Change only observations and the equivalent command-annotated motion."""

    observations: CR1EndToEndVisualObservationsCfg = CR1EndToEndVisualObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = str(ZDW0824_VISUAL_STOP_COMMAND_MOTION)


@configclass
class CR1EndToEndVisualOracleContactGoalPPOEnvCfg(CR1EndToEndVisualPPOEnvCfg):
    """Oracle-goal task with stable-support reference gating and timeout reset."""

    observations: CR1EndToEndVisualOracleContactGoalObservationsCfg = (
        CR1EndToEndVisualOracleContactGoalObservationsCfg()
    )
    scene: CR1EndToEndVisualOracleSceneCfg = CR1EndToEndVisualOracleSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
    )
    terminations: CR1EndToEndVisualOracleTerminationsCfg = (
        CR1EndToEndVisualOracleTerminationsCfg()
    )
    rewards: CR1EndToEndVisualOracleRewardsCfg = (
        CR1EndToEndVisualOracleRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        # The 61.2-s reference can now wait independently at the 26 support
        # anchors plus the inserted middle-standing anchor.  Frame 0 is the
        # initial state and frame 3060 is terminal, so they add no transition.
        # Keep the global horizon just above 61.2 + 27 * 5.5 = 209.7 s so only
        # the per-anchor failure rule decides whether a stalled transition resets.
        self.episode_length_s = 210.0
        motion = self.commands.motion
        motion.stable_anchor_gating_enabled = True
        motion.stable_anchor_confirmation_steps = 4
        motion.stable_anchor_relaxed_after_s = 2.0
        motion.stable_anchor_failure_after_s = 5.5
        motion.stable_anchor_root_height_error_threshold = 0.35
        motion.stable_anchor_root_orientation_threshold = 0.70
        motion.stable_anchor_root_linear_speed_threshold = 0.50
        motion.stable_anchor_root_angular_speed_threshold = 1.50
        motion.stable_anchor_danger_contact_force_threshold = 5.0
        motion.stable_anchor_limb_contact_sensor_names = (
            "left_hand_step_contacts",
            "right_hand_step_contacts",
            "left_foot_step_contacts",
            "right_foot_step_contacts",
        )
        motion.reset_by_stable_support_segments = True
        motion.stable_anchor_ground_contact_sensor_names = (
            "left_hand_ground_contacts",
            "right_hand_ground_contacts",
            "left_foot_ground_contacts",
            "right_foot_ground_contacts",
        )
        motion.stable_anchor_rail_contact_sensor_names = RAIL_CONTACT_SENSOR_NAMES


@configclass
class CR1EndToEndVisualOracleContactGoalAMPPPOEnvCfg(
    CR1EndToEndVisualOracleContactGoalPPOEnvCfg
):
    """Isolated AMP experiment; the underlying task MDP and gate are unchanged."""

    observations: CR1EndToEndVisualOracleContactGoalAMPObservationsCfg = (
        CR1EndToEndVisualOracleContactGoalAMPObservationsCfg()
    )


__all__ = [
    "CR1AMPStyleObservationCfg",
    "CR1EndToEndVisualPPOEnvCfg",
    "CR1EndToEndVisualObservationsCfg",
    "CR1EndToEndVisualOracleContactGoalCfg",
    "CR1EndToEndVisualOracleContactGoalAMPObservationsCfg",
    "CR1EndToEndVisualOracleContactGoalObservationsCfg",
    "CR1EndToEndVisualOracleSceneCfg",
    "CR1EndToEndVisualOracleTerminationsCfg",
    "CR1EndToEndVisualOracleRewardsCfg",
    "CR1EndToEndVisualOracleContactGoalPPOEnvCfg",
    "CR1EndToEndVisualOracleContactGoalAMPPPOEnvCfg",
]
