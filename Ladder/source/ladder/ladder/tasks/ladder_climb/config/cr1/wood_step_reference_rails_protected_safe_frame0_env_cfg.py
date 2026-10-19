from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp

from .wood_step_geometry_dr_env_cfg import (
    WOOD_STEP_BOX_SPECS,
    WOOD_STEP_INCLINED_BOX_NAMES,
    _wood_step_ladder_collection_cfg,
)
from .wood_step_reference_force_diagnostic_env_cfg import (
    CR1ReferenceForceDiagnosticSceneCfg,
)
from .wood_step_reference_force_limited_strict_front_side_env_cfg import (
    CR1ReferenceForceLimitedStrictFrontSideRewardsCfg,
)
from .wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_original_targets_safe_frame0_env_cfg import (
    CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsSafeFrame0EnvCfg,
)


HAND_TARGET_TOP_SOFT_N = 250.0
HAND_SUPPORT_FORCE_SATURATION_N = 100.0
HAND_FRONT_NORMAL_FRACTION = 0.35
RAIL_CONTACT_THRESHOLD_N = 5.0
RAIL_CONTACT_BINARY_WEIGHT = -2.0
RAIL_FORCE_SOFT_N = 5.0
RAIL_FORCE_SCALE_N = 50.0
RAIL_FORCE_EXCESS_WEIGHT = -0.25
MAX_NORMALIZED_RAIL_FORCE_EXCESS = 3.0

_RAIL_BOX_NAMES = {
    "left": "front_zigzag_side_board_left",
    "right": "front_zigzag_side_board_right",
}


def _rail_contact_ladder_collection_cfg():
    ladder_cfg = _wood_step_ladder_collection_cfg(WOOD_STEP_BOX_SPECS)
    for box_name in _RAIL_BOX_NAMES.values():
        ladder_cfg.rigid_objects[box_name].spawn.activate_contact_sensors = True
    return ladder_cfg


def _rail_robot_contact_sensor_cfg(side: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Ladder_{_RAIL_BOX_NAMES[side]}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"],
        history_length=6,
        force_threshold=RAIL_CONTACT_THRESHOLD_N,
        debug_vis=False,
    )


def _rail_binary_contact_term(side: str) -> RewTerm:
    return RewTerm(
        func=mdp.filtered_contact_penalty,
        weight=RAIL_CONTACT_BINARY_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg(
                f"{side}_rail_robot_contacts", body_names=[".*"]
            ),
            "force_threshold": RAIL_CONTACT_THRESHOLD_N,
        },
    )


def _rail_force_excess_term(side: str) -> RewTerm:
    return RewTerm(
        func=mdp.filtered_contact_force_excess_penalty,
        weight=RAIL_FORCE_EXCESS_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg(
                f"{side}_rail_robot_contacts", body_names=[".*"]
            ),
            "force_soft": RAIL_FORCE_SOFT_N,
            "force_scale": RAIL_FORCE_SCALE_N,
            "surface": "all",
            "max_excess": MAX_NORMALIZED_RAIL_FORCE_EXCESS,
        },
    )


@configclass
class CR1RailsProtectedSceneCfg(CR1ReferenceForceDiagnosticSceneCfg):
    """Full ladder with one-to-many robot-contact sensing on each front rail."""

    ladder = _rail_contact_ladder_collection_cfg()
    left_hand_side_board_contacts = None
    right_hand_side_board_contacts = None
    left_rail_robot_contacts = _rail_robot_contact_sensor_cfg("left")
    right_rail_robot_contacts = _rail_robot_contact_sensor_cfg("right")


@configclass
class CR1RailsProtectedRewardsCfg(CR1ReferenceForceLimitedStrictFrontSideRewardsCfg):
    left_rail_any_body_contact = _rail_binary_contact_term("left")
    right_rail_any_body_contact = _rail_binary_contact_term("right")
    left_rail_force_excess = _rail_force_excess_term("left")
    right_rail_force_excess = _rail_force_excess_term("right")


@configclass
class CR1ReferenceRailsProtectedSafeFrame0EnvCfg(
    CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsSafeFrame0EnvCfg
):
    """Safe-frame-0 task with restored rails and strict whole-robot rail avoidance."""

    scene: CR1RailsProtectedSceneCfg = CR1RailsProtectedSceneCfg(
        num_envs=4096, env_spacing=3.0
    )
    rewards: CR1RailsProtectedRewardsCfg = CR1RailsProtectedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.ladder_inclined_box_names = WOOD_STEP_INCLINED_BOX_NAMES
        self.commands.motion.contact_force_soft_limits["hand_top"] = (
            HAND_TARGET_TOP_SOFT_N
        )
        self.commands.motion.contact_front_normal_fraction = HAND_FRONT_NORMAL_FRACTION
        for side in ("left", "right"):
            getattr(
                self.rewards, f"{side}_hand_upward_contact_force"
            ).params["force_saturation"] = HAND_SUPPORT_FORCE_SATURATION_N
            getattr(
                self.rewards, f"{side}_hand_target_top_force_excess"
            ).params["force_soft"] = HAND_TARGET_TOP_SOFT_N
            getattr(
                self.rewards, f"{side}_hand_step_front_contact"
            ).params["min_top_normal_fraction"] = HAND_FRONT_NORMAL_FRACTION
            getattr(
                self.rewards, f"{side}_hand_front_force_excess"
            ).params["min_top_normal_fraction"] = HAND_FRONT_NORMAL_FRACTION
