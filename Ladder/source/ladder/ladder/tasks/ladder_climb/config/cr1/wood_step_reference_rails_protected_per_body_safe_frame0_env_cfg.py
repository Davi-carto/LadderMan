from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ladder.robots.cr1 import CR1_COLLISION_BODY_NAMES
from ladder.tasks.ladder_climb import mdp

from .wood_step_geometry_dr_env_cfg import (
    WOOD_STEP_BOX_SPECS,
    WOOD_STEP_SIDE_BOARD_FILTER_PATHS,
    _wood_step_ladder_collection_cfg,
)
from .wood_step_reference_force_diagnostic_env_cfg import (
    CR1ReferenceForceDiagnosticSceneCfg,
)
from .wood_step_reference_force_limited_strict_front_side_env_cfg import (
    CR1ReferenceForceLimitedStrictFrontSideRewardsCfg,
)
from .wood_step_reference_rails_protected_safe_frame0_env_cfg import (
    MAX_NORMALIZED_RAIL_FORCE_EXCESS,
    RAIL_CONTACT_BINARY_WEIGHT,
    RAIL_CONTACT_THRESHOLD_N,
    RAIL_FORCE_EXCESS_WEIGHT,
    RAIL_FORCE_SCALE_N,
    RAIL_FORCE_SOFT_N,
    CR1ReferenceRailsProtectedSafeFrame0EnvCfg,
)


def _rail_sensor_name(body_name: str) -> str:
    return f"rail_contacts_{body_name}"


RAIL_CONTACT_SENSOR_NAMES = tuple(
    _rail_sensor_name(body_name) for body_name in CR1_COLLISION_BODY_NAMES
)


def _robot_body_rail_contact_sensor_cfg(body_name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
        filter_prim_paths_expr=list(WOOD_STEP_SIDE_BOARD_FILTER_PATHS),
        history_length=1,
        force_threshold=RAIL_CONTACT_THRESHOLD_N,
        debug_vis=False,
    )


def _rail_binary_term(rail_index: int) -> RewTerm:
    return RewTerm(
        func=mdp.whole_robot_rail_binary_contact_penalty,
        weight=RAIL_CONTACT_BINARY_WEIGHT,
        params={
            "sensor_names": RAIL_CONTACT_SENSOR_NAMES,
            "rail_index": rail_index,
            "force_threshold": RAIL_CONTACT_THRESHOLD_N,
            "expected_rail_count": len(WOOD_STEP_SIDE_BOARD_FILTER_PATHS),
        },
    )


def _rail_force_excess_term(rail_index: int) -> RewTerm:
    return RewTerm(
        func=mdp.whole_robot_rail_force_excess_penalty,
        weight=RAIL_FORCE_EXCESS_WEIGHT,
        params={
            "sensor_names": RAIL_CONTACT_SENSOR_NAMES,
            "rail_index": rail_index,
            "force_soft": RAIL_FORCE_SOFT_N,
            "force_scale": RAIL_FORCE_SCALE_N,
            "max_excess": MAX_NORMALIZED_RAIL_FORCE_EXCESS,
            "expected_rail_count": len(WOOD_STEP_SIDE_BOARD_FILTER_PATHS),
        },
    )


@configclass
class CR1RailsProtectedPerBodySceneCfg(CR1ReferenceForceDiagnosticSceneCfg):
    """Full ladder with one-to-many rail filters on every collision-bearing robot body."""

    ladder = _wood_step_ladder_collection_cfg(WOOD_STEP_BOX_SPECS)
    left_hand_side_board_contacts = None
    right_hand_side_board_contacts = None
    for _body_name in CR1_COLLISION_BODY_NAMES:
        locals()[_rail_sensor_name(_body_name)] = _robot_body_rail_contact_sensor_cfg(
            _body_name
        )
    del _body_name


@configclass
class CR1RailsProtectedPerBodyRewardsCfg(
    CR1ReferenceForceLimitedStrictFrontSideRewardsCfg
):
    left_rail_any_body_contact = _rail_binary_term(0)
    right_rail_any_body_contact = _rail_binary_term(1)
    left_rail_force_excess = _rail_force_excess_term(0)
    right_rail_force_excess = _rail_force_excess_term(1)


@configclass
class CR1ReferenceRailsProtectedPerBodySafeFrame0EnvCfg(
    CR1ReferenceRailsProtectedSafeFrame0EnvCfg
):
    """Rail-protected task using valid per-robot-body one-to-many contact filters."""

    scene: CR1RailsProtectedPerBodySceneCfg = CR1RailsProtectedPerBodySceneCfg(
        num_envs=4096, env_spacing=3.0
    )
    rewards: CR1RailsProtectedPerBodyRewardsCfg = (
        CR1RailsProtectedPerBodyRewardsCfg()
    )
