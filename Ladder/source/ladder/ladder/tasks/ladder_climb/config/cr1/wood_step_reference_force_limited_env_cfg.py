from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp

from .wood_step_reference_force_diagnostic_env_cfg import (
    CR1ReferenceForceDiagnosticEnvCfg,
    CR1ReferenceForceDiagnosticSceneCfg,
)
from .wood_step_reference_rate_limited_env_cfg import CR1ReferenceRateLimitedRewardsCfg


HAND_TOP_SOFT_N = 180.0
HAND_UNDESIRED_SOFT_N = 20.0
FOOT_STEP_SOFT_N = 900.0
MAX_NORMALIZED_EXCESS = 3.0


def _force_excess_term(
    sensor_name: str,
    body_name: str,
    *,
    force_soft: float,
    force_scale: float,
    surface: str,
    weight: float,
    limb_name: str | None = None,
    rung_scope: str = "all",
    max_excess: float = MAX_NORMALIZED_EXCESS,
) -> RewTerm:
    params = {
        "sensor_cfg": SceneEntityCfg(sensor_name, body_names=[body_name]),
        "force_soft": force_soft,
        "force_scale": force_scale,
        "surface": surface,
        "max_excess": max_excess,
    }
    if rung_scope != "all":
        params.update(
            command_name="motion",
            limb_name=limb_name,
            rung_scope=rung_scope,
        )
    return RewTerm(
        func=mdp.filtered_contact_force_excess_penalty,
        weight=weight,
        params=params,
    )


@configclass
class CR1ReferenceForceLimitedRewardsCfg(CR1ReferenceRateLimitedRewardsCfg):
    left_hand_target_top_force_excess = _force_excess_term(
        "left_hand_step_contacts",
        "left_sphere_hand_link",
        force_soft=HAND_TOP_SOFT_N,
        force_scale=100.0,
        surface="top",
        weight=-0.05,
        limb_name="left_hand",
        rung_scope="target",
    )
    right_hand_target_top_force_excess = _force_excess_term(
        "right_hand_step_contacts",
        "right_sphere_hand_link",
        force_soft=HAND_TOP_SOFT_N,
        force_scale=100.0,
        surface="top",
        weight=-0.05,
        limb_name="right_hand",
        rung_scope="target",
    )
    left_hand_wrong_rung_top_force_excess = _force_excess_term(
        "left_hand_step_contacts",
        "left_sphere_hand_link",
        force_soft=HAND_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="top",
        weight=-0.10,
        limb_name="left_hand",
        rung_scope="non_target",
    )
    right_hand_wrong_rung_top_force_excess = _force_excess_term(
        "right_hand_step_contacts",
        "right_sphere_hand_link",
        force_soft=HAND_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="top",
        weight=-0.10,
        limb_name="right_hand",
        rung_scope="non_target",
    )
    left_hand_front_force_excess = _force_excess_term(
        "left_hand_step_contacts",
        "left_sphere_hand_link",
        force_soft=HAND_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="front",
        weight=-0.05,
    )
    right_hand_front_force_excess = _force_excess_term(
        "right_hand_step_contacts",
        "right_sphere_hand_link",
        force_soft=HAND_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="front",
        weight=-0.05,
    )
    left_hand_underside_force_excess = _force_excess_term(
        "left_hand_step_contacts",
        "left_sphere_hand_link",
        force_soft=HAND_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="underside",
        weight=-0.05,
    )
    right_hand_underside_force_excess = _force_excess_term(
        "right_hand_step_contacts",
        "right_sphere_hand_link",
        force_soft=HAND_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="underside",
        weight=-0.05,
    )
    left_hand_side_board_force_excess = _force_excess_term(
        "left_hand_side_board_contacts",
        "left_sphere_hand_link",
        force_soft=HAND_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="all",
        weight=-0.05,
    )
    right_hand_side_board_force_excess = _force_excess_term(
        "right_hand_side_board_contacts",
        "right_sphere_hand_link",
        force_soft=HAND_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="all",
        weight=-0.05,
    )
    left_hand_ground_force_excess = _force_excess_term(
        "left_hand_ground_contacts",
        "left_sphere_hand_link",
        force_soft=HAND_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="all",
        weight=-0.05,
    )
    right_hand_ground_force_excess = _force_excess_term(
        "right_hand_ground_contacts",
        "right_sphere_hand_link",
        force_soft=HAND_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="all",
        weight=-0.05,
    )
    left_foot_step_force_excess = _force_excess_term(
        "left_foot_step_contacts",
        "left_ankle_x_link",
        force_soft=FOOT_STEP_SOFT_N,
        force_scale=300.0,
        surface="all",
        weight=-0.05,
    )
    right_foot_step_force_excess = _force_excess_term(
        "right_foot_step_contacts",
        "right_ankle_x_link",
        force_soft=FOOT_STEP_SOFT_N,
        force_scale=300.0,
        surface="all",
        weight=-0.05,
    )


@configclass
class CR1ReferenceForceLimitedEnvCfg(CR1ReferenceForceDiagnosticEnvCfg):
    """Target-rung support reward plus bounded, data-calibrated force penalties."""

    scene: CR1ReferenceForceDiagnosticSceneCfg = CR1ReferenceForceDiagnosticSceneCfg(
        num_envs=4096, env_spacing=3.0
    )
    rewards: CR1ReferenceForceLimitedRewardsCfg = CR1ReferenceForceLimitedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.contact_force_soft_limits = {
            "hand_top": HAND_TOP_SOFT_N,
            "hand_wrong_rung": HAND_UNDESIRED_SOFT_N,
            "hand_front": HAND_UNDESIRED_SOFT_N,
            "hand_underside": HAND_UNDESIRED_SOFT_N,
            "hand_side_board": HAND_UNDESIRED_SOFT_N,
            "hand_ground": HAND_UNDESIRED_SOFT_N,
            "foot_step": FOOT_STEP_SOFT_N,
        }
        self.commands.motion.contact_force_hard_limits = {
            "hand_top": 400.0,
            "hand_wrong_rung": 200.0,
            "hand_front": 200.0,
            "hand_underside": 200.0,
            "hand_side_board": 200.0,
            "hand_ground": 200.0,
            "foot_step": 1500.0,
        }
