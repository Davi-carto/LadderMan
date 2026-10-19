from isaaclab.utils import configclass

from .wood_step_reference_force_limited_env_cfg import (
    CR1ReferenceForceLimitedEnvCfg,
    CR1ReferenceForceLimitedRewardsCfg,
    _force_excess_term,
)


FRONT_SIDE_UNDESIRED_SOFT_N = 10.0
FRONT_SIDE_BINARY_WEIGHT = -0.60
FRONT_SIDE_EXCESS_WEIGHT = -0.10


@configclass
class CR1ReferenceForceLimitedStrictFrontSideRewardsCfg(
    CR1ReferenceForceLimitedRewardsCfg
):
    """Force-limited rewards with stricter front-edge and side-board terms."""

    left_hand_front_force_excess = _force_excess_term(
        "left_hand_step_contacts",
        "left_sphere_hand_link",
        force_soft=FRONT_SIDE_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="front",
        weight=FRONT_SIDE_EXCESS_WEIGHT,
    )
    right_hand_front_force_excess = _force_excess_term(
        "right_hand_step_contacts",
        "right_sphere_hand_link",
        force_soft=FRONT_SIDE_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="front",
        weight=FRONT_SIDE_EXCESS_WEIGHT,
    )
    left_hand_side_board_force_excess = _force_excess_term(
        "left_hand_side_board_contacts",
        "left_sphere_hand_link",
        force_soft=FRONT_SIDE_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="all",
        weight=FRONT_SIDE_EXCESS_WEIGHT,
    )
    right_hand_side_board_force_excess = _force_excess_term(
        "right_hand_side_board_contacts",
        "right_sphere_hand_link",
        force_soft=FRONT_SIDE_UNDESIRED_SOFT_N,
        force_scale=50.0,
        surface="all",
        weight=FRONT_SIDE_EXCESS_WEIGHT,
    )


@configclass
class CR1ReferenceForceLimitedStrictFrontSideEnvCfg(CR1ReferenceForceLimitedEnvCfg):
    """Force-limited task with stricter front-edge and side-board penalties."""

    rewards: CR1ReferenceForceLimitedStrictFrontSideRewardsCfg = (
        CR1ReferenceForceLimitedStrictFrontSideRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        for side in ("left", "right"):
            getattr(self.rewards, f"{side}_hand_step_front_contact").weight = (
                FRONT_SIDE_BINARY_WEIGHT
            )
            getattr(self.rewards, f"{side}_hand_side_board_contact").weight = (
                FRONT_SIDE_BINARY_WEIGHT
            )

        self.commands.motion.contact_force_soft_limits["hand_front"] = (
            FRONT_SIDE_UNDESIRED_SOFT_N
        )
        self.commands.motion.contact_force_soft_limits["hand_side_board"] = (
            FRONT_SIDE_UNDESIRED_SOFT_N
        )
