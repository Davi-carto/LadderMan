from isaaclab.utils import configclass

from .wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_env_cfg import (
    CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsEnvCfg,
)


HAND_UNDERSIDE_SOFT_N = 10.0
HAND_UNDERSIDE_BINARY_WEIGHT = -0.60
HAND_UNDERSIDE_EXCESS_WEIGHT = -0.10


@configclass
class CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsEnvCfg(
    CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsEnvCfg
):
    """No-side-board task with the original runtime hand targets restored."""

    def __post_init__(self):
        super().__post_init__()

        for side in ("left", "right"):
            getattr(
                self.rewards, f"{side}_hand_missing_required_step_contact"
            ).params["target_rung_only"] = True
            getattr(
                self.rewards, f"{side}_hand_step_underside_contact"
            ).weight = HAND_UNDERSIDE_BINARY_WEIGHT
            underside_excess = getattr(
                self.rewards, f"{side}_hand_underside_force_excess"
            )
            underside_excess.params["force_soft"] = HAND_UNDERSIDE_SOFT_N
            underside_excess.weight = HAND_UNDERSIDE_EXCESS_WEIGHT

        self.commands.motion.contact_force_soft_limits["hand_underside"] = (
            HAND_UNDERSIDE_SOFT_N
        )
