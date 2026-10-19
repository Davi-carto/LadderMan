from isaaclab.utils import configclass

from .wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_original_targets_env_cfg import (
    CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsEnvCfg,
)


@configclass
class CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsSafeFrame0EnvCfg(
    CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsEnvCfg
):
    """Original-target task with a smaller frame-0 displacement toward the ladder."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.fixed_start_pose_range = dict(
            self.commands.motion.fixed_start_pose_range
        )
        self.commands.motion.fixed_start_pose_range["x"] = (-0.10, 0.06)
