from isaaclab.utils import configclass

from .wood_step_geometry_dr_env_cfg import (
    WOOD_STEP_NO_SIDE_BOARD_BOX_SPECS,
    WOOD_STEP_NO_SIDE_BOARD_INCLINED_BOX_NAMES,
    _wood_step_ladder_collection_cfg,
)
from .wood_step_reference_force_diagnostic_env_cfg import CR1ReferenceForceDiagnosticSceneCfg
from .wood_step_reference_force_limited_strict_front_side_deploy_obs_env_cfg import (
    CR1ReferenceForceLimitedStrictFrontSideDeployObsEnvCfg,
)


@configclass
class CR1NoSideBoardsSceneCfg(CR1ReferenceForceDiagnosticSceneCfg):
    """Force-diagnostic scene with the two nonphysical front side boards removed."""

    ladder = _wood_step_ladder_collection_cfg(WOOD_STEP_NO_SIDE_BOARD_BOX_SPECS)
    left_hand_side_board_contacts = None
    right_hand_side_board_contacts = None


@configclass
class CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsEnvCfg(
    CR1ReferenceForceLimitedStrictFrontSideDeployObsEnvCfg
):
    """Deployment-observation task using the physical ladder without front side boards."""

    scene: CR1NoSideBoardsSceneCfg = CR1NoSideBoardsSceneCfg(
        num_envs=4096, env_spacing=3.0
    )

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.ladder_inclined_box_names = (
            WOOD_STEP_NO_SIDE_BOARD_INCLINED_BOX_NAMES
        )
        for side in ("left", "right"):
            # These terms reference sensors that are intentionally absent from
            # this scene.  Remove the terms instead of only setting weight=0:
            # IsaacLab resolves SceneEntityCfg even for zero-weight terms.
            setattr(self.rewards, f"{side}_hand_side_board_contact", None)
            setattr(self.rewards, f"{side}_hand_side_board_force_excess", None)
        self.commands.motion.contact_force_soft_limits.pop("hand_side_board", None)
        self.commands.motion.contact_force_hard_limits.pop("hand_side_board", None)
