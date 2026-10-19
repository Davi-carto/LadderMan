from isaaclab.utils import configclass

from .wood_step_geometry_dr_env_cfg import (
    THIN_RAIL_REMAINING_DEPTH_FRACTION,
    THIN_RAIL_WOOD_STEP_BOX_SPECS,
    _wood_step_ladder_collection_cfg,
)
from .wood_step_reference_rails_protected_per_body_safe_frame0_env_cfg import (
    CR1RailsProtectedPerBodySceneCfg,
    CR1ReferenceRailsProtectedPerBodySafeFrame0EnvCfg,
)


RAIL_REMAINING_DEPTH_FRACTION = THIN_RAIL_REMAINING_DEPTH_FRACTION
MISSING_REQUIRED_CONTACT_WEIGHT = -0.40
RAIL_BINARY_CONTACT_WEIGHT = -1.0


@configclass
class CR1RailsProtectedPerBodyThinRailsSceneCfg(
    CR1RailsProtectedPerBodySceneCfg
):
    """Per-body rail sensing with side-board depth reduced by two thirds."""

    ladder = _wood_step_ladder_collection_cfg(THIN_RAIL_WOOD_STEP_BOX_SPECS)


@configclass
class CR1ReferenceRailsProtectedPerBodyThinRailsSafeFrame0EnvCfg(
    CR1ReferenceRailsProtectedPerBodySafeFrame0EnvCfg
):
    """Isolated thin-rail variant; reference motion and contact targets are unchanged."""

    scene: CR1RailsProtectedPerBodyThinRailsSceneCfg = (
        CR1RailsProtectedPerBodyThinRailsSceneCfg(
            num_envs=4096,
            env_spacing=3.0,
        )
    )

    def __post_init__(self):
        super().__post_init__()

        for side in ("left", "right"):
            missing_contact = getattr(
                self.rewards,
                f"{side}_hand_missing_required_step_contact",
            )
            missing_contact.weight = MISSING_REQUIRED_CONTACT_WEIGHT
            missing_contact.params["require_near_target"] = False
            getattr(
                self.rewards,
                f"{side}_rail_any_body_contact",
            ).weight = RAIL_BINARY_CONTACT_WEIGHT
