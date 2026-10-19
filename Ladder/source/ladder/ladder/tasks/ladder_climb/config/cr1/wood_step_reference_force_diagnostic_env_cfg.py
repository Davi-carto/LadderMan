from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .wood_step_geometry_dr_env_cfg import WOOD_STEP_CONTACT_FILTER_PATHS
from .wood_step_reference_rate_limited_env_cfg import (
    CR1ReferenceRateLimitedEnvCfg,
    CR1ReferenceRateLimitedSceneCfg,
)


def _foot_step_contact_sensor(side: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_ankle_x_link",
        filter_prim_paths_expr=list(WOOD_STEP_CONTACT_FILTER_PATHS),
        history_length=6,
        force_threshold=5.0,
        debug_vis=False,
    )


@configclass
class CR1ReferenceForceDiagnosticSceneCfg(CR1ReferenceRateLimitedSceneCfg):
    left_foot_step_contacts = _foot_step_contact_sensor("left")
    right_foot_step_contacts = _foot_step_contact_sensor("right")


@configclass
class CR1ReferenceForceDiagnosticEnvCfg(CR1ReferenceRateLimitedEnvCfg):
    """Isolated task for target-rung rewards and contact-force measurement."""

    scene: CR1ReferenceForceDiagnosticSceneCfg = CR1ReferenceForceDiagnosticSceneCfg(
        num_envs=4096, env_spacing=3.0
    )

    def __post_init__(self):
        super().__post_init__()
        for side in ("left", "right"):
            upward_force = getattr(
                self.rewards, f"{side}_hand_upward_contact_force"
            )
            upward_force.params["target_rung_only"] = True
