"""Recurrent visual DAgger environment with oracle student-only inputs."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp

from .synced_env_cfg import (
    CR1RecurrentVisualMseDaggerObservationsCfg,
    CR1SyncedVisualDaggerEnvCfg,
)


@configclass
class CR1PrivilegedActorInputsCfg(ObsGroup):
    """Oracle task signals used only to test information sufficiency."""

    next_stable_target_pos_b = ObsTerm(
        func=mdp.motion_semantic_target_contact_pos_b,
        params={"command_name": "motion"},
    )
    actual_ladder_contact = ObsTerm(
        func=mdp.actual_ladder_contact_indicator,
        params={"command_name": "motion"},
    )
    limb_linear_velocity_b = ObsTerm(
        func=mdp.robot_limb_lin_vel_b,
        params={"command_name": "motion"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CR1PrivilegedRecurrentVisualObservationsCfg(
    CR1RecurrentVisualMseDaggerObservationsCfg
):
    """Normal recurrent visual inputs plus a privileged oracle tensor."""

    privileged_actor: CR1PrivilegedActorInputsCfg = CR1PrivilegedActorInputsCfg()


@configclass
class CR1SyncedPrivilegedRecurrentVisualMseDaggerEnvCfg(
    CR1SyncedVisualDaggerEnvCfg
):
    """Non-deployable privileged-information DAgger ablation."""

    observations: CR1PrivilegedRecurrentVisualObservationsCfg = (
        CR1PrivilegedRecurrentVisualObservationsCfg()
    )


__all__ = [
    "CR1PrivilegedActorInputsCfg",
    "CR1PrivilegedRecurrentVisualObservationsCfg",
    "CR1SyncedPrivilegedRecurrentVisualMseDaggerEnvCfg",
]
