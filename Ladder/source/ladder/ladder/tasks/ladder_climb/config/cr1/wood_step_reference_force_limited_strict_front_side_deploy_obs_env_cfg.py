from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ladder.tasks.ladder_climb import mdp
from ladder.tasks.ladder_climb.ladder_env_cfg import CriticObservationCfg

from .wood_step_reference_force_limited_strict_front_side_env_cfg import (
    CR1ReferenceForceLimitedStrictFrontSideEnvCfg,
)


@configclass
class CR1DeployActorObservationCfg(ObsGroup):
    """Actor observations available from the CR1 controller and motion player."""

    reference_joint_pos = ObsTerm(
        func=mdp.reference_joint_pos,
        params={"command_name": "motion"},
        history_length=5,
    )
    reference_joint_vel = ObsTerm(
        func=mdp.reference_joint_vel,
        params={"command_name": "motion"},
        history_length=5,
    )
    base_link_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        history_length=5,
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        history_length=5,
    )
    joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=5,
    )
    joint_vel_rel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-0.5, n_max=0.5),
        history_length=5,
    )
    actions = ObsTerm(func=mdp.last_action, history_length=5)
    motion_phase = ObsTerm(
        func=mdp.motion_phase,
        params={"command_name": "motion"},
    )
    contact_indicator = ObsTerm(
        func=mdp.contact_indicator,
        params={"command_name": "motion"},
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CR1DeployObservationsCfg:
    policy: CR1DeployActorObservationCfg = CR1DeployActorObservationCfg()
    critic: CriticObservationCfg = CriticObservationCfg()


@configclass
class CR1ReferenceForceLimitedStrictFrontSideDeployObsEnvCfg(
    CR1ReferenceForceLimitedStrictFrontSideEnvCfg
):
    """Strict front/side task with a deployment-oriented 660-D actor input."""

    observations: CR1DeployObservationsCfg = CR1DeployObservationsCfg()
