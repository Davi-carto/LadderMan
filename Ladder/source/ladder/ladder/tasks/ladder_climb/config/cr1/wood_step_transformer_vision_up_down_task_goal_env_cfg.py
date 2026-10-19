"""Up/down TransformerVision task with supervised four-limb task semantics."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp

from .wood_step_transformer_vision_stop_command_env_cfg import (
    CR1StopCommandTransformerVisionAuxiliaryTargetsCfg,
    CR1StopCommandTransformerVisionObservationsCfg,
)
from .wood_step_transformer_vision_up_down_smooth_output_env_cfg import (
    CR1UpDownSmoothOutputTransformerVisionEnvCfg,
)
from .wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824FullStateObservationCfg,
)


@configclass
class CR1UpDownTaskGoalTransformerVisionAuxiliaryTargetsCfg(
    CR1StopCommandTransformerVisionAuxiliaryTargetsCfg
):
    """Privileged task labels; none of these terms enter the actor directly."""

    task_limb_goal_b = ObsTerm(
        func=mdp.task_limb_goal_pos_b,
        params={"command_name": "motion"},
    )
    task_limb_goal_valid = ObsTerm(
        func=mdp.task_limb_goal_valid,
        params={"command_name": "motion"},
    )
    task_contact_indicator = ObsTerm(
        func=mdp.contact_indicator,
        params={"command_name": "motion"},
    )
    task_motion_joint_pos = ObsTerm(
        func=mdp.motion_semantic_target_joint_pos,
        params={"command_name": "motion"},
    )
    task_motion_root_orientation_6d = ObsTerm(
        func=mdp.motion_semantic_target_root_orientation_6d,
        params={"command_name": "motion"},
    )
    task_motion_valid = ObsTerm(
        func=mdp.motion_semantic_target_valid,
        params={"command_name": "motion"},
    )


@configclass
class CR1UpDownTaskGoalTransformerVisionObservationsCfg(
    CR1StopCommandTransformerVisionObservationsCfg
):
    """Reuse every validated visual group and replace only auxiliary labels."""

    transformer_vision_auxiliary_targets: (
        CR1UpDownTaskGoalTransformerVisionAuxiliaryTargetsCfg
    ) = CR1UpDownTaskGoalTransformerVisionAuxiliaryTargetsCfg()


@configclass
class CR1UpDownTaskGoalTransformerVisionEnvCfg(
    CR1UpDownSmoothOutputTransformerVisionEnvCfg
):
    """Independent task-goal variant with state-locked stable-pose labels."""

    observations: CR1UpDownTaskGoalTransformerVisionObservationsCfg = (
        CR1UpDownTaskGoalTransformerVisionObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_semantic_progress_enabled = True


@configclass
class CR1UpDownTaskGoalFrozenPolicyObservationsCfg(
    CR1UpDownTaskGoalTransformerVisionObservationsCfg
):
    """Task-goal labels plus the successful nonvisual actor's 780-D input."""

    policy: CR1ZDW0824FullStateObservationCfg = CR1ZDW0824FullStateObservationCfg()


@configclass
class CR1UpDownTaskGoalFrozenPolicyEnvCfg(
    CR1UpDownTaskGoalTransformerVisionEnvCfg
):
    """Visual-only pretraining environment driven by the frozen full-state actor."""

    observations: CR1UpDownTaskGoalFrozenPolicyObservationsCfg = (
        CR1UpDownTaskGoalFrozenPolicyObservationsCfg()
    )
