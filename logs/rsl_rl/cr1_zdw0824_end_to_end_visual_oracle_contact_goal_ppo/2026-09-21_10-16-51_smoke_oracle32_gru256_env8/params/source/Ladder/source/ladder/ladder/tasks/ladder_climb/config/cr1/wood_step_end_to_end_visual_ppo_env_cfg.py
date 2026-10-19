"""End-to-end visual reconstruction PPO on the retained blind-task MDP."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp

from .wood_step_transformer_vision_stop_command_env_cfg import (
    CR1StopCommandTransformerVisionDepthCfg,
    CR1StopCommandTransformerVisionInputsCfg,
    CR1StopCommandVisualHeightmapTargetsCfg,
)
from .wood_step_transformer_vision_stop_command_validated_task_env_cfg import (
    ZDW0824_VISUAL_STOP_COMMAND_MOTION,
)
from .wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824FullStateObservationCfg,
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg,
)


@configclass
class CR1EndToEndVisualInputsCfg(CR1StopCommandTransformerVisionInputsCfg):
    """Deployable command-aware history and two distinct camera exposures."""

    depth_updated = ObsTerm(func=mdp.cr1_transformer_vision_depth_updated)

    def __post_init__(self):
        super().__post_init__()
        self.proprio_history.params = dict(self.proprio_history.params)
        self.proprio_history.params["num_proprio_frames"] = 25
        self.proprio_history.params["num_depth_frames"] = 2
        self.proprio_history.params["include_climb_command"] = True
        # The wrapper obtains current proprioception from history[:, -1],
        # retaining the exact same sensor-noise realization as the encoder.


@configclass
class CR1EndToEndVisualAuxiliaryTargetsCfg(ObsGroup):
    """Training-only velocity target; no reference or next-state semantics."""

    base_lin_vel = ObsTerm(func=mdp.cr1_transformer_vision_base_lin_vel_target)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class CR1EndToEndVisualOracleContactGoalCfg(ObsGroup):
    """Oracle task intent and measured state for the isolated diagnostic."""

    next_stable_goal_b = ObsTerm(
        func=mdp.motion_semantic_target_contact_pos_b,
        params={"command_name": "motion"},
    )
    actual_limb_pos_b = ObsTerm(
        func=mdp.robot_limb_pos_b,
        params={"command_name": "motion"},
    )
    actual_ladder_contact = ObsTerm(
        func=mdp.actual_ladder_contact_indicator,
        params={"command_name": "motion"},
    )
    desired_contact = ObsTerm(
        func=mdp.next_stable_contact_indicator,
        params={"command_name": "motion"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class CR1EndToEndVisualObservationsCfg:
    """Keep all privileged labels outside the deployable actor input group."""

    critic: CR1ZDW0824FullStateObservationCfg = CR1ZDW0824FullStateObservationCfg()
    transformer_vision_depth: CR1StopCommandTransformerVisionDepthCfg = (
        CR1StopCommandTransformerVisionDepthCfg()
    )
    transformer_vision_inputs: CR1EndToEndVisualInputsCfg = CR1EndToEndVisualInputsCfg()
    transformer_vision_targets: CR1StopCommandVisualHeightmapTargetsCfg = (
        CR1StopCommandVisualHeightmapTargetsCfg()
    )
    transformer_vision_auxiliary_targets: CR1EndToEndVisualAuxiliaryTargetsCfg = (
        CR1EndToEndVisualAuxiliaryTargetsCfg()
    )


@configclass
class CR1EndToEndVisualOracleContactGoalObservationsCfg(
    CR1EndToEndVisualObservationsCfg
):
    """Base end-to-end observations plus diagnostic-only oracle actor intent."""

    oracle_contact_goal: CR1EndToEndVisualOracleContactGoalCfg = (
        CR1EndToEndVisualOracleContactGoalCfg()
    )


@configclass
class CR1EndToEndVisualPPOEnvCfg(
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg
):
    """Change only observations and the equivalent command-annotated motion."""

    observations: CR1EndToEndVisualObservationsCfg = CR1EndToEndVisualObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = str(ZDW0824_VISUAL_STOP_COMMAND_MOTION)


@configclass
class CR1EndToEndVisualOracleContactGoalPPOEnvCfg(CR1EndToEndVisualPPOEnvCfg):
    """Oracle-goal diagnostic with the base MDP unchanged."""

    observations: CR1EndToEndVisualOracleContactGoalObservationsCfg = (
        CR1EndToEndVisualOracleContactGoalObservationsCfg()
    )


__all__ = [
    "CR1EndToEndVisualPPOEnvCfg",
    "CR1EndToEndVisualObservationsCfg",
    "CR1EndToEndVisualOracleContactGoalCfg",
    "CR1EndToEndVisualOracleContactGoalObservationsCfg",
    "CR1EndToEndVisualOracleContactGoalPPOEnvCfg",
]
