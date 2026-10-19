"""V4 TransformerVision training without a deployable climb command."""

from isaaclab.utils import configclass

from .wood_step_transformer_vision_data_cfg import (
    CR1TransformerVisionAuxiliaryTargetsCfg,
)
from .wood_step_transformer_vision_stop_command_env_cfg import (
    CR1StopCommandThinRailsSceneCfg,
    CR1StopCommandTransformerVisionDepthCfg,
    CR1StopCommandVisualHeightmapTargetsCfg,
    STOP_COMMAND_LADDER_GEOMETRY,
    STOP_COMMAND_MOTION,
    V4_CONTACT_TARGET_OFFSETS,
)
from .wood_step_transformer_vision_thin_rails_ppo_env_cfg import (
    CR1ThinRailsTransformerVisionCriticCfg,
    CR1ThinRailsTransformerVisionInputsCfg,
    CR1ThinRailsTransformerVisionPPOEnvCfg,
)


V4_NO_COMMAND_FILTERED_MOTION = (
    STOP_COMMAND_MOTION.parent
    / "reset_filtered_tol010"
    / (
        "zdw0818_capture5_v3_full_50hz_top_terminal_default_"
        "no_command_reset_filtered_tol010.npz"
    )
)


@configclass
class CR1V4NoCommandTransformerVisionInputsCfg(
    CR1ThinRailsTransformerVisionInputsCfg
):
    """Use the established 85-D contract with a zero command pad."""

    def __post_init__(self):
        super().__post_init__()
        self.proprio_history.params = dict(self.proprio_history.params)
        self.proprio_history.params.pop("climb_command_name", None)
        self.proprio_history.params["include_climb_command"] = False


@configclass
class CR1V4NoCommandTransformerVisionAuxiliaryTargetsCfg(
    CR1TransformerVisionAuxiliaryTargetsCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.next_proprio.params = dict(self.next_proprio.params)
        self.next_proprio.params.pop("climb_command_name", None)
        self.next_proprio.params["include_climb_command"] = False


@configclass
class CR1V4NoCommandTransformerVisionObservationsCfg:
    critic: CR1ThinRailsTransformerVisionCriticCfg = (
        CR1ThinRailsTransformerVisionCriticCfg()
    )
    transformer_vision_depth: CR1StopCommandTransformerVisionDepthCfg = (
        CR1StopCommandTransformerVisionDepthCfg()
    )
    transformer_vision_inputs: CR1V4NoCommandTransformerVisionInputsCfg = (
        CR1V4NoCommandTransformerVisionInputsCfg()
    )
    transformer_vision_targets: CR1StopCommandVisualHeightmapTargetsCfg = (
        CR1StopCommandVisualHeightmapTargetsCfg()
    )
    transformer_vision_auxiliary_targets: (
        CR1V4NoCommandTransformerVisionAuxiliaryTargetsCfg
    ) = CR1V4NoCommandTransformerVisionAuxiliaryTargetsCfg()


@configclass
class CR1V4NoCommandTransformerVisionEnvCfg(
    CR1ThinRailsTransformerVisionPPOEnvCfg
):
    """Full v4 reference with no up/down/stop signal or stop-state sampling."""

    scene: CR1StopCommandThinRailsSceneCfg = CR1StopCommandThinRailsSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
    )
    observations: CR1V4NoCommandTransformerVisionObservationsCfg = (
        CR1V4NoCommandTransformerVisionObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.commands.motion.motion_file = str(V4_NO_COMMAND_FILTERED_MOTION)
        self.commands.motion.ladder_geometry = STOP_COMMAND_LADDER_GEOMETRY
        self.commands.motion.limb_target_offsets = dict(V4_CONTACT_TARGET_OFFSETS)
        self.commands.motion.stop_command_training_enabled = False
        self.commands.motion.stop_hold_reset_probability = 0.0
        self.commands.motion.stop_seek_episode_probability = 0.0
