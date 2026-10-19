"""Visual Hybrid DAgger environment with binary up/down commands."""

from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from ladder.tasks.ladder_climb import mdp
from ..wood_step_transformer_vision_stop_command_env_cfg import (
    CR1StopCommandTransformerVisionDepthCfg,
    CR1StopCommandVisualHeightmapTargetsCfg,
)
from ..wood_step_transformer_vision_thin_rails_ppo_env_cfg import (
    CR1ThinRailsTransformerVisionInputsCfg,
)
from ..wood_step_transformer_vision_up_down_smooth_output_env_cfg import (
    CR1UpDownSmoothOutputTransformerVisionEnvCfg,
)
from ..wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824FullStateObservationCfg,
)


@configclass
class CR1DaggerVisualInputsCfg(CR1ThinRailsTransformerVisionInputsCfg):
    """Five noisy proprio frames with the effective +/-1 motion command."""

    def __post_init__(self):
        super().__post_init__()
        self.proprio_history.params = dict(self.proprio_history.params)
        self.proprio_history.params["num_proprio_frames"] = 5
        self.proprio_history.params["climb_command_name"] = "motion"
        self.proprio_history.params["include_climb_command"] = True


@configclass
class CR1DaggerVisualAuxiliaryTargetsCfg(ObsGroup):
    """Reference-free base-velocity label; no future-state target."""

    base_lin_vel = ObsTerm(func=mdp.cr1_transformer_vision_base_lin_vel_target)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class CR1DaggerVisualObservationsCfg:
    transformer_vision_depth: CR1StopCommandTransformerVisionDepthCfg = (
        CR1StopCommandTransformerVisionDepthCfg()
    )
    transformer_vision_inputs: CR1DaggerVisualInputsCfg = (
        CR1DaggerVisualInputsCfg()
    )
    transformer_vision_targets: CR1StopCommandVisualHeightmapTargetsCfg = (
        CR1StopCommandVisualHeightmapTargetsCfg()
    )
    transformer_vision_auxiliary_targets: CR1DaggerVisualAuxiliaryTargetsCfg = (
        CR1DaggerVisualAuxiliaryTargetsCfg()
    )
    teacher: CR1ZDW0824FullStateObservationCfg = CR1ZDW0824FullStateObservationCfg()
    critic: CR1ZDW0824FullStateObservationCfg = CR1ZDW0824FullStateObservationCfg()


@configclass
class CR1DaggerVisualEnvCfg(CR1UpDownSmoothOutputTransformerVisionEnvCfg):
    """Stage-37 up/down task for a deployable visual DAgger student."""

    observations: CR1DaggerVisualObservationsCfg = CR1DaggerVisualObservationsCfg()


__all__ = ["CR1DaggerVisualEnvCfg", "CR1DaggerVisualObservationsCfg"]
