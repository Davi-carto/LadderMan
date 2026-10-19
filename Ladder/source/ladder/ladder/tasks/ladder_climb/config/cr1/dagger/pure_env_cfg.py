"""Pure visual DAgger environment with all-frame uniform resets."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from ..wood_step_transformer_vision_stop_command_env_cfg import (
    CR1StopCommandTransformerVisionDepthCfg,
)
from ..wood_step_transformer_vision_thin_rails_ppo_env_cfg import (
    CR1ThinRailsTransformerVisionInputsCfg,
)
from ..wood_step_transformer_vision_up_down_smooth_output_env_cfg import (
    CR1UpDownSmoothOutputTransformerVisionEnvCfg,
)
from ..wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824FullStateObservationCfg,
    CR1ZDW0824NoCommandFullStateRewardsCfg,
)


@configclass
class CR1PureVisualDaggerInputsCfg(CR1ThinRailsTransformerVisionInputsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.proprio_history.params = dict(self.proprio_history.params)
        self.proprio_history.params["num_proprio_frames"] = 9
        self.proprio_history.params["climb_command_name"] = "motion"
        self.proprio_history.params["include_climb_command"] = True


@configclass
class CR1PureVisualDaggerObservationsCfg:
    transformer_vision_depth: CR1StopCommandTransformerVisionDepthCfg = (
        CR1StopCommandTransformerVisionDepthCfg()
    )
    transformer_vision_inputs: CR1PureVisualDaggerInputsCfg = (
        CR1PureVisualDaggerInputsCfg()
    )
    teacher: CR1ZDW0824FullStateObservationCfg = CR1ZDW0824FullStateObservationCfg()


@configclass
class CR1PureVisualDaggerEnvCfg(CR1UpDownSmoothOutputTransformerVisionEnvCfg):
    observations: CR1PureVisualDaggerObservationsCfg = CR1PureVisualDaggerObservationsCfg()
    rewards: CR1ZDW0824NoCommandFullStateRewardsCfg = (
        CR1ZDW0824NoCommandFullStateRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        for name, value in vars(self.rewards).items():
            if isinstance(value, RewardTermCfg):
                setattr(self.rewards, name, None)


__all__ = ["CR1PureVisualDaggerEnvCfg", "CR1PureVisualDaggerObservationsCfg"]
