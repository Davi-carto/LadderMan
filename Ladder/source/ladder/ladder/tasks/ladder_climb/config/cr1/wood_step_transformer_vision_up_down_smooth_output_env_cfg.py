"""TransformerVision task with up/down commands and smooth-output reward."""

from isaaclab.utils import configclass

from .wood_step_transformer_vision_stop_command_validated_task_env_cfg import (
    CR1StopCommandValidatedTaskTransformerVisionEnvCfg,
)
from .wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824SmoothOutputRewardsCfg,
)


@configclass
class CR1UpDownSmoothOutputTransformerVisionEnvCfg(
    CR1StopCommandValidatedTaskTransformerVisionEnvCfg
):
    """Train only the recorded up/down commands; do not sample zero commands."""

    rewards: CR1ZDW0824SmoothOutputRewardsCfg = (
        CR1ZDW0824SmoothOutputRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.stop_command_training_enabled = False
        self.commands.motion.stop_hold_reset_probability = 0.0
        self.commands.motion.stop_seek_episode_probability = 0.0
