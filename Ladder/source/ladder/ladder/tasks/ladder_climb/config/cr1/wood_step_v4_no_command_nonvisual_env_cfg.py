"""V4 no-command task using the established nonvisual policy observations."""

from isaaclab.utils import configclass

from .wood_step_reference_force_limited_strict_front_side_deploy_obs_env_cfg import (
    CR1DeployObservationsCfg,
)
from .wood_step_transformer_vision_v4_no_command_env_cfg import (
    CR1V4NoCommandTransformerVisionEnvCfg,
)


@configclass
class CR1V4NoCommandNonVisualEnvCfg(CR1V4NoCommandTransformerVisionEnvCfg):
    """Keep the v4 task unchanged while removing every visual observation."""

    observations: CR1DeployObservationsCfg = CR1DeployObservationsCfg()
