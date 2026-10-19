"""Upward-climbing environment for TransformerVision asymmetric PPO."""

from isaaclab.utils import configclass

from ladder.tasks.ladder_climb.ladder_env_cfg import CriticObservationCfg

from .wood_step_transformer_vision_data_cfg import (
    CR1TransformerVisionAuxiliaryTargetsCfg,
    CR1TransformerVisionInputsCfg,
)
from .wood_step_transformer_vision_env_cfg import (
    CR1FrozenPolicyTransformerVisionEnvCfg,
    CR1TransformerVisionDepthCfg,
)
from .wood_step_visual_targets_cfg import CR1VisualHeightmapTargetsCfg


@configclass
class CR1TransformerVisionPPOObservationsCfg:
    """Actor inputs, asymmetric critic state, and training-only visual labels."""

    critic: CriticObservationCfg = CriticObservationCfg()
    transformer_vision_depth: CR1TransformerVisionDepthCfg = CR1TransformerVisionDepthCfg()
    transformer_vision_inputs: CR1TransformerVisionInputsCfg = CR1TransformerVisionInputsCfg()
    transformer_vision_targets: CR1VisualHeightmapTargetsCfg = CR1VisualHeightmapTargetsCfg()
    transformer_vision_auxiliary_targets: CR1TransformerVisionAuxiliaryTargetsCfg = (
        CR1TransformerVisionAuxiliaryTargetsCfg()
    )


@configclass
class CR1TransformerVisionPPOEnvCfg(CR1FrozenPolicyTransformerVisionEnvCfg):
    """Train a new 136-D visual actor; the old 660-D policy is not observed."""

    observations: CR1TransformerVisionPPOObservationsCfg = CR1TransformerVisionPPOObservationsCfg()

