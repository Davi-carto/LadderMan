"""Frozen-policy CR1 environment that additionally produces Warp TransformerVision data."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp
from ladder.tasks.ladder_climb.ladder_env_cfg import CriticObservationCfg

from .wood_step_reference_force_limited_strict_front_side_deploy_obs_env_cfg import (
    CR1DeployActorObservationCfg,
    CR1ReferenceForceLimitedStrictFrontSideDeployObsEnvCfg,
)
from .wood_step_transformer_vision_data_cfg import (
    CR1TransformerVisionAuxiliaryTargetsCfg,
    CR1TransformerVisionInputsCfg,
)
from .wood_step_visual_targets_cfg import CR1VisualHeightmapTargetsCfg


@configclass
class CR1TransformerVisionDepthCfg(ObsGroup):
    """Accepted Warp D435 depth stream used only by TransformerVision."""

    image = ObsTerm(
        func=mdp.warp_visual_depth,
        params={
            "robot_name": "robot",
            "ladder_name": "ladder",
            "camera_body_name": "d435_head_link",
            "width": 80,
            "height": 60,
            "update_period": 1.0 / 30.0,
            "horizontal_fov_deg": 87.0,
            "vertical_fov_deg": 58.0,
            "min_depth": 0.1,
            "max_depth": 2.0,
            "gaussian_std": 0.005,
            "pixel_dropout_probability": 0.05,
            "rung_focused_mask_probability": 0.1,
            "translation_range": (-0.01, 0.01),
            "rotation_range": (-0.0261799388, 0.0261799388),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CR1FrozenPolicyTransformerVisionObservationsCfg:
    """One environment output containing frozen-policy and visual-pretrain fields."""

    # This is byte-for-byte the nonvisual checkpoint's 660-D policy contract.
    policy: CR1DeployActorObservationCfg = CR1DeployActorObservationCfg()
    critic: CriticObservationCfg = CriticObservationCfg()
    transformer_vision_depth: CR1TransformerVisionDepthCfg = CR1TransformerVisionDepthCfg()
    transformer_vision_inputs: CR1TransformerVisionInputsCfg = CR1TransformerVisionInputsCfg()
    transformer_vision_targets: CR1VisualHeightmapTargetsCfg = CR1VisualHeightmapTargetsCfg()
    transformer_vision_auxiliary_targets: CR1TransformerVisionAuxiliaryTargetsCfg = (
        CR1TransformerVisionAuxiliaryTargetsCfg()
    )


@configclass
class CR1FrozenPolicyTransformerVisionEnvCfg(
    CR1ReferenceForceLimitedStrictFrontSideDeployObsEnvCfg
):
    """Run the trained 660-D nonvisual actor while producing Warp visual labels."""

    observations: CR1FrozenPolicyTransformerVisionObservationsCfg = (
        CR1FrozenPolicyTransformerVisionObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        # Warp uses PhysX tensors directly. Unlike RTX cameras it must not add
        # a render call to every policy step.
        if hasattr(self, "num_rerenders_on_reset"):
            self.num_rerenders_on_reset = 0
        else:
            self.rerender_on_reset = False
