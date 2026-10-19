"""Named temporal inputs and supervision for CR1 TransformerVision."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp


_TRANSFORMER_VISION_HISTORY_PARAMS = {
    "num_proprio_frames": 10,
    "num_depth_frames": 2,
    "depth_height": 60,
    "depth_width": 80,
    "climb_direction_value": 1.0,
}


@configclass
class CR1TransformerVisionInputsCfg(ObsGroup):
    """Temporal visual-encoder inputs, never acting-policy observations."""

    proprio_history = ObsTerm(
        func=mdp.cr1_transformer_vision_history,
        params=_TRANSFORMER_VISION_HISTORY_PARAMS,
    )
    depth_history = ObsTerm(func=mdp.cr1_transformer_vision_depth_history)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class CR1TransformerVisionAuxiliaryTargetsCfg(ObsGroup):
    """Privileged labels for visual pretraining, excluded from the frozen actor."""

    base_lin_vel = ObsTerm(func=mdp.cr1_transformer_vision_base_lin_vel_target)
    next_proprio = ObsTerm(
        func=mdp.cr1_transformer_vision_proprio_frame,
        params={"climb_direction_value": 1.0},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False
