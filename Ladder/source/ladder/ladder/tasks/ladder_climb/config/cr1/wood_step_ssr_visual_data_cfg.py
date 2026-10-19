"""Input and supervision tensors for CR1's future SSR visual encoder."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp


_SSR_HISTORY_PARAMS = {
    "num_proprio_frames": 10,
    "num_depth_frames": 2,
    "depth_height": 60,
    "depth_width": 80,
    "climb_direction_value": 1.0,
}


@configclass
class CR1VisualEncoderInputsCfg(ObsGroup):
    """Temporal inputs for the SSR encoder, separate from the acting policy."""

    proprio_history = ObsTerm(
        func=mdp.cr1_ssr_visual_history,
        params=_SSR_HISTORY_PARAMS,
    )
    depth_history = ObsTerm(func=mdp.cr1_ssr_depth_history)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class CR1VisualAuxiliaryTargetsCfg(ObsGroup):
    """Simulator targets for SSR heads, excluded from actor and critic input."""

    base_lin_vel = ObsTerm(func=mdp.cr1_ssr_base_lin_vel_target)
    next_proprio = ObsTerm(
        func=mdp.cr1_ssr_proprio_frame,
        params={"climb_direction_value": 1.0},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False

