"""Auxiliary four-limb height-map targets for CR1 visual tasks."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp


_HEIGHTMAP_PARAMS = {
    "robot_name": "robot",
    "ladder_name": "ladder",
    "rows": 10,
    "columns": 3,
    "x_extent": 0.315,
    "y_extent": 0.160,
    "ground_height": -0.03,
    "ray_start_height": 2.0,
    "max_ray_distance": 4.0,
}


@configclass
class CR1VisualHeightmapTargetsCfg(ObsGroup):
    """Targets kept outside actor and critic observations for Gate B."""

    relative_height = ObsTerm(
        func=mdp.cr1_limb_heightmap_target,
        params={**_HEIGHTMAP_PARAMS, "output": "relative_height_m"},
    )
    hit_valid = ObsTerm(
        func=mdp.cr1_limb_heightmap_target,
        params={**_HEIGHTMAP_PARAMS, "output": "hit_valid"},
    )
    hit_class = ObsTerm(
        func=mdp.cr1_limb_heightmap_target,
        params={**_HEIGHTMAP_PARAMS, "output": "hit_class"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        # Preserve [B, 4, 10, 3] tensors for later auxiliary-loss plumbing.
        self.concatenate_terms = False
