"""CR1 visual environment backed by the native batched Warp renderer.

Unlike ``wood_step_visual_env_cfg``, this scene has no RTX/TiledCamera.  Its
actor depth term reads current IsaacLab state tensors and renders depth on the
GPU through ``mdp.warp_visual_depth``.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp
from ladder.tasks.ladder_climb.ladder_env_cfg import CriticObservationCfg

from .wood_step_geometry_dr_env_cfg import CR1WoodStepGeometryDREnvCfg
from .wood_step_visual_targets_cfg import CR1VisualHeightmapTargetsCfg
from .wood_step_ssr_visual_data_cfg import (
    CR1VisualAuxiliaryTargetsCfg,
    CR1VisualEncoderInputsCfg,
)


@configclass
class CR1WarpVisualProprioceptionCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    root_orientation = ObsTerm(func=mdp.root_orientation_6d)
    joint_pos_history = ObsTerm(func=mdp.joint_pos, history_length=9)
    action_history = ObsTerm(func=mdp.last_action, history_length=3)
    climb_direction = ObsTerm(func=mdp.climb_direction, params={"direction": 1.0})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CR1WarpVisualDepthCfg(ObsGroup):
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
class CR1WarpVisualObservationsCfg:
    visual_proprio: CR1WarpVisualProprioceptionCfg = CR1WarpVisualProprioceptionCfg()
    visual_depth: CR1WarpVisualDepthCfg = CR1WarpVisualDepthCfg()
    critic: CriticObservationCfg = CriticObservationCfg()
    visual_targets: CR1VisualHeightmapTargetsCfg = CR1VisualHeightmapTargetsCfg()
    visual_encoder_inputs: CR1VisualEncoderInputsCfg = CR1VisualEncoderInputsCfg()
    visual_aux_targets: CR1VisualAuxiliaryTargetsCfg = CR1VisualAuxiliaryTargetsCfg()


@configclass
class CR1WoodStepWarpVisualEnvCfg(CR1WoodStepGeometryDREnvCfg):
    """Geometry-DR visual task using Warp rather than the Isaac RTX renderer."""

    observations: CR1WarpVisualObservationsCfg = CR1WarpVisualObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # No TiledCamera is present, so ManagerBasedRLEnv can skip sim.render()
        # during every physics step. Warp depth is evaluated in the observation
        # manager from PhysX tensor data after the final decimation step.
        if hasattr(self, "num_rerenders_on_reset"):
            self.num_rerenders_on_reset = 0
        else:
            self.rerender_on_reset = False
