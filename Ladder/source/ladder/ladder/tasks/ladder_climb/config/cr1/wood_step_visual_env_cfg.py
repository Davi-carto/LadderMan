import math

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp
from ladder.tasks.ladder_climb.ladder_env_cfg import (
    CriticObservationCfg,
    EventCfg,
)
from ladder.utils.robot_semantics import spawn_cr1_with_link_semantics

from .wood_step_geometry_dr_env_cfg import (
    CR1WoodStepGeometryDREnvCfg,
    CR1WoodStepGeometryDRSceneCfg,
)
from .wood_step_visual_targets_cfg import CR1VisualHeightmapTargetsCfg
from .wood_step_ssr_visual_data_cfg import (
    CR1VisualAuxiliaryTargetsCfg,
    CR1VisualEncoderInputsCfg,
)


@configclass
class CR1VisualProprioceptionCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    root_orientation = ObsTerm(func=mdp.root_orientation_6d)
    joint_pos_history = ObsTerm(func=mdp.joint_pos, history_length=9)
    action_history = ObsTerm(func=mdp.last_action, history_length=3)
    climb_direction = ObsTerm(func=mdp.climb_direction, params={"direction": 1.0})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CR1VisualDepthCfg(ObsGroup):
    image = ObsTerm(
        func=mdp.visual_depth,
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "min_depth": 0.1,
            "max_depth": 2.0,
            "gaussian_std": 0.005,
            "pixel_dropout_probability": 0.05,
            "rung_focused_mask_probability": 0.1,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CR1VisualObservationsCfg:
    visual_proprio: CR1VisualProprioceptionCfg = CR1VisualProprioceptionCfg()
    visual_depth: CR1VisualDepthCfg = CR1VisualDepthCfg()
    critic: CriticObservationCfg = CriticObservationCfg()
    visual_targets: CR1VisualHeightmapTargetsCfg = CR1VisualHeightmapTargetsCfg()
    visual_encoder_inputs: CR1VisualEncoderInputsCfg = CR1VisualEncoderInputsCfg()
    visual_aux_targets: CR1VisualAuxiliaryTargetsCfg = CR1VisualAuxiliaryTargetsCfg()


@configclass
class CR1WoodStepVisualSceneCfg(CR1WoodStepGeometryDRSceneCfg):
    depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_head_link/DepthCamera",
        update_period=0.02,
        height=60,
        width=80,
        data_types=["distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=45.5,
            clipping_range=(0.1, 2.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        depth_clipping_behavior="max",
        semantic_filter=["class"],
        colorize_semantic_segmentation=False,
    )


@configclass
class CR1VisualEventCfg(EventCfg):
    camera_extrinsics = EventTerm(
        func=mdp.randomize_camera_extrinsics,
        mode="reset",
        params={
            "camera_cfg": SceneEntityCfg("depth_camera"),
            "parent_cfg": SceneEntityCfg("robot", body_names=["d435_head_link"]),
            "translation_range": (-0.01, 0.01),
            "rotation_range": (-math.radians(1.5), math.radians(1.5)),
        },
    )


@configclass
class CR1WoodStepVisualEnvCfg(CR1WoodStepGeometryDREnvCfg):
    scene: CR1WoodStepVisualSceneCfg = CR1WoodStepVisualSceneCfg(num_envs=256, env_spacing=3.0)
    observations: CR1VisualObservationsCfg = CR1VisualObservationsCfg()
    events: CR1VisualEventCfg = CR1VisualEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.robot.spawn.func = spawn_cr1_with_link_semantics
        self.scene.depth_camera.update_period = self.decimation * self.sim.dt
        if hasattr(self, "num_rerenders_on_reset"):
            self.num_rerenders_on_reset = 1
        else:
            self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"
