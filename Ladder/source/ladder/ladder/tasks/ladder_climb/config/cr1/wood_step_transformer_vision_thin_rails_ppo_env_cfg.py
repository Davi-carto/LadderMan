"""Independent ThinRails environment for scratch TransformerVision PPO."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp
from ladder.tasks.ladder_climb.ladder_env_cfg import CriticObservationCfg, EventCfg

from .wood_step_reference_rails_protected_per_body_thin_rails_safe_frame0_env_cfg import (
    CR1ReferenceRailsProtectedPerBodyThinRailsSafeFrame0EnvCfg,
)
from .wood_step_reference_force_limited_strict_front_side_deploy_obs_env_cfg import (
    CR1DeployActorObservationCfg,
)
from .wood_step_transformer_vision_data_cfg import (
    CR1TransformerVisionAuxiliaryTargetsCfg,
    CR1TransformerVisionInputsCfg,
    _TRANSFORMER_VISION_HISTORY_PARAMS,
)
from .wood_step_transformer_vision_env_cfg import CR1TransformerVisionDepthCfg
from .wood_step_visual_targets_cfg import (
    CR1VisualHeightmapTargetsCfg,
    _HEIGHTMAP_PARAMS,
)


THIN_RAIL_BOX_SPEC_SET = "thin_rails"
_X2_MOTOR_RANDOMIZATION_PARAMS = {
    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    "motor_strength_range": (0.8, 1.2),
    "stiffness_factor_range": (0.9, 1.1),
    "damping_factor_range": (0.9, 1.1),
}


@configclass
class CR1ThinRailsTransformerVisionDepthCfg(CR1TransformerVisionDepthCfg):
    """Warp depth rendered from the same thin-rail boxes as PhysX."""

    def __post_init__(self):
        super().__post_init__()
        self.image.params = dict(self.image.params)
        self.image.params["ladder_box_spec_set"] = THIN_RAIL_BOX_SPEC_SET
        self.image.params["horizontal_fov_randomization_range_deg"] = (-1.0, 1.0)
        self.image.params["vertical_fov_randomization_range_deg"] = (-1.0, 1.0)


@configclass
class CR1ThinRailsTransformerVisionInputsCfg(CR1TransformerVisionInputsCfg):
    """Centered depth plus X2-aligned input-only proprioceptive noise."""

    def __post_init__(self):
        super().__post_init__()
        self.proprio_history.params = {
            **_TRANSFORMER_VISION_HISTORY_PARAMS,
            "depth_mean": 1.0,
            "depth_scale": 1.0,
            "add_proprio_noise": True,
            "root_angular_velocity_noise": 0.5,
            "orientation_noise_rad": 0.05,
            "joint_position_noise_normalized": 0.01,
            "joint_velocity_noise": 1.5,
        }


@configclass
class CR1ThinRailsVisualHeightmapTargetsCfg(CR1VisualHeightmapTargetsCfg):
    """Four-limb auxiliary labels queried from the thin-rail boxes."""

    def __post_init__(self):
        super().__post_init__()
        for term_name in ("relative_height", "hit_valid", "hit_class"):
            term = getattr(self, term_name)
            term.params = dict(term.params)
            term.params["ladder_box_spec_set"] = THIN_RAIL_BOX_SPEC_SET


@configclass
class CR1ThinRailsTransformerVisionCriticCfg(CriticObservationCfg):
    """Existing 631-D critic plus clean terrain and current dynamics."""

    joint_vel = ObsTerm(func=mdp.joint_vel, scale=0.1)
    ladder_params = ObsTerm(func=mdp.ladder_params, params={"command_name": "motion"})
    limb_heightmaps = ObsTerm(
        func=mdp.cr1_limb_heightmap_target,
        params={
            **_HEIGHTMAP_PARAMS,
            "ladder_box_spec_set": THIN_RAIL_BOX_SPEC_SET,
            "output": "relative_height_flat",
        },
    )


@configclass
class CR1ThinRailsTransformerVisionEventsCfg(EventCfg):
    """ThinRails-only X2-style actuator variation around the CR1 model."""

    motor_strength = EventTerm(
        func=mdp.randomize_actuator_strength_and_gains,
        mode="reset",
        params=dict(_X2_MOTOR_RANDOMIZATION_PARAMS),
    )
    motor_strength_interval = EventTerm(
        func=mdp.randomize_actuator_strength_and_gains,
        mode="interval",
        interval_range_s=(8.0, 8.0),
        is_global_time=False,
        params=dict(_X2_MOTOR_RANDOMIZATION_PARAMS),
    )


@configclass
class CR1ThinRailsTransformerVisionPPOObservationsCfg:
    """Unchanged 136-D actor inputs, 780-D critic, and auxiliary labels."""

    critic: CR1ThinRailsTransformerVisionCriticCfg = (
        CR1ThinRailsTransformerVisionCriticCfg()
    )
    transformer_vision_depth: CR1ThinRailsTransformerVisionDepthCfg = (
        CR1ThinRailsTransformerVisionDepthCfg()
    )
    transformer_vision_inputs: CR1ThinRailsTransformerVisionInputsCfg = (
        CR1ThinRailsTransformerVisionInputsCfg()
    )
    transformer_vision_targets: CR1ThinRailsVisualHeightmapTargetsCfg = (
        CR1ThinRailsVisualHeightmapTargetsCfg()
    )
    transformer_vision_auxiliary_targets: CR1TransformerVisionAuxiliaryTargetsCfg = (
        CR1TransformerVisionAuxiliaryTargetsCfg()
    )


@configclass
class CR1ThinRailsTransformerVisionPreflightObservationsCfg(
    CR1ThinRailsTransformerVisionPPOObservationsCfg
):
    """Training observations plus the frozen policy input for no-update inspection."""

    policy: CR1DeployActorObservationCfg = CR1DeployActorObservationCfg()


@configclass
class CR1ThinRailsTransformerVisionPPOEnvCfg(
    CR1ReferenceRailsProtectedPerBodyThinRailsSafeFrame0EnvCfg
):
    """Latest ThinRails task with Warp vision and scratch visual PPO data."""

    observations: CR1ThinRailsTransformerVisionPPOObservationsCfg = (
        CR1ThinRailsTransformerVisionPPOObservationsCfg()
    )
    events: CR1ThinRailsTransformerVisionEventsCfg = (
        CR1ThinRailsTransformerVisionEventsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self, "num_rerenders_on_reset"):
            self.num_rerenders_on_reset = 0
        else:
            self.rerender_on_reset = False
        # Keep the 50-Hz policy with four physics steps; delay is 0 or 5 ms.
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.actions.joint_pos.min_control_delay = 0
        self.actions.joint_pos.max_control_delay = 1


@configclass
class CR1ThinRailsTransformerVisionPreflightEnvCfg(
    CR1ThinRailsTransformerVisionPPOEnvCfg
):
    """No-update preflight driven by the established 660-D ThinRails policy."""

    observations: CR1ThinRailsTransformerVisionPreflightObservationsCfg = (
        CR1ThinRailsTransformerVisionPreflightObservationsCfg()
    )
