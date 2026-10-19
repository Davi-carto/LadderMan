"""Independent TransformerVision task with safe zero-command stopping."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from ladder.tasks.ladder_climb import mdp
from ladder.tasks.ladder_climb.ladder_geometry import generate_ladder
from ladder.utils.ladder_presets import HOLOSOMA_CR1_WOOD_STEP

from .wood_step_env_cfg import REPO_ROOT
from .wood_step_geometry_dr_env_cfg import (
    STOP_COMMAND_GROUNDED_THIN_RAIL_BOX_SPECS,
    STOP_COMMAND_LADDER_Z_OFFSET,
    _wood_step_ladder_collection_cfg,
)
from .wood_step_reference_rails_protected_per_body_thin_rails_safe_frame0_env_cfg import (
    CR1RailsProtectedPerBodyThinRailsSceneCfg,
)
from .wood_step_transformer_vision_data_cfg import (
    CR1TransformerVisionAuxiliaryTargetsCfg,
)
from .wood_step_transformer_vision_thin_rails_ppo_env_cfg import (
    CR1ThinRailsTransformerVisionCriticCfg,
    CR1ThinRailsTransformerVisionDepthCfg,
    CR1ThinRailsTransformerVisionInputsCfg,
    CR1ThinRailsTransformerVisionPPOEnvCfg,
    CR1ThinRailsVisualHeightmapTargetsCfg,
)


STOP_COMMAND_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "zdw0818_capture5"
    / "v3_foot160_pitch0"
    / "07_top_default_command_preview"
    / "v4_terminal_default_stop_command"
    / "zdw0818_capture5_v3_full_50hz_top_terminal_default_command_stop_preview.npz"
)
STOP_COMMAND_LADDER_GEOMETRY = generate_ladder(
    phi=HOLOSOMA_CR1_WOOD_STEP.phi_deg,
    rung_spacing=HOLOSOMA_CR1_WOOD_STEP.rung_spacing,
    rung_length=HOLOSOMA_CR1_WOOD_STEP.rung_length,
    rung_radius_or_width=HOLOSOMA_CR1_WOOD_STEP.rung_width,
    num_rungs=HOLOSOMA_CR1_WOOD_STEP.num_rungs,
    origin=(
        HOLOSOMA_CR1_WOOD_STEP.target_origin[0],
        HOLOSOMA_CR1_WOOD_STEP.target_origin[1],
        HOLOSOMA_CR1_WOOD_STEP.target_origin[2] + STOP_COMMAND_LADDER_Z_OFFSET,
    ),
)
STOP_COMMAND_LADDER_BOX_SPEC_SET = "thin_rails_grounded"
V4_CONTACT_TARGET_OFFSETS = {
    "left_hand": (-0.015, 0.21, 0.045),
    "right_hand": (-0.01, -0.22, 0.045),
    "left_foot": (0.0, 0.13, 0.0),
    "right_foot": (0.0, -0.18, 0.0),
}


@configclass
class CR1StopCommandThinRailsSceneCfg(CR1RailsProtectedPerBodyThinRailsSceneCfg):
    """Thin-rail scene aligned with the new base-grounded reference."""

    ladder = _wood_step_ladder_collection_cfg(
        STOP_COMMAND_GROUNDED_THIN_RAIL_BOX_SPECS
    )


@configclass
class CR1StopCommandTransformerVisionDepthCfg(CR1ThinRailsTransformerVisionDepthCfg):
    def __post_init__(self):
        super().__post_init__()
        self.image.params = dict(self.image.params)
        self.image.params["ladder_box_spec_set"] = STOP_COMMAND_LADDER_BOX_SPEC_SET


@configclass
class CR1StopCommandTransformerVisionInputsCfg(CR1ThinRailsTransformerVisionInputsCfg):
    """Use the planner's effective {-1, 0, +1} command in visual history."""

    def __post_init__(self):
        super().__post_init__()
        self.proprio_history.params = dict(self.proprio_history.params)
        self.proprio_history.params["climb_command_name"] = "motion"


@configclass
class CR1StopCommandTransformerVisionAuxiliaryTargetsCfg(
    CR1TransformerVisionAuxiliaryTargetsCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.next_proprio.params = dict(self.next_proprio.params)
        self.next_proprio.params["climb_command_name"] = "motion"


@configclass
class CR1StopCommandVisualHeightmapTargetsCfg(CR1ThinRailsVisualHeightmapTargetsCfg):
    def __post_init__(self):
        super().__post_init__()
        for term_name in ("relative_height", "hit_valid", "hit_class"):
            term = getattr(self, term_name)
            term.params = dict(term.params)
            term.params["ladder_box_spec_set"] = STOP_COMMAND_LADDER_BOX_SPEC_SET


@configclass
class CR1StopCommandTransformerVisionCriticCfg(
    CR1ThinRailsTransformerVisionCriticCfg
):
    climb_command = ObsTerm(
        func=mdp.motion_climb_command,
        params={"command_name": "motion"},
    )


@configclass
class CR1StopCommandTransformerVisionObservationsCfg:
    critic: CR1StopCommandTransformerVisionCriticCfg = (
        CR1StopCommandTransformerVisionCriticCfg()
    )
    transformer_vision_depth: CR1StopCommandTransformerVisionDepthCfg = (
        CR1StopCommandTransformerVisionDepthCfg()
    )
    transformer_vision_inputs: CR1StopCommandTransformerVisionInputsCfg = (
        CR1StopCommandTransformerVisionInputsCfg()
    )
    transformer_vision_targets: CR1StopCommandVisualHeightmapTargetsCfg = (
        CR1StopCommandVisualHeightmapTargetsCfg()
    )
    transformer_vision_auxiliary_targets: (
        CR1StopCommandTransformerVisionAuxiliaryTargetsCfg
    ) = CR1StopCommandTransformerVisionAuxiliaryTargetsCfg()


@configclass
class CR1StopCommandTransformerVisionEnvCfg(CR1ThinRailsTransformerVisionPPOEnvCfg):
    """New reference and command-state task; existing visual tasks are unchanged."""

    scene: CR1StopCommandThinRailsSceneCfg = CR1StopCommandThinRailsSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
    )
    observations: CR1StopCommandTransformerVisionObservationsCfg = (
        CR1StopCommandTransformerVisionObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.commands.motion.motion_file = str(STOP_COMMAND_MOTION)
        self.commands.motion.ladder_geometry = STOP_COMMAND_LADDER_GEOMETRY
        self.commands.motion.limb_target_offsets = dict(V4_CONTACT_TARGET_OFFSETS)
        self.commands.motion.stop_command_training_enabled = True
        self.commands.motion.stop_hold_reset_probability = 0.05
        self.commands.motion.stop_seek_episode_probability = 0.05
        self.commands.motion.stop_hold_time_range_s = (1.0, 3.0)
