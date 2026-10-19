"""Stop-command TransformerVision task with the validated nonvisual MDP design."""

from isaaclab.utils import configclass

from .wood_step_transformer_vision_stop_command_env_cfg import (
    CR1StopCommandTransformerVisionEnvCfg,
)
from .wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824NoCommandFullStateRewardsCfg,
    ZDW0824_CONTACT_TARGET_OFFSETS,
    ZDW0824_LADDER_GEOMETRY,
    ZDW0824_MIDDLE_STAND_MOTION,
)


ZDW0824_VISUAL_STOP_COMMAND_MOTION = (
    ZDW0824_MIDDLE_STAND_MOTION.parents[1]
    / "37_visual_up_down_stop_command"
    / (
        "zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_"
        "reset5mm_visual_up_down_stop_50hz_motion.npz"
    )
)
ZDW0824_TOP_DEFAULT_STAND_START_FRAME = 1500


@configclass
class CR1StopCommandValidatedTaskTransformerVisionEnvCfg(
    CR1StopCommandTransformerVisionEnvCfg
):
    """Visual observations and commands with the validated nonvisual MDP."""

    rewards: CR1ZDW0824NoCommandFullStateRewardsCfg = (
        CR1ZDW0824NoCommandFullStateRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        # Stage 37 appends command-only arrays to the validated Stage-36
        # reference; every original pose, contact, and reset array is intact.
        self.commands.motion.motion_file = str(ZDW0824_VISUAL_STOP_COMMAND_MOTION)
        self.commands.motion.ladder_geometry = ZDW0824_LADDER_GEOMETRY
        self.commands.motion.limb_target_offsets = dict(
            ZDW0824_CONTACT_TARGET_OFFSETS
        )
        self.episode_length_s = 90.0

        # Match the reset mixture that learned the complete nonvisual motion.
        # Frame 1500 is the first frame of this reference's
        # 100-frame ladder-top default-standing hold.
        self.commands.motion.fixed_start_probability = 0.20
        self.commands.motion.fixed_start_pose_randomization_probability = 0.0
        self.commands.motion.secondary_fixed_start_frame = (
            ZDW0824_TOP_DEFAULT_STAND_START_FRAME
        )
        self.commands.motion.secondary_fixed_start_probability = 0.10
        self.commands.motion.secondary_fixed_start_requires_nominal_ladder = True
        self.commands.motion.uniform_start_probability = 0.35

        self.commands.motion.fixed_start_pose_range = dict(
            self.commands.motion.fixed_start_pose_range
        )
        self.commands.motion.fixed_start_pose_range["x"] = (-0.06, 0.06)
        self.commands.motion.fixed_start_pose_range["y"] = (-0.06, 0.06)
