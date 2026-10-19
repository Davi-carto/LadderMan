from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ladder.robots.cr1 import CR1_ACTION_SCALE
from ladder.tasks.ladder_climb import mdp
from ladder.tasks.ladder_climb.ladder_env_cfg import RewardsCfg

from .wood_step_geometry_dr_env_cfg import CR1WoodStepGeometryDRSceneCfg
from .wood_step_geometry_dr_relaxed_reset_nominal30_env_cfg import (
    CR1WoodStepGeometryDRRelaxedResetNominal30EnvCfg,
)
from .wood_step_env_cfg import REPO_ROOT


CR1_REPAIRED_TRAINING_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "restored_prefix_v300_stride2_slow50"
    / "smooth_start"
    / "reference_repaired"
    / "right_hand_rung3_hold_to195"
    / "climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_arc_uniform_contact_safe_right_hand_rung3_hold_to195_train_ready_5grid.npz"
)


def _ground_contact_sensor(side: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_sphere_hand_link",
        filter_prim_paths_expr=["/World/ground", "{ENV_REGEX_NS}/Ladder_base_platform"],
        history_length=6,
        force_threshold=5.0,
        debug_vis=False,
    )


@configclass
class CR1ReferenceRateLimitedSceneCfg(CR1WoodStepGeometryDRSceneCfg):
    left_hand_ground_contacts = _ground_contact_sensor("left")
    right_hand_ground_contacts = _ground_contact_sensor("right")


@configclass
class CR1ReferenceRateLimitedRewardsCfg(RewardsCfg):
    left_hand_step_front_contact = RewTerm(
        func=mdp.step_surface_orientation_contact_penalty,
        weight=-0.40,
        params={
            "sensor_cfg": SceneEntityCfg(
                "left_hand_step_contacts", body_names=["left_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
            "min_top_normal_fraction": 0.5,
            "surface": "front",
        },
    )
    right_hand_step_front_contact = RewTerm(
        func=mdp.step_surface_orientation_contact_penalty,
        weight=-0.40,
        params={
            "sensor_cfg": SceneEntityCfg(
                "right_hand_step_contacts", body_names=["right_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
            "min_top_normal_fraction": 0.5,
            "surface": "front",
        },
    )
    left_hand_step_underside_contact = RewTerm(
        func=mdp.step_surface_orientation_contact_penalty,
        weight=-0.40,
        params={
            "sensor_cfg": SceneEntityCfg(
                "left_hand_step_contacts", body_names=["left_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
            "min_top_normal_fraction": 0.5,
            "surface": "underside",
        },
    )
    right_hand_step_underside_contact = RewTerm(
        func=mdp.step_surface_orientation_contact_penalty,
        weight=-0.40,
        params={
            "sensor_cfg": SceneEntityCfg(
                "right_hand_step_contacts", body_names=["right_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
            "min_top_normal_fraction": 0.5,
            "surface": "underside",
        },
    )
    left_hand_ground_contact = RewTerm(
        func=mdp.filtered_contact_penalty,
        weight=-0.40,
        params={
            "sensor_cfg": SceneEntityCfg(
                "left_hand_ground_contacts", body_names=["left_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
        },
    )
    right_hand_ground_contact = RewTerm(
        func=mdp.filtered_contact_penalty,
        weight=-0.40,
        params={
            "sensor_cfg": SceneEntityCfg(
                "right_hand_ground_contacts", body_names=["right_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
        },
    )
    left_hand_missing_required_step_contact = RewTerm(
        func=mdp.missing_required_step_contact_penalty,
        weight=-0.20,
        params={
            "command_name": "motion",
            "limb_name": "left_hand",
            "sensor_cfg": SceneEntityCfg(
                "left_hand_step_contacts", body_names=["left_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
            "target_distance_threshold": 0.08,
            "grace_steps": 10,
        },
    )
    right_hand_missing_required_step_contact = RewTerm(
        func=mdp.missing_required_step_contact_penalty,
        weight=-0.20,
        params={
            "command_name": "motion",
            "limb_name": "right_hand",
            "sensor_cfg": SceneEntityCfg(
                "right_hand_step_contacts", body_names=["right_sphere_hand_link"]
            ),
            "force_threshold": 5.0,
            "target_distance_threshold": 0.08,
            "grace_steps": 10,
        },
    )


@configclass
class CR1ReferenceRateLimitedEnvCfg(CR1WoodStepGeometryDRRelaxedResetNominal30EnvCfg):
    """Nominal30 geometry-DR task with hardware-oriented target filtering."""

    scene: CR1ReferenceRateLimitedSceneCfg = CR1ReferenceRateLimitedSceneCfg(
        num_envs=4096, env_spacing=3.0
    )
    rewards: CR1ReferenceRateLimitedRewardsCfg = CR1ReferenceRateLimitedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.commands.motion.motion_file = str(CR1_REPAIRED_TRAINING_MOTION)
        # Replace broad hand-contact penalties with filtered surface classes.
        for side in ("left", "right"):
            hand_position_params = getattr(
                self.rewards, f"{side}_hand_contact_position"
            ).params
            if "temperature" in hand_position_params:
                hand_position_params["temperature"] = 0.01
            getattr(self.rewards, f"{side}_hand_swing_contact").weight = 0.0
            getattr(self.rewards, f"{side}_hand_downward_contact").weight = 0.0
            getattr(self.rewards, f"{side}_hand_step_surface_contact").weight = 0.0
            getattr(self.rewards, f"{side}_hand_side_board_contact").weight = -0.40
        self.commands.motion.fixed_start_probability = 0.30
        self.commands.motion.uniform_start_probability = 0.35
        self.commands.motion.fixed_start_pose_range = {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "z": (0.0, 0.01),
            "roll": (-0.02, 0.02),
            "pitch": (-0.20, 0.20),
            "yaw": (-0.20, 0.20),
        }
        self.commands.motion.fixed_start_pose_randomization_probability = 2.0 / 3.0
        self.actions.joint_pos = mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            use_default_offset=True,
            # This is a conservative training envelope (25-31% of the URDF
            # velocity limits), pending measured CR1 controller target limits.
            max_target_velocity=6.0,
        )
        self.actions.joint_pos.scale = CR1_ACTION_SCALE
