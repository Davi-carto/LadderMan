from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ladder.robots.cr1 import CR1_ACTION_SCALE, CR1_CFG
from ladder.tasks.ladder_climb.ladder_env_cfg import LadderEnvCfg, LadderSceneCfg
from ladder.tasks.ladder_climb.ladder_geometry import generate_ladder
from ladder.utils.contact_schema import (
    CR1_CONTACT_DISTANCE_THRESHOLDS,
    CR1_COMMAND_CONTACT_TARGET_OFFSETS,
    CR1_INITIAL_SOLE_TARGET_Z,
    CR1_LIMB_BODY_NAMES,
)
from ladder.utils.ladder_presets import HOLOSOMA_CR1_WOOD_STEP


REPO_ROOT = Path(__file__).resolve().parents[8]
WOOD_STEP_LADDER_USD = (
    REPO_ROOT / "assets" / "ladders" / "ladder_A_aframe" / "wood_step_ladder_physics.usda"
)
CR1_TRAINING_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "restored_prefix_v300_stride2_slow50"
    / "smooth_start"
    / "climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_train_ready.npz"
)
CR1_GROUND_Z = CR1_INITIAL_SOLE_TARGET_Z
WOOD_STEP_GEOMETRY = generate_ladder(
    phi=HOLOSOMA_CR1_WOOD_STEP.phi_deg,
    rung_spacing=HOLOSOMA_CR1_WOOD_STEP.rung_spacing,
    rung_length=HOLOSOMA_CR1_WOOD_STEP.rung_length,
    rung_radius_or_width=HOLOSOMA_CR1_WOOD_STEP.rung_width,
    num_rungs=HOLOSOMA_CR1_WOOD_STEP.num_rungs,
    origin=HOLOSOMA_CR1_WOOD_STEP.target_origin,
)


@configclass
class CR1WoodStepSceneCfg(LadderSceneCfg):
    rung_00 = None
    rung_01 = None
    rung_02 = None
    rung_03 = None
    rung_04 = None
    rung_05 = None
    rung_06 = None
    rung_07 = None
    rung_08 = None
    rung_09 = None
    rung_10 = None
    rung_11 = None
    rail_left = None
    rail_right = None
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            color=(0.12, 0.12, 0.12),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, CR1_GROUND_Z)),
        collision_group=-1,
    )
    ladder = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ladder",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WOOD_STEP_LADDER_USD)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.38, -0.04, 0.0),
            rot=(2**-0.5, 0.0, 0.0, -(2**-0.5)),
        ),
        collision_group=0,
    )


@configclass
class CR1WoodStepEnvCfg(LadderEnvCfg):
    scene: CR1WoodStepSceneCfg = CR1WoodStepSceneCfg(num_envs=4096, env_spacing=3.0)

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain = None
        self.sim.physics_material = self.scene.ground.spawn.physics_material
        self.scene.robot = CR1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = CR1_ACTION_SCALE
        self.commands.motion.motion_file = str(CR1_TRAINING_MOTION)
        self.commands.motion.anchor_body_name = "body"
        self.commands.motion.body_names = [
            "base_link",
            "left_hip_x_link",
            "left_knee_link",
            "left_ankle_x_link",
            "right_hip_x_link",
            "right_knee_link",
            "right_ankle_x_link",
            "body",
            "left_shoulder_x_link",
            "left_elbow_link",
            "left_wrist_x_link",
            "left_sphere_hand_link",
            "right_shoulder_x_link",
            "right_elbow_link",
            "right_wrist_x_link",
            "right_sphere_hand_link",
        ]
        self.commands.motion.ladder_geometry = WOOD_STEP_GEOMETRY
        self.commands.motion.limb_body_names = CR1_LIMB_BODY_NAMES
        self.commands.motion.limb_target_offsets = CR1_COMMAND_CONTACT_TARGET_OFFSETS
        self.commands.motion.limb_target_distance_thresholds = CR1_CONTACT_DISTANCE_THRESHOLDS
        self.commands.motion.fixed_start_frame = 0
        self.commands.motion.fixed_start_probability = 0.20
        self.commands.motion.uniform_start_probability = 0.40
        self.commands.motion.pose_range = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.01),
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
            "yaw": (-0.02, 0.02),
        }
        self.commands.motion.velocity_range = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.02, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        }
        self.events.torso_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="body")

        self.rewards.motion_body_pos_asym.params["lower_body_names"] = [
            "base_link",
            "left_hip_x_link",
            "left_knee_link",
            "left_ankle_x_link",
            "right_hip_x_link",
            "right_knee_link",
            "right_ankle_x_link",
        ]
        self.rewards.motion_body_pos_asym.params["upper_body_names"] = [
            "body",
            "left_shoulder_x_link",
            "left_elbow_link",
            "left_wrist_x_link",
            "left_sphere_hand_link",
            "right_shoulder_x_link",
            "right_elbow_link",
            "right_wrist_x_link",
            "right_sphere_hand_link",
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                r"^(?!left_ankle_x_link$)(?!right_ankle_x_link$)(?!left_sphere_hand_link$)(?!right_sphere_hand_link$).+$"
            ],
        )
        for side in ("left", "right"):
            sphere_hand_sensor_cfg = SceneEntityCfg(
                "contact_forces", body_names=[f"{side}_sphere_hand_link"]
            )
            getattr(self.rewards, f"{side}_hand_swing_contact").params[
                "sensor_cfg"
            ] = sphere_hand_sensor_cfg
            getattr(self.rewards, f"{side}_hand_downward_contact").params[
                "sensor_cfg"
            ] = sphere_hand_sensor_cfg
            getattr(self.rewards, f"{side}_hand_upward_contact_force").params[
                "sensor_cfg"
            ] = sphere_hand_sensor_cfg
        # The 23.52 s reference completes before this fallback episode timeout.
        self.episode_length_s = 25.0
