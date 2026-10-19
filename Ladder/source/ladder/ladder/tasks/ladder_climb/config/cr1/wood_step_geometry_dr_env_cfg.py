import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ladder.utils.ladder_presets import HOLOSOMA_CR1_WOOD_STEP
from ladder.utils.wood_step_ladder import (
    WOOD_STEP_FIXED_BOX_NAMES,
    WOOD_STEP_INCLINED_BOX_NAMES,
    WOOD_STEP_RUNG_BOX_NAMES,
    load_wood_step_box_specs,
    thin_front_side_board_specs,
    WoodStepBoxSpec,
)

from .wood_step_env_cfg import CR1WoodStepEnvCfg, CR1WoodStepSceneCfg, REPO_ROOT


WOOD_STEP_LADDER_URDF = (
    REPO_ROOT / "assets" / "ladders" / "ladder_A_aframe" / "wood_step_ladder_sim.urdf"
)
CR1_GEOMETRY_DR_TRAINING_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "restored_prefix_v300_stride2_slow50"
    / "smooth_start"
    / "ladder_geometry_dr"
    / "climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_train_ready_ladder_dr.npz"
)
WOOD_STEP_BOX_SPECS = load_wood_step_box_specs(
    WOOD_STEP_LADDER_URDF,
    asset_translation=HOLOSOMA_CR1_WOOD_STEP.asset_translation,
    asset_yaw_deg=HOLOSOMA_CR1_WOOD_STEP.asset_yaw_deg,
)
THIN_RAIL_REMAINING_DEPTH_FRACTION = 1.0 / 3.0
THIN_RAIL_WOOD_STEP_BOX_SPECS = thin_front_side_board_specs(
    WOOD_STEP_BOX_SPECS,
    remaining_depth_fraction=THIN_RAIL_REMAINING_DEPTH_FRACTION,
)
STOP_COMMAND_LADDER_Z_OFFSET = -0.03
STOP_COMMAND_GROUNDED_WOOD_STEP_BOX_SPECS = tuple(
    WoodStepBoxSpec(
        name=spec.name,
        size=spec.size,
        position=(
            spec.position[0],
            spec.position[1],
            spec.position[2] + STOP_COMMAND_LADDER_Z_OFFSET,
        ),
        quaternion=spec.quaternion,
    )
    for spec in WOOD_STEP_BOX_SPECS
)
STOP_COMMAND_GROUNDED_THIN_RAIL_BOX_SPECS = thin_front_side_board_specs(
    STOP_COMMAND_GROUNDED_WOOD_STEP_BOX_SPECS,
    remaining_depth_fraction=THIN_RAIL_REMAINING_DEPTH_FRACTION,
)
WOOD_STEP_NO_SIDE_BOARD_BOX_SPECS = tuple(
    spec
    for spec in WOOD_STEP_BOX_SPECS
    if not spec.name.startswith("front_zigzag_side_board_")
)
WOOD_STEP_NO_SIDE_BOARD_INCLINED_BOX_NAMES = tuple(
    name
    for name in WOOD_STEP_INCLINED_BOX_NAMES
    if not name.startswith("front_zigzag_side_board_")
)
WOOD_STEP_CONTACT_FILTER_PATHS = tuple(
    f"{{ENV_REGEX_NS}}/Ladder_{name}" for name in WOOD_STEP_RUNG_BOX_NAMES
)
WOOD_STEP_SIDE_BOARD_FILTER_PATHS = tuple(
    f"{{ENV_REGEX_NS}}/Ladder_{name}"
    for name in WOOD_STEP_INCLINED_BOX_NAMES
    if name.startswith("front_zigzag_side_board_")
)


def _box_color(name: str) -> tuple[float, float, float]:
    if name.startswith("step_") or name == "top_platform":
        return (0.86, 0.55, 0.25)
    if name.startswith("front_") or name.startswith("top_high_post"):
        return (0.78, 0.45, 0.20)
    if name.startswith("side_limit_bar"):
        return (0.55, 0.55, 0.55)
    return (0.70, 0.40, 0.18)


def _wood_step_ladder_collection_cfg(box_specs=WOOD_STEP_BOX_SPECS) -> RigidObjectCollectionCfg:
    rigid_objects = {}
    for spec in box_specs:
        rigid_objects[spec.name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Ladder_{spec.name}",
            spawn=sim_utils.CuboidCfg(
                size=spec.size,
                semantic_tags=[
                    ("class", "rung" if spec.name in WOOD_STEP_RUNG_BOX_NAMES else "ladder")
                ],
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_box_color(spec.name)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=spec.position, rot=spec.quaternion),
            collision_group=0,
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


def _hand_surface_contact_sensor_cfg(side: str, filter_paths: list[str]) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_sphere_hand_link",
        filter_prim_paths_expr=list(filter_paths),
        history_length=6,
        force_threshold=5.0,
        debug_vis=False,
    )


@configclass
class CR1WoodStepGeometryDRSceneCfg(CR1WoodStepSceneCfg):
    ladder = _wood_step_ladder_collection_cfg()
    left_hand_step_contacts = _hand_surface_contact_sensor_cfg(
        "left", list(WOOD_STEP_CONTACT_FILTER_PATHS)
    )
    right_hand_step_contacts = _hand_surface_contact_sensor_cfg(
        "right", list(WOOD_STEP_CONTACT_FILTER_PATHS)
    )
    left_hand_side_board_contacts = _hand_surface_contact_sensor_cfg(
        "left", list(WOOD_STEP_SIDE_BOARD_FILTER_PATHS)
    )
    right_hand_side_board_contacts = _hand_surface_contact_sensor_cfg(
        "right", list(WOOD_STEP_SIDE_BOARD_FILTER_PATHS)
    )


@configclass
class CR1WoodStepGeometryDREnvCfg(CR1WoodStepEnvCfg):
    scene: CR1WoodStepGeometryDRSceneCfg = CR1WoodStepGeometryDRSceneCfg(
        num_envs=4096, env_spacing=3.0
    )

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.motion_file = str(CR1_GEOMETRY_DR_TRAINING_MOTION)
        self.commands.motion.ladder_asset_name = "ladder"
        self.commands.motion.ladder_inclination_delta_range = (
            -math.radians(1.0),
            math.radians(1.0),
        )
        self.commands.motion.ladder_spacing_delta_range = (-0.01, 0.01)
        self.commands.motion.ladder_fixed_box_names = WOOD_STEP_FIXED_BOX_NAMES
        self.commands.motion.ladder_rung_box_names = WOOD_STEP_RUNG_BOX_NAMES
        self.commands.motion.ladder_inclined_box_names = WOOD_STEP_INCLINED_BOX_NAMES
        self.commands.motion.ladder_relative_observation_frame = "ladder"
        for term_name in (
            "left_hand_step_surface_contact",
            "right_hand_step_surface_contact",
            "left_hand_side_board_contact",
            "right_hand_side_board_contact",
        ):
            getattr(self.rewards, term_name).weight = -0.20
        for side in ("left", "right"):
            upward_force = getattr(self.rewards, f"{side}_hand_upward_contact_force")
            upward_force.params["sensor_cfg"] = SceneEntityCfg(
                f"{side}_hand_step_contacts", body_names=[f"{side}_sphere_hand_link"]
            )
            upward_force.params["filtered_contacts_only"] = True
