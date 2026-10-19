"""Inspect four limb-centered CR1 ladder height-map targets.

This Gate-A diagnostic replays a reference motion kinematically. It does not
add observations, alter the acting policy, or change PPO.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
DEFAULT_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "restored_prefix_v300_stride2_slow50"
    / "smooth_start"
    / "ladder_geometry_dr"
    / "climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_train_ready_ladder_dr.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "cr1_limb_heightmaps"
_WINDOW_NAME = "CR1 limb ladder height maps"
_LIMB_ORDER = ("left_hand", "right_hand", "left_foot", "right_foot")
_RAIL_NAMES = (
    "front_zigzag_side_board_left",
    "front_zigzag_side_board_right",
)


parser = argparse.ArgumentParser(description="Inspect CR1 four-limb local ladder maps.")
parser.add_argument(
    "--task",
    type=str,
    default="Ladder-Climb-CR1-Geometry-DR-v0",
    help="Registered geometry-randomized CR1 task used to create the scene.",
)
parser.add_argument("--motion_file", type=Path, default=DEFAULT_MOTION, help="Motion npz.")
parser.add_argument("--reference_frame", type=int, default=0, help="First reference frame.")
parser.add_argument("--end_frame", type=int, default=None, help="Inclusive final replay frame.")
parser.add_argument("--play", action="store_true", help="Replay the selected reference frames.")
parser.add_argument("--playback_fps", type=float, default=None, help="Defaults to motion FPS.")
parser.add_argument("--loop", action="store_true", help="Loop the selected frame range.")
parser.add_argument(
    "--ladder_inclination_delta_deg",
    type=float,
    default=0.0,
    help="Deterministic inclination offset from the nominal ladder in degrees.",
)
parser.add_argument(
    "--ladder_spacing_delta_m",
    type=float,
    default=0.0,
    help="Deterministic rung-spacing offset from nominal in meters.",
)
parser.add_argument(
    "--height_min",
    type=float,
    default=0.0,
    help="Minimum raw world hit-z value used for display colors.",
)
parser.add_argument(
    "--height_max",
    type=float,
    default=2.0,
    help="Maximum raw world hit-z value used for display colors.",
)
parser.add_argument("--display_scale", type=float, default=1.0, help="Dashboard scale.")
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT, help="Capture directory.")
parser.add_argument("--no_show", action="store_true", help="Save one capture without a window.")
parser.add_argument(
    "--smoke_steps", type=int, default=0, help="Close after N UI/playback updates; zero is unlimited."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.reference_frame < 0:
    parser.error("--reference_frame must be non-negative")
if args_cli.playback_fps is not None and args_cli.playback_fps <= 0.0:
    parser.error("--playback_fps must be positive")
if args_cli.height_max <= args_cli.height_min:
    parser.error("--height_max must be greater than --height_min")
if args_cli.display_scale <= 0.0:
    parser.error("--display_scale must be positive")
if args_cli.smoke_steps < 0:
    parser.error("--smoke_steps must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply

import ladder.tasks  # noqa: F401
from ladder.tasks.ladder_climb.config.cr1.wood_step_geometry_dr_env_cfg import (
    WOOD_STEP_BOX_SPECS,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_env_cfg import CR1_GROUND_Z
from ladder.utils.contact_schema import CR1_VIRTUAL_BODY_SPECS
from ladder.utils.limb_heightmaps import (
    HEIGHTMAP_CLASS_NAMES,
    GROUND,
    INVALID,
    LADDER_OTHER,
    NO_HIT,
    RAIL,
    RUNG,
    LimbHeightMapCfg,
    sample_limb_heightmap,
)
from ladder.utils.wood_step_ladder import WOOD_STEP_RUNG_BOX_NAMES


_MAP_CONFIGS = {
    "left_hand": LimbHeightMapCfg(10, 3, 0.315, 0.160),
    "right_hand": LimbHeightMapCfg(10, 3, 0.315, 0.160),
    "left_foot": LimbHeightMapCfg(10, 3, 0.315, 0.160),
    "right_foot": LimbHeightMapCfg(10, 3, 0.315, 0.160),
}
_CLASS_COLORS_BGR = np.asarray(
    (
        (35, 35, 35),       # no-hit
        (0, 165, 255),      # rung
        (255, 120, 40),     # rail
        (120, 90, 180),     # other ladder
        (70, 170, 70),      # ground
        (255, 0, 255),      # invalid
    ),
    dtype=np.uint8,
)
_LIMB_MARKER_COLORS = {
    "left_hand": (0.0, 0.85, 1.0),
    "right_hand": (0.2, 0.35, 1.0),
    "left_foot": (0.15, 1.0, 0.25),
    "right_foot": (1.0, 0.75, 0.05),
}


def _make_height_point_markers() -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/CR1LimbHeightMaps/hit_points",
            markers={
                limb_name: sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
                for limb_name, color in _LIMB_MARKER_COLORS.items()
            },
        )
    )


def _visualize_height_points(marker: VisualizationMarkers | None, maps) -> None:
    if marker is None:
        return
    positions = []
    marker_indices = []
    for limb_index, limb_name in enumerate(_LIMB_ORDER):
        result = maps[limb_name]
        hit = result.valid[0] & (result.hit_class[0] != NO_HIT)
        limb_points = result.points_w[0][hit].clone()
        limb_points[:, 2] += 0.01
        positions.append(limb_points)
        marker_indices.append(
            torch.full(
                (limb_points.shape[0],),
                limb_index,
                dtype=torch.long,
                device=limb_points.device,
            )
        )
    positions = torch.cat(positions, dim=0)
    marker_indices = torch.cat(marker_indices, dim=0)
    if positions.shape[0] == 0:
        marker.set_visibility(False)
        return
    marker.set_visibility(True)
    marker.visualize(positions, marker_indices=marker_indices)


def _disable_nonvisual_randomization(env_cfg) -> None:
    for event_name in (
        "robot_material",
        "robot_mass",
        "joint_position_offset",
        "torso_com",
        "root_velocity_push",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    env_cfg.commands.motion.pose_range = {
        name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.commands.motion.fixed_start_pose_range = {}
    env_cfg.commands.motion.fixed_start_pose_randomization_probability = 0.0
    env_cfg.commands.motion.velocity_range = {
        name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    inclination_delta = math.radians(args_cli.ladder_inclination_delta_deg)
    spacing_delta = args_cli.ladder_spacing_delta_m
    env_cfg.commands.motion.ladder_inclination_delta_range = (
        inclination_delta,
        inclination_delta,
    )
    env_cfg.commands.motion.ladder_spacing_delta_range = (spacing_delta, spacing_delta)
    env_cfg.commands.motion.nominal_ladder_geometry_probability = (
        1.0 if inclination_delta == 0.0 and spacing_delta == 0.0 else 0.0
    )


def _write_reference_frame(unwrapped, reference_frame: int) -> None:
    command = unwrapped.command_manager.get_term("motion")
    robot = unwrapped.scene.articulations["robot"]
    command.time_steps.fill_(reference_frame)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = command.body_pos_w[:, 0]
    root_state[:, 3:7] = command.body_quat_w[:, 0]
    root_state[:, 7:10] = command.body_lin_vel_w[:, 0]
    root_state[:, 10:13] = command.body_ang_vel_w[:, 0]
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(command.joint_pos, command.joint_vel)
    unwrapped.sim.render()
    unwrapped.scene.update(unwrapped.physics_dt)


def _limb_centers_w(robot) -> dict[str, torch.Tensor]:
    centers = {}
    for side in ("left", "right"):
        hand_index = robot.body_names.index(f"{side}_sphere_hand_link")
        centers[f"{side}_hand"] = robot.data.body_pos_w[:, hand_index]

        sole_name = f"{side}_sole_contact_point"
        ankle_name, sole_offset = CR1_VIRTUAL_BODY_SPECS[sole_name]
        ankle_index = robot.body_names.index(ankle_name)
        offset = torch.tensor(
            sole_offset,
            dtype=robot.data.body_pos_w.dtype,
            device=robot.data.body_pos_w.device,
        ).expand(robot.data.body_pos_w.shape[0], -1)
        centers[f"{side}_foot"] = (
            robot.data.body_pos_w[:, ankle_index]
            + quat_apply(robot.data.body_quat_w[:, ankle_index], offset)
        )
    return centers


def _runtime_box_tensors(ladder, device, dtype):
    specs_by_name = {spec.name: spec for spec in WOOD_STEP_BOX_SPECS}
    missing = sorted(set(ladder.object_names) - set(specs_by_name))
    if missing:
        raise ValueError(f"missing wood-step box specs for runtime objects: {missing}")
    sizes = torch.tensor(
        [specs_by_name[name].size for name in ladder.object_names],
        dtype=dtype,
        device=device,
    )
    classes = torch.tensor(
        [
            RUNG
            if name in WOOD_STEP_RUNG_BOX_NAMES
            else RAIL
            if name in _RAIL_NAMES
            else LADDER_OTHER
            for name in ladder.object_names
        ],
        dtype=torch.long,
        device=device,
    )
    return sizes, classes


def _sample_maps(unwrapped):
    robot = unwrapped.scene.articulations["robot"]
    ladder = unwrapped.scene.rigid_object_collections["ladder"]
    centers = _limb_centers_w(robot)
    box_sizes, box_classes = _runtime_box_tensors(
        ladder, robot.data.body_pos_w.device, robot.data.body_pos_w.dtype
    )
    maps = {
        name: sample_limb_heightmap(
            limb_centers_w=centers[name],
            box_positions_w=ladder.data.object_pos_w,
            box_quaternions_wxyz=ladder.data.object_quat_w,
            box_sizes=box_sizes,
            box_classes=box_classes,
            cfg=_MAP_CONFIGS[name],
            ground_height_w=torch.full(
                (centers[name].shape[0],),
                CR1_GROUND_Z,
                dtype=centers[name].dtype,
                device=centers[name].device,
            ),
        )
        for name in _LIMB_ORDER
    }
    rung_indices = [
        index for index, name in enumerate(ladder.object_names) if name in WOOD_STEP_RUNG_BOX_NAMES
    ]
    rung_centers_w = ladder.data.object_pos_w[:, rung_indices]
    return maps, centers, rung_centers_w, box_sizes, box_classes


def _map_image(values: np.ndarray, classes: np.ndarray, *, class_view: bool) -> np.ndarray:
    if class_view:
        return _CLASS_COLORS_BGR[classes]
    normalized = np.round(
        np.clip(
            (values - args_cli.height_min) / (args_cli.height_max - args_cli.height_min),
            0.0,
            1.0,
        )
        * 255.0
    ).astype(np.uint8)
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    image[classes == NO_HIT] = _CLASS_COLORS_BGR[NO_HIT]
    image[classes == INVALID] = _CLASS_COLORS_BGR[INVALID]
    return image


def _panel(
    limb_name: str,
    result,
    rung_centers_w: np.ndarray,
    *,
    class_view: bool,
    frame: int,
) -> np.ndarray:
    cfg = _MAP_CONFIGS[limb_name]
    world_height = result.raw_hit_z_w[0].detach().cpu().numpy()
    hit_class = result.hit_class[0].detach().cpu().numpy()
    # Matrix row zero is negative world X; flip it so world +X points upward.
    image = np.flipud(
        _map_image(world_height, hit_class, class_view=class_view)
    ).copy()
    map_width = max(120, int(round(cfg.y_extent * 420.0)))
    map_height = max(210, int(round(cfg.x_extent * 420.0)))
    image = cv2.resize(image, (map_width, map_height), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((430, 330, 3), 20, dtype=np.uint8)
    x0 = (canvas.shape[1] - map_width) // 2
    y0 = 55 + (300 - map_height) // 2
    y0 = max(55, y0)
    image = image[: min(map_height, canvas.shape[0] - y0 - 32)]
    map_height = image.shape[0]
    canvas[y0 : y0 + map_height, x0 : x0 + map_width] = image

    center_x = x0 + map_width // 2
    center_y = y0 + map_height // 2
    cv2.drawMarker(canvas, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.arrowedLine(canvas, (x0 - 14, y0 + map_height), (x0 - 14, y0 + 5), (255, 255, 255), 2)
    cv2.arrowedLine(
        canvas,
        (x0, y0 + map_height + 14),
        (x0 + map_width - 5, y0 + map_height + 14),
        (255, 255, 255),
        2,
    )
    cv2.putText(canvas, "+x", (x0 - 30, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(canvas, "+y", (x0 + map_width - 24, y0 + map_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    center_w = (
        result.plane_points_w[0, cfg.rows // 2, cfg.columns // 2]
        .detach()
        .cpu()
        .numpy()
    )
    x_axis_w = result.x_axis_w[0].detach().cpu().numpy()
    y_axis_w = result.y_axis_w[0].detach().cpu().numpy()
    for rung_index, rung_center_w in enumerate(rung_centers_w):
        relative_w = rung_center_w - center_w
        relative_x = float(np.dot(relative_w, x_axis_w))
        relative_y = float(np.dot(relative_w, y_axis_w))
        if (
            abs(relative_x) > 0.5 * cfg.x_extent
            or abs(relative_y) > 0.5 * cfg.y_extent
        ):
            continue
        x = int(
            round(x0 + (relative_y / cfg.y_extent + 0.5) * (map_width - 1))
        )
        y = int(
            round(y0 + (0.5 - relative_x / cfg.x_extent) * (map_height - 1))
        )
        cv2.drawMarker(canvas, (x, y), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 10, 1)
        cv2.putText(canvas, str(rung_index + 1), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 0), 1)

    valid = result.valid[0]
    hit = result.hit_valid[0]
    finite_values = result.raw_hit_z_w[0][hit]
    value_min = float(finite_values.min()) if finite_values.numel() else math.nan
    value_max = float(finite_values.max()) if finite_values.numel() else math.nan
    view_name = "classes" if class_view else "raw world hit z (m)"
    cv2.putText(
        canvas,
        f"{limb_name} | {view_name}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (240, 240, 240),
        1,
    )
    cv2.putText(
        canvas,
        f"shape={cfg.rows}x{cfg.columns} frame={frame}",
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (205, 205, 205),
        1,
    )
    cv2.putText(
        canvas,
        f"range=[{value_min:+.3f},{value_max:+.3f}] valid={float(valid.float().mean()):.3f}",
        (8, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (205, 205, 205),
        1,
    )
    return canvas


def _dashboard(maps, rung_centers_w: torch.Tensor, frame: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rung_centers_w_numpy = rung_centers_w[0].detach().cpu().numpy()
    panels = {}
    rows = []
    for pair in (("left_hand", "right_hand"), ("left_foot", "right_foot")):
        row = []
        for limb_name in pair:
            for class_view in (False, True):
                suffix = "classes" if class_view else "height"
                panel = _panel(
                    limb_name,
                    maps[limb_name],
                    rung_centers_w_numpy,
                    class_view=class_view,
                    frame=frame,
                )
                panels[f"{limb_name}_{suffix}"] = panel
                row.append(panel)
        rows.append(np.concatenate(row, axis=1))
    dashboard = np.concatenate(rows, axis=0)
    legend = np.full((30, dashboard.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(
        legend,
        "classes: no-hit=dark | rung=orange | rail=blue | ladder=purple | ground=green | invalid=magenta",
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )
    dashboard = np.concatenate((dashboard, legend), axis=0)
    return dashboard, panels


def _metadata(unwrapped, maps, centers, rung_centers_w, frame: int) -> dict:
    command = unwrapped.command_manager.get_term("motion")
    ladder = unwrapped.scene.rigid_object_collections["ladder"]
    result = {
        "reference_frame": frame,
        "ladder_inclination_deg": float(torch.rad2deg(command.ladder_phi[0])),
        "ladder_rung_spacing_m": float(command.ladder_spacing[0]),
        "ladder_origin_w": command.ladder_origins_w[0].detach().cpu().tolist(),
        "ladder_object_names": list(ladder.object_names),
        "ladder_positions_w": ladder.data.object_pos_w[0].detach().cpu().tolist(),
        "ladder_quaternions_wxyz": ladder.data.object_quat_w[0].detach().cpu().tolist(),
        "rung_centers_w": rung_centers_w[0].detach().cpu().tolist(),
        "class_names": {str(key): value for key, value in HEIGHTMAP_CLASS_NAMES.items()},
        "height_definition": "world-space hit z from gravity-aligned downward rays",
        "display_definition": "raw world-space hit z",
        "training_definition": "limb reference z minus raw world-space hit z in meters",
        "ray_definition": "world-XY horizontal grid centered at each limb; world direction (0, 0, -1)",
        "no_hit_definition": "valid diagnostic gap with raw hit z stored as 0.0 m",
        "maps": {},
    }
    for limb_name, map_result in maps.items():
        hit_class = map_result.hit_class[0]
        valid = map_result.valid[0]
        hit = map_result.hit_valid[0]
        values = map_result.raw_hit_z_w[0][hit]
        relative_values = map_result.relative_height_m[0][hit]
        result["maps"][limb_name] = {
            "shape": list(map_result.height.shape[1:]),
            "coverage_world_xy_m": [
                _MAP_CONFIGS[limb_name].x_extent,
                _MAP_CONFIGS[limb_name].y_extent,
            ],
            "limb_center_w": centers[limb_name][0].detach().cpu().tolist(),
            "limb_center_xy": map_result.center_xy[0].detach().cpu().tolist(),
            "world_hit_z_min_max_m": [float(values.min()), float(values.max())]
            if values.numel()
            else [None, None],
            "relative_height_min_max_m": [float(relative_values.min()), float(relative_values.max())]
            if relative_values.numel()
            else [None, None],
            "valid_fraction": float(valid.float().mean()),
            "hit_valid_fraction": float(hit.float().mean()),
            "class_fractions": {
                HEIGHTMAP_CLASS_NAMES[class_id]: float((hit_class == class_id).float().mean())
                for class_id in (NO_HIT, RUNG, RAIL, LADDER_OTHER, GROUND, INVALID)
            },
        }
    return result


def _save_capture(unwrapped, maps, centers, rung_centers_w, dashboard, panels, frame: int) -> None:
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = (
        f"phi_delta_{args_cli.ladder_inclination_delta_deg:+.2f}deg_"
        f"spacing_delta_{args_cli.ladder_spacing_delta_m:+.3f}m"
    ).replace("+", "p").replace("-", "m")
    stem = f"frame_{frame:05d}_{geometry}"
    cv2.imwrite(str(output_dir / f"{stem}_dashboard.png"), dashboard)
    for name, image in panels.items():
        cv2.imwrite(str(output_dir / f"{stem}_{name}.png"), image)
    for limb_name, result in maps.items():
        np.save(output_dir / f"{stem}_{limb_name}_height_m.npy", result.height[0].detach().cpu().numpy())
        np.save(
            output_dir / f"{stem}_{limb_name}_relative_height_m.npy",
            result.relative_height_m[0].detach().cpu().numpy(),
        )
        np.save(output_dir / f"{stem}_{limb_name}_class.npy", result.hit_class[0].detach().cpu().numpy())
        np.save(output_dir / f"{stem}_{limb_name}_valid.npy", result.valid[0].detach().cpu().numpy())
        np.save(output_dir / f"{stem}_{limb_name}_hit_valid.npy", result.hit_valid[0].detach().cpu().numpy())
        np.save(output_dir / f"{stem}_{limb_name}_points_w.npy", result.points_w[0].detach().cpu().numpy())
    metadata = _metadata(unwrapped, maps, centers, rung_centers_w, frame)
    with (output_dir / f"{stem}_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    print(f"[LIMB HEIGHTMAP] Saved frame {frame} to {output_dir}", flush=True)
    for limb_name, summary in metadata["maps"].items():
        print(
            f"[LIMB HEIGHTMAP] {limb_name}: shape={tuple(summary['shape'])} "
            f"world_hit_z={summary['world_hit_z_min_max_m']} valid={summary['valid_fraction']:.3f} "
            f"relative_m={summary['relative_height_min_max_m']} hit_valid={summary['hit_valid_fraction']:.3f} "
            f"classes={summary['class_fractions']}",
            flush=True,
        )


def _opencv_highgui_available() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", maxsplit=1)[1].strip().upper() != "NONE"
    return True


class _DashboardWindow:
    def __init__(self) -> None:
        self._backend = "opencv" if _opencv_highgui_available() else "omni"
        self._window = None
        self._provider = None
        self._omni_ui = None
        if self._backend == "omni":
            import omni.ui

            self._omni_ui = omni.ui
            print("[LIMB HEIGHTMAP] OpenCV HighGUI unavailable; using Isaac Sim window.", flush=True)

    def show(self, image: np.ndarray) -> bool:
        resized = cv2.resize(
            image,
            dsize=None,
            fx=args_cli.display_scale,
            fy=args_cli.display_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        if self._backend == "opencv":
            cv2.imshow(_WINDOW_NAME, resized)
            return (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q"))
        rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
        height, width = rgba.shape[:2]
        if self._provider is None:
            self._provider = self._omni_ui.ByteImageProvider()
            self._window = self._omni_ui.Window(
                _WINDOW_NAME, width=width, height=height + 30, visible=True
            )
            with self._window.frame:
                self._omni_ui.ImageWithProvider(
                    self._provider,
                    fill_policy=self._omni_ui.IwpFillPolicy.IWP_STRETCH,
                )
        self._provider.set_bytes_data(rgba.flatten().data, [width, height])
        return False

    def close(self) -> None:
        if self._backend == "opencv":
            cv2.destroyAllWindows()
        elif self._window is not None:
            self._window.visible = False
            self._window.destroy()


def _capture(unwrapped, frame: int):
    maps, centers, rung_centers_w, _, _ = _sample_maps(unwrapped)
    dashboard, panels = _dashboard(maps, rung_centers_w, frame)
    return maps, centers, rung_centers_w, dashboard, panels


def main() -> None:
    motion_file = args_cli.motion_file.expanduser().resolve()
    if not motion_file.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_cfg.commands.motion.fixed_reference_frame = args_cli.reference_frame
    env_cfg.commands.motion.advance_reference = False
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    # Keep the viewport user-controlled instead of resetting it to the robot every render.
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = None
    _disable_nonvisual_randomization(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    window = None
    height_point_marker = None
    try:
        unwrapped = env.unwrapped
        env.reset()
        if not args_cli.headless:
            height_point_marker = _make_height_point_markers()
        command = unwrapped.command_manager.get_term("motion")
        frame_count = command.motion.time_step_total
        if args_cli.reference_frame >= frame_count:
            raise ValueError(
                f"--reference_frame must be in [0, {frame_count}), got {args_cli.reference_frame}"
            )
        end_frame = frame_count - 1 if args_cli.end_frame is None else args_cli.end_frame
        if end_frame < args_cli.reference_frame or end_frame >= frame_count:
            raise ValueError(
                f"--end_frame must be in [{args_cli.reference_frame}, {frame_count}), got {end_frame}"
            )
        playback_fps = command.motion.fps if args_cli.playback_fps is None else args_cli.playback_fps
        print(
            "[LIMB HEIGHTMAP] "
            "all-limbs=10x3/0.315x0.160m "
            "grid=world-XY-horizontal rays=world-(0,0,-1) "
            f"geometry_delta=({args_cli.ladder_inclination_delta_deg:+.3f}deg, "
            f"{args_cli.ladder_spacing_delta_m:+.4f}m)",
            flush=True,
        )
        if not args_cli.no_show and not args_cli.headless:
            window = _DashboardWindow()

        if not args_cli.play:
            _write_reference_frame(unwrapped, args_cli.reference_frame)
            maps, centers, rung_centers_w, dashboard, panels = _capture(
                unwrapped, args_cli.reference_frame
            )
            _visualize_height_points(height_point_marker, maps)
            unwrapped.sim.render()
            _save_capture(
                unwrapped,
                maps,
                centers,
                rung_centers_w,
                dashboard,
                panels,
                args_cli.reference_frame,
            )
            if window is None:
                return
            quit_requested = window.show(dashboard)
            update_count = 0
            while simulation_app.is_running() and not quit_requested:
                simulation_app.update()
                quit_requested = window.show(dashboard)
                update_count += 1
                if args_cli.smoke_steps and update_count >= args_cli.smoke_steps:
                    break
            return

        print(
            f"[LIMB HEIGHTMAP] Playing frames {args_cli.reference_frame}..{end_frame} "
            f"at {playback_fps:g} fps (loop={args_cli.loop}).",
            flush=True,
        )
        frame = args_cli.reference_frame
        update_count = 0
        next_frame_time = time.perf_counter()
        quit_requested = False
        while simulation_app.is_running() and not quit_requested:
            _write_reference_frame(unwrapped, frame)
            maps, centers, rung_centers_w, dashboard, panels = _capture(unwrapped, frame)
            _visualize_height_points(height_point_marker, maps)
            unwrapped.sim.render()
            if update_count == 0:
                _save_capture(
                    unwrapped,
                    maps,
                    centers,
                    rung_centers_w,
                    dashboard,
                    panels,
                    frame,
                )
            if window is not None:
                quit_requested = window.show(dashboard)
            print(f"\r[LIMB HEIGHTMAP] frame={frame}/{end_frame}", end="", flush=True)
            update_count += 1
            if args_cli.smoke_steps and update_count >= args_cli.smoke_steps:
                break
            if frame == end_frame:
                if not args_cli.loop:
                    break
                frame = args_cli.reference_frame
            else:
                frame += 1
            next_frame_time += 1.0 / playback_fps
            time.sleep(max(0.0, next_frame_time - time.perf_counter()))
        print(flush=True)
    finally:
        if window is not None:
            window.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
