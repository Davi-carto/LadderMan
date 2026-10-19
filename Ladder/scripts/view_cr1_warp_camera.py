"""Compare RTX and custom Warp depth for one CR1 wood-step ladder scene.

This is a gated camera diagnostic. It does not replace the registered visual
task's ``TiledCamera`` or alter policy observations.
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
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "cr1_warp_camera"


parser = argparse.ArgumentParser(description="Compare CR1 RTX and Warp ladder-camera outputs.")
parser.add_argument(
    "--task",
    type=str,
    default="Ladder-Climb-CR1-Geometry-DR-Visual-v0",
    help="Registered CR1 visual task used to create the scene.",
)
parser.add_argument("--motion_file", type=Path, default=DEFAULT_MOTION, help="Geometry-DR motion npz.")
parser.add_argument(
    "--reference_frame",
    type=int,
    default=0,
    help="Reference frame to display, or first frame when --play is used.",
)
parser.add_argument("--play", action="store_true", help="Replay the reference motion frame by frame.")
parser.add_argument(
    "--end_frame",
    type=int,
    default=None,
    help="Inclusive final playback frame; defaults to the final motion frame.",
)
parser.add_argument(
    "--playback_fps",
    type=float,
    default=None,
    help="Playback rate; defaults to the reference motion fps.",
)
parser.add_argument("--loop", action="store_true", help="Loop the selected frame range during playback.")
parser.add_argument(
    "--camera_width", type=int, default=80, help="Full render width before optional crop."
)
parser.add_argument(
    "--camera_height", type=int, default=60, help="Full render height before optional crop."
)
parser.add_argument("--horizontal_fov_deg", type=float, default=87.0, help="Horizontal pinhole FOV.")
parser.add_argument("--vertical_fov_deg", type=float, default=58.0, help="Vertical pinhole FOV.")
parser.add_argument("--min_depth", type=float, default=0.1, help="Near depth in meters.")
parser.add_argument("--max_depth", type=float, default=2.0, help="Far depth in meters.")
parser.add_argument("--crop_top", type=int, default=0, help="Rows cropped from the top after rendering.")
parser.add_argument("--crop_bottom", type=int, default=0, help="Rows cropped from the bottom after rendering.")
parser.add_argument("--crop_left", type=int, default=0, help="Columns cropped from the left after rendering.")
parser.add_argument("--crop_right", type=int, default=0, help="Columns cropped from the right after rendering.")
parser.add_argument("--display_scale", type=float, default=5.0, help="Nearest-neighbor display enlargement.")
parser.add_argument("--difference_max", type=float, default=0.25, help="Depth difference display maximum.")
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT, help="Capture output directory.")
parser.add_argument("--no_show", action="store_true", help="Save one capture without image windows.")
parser.add_argument(
    "--warp_debug",
    action="store_true",
    help="Log kernel-read camera/geometry values and isolated Warp render passes.",
)
parser.add_argument(
    "--smoke_steps",
    type=int,
    default=0,
    help="Close after this many static UI ticks or playback frames; zero has no limit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.reference_frame < 0:
    parser.error("--reference_frame must be non-negative")
if args_cli.display_scale <= 0.0:
    parser.error("--display_scale must be positive")
if args_cli.difference_max <= 0.0:
    parser.error("--difference_max must be positive")
if args_cli.playback_fps is not None and args_cli.playback_fps <= 0.0:
    parser.error("--playback_fps must be positive")
if args_cli.smoke_steps < 0:
    parser.error("--smoke_steps must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch
import warp as wp

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    quat_apply,
    quat_apply_inverse,
)

import ladder.tasks  # noqa: F401
from ladder.robots.cr1 import CR1_URDF
from ladder.tasks.ladder_climb.config.cr1.wood_step_env_cfg import CR1_GROUND_Z
from ladder.tasks.ladder_climb.config.cr1.wood_step_geometry_dr_env_cfg import (
    WOOD_STEP_BOX_SPECS,
)
from ladder.utils.warp_camera import (
    BACKGROUND,
    GROUND,
    HIT_CLASS_NAMES,
    LADDER_OTHER,
    ROBOT,
    RUNG,
    CR1WarpCamera,
    WarpCameraCfg,
    WARP_CENTER_RAY_VALUE_NAMES,
    WARP_DIAGNOSTIC_COUNT_NAMES,
    aperture_from_fov,
    pinhole_fov_remap_grid,
)
from ladder.utils.wood_step_ladder import WOOD_STEP_RUNG_BOX_NAMES


_WINDOW_NAMES = (
    "RTX depth (native FOV, m)",
    "Warp depth (m)",
    "|RTX remapped-Warp| diagnostic (m)",
    "Warp hit classes",
    "Warp rung mask",
    "Warp robot mask",
    "Warp RFM mask (rungs+robot)",
    "Warp RFM depth (rungs+robot)",
)
_CAMERA_BODY_NAME = "d435_head_link"


def _disable_randomization(env_cfg) -> None:
    for event_name in (
        "robot_material",
        "robot_mass",
        "joint_position_offset",
        "torso_com",
        "root_velocity_push",
        "camera_extrinsics",
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
    env_cfg.commands.motion.ladder_inclination_delta_range = (0.0, 0.0)
    env_cfg.commands.motion.ladder_spacing_delta_range = (0.0, 0.0)
    env_cfg.commands.motion.nominal_ladder_geometry_probability = 1.0


def _configure_rtx_camera(env_cfg, cfg: WarpCameraCfg) -> None:
    focal_length = 24.0
    camera_cfg = env_cfg.scene.depth_camera
    camera_cfg.width = cfg.width
    camera_cfg.height = cfg.height
    camera_cfg.spawn.focal_length = focal_length
    camera_cfg.spawn.horizontal_aperture = aperture_from_fov(
        cfg.horizontal_fov_deg, focal_length
    )
    # Isaac Sim enforces square pixels for USD cameras, so at 80x60 the
    # vertical FOV is derived from the horizontal FOV and image aspect ratio.
    camera_cfg.spawn.vertical_aperture = (
        camera_cfg.spawn.horizontal_aperture * cfg.height / cfg.width
    )
    camera_cfg.spawn.clipping_range = (cfg.min_depth, cfg.max_depth)
    camera_cfg.depth_clipping_behavior = "max"


def _depth_2d(value: torch.Tensor) -> torch.Tensor:
    image = value[0]
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected one HxW depth image, got {tuple(image.shape)}")
    return image.to(dtype=torch.float32)


def _depth_color(depth: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    clipped = np.clip(depth, minimum, maximum)
    normalized = np.round((clipped - minimum) / (maximum - minimum) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _difference_color(difference: np.ndarray) -> np.ndarray:
    normalized = np.round(
        np.clip(difference / args_cli.difference_max, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def _class_color(hit_class: np.ndarray) -> np.ndarray:
    palette_bgr = np.asarray(
        (
            (0, 0, 0),
            (0, 165, 255),
            (80, 220, 80),
            (220, 80, 80),
            (150, 150, 150),
        ),
        dtype=np.uint8,
    )
    return palette_bgr[hit_class]


def _masked_depth_color(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = _depth_color(depth, args_cli.min_depth, args_cli.max_depth)
    color[~mask] = 0
    return color


def _opencv_highgui_available() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", maxsplit=1)[1].strip().upper() != "NONE"
    return True


def _bgr_or_gray_to_rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        rgb = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 3:
        rgb = image[..., ::-1]
    else:
        raise ValueError(f"Expected BGR or grayscale image, got {image.shape}")
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate((rgb, alpha), axis=-1))


class _ImageWindows:
    def __init__(self) -> None:
        self._backend = "opencv" if _opencv_highgui_available() else "omni"
        self._windows = {}
        self._providers = {}
        self._omni_ui = None
        if self._backend == "omni":
            import omni.ui

            self._omni_ui = omni.ui
            print("[WARP CAMERA] OpenCV HighGUI unavailable; using Isaac Sim windows.", flush=True)

    def show(self, name: str, image: np.ndarray) -> bool:
        resized = cv2.resize(
            image,
            dsize=None,
            fx=args_cli.display_scale,
            fy=args_cli.display_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        if self._backend == "opencv":
            cv2.imshow(name, resized)
            key = cv2.waitKey(1) & 0xFF
            return key in (27, ord("q"), ord("Q"))

        rgba = _bgr_or_gray_to_rgba(resized)
        height, width = rgba.shape[:2]
        provider = self._providers.get(name)
        if provider is None:
            index = _WINDOW_NAMES.index(name)
            provider = self._omni_ui.ByteImageProvider()
            window = self._omni_ui.Window(
                name,
                width=width,
                height=height + 30,
                position_x=(index % 4) * width,
                position_y=10 + (index // 4) * (height + 30),
                visible=True,
            )
            with window.frame:
                self._omni_ui.ImageWithProvider(
                    provider,
                    fill_policy=self._omni_ui.IwpFillPolicy.IWP_STRETCH,
                )
            self._windows[name] = window
            self._providers[name] = provider
        provider.set_bytes_data(rgba.flatten().data, [width, height])
        return False

    def close(self) -> None:
        if self._backend == "opencv":
            cv2.destroyAllWindows()
            return
        for window in self._windows.values():
            window.visible = False
            window.destroy()
        self._windows.clear()
        self._providers.clear()


def _save_capture(
    images: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    metadata: dict,
    reference_frame: int,
) -> Path:
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{reference_frame:05d}"
    suffixes = {
        _WINDOW_NAMES[0]: "rtx_depth",
        _WINDOW_NAMES[1]: "warp_depth",
        _WINDOW_NAMES[2]: "depth_difference",
        _WINDOW_NAMES[3]: "warp_classes",
        _WINDOW_NAMES[4]: "warp_rung_mask",
        _WINDOW_NAMES[5]: "warp_robot_mask",
        _WINDOW_NAMES[6]: "warp_rfm_mask",
        _WINDOW_NAMES[7]: "warp_rfm_depth",
    }
    for name, image in images.items():
        cv2.imwrite(str(output_dir / f"{stem}_{suffixes[name]}.png"), image)
    for name, array in arrays.items():
        np.save(output_dir / f"{stem}_{name}.npy", array)
    metadata_path = output_dir / f"{stem}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    print(f"[WARP CAMERA] Saved comparison to {output_dir}", flush=True)
    return output_dir


def _warp_diagnostic_summary(output, ladder_object_names: list[str]) -> dict:
    if output.diagnostic_counts is None or output.center_ray_values is None:
        return {}
    counts = output.diagnostic_counts.detach().cpu().tolist()
    center_values = output.center_ray_values.detach().cpu().tolist()
    return {
        "counts": dict(zip(WARP_DIAGNOSTIC_COUNT_NAMES, counts)),
        "center_ray": dict(
            zip(WARP_CENTER_RAY_VALUE_NAMES, center_values[: len(WARP_CENTER_RAY_VALUE_NAMES)])
        ),
        "center_ray_ladder_box_distances_m": dict(
            zip(
                ladder_object_names,
                center_values[
                    len(WARP_CENTER_RAY_VALUE_NAMES) : len(WARP_CENTER_RAY_VALUE_NAMES)
                    + len(ladder_object_names)
                ],
            )
        ),
    }


def _warp_class_counts(output) -> dict[str, int]:
    hit_class = output.hit_class.detach().cpu().numpy()
    return {
        HIT_CLASS_NAMES[class_id]: int(np.count_nonzero(hit_class == class_id))
        for class_id in (BACKGROUND, RUNG, ROBOT, LADDER_OTHER, GROUND)
    }


def _project_robot_hands(
    robot,
    camera_position: torch.Tensor,
    camera_quaternion: torch.Tensor,
    cfg: WarpCameraCfg,
) -> dict[str, dict]:
    projections = {}
    for body_name in ("left_sphere_hand_link", "right_sphere_hand_link"):
        body_index = robot.body_names.index(body_name)
        world_position = robot.data.body_pos_w[0, body_index]
        camera_position_local = quat_apply_inverse(
            camera_quaternion.reshape(1, 4),
            (world_position - camera_position).reshape(1, 3),
        )[0]
        depth = float(-camera_position_local[2])
        if depth > 0.0:
            normalized_x = float(camera_position_local[0] / depth / cfg.horizontal_tangent)
            normalized_y = float(camera_position_local[1] / depth / cfg.vertical_tangent)
            pixel_column = (normalized_x + 1.0) * 0.5 * cfg.width - 0.5
            pixel_row = (1.0 - normalized_y) * 0.5 * cfg.height - 0.5
            center_in_view = (
                cfg.min_depth <= depth <= cfg.max_depth
                and abs(normalized_x) <= 1.0
                and abs(normalized_y) <= 1.0
            )
        else:
            normalized_x = normalized_y = pixel_column = pixel_row = None
            center_in_view = False
        projections[body_name] = {
            "world_position": world_position.detach().cpu().tolist(),
            "camera_position": camera_position_local.detach().cpu().tolist(),
            "image_plane_depth_m": depth,
            "normalized_xy": [normalized_x, normalized_y],
            "pixel_column_row": [pixel_column, pixel_row],
            "center_in_warp_view": center_in_view,
        }
    return projections


def _capture(unwrapped, cfg: WarpCameraCfg, renderer: CR1WarpCamera, reference_frame: int):
    camera = unwrapped.scene.sensors["depth_camera"]
    robot = unwrapped.scene.articulations["robot"]
    ladder = unwrapped.scene.rigid_object_collections["ladder"]
    camera_body_index = robot.body_names.index(_CAMERA_BODY_NAME)
    # TiledCameraCfg returns its initialization pose by default.  The diagnostic
    # disables extrinsic randomization, so derive the current sensor pose from
    # its attached D435 link to keep Warp synchronized with the moving camera.
    camera_position = robot.data.body_pos_w[0, camera_body_index]
    camera_quaternion = convert_camera_frame_orientation_convention(
        robot.data.body_quat_w[0, camera_body_index].reshape(1, 4),
        origin="world",
        target="opengl",
    )[0]
    rtx_depth_full = _depth_2d(camera.data.output["distance_to_image_plane"])
    intrinsics = camera.data.intrinsic_matrices[0].detach().cpu().numpy()
    remap_x, remap_y = pinhole_fov_remap_grid(
        intrinsics,
        output_width=cfg.width,
        output_height=cfg.height,
        horizontal_fov_deg=cfg.horizontal_fov_deg,
        vertical_fov_deg=cfg.vertical_fov_deg,
    )
    rtx_raw = rtx_depth_full.detach().cpu().numpy()
    rtx_native = cfg.crop(rtx_raw)
    rtx_aligned_full = cv2.remap(
        rtx_raw,
        remap_x,
        remap_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=cfg.max_depth,
    )
    rtx_aligned = cfg.crop(rtx_aligned_full)
    output = renderer.render(
        camera_position_w=camera_position,
        camera_quaternion_wxyz_opengl=camera_quaternion,
        robot_body_positions_w=robot.data.body_pos_w[0],
        robot_body_quaternions_wxyz=robot.data.body_quat_w[0],
        ladder_positions_w=ladder.data.object_pos_w[0],
        ladder_quaternions_wxyz=ladder.data.object_quat_w[0],
        ground_height=CR1_GROUND_Z,
        collect_diagnostics=args_cli.warp_debug,
    )
    warp_diagnostics = _warp_diagnostic_summary(output, list(ladder.object_names))
    warp_depth = output.depth.detach().cpu().numpy()
    hit_class = output.hit_class.detach().cpu().numpy()
    rung_mask = hit_class == RUNG
    robot_mask = hit_class == ROBOT
    rfm_mask = rung_mask | robot_mask
    difference = np.abs(rtx_aligned - warp_depth)
    images = {
        _WINDOW_NAMES[0]: _depth_color(rtx_native, cfg.min_depth, cfg.max_depth),
        _WINDOW_NAMES[1]: _depth_color(warp_depth, cfg.min_depth, cfg.max_depth),
        _WINDOW_NAMES[2]: _difference_color(difference),
        _WINDOW_NAMES[3]: _class_color(hit_class),
        _WINDOW_NAMES[4]: rung_mask.astype(np.uint8) * 255,
        _WINDOW_NAMES[5]: robot_mask.astype(np.uint8) * 255,
        _WINDOW_NAMES[6]: rfm_mask.astype(np.uint8) * 255,
        _WINDOW_NAMES[7]: _masked_depth_color(warp_depth, rfm_mask),
    }
    arrays = {
        "rtx_depth_raw_m": rtx_raw,
        "rtx_depth_m": rtx_native,
        "rtx_depth_warp_fov_m": rtx_aligned,
        "warp_depth_m": warp_depth,
        "depth_difference_m": difference,
        "warp_hit_class": hit_class,
        "ladder_positions_w": ladder.data.object_pos_w[0].detach().cpu().numpy(),
        "ladder_quaternions_wxyz": ladder.data.object_quat_w[0].detach().cpu().numpy(),
    }
    if args_cli.warp_debug:
        arrays["warp_diagnostic_counts"] = output.diagnostic_counts.detach().cpu().numpy().copy()
        arrays["warp_center_ray_values"] = output.center_ray_values.detach().cpu().numpy().copy()
    camera_forward = quat_apply(
        camera_quaternion.reshape(1, 4),
        torch.tensor([[0.0, 0.0, -1.0]], device=camera_position.device),
    )[0]
    rung_indexes = [
        index for index, name in enumerate(ladder.object_names) if name in WOOD_STEP_RUNG_BOX_NAMES
    ]
    rung_center = ladder.data.object_pos_w[0, rung_indexes].mean(dim=0)
    rung_offset = rung_center - camera_position
    rung_forward_projection = torch.dot(rung_offset, camera_forward)
    measured_horizontal_fov = math.degrees(2.0 * math.atan(cfg.width / (2.0 * intrinsics[0, 0])))
    measured_vertical_fov = math.degrees(2.0 * math.atan(cfg.height / (2.0 * intrinsics[1, 1])))
    center_column = cfg.width // 2
    center_row = cfg.height // 2
    center_screen_x = 2.0 * (center_column + 0.5) / cfg.width - 1.0
    center_screen_y = 1.0 - 2.0 * (center_row + 0.5) / cfg.height
    center_camera_ray = torch.tensor(
        [
            center_screen_x * cfg.horizontal_tangent,
            center_screen_y * cfg.vertical_tangent,
            -1.0,
        ],
        dtype=torch.float32,
        device=camera_position.device,
    )
    expected_center_world_ray = torch.nn.functional.normalize(
        quat_apply(camera_quaternion.reshape(1, 4), center_camera_ray.reshape(1, 3)), dim=-1
    )[0]
    metadata = {
        "reference_frame": reference_frame,
        "full_resolution": [cfg.width, cfg.height],
        "output_resolution": [cfg.output_width, cfg.output_height],
        "crop_top_bottom_left_right": [
            cfg.crop_top,
            cfg.crop_bottom,
            cfg.crop_left,
            cfg.crop_right,
        ],
        "requested_fov_deg": [cfg.horizontal_fov_deg, cfg.vertical_fov_deg],
        "rtx_intrinsic_fov_deg": [measured_horizontal_fov, measured_vertical_fov],
        "rtx_intrinsic_matrix": intrinsics.tolist(),
        "rtx_to_warp_remap_source_pixel_bounds": {
            "column": [float(remap_x.min()), float(remap_x.max())],
            "row": [float(remap_y.min()), float(remap_y.max())],
        },
        "depth_comparison": (
            "The RTX and Warp depth windows retain their native FOVs. Only the diagnostic "
            "difference remaps RTX depth to the Warp pinhole ray grid before subtraction."
        ),
        "camera_position_w": camera_position.detach().cpu().tolist(),
        "camera_quaternion_wxyz_opengl": camera_quaternion.detach().cpu().tolist(),
        "warp_camera_pose_source": "current robot d435_head_link pose",
        "camera_parent_position_w": robot.data.body_pos_w[0, camera_body_index].detach().cpu().tolist(),
        "camera_parent_quaternion_wxyz": robot.data.body_quat_w[0, camera_body_index]
        .detach()
        .cpu()
        .tolist(),
        "robot_hand_projections": _project_robot_hands(
            robot, camera_position, camera_quaternion, cfg
        ),
        "camera_forward_w": camera_forward.detach().cpu().tolist(),
        "mean_rung_center_w": rung_center.detach().cpu().tolist(),
        "mean_rung_forward_projection_m": float(rung_forward_projection),
        "ladder_object_names": list(ladder.object_names),
        "ladder_positions_w": ladder.data.object_pos_w[0].detach().cpu().tolist(),
        "ladder_quaternions_wxyz": ladder.data.object_quat_w[0].detach().cpu().tolist(),
        "depth_range_m": [cfg.min_depth, cfg.max_depth],
        "mean_absolute_depth_difference_m": float(difference.mean()),
        "p95_absolute_depth_difference_m": float(np.quantile(difference, 0.95)),
        "hit_class_names": {str(key): value for key, value in HIT_CLASS_NAMES.items()},
        "hit_pixel_fractions": {
            HIT_CLASS_NAMES[class_id]: float(np.mean(hit_class == class_id))
            for class_id in (BACKGROUND, RUNG, ROBOT, LADDER_OTHER, GROUND)
        },
    }
    if args_cli.warp_debug:
        metadata["warp_debug"] = {
            "software": {
                "device": str(unwrapped.device),
                "torch_version": torch.__version__,
                "warp_version": wp.__version__,
                "torch_cuda_stream": (
                    int(torch.cuda.current_stream(camera_position.device).cuda_stream)
                    if camera_position.device.type == "cuda"
                    else None
                ),
            },
            "torch_expected_center_ray": {
                "pixel_column_row": [center_column, center_row],
                "camera_ray": center_camera_ray.detach().cpu().tolist(),
                "world_ray": expected_center_world_ray.detach().cpu().tolist(),
                "camera_quaternion_xyzw": camera_quaternion[[1, 2, 3, 0]].detach().cpu().tolist(),
            },
            "full_render": {
                "class_counts": _warp_class_counts(output),
                **warp_diagnostics,
            },
            "isolation_passes": {},
        }
        for pass_name, include_robot, include_ladder, include_ground in (
            ("ladder_only", False, True, False),
            ("robot_only", True, False, False),
            ("ground_only", False, False, True),
        ):
            isolated_output = renderer.render(
                camera_position_w=camera_position,
                camera_quaternion_wxyz_opengl=camera_quaternion,
                robot_body_positions_w=robot.data.body_pos_w[0],
                robot_body_quaternions_wxyz=robot.data.body_quat_w[0],
                ladder_positions_w=ladder.data.object_pos_w[0],
                ladder_quaternions_wxyz=ladder.data.object_quat_w[0],
                ground_height=CR1_GROUND_Z,
                include_robot=include_robot,
                include_ladder=include_ladder,
                include_ground=include_ground,
                collect_diagnostics=True,
            )
            isolated_diagnostics = _warp_diagnostic_summary(
                isolated_output, list(ladder.object_names)
            )
            metadata["warp_debug"]["isolation_passes"][pass_name] = {
                "class_counts": _warp_class_counts(isolated_output),
                "depth_min_max_m": [
                    float(isolated_output.depth.min()),
                    float(isolated_output.depth.max()),
                ],
                "counts": isolated_diagnostics["counts"],
                "center_ray": isolated_diagnostics["center_ray"],
            }
        print("[WARP DEBUG] Kernel and isolation diagnostics:", flush=True)
        print(json.dumps(metadata["warp_debug"], indent=2, sort_keys=True), flush=True)
    return images, arrays, metadata


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
    # Pull the image rendered from the just-written robot state before Warp
    # consumes the matching current D435 link pose.
    camera = unwrapped.scene.sensors["depth_camera"]
    camera.reset()
    camera.update(0.0, force_recompute=True)


def _print_window_legend() -> None:
    descriptions = (
        "Isaac Sim RTX metric depth in its native square-pixel FOV",
        "custom Warp metric depth for robot, ladder, and ground",
        "diagnostic-only absolute difference after remapping RTX onto the Warp ray grid",
        "nearest Warp hit class (rung/robot/other ladder/ground)",
        "binary rung-only mask",
        "binary robot-only mask",
        "binary RFM mask: rungs plus robot",
        "Warp depth retained only where the RFM mask is true",
    )
    print("[WARP CAMERA] Eight diagnostic windows:", flush=True)
    for index, (name, description) in enumerate(zip(_WINDOW_NAMES, descriptions), start=1):
        print(f"  {index}. {name}: {description}", flush=True)


def _print_capture_summary(cfg: WarpCameraCfg, metadata: dict) -> None:
    print(
        "[WARP CAMERA] "
        f"output={cfg.output_width}x{cfg.output_height} "
        f"requested_fov=({cfg.horizontal_fov_deg:.3f}, {cfg.vertical_fov_deg:.3f})deg "
        f"rtx_fov=({metadata['rtx_intrinsic_fov_deg'][0]:.3f}, "
        f"{metadata['rtx_intrinsic_fov_deg'][1]:.3f})deg",
        flush=True,
    )
    print(
        "[WARP CAMERA] "
        f"mean_abs_difference={metadata['mean_absolute_depth_difference_m']:.6f}m "
        f"p95_abs_difference={metadata['p95_absolute_depth_difference_m']:.6f}m "
        f"rung_forward_projection={metadata['mean_rung_forward_projection_m']:.6f}m "
        f"class_fractions={metadata['hit_pixel_fractions']}",
        flush=True,
    )


def main() -> None:
    motion_file = args_cli.motion_file.expanduser().resolve()
    if not motion_file.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    cfg = WarpCameraCfg(
        width=args_cli.camera_width,
        height=args_cli.camera_height,
        horizontal_fov_deg=args_cli.horizontal_fov_deg,
        vertical_fov_deg=args_cli.vertical_fov_deg,
        min_depth=args_cli.min_depth,
        max_depth=args_cli.max_depth,
        crop_top=args_cli.crop_top,
        crop_bottom=args_cli.crop_bottom,
        crop_left=args_cli.crop_left,
        crop_right=args_cli.crop_right,
    )

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_cfg.commands.motion.fixed_reference_frame = args_cli.reference_frame
    env_cfg.commands.motion.advance_reference = False
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    _disable_randomization(env_cfg)
    _configure_rtx_camera(env_cfg, cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    windows = None
    try:
        unwrapped = env.unwrapped
        env.reset()
        robot = unwrapped.scene.articulations["robot"]
        ladder = unwrapped.scene.rigid_object_collections["ladder"]
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
        renderer = CR1WarpCamera(
            cfg,
            robot_urdf=CR1_URDF,
            robot_body_names=robot.body_names,
            ladder_box_specs=WOOD_STEP_BOX_SPECS,
            ladder_object_names=ladder.object_names,
            rung_box_names=WOOD_STEP_RUNG_BOX_NAMES,
            device=unwrapped.device,
        )
        _print_window_legend()

        if args_cli.play:
            if not args_cli.no_show and not args_cli.headless:
                windows = _ImageWindows()
            print(
                f"[WARP CAMERA] Playing reference frames {args_cli.reference_frame}..{end_frame} "
                f"at {playback_fps:g} fps (loop={args_cli.loop}).",
                flush=True,
            )
            frame = args_cli.reference_frame
            update_count = 0
            quit_requested = False
            next_frame_time = time.perf_counter()
            while simulation_app.is_running() and not quit_requested:
                _write_reference_frame(unwrapped, frame)
                images, arrays, metadata = _capture(unwrapped, cfg, renderer, frame)
                if update_count == 0:
                    _save_capture(images, arrays, metadata, frame)
                    _print_capture_summary(cfg, metadata)
                if windows is not None:
                    for name in _WINDOW_NAMES:
                        quit_requested = windows.show(name, images[name]) or quit_requested
                print(f"\r[WARP CAMERA] frame={frame}/{end_frame}", end="", flush=True)

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
            return

        _write_reference_frame(unwrapped, args_cli.reference_frame)
        images, arrays, metadata = _capture(
            unwrapped, cfg, renderer, args_cli.reference_frame
        )
        _save_capture(images, arrays, metadata, args_cli.reference_frame)
        _print_capture_summary(cfg, metadata)

        if args_cli.no_show or args_cli.headless:
            return
        windows = _ImageWindows()
        quit_requested = False
        for name in _WINDOW_NAMES:
            quit_requested = windows.show(name, images[name]) or quit_requested
        print(
            "[WARP CAMERA] Static reference-frame comparison is ready. "
            "Close Isaac Sim, or press q/Esc with OpenCV windows, to exit.",
            flush=True,
        )
        update_count = 0
        while simulation_app.is_running() and not quit_requested:
            simulation_app.update()
            update_count += 1
            if args_cli.smoke_steps and update_count >= args_cli.smoke_steps:
                break
            if windows._backend == "opencv":
                key = cv2.waitKey(1) & 0xFF
                quit_requested = key in (27, ord("q"), ord("Q"))
    finally:
        if windows is not None:
            windows.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
