"""Inspect a moving RGB/depth camera view of the LadderMan ladder.

This is intentionally a standalone sensor diagnostic.  It does not create a
Gym task, policy, observation term, or training configuration.

Example::

    ./isaaclab.sh -p Ladder/scripts/view_ladder_camera.py --enable_cameras

The camera follows the ladder's climbing direction from the lower rungs to the
top and then returns to the lower end.  OpenCV shows the RGB and metric depth
images in separate windows when a GUI is available; periodic captures are
written to ``outputs/ladder_camera_test`` by default.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LADDER = REPO_ROOT / "assets" / "ladders" / "ladder_A_aframe" / "wood_step_ladder_physics.usda"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "ladder_camera_test"


parser = argparse.ArgumentParser(description="Show a moving RGB and depth view of one LadderMan ladder.")
parser.add_argument("--ladder_asset", type=Path, default=DEFAULT_LADDER, help="USD/USDA ladder asset to load.")
parser.add_argument("--ladder_x", type=float, default=0.0, help="Ladder X translation in meters.")
parser.add_argument("--ladder_y", type=float, default=0.0, help="Ladder Y translation in meters.")
parser.add_argument("--ladder_z", type=float, default=0.0, help="Ladder Z translation in meters.")
parser.add_argument("--ladder_yaw_deg", type=float, default=0.0, help="Ladder world yaw in degrees.")
parser.add_argument("--width", type=int, default=640, help="Camera image width in pixels.")
parser.add_argument("--height", type=int, default=480, help="Camera image height in pixels.")
parser.add_argument("--speed", type=float, default=0.20, help="Camera travel speed along the ladder in m/s.")
parser.add_argument(
    "--side_offset",
    type=float,
    default=0.90,
    help="Camera offset to the side of the ladder in meters.",
)
parser.add_argument(
    "--front_offset",
    type=float,
    default=1.05,
    help="Camera offset toward the front of the ladder in meters.",
)
parser.add_argument(
    "--above_offset",
    type=float,
    default=0.75,
    help="Camera height above the tracked ladder point in meters.",
)
parser.add_argument(
    "--depth_max",
    type=float,
    default=3.0,
    help="Far clipping and display range for metric depth in meters.",
)
parser.add_argument(
    "--save_interval_steps",
    type=int,
    default=200,
    help="Save a paired RGB/depth capture every N simulation steps; 0 disables saving.",
)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for diagnostic captures.")
parser.add_argument("--no_show", action="store_true", help="Do not open OpenCV windows; still run and save captures.")
parser.add_argument("--smoke_steps", type=int, default=0, help="Run this many simulation steps and exit; 0 keeps running.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.width <= 0 or args_cli.height <= 0:
    parser.error("--width and --height must be positive")
if args_cli.speed <= 0.0:
    parser.error("--speed must be positive")
if args_cli.depth_max <= 0.1:
    parser.error("--depth_max must be greater than 0.1 m")
if args_cli.save_interval_steps < 0 or args_cli.smoke_steps < 0:
    parser.error("--save_interval_steps and --smoke_steps must be non-negative")

# Camera sensors are disabled by default in Isaac Lab, including for a GUI run.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationContext


def check_ladder_asset(path: Path) -> Path:
    """Resolve and validate the ladder asset before Isaac Sim starts stepping."""

    asset_path = path.expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Ladder asset not found: {asset_path}")
    if asset_path.suffix.lower() not in {".usd", ".usda"}:
        raise ValueError(f"Expected a USD/USDA ladder asset, got: {asset_path}")
    return asset_path


def yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    """Return a world-Z yaw quaternion in Isaac Lab's wxyz convention."""

    half_yaw = 0.5 * math.radians(yaw_deg)
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def rotate_ladder_xy(point: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a point from ladder-local coordinates into the world XY plane."""

    yaw = math.radians(yaw_deg)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotated = point.copy()
    rotated[0] = cos_yaw * point[0] - sin_yaw * point[1]
    rotated[1] = sin_yaw * point[0] + cos_yaw * point[1]
    return rotated


def design_scene() -> TiledCamera:
    """Spawn a ground plane, light, ladder, and one RGB/depth TiledCamera."""

    ladder_asset = check_ladder_asset(args_cli.ladder_asset)

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    ladder_cfg = sim_utils.UsdFileCfg(usd_path=str(ladder_asset))
    ladder_cfg.func(
        "/World/Ladder",
        ladder_cfg,
        translation=(args_cli.ladder_x, args_cli.ladder_y, args_cli.ladder_z),
        orientation=yaw_quat_wxyz(args_cli.ladder_yaw_deg),
    )

    camera_cfg = TiledCameraCfg(
        prim_path="/World/LadderCamera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, args_cli.depth_max),
        ),
        depth_clipping_behavior="max",
    )
    camera = TiledCamera(cfg=camera_cfg)

    print(f"[INFO] Ladder asset: {ladder_asset}")
    print(f"[INFO] Camera output: {args_cli.width} x {args_cli.height}, metric depth in meters")
    return camera


def ladder_camera_pose(progress: float) -> tuple[np.ndarray, np.ndarray]:
    """Return an above-and-diagonal eye/target pair along the ladder.

    The local rung centers below come from the first and seventh rungs in the
    checked-in wood-step ladder asset.  The camera follows this span and tracks
    slightly upward so the upcoming rungs stay visible.
    """

    lower = np.asarray((0.0, -0.065, 0.175), dtype=np.float32)
    upper = np.asarray((0.0, -0.875, 1.555), dtype=np.float32)
    ladder_point = lower + float(progress) * (upper - lower)

    # +X is a lateral view of the rungs and +Y is in front of the ladder face.
    eye_local = ladder_point + np.asarray(
        (args_cli.side_offset, args_cli.front_offset, args_cli.above_offset), dtype=np.float32
    )
    target_local = ladder_point + 0.15 * (upper - lower)

    translation = np.asarray((args_cli.ladder_x, args_cli.ladder_y, args_cli.ladder_z), dtype=np.float32)
    eye = rotate_ladder_xy(eye_local, args_cli.ladder_yaw_deg) + translation
    target = rotate_ladder_xy(target_local, args_cli.ladder_yaw_deg) + translation
    return eye, target


def camera_progress(sim_time: float, span_length: float) -> float:
    """Move from bottom to top and back, at the requested metric speed."""

    travel = (sim_time * args_cli.speed) / span_length
    phase = travel % 2.0
    return phase if phase <= 1.0 else 2.0 - phase


def as_numpy(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a camera tensor to a CPU NumPy array without changing its values."""

    if isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy()
    return np.asarray(image)


def rgb_frame(camera: TiledCamera) -> np.ndarray:
    """Return the first RGB frame as uint8 RGB with no alpha channel."""

    image = as_numpy(camera.data.output["rgb"])[0]
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    return image


def depth_frame(camera: TiledCamera) -> np.ndarray:
    """Return the first distance-to-image-plane frame in meters as HxW."""

    depth = as_numpy(camera.data.output["distance_to_image_plane"])[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return np.nan_to_num(depth.astype(np.float32), nan=args_cli.depth_max, posinf=args_cli.depth_max, neginf=0.05)


def depth_visualization(depth: np.ndarray) -> np.ndarray:
    """Colorize metric depth using one fixed range so motion is comparable."""

    clipped = np.clip(depth, 0.05, args_cli.depth_max)
    normalized = ((clipped - 0.05) / (args_cli.depth_max - 0.05) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)


def save_capture(output_dir: Path, step: int, rgb: np.ndarray, depth: np.ndarray) -> None:
    """Save RGB, colorized depth, and raw metric depth for later inspection."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"rgb_{step:07d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"depth_{step:07d}.png"), depth_visualization(depth))
    np.save(output_dir / f"depth_{step:07d}.npy", depth)


def run_simulator(sim: SimulationContext, camera: TiledCamera) -> None:
    """Step physics, move the camera, and display/save its two outputs."""

    sim_dt = sim.get_physics_dt()
    span_length = float(np.linalg.norm(np.asarray((0.0, -0.875, 1.555)) - np.asarray((0.0, -0.065, 0.175))))
    show_windows = not args_cli.no_show and not args_cli.headless
    if show_windows:
        cv2.namedWindow("Ladder RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Ladder depth (m)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ladder RGB", args_cli.width, args_cli.height)
        cv2.resizeWindow("Ladder depth (m)", args_cli.width, args_cli.height)

    sim_time = 0.0
    step = 0
    printed_shape = False
    try:
        while simulation_app.is_running():
            progress = camera_progress(sim_time, span_length)
            eye, target = ladder_camera_pose(progress)
            camera.set_world_poses_from_view(
                torch.as_tensor(eye, device=sim.device).view(1, 3),
                torch.as_tensor(target, device=sim.device).view(1, 3),
            )

            sim.step()
            camera.update(sim_dt)
            sim_time += sim_dt
            step += 1

            if "rgb" not in camera.data.output or "distance_to_image_plane" not in camera.data.output:
                continue
            rgb = rgb_frame(camera)
            depth = depth_frame(camera)
            if not printed_shape:
                print(
                    f"[INFO] rgb_shape={rgb.shape} depth_shape={depth.shape} "
                    f"depth_range=({float(depth.min()):.4f}, {float(depth.max()):.4f}) m"
                )
                printed_shape = True

            if show_windows:
                cv2.imshow("Ladder RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow("Ladder depth (m)", depth_visualization(depth))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("[INFO] OpenCV quit key received.")
                    break

            if args_cli.save_interval_steps and step % args_cli.save_interval_steps == 0:
                save_capture(args_cli.output_dir.expanduser().resolve(), step, rgb, depth)
                print(f"[INFO] Saved RGB/depth capture at step {step} to {args_cli.output_dir}")

            if args_cli.smoke_steps and step >= args_cli.smoke_steps:
                print(f"[INFO] Smoke test completed for {step} steps.")
                break
    finally:
        if show_windows:
            cv2.destroyAllWindows()


def main() -> None:
    """Create the standalone scene and run the camera diagnostic."""

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.2, 3.0, 2.5], target=[0.0, -0.55, 0.9])
    camera = design_scene()
    sim.reset()
    print("[INFO] Scene setup complete. Press q or Esc in either image window to stop.")
    run_simulator(sim, camera)


if __name__ == "__main__":
    main()
    simulation_app.close()
