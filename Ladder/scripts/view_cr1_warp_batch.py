"""Validate batched Warp depth rendering on CR1 geometry-randomized ladders.

This is a renderer acceptance tool, not a training entry point. It kinematically
replays the reference motion in many IsaacLab environments and renders native
Warp depth/classes in one launch per frame. Image display and capture are kept
outside the timed renderer path.
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "cr1_warp_batch"
_CAMERA_BODY_NAME = "d435_head_link"
_WINDOW_NAMES = (
    "Warp batch depth (m)",
    "Warp batch hit classes",
    "Warp batch RFM mask (rungs+robot)",
    "Warp batch RFM depth (rungs+robot)",
)


parser = argparse.ArgumentParser(description="Validate batched CR1 Warp depth rendering.")
parser.add_argument(
    "--task",
    type=str,
    default="Ladder-Climb-CR1-Geometry-DR-v0",
    help="Registered nonvisual geometry-randomized task used to create the scene.",
)
parser.add_argument("--motion_file", type=Path, default=DEFAULT_MOTION, help="Motion npz.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments.")
parser.add_argument("--reference_frame", type=int, default=0, help="First replay frame.")
parser.add_argument("--end_frame", type=int, default=None, help="Inclusive final replay frame.")
parser.add_argument("--playback_fps", type=float, default=None, help="Replay rate; defaults to motion FPS.")
parser.add_argument("--loop", action="store_true", help="Loop the selected reference-frame range.")
parser.add_argument("--display_env", type=int, default=0, help="Environment index displayed in image windows.")
parser.add_argument("--display_interval", type=int, default=5, help="Display/capture every N replay frames.")
parser.add_argument("--log_interval", type=int, default=25, help="Print batch statistics every N frames.")
parser.add_argument("--warmup_frames", type=int, default=5, help="Untimed Warp frames before statistics.")
parser.add_argument("--camera_width", type=int, default=80, help="Warp depth width.")
parser.add_argument("--camera_height", type=int, default=60, help="Warp depth height.")
parser.add_argument("--horizontal_fov_deg", type=float, default=87.0, help="Warp horizontal FOV.")
parser.add_argument("--vertical_fov_deg", type=float, default=58.0, help="Warp vertical FOV.")
parser.add_argument("--min_depth", type=float, default=0.1, help="Near depth in meters.")
parser.add_argument("--max_depth", type=float, default=2.0, help="Far depth in meters.")
parser.add_argument("--display_scale", type=float, default=5.0, help="Nearest-neighbor image enlargement.")
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT, help="Capture output directory.")
parser.add_argument("--no_show", action="store_true", help="Do not create image windows.")
parser.add_argument(
    "--disable_ladder_randomization",
    action="store_true",
    help="Use the nominal ladder in every environment instead of per-environment geometry randomization.",
)
parser.add_argument("--smoke_steps", type=int, default=0, help="Close after N replay frames; zero is unlimited.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.reference_frame < 0:
    parser.error("--reference_frame must be non-negative")
if args_cli.display_env < 0 or args_cli.display_env >= args_cli.num_envs:
    parser.error("--display_env must be in [0, --num_envs)")
if args_cli.display_interval <= 0 or args_cli.log_interval <= 0:
    parser.error("--display_interval and --log_interval must be positive")
if args_cli.warmup_frames < 0:
    parser.error("--warmup_frames must be non-negative")
if args_cli.display_scale <= 0.0:
    parser.error("--display_scale must be positive")
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

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import convert_camera_frame_orientation_convention

import ladder.tasks  # noqa: F401
from ladder.robots.cr1 import CR1_URDF
from ladder.tasks.ladder_climb.config.cr1.wood_step_env_cfg import CR1_GROUND_Z
from ladder.tasks.ladder_climb.config.cr1.wood_step_geometry_dr_env_cfg import WOOD_STEP_BOX_SPECS
from ladder.utils.warp_camera import (
    HIT_CLASS_NAMES,
    ROBOT,
    RUNG,
    CR1WarpBatchCamera,
    WarpCameraCfg,
)
from ladder.utils.wood_step_ladder import WOOD_STEP_RUNG_BOX_NAMES


def _disable_nonvisual_randomization(env_cfg) -> None:
    """Freeze robot/reset noise while retaining ladder geometry randomization."""

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
    if args_cli.disable_ladder_randomization:
        env_cfg.commands.motion.ladder_inclination_delta_range = (0.0, 0.0)
        env_cfg.commands.motion.ladder_spacing_delta_range = (0.0, 0.0)
        env_cfg.commands.motion.nominal_ladder_geometry_probability = 1.0


def _depth_color(depth: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    normalized = np.round(
        (np.clip(depth, minimum, maximum) - minimum) / (maximum - minimum) * 255.0
    ).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


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


def _bgr_or_gray_to_rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        rgb = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 3:
        rgb = image[..., ::-1]
    else:
        raise ValueError(f"Expected grayscale or BGR image, got {image.shape}")
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate((rgb, alpha), axis=-1))


def _opencv_highgui_available() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", maxsplit=1)[1].strip().upper() != "NONE"
    return True


class _ImageWindows:
    def __init__(self) -> None:
        self._backend = "opencv" if _opencv_highgui_available() else "omni"
        self._windows = {}
        self._providers = {}
        self._omni_ui = None
        if self._backend == "omni":
            import omni.ui

            self._omni_ui = omni.ui
            print("[WARP BATCH] OpenCV HighGUI unavailable; using Isaac Sim windows.", flush=True)

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
            return (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q"))

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
                position_x=(index % 2) * width,
                position_y=10 + (index // 2) * (height + 30),
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


def _current_camera_pose(robot) -> tuple[torch.Tensor, torch.Tensor]:
    body_index = robot.body_names.index(_CAMERA_BODY_NAME)
    positions = robot.data.body_pos_w[:, body_index].contiguous()
    quaternions_opengl = convert_camera_frame_orientation_convention(
        robot.data.body_quat_w[:, body_index], origin="world", target="opengl"
    ).contiguous()
    return positions, quaternions_opengl


def _render_batch(unwrapped, renderer: CR1WarpBatchCamera):
    robot = unwrapped.scene.articulations["robot"]
    ladder = unwrapped.scene.rigid_object_collections["ladder"]
    camera_positions, camera_quaternions = _current_camera_pose(robot)
    if torch.cuda.is_available() and str(unwrapped.device).startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        start.record()
        output = renderer.render(
            camera_positions_w=camera_positions,
            camera_quaternions_wxyz_opengl=camera_quaternions,
            robot_body_positions_w=robot.data.body_pos_w,
            robot_body_quaternions_wxyz=robot.data.body_quat_w,
            ladder_positions_w=ladder.data.object_pos_w,
            ladder_quaternions_wxyz=ladder.data.object_quat_w,
            ground_height=CR1_GROUND_Z,
        )
        finish.record()
        finish.synchronize()
        elapsed_ms = start.elapsed_time(finish)
    else:
        start_time = time.perf_counter()
        output = renderer.render(
            camera_positions_w=camera_positions,
            camera_quaternions_wxyz_opengl=camera_quaternions,
            robot_body_positions_w=robot.data.body_pos_w,
            robot_body_quaternions_wxyz=robot.data.body_quat_w,
            ladder_positions_w=ladder.data.object_pos_w,
            ladder_quaternions_wxyz=ladder.data.object_quat_w,
            ground_height=CR1_GROUND_Z,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return output, elapsed_ms, camera_positions, camera_quaternions


def _class_statistics(hit_class: torch.Tensor) -> dict[str, float]:
    counts = torch.bincount(hit_class.reshape(-1), minlength=len(HIT_CLASS_NAMES)).cpu().tolist()
    total = max(sum(counts), 1)
    return {HIT_CLASS_NAMES[index]: count / total for index, count in enumerate(counts)}


def _images_for_display(output) -> dict[str, np.ndarray]:
    hit_class = output.hit_class[args_cli.display_env].detach().cpu().numpy()
    depth = output.depth[args_cli.display_env].detach().cpu().numpy()
    rfm_mask = (hit_class == RUNG) | (hit_class == ROBOT)
    rfm_depth = _depth_color(depth, args_cli.min_depth, args_cli.max_depth)
    rfm_depth[~rfm_mask] = 0
    return {
        _WINDOW_NAMES[0]: _depth_color(depth, args_cli.min_depth, args_cli.max_depth),
        _WINDOW_NAMES[1]: _class_color(hit_class),
        _WINDOW_NAMES[2]: rfm_mask.astype(np.uint8) * 255,
        _WINDOW_NAMES[3]: rfm_depth,
    }


def _save_capture(output, images: dict[str, np.ndarray], metadata: dict, reference_frame: int) -> None:
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{reference_frame:05d}_env_{args_cli.display_env:04d}"
    suffixes = ("depth", "classes", "rfm_mask", "rfm_depth")
    for name, suffix in zip(_WINDOW_NAMES, suffixes):
        cv2.imwrite(str(output_dir / f"{stem}_{suffix}.png"), images[name])
    np.save(output_dir / f"{stem}_depth_m.npy", output.depth[args_cli.display_env].detach().cpu().numpy())
    np.save(output_dir / f"{stem}_hit_class.npy", output.hit_class[args_cli.display_env].detach().cpu().numpy())
    with (output_dir / f"{stem}_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    print(f"[WARP BATCH] Saved env-{args_cli.display_env} capture to {output_dir}", flush=True)


def main() -> None:
    motion_file = args_cli.motion_file.expanduser().resolve()
    if not motion_file.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    camera_cfg = WarpCameraCfg(
        width=args_cli.camera_width,
        height=args_cli.camera_height,
        horizontal_fov_deg=args_cli.horizontal_fov_deg,
        vertical_fov_deg=args_cli.vertical_fov_deg,
        min_depth=args_cli.min_depth,
        max_depth=args_cli.max_depth,
    )
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_cfg.commands.motion.fixed_reference_frame = args_cli.reference_frame
    env_cfg.commands.motion.advance_reference = False
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    _disable_nonvisual_randomization(env_cfg)

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
        renderer = CR1WarpBatchCamera(
            camera_cfg,
            num_envs=args_cli.num_envs,
            robot_urdf=CR1_URDF,
            robot_body_names=robot.body_names,
            ladder_box_specs=WOOD_STEP_BOX_SPECS,
            ladder_object_names=ladder.object_names,
            rung_box_names=WOOD_STEP_RUNG_BOX_NAMES,
            device=unwrapped.device,
        )
        if not args_cli.no_show and not args_cli.headless:
            windows = _ImageWindows()
        print(
            "[WARP BATCH] "
            f"envs={args_cli.num_envs} image={camera_cfg.output_width}x{camera_cfg.output_height} "
            f"fov=({camera_cfg.horizontal_fov_deg:g}, {camera_cfg.vertical_fov_deg:g})deg "
            f"ladder_randomization={not args_cli.disable_ladder_randomization}",
            flush=True,
        )
        print(
            f"[WARP BATCH] Replaying frames {args_cli.reference_frame}..{end_frame} at "
            f"{playback_fps:g} fps (loop={args_cli.loop}). Press q/Esc in OpenCV windows to quit.",
            flush=True,
        )

        _write_reference_frame(unwrapped, args_cli.reference_frame)
        for _ in range(args_cli.warmup_frames):
            _render_batch(unwrapped, renderer)
        print(
            f"[WARP BATCH] Completed {args_cli.warmup_frames} untimed renderer warmup frames.",
            flush=True,
        )

        frame = args_cli.reference_frame
        update_count = 0
        quit_requested = False
        next_frame_time = time.perf_counter()
        while simulation_app.is_running() and not quit_requested:
            _write_reference_frame(unwrapped, frame)
            output, elapsed_ms, camera_positions, camera_quaternions = _render_batch(unwrapped, renderer)
            if update_count % args_cli.log_interval == 0:
                statistics = _class_statistics(output.hit_class)
                rays = args_cli.num_envs * camera_cfg.width * camera_cfg.height
                print(
                    "[WARP BATCH] "
                    f"frame={frame} render_ms={elapsed_ms:.3f} "
                    f"Mrays/s={rays / max(elapsed_ms, 1.0e-6) / 1000.0:.3f} "
                    f"class_fractions={statistics}",
                    flush=True,
                )
            if update_count % args_cli.display_interval == 0:
                images = _images_for_display(output)
                metadata = {
                    "reference_frame": frame,
                    "num_envs": args_cli.num_envs,
                    "display_env": args_cli.display_env,
                    "render_ms": elapsed_ms,
                    "camera_position_w": camera_positions[args_cli.display_env].detach().cpu().tolist(),
                    "camera_quaternion_wxyz_opengl": camera_quaternions[
                        args_cli.display_env
                    ].detach().cpu().tolist(),
                    "class_fractions": _class_statistics(output.hit_class),
                    "warp_camera_pose_source": "current robot d435_head_link pose",
                    "ladder_positions_w": ladder.data.object_pos_w[
                        args_cli.display_env
                    ].detach().cpu().tolist(),
                    "ladder_quaternions_wxyz": ladder.data.object_quat_w[
                        args_cli.display_env
                    ].detach().cpu().tolist(),
                }
                if update_count == 0:
                    _save_capture(output, images, metadata, frame)
                if windows is not None:
                    for name in _WINDOW_NAMES:
                        quit_requested = windows.show(name, images[name]) or quit_requested
            print(f"\r[WARP BATCH] frame={frame}/{end_frame}", end="", flush=True)

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
        if windows is not None:
            windows.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
