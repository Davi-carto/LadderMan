"""Capture and display CR1 LadderMan depth observation variants."""

from __future__ import annotations

import argparse
import os
import sys
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
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "cr1_visual_observations.png"


parser = argparse.ArgumentParser(description="Display raw, noisy, and rung-focused CR1 depth images.")
parser.add_argument(
    "--task",
    type=str,
    default="Ladder-Climb-CR1-Geometry-DR-Visual-v0",
    help="Registered visual Ladder task.",
)
parser.add_argument("--motion_file", type=Path, default=DEFAULT_MOTION, help="Geometry-DR motion npz.")
parser.add_argument("--reference_frame", type=int, default=0, help="Reference frame used for capture.")
parser.add_argument("--warmup_steps", type=int, default=3, help="Zero-action steps before capture.")
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output comparison PNG.")
parser.add_argument("--no_show", action="store_true", help="Save without opening a Matplotlib window.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ladder.tasks  # noqa: F401
from ladder.utils.visual_depth import (
    add_depth_noise_and_dropout,
    apply_rung_focused_masking,
    prepare_metric_depth,
    rfm_keep_mask_from_semantic,
)


def _disable_randomization(env_cfg) -> None:
    for event_name in (
        "robot_material",
        "robot_mass",
        "joint_position_offset",
        "torso_com",
        "camera_extrinsics",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    env_cfg.commands.motion.pose_range = {
        name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.commands.motion.fixed_start_pose_range = {}
    env_cfg.commands.motion.velocity_range = {
        name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.commands.motion.ladder_inclination_delta_range = (0.0, 0.0)
    env_cfg.commands.motion.ladder_spacing_delta_range = (0.0, 0.0)


def _capture(unwrapped) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    camera = unwrapped.scene.sensors["depth_camera"]
    raw = prepare_metric_depth(camera.data.output["distance_to_image_plane"])
    rfm_keep_mask = rfm_keep_mask_from_semantic(
        camera.data.output["semantic_segmentation"],
        camera.data.info.get("semantic_segmentation"),
    )
    noisy = add_depth_noise_and_dropout(raw)
    rung_focused, applied = apply_rung_focused_masking(raw, rfm_keep_mask, probability=1.0)
    if not bool(applied.all()):
        raise RuntimeError("Forced rung-focused masking was not applied")
    return raw, noisy, rfm_keep_mask, rung_focused


def _display(
    raw: torch.Tensor,
    training_depth: torch.Tensor,
    noisy: torch.Tensor,
    rfm_keep_mask: torch.Tensor,
    rung_focused: torch.Tensor,
    output: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ladderman_matplotlib")
    import matplotlib.pyplot as plt

    images = [
        (raw[0, 0].cpu().numpy(), "Raw simulated depth", "viridis_r", 0.1, 2.0),
        (
            training_depth[0, 0].cpu().numpy(),
            "Actual visual_depth observation",
            "viridis_r",
            0.0,
            2.0,
        ),
        (noisy[0, 0].cpu().numpy(), "Gaussian noise + 5% dropout", "viridis_r", 0.1, 2.0),
        (
            rfm_keep_mask[0, 0].float().cpu().numpy(),
            "RFM keep mask (rungs+robot)",
            "gray",
            0.0,
            1.0,
        ),
        (rung_focused[0, 0].cpu().numpy(), "Forced RFM preview", "viridis_r", 0.1, 2.0),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (image, title, cmap, vmin, vmax) in zip(axes.flat, images):
        artist = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat[len(images) :]:
        axis.set_axis_off()

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(f"[VISUAL] Saved comparison image: {output}")
    if not args_cli.no_show and not args_cli.headless:
        plt.show()
    else:
        plt.close(figure)


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
    _disable_randomization(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        observations, _ = env.reset()
        actions = torch.zeros(
            (1, unwrapped.action_manager.total_action_dim),
            dtype=torch.float32,
            device=unwrapped.device,
        )
        for _ in range(args_cli.warmup_steps):
            observations, _, _, _, _ = env.step(actions)

        try:
            training_depth = observations["visual_depth"]
        except (KeyError, TypeError, IndexError) as error:
            raise RuntimeError(
                "Visual environment did not return the visual_depth observation group"
            ) from error
        expected_shape = (1, 1, 60, 80)
        if tuple(training_depth.shape) != expected_shape:
            raise RuntimeError(
                f"Expected visual_depth shape {expected_shape}, got {tuple(training_depth.shape)}"
            )
        if not bool(torch.isfinite(training_depth).all()):
            raise RuntimeError("visual_depth contains non-finite values")

        raw, noisy, rfm_keep_mask, rung_focused = _capture(unwrapped)
        dropout_fraction = float((noisy == 0.0).float().mean())
        rfm_keep_fraction = float(rfm_keep_mask.float().mean())
        training_zero_fraction = float((training_depth == 0.0).float().mean())
        print(f"[VISUAL] raw_shape={tuple(raw.shape)} range=({float(raw.min()):.4f}, {float(raw.max()):.4f})m")
        print(
            f"[VISUAL] training_shape={tuple(training_depth.shape)} "
            f"range=({float(training_depth.min()):.4f}, {float(training_depth.max()):.4f})m "
            f"zero_fraction={training_zero_fraction:.4f}"
        )
        print(
            f"[VISUAL] dropout_fraction={dropout_fraction:.4f} "
            f"rfm_keep_pixel_fraction={rfm_keep_fraction:.4f}"
        )
        if rfm_keep_fraction == 0.0:
            raise RuntimeError("RFM keep mask is empty; check rung and robot semantic tags")
        _display(raw, training_depth, noisy, rfm_keep_mask, rung_focused, args_cli.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
