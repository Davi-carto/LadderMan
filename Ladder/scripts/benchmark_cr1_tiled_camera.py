"""Benchmark the current RTX/TiledCamera visual-observation training path.

This script deliberately measures a complete ManagerBasedRLEnv control step:
physics, RTX rendering, camera update, and the existing RFM/noise depth
observation term.  It is the baseline for deciding whether the native Warp
camera should replace ``TiledCamera`` in the training task.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


parser = argparse.ArgumentParser(
    description="Benchmark CR1's existing RTX TiledCamera visual-observation path."
)
parser.add_argument(
    "--task",
    type=str,
    default="Ladder-Climb-CR1-Geometry-DR-Visual-v0",
    help="Visual task whose TiledCamera and observation processing are benchmarked.",
)
parser.add_argument("--num_envs", type=int, default=256, help="Parallel environment count.")
parser.add_argument("--warmup_steps", type=int, default=20, help="Untimed complete environment steps.")
parser.add_argument("--steps", type=int, default=300, help="Timed complete environment steps.")
parser.add_argument("--log_interval", type=int, default=25, help="Print every N timed steps.")
parser.add_argument(
    "--freeze_nonvisual_randomization",
    action="store_true",
    help="Freeze robot/reset perturbations while retaining per-environment ladder geometry randomization.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative")
if args_cli.steps <= 0:
    parser.error("--steps must be positive")
if args_cli.log_interval <= 0:
    parser.error("--log_interval must be positive")

# TiledCamera is an RTX sensor.  Keep the benchmark self-contained so its
# result cannot silently become a non-rendering physics-only measurement.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ladder.tasks  # noqa: F401


def _freeze_nonvisual_randomization(env_cfg) -> None:
    """Match the Warp acceptance setup while retaining ladder geometry DR."""

    for event_name in (
        "robot_material",
        "robot_mass",
        "joint_position_offset",
        "torso_com",
        "camera_extrinsics",
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


def _timed_step(env, actions: torch.Tensor) -> float:
    """Time an end-to-end environment step, including visual observations."""

    if torch.cuda.is_available() and str(env.unwrapped.device).startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        start.record()
        env.step(actions)
        finish.record()
        finish.synchronize()
        return start.elapsed_time(finish)

    start_time = time.perf_counter()
    env.step(actions)
    return (time.perf_counter() - start_time) * 1000.0


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    if args_cli.freeze_nonvisual_randomization:
        _freeze_nonvisual_randomization(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        env.reset()
        action_dim = unwrapped.action_manager.total_action_dim
        actions = torch.zeros((args_cli.num_envs, action_dim), device=unwrapped.device)
        camera = unwrapped.scene.sensors["depth_camera"]
        print(
            "[TILED CAMERA] "
            f"envs={args_cli.num_envs} image={camera.cfg.width}x{camera.cfg.height} "
            f"update_period={camera.cfg.update_period:g}s "
            f"nonvisual_randomization={'frozen' if args_cli.freeze_nonvisual_randomization else 'on'}",
            flush=True,
        )

        for _ in range(args_cli.warmup_steps):
            env.step(actions)
        print(f"[TILED CAMERA] Completed {args_cli.warmup_steps} untimed full-step warmup frames.", flush=True)

        samples_ms: list[float] = []
        for step in range(args_cli.steps):
            elapsed_ms = _timed_step(env, actions)
            samples_ms.append(elapsed_ms)
            if step % args_cli.log_interval == 0 or step == args_cli.steps - 1:
                recent = samples_ms[-args_cli.log_interval :]
                mean_ms = statistics.fmean(recent)
                transitions_per_second = args_cli.num_envs * 1000.0 / max(mean_ms, 1.0e-6)
                print(
                    "[TILED CAMERA] "
                    f"step={step + 1}/{args_cli.steps} step_ms={elapsed_ms:.3f} "
                    f"recent_mean_ms={mean_ms:.3f} transitions/s={transitions_per_second:.1f}",
                    flush=True,
                )

        mean_ms = statistics.fmean(samples_ms)
        median_ms = statistics.median(samples_ms)
        p95_ms = sorted(samples_ms)[max(0, int(round(0.95 * (len(samples_ms) - 1))))]
        print(
            "[TILED CAMERA] SUMMARY "
            f"steps={len(samples_ms)} mean_ms={mean_ms:.3f} median_ms={median_ms:.3f} "
            f"p95_ms={p95_ms:.3f} transitions/s={args_cli.num_envs * 1000.0 / mean_ms:.1f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
