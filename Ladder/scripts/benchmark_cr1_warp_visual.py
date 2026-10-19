"""Validate and benchmark CR1's Warp-backed visual-observation task.

It measures full ManagerBasedRLEnv steps after Warp depth has been integrated
into the actor observation.  This is intentionally separate from PPO training
and from the RTX/TiledCamera benchmark.
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

parser = argparse.ArgumentParser(description="Validate CR1 Warp visual observations and throughput.")
parser.add_argument(
    "--task",
    type=str,
    default="Ladder-Climb-CR1-Geometry-DR-WarpVisual-v0",
    help="Warp visual task to benchmark.",
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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ladder.tasks  # noqa: F401


def _freeze_nonvisual_randomization(env_cfg) -> None:
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


def _timed_step(env, actions: torch.Tensor):
    if torch.cuda.is_available() and str(env.unwrapped.device).startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        start.record()
        observations, *_ = env.step(actions)
        finish.record()
        finish.synchronize()
        return observations, start.elapsed_time(finish)
    start_time = time.perf_counter()
    observations, *_ = env.step(actions)
    return observations, (time.perf_counter() - start_time) * 1000.0


def _validate_depth(observations: dict[str, torch.Tensor], num_envs: int) -> dict[str, float]:
    depth = observations["visual_depth"]
    expected = (num_envs, 1, 60, 80)
    if tuple(depth.shape) != expected:
        raise RuntimeError(f"Warp visual depth shape must be {expected}, got {tuple(depth.shape)}")
    if not torch.isfinite(depth).all():
        raise RuntimeError("Warp visual depth contains non-finite values")
    minimum = float(depth.min().item())
    maximum = float(depth.max().item())
    if minimum < 0.0 or maximum > 2.0 + 1.0e-5:
        raise RuntimeError(f"Warp visual depth outside [0, 2]: min={minimum}, max={maximum}")
    return {
        "min": minimum,
        "max": maximum,
        "zero_fraction": float((depth == 0.0).float().mean().item()),
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    if args_cli.freeze_nonvisual_randomization:
        _freeze_nonvisual_randomization(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        observations, _ = env.reset()
        initial_stats = _validate_depth(observations, args_cli.num_envs)
        action_dim = unwrapped.action_manager.total_action_dim
        actions = torch.zeros((args_cli.num_envs, action_dim), device=unwrapped.device)
        print(
            "[WARP VISUAL] "
            f"envs={args_cli.num_envs} depth_shape={tuple(observations['visual_depth'].shape)} "
            f"initial_depth={initial_stats} "
            f"rtx_sensors={unwrapped.sim.has_rtx_sensors()} "
            f"nonvisual_randomization={'frozen' if args_cli.freeze_nonvisual_randomization else 'on'}",
            flush=True,
        )
        if unwrapped.sim.has_rtx_sensors():
            raise RuntimeError("Warp visual task unexpectedly contains an RTX sensor")

        for _ in range(args_cli.warmup_steps):
            env.step(actions)
        print(f"[WARP VISUAL] Completed {args_cli.warmup_steps} untimed full-step warmup frames.", flush=True)

        samples_ms: list[float] = []
        for step in range(args_cli.steps):
            observations, elapsed_ms = _timed_step(env, actions)
            samples_ms.append(elapsed_ms)
            if step % args_cli.log_interval == 0 or step == args_cli.steps - 1:
                stats = _validate_depth(observations, args_cli.num_envs)
                recent = samples_ms[-args_cli.log_interval :]
                mean_ms = statistics.fmean(recent)
                transitions_per_second = args_cli.num_envs * 1000.0 / max(mean_ms, 1.0e-6)
                print(
                    "[WARP VISUAL] "
                    f"step={step + 1}/{args_cli.steps} step_ms={elapsed_ms:.3f} "
                    f"recent_mean_ms={mean_ms:.3f} transitions/s={transitions_per_second:.1f} "
                    f"depth={stats}",
                    flush=True,
                )

        mean_ms = statistics.fmean(samples_ms)
        median_ms = statistics.median(samples_ms)
        p95_ms = sorted(samples_ms)[max(0, int(round(0.95 * (len(samples_ms) - 1))))]
        print(
            "[WARP VISUAL] SUMMARY "
            f"steps={len(samples_ms)} mean_ms={mean_ms:.3f} median_ms={median_ms:.3f} "
            f"p95_ms={p95_ms:.3f} transitions/s={args_cli.num_envs * 1000.0 / mean_ms:.1f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
