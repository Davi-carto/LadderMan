"""Check CR1 visual-task height-map and SSR data tensors during reset and step."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

parser = argparse.ArgumentParser(description="Validate CR1 limb height-map targets.")
parser.add_argument(
    "--task",
    type=str,
    default="Ladder-Climb-CR1-Geometry-DR-WarpVisual-v0",
    help="CR1 visual task to check (RTX or Warp).",
)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.steps <= 0:
    parser.error("--steps must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ladder.tasks  # noqa: F401


EXPECTED_TARGET_SHAPE = (args_cli.num_envs, 4, 10, 3)
EXPECTED_PROPRIO_SHAPE = (args_cli.num_envs, 10, 85)
EXPECTED_DEPTH_HISTORY_SHAPE = (args_cli.num_envs, 2, 60, 80)


def _check(observations, label: str) -> None:
    targets = observations["visual_targets"]
    if not isinstance(targets, dict):
        raise RuntimeError(f"visual_targets must remain a dictionary, got {type(targets).__name__}")
    relative = targets["relative_height"]
    hit_valid = targets["hit_valid"]
    if tuple(relative.shape) != EXPECTED_TARGET_SHAPE:
        raise RuntimeError(
            f"{label} relative height shape must be {EXPECTED_TARGET_SHAPE}, got {tuple(relative.shape)}"
        )
    if tuple(hit_valid.shape) != EXPECTED_TARGET_SHAPE:
        raise RuntimeError(
            f"{label} hit-valid shape must be {EXPECTED_TARGET_SHAPE}, got {tuple(hit_valid.shape)}"
        )
    if not torch.isfinite(relative).all() or not torch.isfinite(hit_valid).all():
        raise RuntimeError(f"{label} target tensors contain non-finite values")
    if not torch.all((hit_valid == 0.0) | (hit_valid == 1.0)):
        raise RuntimeError(f"{label} hit-valid tensor is not binary")
    if torch.any(relative[hit_valid == 0.0] != 0.0):
        raise RuntimeError(f"{label} no-hit relative targets must be zero")
    print(
        f"[HEIGHTMAP TARGETS] {label} shape={tuple(relative.shape)} "
        f"valid_fraction={float(hit_valid.mean()):.4f} "
        f"relative_range=({float(relative.min()):.4f}, {float(relative.max()):.4f})",
        flush=True,
    )


def _check_ssr_data(observations, label: str, *, reset: bool) -> None:
    inputs = observations["visual_encoder_inputs"]
    targets = observations["visual_aux_targets"]
    if not isinstance(inputs, dict) or not isinstance(targets, dict):
        raise RuntimeError("SSR inputs and auxiliary targets must remain dictionaries")
    proprio_history = inputs["proprio_history"]
    depth_history = inputs["depth_history"]
    base_lin_vel = targets["base_lin_vel"]
    next_proprio = targets["next_proprio"]
    if tuple(proprio_history.shape) != EXPECTED_PROPRIO_SHAPE:
        raise RuntimeError(
            f"{label} proprio history must be {EXPECTED_PROPRIO_SHAPE}, got {tuple(proprio_history.shape)}"
        )
    if tuple(depth_history.shape) != EXPECTED_DEPTH_HISTORY_SHAPE:
        raise RuntimeError(
            f"{label} depth history must be {EXPECTED_DEPTH_HISTORY_SHAPE}, got {tuple(depth_history.shape)}"
        )
    if tuple(base_lin_vel.shape) != (args_cli.num_envs, 3):
        raise RuntimeError(f"{label} velocity target must have shape [B, 3]")
    if tuple(next_proprio.shape) != (args_cli.num_envs, 85):
        raise RuntimeError(f"{label} next proprio target must have shape [B, 85]")
    for name, tensor in {
        "proprio_history": proprio_history,
        "depth_history": depth_history,
        "base_lin_vel": base_lin_vel,
        "next_proprio": next_proprio,
    }.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{label} {name} contains non-finite values")
    torch.testing.assert_close(proprio_history[:, -1], next_proprio)
    torch.testing.assert_close(depth_history[:, -1], observations["visual_depth"][:, 0])
    if reset:
        torch.testing.assert_close(proprio_history, proprio_history[:, :1].expand_as(proprio_history))
        torch.testing.assert_close(depth_history, depth_history[:, :1].expand_as(depth_history))
    print(
        f"[SSR DATA] {label} proprio={tuple(proprio_history.shape)} "
        f"depth={tuple(depth_history.shape)} velocity={tuple(base_lin_vel.shape)} "
        f"proprio_lag_mean={float(torch.abs(proprio_history[:, -1] - proprio_history[:, 0]).mean()):.6f} "
        f"depth_lag_mean={float(torch.abs(depth_history[:, -1] - depth_history[:, 0]).mean()):.6f}",
        flush=True,
    )


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        observations, _ = env.reset()
        _check(observations, "reset")
        _check_ssr_data(observations, "reset", reset=True)
        actor_dims = {
            name: unwrapped.observation_manager.group_obs_dim[name]
            for name in ("visual_proprio", "visual_depth")
        }
        print(
            f"[HEIGHTMAP TARGETS] actor_group_dims={actor_dims} "
            f"critic_dim={unwrapped.observation_manager.group_obs_dim['critic']} "
            f"target_group_dims={unwrapped.observation_manager.group_obs_dim['visual_targets']} "
            f"encoder_input_dims={unwrapped.observation_manager.group_obs_dim['visual_encoder_inputs']} "
            f"aux_target_dims={unwrapped.observation_manager.group_obs_dim['visual_aux_targets']}",
            flush=True,
        )
        action_dim = unwrapped.action_manager.total_action_dim
        actions = torch.zeros((args_cli.num_envs, action_dim), device=unwrapped.device)
        for step in range(args_cli.steps):
            observations, *_ = env.step(actions)
            _check(observations, f"step={step + 1}")
            _check_ssr_data(observations, f"step={step + 1}", reset=False)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
