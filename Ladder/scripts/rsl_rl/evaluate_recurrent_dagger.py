"""Bounded first-episode recurrent student/teacher evaluation without optimizers."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "Ladder/source/ladder"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-PPO-Dagger-Synced-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--motion_file", required=True)
parser.add_argument("--output_dir", required=True, help="New output directory; existing evaluations are preserved.")
parser.add_argument("--reference_frame", "--fixed_reference_frame", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_steps", type=int, default=3500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--controller", choices=("student_mean", "teacher", "fixed_std", "checkpoint_std"), default="student_mean")
parser.add_argument("--fixed_std", type=float, default=0.1)
parser.add_argument("--nominal_ladder", action="store_true", help="Explicitly disable ladder geometry randomization only.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if min(args_cli.num_envs, args_cli.max_steps) < 1 or min(args_cli.reference_frame, args_cli.seed) < 0:
    parser.error("Positive env/step counts and non-negative reference frame/seed are required")
if args_cli.fixed_std < 0:
    parser.error("fixed_std must be non-negative")
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_tasks.utils import parse_env_cfg

import ladder.tasks  # noqa: F401
from ladder.tasks.ladder_climb.config.cr1.dagger.pure_wrapper import as_pure_visual_dagger_tensordict
from ladder.tasks.ladder_climb.config.cr1.dagger.recurrent_evaluation import FirstEpisodeAccumulator
from ladder.tasks.ladder_climb.config.cr1.dagger.recurrent_visual_model import (
    CR1RecurrentVisualDaggerStudent, RECURRENT_VISUAL_DAGGER_DATA_CONTRACT,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.privileged_recurrent_visual_model import (
    CR1PrivilegedRecurrentVisualDaggerStudent,
    CR1PrivilegedRecurrentVisualLegacyDaggerStudent,
    PRIVILEGED_RECURRENT_VISUAL_DAGGER_DATA_CONTRACT,
    PRIVILEGED_RECURRENT_VISUAL_LEGACY_DAGGER_DATA_CONTRACT,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.recurrent_visual_ppo_model import (
    CR1RecurrentVisualPpoStudent, RECURRENT_VISUAL_PPO_DAGGER_DATA_CONTRACT,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.recurrent_visual_ln_mse_algorithm import (
    RECURRENT_VISUAL_LN_MSE_DAGGER_DATA_CONTRACT,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.student_teacher import load_frozen_teacher
from ladder.utils.depth_camera_config import configure_depth_camera_timing


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}".encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def capture_state(env):
    motion = env.command_manager.get_term("motion")
    robot = motion.robot.data
    return {key: value.detach().clone() for key, value in {
        "reference_frame": motion.time_steps,
        "command": motion.effective_climb_command.squeeze(-1),
        "root_pos": robot.root_pos_w - env.scene.env_origins,
        "root_quat": robot.root_quat_w,
        "root_tilt_rad": torch.acos((-robot.projected_gravity_b[:, 2]).clamp(-1, 1)),
        "anchor_pos_error_m": (motion.robot_anchor_pos_w - motion.anchor_pos_w).norm(dim=-1),
        "anchor_rot_error_rad": quat_error_magnitude(motion.robot_anchor_quat_w, motion.anchor_quat_w),
    }.items()}


def main():
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    motion_path = Path(args_cli.motion_file).expanduser().resolve()
    output = Path(args_cli.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = payload.get("data_contract")
    classes = {
        RECURRENT_VISUAL_DAGGER_DATA_CONTRACT: CR1RecurrentVisualDaggerStudent,
        RECURRENT_VISUAL_PPO_DAGGER_DATA_CONTRACT: CR1RecurrentVisualPpoStudent,
        RECURRENT_VISUAL_LN_MSE_DAGGER_DATA_CONTRACT: CR1RecurrentVisualPpoStudent,
        PRIVILEGED_RECURRENT_VISUAL_LEGACY_DAGGER_DATA_CONTRACT: CR1PrivilegedRecurrentVisualLegacyDaggerStudent,
        PRIVILEGED_RECURRENT_VISUAL_DAGGER_DATA_CONTRACT: CR1PrivilegedRecurrentVisualDaggerStudent,
    }
    if contract not in classes:
        raise ValueError(f"Unsupported recurrent student data contract: {contract}")
    teacher_path = Path(payload["teacher_checkpoint"]).expanduser().resolve()
    teacher_hash = sha256(teacher_path)
    if teacher_hash != payload["teacher_checkpoint_sha256"]:
        raise RuntimeError("Source teacher checkpoint hash differs from student checkpoint")
    training_manifest_path = next(checkpoint.parent.glob("recurrent_visual_*_diagnostics/manifest.json"), None)
    motion_hash = sha256(motion_path)
    if training_manifest_path is not None:
        training_manifest = json.loads(training_manifest_path.read_text())
        if training_manifest.get("motion_file_sha256", motion_hash) != motion_hash:
            raise RuntimeError("Evaluation motion differs from the training motion")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.commands.motion.motion_file = str(motion_path)
    env_cfg.commands.motion.fixed_reference_frame = args_cli.reference_frame
    if args_cli.nominal_ladder:
        env_cfg.commands.motion.ladder_inclination_delta_range = (0.0, 0.0)
        env_cfg.commands.motion.ladder_spacing_delta_range = (0.0, 0.0)
        env_cfg.commands.motion.nominal_ladder_geometry_probability = 1.0
    configure_depth_camera_timing(env_cfg, checkpoint=checkpoint)
    dump_yaml(str(output / "env.yaml"), env_cfg)
    # Load weights before constructing the environment, whose seed initializes
    # startup/reset randomization independently of network initialization draws.
    state = payload["policy_state_dict"]
    hidden_dims = [value.shape[0] for key, value in sorted(
        state.items(), key=lambda item: int(item[0].split('.')[1]) if re.fullmatch(r"feature_mlp\.\d+\.weight", item[0]) else -1
    ) if re.fullmatch(r"feature_mlp\.\d+\.weight", key)]
    student = classes[contract](state["action_head.weight"].shape[0], hidden_dims, 0.1)
    student.load_state_dict(state, strict=True)
    student.to(env_cfg.sim.device).eval().requires_grad_(False)
    teacher = load_frozen_teacher(str(teacher_path), observation_dim=780,
                                 action_dim=student.action_dim, hidden_dims=[512, 256, 256], device=env_cfg.sim.device)
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = gym_env.unwrapped
    motion = env.command_manager.get_term("motion")
    convert = lambda observations: as_pure_visual_dagger_tensordict(
        observations, num_envs=env.num_envs, device=env.device,
    )
    observations = convert(gym_env.reset()[0])
    initial = capture_state(env)
    initial_tensors = {**initial, "joint_pos": motion.robot.data.joint_pos,
                       **{f"observation/{key}": value for key, value in observations.items()}}
    initial_tensors = {key: value.detach().cpu().clone() for key, value in initial_tensors.items()}
    torch.save(initial_tensors, output / "initial_state.pt")
    downward = (motion.motion.climb_direction < 0).nonzero().flatten()
    downward_start = int(downward[0]) if downward.numel() else None
    accounting = FirstEpisodeAccumulator(initial, env.step_dt, downward_start)
    generator = torch.Generator(device=env.device).manual_seed(args_cli.seed + 1000003)
    std = student.output_std.detach()
    if args_cli.controller == "fixed_std":
        std = torch.full_like(std, args_cli.fixed_std)
    manifest = {
        "args": vars(args_cli), "task": args_cli.task, "data_contract": contract,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint),
                       "iteration": payload.get("iter"), "update_iteration": payload.get("update_iteration")},
        "teacher_checkpoint": {"path": str(teacher_path), "sha256": teacher_hash},
        "motion_file": {"path": str(motion_path), "sha256": motion_hash, "frames": motion.motion.time_step_total},
        "training_manifest": str(training_manifest_path) if training_manifest_path else None,
        "environment_config_file": "env.yaml", "environment_config_sha256": sha256(output / "env.yaml"),
        "step_dt": env.step_dt, "reference_frame": args_cli.reference_frame,
        "downward_start_frame": downward_start,
        "initial_tensor_sha256": {key: tensor_sha256(value) for key, value in initial_tensors.items()},
        "student_network": repr(student), "checkpoint_action_std": student.output_std.tolist(),
        "executed_action_std": std.tolist() if args_cli.controller in ("fixed_std", "checkpoint_std") else [0.0] * student.action_dim,
        "observation_shapes": {key: list(value.shape) for key, value in observations.items()},
        "control_delay_physics_steps": [env_cfg.actions.joint_pos.min_control_delay, env_cfg.actions.joint_pos.max_control_delay],
        "protocol": {
            "episode_sampling": "first episode per environment; no replacement by autoreset episodes",
            "terminal_state_timing": "post-physics after termination compute, before reward/reset/reference advancement",
            "success": "motion_complete without any simultaneous non-timeout failure term",
            "censoring": "unfinished first episodes at evaluation budget are censored, never successes",
            "randomization": "training randomization retained; only fixed frame and explicit nominal_ladder override",
            "matching": "matched initial seed; future observation/reset RNG streams need not stay paired",
            "sampling_rng": "separate generator for action noise, seed + 1000003",
            "ascent_marker": "reached first negative-command frame without simultaneous failure; not a separate geometric success test",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    original_compute = env.termination_manager.compute
    terminal = {}

    def compute_and_capture():
        dones = original_compute()
        terminal.clear()
        terminal.update(state=capture_state(env), dones=dones.clone(), flags={
            name: env.termination_manager.get_term(name).clone()
            for name in env.termination_manager.active_terms
        })
        return dones

    env.termination_manager.compute = compute_and_capture
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for step in range(args_cli.max_steps):
                mean = student(observations)
                teacher_mean = teacher(observations["teacher"])
                actions = teacher_mean if args_cli.controller == "teacher" else mean
                if args_cli.controller in ("fixed_std", "checkpoint_std"):
                    actions = actions + torch.randn(actions.shape, generator=generator, device=env.device) * std
                frame_before = motion.time_steps.clone()
                raw_obs, rewards, terminated, truncated, _ = gym_env.step(actions)
                dones = terminated | truncated
                if not torch.equal(terminal["state"]["reference_frame"], frame_before) or not torch.equal(dones, terminal["dones"]):
                    raise RuntimeError("Terminal-state capture lost reference/done alignment")
                accounting.add(terminal["state"], rewards, dones, terminal["flags"],
                               (mean - teacher_mean).square().mean(-1))
                student.reset(dones)
                observations = convert(raw_obs)
                if step % 250 == 0 or not accounting.active.any():
                    print(f"[EVAL] step={step + 1} active={int(accounting.active.sum())}/{env.num_envs} "
                          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
                if not accounting.active.any():
                    break
        episodes, summary = accounting.report()
        summary.update(controller=args_cli.controller, wall_time_seconds=time.monotonic() - started,
                       reference_frame=args_cli.reference_frame)
        (output / "episodes.json").write_text(json.dumps(episodes, indent=2, allow_nan=False) + "\n")
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print("[EVAL] " + json.dumps(summary), flush=True)
        print(f"[EVAL] Results: {output}", flush=True)
    finally:
        env.termination_manager.compute = original_compute
        gym_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
