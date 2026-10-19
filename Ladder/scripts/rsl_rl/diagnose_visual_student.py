"""Bounded, no-update teacher/student shadow evaluation on identical visited states.

Only the selected controller executes. The other actor supplies labels, not
interventions. Post-action state is captured before automatic resets without
activating RecorderManager (which would add an observation/history computation).
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "Ladder" / "source" / "ladder"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output_dir", required=True, help="New directory; never overwrite an existing evaluation.")
parser.add_argument("--controller", choices=("student", "teacher"), required=True)
parser.add_argument("--condition", choices=("normal", "clean"), required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=3200)
parser.add_argument("--reference_frame", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1 or args_cli.max_steps < 1:
    parser.error("num_envs and max_steps must be positive")
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_tasks.utils import parse_env_cfg

import ladder.tasks  # noqa: F401
from ladder.tasks.ladder_climb.config.cr1.dagger.pure_model import (
    CR1PureVisualDaggerStudent,
    PURE_VISUAL_DAGGER_DATA_CONTRACT,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.pure_wrapper import as_pure_visual_dagger_tensordict
from ladder.tasks.ladder_climb.config.cr1.dagger.shadow_analysis import summarize_shadow_trace
from ladder.tasks.ladder_climb.config.cr1.dagger.shadow_config import configure_shadow_eval
from ladder.tasks.ladder_climb.config.cr1.dagger.student_teacher import load_frozen_teacher
from ladder.utils.contact_metrics import classify_limb_contacts
from ladder.utils.depth_camera_config import configure_depth_camera_timing


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu_copy(values):
    return {key: value.detach().cpu().clone() for key, value in values.items()}


def capture_state(env):
    """Fresh physical measurements; command.metrics may lag reference advancement."""
    motion = env.command_manager.get_term("motion")
    robot = motion.robot.data
    forces = motion._contact_sensor.data.net_forces_w[:, motion._contact_limb_body_indexes]
    required, physical, target = classify_limb_contacts(
        motion.contact_indicator, motion.robot_limb_pos_w, motion.target_contact_pos_w,
        forces, motion._limb_target_distance_thresholds, motion.cfg.contact_force_threshold,
    )
    return {
        "anchor_pos_error": (motion.robot_anchor_pos_w - motion.anchor_pos_w).norm(dim=-1),
        "anchor_rot_error": quat_error_magnitude(motion.robot_anchor_quat_w, motion.anchor_quat_w),
        "root_tilt": torch.acos((-robot.projected_gravity_b[:, 2]).clamp(-1, 1)),
        "root_ang_vel": robot.root_ang_vel_b,
        "root_pos": robot.root_pos_w - env.scene.env_origins,
        "root_quat": robot.root_quat_w,
        "root_lin_vel": robot.root_lin_vel_w,
        "joint_pos": motion.robot_joint_pos,
        "joint_vel": motion.robot_joint_vel,
        "reference_joint_pos": motion.joint_pos,
        "reference_anchor_pos": motion.anchor_pos_w - env.scene.env_origins,
        "reference_anchor_quat": motion.anchor_quat_w,
        "required_contact": required,
        "physical_contact": physical,
        "target_contact": target,
        "contact_forces": forces,
        "contact_target_error": (motion.robot_limb_pos_w - motion.target_contact_pos_w).norm(dim=-1),
    }


def write_scalar_csv(path, trace, dt):
    scalar_keys = [key for key, value in trace.items() if value.ndim == 2]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["action_step", "env_id", "time_before_action_s", *scalar_keys,
                         "action_rmse", "feature_mse", "missing_required_contacts"])
        for step in range(trace["valid"].shape[0]):
            for env_id in range(trace["valid"].shape[1]):
                if not trace["valid"][step, env_id]:
                    continue
                action_error = trace["student_action"][step, env_id] - trace["teacher_action"][step, env_id]
                feature_error = trace["student_feature"][step, env_id] - trace["teacher_feature"][step, env_id]
                missing = trace["post_required_contact"][step, env_id] & ~trace["post_physical_contact"][step, env_id]
                writer.writerow([step, env_id, step * dt,
                                 *(trace[key][step, env_id].item() for key in scalar_keys),
                                 action_error.square().mean().sqrt().item(),
                                 feature_error.square().mean().item(), missing.sum().item()])


def main():
    checkpoint = Path(args_cli.checkpoint).resolve()
    training_manifest_path = checkpoint.parent / "teacher_bc" / "manifest.json"
    training_manifest = json.loads(training_manifest_path.read_text())
    output = Path(args_cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher_path = Path(training_manifest["teacher_checkpoint"]["path"])
    motion_path = Path(training_manifest["motion"]["path"])
    for key, path in (("teacher_checkpoint", teacher_path), ("motion", motion_path)):
        if sha256(path) != training_manifest[key]["sha256"]:
            raise RuntimeError(f"{key} differs from the training manifest")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["data_contract"] != PURE_VISUAL_DAGGER_DATA_CONTRACT:
        raise RuntimeError("This diagnostic only supports the pure visual DAgger student")

    task = "Ladder-Climb-CR1-ZDW0824-UpDown-Pure-Visual-Dagger-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.commands.motion.motion_file = str(motion_path)
    configure_depth_camera_timing(env_cfg, checkpoint=checkpoint)
    config_manifest = configure_shadow_eval(
        env_cfg, args_cli.reference_frame, args_cli.seed, clean=args_cli.condition == "clean",
    )
    dump_yaml(str(output / "env.yaml"), env_cfg)
    gym_env = gym.make(task, cfg=env_cfg)
    env = gym_env.unwrapped
    action_term = env.action_manager.get_term("joint_pos")
    motion = env.command_manager.get_term("motion")

    def convert(observations):
        return as_pure_visual_dagger_tensordict(observations, num_envs=env.num_envs, device=env.device)

    # Exactly one observation computation from reset, then the normal env.step
    # observation. No extra sensor draws or history updates for shadow labels.
    observations = convert(gym_env.reset()[0])
    hidden_dims = [value.shape[0] for key, value in payload["policy_state_dict"].items()
                   if key.startswith("feature_mlp.") and key.endswith("weight")]
    student = CR1PureVisualDaggerStudent(env.action_manager.total_action_dim, hidden_dims, 1.0).to(env.device)
    student.load_state_dict(payload["policy_state_dict"], strict=True)
    student.eval().requires_grad_(False)
    teacher = load_frozen_teacher(
        str(teacher_path), observation_dim=observations["teacher"].shape[-1],
        action_dim=env.action_manager.total_action_dim, hidden_dims=[512, 256, 256], device=env.device,
    )
    teacher_initial = cpu_copy(teacher.state_dict())
    manifest = {
        "schema_version": 1, "args": vars(args_cli), "task": task,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "iter": payload.get("iter")},
        "training_manifest": str(training_manifest_path), "configuration": config_manifest,
        "teacher_checkpoint": training_manifest["teacher_checkpoint"], "motion": training_manifest["motion"],
        "student_network": str(student), "teacher_network": str(teacher),
        "step_dt": env.step_dt, "max_episode_length": env.max_episode_length,
        "action_joint_names": action_term._joint_names, "limb_names": motion.limb_names,
        "contact_force_threshold": motion.cfg.contact_force_threshold,
        "target_contact_distance_thresholds": motion._limb_target_distance_thresholds.cpu().tolist(),
        "termination_names": env.termination_manager.active_terms,
        "observation_shapes": {key: list(value.shape) for key, value in observations.items()},
        "protocol": {
            "control": "deterministic selected actor mean; no sampling, mixing, intervention or optimization",
            "episodes": "first episode per environment only; capped episodes are censored, not successes",
            "post_state": "after termination compute, before reset and reference advancement",
            "reference_alignment": "pre and post measurements compare against the same pre-action reference frame",
            "observation_timing": "reset output then env.step outputs; no extra observation computation",
            "trace": "all inputs, actor means/features, pre/post states, actual delayed/rate-limited targets and terminations",
            "comparison": "same initial seed/frame/config per condition, not a claim of paired future noise or GPU determinism",
        },
    }
    with (output / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    torch.save({"action_scale": action_term._scale, "action_offset": action_term._offset,
                "max_target_delta": action_term._max_target_delta,
                "env_origins": env.scene.env_origins}, output / "action_metadata.pt")

    # A normal RecorderTerm causes an extra observation_manager.compute in
    # ManagerBasedRLEnv.step. This instance-local read-only hook avoids that.
    original_compute = env.termination_manager.compute
    post = {}

    def compute_and_capture():
        done = original_compute()
        delay_buffer = action_term._action_delay_buffer
        state = {f"post_{key}": value for key, value in capture_state(env).items()}
        state.update({
            "done": done,
            "post_reference_frame": motion.time_steps,
            "delayed_action": action_term.raw_actions,
            "processed_joint_target": action_term.processed_actions,
            "delay": delay_buffer.time_lags if delay_buffer is not None else torch.zeros_like(motion.time_steps),
            **{f"termination/{name}": env.termination_manager.get_term(name)
               for name in env.termination_manager.active_terms},
        })
        post.clear()
        post.update(cpu_copy(state))
        return done

    env.termination_manager.compute = compute_and_capture
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    rows = []
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for step in range(args_cli.max_steps):
                student_action, student_feature = student.action_mean_and_feature(observations)
                teacher_feature = teacher[:-1](observations["teacher"])
                teacher_action = teacher[-1](teacher_feature)
                control = student_action if args_cli.controller == "student" else teacher_action
                row = cpu_copy({
                    "valid": alive, "reference_frame": motion.time_steps,
                    "student_action": student_action, "teacher_action": teacher_action,
                    "student_feature": student_feature, "teacher_feature": teacher_feature,
                    **{f"input/{key}": value for key, value in observations.items()},
                    **{f"pre_{key}": value for key, value in capture_state(env).items()},
                })
                next_obs, _, terminated, truncated, _ = gym_env.step(control)
                if not torch.equal(row["reference_frame"], post["post_reference_frame"]):
                    raise RuntimeError("Reference advanced before post-action capture")
                if not torch.equal((terminated | truncated).cpu(), post["done"]):
                    raise RuntimeError("Captured termination differs from env.step")
                row.update(post)
                rows.append(row)
                observations = convert(next_obs)
                alive &= ~(terminated | truncated)
                if step % 100 == 0 or not alive.any():
                    print(f"[SHADOW] {args_cli.condition}/{args_cli.controller} step={step + 1} "
                          f"alive={alive.sum().item()}/{env.num_envs} elapsed={time.monotonic() - started:.1f}s", flush=True)
                if not alive.any():
                    break
        trace = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        torch.save(trace, output / "trace.pt")
        summary = summarize_shadow_trace(trace, env.step_dt, args_cli.controller)
        summary["condition"] = args_cli.condition
        summary["wall_time_seconds"] = time.monotonic() - started
        summary["weights_unchanged"] = all(
            torch.equal(value.cpu(), payload["policy_state_dict"][key]) for key, value in student.state_dict().items()
        ) and all(torch.equal(value.cpu(), teacher_initial[key]) for key, value in teacher.state_dict().items())
        with (output / "summary.json").open("w") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        write_scalar_csv(output / "steps.csv", trace, env.step_dt)
        print(f"[SHADOW] Finished: {output / 'summary.json'}", flush=True)
    finally:
        env.termination_manager.compute = original_compute
        gym_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
