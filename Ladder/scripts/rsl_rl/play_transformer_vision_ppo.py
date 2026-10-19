"""Play and inspect a complete CR1 TransformerVision PPO checkpoint.

The policy controls the robot with its native recurrent visual actor.  The
same encoder evaluation used to produce each action also drives a live depth
and limb-height-map dashboard, so diagnostics never advance a second GRU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

DEFAULT_TASK = (
    "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
    "SafeFrame0-WarpTransformerVision-PPO-v0"
)
DEFAULT_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "restored_prefix_v300_stride2_slow50"
    / "smooth_start"
    / "ladder_geometry_dr"
    / "climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_train_ready_ladder_dr.npz"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "isaaclab_v2.1.0"
    / "logs"
    / "rsl_rl"
    / "cr1_transformer_vision_ppo_thin_rails_scratch"
    / "2026-08-19_16-33-44_resume1600_to8000_save400"
    / "model_7999.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "cr1_transformer_vision_ppo_play"
LIMB_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")
PHYSICAL_CONTACT_BODY_NAMES = {
    "left_hand": "left_sphere_hand_link",
    "right_hand": "right_sphere_hand_link",
    "left_foot": "left_ankle_x_link",
    "right_foot": "right_ankle_x_link",
}
LIMB_COLORS = {
    "left_hand": (0.0, 0.85, 1.0),
    "right_hand": (0.2, 0.35, 1.0),
    "left_foot": (0.15, 1.0, 0.25),
    "right_foot": (1.0, 0.75, 0.05),
}
PROPRIO_FIELDS = (
    ("root_angular_velocity", slice(0, 3)),
    ("root_orientation_6d", slice(3, 9)),
    ("joint_position", slice(9, 34)),
    ("joint_velocity", slice(34, 59)),
    ("action", slice(59, 84)),
    ("climb_direction", slice(84, 85)),
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=DEFAULT_TASK)
parser.add_argument("--motion_file", type=Path, default=DEFAULT_MOTION)
parser.add_argument(
    "--depth_update_hz", type=float, default=None,
    help="Override Warp depth refresh Hz (0: policy rate). Default: checkpoint camera timing.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--display_env", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=0, help="Zero runs until the viewer closes.")
parser.add_argument("--print_interval", type=int, default=25)
parser.add_argument("--save_interval", type=int, default=200, help="PNG interval; zero disables periodic captures.")
parser.add_argument("--display_scale", type=float, default=0.75)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--no_show", action="store_true", help="Disable dashboard and 3D height-map markers.")
parser.add_argument("--show_reference_guidance", action="store_true",
                    help="Show selected whole-body reference points (cyan) and ankle targets (green).")
parser.add_argument("--start_paused", action="store_true")
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--reference_frame", type=int, default=None)
parser.add_argument(
    "--reset_at_step",
    type=int,
    default=0,
    help="Request one automatic reset after this many control steps; zero disables it.",
)
parser.add_argument(
    "--zero_command_at_step",
    type=int,
    default=0,
    help="Request one safe zero-command stop after this many control steps; zero disables it.",
)
parser.add_argument(
    "--resume_command_at_step",
    type=int,
    default=0,
    help="Resume after a zero-command hold at this control step; zero disables it.",
)
parser.add_argument(
    "--disable_randomization",
    action="store_true",
    help="Use nominal reset, ladder, camera, sensor, actuator, and control-delay settings.",
)
parser.add_argument(
    "--enable_disturbances",
    action="store_true",
    help="Keep the training-time interval root pushes; they are disabled by default for inspection.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(checkpoint=str(DEFAULT_CHECKPOINT))
args_cli = parser.parse_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if not 0 <= args_cli.display_env < args_cli.num_envs:
    parser.error("--display_env must be in [0, --num_envs)")
if (
    args_cli.max_steps < 0
    or args_cli.reset_at_step < 0
    or args_cli.zero_command_at_step < 0
    or args_cli.resume_command_at_step < 0
    or args_cli.print_interval <= 0
    or args_cli.save_interval < 0
):
    parser.error(
        "step/interval arguments must be nonnegative and --print_interval positive"
    )
if args_cli.display_scale <= 0.0:
    parser.error("--display_scale must be positive")
if args_cli.headless and not args_cli.no_show:
    parser.error("--headless requires --no_show")
if args_cli.start_paused and args_cli.headless:
    parser.error("--start_paused requires an interactive viewer")
if "/log/level" not in args_cli.kit_args:
    args_cli.kit_args = f"{args_cli.kit_args} --/log/level=error".strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import cv2
import gymnasium as gym
import numpy as np
import omni.timeline
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_tasks.utils import parse_env_cfg
from ladder.utils.depth_camera_config import configure_depth_camera_timing
from rsl_rl.runners import OnPolicyRunner

import ladder.tasks  # noqa: F401
from ladder.rsl_rl.modules import (
    LEGACY_TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT,
    VARIATIONAL_TRANSFORMER_VISION_DATA_CONTRACT,
)
from ladder.rsl_rl.modules.transformer_vision_actor import DEPTH_HISTORY_KEY
from ladder.rsl_rl.runners import TransformerVisionPPOVecEnvWrapper
from ladder.rsl_rl.storage.transformer_vision_ppo_storage import (
    HEIGHTMAP_HIT_CLASS_KEY,
    HEIGHTMAP_KEY,
    HEIGHTMAP_VALID_KEY,
    MOTION_JOINT_POS_KEY,
    MOTION_ROOT_ORIENTATION_KEY,
    MOTION_TARGET_VALID_KEY,
    NEXT_PROPRIO_KEY,
    VELOCITY_KEY,
)
from ladder.utils.limb_heightmaps import RUNG, relative_height_to_world_points


carb.logging.acquire_logging().set_level_threshold(carb.logging.LEVEL_ERROR)


def _stats(values: torch.Tensor) -> dict[str, float | int | None]:
    finite = values.detach()[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(finite.numel()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)),
    }


def _prediction_maps(output) -> torch.Tensor:
    return torch.stack(
        (output.hand_left_map, output.hand_right_map, output.foot_left_map, output.foot_right_map),
        dim=-3,
    )


def _masked_mse_per_env(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    axes = tuple(range(1, prediction.ndim))
    weight = weight.to(dtype=prediction.dtype)
    return ((prediction - target).square() * weight).sum(dim=axes) / weight.sum(dim=axes).clamp_min(1.0)


def _map_metrics(output, observations) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = _prediction_maps(output)
    target = observations[HEIGHTMAP_KEY]
    valid = observations[HEIGHTMAP_VALID_KEY] > 0.5
    rung = valid & (observations[HEIGHTMAP_HIT_CLASS_KEY] == RUNG)
    non_rung = valid & ~rung
    weighted = valid.to(prediction.dtype) * (1.0 + rung.to(prediction.dtype))
    metrics: dict[str, torch.Tensor] = {}
    for index, name in enumerate(LIMB_NAMES):
        metrics[f"{name}_weighted_mse"] = _masked_mse_per_env(
            prediction[:, index], target[:, index], weighted[:, index]
        )
        metrics[f"{name}_rung_mse"] = _masked_mse_per_env(
            prediction[:, index], target[:, index], rung[:, index]
        )
        metrics[f"{name}_non_rung_mse"] = _masked_mse_per_env(
            prediction[:, index], target[:, index], non_rung[:, index]
        )
        metrics[f"{name}_rung_fraction"] = rung[:, index].sum(dim=(1, 2)) / valid[
            :, index
        ].sum(dim=(1, 2)).clamp_min(1)
    return prediction, metrics


def _velocity_metrics(output, observations) -> dict[str, torch.Tensor]:
    target = observations[VELOCITY_KEY]
    error = output.velocity - target
    model_mse = error.square().mean(dim=-1)
    zero_mse = target.square().mean(dim=-1)
    skill = torch.full_like(model_mse, torch.nan)
    meaningful_baseline = zero_mse > 1.0e-8
    skill[meaningful_baseline] = 1.0 - model_mse[meaningful_baseline] / zero_mse[meaningful_baseline]
    return {
        "velocity_mse": model_mse,
        "velocity_zero_mse": zero_mse,
        "velocity_skill": skill,
        "velocity_error_norm": torch.linalg.vector_norm(error, dim=-1),
    }


def _future_metrics(output, next_observations, dones: torch.Tensor) -> dict[str, torch.Tensor]:
    if getattr(output, "next_proprio", None) is None:
        return {}
    target = next_observations[NEXT_PROPRIO_KEY][..., : output.next_proprio.shape[-1]]
    error = (output.next_proprio - target).square()
    metrics = {"future_mse": error.mean(dim=-1)}
    for name, field_slice in PROPRIO_FIELDS:
        if field_slice.stop <= error.shape[-1]:
            metrics[f"future_{name}_mse"] = error[:, field_slice].mean(dim=-1)
    valid = ~dones.to(dtype=torch.bool)
    return {name: value[valid] for name, value in metrics.items()}


def _motion_metrics(output, observations) -> dict[str, torch.Tensor]:
    if output.motion_joint_pos is None or output.motion_root_orientation_6d is None:
        return {}
    valid = observations[MOTION_TARGET_VALID_KEY].to(dtype=torch.bool)
    joint_mse = (
        output.motion_joint_pos - observations[MOTION_JOINT_POS_KEY]
    ).square().mean(dim=-1)
    orientation_mse = (
        output.motion_root_orientation_6d
        - observations[MOTION_ROOT_ORIENTATION_KEY]
    ).square().mean(dim=-1)
    invalid = torch.full_like(joint_mse, torch.nan)
    return {
        "motion_joint_mse": torch.where(valid, joint_mse, invalid),
        "motion_root_orientation_mse": torch.where(
            valid, orientation_mse, invalid
        ),
    }


def _contact_forces(env) -> dict[str, torch.Tensor]:
    sensor = env.scene.sensors["contact_forces"]
    name_to_index = {name: index for index, name in enumerate(sensor.body_names)}
    return {
        limb: torch.linalg.vector_norm(sensor.data.net_forces_w[:, name_to_index[body]], dim=-1)
        for limb, body in PHYSICAL_CONTACT_BODY_NAMES.items()
        if body in name_to_index
    }


def _command_metrics(env) -> dict[str, object]:
    command = env.command_manager.get_term("motion")
    env_id = args_cli.display_env
    if hasattr(command, "task_index"):
        ground_sensors = command._stable_anchor_ground_contact_sensors
        return {
            "task_index": int(command.task_index[env_id]),
            "single_task_episode": bool(command.cfg.single_task_episode),
            "sequence_start_task_index": int(command.metrics["sequence_start_task_index"][env_id]),
            "completed_support_transitions": int(command.metrics["task_completed_count"][env_id]),
            "sequence_completion_fraction": float(command.metrics["sequence_completion_fraction"][env_id]),
            "reference_frame": int(command.time_steps[env_id]),
            "reference_guidance": (None if getattr(command, "reference_guidance", None) is None else {
                name: value[env_id].tolist() for name, value in command.reference_guidance.diagnostics.items()
            }),
            "reference_frames": command.motion.time_step_total,
            "episode_step": int(env.episode_length_buf[env_id]),
            "effective_climb_command": float(command.effective_climb_command[env_id, 0]),
            "active_limbs": command.active_limbs[env_id].tolist(),
            "source_support": command.source_ids[env_id].tolist(),
            "target_support": command.target_ids[env_id].tolist(),
            "actual_support": command.actual_stable_support_rung_id[env_id].tolist(),
            "current_expected_support": command.current_expected_support_ids[env_id].tolist(),
            "contact_phases": (None if command.contact_phases is None else
                               command.contact_phases.phase[env_id].tolist()),
            "protected_support": (None if command.contact_phases is None else
                                   command.contact_phases.protected[env_id].tolist()),
            "release_milestones": (None if command.contact_phases is None else
                                   command.contact_phases.released[env_id].tolist()),
            "acquired_milestones": (None if command.contact_phases is None else
                                    command.contact_phases.acquired[env_id].tolist()),
            "deadline_s": float(command.task_deadline_steps[env_id]) * env.step_dt,
            "body_target_w_m": (command.body_goal_pos_w[env_id].tolist()
                                if getattr(command.cfg, "contact_task_body_goal_enabled", False) else None),
            "body_target_error_m": (float((command.body_goal_pos_w[env_id]
                - command.robot_anchor_pos_w[env_id]).norm())
                if getattr(command.cfg, "contact_task_body_goal_enabled", False) else None),
            "body_goal_progress_m": (float(command.metrics["body_goal_progress_m"][env_id])
                if getattr(command.cfg, "contact_task_body_goal_enabled", False) else None),
            "body_completion_error_m": (float(command.metrics["body_completion_error_m"][env_id])
                if getattr(command.cfg, "contact_task_body_goal_enabled", False) else None),
            "foot_safety": {name: value[env_id].tolist() for name, value in
                            getattr(command, "foot_reward_diagnostics", {}).items()},
            "ground_net_force_n": [
                float(torch.linalg.vector_norm(sensor.data.net_forces_w[env_id]))
                for sensor in ground_sensors
            ],
            "ground_filtered_force_n": [
                float(torch.linalg.vector_norm(sensor.data.force_matrix_w[env_id], dim=-1).amax())
                for sensor in ground_sensors
            ],
            "limb_height_w_m": command.robot_limb_pos_w[env_id, :, 2].tolist(),
            "target_error_m": (command.robot_limb_pos_w[env_id] - command.goal_pos_w[env_id]).norm(dim=-1).tolist(),
            "elapsed_s": float(command.task_steps[env_id]) * env.step_dt,
        }
    limb_error = torch.linalg.vector_norm(
        command.robot_limb_pos_w[env_id] - command.target_contact_pos_w[env_id], dim=-1
    )
    joint_error = (command.robot_joint_pos[env_id] - command.joint_pos[env_id]).abs()
    body_error = torch.linalg.vector_norm(
        command.robot_body_pos_w[env_id] - command.body_pos_relative_w[env_id], dim=-1
    )
    forces = _contact_forces(env)
    stop_state = int(command._stop_state[env_id])
    stop_state_name = {0: "RUN", 1: "SEEK", 2: "HOLD"}.get(
        stop_state, f"UNKNOWN({stop_state})"
    )
    return {
        "reference_frame": int(command.time_steps[env_id]),
        "reference_frames": int(command.motion.time_step_total),
        "effective_climb_command": float(command.effective_climb_command[env_id, 0]),
        "stop_state": stop_state_name,
        "stop_target_frame": int(command._stop_target_frame[env_id]),
        "episode_step": int(env.episode_length_buf[env_id]),
        "ladder_inclination_deg": math.degrees(float(command.ladder_phi[env_id])),
        "ladder_rung_spacing_m": float(command.ladder_spacing[env_id]),
        "anchor_position_error_m": float(
            torch.linalg.vector_norm(command.robot_anchor_pos_w[env_id] - command.anchor_pos_w[env_id])
        ),
        "anchor_orientation_error_rad": float(
            quat_error_magnitude(command.robot_anchor_quat_w[env_id], command.anchor_quat_w[env_id])
        ),
        "joint_abs_error_mean_rad": float(joint_error.mean()),
        "joint_abs_error_max_rad": float(joint_error.max()),
        "body_position_error_mean_m": float(body_error.mean()),
        "body_position_error_max_m": float(body_error.max()),
        "limbs": {
            name: {
                "required": bool(command.contact_indicator[env_id, index] > 0.5),
                "rung_id": int(command.contact_rung_id[env_id, index]),
                "target_error_m": float(limb_error[index]),
                "physical_force_n": float(forces.get(name, torch.zeros(env.num_envs, device=env.device))[env_id]),
            }
            for index, name in enumerate(command.limb_names)
        },
    }


def _termination_values(env, env_id: int) -> dict[str, bool]:
    return {
        name: bool(values[0])
        for name, values in env.termination_manager.get_active_iterable_terms(env_idx=env_id)
    }


class _MetricAccumulator:
    def __init__(self) -> None:
        self._sum: dict[str, float] = {}
        self._count: dict[str, int] = {}
        self._minimum: dict[str, float] = {}
        self._maximum: dict[str, float] = {}

    def add(self, name: str, values: torch.Tensor) -> None:
        finite = values.detach()[torch.isfinite(values)]
        if finite.numel() == 0:
            return
        self._sum[name] = self._sum.get(name, 0.0) + float(finite.sum())
        self._count[name] = self._count.get(name, 0) + int(finite.numel())
        self._minimum[name] = min(self._minimum.get(name, math.inf), float(finite.min()))
        self._maximum[name] = max(self._maximum.get(name, -math.inf), float(finite.max()))

    def report(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "count": self._count[name],
                "mean": self._sum[name] / self._count[name],
                "min": self._minimum[name],
                "max": self._maximum[name],
            }
            for name in sorted(self._count)
        }


class _EvaluationLog:
    def __init__(self, output_dir: Path, checkpoint: Path, checkpoint_iteration: int, num_envs: int) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.output_dir / "events.jsonl").open("w", encoding="utf-8")
        self.checkpoint = checkpoint
        self.checkpoint_iteration = checkpoint_iteration
        self.metrics = _MetricAccumulator()
        self.total_steps = 0
        self.total_transitions = 0
        self.episodes: list[dict[str, object]] = []
        self._episode_rewards = torch.zeros(num_envs)
        self._episode_lengths = torch.zeros(num_envs, dtype=torch.long)

    def add_visual(self, metrics: dict[str, torch.Tensor], output) -> None:
        for name, value in metrics.items():
            self.metrics.add(name, value)
        if hasattr(output, "recurrent_feature"):
            self.metrics.add("recurrent_feature_abs", output.recurrent_feature.abs())
        if hasattr(output, "dynamic_latent"):
            self.metrics.add("dynamic_abs", output.dynamic_latent.abs())
        if getattr(output, "motion_latent", None) is not None:
            self.metrics.add("motion_latent_abs", output.motion_latent.abs())
        if (
            getattr(output, "dynamic_mu", None) is not None
            and getattr(output, "dynamic_logvar", None) is not None
        ):
            self.metrics.add("dynamic_mu_abs", output.dynamic_mu.abs())
            self.metrics.add("dynamic_logvar", output.dynamic_logvar)
            self.metrics.add("dynamic_std", torch.exp(0.5 * output.dynamic_logvar))

    def process_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        env,
        terminal_frames: torch.Tensor,
        future: dict[str, torch.Tensor],
    ) -> None:
        rewards_cpu = rewards.detach().cpu()
        self._episode_rewards += rewards_cpu
        self._episode_lengths += 1
        self.total_steps += 1
        self.total_transitions += int(rewards.numel())
        for name, value in future.items():
            self.metrics.add(name, value)
        for env_id in torch.nonzero(dones.detach().cpu(), as_tuple=False).flatten().tolist():
            causes = _termination_values(env, env_id)
            episode = {
                "env_id": env_id,
                "length": int(self._episode_lengths[env_id]),
                "reward": float(self._episode_rewards[env_id]),
                "terminal_reference_frame": int(terminal_frames[env_id]),
                "causes": causes,
                "motion_complete": bool(causes.get("motion_complete", False)),
                "task_success": bool(causes.get("task_success", False)),
            }
            command = env.command_manager.get_term("motion")
            if hasattr(command, "last_episode_metrics"):
                episode["single_task_episode"] = bool(command.cfg.single_task_episode)
                episode["contact_sequence"] = {
                    name: float(values[env_id])
                    for name, values in command.last_episode_metrics.items()
                }
            self.episodes.append(episode)
            self.write({"type": "episode", **episode})
            self._episode_rewards[env_id] = 0.0
            self._episode_lengths[env_id] = 0

    def manual_reset(self, terminal_frames: torch.Tensor) -> None:
        """Close active trajectories explicitly so reset-time statistics never leak across episodes."""

        for env_id in torch.nonzero(self._episode_lengths > 0, as_tuple=False).flatten().tolist():
            episode = {
                "env_id": env_id,
                "length": int(self._episode_lengths[env_id]),
                "reward": float(self._episode_rewards[env_id]),
                "terminal_reference_frame": int(terminal_frames[env_id]),
                "causes": {"manual_reset": True},
                "motion_complete": False,
            }
            self.episodes.append(episode)
            self.write({"type": "episode", **episode})
        self._episode_rewards.zero_()
        self._episode_lengths.zero_()

    def write(self, record: dict[str, object]) -> None:
        self._jsonl.write(json.dumps(record, sort_keys=True) + "\n")
        self._jsonl.flush()

    def close(self) -> Path:
        completed = sum(bool(episode["motion_complete"]) for episode in self.episodes)
        summary = {
            "purpose": "No-update deterministic playback of a complete TransformerVision PPO checkpoint.",
            "checkpoint": str(self.checkpoint),
            "checkpoint_iteration": self.checkpoint_iteration,
            "total_control_steps": self.total_steps,
            "total_transitions": self.total_transitions,
            "episodes_finished": len(self.episodes),
            "motion_complete_episodes": completed,
            "task_success_episodes": sum(bool(e.get("task_success", False)) for e in self.episodes),
            "sequence_complete_episodes": sum(
                bool(e.get("contact_sequence", {}).get("sequence_success", False))
                and not e.get("single_task_episode", True) for e in self.episodes
            ),
            "motion_complete_fraction": None if not self.episodes else completed / len(self.episodes),
            "metrics": self.metrics.report(),
            "episodes": self.episodes,
        }
        path = self.output_dir / "summary.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
        self._jsonl.close()
        return path


def _colorize(
    values: np.ndarray,
    lower: float,
    upper: float,
    invalid: np.ndarray,
    *,
    flip_vertical: bool = True,
) -> np.ndarray:
    normalized = np.round(np.clip((values - lower) / max(upper - lower, 1.0e-8), 0.0, 1.0) * 255).astype(
        np.uint8
    )
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    image[invalid] = (35, 35, 35)
    return np.flipud(image) if flip_vertical else image


def _tile(image: np.ndarray, title: str, detail: str, *, width: int, height: int) -> np.ndarray:
    canvas = np.full((height + 52, width + 12, 3), 18, dtype=np.uint8)
    canvas[30 : 30 + height, 6 : 6 + width] = cv2.resize(
        image, (width, height), interpolation=cv2.INTER_NEAREST
    )
    cv2.putText(canvas, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (245, 245, 245), 1)
    cv2.putText(canvas, detail, (6, height + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (205, 205, 205), 1)
    return canvas


def _dashboard(
    output,
    observations,
    metrics: dict[str, torch.Tensor],
    *,
    env_id: int,
    total_step: int,
    reference_frame: int,
    overlay_mode: str,
) -> np.ndarray:
    prediction = _prediction_maps(output)[env_id].detach().cpu().numpy()
    target = observations[HEIGHTMAP_KEY][env_id].detach().cpu().numpy()
    valid = (observations[HEIGHTMAP_VALID_KEY][env_id] > 0.5).detach().cpu().numpy()
    visible = np.concatenate((target[valid], prediction[valid])) if np.any(valid) else np.zeros(1)
    value_limit = max(float(np.abs(visible).max()), 0.05)
    error = np.abs(prediction - target)
    error_limit = max(float(error[valid].max()) if np.any(valid) else 0.0, 0.01)
    rows = []
    for index, limb in enumerate(LIMB_NAMES):
        invalid = ~valid[index]
        tiles = (
            _tile(
                _colorize(target[index], -value_limit, value_limit, invalid),
                "target relative height (m)",
                f"range=[{-value_limit:+.3f},{value_limit:+.3f}]",
                width=180,
                height=120,
            ),
            _tile(
                _colorize(prediction[index], -value_limit, value_limit, invalid),
                "prediction relative height (m)",
                f"weighted MSE={float(metrics[f'{limb}_weighted_mse'][env_id]):.6f}",
                width=180,
                height=120,
            ),
            _tile(
                _colorize(error[index], 0.0, error_limit, invalid),
                "absolute error (m)",
                f"rung/non={float(metrics[f'{limb}_rung_mse'][env_id]):.5f}/"
                f"{float(metrics[f'{limb}_non_rung_mse'][env_id]):.5f}",
                width=180,
                height=120,
            ),
        )
        row = np.concatenate(tiles, axis=1)
        label = np.full((25, row.shape[1], 3), 26, dtype=np.uint8)
        cv2.putText(label, limb, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        rows.append(np.concatenate((label, row), axis=0))

    width = rows[0].shape[1]
    header = np.full((70, width, 3), 14, dtype=np.uint8)
    cv2.putText(
        header,
        f"TransformerVision PPO | step={total_step} frame={reference_frame} env={env_id} | 3D={overlay_mode}",
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (250, 250, 250),
        1,
    )
    cv2.putText(
        header,
        "columns: target | prediction | absolute error; TURBO: blue=low, red=high, gray=invalid",
        (7, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (210, 210, 210),
        1,
    )
    cv2.putText(
        header,
        "H: target/prediction/off   R: reset   P: pause   N: one step   I: print   Q: quit",
        (7, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (210, 210, 210),
        1,
    )
    centered_depth = observations[DEPTH_HISTORY_KEY][env_id, -1]
    depth_m = centered_depth + 1.0
    depth_np = depth_m.detach().cpu().numpy()
    invalid_depth = ~np.isfinite(depth_np) | (depth_np <= 0.0)
    depth_tile = _tile(
        # Warp depth is already stored in camera image row order.  Only the
        # limb-map panels need the display flip used above.
        _colorize(depth_np, 0.1, 2.0, invalid_depth, flip_vertical=False),
        "latest processed Warp depth reconstructed to meters",
        f"centered input range=({float(centered_depth.min()):+.3f},{float(centered_depth.max()):+.3f})",
        width=width - 12,
        height=150,
    )
    return np.concatenate((header, depth_tile, *rows), axis=0)


def _opencv_highgui_available() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", maxsplit=1)[1].strip().upper() != "NONE"
    return True


class _DashboardWindow:
    def __init__(self, scale: float) -> None:
        self._scale = scale
        self._backend = "opencv" if _opencv_highgui_available() else "omni"
        self._window = None
        self._provider = None
        self._omni_ui = None
        if self._backend == "omni":
            import omni.ui

            self._omni_ui = omni.ui
            print("[VISION PLAY] OpenCV HighGUI unavailable; using an Isaac Sim image window.", flush=True)

    def show(self, image: np.ndarray) -> None:
        image = cv2.resize(
            image, dsize=None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_NEAREST
        )
        if self._backend == "opencv":
            cv2.imshow("CR1 TransformerVision PPO", image)
            cv2.waitKey(1)
            return
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        height, width = rgba.shape[:2]
        if self._provider is None:
            self._provider = self._omni_ui.ByteImageProvider()
            self._window = self._omni_ui.Window(
                "CR1 TransformerVision PPO", width=width, height=height + 30, visible=True
            )
            with self._window.frame:
                self._omni_ui.ImageWithProvider(
                    self._provider, fill_policy=self._omni_ui.IwpFillPolicy.IWP_STRETCH
                )
        self._provider.set_bytes_data(rgba.flatten().data, [width, height])

    def close(self) -> None:
        if self._backend == "opencv":
            cv2.destroyAllWindows()
        elif self._window is not None:
            self._window.visible = False
            self._window.destroy()


def _make_markers() -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/CR1TransformerVisionPPO/limb_heightmaps",
            markers={
                name: sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
                for name, color in LIMB_COLORS.items()
            },
        )
    )


class _HeightMapOverlay:
    def __init__(self, marker: VisualizationMarkers, env_id: int) -> None:
        self._marker = marker
        self._env_id = env_id
        self._modes = ("target", "prediction", "off")
        self._mode_index = 0
        self._target_points = None
        self._prediction_points = None
        self._valid = None

    @property
    def mode(self) -> str:
        return self._modes[self._mode_index]

    def cycle(self) -> None:
        self._mode_index = (self._mode_index + 1) % len(self._modes)
        self._render()
        print(f"[VISION PLAY][CONTROL] 3D height-map overlay={self.mode}", flush=True)

    def update(self, plane_points_w, target, prediction, valid) -> None:
        self._target_points = relative_height_to_world_points(plane_points_w, target)
        self._prediction_points = relative_height_to_world_points(plane_points_w, prediction)
        self._valid = valid.to(dtype=torch.bool)
        self._render()

    def _render(self) -> None:
        if self.mode == "off" or self._valid is None:
            self._marker.set_visibility(False)
            return
        source = self._target_points if self.mode == "target" else self._prediction_points
        positions = []
        marker_indices = []
        for index in range(len(LIMB_NAMES)):
            valid = self._valid[self._env_id, index]
            points = source[self._env_id, index][valid].clone()
            points[:, 2] += 0.01
            positions.append(points)
            marker_indices.append(
                torch.full((points.shape[0],), index, dtype=torch.long, device=points.device)
            )
        positions = torch.cat(positions)
        marker_indices = torch.cat(marker_indices)
        if positions.numel() == 0:
            self._marker.set_visibility(False)
            return
        self._marker.set_visibility(True)
        self._marker.visualize(positions, marker_indices=marker_indices)

    def close(self) -> None:
        self._marker.set_visibility(False)


class _Controller:
    def __init__(self, *, paused: bool, enabled: bool) -> None:
        self.paused = paused
        self.step = False
        self.reset = False
        self.info = False
        self.cycle = False
        self.zero_command = False
        self.resume_command = False
        self.quit = False
        self._input = None
        self._keyboard = None
        self._subscription = None
        self._timeline = omni.timeline.get_timeline_interface()
        if enabled:
            from omni import appwindow

            window = appwindow.get_default_app_window()
            if window is not None:
                self._input = carb.input.acquire_input_interface()
                self._keyboard = window.get_keyboard()
                self._subscription = self._input.subscribe_to_keyboard_events(
                    self._keyboard, self._on_event
                )

    def _on_event(self, event, *args) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input.name
        if key == "P":
            self.paused = not self.paused
            print(f"[VISION PLAY][CONTROL] {'paused' if self.paused else 'playing'}", flush=True)
        elif key == "N" and self.paused:
            self.step = True
        elif key == "R":
            self.reset = True
            # The key can arrive from inside an Isaac render update.  Keep the
            # timeline live so the current render/physics call can return to
            # the main loop, where the actual environment reset is performed.
            self.ensure_playing()
            print("[VISION PLAY][CONTROL] reset requested; waiting for current render step", flush=True)
        elif key == "I":
            self.info = True
        elif key == "H":
            self.cycle = True
        elif key in ("KEY_0", "0"):
            self.zero_command = True
            print("[VISION PLAY][CONTROL] zero command requested", flush=True)
        elif key == "C":
            self.resume_command = True
            print("[VISION PLAY][CONTROL] climb resume requested", flush=True)
        elif key in ("Q", "ESCAPE"):
            self.quit = True
        return True

    def ensure_playing(self) -> None:
        if not self._timeline.is_playing():
            self._timeline.play()

    def close(self) -> None:
        if self._subscription is None:
            return
        unsubscribe = getattr(self._input, "unsubscribe_to_keyboard_events", None)
        if unsubscribe is None:
            unsubscribe = self._input.unsubscribe_from_keyboard_events
        unsubscribe(self._keyboard, self._subscription)


def _disable_randomization(env_cfg) -> None:
    for name in (
        "robot_material",
        "robot_mass",
        "joint_position_offset",
        "torso_com",
        "camera_extrinsics",
        "root_velocity_push",
        "motor_strength",
        "motor_strength_interval",
    ):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    command = env_cfg.commands.motion
    command.pose_range = {name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")}
    command.fixed_start_pose_range = {}
    command.velocity_range = {
        name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    command.joint_position_range = (0.0, 0.0)
    command.ladder_inclination_delta_range = (0.0, 0.0)
    command.ladder_spacing_delta_range = (0.0, 0.0)
    env_cfg.actions.joint_pos.min_control_delay = 0
    env_cfg.actions.joint_pos.max_control_delay = 0
    history = env_cfg.observations.transformer_vision_inputs.proprio_history.params
    history["add_proprio_noise"] = False
    depth = env_cfg.observations.transformer_vision_depth.image.params
    for name in ("gaussian_std", "pixel_dropout_probability", "rung_focused_mask_probability"):
        depth[name] = 0.0
    depth["translation_range"] = (0.0, 0.0)
    depth["rotation_range"] = (0.0, 0.0)
    depth["horizontal_fov_randomization_range_deg"] = (0.0, 0.0)
    depth["vertical_fov_randomization_range_deg"] = (0.0, 0.0)


def _print_setup(env, observations, actor, checkpoint: Path, checkpoint_payload, env_cfg) -> None:
    depth_cfg = env_cfg.observations.transformer_vision_depth.image.params
    history_cfg = env_cfg.observations.transformer_vision_inputs.proprio_history.params
    policy_parameters = sum(parameter.numel() for parameter in actor.parameters())
    critic_parameters = sum(
        value.numel() for value in checkpoint_payload["critic_state_dict"].values() if torch.is_tensor(value)
    )
    print("\n[VISION PLAY][SETUP] complete TransformerVision PPO playback", flush=True)
    print(
        f"  checkpoint={checkpoint} iteration={checkpoint_payload.get('iter')} "
        f"policy_params={policy_parameters} critic_params={critic_parameters}"
    )
    print(f"  task={args_cli.task} motion={env_cfg.commands.motion.motion_file}")
    print(
        f"  envs={env.num_envs} display_env={args_cli.display_env} device={env.device} "
        f"step_dt={env.step_dt:.4f}s deterministic_actor=1"
    )
    print(
        "  inputs: "
        + " ".join(f"{name}={tuple(value.shape)}" for name, value in observations.items())
    )
    print(
        f"  depth: image={depth_cfg['width']}x{depth_cfg['height']} "
        f"fov=({depth_cfg['horizontal_fov_deg']},{depth_cfg['vertical_fov_deg']})deg "
        f"range=({depth_cfg['min_depth']},{depth_cfg['max_depth']})m "
        f"mean={history_cfg.get('depth_mean')} scale={history_cfg.get('depth_scale')} "
        f"noise_std={depth_cfg.get('gaussian_std')} dropout={depth_cfg.get('pixel_dropout_probability')} "
        f"rung_mask_probability={depth_cfg.get('rung_focused_mask_probability')}"
    )
    print(
        f"  randomization={'disabled' if args_cli.disable_randomization else 'training-style'} "
        f"root_disturbances={int(args_cli.enable_disturbances)}"
    )
    if actor.encoder_variant == "end_to_end_policy_only":
        if getattr(actor, "use_contact_task", False):
            print(f"  EVENT CONTACT TASK: GRU256 + command/state{actor.oracle_input_dim} with measured/current-expected/next-stable "
                  "support IDs (-1=none, 0=ground, 1..7=rung); no reference clock")
            if actor.use_contact_task_phases:
                print("  contact phase per limb: 0=prepare, 1=release, 2=move, 3=acquire; "
                      "expected IDs update from measured execution, NOT reference time")
            if actor.use_contact_task_body_goal:
                print("  body goal: next-stable body-anchor error XYZ in current anchor frame; "
                      "soft approach reward, NOT whole-body center of mass; privileged oracle goal")
        elif actor.use_oracle_contact_goal:
            print(
                "  ORACLE DIAGNOSTIC: actor uses GRU256 plus next-stable goal12 and three "
                "12-D support relations (actual/current expected/next stable); NOT deployable"
            )
        print("  encoder: proprio128 + depth128 -> GRU256; no Transformer")
        print("  policy-only encoder; no velocity or height-map reconstruction")
    elif actor.encoder_variant == "end_to_end_reconstruction":
        print("  map_order=left_hand,right_hand,left_foot,right_foot shape=[4,10,3] units=relative meters")
        print("  dynamics: none; velocity 3 + four height-map latents 32; end-to-end PPO")
        print("  losses: velocity=2 hand=2 foot=2; height points rung:non-rung=2:1")
    elif actor.encoder_variant == "variational_legacy":
        print("  dynamics: legacy VAE, future target=full proprio 85, KL=1")
    else:
        print("  dynamics: deterministic 16-D latent, future target=physical state 59, no sampling/KL")
    if actor.encoder_variant not in ("end_to_end_reconstruction", "end_to_end_policy_only"):
        print("  losses: velocity=2 hand=2 foot=2 future=5; height points rung:non-rung=2:1")
    controls = "0=safe stop, C=resume, R=reset, P=pause, N=single-step, I=print, Q=quit"
    if actor.encoder_variant != "end_to_end_policy_only":
        controls = "0=safe stop, C=resume, H=target/prediction/off, R=reset, P=pause, N=single-step, I=print, Q=quit"
    if getattr(actor, "use_contact_task", False):
        controls = "R=reset, P=pause, N=single-step, I=print, Q=quit (0/C unavailable)"
    print(
        f"  controls: {controls}\n"
    )


def _print_frame(record: dict[str, object]) -> None:
    action = record["action"]
    command = record["command"]
    print(
        f"\n[VISION PLAY][{record['label']}] step={record['total_step']} "
        f"episode_step={command['episode_step']} frame={command['reference_frame']}/"
        f"{command['reference_frames'] - 1} reward={record['reward_after']:+.5f} "
        f"done={int(record['done'])} overlay={record['overlay_mode']}",
        flush=True,
    )
    print(
        f"  action: l2={action['l2']:.4f} min={action['min']:+.4f} max={action['max']:+.4f} "
        f"policy_std_mean={record['policy_std_mean']:.4f}"
    )
    if record.get("policy_only"):
        print(f"  recurrent_feature_l2={record['recurrent_feature_l2']:.4f}")
        if "task_index" in command:
            print(f"  task={command['task_index']} active={command['active_limbs']} "
                  f"target={command['target_support']} actual={command['actual_support']} "
                  f"elapsed={command['elapsed_s']:.2f}s errors={command['target_error_m']}")
            if command.get("contact_phases") is not None:
                print(f"  phases={command['contact_phases']} expected={command['current_expected_support']} "
                      f"protected={command['protected_support']} deadline={command['deadline_s']:.2f}s")
            if command.get("body_target_error_m") is not None:
                print(f"  body_goal_error={command['body_target_error_m']:.3f}m "
                      f"body_progress={command['body_goal_progress_m']:+.5f}m "
                      f"last_completion_error={command['body_completion_error_m']:.3f}m")
            if command.get("foot_safety"):
                foot = command["foot_safety"]
                print(f"  feet [L,R]: swing={foot['swing_mask']} toe_up={foot['toe_up_deg']}deg "
                      f"side_tilt={foot['side_tilt_deg']}deg "
                      f"front_side={foot['front_force_n']}N underside={foot['underside_force_n']}N")
            if command.get("reference_guidance"):
                guide = command["reference_guidance"]
                print(f"  reference guide={guide['frame']:.1f} match={guide['matched_frame']} "
                      f"valid={guide['valid']} weighted_rate={guide['weighted_rate']:.3f} "
                      f"ankle_targets_w={guide['foot_targets_w']}")
                print(f"  reference next={guide.get('next_frame')} pause_reason={guide.get('pause_reason')} "
                      "(0=advance,1=phase,2=end,3=conflict,4=inactive/unsafe)")
                print(f"  reference pause_age={guide['pause_age_s']:.2f}s scale={guide['reward_scale']:.3f} "
                      f"raw_rate={guide['raw_weighted_rate']:.3f} applied_rate={guide['weighted_rate']:.3f}")
            print(f"  ground: net_force={command['ground_net_force_n']}N "
                  f"filtered_force={command['ground_filtered_force_n']}N "
                  f"limb_z={command['limb_height_w_m']}m")
            if record["done"]:
                print(f"  termination={record['terminations']}")
            return
        print(
            f"  depth: meters=({record['depth']['min_m']:.3f},{record['depth']['max_m']:.3f}) "
            f"invalid_fraction={record['depth']['invalid_fraction']:.3f} "
            f"centered=({record['depth']['centered_min']:+.3f},{record['depth']['centered_max']:+.3f})"
        )
        print(
            f"  tracking: anchor_pos={command['anchor_position_error_m']:.4f}m "
            f"anchor_ori={command['anchor_orientation_error_rad']:.4f}rad "
            f"joint_mean/max={command['joint_abs_error_mean_rad']:.4f}/"
            f"{command['joint_abs_error_max_rad']:.4f}rad "
            f"body_mean/max={command['body_position_error_mean_m']:.4f}/"
            f"{command['body_position_error_max_m']:.4f}m"
        )
        print(
            f"  command: effective={command['effective_climb_command']:+.0f} "
            f"state={command['stop_state']} target_frame={command['stop_target_frame']}"
        )
        if record["done"]:
            print(f"  termination={record['terminations']}")
        return
    velocity = record["velocity"]
    dynamic = record["dynamic"]
    velocity_skill = velocity["skill"]
    velocity_skill_text = "n/a" if velocity_skill is None else f"{velocity_skill:+.3f}"
    print(
        f"  velocity: target={velocity['target']} prediction={velocity['prediction']} "
        f"mse={velocity['mse']:.6f} zero_mse={velocity['zero_mse']:.6f} skill={velocity_skill_text}"
    )
    if dynamic["type"] == "vae":
        print(
            f"  dynamic(VAE): mu_abs_mean={dynamic['abs_mean']:.6f} "
            f"mu_abs_max={dynamic['abs_max']:.6f} logvar_mean={dynamic['logvar_mean']:+.6f} "
            f"std_mean={dynamic['std_mean']:.6f} latent_l2={record['latent_l2']:.4f}"
        )
    elif dynamic["type"] == "none":
        print(f"  latent: velocity+heightmaps only, l2={record['latent_l2']:.4f}")
    else:
        print(
            f"  dynamic(deterministic): abs_mean={dynamic['abs_mean']:.6f} "
            f"abs_max={dynamic['abs_max']:.6f} latent_l2={record['latent_l2']:.4f}"
        )
    print(
        f"  depth: meters=({record['depth']['min_m']:.3f},{record['depth']['max_m']:.3f}) "
        f"invalid_fraction={record['depth']['invalid_fraction']:.3f} "
        f"centered=({record['depth']['centered_min']:+.3f},{record['depth']['centered_max']:+.3f})"
    )
    for limb in LIMB_NAMES:
        maps = record["heightmaps"][limb]
        contact = command["limbs"][limb]
        print(
            f"  {limb}: map_weighted={maps['weighted_mse']:.6f} "
            f"rung={maps['rung_mse']:.6f} non_rung={maps['non_rung_mse']:.6f} "
            f"required={int(contact['required'])} rung_id={contact['rung_id']} "
            f"target_error={contact['target_error_m']:.4f}m force={contact['physical_force_n']:.2f}N"
        )
    future = record.get("future")
    if future is not None:
        print(
            "  future: "
            + " ".join(f"{name}={value:.6f}" for name, value in future.items())
        )
    print(
        f"  tracking: anchor_pos={command['anchor_position_error_m']:.4f}m "
        f"anchor_ori={command['anchor_orientation_error_rad']:.4f}rad "
        f"joint_mean/max={command['joint_abs_error_mean_rad']:.4f}/"
        f"{command['joint_abs_error_max_rad']:.4f}rad "
        f"body_mean/max={command['body_position_error_mean_m']:.4f}/"
        f"{command['body_position_error_max_m']:.4f}m"
    )
    print(
        f"  command: effective={command['effective_climb_command']:+.0f} "
        f"state={command['stop_state']} target_frame={command['stop_target_frame']}"
    )
    if record["done"]:
        print(f"  termination={record['terminations']}")


def _configure_end_to_end_playback(runner_cfg: dict, num_envs: int) -> dict[str, bool]:
    """Make a training runner config safe for actor-only playback."""
    algorithm_cfg = runner_cfg["algorithm"]
    algorithm_cfg["num_mini_batches"] = min(
        int(algorithm_cfg["num_mini_batches"]), int(num_envs)
    )
    return {
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
    }


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    motion_file = args_cli.motion_file.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TransformerVision PPO checkpoint was not found: {checkpoint}")
    if not motion_file.is_file():
        raise FileNotFoundError(f"Motion file was not found: {motion_file}")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    output_dir = args_cli.output_dir.expanduser().resolve() / checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = None
    env_cfg.viewer.body_name = None
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_cfg.commands.motion.fixed_reference_frame = args_cli.reference_frame
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    if args_cli.disable_randomization:
        _disable_randomization(env_cfg)
    elif not args_cli.enable_disturbances and hasattr(env_cfg.events, "root_velocity_push"):
        env_cfg.events.root_velocity_push = None

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner_cfg = agent_cfg.to_dict()
    if runner_cfg.get("policy", {}).get("use_contact_task", False) and (
        args_cli.zero_command_at_step or args_cli.resume_command_at_step
    ):
        raise ValueError("Event contact tasks have no reference-clock stop/resume")
    end_to_end = runner_cfg.get("use_end_to_end_visual_ppo", False)
    if runner_cfg.get("use_end_to_end_visual_amp_ppo", False):
        end_to_end = True
    if not runner_cfg.get("use_transformer_vision_runner", False) and not end_to_end:
        raise RuntimeError("This script requires a TransformerVision PPO task configuration")
    if checkpoint_payload.get("data_contract") == VARIATIONAL_TRANSFORMER_VISION_DATA_CONTRACT:
        runner_cfg["algorithm"]["visual_architecture"] = "variational_legacy"
    elif checkpoint_payload.get("data_contract") == LEGACY_TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT:
        runner_cfg["algorithm"]["visual_architecture"] = "task_goal_legacy"

    configure_depth_camera_timing(env_cfg, checkpoint=checkpoint, update_hz=args_cli.depth_update_hz)
    wrapper_class = TransformerVisionPPOVecEnvWrapper
    if runner_cfg.get("use_end_to_end_visual_amp_ppo", False):
        from ladder.rsl_rl.runners.end_to_end_visual_amp_wrapper import (
            EndToEndVisualAMPPPOVecEnvWrapper,
        )

        wrapper_class = EndToEndVisualAMPPPOVecEnvWrapper
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif end_to_end:
        from ladder.rsl_rl.runners.end_to_end_visual_wrapper import EndToEndVisualPPOVecEnvWrapper

        wrapper_class = EndToEndVisualPPOVecEnvWrapper
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    env = wrapper_class(
        raw_env,
        clip_actions=agent_cfg.clip_actions,
        clip_observations=getattr(agent_cfg, "clip_observations", None),
        **({
            "num_frames": runner_cfg["algorithm"].get("amp_num_frames", 5),
            "history_stride": runner_cfg["algorithm"].get("amp_history_stride", 2),
            "anchor_hold_probability": runner_cfg["algorithm"].get("amp_anchor_hold_probability", 0.2),
            "balance_directions": runner_cfg["algorithm"].get("amp_balance_directions", True),
        } if runner_cfg.get("use_end_to_end_visual_amp_ppo", False) else {}),
    )
    playback_load_cfg = None
    if end_to_end:
        configured_mini_batches = int(runner_cfg["algorithm"]["num_mini_batches"])
        playback_load_cfg = _configure_end_to_end_playback(runner_cfg, env.num_envs)
        effective_mini_batches = int(runner_cfg["algorithm"]["num_mini_batches"])
        if effective_mini_batches != configured_mini_batches:
            print(
                "[VISION PLAY] End-to-end inference reduced num_mini_batches "
                f"from {configured_mini_batches} to {effective_mini_batches} for "
                f"{env.num_envs} playback environment(s).",
                flush=True,
            )
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    if end_to_end:
        runner.load(str(checkpoint), load_cfg=playback_load_cfg, strict=True)
    else:
        runner.load(str(checkpoint))
    actor = runner.alg.get_policy().to(env.device).eval()
    if not hasattr(actor, "forward_with_output"):
        raise RuntimeError("TransformerVisionActor.forward_with_output is unavailable")
    actor.reset()
    observations = env.get_observations()
    _print_setup(env.unwrapped, observations, actor, checkpoint, checkpoint_payload, env_cfg)
    policy_only = actor.encoder_variant == "end_to_end_policy_only"

    evaluation = _EvaluationLog(
        output_dir, checkpoint, int(checkpoint_payload.get("iter", -1)), env.num_envs
    )
    controller = _Controller(paused=args_cli.start_paused, enabled=not args_cli.headless)
    window = None if args_cli.no_show or policy_only else _DashboardWindow(args_cli.display_scale)
    overlay = (
        None
        if args_cli.no_show or policy_only
        else _HeightMapOverlay(_make_markers(), args_cli.display_env)
    )
    latest_dashboard = None
    guidance_markers = None
    if args_cli.show_reference_guidance:
        command = env.unwrapped.command_manager.get_term("motion")
        if getattr(command, "reference_guidance", None) is None:
            raise ValueError("--show_reference_guidance requires the Reference task variant")
        guidance_markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/CR1ReferenceGuidance",
            markers={name: sim_utils.SphereCfg(
                radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
                for name, radius, color in (("body", .018, (0., .8, 1.)), ("ankle", .025, (.1, 1., .1)))},
        ))
    total_step = 0
    automatic_reset_issued = False
    automatic_zero_command_issued = False
    automatic_resume_command_issued = False
    force_print = True

    try:
        while simulation_app.is_running() and not controller.quit:
            if controller.reset:
                reset_start = time.time()
                print("[VISION PLAY][RESET] begin", flush=True)
                controller.ensure_playing()
                evaluation.manual_reset(
                    env.unwrapped.command_manager.get_term("motion").time_steps.detach().cpu()
                )
                print("[VISION PLAY][RESET] episode statistics closed", flush=True)
                observations = env.reset()
                print("[VISION PLAY][RESET] environment state and observations reset", flush=True)
                actor.reset()
                controller.ensure_playing()
                controller.reset = False
                latest_dashboard = None
                force_print = True
                reset_frame = int(
                    env.unwrapped.command_manager.get_term("motion").time_steps[args_cli.display_env]
                )
                print(
                    f"[VISION PLAY][RESET] complete frame={reset_frame} actor_GRU=cleared "
                    f"elapsed_ms={(time.time() - reset_start) * 1000.0:.1f}",
                    flush=True,
                )
            if controller.cycle:
                if overlay is not None:
                    overlay.cycle()
                controller.cycle = False
                force_print = True
            if controller.info:
                force_print = True
                controller.info = False
            motion_command = env.unwrapped.command_manager.get_term("motion")
            if getattr(actor, "use_contact_task", False) and (
                controller.zero_command or controller.resume_command
            ):
                print("[VISION PLAY][CONTROL] 0/C unavailable for event contact tasks", flush=True)
                controller.zero_command = controller.resume_command = False
            if controller.zero_command:
                motion_command.request_zero_command()
                controller.zero_command = False
                force_print = True
                print(
                    "[VISION PLAY][CONTROL] safe-stop targets="
                    f"{motion_command._stop_target_frame.detach().cpu().tolist()}",
                    flush=True,
                )
            if controller.resume_command:
                motion_command.resume_climb_command()
                controller.resume_command = False
                force_print = True

            single_step = controller.step
            controller.step = False
            if controller.paused and not single_step:
                env.unwrapped.sim.render()
                if window is not None and latest_dashboard is not None:
                    window.show(latest_dashboard)
                time.sleep(0.01)
                continue

            start = time.time()
            controller.ensure_playing()
            command_before = _command_metrics(env.unwrapped)
            terminal_frames = env.unwrapped.command_manager.get_term("motion").time_steps.clone()
            with torch.no_grad():
                actions, output = actor.forward_with_output(observations)
            if policy_only:
                predictions, map_metrics, velocity_metrics, motion_metrics = None, {}, {}, {}
            else:
                predictions, map_metrics = _map_metrics(output, observations)
                velocity_metrics = _velocity_metrics(output, observations)
                motion_metrics = _motion_metrics(output, observations)
            visual_metrics = {**map_metrics, **velocity_metrics, **motion_metrics}
            evaluation.add_visual(visual_metrics, output)

            overlay_mode = "off"
            if overlay is not None:
                cache = getattr(env.unwrapped, "_cr1_limb_heightmap_target_cache", None)
                if cache is None or "plane_points_w" not in cache:
                    raise RuntimeError("Current limb height-map world grid is unavailable")
                overlay.update(
                    cache["plane_points_w"],
                    observations[HEIGHTMAP_KEY],
                    predictions,
                    observations[HEIGHTMAP_VALID_KEY] > 0.5,
                )
                overlay_mode = overlay.mode
            if window is not None:
                latest_dashboard = _dashboard(
                    output,
                    observations,
                    map_metrics,
                    env_id=args_cli.display_env,
                    total_step=total_step,
                    reference_frame=int(command_before["reference_frame"]),
                    overlay_mode=overlay_mode,
                )
                window.show(latest_dashboard)
                if args_cli.save_interval > 0 and total_step % args_cli.save_interval == 0:
                    cv2.imwrite(str(output_dir / f"dashboard_step_{total_step:06d}.png"), latest_dashboard)

            # Keyboard events are dispatched while Isaac renders the viewer
            # and dashboard.  If R arrived during that update, do not execute
            # one more action from the old episode; service the reset at the
            # top of the next loop iteration instead.
            if controller.reset:
                continue

            next_observations, rewards, dones, _ = env.step(actions)
            if guidance_markers is not None:
                guidance = env.unwrapped.command_manager.get_term("motion").reference_guidance
                visible = bool(guidance.diagnostics.get("valid", dones.new_zeros(dones.shape))[args_cli.display_env])
                guidance_markers.set_visibility(visible)
                if visible:
                    body = guidance.body_targets_w[args_cli.display_env]
                    points = torch.cat((body, body[guidance.feet]))
                    indices = torch.cat((torch.zeros(len(body), device=body.device, dtype=torch.long),
                                         torch.ones(2, device=body.device, dtype=torch.long)))
                    guidance_markers.visualize(points, marker_indices=indices)
            future = _future_metrics(output, next_observations, dones)
            evaluation.process_step(rewards, dones, env.unwrapped, terminal_frames, future)
            actor.reset(dones.to(dtype=torch.bool))
            total_step += 1
            if (
                args_cli.reset_at_step > 0
                and not automatic_reset_issued
                and total_step >= args_cli.reset_at_step
            ):
                controller.reset = True
                automatic_reset_issued = True
                print(
                    f"[VISION PLAY][CONTROL] automatic reset requested at step={total_step}",
                    flush=True,
                )
            if (
                args_cli.zero_command_at_step > 0
                and not automatic_zero_command_issued
                and total_step >= args_cli.zero_command_at_step
            ):
                controller.zero_command = True
                automatic_zero_command_issued = True
            if (
                args_cli.resume_command_at_step > 0
                and not automatic_resume_command_issued
                and total_step >= args_cli.resume_command_at_step
            ):
                controller.resume_command = True
                automatic_resume_command_issued = True

            env_id = args_cli.display_env
            depth_centered = observations[DEPTH_HISTORY_KEY][env_id, -1]
            depth_m = depth_centered + 1.0
            record: dict[str, object] = {
                "type": "step",
                "label": "DONE" if bool(dones[env_id]) else "PERIODIC",
                "total_step": total_step,
                "reward_after": float(rewards[env_id]),
                "done": bool(dones[env_id]),
                "terminations": _termination_values(env.unwrapped, env_id),
                "overlay_mode": overlay_mode,
                "policy_std_mean": float(actor.output_std.mean()),
                "action": {
                    "l2": float(torch.linalg.vector_norm(actions[env_id])),
                    "min": float(actions[env_id].min()),
                    "max": float(actions[env_id].max()),
                },
                "depth": {
                    "min_m": float(depth_m.min()),
                    "max_m": float(depth_m.max()),
                    "invalid_fraction": float((depth_m <= 0.0).float().mean()),
                    "centered_min": float(depth_centered.min()),
                    "centered_max": float(depth_centered.max()),
                },
                "command": command_before,
            }
            if policy_only:
                record.update({
                    "policy_only": True,
                    "recurrent_feature_l2": float(
                        torch.linalg.vector_norm(output.recurrent_feature[env_id])
                    ),
                })
            else:
                velocity_target = observations[VELOCITY_KEY][env_id]
                velocity_prediction = output.velocity[env_id]
                velocity_skill = velocity_metrics["velocity_skill"][env_id]
                dynamic_values = output.dynamic_latent[env_id]
                dynamic_record = {
                    "type": "deterministic" if dynamic_values.numel() else "none",
                    "abs_mean": float(dynamic_values.abs().mean()) if dynamic_values.numel() else 0.0,
                    "abs_max": float(dynamic_values.abs().max()) if dynamic_values.numel() else 0.0,
                }
                if output.dynamic_mu is not None and output.dynamic_logvar is not None:
                    dynamic_std = torch.exp(0.5 * output.dynamic_logvar[env_id])
                    dynamic_record = {
                        "type": "vae",
                        "abs_mean": float(output.dynamic_mu[env_id].abs().mean()),
                        "abs_max": float(output.dynamic_mu[env_id].abs().max()),
                        "logvar_mean": float(output.dynamic_logvar[env_id].mean()),
                        "std_mean": float(dynamic_std.mean()),
                    }
                record.update({
                    "velocity": {
                        "target": [float(value) for value in velocity_target],
                        "prediction": [float(value) for value in velocity_prediction],
                        "mse": float(velocity_metrics["velocity_mse"][env_id]),
                        "zero_mse": float(velocity_metrics["velocity_zero_mse"][env_id]),
                        "skill": float(velocity_skill) if torch.isfinite(velocity_skill) else None,
                    },
                    "dynamic": dynamic_record,
                    "latent_l2": float(torch.linalg.vector_norm(output.latent[env_id])),
                    "heightmaps": {
                        limb: {
                            "weighted_mse": float(map_metrics[f"{limb}_weighted_mse"][env_id]),
                            "rung_mse": float(map_metrics[f"{limb}_rung_mse"][env_id]),
                            "non_rung_mse": float(map_metrics[f"{limb}_non_rung_mse"][env_id]),
                            "rung_fraction": float(map_metrics[f"{limb}_rung_fraction"][env_id]),
                        }
                        for limb in LIMB_NAMES
                    },
                    "future": None
                    if bool(dones[env_id]) or not future
                    else {
                        name: float(value[env_id - int(dones[:env_id].sum())])
                        for name, value in future.items()
                    },
                })
            should_print = force_print or single_step or bool(dones[env_id]) or total_step % args_cli.print_interval == 0
            if should_print:
                record["label"] = "MANUAL" if force_print else record["label"]
                _print_frame(record)
                evaluation.write(record)
                force_print = False
            if bool(dones[env_id]):
                print(
                    f"[VISION PLAY][EPISODE] frame={int(terminal_frames[env_id])} "
                    f"causes={record['terminations']}",
                    flush=True,
                )

            observations = next_observations
            if args_cli.max_steps > 0 and total_step >= args_cli.max_steps:
                break
            if args_cli.real_time:
                remaining = env.unwrapped.step_dt - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if latest_dashboard is not None:
            cv2.imwrite(str(output_dir / "dashboard_final.png"), latest_dashboard)
        summary_path = evaluation.close()
        print(f"[VISION PLAY][SUMMARY] {summary_path}", flush=True)
        controller.close()
        if overlay is not None:
            overlay.close()
        if window is not None:
            window.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
