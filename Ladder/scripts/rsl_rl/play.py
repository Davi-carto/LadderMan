"""Play a LadderMan RSL-RL checkpoint."""

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


REPO_ROOT = Path(__file__).resolve().parents[3]
LADDER_SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(LADDER_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(LADDER_SOURCE_DIR))


parser = argparse.ArgumentParser(description="Play a LadderMan RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Ladder-Climb-CR1-v0", help="Name of the task.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to LadderMan climbing motion.npz.")
parser.add_argument(
    "--depth_update_hz", type=float, default=None,
    help="Override Warp depth refresh Hz (0: policy rate). Default: checkpoint camera timing.",
)
parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    default=None,
    help="Frozen teacher checkpoint required to construct a Hybrid DAgger policy.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
parser.add_argument(
    "--fast_mode",
    "--fast-mode",
    action="store_true",
    help=(
        "Minimize play-only diagnostics and policy export work. The task physics, "
        "rewards, terminations, randomization, and optional disturbances remain active."
    ),
)
parser.add_argument(
    "--performance_interval",
    type=int,
    default=250,
    help="Report fast-mode wall-clock and control-step performance every N steps.",
)
parser.add_argument(
    "--visual_diagnostics",
    action="store_true",
    help="Attach a diagnostic-only CR1 camera and show RGB, depth, and rung-plus-robot RFM windows.",
)
parser.add_argument(
    "--camera_width",
    type=int,
    default=640,
    help="Diagnostic camera image width in pixels.",
)
parser.add_argument(
    "--camera_height",
    type=int,
    default=480,
    help="Diagnostic camera image height in pixels.",
)
parser.add_argument(
    "--camera_output_dir",
    type=str,
    default=str(REPO_ROOT / "outputs" / "cr1_camera_diagnostics"),
    help="Directory for periodic diagnostic camera captures.",
)
parser.add_argument(
    "--camera_save_interval",
    type=int,
    default=200,
    help="Save diagnostic captures every N control steps; zero disables saving.",
)
parser.add_argument(
    "--camera_display_scale",
    type=float,
    default=1.0,
    help="Display scale applied to the diagnostic camera images.",
)
parser.add_argument(
    "--camera_no_show",
    action="store_true",
    help="Do not open diagnostic image windows; captures and terminal statistics remain enabled.",
)
parser.add_argument(
    "--camera_pos",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(0.0, 0.0, 0.0),
    help="Camera offset position from d435_head_link in meters.",
)
parser.add_argument(
    "--camera_rot",
    type=float,
    nargs=4,
    metavar=("W", "X", "Y", "Z"),
    default=(1.0, 0.0, 0.0, 0.0),
    help="Camera offset quaternion (w x y z) from d435_head_link.",
)
parser.add_argument(
    "--camera_convention",
    choices=("world", "ros", "opengl"),
    default="world",
    help="Convention used by --camera_pos and --camera_rot.",
)
parser.add_argument("--print_interval", type=int, default=10, help="Print diagnostics every N control frames.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many control steps; zero runs indefinitely.")
parser.add_argument("--force_stats", action="store_true", help="Collect contact-frame force quantiles during play.")
parser.add_argument("--force_stats_file", type=str, default=None, help="Optional JSON output path for force statistics.")
parser.add_argument(
    "--lower_body_jitter_stats",
    action="store_true",
    help="Collect read-only per-joint lower-body jitter statistics for environment 0.",
)
parser.add_argument(
    "--lower_body_jitter_stats_file",
    type=str,
    default=None,
    help="Optional JSON output path for lower-body jitter statistics.",
)
parser.add_argument(
    "--jitter_window_frames",
    type=int,
    default=25,
    help="Window length used to locate the worst lower-body jitter intervals.",
)
parser.add_argument(
    "--force_limits_file",
    type=str,
    default=None,
    help="Optional JSON soft/hard force limits used only for diagnostic exceedance statistics.",
)
parser.add_argument(
    "--force_stats_warmup_steps",
    type=int,
    default=5,
    help="Ignore this many control steps after each reset when collecting force statistics.",
)
parser.add_argument("--start_paused", action="store_true", help="Start paused for keyboard single-step playback.")
parser.add_argument(
    "--reference_frame",
    type=int,
    default=None,
    help="Use a fixed reset reference frame instead of adaptive reset sampling.",
)
parser.add_argument(
    "--uniform_reset_sampling",
    action="store_true",
    help="Sample every reset uniformly from the motion file's reset-valid frames.",
)
parser.add_argument(
    "--disable_randomization",
    action="store_true",
    help="Disable domain/reset-state randomization and physics-step control delay for comparison.",
)
parser.add_argument(
    "--base_reset_noise_only",
    action="store_true",
    help="Disable physics/joint randomization but retain the configured base pose and velocity reset noise.",
)
parser.add_argument(
    "--disable_ladder_randomization",
    action="store_true",
    help="Use nominal ladder inclination and rung spacing while retaining other randomization.",
)
parser.add_argument(
    "--enable_disturbances",
    action="store_true",
    help="Enable the training-style interval root velocity pushes during play.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.force_stats_file is not None or args_cli.force_limits_file is not None:
    args_cli.force_stats = True
if args_cli.lower_body_jitter_stats_file is not None:
    args_cli.lower_body_jitter_stats = True

if args_cli.disable_randomization and args_cli.base_reset_noise_only:
    parser.error("--disable_randomization and --base_reset_noise_only are mutually exclusive")
if args_cli.disable_ladder_randomization and (
    args_cli.disable_randomization or args_cli.base_reset_noise_only
):
    parser.error(
        "--disable_ladder_randomization cannot be combined with "
        "--disable_randomization or --base_reset_noise_only"
    )
if args_cli.enable_disturbances and args_cli.disable_randomization:
    parser.error("--enable_disturbances cannot be combined with --disable_randomization")
if args_cli.print_interval <= 0:
    parser.error("--print_interval must be positive")
if args_cli.performance_interval <= 0:
    parser.error("--performance_interval must be positive")
if args_cli.max_steps < 0:
    parser.error("--max_steps must be nonnegative")
if args_cli.force_stats_warmup_steps < 0:
    parser.error("--force_stats_warmup_steps must be nonnegative")
if args_cli.jitter_window_frames <= 0:
    parser.error("--jitter_window_frames must be positive")
if args_cli.camera_save_interval < 0:
    parser.error("--camera_save_interval must be non-negative")
if args_cli.camera_width <= 0 or args_cli.camera_height <= 0:
    parser.error("--camera_width and --camera_height must be positive")
if args_cli.camera_display_scale <= 0.0:
    parser.error("--camera_display_scale must be positive")
if args_cli.start_paused and args_cli.headless:
    parser.error("--start_paused requires an interactive viewer")
if args_cli.reference_frame is not None and args_cli.uniform_reset_sampling:
    parser.error("--reference_frame and --uniform_reset_sampling are mutually exclusive")
if args_cli.visual_diagnostics and args_cli.headless and not args_cli.camera_no_show:
    parser.error("--visual_diagnostics requires a GUI; use --camera_no_show for capture-only diagnostics")
if args_cli.fast_mode and (args_cli.video or args_cli.visual_diagnostics or args_cli.force_stats):
    parser.error("--fast_mode cannot be combined with video, visual diagnostics, or force statistics")
if "/log/level" not in args_cli.kit_args:
    args_cli.kit_args = f"{args_cli.kit_args} --/log/level=error".strip()

if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.task.endswith("-Visual-v0"):
    args_cli.enable_cameras = True
if args_cli.visual_diagnostics:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import omni.timeline
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from ladder.utils.depth_camera_config import configure_depth_camera_timing

import ladder.tasks  # noqa: F401
from rsl_rl_compat import (
    convert_legacy_distillation_runner_cfg,
    convert_legacy_runner_cfg,
    convert_split_actor_checkpoint_to_actor_module_state_dict,
    is_split_actor_critic_checkpoint,
    load_checkpoint_payload,
)
from ladder.tasks.ladder_climb.mdp.rewards import whole_robot_rail_force_max
from ladder.utils.camera_diagnostics import (
    CAMERA_DIAGNOSTIC_NAME,
    CameraDiagnosticSettings,
    CameraDiagnostics,
    attach_cr1_camera_diagnostic,
)
from ladder.utils.contact_force import filtered_surface_force_max, target_rung_forces


carb.logging.acquire_logging().set_level_threshold(carb.logging.LEVEL_ERROR)
logging.disable(logging.WARNING)


PHYSICAL_CONTACT_BODY_NAMES = {
    "left_hand": "left_sphere_hand_link",
    "right_hand": "right_sphere_hand_link",
    "left_foot": "left_ankle_x_link",
    "right_foot": "right_ankle_x_link",
}
CONTACT_FORCE_THRESHOLD_N = 5.0
HAND_SURFACE_REWARD_TERMS = {
    "left_hand": {
        "step_front": "left_hand_step_front_contact",
        "step_underside": "left_hand_step_underside_contact",
        "side_board": "left_hand_side_board_contact",
        "ground": "left_hand_ground_contact",
        "missing_support": "left_hand_missing_required_step_contact",
    },
    "right_hand": {
        "step_front": "right_hand_step_front_contact",
        "step_underside": "right_hand_step_underside_contact",
        "side_board": "right_hand_side_board_contact",
        "ground": "right_hand_ground_contact",
        "missing_support": "right_hand_missing_required_step_contact",
    },
}


def _whole_robot_rail_sensor_names(env) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in env.scene.sensors
            if name.startswith("rail_contacts_")
        )
    )


class PlayController:
    """Keyboard state for interactive policy playback."""

    def __init__(self, *, paused: bool, enabled: bool):
        self.paused = paused
        self._step_requested = False
        self._reset_requested = False
        self._info_requested = False
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._timeline = omni.timeline.get_timeline_interface()

        if enabled:
            from omni import appwindow

            app_window = appwindow.get_default_app_window()
            if app_window is not None:
                self._input = carb.input.acquire_input_interface()
                self._keyboard = app_window.get_keyboard()
                self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                    self._keyboard, self._on_keyboard_event
                )

    def close(self) -> None:
        if self._keyboard_sub is not None:
            unsubscribe = getattr(self._input, "unsubscribe_to_keyboard_events", None)
            if unsubscribe is None:
                unsubscribe = self._input.unsubscribe_from_keyboard_events
            unsubscribe(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def consume_step_request(self) -> bool:
        requested = self._step_requested
        self._step_requested = False
        return requested

    def consume_reset_request(self) -> bool:
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def consume_info_request(self) -> bool:
        requested = self._info_requested
        self._info_requested = False
        return requested

    def ensure_timeline_playing(self) -> None:
        if not self._timeline.is_playing():
            self._timeline.play()

    def _on_keyboard_event(self, event, *args) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        key = event.input.name
        if key == "P":
            self.paused = not self.paused
            if not self.paused:
                self.ensure_timeline_playing()
            state = "PAUSED" if self.paused else "PLAYING"
            print(f"[CONTROL] {state}", flush=True)
        elif key == "N":
            if self.paused:
                self.ensure_timeline_playing()
                self._step_requested = True
        elif key == "R":
            self._reset_requested = True
        elif key == "I":
            self._info_requested = True
        return True


class LowerBodyJitterStatsAccumulator:
    """Collect read-only control-rate lower-body motion statistics for env 0."""

    _JOINT_TOKENS = ("_hip_", "_knee_", "_ankle_")

    def __init__(
        self,
        env,
        *,
        checkpoint: str,
        motion_file: str,
        output_path: str,
        window_frames: int,
    ):
        self._env = env
        self._command = env.command_manager.get_term("motion")
        self._robot = env.scene["robot"]
        self._action_term = env.action_manager._terms.get("joint_pos")
        if self._action_term is None:
            raise RuntimeError("lower-body jitter statistics require the joint_pos action term")

        self._joint_names = [
            name
            for name in self._robot.joint_names
            if any(token in name for token in self._JOINT_TOKENS)
        ]
        if not self._joint_names:
            raise RuntimeError("no hip, knee, or ankle joints were found")
        robot_name_to_index = {
            name: index for index, name in enumerate(self._robot.joint_names)
        }
        action_name_to_index = {
            name: index for index, name in enumerate(self._action_term._joint_names)
        }
        missing_actions = [
            name for name in self._joint_names if name not in action_name_to_index
        ]
        if missing_actions:
            raise RuntimeError(
                "lower-body joints are missing from joint_pos actions: "
                f"{missing_actions}"
            )
        self._robot_joint_ids = [
            robot_name_to_index[name] for name in self._joint_names
        ]
        self._action_joint_ids = [
            action_name_to_index[name] for name in self._joint_names
        ]
        self._dt = float(env.step_dt)
        self._checkpoint = os.path.abspath(checkpoint)
        self._motion_file = os.path.abspath(motion_file)
        self._output_path = os.path.abspath(output_path)
        output_root, _ = os.path.splitext(self._output_path)
        self._trace_output_path = f"{output_root}_per_frame.csv"
        self._window_frames = int(window_frames)
        self._reference_frames: list[int] = []
        self._action_source_frames: list[int] = []
        self._signals: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "actual_position_rad",
                "reference_position_rad",
                "position_error_rad",
                "actual_velocity_rad_s",
                "reference_velocity_rad_s",
                "velocity_error_rad_s",
                "actual_acceleration_rad_s2",
                "reference_acceleration_rad_s2",
                "acceleration_error_rad_s2",
                "position_error_delta_rad_per_step",
                "raw_action",
                "action_delta_per_step",
                "processed_target_rad",
                "processed_target_delta_rad_per_step",
            )
        }
        self._previous_actual_velocity: torch.Tensor | None = None
        self._previous_reference_velocity: torch.Tensor | None = None
        self._previous_velocity_error: torch.Tensor | None = None
        self._previous_position_error: torch.Tensor | None = None
        self._previous_raw_action: torch.Tensor | None = None
        self._previous_processed_target: torch.Tensor | None = None
        self._finished = False
        self._finish_reason = ""

        if env.num_envs != 1:
            print(
                "[PLAY][JITTER] num_envs is not 1; statistics use environment 0 only.",
                flush=True,
            )

    @staticmethod
    def _nan_like(value: torch.Tensor) -> torch.Tensor:
        return torch.full_like(value, float("nan"))

    def record(self, *, action_source_frame: int | None = None) -> None:
        if self._finished:
            return
        joint_ids = self._robot_joint_ids
        action_ids = self._action_joint_ids
        actual_position = self._robot.data.joint_pos[0, joint_ids].detach()
        reference_position = self._command.joint_pos[0, joint_ids].detach()
        actual_velocity = self._robot.data.joint_vel[0, joint_ids].detach()
        reference_velocity = self._command.joint_vel[0, joint_ids].detach()
        position_error = actual_position - reference_position
        velocity_error = actual_velocity - reference_velocity
        raw_action = self._action_term.raw_actions[0, action_ids].detach()
        processed_target = self._action_term.processed_actions[0, action_ids].detach()

        if self._previous_actual_velocity is None:
            actual_acceleration = self._nan_like(actual_velocity)
            reference_acceleration = self._nan_like(reference_velocity)
            acceleration_error = self._nan_like(velocity_error)
            position_error_delta = self._nan_like(position_error)
        else:
            actual_acceleration = (
                actual_velocity - self._previous_actual_velocity
            ) / self._dt
            reference_acceleration = (
                reference_velocity - self._previous_reference_velocity
            ) / self._dt
            acceleration_error = (
                velocity_error - self._previous_velocity_error
            ) / self._dt
            position_error_delta = position_error - self._previous_position_error

        if self._previous_raw_action is None:
            action_delta = self._nan_like(raw_action)
            processed_target_delta = self._nan_like(processed_target)
        else:
            action_delta = raw_action - self._previous_raw_action
            processed_target_delta = (
                processed_target - self._previous_processed_target
            )

        values = {
            "actual_position_rad": actual_position,
            "reference_position_rad": reference_position,
            "position_error_rad": position_error,
            "actual_velocity_rad_s": actual_velocity,
            "reference_velocity_rad_s": reference_velocity,
            "velocity_error_rad_s": velocity_error,
            "actual_acceleration_rad_s2": actual_acceleration,
            "reference_acceleration_rad_s2": reference_acceleration,
            "acceleration_error_rad_s2": acceleration_error,
            "position_error_delta_rad_per_step": position_error_delta,
            "raw_action": raw_action,
            "action_delta_per_step": action_delta,
            "processed_target_rad": processed_target,
            "processed_target_delta_rad_per_step": processed_target_delta,
        }
        for name, value in values.items():
            self._signals[name].append(value.to(device="cpu", dtype=torch.float32))
        self._reference_frames.append(int(self._command.time_steps[0]))
        self._action_source_frames.append(
            -1 if action_source_frame is None else int(action_source_frame)
        )

        self._previous_actual_velocity = actual_velocity.clone()
        self._previous_reference_velocity = reference_velocity.clone()
        self._previous_velocity_error = velocity_error.clone()
        self._previous_position_error = position_error.clone()
        self._previous_raw_action = raw_action.clone()
        self._previous_processed_target = processed_target.clone()

    @staticmethod
    def _signal_summary(
        values: torch.Tensor, reference_frames: torch.Tensor
    ) -> dict[str, float | int]:
        finite = torch.isfinite(values)
        if not torch.any(finite):
            return {"count": 0}
        valid_values = values[finite]
        absolute = valid_values.abs()
        finite_indexes = torch.nonzero(finite, as_tuple=False).flatten()
        worst_valid_index = int(torch.argmax(absolute))
        worst_index = int(finite_indexes[worst_valid_index])
        return {
            "count": int(valid_values.numel()),
            "mean": float(valid_values.mean()),
            "mean_abs": float(absolute.mean()),
            "rms": float(torch.sqrt(torch.mean(valid_values.square()))),
            "p95_abs": float(torch.quantile(absolute, 0.95)),
            "max_abs": float(absolute[worst_valid_index]),
            "min": float(valid_values.min()),
            "max": float(valid_values.max()),
            "worst_reference_frame": int(reference_frames[worst_index]),
        }

    @staticmethod
    def _window_rms(values: torch.Tensor, window_frames: int) -> torch.Tensor:
        finite = torch.isfinite(values)
        squared = torch.where(finite, values.square(), torch.zeros_like(values))
        counts = finite.to(dtype=torch.float32)
        squared_prefix = torch.cat((torch.zeros(1), torch.cumsum(squared, dim=0)))
        count_prefix = torch.cat((torch.zeros(1), torch.cumsum(counts, dim=0)))
        squared_sum = squared_prefix[window_frames:] - squared_prefix[:-window_frames]
        count = count_prefix[window_frames:] - count_prefix[:-window_frames]
        return torch.sqrt(squared_sum / count.clamp_min(1.0))

    @classmethod
    def _worst_joint_interval(
        cls,
        values: torch.Tensor,
        reference_frames: torch.Tensor,
        window_frames: int,
    ) -> dict[str, float | int]:
        effective_window = min(window_frames, int(values.shape[0]))
        if effective_window <= 0:
            return {}
        scores = cls._window_rms(values, effective_window)
        if scores.numel() == 0:
            return {}
        start = int(torch.argmax(scores))
        end = start + effective_window - 1
        return {
            "start_reference_frame": int(reference_frames[start]),
            "end_reference_frame": int(reference_frames[end]),
            "window_frames": effective_window,
            "rms": float(scores[start]),
        }

    @classmethod
    def _worst_overall_intervals(
        cls,
        acceleration_error: torch.Tensor,
        reference_frames: torch.Tensor,
        window_frames: int,
        count: int = 3,
    ) -> list[dict[str, float | int]]:
        effective_window = min(window_frames, int(acceleration_error.shape[0]))
        if effective_window <= 0:
            return []
        finite = torch.isfinite(acceleration_error)
        per_frame_energy = torch.where(
            finite,
            acceleration_error.square(),
            torch.zeros_like(acceleration_error),
        ).sum(dim=-1) / finite.sum(dim=-1).clamp_min(1)
        scores = cls._window_rms(torch.sqrt(per_frame_energy), effective_window)
        selected: list[dict[str, float | int]] = []
        selected_indexes: list[tuple[int, int]] = []
        for candidate in torch.argsort(scores, descending=True).tolist():
            start = int(candidate)
            end = start + effective_window - 1
            if any(start <= other_end and end >= other_start for other_start, other_end in selected_indexes):
                continue
            selected_indexes.append((start, end))
            selected.append(
                {
                    "start_reference_frame": int(reference_frames[start]),
                    "end_reference_frame": int(reference_frames[end]),
                    "window_frames": effective_window,
                    "acceleration_error_rms_rad_s2": float(scores[start]),
                }
            )
            if len(selected) >= count:
                break
        return selected

    def report(self) -> dict[str, object]:
        if not self._reference_frames:
            return {
                "metadata": {
                    "checkpoint": self._checkpoint,
                    "motion_file": self._motion_file,
                    "finish_reason": self._finish_reason,
                    "sample_count": 0,
                }
            }
        reference_frames = torch.tensor(self._reference_frames, dtype=torch.long)
        signals = {
            name: torch.stack(values, dim=0)
            for name, values in self._signals.items()
        }
        per_joint: dict[str, object] = {}
        for joint_index, joint_name in enumerate(self._joint_names):
            joint_signals = {
                name: self._signal_summary(value[:, joint_index], reference_frames)
                for name, value in signals.items()
            }
            joint_signals["worst_acceleration_error_interval"] = (
                self._worst_joint_interval(
                    signals["acceleration_error_rad_s2"][:, joint_index],
                    reference_frames,
                    self._window_frames,
                )
            )
            per_joint[joint_name] = joint_signals

        aggregate = {}
        repeated_frames = reference_frames.repeat_interleave(len(self._joint_names))
        for name, value in signals.items():
            aggregate[name] = self._signal_summary(value.flatten(), repeated_frames)
        return {
            "metadata": {
                "checkpoint": self._checkpoint,
                "motion_file": self._motion_file,
                "finish_reason": self._finish_reason,
                "dt_s": self._dt,
                "control_hz": 1.0 / self._dt,
                "sample_count": len(self._reference_frames),
                "start_reference_frame": self._reference_frames[0],
                "end_reference_frame": self._reference_frames[-1],
                "action_source_start_frame": self._action_source_frames[0],
                "action_source_end_frame": self._action_source_frames[-1],
                "joint_names": self._joint_names,
                "window_frames": self._window_frames,
                "acceleration_definition": "control-rate finite difference of joint velocity",
                "per_frame_trace_file": self._trace_output_path,
            },
            "aggregate": aggregate,
            "per_joint": per_joint,
            "worst_overall_intervals": self._worst_overall_intervals(
                signals["acceleration_error_rad_s2"],
                reference_frames,
                self._window_frames,
            ),
        }

    def _write_per_frame_trace(self) -> None:
        signal_names = tuple(self._signals)
        signals = {
            name: torch.stack(values, dim=0)
            for name, values in self._signals.items()
        }
        with open(self._trace_output_path, "w", encoding="utf-8", newline="") as trace_file:
            writer = csv.writer(trace_file)
            writer.writerow(
                (
                    "reference_frame",
                    "action_source_frame",
                    "joint_name",
                    *signal_names,
                )
            )
            for frame_index, reference_frame in enumerate(self._reference_frames):
                for joint_index, joint_name in enumerate(self._joint_names):
                    writer.writerow(
                        (
                            reference_frame,
                            self._action_source_frames[frame_index],
                            joint_name,
                            *(
                                float(signals[name][frame_index, joint_index])
                                for name in signal_names
                            ),
                        )
                    )

    def finish(self, *, reason: str) -> None:
        if self._finished:
            return
        self._finished = True
        self._finish_reason = reason
        report = self.report()
        metadata = report["metadata"]
        print(
            "\n[PLAY][JITTER] "
            f"samples={metadata['sample_count']} "
            f"reference_frames={metadata.get('start_reference_frame', -1)}.."
            f"{metadata.get('end_reference_frame', -1)} reason={reason}",
            flush=True,
        )
        if "aggregate" in report:
            aggregate = report["aggregate"]
            print(
                "[PLAY][JITTER][AGGREGATE] "
                f"position_error_rms={aggregate['position_error_rad']['rms']:.5f}rad "
                f"velocity_error_rms={aggregate['velocity_error_rad_s']['rms']:.5f}rad/s "
                f"acceleration_error_p95={aggregate['acceleration_error_rad_s2']['p95_abs']:.3f}rad/s^2 "
                f"action_delta_p95={aggregate['action_delta_per_step']['p95_abs']:.5f}/step "
                f"target_delta_p95={aggregate['processed_target_delta_rad_per_step']['p95_abs']:.5f}rad/step",
                flush=True,
            )
            for joint_name, values in report["per_joint"].items():
                print(
                    f"[PLAY][JITTER][JOINT] {joint_name} "
                    f"pos_err_rms={values['position_error_rad']['rms']:.5f}rad "
                    f"vel_err_rms={values['velocity_error_rad_s']['rms']:.5f}rad/s "
                    f"acc_err_p95={values['acceleration_error_rad_s2']['p95_abs']:.3f}rad/s^2 "
                    f"action_delta_p95={values['action_delta_per_step']['p95_abs']:.5f}/step",
                    flush=True,
                )
            for interval in report["worst_overall_intervals"]:
                print(
                    "[PLAY][JITTER][INTERVAL] "
                    f"frames={interval['start_reference_frame']}..{interval['end_reference_frame']} "
                    f"acceleration_error_rms={interval['acceleration_error_rms_rad_s2']:.3f}rad/s^2",
                    flush=True,
                )

        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        with open(self._output_path, "w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, sort_keys=True)
        self._write_per_frame_trace()
        print(f"[PLAY][JITTER] wrote {self._output_path}", flush=True)
        print(f"[PLAY][JITTER] wrote {self._trace_output_path}", flush=True)


def _observations_only(result):
    return result[0] if isinstance(result, tuple) else result


def _format_vector(value: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(item):+.4f}" for item in value) + "]"


def _contact_force_diagnostics(env) -> dict[str, dict[str, float | torch.Tensor]]:
    sensor = env.scene.sensors["contact_forces"]
    current_forces = sensor.data.net_forces_w[0]
    history = sensor.data.net_forces_w_history[0]
    history_norms = torch.linalg.norm(history, dim=-1)
    peak_norms = torch.max(history_norms, dim=0).values
    body_to_index = {name: index for index, name in enumerate(sensor.body_names)}
    return {
        limb_name: {
            "current_w": current_forces[body_to_index[body_name]],
            "current_norm": float(torch.linalg.norm(current_forces[body_to_index[body_name]])),
            "peak_norm": float(peak_norms[body_to_index[body_name]]),
        }
        for limb_name, body_name in PHYSICAL_CONTACT_BODY_NAMES.items()
        if body_name in body_to_index
    }


def _filtered_force_diagnostic(env, sensor_name: str) -> dict[str, float | int | torch.Tensor]:
    sensor = env.scene.sensors[sensor_name]
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None or force_matrix_w.numel() == 0:
        return {
            "max_norm": 0.0,
            "max_force_w": torch.zeros(3, device=env.device),
            "filter_count": 0,
        }
    forces_w = force_matrix_w[0].reshape(-1, 3)
    norms = torch.linalg.vector_norm(forces_w, dim=-1)
    max_index = int(torch.argmax(norms))
    return {
        "max_norm": float(norms[max_index]),
        "max_force_w": forces_w[max_index],
        "filter_count": force_matrix_w.shape[2],
    }


def _rung_force_diagnostic(env, sensor_name: str, rung_id: int) -> dict[str, object]:
    sensor = env.scene.sensors[sensor_name]
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None or force_matrix_w.shape[2] == 0:
        zero = torch.zeros(3, device=env.device)
        return {"target_force_w": zero, "target_norm": 0.0, "max_rung": -1, "max_norm": 0.0}
    per_rung_force_w = force_matrix_w[0].sum(dim=0)
    per_rung_norm = torch.linalg.vector_norm(per_rung_force_w, dim=-1)
    max_rung = int(torch.argmax(per_rung_norm))
    if 0 <= rung_id < per_rung_force_w.shape[0]:
        target_force_w = per_rung_force_w[rung_id]
    else:
        target_force_w = torch.zeros(3, device=env.device)
    return {
        "target_force_w": target_force_w,
        "target_norm": float(torch.linalg.vector_norm(target_force_w)),
        "max_rung": max_rung,
        "max_norm": float(per_rung_norm[max_rung]),
    }


def _load_force_limits(path: str | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    resolved = os.path.abspath(path)
    with open(resolved, encoding="utf-8") as input_file:
        raw_limits = json.load(input_file)
    if not isinstance(raw_limits, dict):
        raise ValueError("force limits JSON must be an object keyed by force metric name")

    limits = {}
    for name, values in raw_limits.items():
        if not isinstance(name, str) or not isinstance(values, dict):
            raise ValueError("each force limit must be an object with numeric soft and hard values")
        if set(values) != {"soft", "hard"}:
            raise ValueError(f"force limit {name!r} must contain exactly 'soft' and 'hard'")
        soft = float(values["soft"])
        hard = float(values["hard"])
        if not math.isfinite(soft) or not math.isfinite(hard) or soft < 0.0 or hard <= soft:
            raise ValueError(
                f"force limit {name!r} must satisfy finite 0 <= soft < hard, got {values}"
            )
        limits[name] = {"soft": soft, "hard": hard}
    return limits


class ForceStatsAccumulator:
    """Collect run-wide and per-episode force distributions for play diagnostics."""

    def __init__(self, env, *, force_limits: dict[str, dict[str, float]], warmup_steps: int):
        self._env = env
        self._force_limits = force_limits
        self._warmup_steps = warmup_steps
        self._samples: dict[str, list[float]] = {}
        self._episode_samples: list[dict[str, list[float]]] = [
            {} for _ in range(env.num_envs)
        ]
        self._episode_steps = torch.zeros(env.num_envs, dtype=torch.long)
        self._episode_eligible: dict[str, torch.Tensor] = {}
        self._episode_over_soft: dict[str, torch.Tensor] = {}
        self._episode_hard_events: dict[str, torch.Tensor] = {}
        self._previous_over_hard: dict[str, torch.Tensor] = {}
        self._episodes: list[dict[str, object]] = []
        self._episode_indexes = [0] * env.num_envs
        self._previous_foot_step_force: dict[str, torch.Tensor] = {}

    @staticmethod
    def _limit_key(name: str) -> str:
        for side in ("left_", "right_"):
            if name.startswith(side):
                return name[len(side) :]
        return name

    def _limit(self, name: str) -> dict[str, float] | None:
        return self._force_limits.get(name, self._force_limits.get(self._limit_key(name)))

    def _append(self, name: str, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        values = values.detach()
        eligible = self._env.episode_length_buf >= self._warmup_steps
        if mask is not None:
            eligible &= mask
        eligible &= torch.isfinite(values)

        contact = eligible & (values > CONTACT_FORCE_THRESHOLD_N)
        contact_env_ids = torch.nonzero(contact, as_tuple=False).flatten()
        if contact_env_ids.numel() > 0:
            contact_values = values[contact_env_ids].cpu().tolist()
            self._samples.setdefault(name, []).extend(contact_values)
            for env_id, value in zip(contact_env_ids.cpu().tolist(), contact_values):
                self._episode_samples[env_id].setdefault(name, []).append(value)

        limit = self._limit(name)
        if limit is None:
            return
        eligible_cpu = eligible.cpu()
        values_cpu = values.cpu()
        if name not in self._episode_eligible:
            self._episode_eligible[name] = torch.zeros(self._env.num_envs, dtype=torch.long)
            self._episode_over_soft[name] = torch.zeros(self._env.num_envs, dtype=torch.long)
            self._episode_hard_events[name] = torch.zeros(self._env.num_envs, dtype=torch.long)
            self._previous_over_hard[name] = torch.zeros(self._env.num_envs, dtype=torch.bool)
        self._episode_eligible[name] += eligible_cpu.long()
        self._episode_over_soft[name] += (eligible_cpu & (values_cpu > limit["soft"])).long()
        over_hard = eligible_cpu & (values_cpu > limit["hard"])
        self._episode_hard_events[name] += (
            over_hard & ~self._previous_over_hard[name]
        ).long()
        self._previous_over_hard[name] = over_hard

    def update(self) -> None:
        command = self._env.command_manager.get_term("motion")
        self._episode_steps += (
            self._env.episode_length_buf >= self._warmup_steps
        ).detach().cpu().long()
        foot_step_forces: dict[str, torch.Tensor] = {}
        foot_required: dict[str, torch.Tensor] = {}

        for side, limb_name in (("left", "left_hand"), ("right", "right_hand")):
            sensor_name = f"{side}_hand_step_contacts"
            if sensor_name not in self._env.scene.sensors:
                continue
            forces_w = self._env.scene.sensors[sensor_name].data.force_matrix_w
            if forces_w is None:
                continue
            hand_surface_forces = []
            for surface in ("top", "front", "underside"):
                surface_force = filtered_surface_force_max(
                    forces_w,
                    surface=surface,
                    min_top_normal_fraction=(
                        command.cfg.contact_front_normal_fraction
                        if surface == "front"
                        else 0.5
                    ),
                )
                self._append(
                    f"{side}_hand_{surface}",
                    surface_force,
                )
                hand_surface_forces.append(surface_force)
            limb_index = command.limb_names.index(limb_name)
            rung_ids = command.contact_rung_id[:, limb_index : limb_index + 1]
            required = command.contact_indicator[:, limb_index] > 0.5
            target_forces_w = target_rung_forces(forces_w, rung_ids)[:, 0]
            target_upward = torch.clamp_min(target_forces_w[:, 2], 0.0)
            self._append(f"{side}_hand_target_rung_upward", target_upward, required)

            force_norm = torch.linalg.vector_norm(forces_w[:, 0], dim=-1)
            filter_ids = torch.arange(force_norm.shape[1], device=force_norm.device).unsqueeze(0)
            valid_target = (rung_ids[:, 0] >= 0) & (rung_ids[:, 0] < force_norm.shape[1])
            wrong_mask = filter_ids != rung_ids[:, 0].unsqueeze(-1)
            wrong_force = torch.where(wrong_mask, force_norm, torch.zeros_like(force_norm)).amax(dim=-1)
            self._append(
                f"{side}_hand_wrong_rung",
                wrong_force,
                required & valid_target,
            )

            for surface_name, filtered_sensor_name in (
                ("side_board", f"{side}_hand_side_board_contacts"),
                ("ground", f"{side}_hand_ground_contacts"),
            ):
                if filtered_sensor_name not in self._env.scene.sensors:
                    continue
                surface_forces_w = self._env.scene.sensors[filtered_sensor_name].data.force_matrix_w
                if surface_forces_w is not None:
                    surface_force = filtered_surface_force_max(
                        surface_forces_w, surface="all"
                    )
                    self._append(
                        f"{side}_hand_{surface_name}",
                        surface_force,
                    )
                    hand_surface_forces.append(surface_force)
            self._append(
                f"{side}_hand_force",
                torch.stack(hand_surface_forces, dim=-1).amax(dim=-1),
            )

        rail_sensor_names = _whole_robot_rail_sensor_names(self._env)
        if rail_sensor_names:
            rail_force_max = whole_robot_rail_force_max(
                self._env, rail_sensor_names, expected_rail_count=2
            )
            self._append("left_rail_any_body", rail_force_max[:, 0])
            self._append("right_rail_any_body", rail_force_max[:, 1])

        for side, limb_name in (("left", "left_foot"), ("right", "right_foot")):
            sensor_name = f"{side}_foot_step_contacts"
            if sensor_name not in self._env.scene.sensors:
                continue
            forces_w = self._env.scene.sensors[sensor_name].data.force_matrix_w
            if forces_w is None:
                continue
            step_force = filtered_surface_force_max(forces_w, surface="all")
            foot_step_forces[side] = step_force
            limb_index = command.limb_names.index(limb_name)
            required = command.contact_indicator[:, limb_index] > 0.5
            foot_required[side] = required
            self._append(f"{side}_foot_step", step_force)
            target_forces_w = target_rung_forces(
                forces_w,
                command.contact_rung_id[:, limb_index : limb_index + 1],
            )[:, 0]
            self._append(
                f"{side}_foot_target_rung",
                torch.linalg.vector_norm(target_forces_w, dim=-1),
                required,
            )
            previous = self._previous_foot_step_force.get(side)
            if previous is not None and previous.shape == step_force.shape:
                contact_began = (previous <= CONTACT_FORCE_THRESHOLD_N) & (
                    step_force > CONTACT_FORCE_THRESHOLD_N
                )
                self._append(f"{side}_foot_contact_onset", step_force, contact_began)
            self._previous_foot_step_force[side] = step_force.detach().clone()

        if set(foot_step_forces) == {"left", "right"}:
            left_contact = foot_step_forces["left"] > CONTACT_FORCE_THRESHOLD_N
            right_contact = foot_step_forces["right"] > CONTACT_FORCE_THRESHOLD_N
            self._append(
                "left_foot_single_support",
                foot_step_forces["left"],
                left_contact & ~right_contact & foot_required["left"],
            )
            self._append(
                "right_foot_single_support",
                foot_step_forces["right"],
                right_contact & ~left_contact & foot_required["right"],
            )

    @staticmethod
    def _sample_summary(samples: list[float]) -> dict[str, float | int]:
        values = torch.tensor(samples, dtype=torch.float32)
        return {
            "count": int(values.numel()),
            "mean": float(values.mean()),
            "p50": float(torch.quantile(values, 0.50)),
            "p95": float(torch.quantile(values, 0.95)),
            "max": float(values.max()),
        }

    def finish_episodes(self, dones: torch.Tensor) -> None:
        for env_id in torch.nonzero(dones.detach().cpu(), as_tuple=False).flatten().tolist():
            self._finish_episode(env_id, complete=True)

    def _finish_episode(self, env_id: int, *, complete: bool) -> None:
        if self._episode_steps[env_id] == 0 and not self._episode_samples[env_id]:
            self._reset_episode_state(env_id)
            return
        stats = {
            name: self._sample_summary(samples)
            for name, samples in sorted(self._episode_samples[env_id].items())
            if samples
        }
        for name, eligible_count in self._episode_eligible.items():
            count = int(eligible_count[env_id])
            if count == 0:
                continue
            stats.setdefault(name, {})["over_soft_fraction"] = (
                int(self._episode_over_soft[name][env_id]) / count
            )
            stats[name]["hard_collision_count"] = int(
                self._episode_hard_events[name][env_id]
            )
        self._episodes.append(
            {
                "env_id": env_id,
                "episode_index": self._episode_indexes[env_id],
                "complete": complete,
                "sampled_control_steps": int(self._episode_steps[env_id]),
                "metrics": stats,
            }
        )
        self._episode_indexes[env_id] += 1
        self._reset_episode_state(env_id)

    def _reset_episode_state(self, env_id: int) -> None:
        self._episode_samples[env_id] = {}
        self._episode_steps[env_id] = 0
        for values in self._episode_eligible.values():
            values[env_id] = 0
        for values in self._episode_over_soft.values():
            values[env_id] = 0
        for values in self._episode_hard_events.values():
            values[env_id] = 0
        for values in self._previous_over_hard.values():
            values[env_id] = False
        for side in ("left", "right"):
            previous = self._previous_foot_step_force.get(side)
            if previous is not None:
                previous[env_id] = 0.0

    def finish_active_episodes(self, *, complete: bool) -> None:
        for env_id in range(self._env.num_envs):
            self._finish_episode(env_id, complete=complete)

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            name: self._sample_summary(samples)
            for name, samples in sorted(self._samples.items())
            if samples
        }

    def print_summary(self) -> dict[str, dict[str, float | int]]:
        summary = self.summary()
        print("\n[PLAY][FORCE_STATS] contact frames only (>5 N)", flush=True)
        for name, values in summary.items():
            print(
                f"  {name}: count={values['count']} mean={values['mean']:.2f}N "
                f"p50={values['p50']:.2f}N p95={values['p95']:.2f}N "
                f"max={values['max']:.2f}N"
            )
        return summary

    def report(self) -> dict[str, object]:
        return {
            "metadata": {
                "contact_force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
                "reset_warmup_steps": self._warmup_steps,
                "force_limits": self._force_limits,
            },
            "aggregate_contact_frames": self.summary(),
            "episodes": self._episodes,
        }


def _hand_surface_diagnostics(env, limb_name: str) -> dict[str, dict[str, object]]:
    diagnostics = {}
    for surface_name, reward_name in HAND_SURFACE_REWARD_TERMS[limb_name].items():
        if reward_name not in env.reward_manager.active_terms:
            continue
        reward_cfg = env.reward_manager.get_term_cfg(reward_name)
        sensor_name = reward_cfg.params["sensor_cfg"].name
        if sensor_name not in env.scene.sensors:
            continue
        force = _filtered_force_diagnostic(env, sensor_name)
        diagnostic_params = {
            name: reward_cfg.params[name]
            for name in (
                "force_threshold",
                "min_top_normal_fraction",
                "target_distance_threshold",
                "grace_steps",
            )
            if name in reward_cfg.params
        }
        diagnostics[surface_name] = {
            **force,
            "sensor_name": sensor_name,
            **diagnostic_params,
            "penalty_triggered": bool(reward_cfg.func(env, **reward_cfg.params)[0] > 0.5),
        }
    return diagnostics


def _hand_surface_contact_state(env) -> tuple[bool, ...]:
    state = []
    for limb_name in HAND_SURFACE_REWARD_TERMS:
        diagnostics = _hand_surface_diagnostics(env, limb_name)
        state.extend(
            bool(surface["penalty_triggered"])
            for surface in diagnostics.values()
        )
    return tuple(state)


def _contact_flag(required: bool, error_m: float, force_n: float, target_threshold_m: float) -> str:
    in_contact = force_n > CONTACT_FORCE_THRESHOLD_N
    off_target = error_m > target_threshold_m
    if not required and in_contact:
        return "INACTIVE_PHASE_CONTACT"
    if required and in_contact and off_target:
        return "OFF_TARGET_CONTACT"
    if required and not in_contact and off_target:
        return "MISSED_TARGET"
    if required and not in_contact:
        return "NEAR_TARGET_NO_FORCE"
    if required:
        return "ON_TARGET_CONTACT"
    return "clear"


def _termination_values(env) -> dict[str, bool]:
    return {
        name: bool(values[0])
        for name, values in env.termination_manager.get_active_iterable_terms(env_idx=0)
    }


def _print_runtime_diagnostics(env, *, total_step: int, reward, actions, label: str) -> None:
    command = env.command_manager.get_term("motion")
    frame = int(command.time_steps[0])
    frame_total = command.motion.time_step_total
    episode_step = int(env.episode_length_buf[0])

    joint_abs_error = torch.abs(command.robot_joint_pos[0] - command.joint_pos[0])
    body_position_error = torch.linalg.norm(
        command.robot_body_pos_w[0] - command.body_pos_relative_w[0], dim=-1
    )
    body_orientation_error = quat_error_magnitude(command.robot_body_quat_w[0], command.body_quat_relative_w[0])
    anchor_position_error = torch.linalg.norm(command.robot_anchor_pos_w[0] - command.anchor_pos_w[0])
    anchor_orientation_error = quat_error_magnitude(command.robot_anchor_quat_w[0], command.anchor_quat_w[0])
    reference_anchor_speed = torch.linalg.norm(command.anchor_lin_vel_w[0])
    robot_anchor_speed = torch.linalg.norm(command.robot_anchor_lin_vel_w[0])
    worst_body_index = int(torch.argmax(body_position_error))
    force_diagnostics = _contact_force_diagnostics(env)

    reward_value = "n/a" if reward is None else f"{float(reward[0]):+.6f}"
    print(
        f"\n[PLAY][{label}] total_step={total_step} episode_step={episode_step} "
        f"reference_frame={frame}/{frame_total - 1} reward={reward_value}",
        flush=True,
    )
    if actions is not None:
        print(
            f"  action: l2={float(torch.linalg.norm(actions[0])):.4f} "
            f"min={float(actions[0].min()):+.4f} max={float(actions[0].max()):+.4f}"
        )
    print(
        f"  tracking: anchor_pos={float(anchor_position_error):.4f}m "
        f"anchor_ori={float(anchor_orientation_error):.4f}rad "
        f"anchor_speed_ref={float(reference_anchor_speed):.4f}m/s "
        f"anchor_speed_robot={float(robot_anchor_speed):.4f}m/s "
        f"joint_abs_mean={float(joint_abs_error.mean()):.4f}rad "
        f"joint_abs_max={float(joint_abs_error.max()):.4f}rad"
    )
    print(
        f"  body: pos_mean={float(body_position_error.mean()):.4f}m "
        f"pos_max={float(body_position_error.max()):.4f}m({command.cfg.body_names[worst_body_index]}) "
        f"ori_mean={float(body_orientation_error.mean()):.4f}rad "
        f"ori_max={float(body_orientation_error.max()):.4f}rad"
    )
    print(
        f"  ladder_sample: phi={math.degrees(float(command.ladder_phi[0])):.3f}deg "
        f"spacing={float(command.ladder_spacing[0]):.4f}m"
    )
    print(
        "  contact_metrics: "
        f"required_ref={float(command.metrics['required_contact_fraction'][0]):.3f} "
        f"required_physical={float(command.metrics['required_physical_contact_fraction'][0]):.3f} "
        f"required_target={float(command.metrics['required_target_contact_fraction'][0]):.3f} "
        f"swing_physical={float(command.metrics['swing_physical_contact_fraction'][0]):.3f}"
    )
    rail_sensor_names = _whole_robot_rail_sensor_names(env)
    if rail_sensor_names:
        rail_force_max = whole_robot_rail_force_max(
            env, rail_sensor_names, expected_rail_count=2
        )[0]
        print(
            f"  whole_robot_rail_contact: sensors={len(rail_sensor_names)} "
            f"left={float(rail_force_max[0]):.2f}N "
            f"right={float(rail_force_max[1]):.2f}N "
            f"force_gate={CONTACT_FORCE_THRESHOLD_N:.2f}N"
        )

    contact_errors = torch.linalg.norm(command.robot_limb_pos_w[0] - command.target_contact_pos_w[0], dim=-1)
    for limb_index, limb_name in enumerate(command.limb_names):
        required = bool(command.contact_indicator[0, limb_index] > 0.5)
        rung_id = int(command.contact_rung_id[0, limb_index])
        actual = command.robot_limb_pos_w[0, limb_index]
        target = command.target_contact_pos_w[0, limb_index]
        delta = actual - target
        error_m = float(contact_errors[limb_index])
        force = force_diagnostics.get(limb_name)
        current_force_n = 0.0 if force is None else float(force["current_norm"])
        peak_force_n = 0.0 if force is None else float(force["peak_norm"])
        current_force_w = torch.zeros(3, device=env.device) if force is None else force["current_w"]
        target_threshold_m = float(command._limb_target_distance_thresholds[limb_index])
        flag = _contact_flag(required, error_m, current_force_n, target_threshold_m)
        print(
            f"  {limb_name}: required={int(required)} rung={rung_id} "
            f"error={error_m:.4f}m target_gate={target_threshold_m:.4f}m "
            f"force_now={current_force_n:.2f}N force_peak6={peak_force_n:.2f}N flag={flag}"
        )
        print(
            f"    actual_w={_format_vector(actual)} target_w={_format_vector(target)} "
            f"delta_w={_format_vector(delta)} total_force_w={_format_vector(current_force_w)} "
            f"physical_body={PHYSICAL_CONTACT_BODY_NAMES.get(limb_name, 'unknown')}"
        )
        step_sensor_name = f"{limb_name}_step_contacts"
        if step_sensor_name in env.scene.sensors:
            rung_force = _rung_force_diagnostic(env, step_sensor_name, rung_id)
            print(
                f"    rung_filtered: target_rung={rung_id} "
                f"target_force={float(rung_force['target_norm']):.2f}N "
                f"target_force_w={_format_vector(rung_force['target_force_w'])} "
                f"max_rung={int(rung_force['max_rung'])} "
                f"max_force={float(rung_force['max_norm']):.2f}N"
            )
        surface = (
            _hand_surface_diagnostics(env, limb_name)
            if limb_name in HAND_SURFACE_REWARD_TERMS
            else {}
        )
        if surface:
            rung_index = min(max(rung_id, 0), command.ladder_geometry.num_rungs - 1)
            rung_position = command.rung_positions_w[0, rung_index]
            print(f"    target_minus_rung_w={_format_vector(target - rung_position)}")
            step = surface["step_front"]
            step_underside = surface["step_underside"]
            side_board = surface.get("side_board")
            ground = surface["ground"]
            missing_support = surface["missing_support"]
            step_force_norm = float(step["max_norm"])
            step_normal_z_fraction = (
                0.0
                if step_force_norm <= 1.0e-6
                else float(step["max_force_w"][2]) / step_force_norm
            )
            front_normal_fraction = float(step["min_top_normal_fraction"])
            underside_normal_fraction = float(
                step_underside["min_top_normal_fraction"]
            )
            if step_force_norm <= float(step["force_threshold"]):
                step_class = "CLEAR"
            elif step_normal_z_fraction <= -underside_normal_fraction:
                step_class = "UNDERSIDE"
            elif abs(step_normal_z_fraction) < front_normal_fraction:
                step_class = "FRONT_EDGE"
            elif step_normal_z_fraction >= 0.5:
                step_class = "TOP_ALLOWED"
            else:
                step_class = "TRANSITIONAL_NORMAL"
            print(
                f"    step: force={step_force_norm:.2f}N "
                f"force_w={_format_vector(step['max_force_w'])} "
                f"force_gate={float(step['force_threshold']):.2f}N "
                f"normal_z_fraction={step_normal_z_fraction:+.3f} class={step_class} "
                f"front_penalty={int(step['penalty_triggered'])} "
                f"underside_penalty={int(step_underside['penalty_triggered'])}"
            )
            step_sensor_name = f"{limb_name}_step_contacts"
            if step_sensor_name in env.scene.sensors:
                step_forces_w = env.scene.sensors[step_sensor_name].data.force_matrix_w
                if step_forces_w is not None:
                    print(
                        "    step_surface_force_max: "
                        f"top={float(filtered_surface_force_max(step_forces_w, surface='top')[0]):.2f}N "
                        f"front={float(filtered_surface_force_max(step_forces_w, surface='front', min_top_normal_fraction=front_normal_fraction)[0]):.2f}N "
                        f"underside={float(filtered_surface_force_max(step_forces_w, surface='underside')[0]):.2f}N "
                        f"all={float(filtered_surface_force_max(step_forces_w, surface='all')[0]):.2f}N"
                    )
            if side_board is None:
                print("    side_board: disabled (geometry not spawned)")
            else:
                print(
                    f"    side_board: force={float(side_board['max_norm']):.2f}N "
                    f"force_w={_format_vector(side_board['max_force_w'])} "
                    f"force_gate={float(side_board['force_threshold']):.2f}N "
                    f"penalty_triggered={int(side_board['penalty_triggered'])}"
                )
            print(
                f"    ground: force={float(ground['max_norm']):.2f}N "
                f"force_w={_format_vector(ground['max_force_w'])} "
                f"force_gate={float(ground['force_threshold']):.2f}N "
                f"penalty_triggered={int(ground['penalty_triggered'])}"
            )
            print(
                f"    missing_step_support: grace={int(missing_support['grace_steps'])}frames "
                f"distance_gate={float(missing_support['target_distance_threshold']):.4f}m "
                f"upward_force_gate={float(missing_support['force_threshold']):.2f}N "
                f"penalty_triggered={int(missing_support['penalty_triggered'])}"
            )

    reward_terms = {
        name: round(float(values[0]), 5)
        for name, values in env.reward_manager.get_active_iterable_terms(env_idx=0)
    }
    print(f"  weighted_reward_rate={reward_terms}")
    print(f"  termination_terms={_termination_values(env)}", flush=True)


def _print_control_delay(env, *, label: str) -> None:
    """Report environment 0's actual episode delay, including fast-mode resets."""
    action_term = env.action_manager.get_term("joint_pos")
    delay_buffer = getattr(action_term, "_action_delay_buffer", None)
    delay_steps = 0 if delay_buffer is None else int(delay_buffer.time_lags[0])
    command = env.command_manager.get_term("motion")
    print(
        f"[PLAY][DELAY] {label} env=0 reference_frame={int(command.time_steps[0])} "
        f"control_delay_physics_steps={delay_steps} "
        f"control_delay_ms={delay_steps * env.physics_dt * 1000.0:.1f}",
        flush=True,
    )


def _print_setup(env, obs, *, motion_file: str, checkpoint: str, agent_cfg) -> None:
    command = env.command_manager.get_term("motion")
    geometry = command.ladder_geometry
    if isinstance(obs, torch.Tensor):
        observation_shapes = {"policy": tuple(obs.shape)}
    elif hasattr(obs, "items"):
        observation_shapes = {
            name: tuple(value.shape)
            for name, value in obs.items()
            if isinstance(value, torch.Tensor)
        }
    else:
        raise TypeError(f"Unsupported observation type: {type(obs).__name__}")
    reward_weights = {
        name: float(env.reward_manager.get_term_cfg(name).weight) for name in env.reward_manager.active_terms
    }
    termination_cfg = {
        name: {
            "time_out": bool(env.termination_manager.get_term_cfg(name).time_out),
            "params": env.termination_manager.get_term_cfg(name).params,
        }
        for name in env.termination_manager.active_terms
    }
    valid_count = int(command.motion.reset_valid_frame_indexes.numel())

    print("\n[PLAY][SETUP]")
    print(f"  task={args_cli.task}")
    print(f"  checkpoint={checkpoint}")
    print(f"  motion_file={motion_file}")
    print(
        f"  device={env.device} num_envs={env.num_envs} step_dt={env.step_dt:g}s "
        f"control_fps={1.0 / env.step_dt:g}"
    )
    print(
        f"  reference_frames={command.motion.time_step_total} fps={command.motion.fps:g} "
        f"reset_valid_frames={valid_count}/{command.motion.time_step_total} "
        f"fixed_reference_frame={command.cfg.fixed_reference_frame}"
    )
    print(f"  observation_shapes={observation_shapes} obs_groups={agent_cfg.obs_groups}")
    print(f"  action_dim={env.action_manager.total_action_dim}")
    print(
        f"  ladder: phi={math.degrees(geometry.phi):.3f}deg spacing={geometry.rung_spacing:.4f}m "
        f"rung_length={geometry.rung_length:.4f}m rung_width={geometry.rung_width:.4f}m "
        f"num_rungs={geometry.num_rungs} origin={geometry.origin}"
    )
    print(f"  limb_phase_points={command.cfg.limb_body_names}")
    print(f"  physical_contact_bodies={PHYSICAL_CONTACT_BODY_NAMES}")
    print(f"  limb_target_offsets={command.cfg.limb_target_offsets}")
    print(
        f"  contact_metric_config: sensor={command.cfg.contact_sensor_name} "
        f"force>{command.cfg.contact_force_threshold:g}N "
        f"target_distance={command.cfg.limb_target_distance_thresholds}"
    )
    for limb_name, reward_terms in HAND_SURFACE_REWARD_TERMS.items():
        for surface_name, reward_name in reward_terms.items():
            if reward_name not in env.reward_manager.active_terms:
                continue
            reward_cfg = env.reward_manager.get_term_cfg(reward_name)
            sensor_name = reward_cfg.params["sensor_cfg"].name
            if sensor_name not in env.scene.sensors:
                continue
            sensor = env.scene.sensors[sensor_name]
            force_matrix = sensor.data.force_matrix_w
            diagnostic_params = {
                name: reward_cfg.params[name]
                for name in (
                    "force_threshold",
                    "min_top_normal_fraction",
                    "target_distance_threshold",
                    "grace_steps",
                )
                if name in reward_cfg.params
            }
            print(
                f"  {limb_name}_{surface_name}_contact: sensor={sensor_name} "
                f"filters={sensor.cfg.filter_prim_paths_expr} "
                f"force_matrix_shape={None if force_matrix is None else tuple(force_matrix.shape)} "
                f"params={diagnostic_params}"
            )
    print(f"  reward_weights={reward_weights}")
    print(f"  termination_cfg={termination_cfg}")
    print(f"  contact_flags: total normal force>{CONTACT_FORCE_THRESHOLD_N:g}N")
    print(
        "  controls: P=play/pause, N=single step while paused, "
        "I=print diagnostics, R=reset"
    )


def main():
    force_limits = _load_force_limits(args_cli.force_limits_file)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Asset-follow mode rewrites the viewport pose every render frame and prevents manual camera control.
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = None
    env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
    env_cfg.commands.motion.fixed_reference_frame = args_cli.reference_frame
    if args_cli.uniform_reset_sampling:
        env_cfg.commands.motion.fixed_start_probability = 0.0
        env_cfg.commands.motion.secondary_fixed_start_probability = 0.0
        env_cfg.commands.motion.uniform_start_probability = 1.0
        print(
            "[INFO] Play resets use uniform sampling across the motion file's "
            "reset-valid frames."
        )
    if args_cli.fast_mode:
        disabled_reward_terms = []
        for term_name, term_cfg in vars(env_cfg.rewards).items():
            if term_name.startswith("_") or term_cfg is None:
                continue
            setattr(env_cfg.rewards, term_name, None)
            disabled_reward_terms.append(term_name)
        if hasattr(env_cfg.commands.motion, "debug_vis"):
            env_cfg.commands.motion.debug_vis = False
        disabled_contact_sensors = []
        primary_contact_sensor_name = env_cfg.commands.motion.contact_sensor_name
        privileged_contact_sensor_names = {
            "left_hand_step_contacts",
            "right_hand_step_contacts",
            "left_foot_step_contacts",
            "right_foot_step_contacts",
        } if hasattr(env_cfg.observations, "privileged_actor") else set()
        for sensor_name, sensor_cfg in vars(env_cfg.scene).items():
            if not isinstance(sensor_cfg, ContactSensorCfg):
                continue
            sensor_cfg.debug_vis = False
            if (
                sensor_name != primary_contact_sensor_name
                and sensor_name not in privileged_contact_sensor_names
            ):
                setattr(env_cfg.scene, sensor_name, None)
                disabled_contact_sensors.append(sensor_name)
        print(
            f"[FAST] Disabled {len(disabled_reward_terms)} play-only reward evaluations "
            f"and {len(disabled_contact_sensors)} reward-only filtered contact sensors. "
            "Physics collisions, the primary contact sensor, observations, terminations, "
            "randomization, and disturbances are unchanged.",
            flush=True,
        )
    if args_cli.visual_diagnostics:
        attach_cr1_camera_diagnostic(
            env_cfg.scene,
            width=args_cli.camera_width,
            height=args_cli.camera_height,
            position=tuple(args_cli.camera_pos),
            rotation=tuple(args_cli.camera_rot),
            convention=args_cli.camera_convention,
            min_depth=0.1,
            max_depth=2.0,
        )
        print(
            "[CAMERA] diagnostic camera attached to d435_head_link; "
            f"resolution={args_cli.camera_width}x{args_cli.camera_height} "
            f"offset={tuple(args_cli.camera_pos)} rotation={tuple(args_cli.camera_rot)} "
            f"convention={args_cli.camera_convention}",
            flush=True,
        )
    if args_cli.disable_randomization:
        for event_name in (
            "robot_material",
            "robot_mass",
            "joint_position_offset",
            "torso_com",
            "camera_extrinsics",
            "motor_strength",
            "motor_strength_interval",
            "root_velocity_push",
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
        for delay_field in ("min_control_delay", "max_control_delay"):
            if hasattr(env_cfg.actions.joint_pos, delay_field):
                setattr(env_cfg.actions.joint_pos, delay_field, 0)
        print("[INFO] Domain/reset-state randomization and control delay disabled for play.")
    elif args_cli.base_reset_noise_only:
        for event_name in (
            "robot_material",
            "robot_mass",
            "joint_position_offset",
            "torso_com",
            "camera_extrinsics",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
        env_cfg.commands.motion.ladder_inclination_delta_range = (0.0, 0.0)
        env_cfg.commands.motion.ladder_spacing_delta_range = (0.0, 0.0)
        print("[INFO] Physics/joint randomization disabled; base reset noise retained for play.")
    elif args_cli.disable_ladder_randomization:
        env_cfg.commands.motion.ladder_inclination_delta_range = (0.0, 0.0)
        env_cfg.commands.motion.ladder_spacing_delta_range = (0.0, 0.0)
        print("[INFO] Ladder geometry randomization disabled for play; other randomization retained.")
    if not args_cli.enable_disturbances and hasattr(env_cfg.events, "root_velocity_push"):
        env_cfg.events.root_velocity_push = None
        print("[INFO] Interval root disturbances disabled for play; use --enable_disturbances to enable them.")
    elif args_cli.enable_disturbances:
        print("[INFO] Interval root velocity disturbances enabled for play.")
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    checkpoint_payload = load_checkpoint_payload(resume_path)
    split_actor_critic_checkpoint = is_split_actor_critic_checkpoint(checkpoint_payload)
    if split_actor_critic_checkpoint:
        print(
            "[INFO] Split actor/critic checkpoint detected; play will select "
            "the loader matching the installed RSL-RL API.",
            flush=True,
        )

    configure_depth_camera_timing(env_cfg, checkpoint=resume_path, update_hz=args_cli.depth_update_hz)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    runner_cfg = agent_cfg.to_dict()
    transformer_vision_ppo = runner_cfg.get("use_transformer_vision_runner", False)
    visual_dagger = runner_cfg.get("use_dagger_visual_runner", False)
    pure_visual_dagger = runner_cfg.get("use_pure_dagger_visual_runner", False)
    visual_mse_dagger = runner_cfg.get("use_visual_mse_dagger_runner", False)
    recurrent_visual_mse_dagger = runner_cfg.get(
        "use_recurrent_visual_mse_dagger_runner", False
    )
    recurrent_visual_ppo_dagger = runner_cfg.get(
        "use_recurrent_visual_ppo_dagger_runner", False
    )
    privileged_recurrent_visual_mse_dagger = runner_cfg.get(
        "use_privileged_recurrent_visual_mse_dagger_runner", False
    )
    same_input_dagger = runner_cfg.get("use_same_input_dagger_runner", False)
    visual_policy = transformer_vision_ppo or visual_dagger or pure_visual_dagger
    recurrent_visual_policy = (
        transformer_vision_ppo or visual_dagger or recurrent_visual_mse_dagger
    )
    if pure_visual_dagger:
        from ladder.tasks.ladder_climb.config.cr1.dagger.pure_wrapper import (
            CR1PureVisualDaggerVecEnvWrapper,
        )

        if args_cli.teacher_checkpoint is None and visual_mse_dagger:
            args_cli.teacher_checkpoint = checkpoint_payload.get("teacher_checkpoint")
        if args_cli.teacher_checkpoint is None:
            raise ValueError("Pure visual DAgger play requires --teacher_checkpoint")
        teacher_checkpoint = os.path.abspath(os.path.expanduser(args_cli.teacher_checkpoint))
        if not os.path.isfile(teacher_checkpoint):
            raise FileNotFoundError(
                f"Pure visual DAgger teacher checkpoint was not found: {teacher_checkpoint}"
            )
        runner_cfg["algorithm"]["teacher_checkpoint"] = teacher_checkpoint
        agent_cfg.algorithm.teacher_checkpoint = teacher_checkpoint
        env = CR1PureVisualDaggerVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if recurrent_visual_ppo_dagger:
            print(
                "[INFO] Playing the normalized recurrent Visual PPO-DAgger student; "
                "all environments execute student action means.",
                flush=True,
            )
        elif runner_cfg.get("use_recurrent_visual_ln_mse_dagger_runner", False):
            print(
                "[INFO] Playing the LayerNorm-only recurrent MSE student; "
                "all environments execute deterministic student action means.",
                flush=True,
            )
        elif recurrent_visual_mse_dagger:
            if privileged_recurrent_visual_mse_dagger:
                print(
                    "[INFO] Playing the privileged recurrent Visual MSE-DAgger "
                    "student; this checkpoint requires oracle task inputs and is "
                    "not deployable.",
                    flush=True,
                )
            else:
                print(
                    "[INFO] Playing the recurrent 25-frame Visual MSE-DAgger student.",
                    flush=True,
                )
        elif visual_mse_dagger:
            print(
                "[INFO] Playing the 100% student-rollout Visual MSE-DAgger student.",
                flush=True,
            )
        else:
            print("[INFO] Playing the pure Visual DAgger student.", flush=True)
    elif visual_dagger:
        from ladder.tasks.ladder_climb.config.cr1.dagger.visual_wrapper import (
            CR1DaggerVisualVecEnvWrapper,
        )

        if args_cli.teacher_checkpoint is None:
            raise ValueError("Visual DAgger play requires --teacher_checkpoint")
        teacher_checkpoint = os.path.abspath(os.path.expanduser(args_cli.teacher_checkpoint))
        if not os.path.isfile(teacher_checkpoint):
            raise FileNotFoundError(
                f"Visual DAgger teacher checkpoint was not found: {teacher_checkpoint}"
            )
        runner_cfg["algorithm"]["teacher_checkpoint"] = teacher_checkpoint
        agent_cfg.algorithm.teacher_checkpoint = teacher_checkpoint
        env = CR1DaggerVisualVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print("[INFO] Playing the native recurrent Visual DAgger student.", flush=True)
    elif transformer_vision_ppo:
        from ladder.rsl_rl.runners import TransformerVisionPPOVecEnvWrapper

        env = TransformerVisionPPOVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
            clip_observations=getattr(agent_cfg, "clip_observations", None),
        )
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    try:
        from rsl_rl.models import MLPModel  # noqa: F401
    except ImportError:
        uses_split_model_api = False
    else:
        uses_split_model_api = True
        if same_input_dagger:
            if "student_state_dict" not in checkpoint_payload:
                raise ValueError(
                    "Same-input DAgger play requires a distilled student "
                    "checkpoint containing student_state_dict; a teacher RL "
                    "checkpoint must not be played through this task."
                )
            teacher_checkpoint = checkpoint_payload.get("teacher_checkpoint")
            if not teacher_checkpoint or not os.path.isfile(teacher_checkpoint):
                raise FileNotFoundError(
                    "The distilled checkpoint's source teacher checkpoint is "
                    f"unavailable: {teacher_checkpoint!r}"
                )
            runner_cfg["algorithm"]["teacher_checkpoint"] = teacher_checkpoint
            agent_cfg.algorithm.teacher_checkpoint = teacher_checkpoint
            runner_cfg = convert_legacy_distillation_runner_cfg(runner_cfg)
        elif not visual_policy:
            runner_cfg = convert_legacy_runner_cfg(runner_cfg)
    if split_actor_critic_checkpoint and not uses_split_model_api:
        # The legacy checkpoint has no observation-normalizer state. Its actor
        # was exported with Identity normalization, so replay must use raw obs.
        runner_cfg["empirical_normalization"] = False
    runner_class = DistillationRunner if same_input_dagger else OnPolicyRunner
    ppo_runner = runner_class(
        env, runner_cfg, log_dir=None, device=agent_cfg.device
    )
    if same_input_dagger:
        ppo_runner.load(
            resume_path,
            load_cfg={
                "student": True,
                "teacher": False,
                "optimizer": False,
                "iteration": False,
            },
            strict=True,
        )
    elif visual_mse_dagger:
        ppo_runner.load(
            resume_path,
            load_cfg={
                "actor": True,
                "optimizer": False,
                "iteration": False,
                **({"critic": False} if recurrent_visual_ppo_dagger else {}),
            },
            strict=True,
        )
    elif split_actor_critic_checkpoint:
        if uses_split_model_api:
            ppo_runner.alg.get_policy().load_state_dict(checkpoint_payload["actor_state_dict"])
        else:
            actor_state_dict = convert_split_actor_checkpoint_to_actor_module_state_dict(
                checkpoint_payload
            )
            ppo_runner.alg.policy.actor.load_state_dict(actor_state_dict)
    else:
        ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.fast_mode:
        observation_manager = env.unwrapped.observation_manager
        if visual_policy:
            actor_group_names = ("transformer_vision_inputs",)
            if privileged_recurrent_visual_mse_dagger:
                actor_group_names += ("privileged_actor",)
            env.actor_only = True
        elif same_input_dagger:
            actor_group_names = tuple(ppo_runner.alg.get_policy().obs_groups)
        else:
            actor_group_names = tuple(agent_cfg.obs_groups["actor"])
        available_group_names = tuple(observation_manager.active_terms)
        missing_actor_groups = set(actor_group_names) - set(available_group_names)
        if missing_actor_groups:
            raise RuntimeError(
                f"Fast-mode actor observation groups are missing: {sorted(missing_actor_groups)}"
            )

        def _compute_actor_observations_only(self, update_history: bool = False):
            obs_buffer = {
                group_name: self.compute_group(group_name, update_history=update_history)
                for group_name in actor_group_names
            }
            # RSL-RL's TensorDict wrapper keeps all configured group keys, but
            # inference reads only actor_group_names. Reuse the first actor tensor
            # for unused critic-only groups instead of evaluating them a second time.
            fallback = obs_buffer[actor_group_names[0]]
            for group_name in available_group_names:
                obs_buffer.setdefault(group_name, fallback)
            self._obs_buffer = obs_buffer
            return obs_buffer

        observation_manager.compute = MethodType(
            _compute_actor_observations_only, observation_manager
        )
        print(
            f"[FAST] Computing actor groups {actor_group_names} once; "
            "critic-only observation evaluation is skipped.",
            flush=True,
        )
        motion_command = env.unwrapped.command_manager.get_term("motion")

        def _skip_training_metrics(self):
            return None

        motion_command._update_metrics = MethodType(_skip_training_metrics, motion_command)
        print(
            "[FAST] Per-step training-log metrics are skipped; reference advancement "
            "and termination checks remain active.",
            flush=True,
        )

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if args_cli.fast_mode:
        print("[FAST] Skipping JIT/ONNX export during playback.", flush=True)
    elif visual_policy:
        print(
            "[INFO] Recurrent visual export is deferred; playing the native checkpoint.",
            flush=True,
        )
    elif hasattr(ppo_runner, "export_policy_to_jit"):
        ppo_runner.export_policy_to_jit(export_model_dir, filename="policy.pt")
        ppo_runner.export_policy_to_onnx(export_model_dir, filename="policy.onnx")
    else:
        policy_nn = ppo_runner.alg.policy
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    dt = env.unwrapped.step_dt
    obs = _observations_only(env.get_observations())
    unwrapped = env.unwrapped
    controller = PlayController(paused=args_cli.start_paused, enabled=not args_cli.headless)
    camera_diagnostics = (
        CameraDiagnostics(
            CameraDiagnosticSettings(
                output_dir=Path(args_cli.camera_output_dir).resolve(),
                save_interval=args_cli.camera_save_interval,
                display_scale=args_cli.camera_display_scale,
                show=not args_cli.camera_no_show,
            )
        )
        if args_cli.visual_diagnostics
        else None
    )
    force_stats = (
        ForceStatsAccumulator(
            unwrapped,
            force_limits=force_limits,
            warmup_steps=args_cli.force_stats_warmup_steps,
        )
        if args_cli.force_stats
        else None
    )
    if args_cli.lower_body_jitter_stats:
        jitter_output_path = args_cli.lower_body_jitter_stats_file
        if jitter_output_path is None:
            reference_label = (
                "random"
                if args_cli.reference_frame is None
                else f"frame{args_cli.reference_frame}"
            )
            jitter_output_path = str(
                REPO_ROOT
                / "outputs"
                / "lower_body_jitter"
                / f"{Path(resume_path).stem}_{reference_label}.json"
            )
        lower_body_jitter_stats = LowerBodyJitterStatsAccumulator(
            unwrapped,
            checkpoint=resume_path,
            motion_file=env_cfg.commands.motion.motion_file,
            output_path=jitter_output_path,
            window_frames=args_cli.jitter_window_frames,
        )
    else:
        lower_body_jitter_stats = None
    _print_setup(
        unwrapped,
        obs,
        motion_file=env_cfg.commands.motion.motion_file,
        checkpoint=resume_path,
        agent_cfg=agent_cfg,
    )
    _print_control_delay(unwrapped, label="INITIAL")
    if lower_body_jitter_stats is not None:
        lower_body_jitter_stats.record()
    timestep = 0
    last_reward = None
    last_actions = None
    if args_cli.fast_mode:
        print(
            "[FAST] Runtime contact polling and automatic full diagnostics disabled; "
            "press I for a lightweight frame/action status line.",
            flush=True,
        )
    else:
        last_indicator = unwrapped.command_manager.get_term("motion").contact_indicator[0].clone()
        last_surface_contact_state = _hand_surface_contact_state(unwrapped)
        _print_runtime_diagnostics(
            unwrapped, total_step=timestep, reward=last_reward, actions=last_actions, label="INITIAL"
        )
    if camera_diagnostics is not None:
        camera_diagnostics.update(unwrapped.scene.sensors[CAMERA_DIAGNOSTIC_NAME], timestep)

    performance_window_start = time.perf_counter()
    performance_window_steps = 0
    performance_compute_time = 0.0
    performance_deadline_misses = 0

    try:
        while simulation_app.is_running():
            if controller.consume_reset_request():
                if force_stats is not None:
                    force_stats.finish_active_episodes(complete=False)
                if lower_body_jitter_stats is not None:
                    lower_body_jitter_stats.finish(reason="manual_reset")
                obs = _observations_only(env.reset())
                if recurrent_visual_policy:
                    policy.reset()
                last_reward = None
                last_actions = None
                if not args_cli.fast_mode:
                    last_indicator = unwrapped.command_manager.get_term("motion").contact_indicator[0].clone()
                    last_surface_contact_state = _hand_surface_contact_state(unwrapped)
                print("[CONTROL] environment reset", flush=True)
                _print_control_delay(unwrapped, label="AFTER_MANUAL_RESET")
                if not args_cli.fast_mode:
                    _print_runtime_diagnostics(
                        unwrapped, total_step=timestep, reward=None, actions=None, label="MANUAL_RESET"
                    )
                if camera_diagnostics is not None:
                    if camera_diagnostics.update(
                        unwrapped.scene.sensors[CAMERA_DIAGNOSTIC_NAME], timestep
                    ):
                        break

            if controller.consume_info_request():
                if args_cli.fast_mode:
                    command = unwrapped.command_manager.get_term("motion")
                    action_norm = (
                        float(torch.linalg.vector_norm(last_actions[0]))
                        if last_actions is not None
                        else float("nan")
                    )
                    print(
                        f"[FAST][INFO] total_step={timestep} "
                        f"reference_frame={int(command.time_steps[0])}/"
                        f"{command.motion.time_step_total - 1} action_l2_norm={action_norm:.4f}",
                        flush=True,
                    )
                else:
                    _print_runtime_diagnostics(
                        unwrapped,
                        total_step=timestep,
                        reward=last_reward,
                        actions=last_actions,
                        label="MANUAL_INFO",
                    )
                _print_control_delay(unwrapped, label="MANUAL_INFO")

            single_step = controller.consume_step_request()
            if controller.paused and not single_step:
                unwrapped.sim.render()
                if camera_diagnostics is not None:
                    if camera_diagnostics.update(
                        unwrapped.scene.sensors[CAMERA_DIAGNOSTIC_NAME], timestep
                    ):
                        break
                time.sleep(0.01)
                continue

            start_time = time.perf_counter()
            controller.ensure_timeline_playing()
            if force_stats is not None:
                force_stats.update()
            command = unwrapped.command_manager.get_term("motion")
            frame_before_step = int(command.time_steps[0])
            with torch.no_grad():
                actions = policy(obs)
                obs, reward, dones, _ = env.step(actions)
                if recurrent_visual_policy:
                    policy.reset(dones.to(dtype=torch.bool))
            compute_time = time.perf_counter() - start_time
            timestep += 1
            if camera_diagnostics is not None:
                if camera_diagnostics.update(unwrapped.scene.sensors[CAMERA_DIAGNOSTIC_NAME], timestep):
                    break
            if force_stats is not None:
                force_stats.finish_episodes(dones)

            done = bool(dones[0])
            if lower_body_jitter_stats is not None:
                if done:
                    lower_body_jitter_stats.finish(
                        reason=f"termination_at_source_frame_{frame_before_step}"
                    )
                else:
                    lower_body_jitter_stats.record(
                        action_source_frame=frame_before_step
                    )
            if done:
                print(
                    f"\n[PLAY][DONE] terminal_reference_frame={frame_before_step} "
                    f"causes={_termination_values(unwrapped)}",
                    flush=True,
                )
                _print_control_delay(unwrapped, label="AFTER_AUTO_RESET")

            if not args_cli.fast_mode:
                command = unwrapped.command_manager.get_term("motion")
                current_indicator = command.contact_indicator[0].clone()
                contact_changed = not torch.equal(current_indicator, last_indicator)
                surface_contact_state = _hand_surface_contact_state(unwrapped)
                surface_contact_began = any(
                    current and not previous
                    for current, previous in zip(surface_contact_state, last_surface_contact_state)
                )

                if (
                    single_step
                    or done
                    or surface_contact_began
                    or contact_changed
                    or timestep % args_cli.print_interval == 0
                ):
                    if single_step:
                        label = "SINGLE_STEP"
                    elif done:
                        label = "AFTER_AUTO_RESET"
                    elif surface_contact_began:
                        label = "FILTERED_CONTACT_BEGIN"
                    elif contact_changed:
                        label = "CONTACT_PHASE_CHANGE"
                    else:
                        label = "PERIODIC"
                    _print_runtime_diagnostics(
                        unwrapped, total_step=timestep, reward=reward, actions=actions, label=label
                    )

                last_indicator = current_indicator
                last_surface_contact_state = surface_contact_state

            last_reward = reward
            last_actions = actions
            if args_cli.video and timestep == args_cli.video_length:
                break
            if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
                break
            sleep_time = dt - (time.perf_counter() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
            if args_cli.fast_mode:
                performance_window_steps += 1
                performance_compute_time += compute_time
                performance_deadline_misses += int(compute_time > dt)
                if performance_window_steps >= args_cli.performance_interval:
                    wall_time = time.perf_counter() - performance_window_start
                    wall_hz = performance_window_steps / max(wall_time, 1.0e-9)
                    compute_hz = performance_window_steps / max(performance_compute_time, 1.0e-9)
                    miss_fraction = performance_deadline_misses / performance_window_steps
                    print(
                        f"[FAST][PERF] steps={performance_window_steps} "
                        f"wall_hz={wall_hz:.2f} compute_hz={compute_hz:.2f} "
                        f"real_time_factor={wall_hz * dt:.3f} "
                        f"deadline_miss_fraction={miss_fraction:.3f}",
                        flush=True,
                    )
                    performance_window_start = time.perf_counter()
                    performance_window_steps = 0
                    performance_compute_time = 0.0
                    performance_deadline_misses = 0
    finally:
        if lower_body_jitter_stats is not None:
            lower_body_jitter_stats.finish(reason="play_exit")
        if force_stats is not None:
            force_stats.finish_active_episodes(complete=False)
            force_stats.print_summary()
            if args_cli.force_stats_file is not None:
                output_path = os.path.abspath(args_cli.force_stats_file)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as output_file:
                    json.dump(force_stats.report(), output_file, indent=2, sort_keys=True)
                print(f"[PLAY][FORCE_STATS] wrote {output_path}", flush=True)
        controller.close()
        if camera_diagnostics is not None:
            camera_diagnostics.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
