"""Validate TransformerVision on trajectories from the frozen CR1 actor.

This is deliberately an auxiliary-only acceptance tool.  The established
nonvisual RSL-RL policy controls the robot; the TransformerVision encoder only
predicts its configured auxiliary targets.  Neither PPO nor an optimizer step
is run here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs"
    / "rsl_rl"
    / "cr1_nonvisual_zdw0824_no_command_full_state_middle_stand"
    / "2026-08-28_18-07-30_stage36_two_anchor_20_10_35_35_frame0_nominal_to10000"
    / "model_9999.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "cr1_transformer_vision_pretrain"
_LIMB_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")
_PROPRIO_FIELDS = (
    ("root_angular_velocity", slice(0, 3)),
    ("root_orientation_6d", slice(3, 9)),
    ("joint_position", slice(9, 34)),
    ("joint_velocity", slice(34, 59)),
    ("action", slice(59, 84)),
    ("climb_direction", slice(84, 85)),
)


parser = argparse.ArgumentParser(
    description="No-update TransformerVision diagnostic under CR1's frozen nonvisual policy."
)
parser.add_argument(
    "--task",
    type=str,
    default=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "ZDW0824-WarpTransformerVision-UpDown-TaskGoal-FrozenPolicy-v0"
    ),
    help="Frozen-policy Warp TransformerVision task.",
)
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Frozen split RSL-RL checkpoint.")
parser.add_argument(
    "--depth_update_hz", type=float, default=None,
    help="Override Warp depth refresh Hz (0: policy rate). Default: visual checkpoint camera timing.",
)
parser.add_argument(
    "--vision_checkpoint",
    type=Path,
    default=None,
    help="Optional TransformerVision checkpoint to inspect without updating it.",
)
parser.add_argument(
    "--visual_architecture",
    choices=("deterministic", "task_goal"),
    default="task_goal",
    help="Visual encoder contract used by --vision_checkpoint.",
)
parser.add_argument("--num_envs", type=int, default=32, help="Parallel environments in the rollout.")
parser.add_argument("--steps", type=int, default=24, help="Control steps stored for replay.")
parser.add_argument("--display_env", type=int, default=0, help="Environment index to save in the dashboard.")
parser.add_argument("--display_step", type=int, default=-1, help="Rollout step to display; -1 selects the final step.")
parser.add_argument("--display_scale", type=float, default=1.0, help="Dashboard enlargement factor.")
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for PNG, NPY, and JSON.")
parser.add_argument(
    "--geometry_only",
    action="store_true",
    help="Show processed Warp depth and true height maps; hide untrained predictions and errors.",
)
parser.add_argument("--no_show", action="store_true", help="Save the dashboard without opening an image window.")
parser.add_argument("--seed", type=int, default=42, help="Torch seed for deterministic diagnostic sampling.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")
if args_cli.steps <= 0:
    parser.error("--steps must be positive")
if args_cli.display_env < 0 or args_cli.display_env >= args_cli.num_envs:
    parser.error("--display_env must be in [0, --num_envs)")
if args_cli.display_step < -args_cli.steps or args_cli.display_step >= args_cli.steps:
    parser.error("--display_step must be in [-steps, steps)")
if args_cli.display_scale <= 0.0:
    parser.error("--display_scale must be positive")
if args_cli.headless and not args_cli.no_show:
    parser.error("--headless requires --no_show")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import carb
import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_tasks.utils import parse_env_cfg

import ladder.tasks  # noqa: F401
from ladder.robots.cr1 import CR1_ACTION_SCALE
from ladder.rsl_rl.modules import TaskGoalTransformerVisionEncoder
from ladder.rsl_rl.runners import FrozenPolicyVisionRunner, TransformerVisionVecEnvWrapper
from ladder.tasks.ladder_climb.mdp.observations import (
    CR1_TRANSFORMER_VISION_ANGULAR_VELOCITY_SCALE,
    CR1_TRANSFORMER_VISION_JOINT_VELOCITY_SCALE,
    CR1_TRANSFORMER_VISION_PROPRIO_CLIP,
)
from ladder.utils.limb_heightmaps import RUNG, relative_height_to_world_points
from ladder.utils.depth_camera_config import configure_depth_camera_timing


_LIMB_MARKER_COLORS = {
    "left_hand": (0.0, 0.85, 1.0),
    "right_hand": (0.2, 0.35, 1.0),
    "left_foot": (0.15, 1.0, 0.25),
    "right_foot": (1.0, 0.75, 0.05),
}


def _finite_stats(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
    }


def _masked_mse_and_count(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, int]:
    valid = valid.to(dtype=prediction.dtype)
    count = int(valid.sum())
    return ((prediction - target).square() * valid).sum() / valid.sum().clamp_min(1.0), count


def _prediction_maps(output) -> torch.Tensor:
    return torch.stack(
        (
            output.hand_left_map,
            output.hand_right_map,
            output.foot_left_map,
            output.foot_right_map,
        ),
        dim=-3,
    )


def _per_limb_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    hit_class: torch.Tensor,
) -> dict[str, dict[str, float | int | dict[str, float | int | None]]]:
    metrics = {}
    for limb_index, limb_name in enumerate(_LIMB_NAMES):
        limb_valid = valid[..., limb_index, :, :]
        limb_rung = limb_valid & (hit_class[..., limb_index, :, :] == RUNG)
        limb_non_rung = limb_valid & ~limb_rung
        prediction = predictions[..., limb_index, :, :]
        target = targets[..., limb_index, :, :]
        mse, count = _masked_mse_and_count(
            prediction,
            target,
            limb_valid,
        )
        rung_mse, rung_count = _masked_mse_and_count(prediction, target, limb_rung)
        non_rung_mse, non_rung_count = _masked_mse_and_count(prediction, target, limb_non_rung)
        weight = limb_valid.to(dtype=prediction.dtype) * (
            1.0 + limb_rung.to(dtype=prediction.dtype)
        )
        weighted_mse = ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1.0)
        if count == 0:
            raise RuntimeError(f"{limb_name} has no valid height-map target values")
        metrics[limb_name] = {
            "mse": float(mse.detach()),
            "weighted_mse_2_to_1": float(weighted_mse.detach()),
            "rung_mse": None if rung_count == 0 else float(rung_mse.detach()),
            "non_rung_mse": None if non_rung_count == 0 else float(non_rung_mse.detach()),
            "valid_elements": count,
            "rung_elements": rung_count,
            "non_rung_elements": non_rung_count,
            "rung_fraction": rung_count / count,
            "target": _finite_stats(target[limb_valid]),
            "prediction": _finite_stats(prediction.detach()[limb_valid]),
        }
    return metrics


def _series_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int | None]:
    prediction = prediction.to(dtype=torch.float64)
    target = target.to(dtype=torch.float64)
    error = prediction - target
    model_mse = torch.mean(error.square())
    zero_mse = torch.mean(target.square())
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    correlation_denominator = torch.sqrt(
        target_centered.square().sum() * prediction_centered.square().sum()
    )
    correlation = None
    if correlation_denominator > 1.0e-12:
        correlation = float((target_centered * prediction_centered).sum() / correlation_denominator)
    return {
        "count": int(target.numel()),
        "model_mse": float(model_mse),
        "model_rmse": float(torch.sqrt(model_mse)),
        "model_mae": float(torch.mean(error.abs())),
        "zero_baseline_mse": float(zero_mse),
        "zero_baseline_rmse": float(torch.sqrt(zero_mse)),
        "skill_score_vs_zero": None if zero_mse <= 1.0e-12 else float(1.0 - model_mse / zero_mse),
        "target_mean": float(target.mean()),
        "target_std": float(target.std(unbiased=False)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std(unbiased=False)),
        "correlation": correlation,
    }


def _velocity_metrics(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> dict[str, object]:
    selected_prediction = prediction[valid]
    selected_target = target[valid]
    if selected_target.numel() == 0:
        raise RuntimeError("Velocity diagnostic has no valid samples")
    return {
        "units": "m/s",
        "overall": _series_metrics(selected_prediction.reshape(-1), selected_target.reshape(-1)),
        "per_axis": {
            axis: _series_metrics(selected_prediction[:, index], selected_target[:, index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "interpretation": "skill_score_vs_zero > 0 means better than always predicting zero; <= 0 means no better than zero.",
    }


def _future_proprio_metrics(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> dict[str, object]:
    target = target[..., : prediction.shape[-1]]
    selected_prediction = prediction[valid]
    selected_target = target[valid]
    if selected_target.numel() == 0:
        raise RuntimeError("Future-proprio diagnostic has no valid transitions")
    return {
        "overall": _series_metrics(selected_prediction.reshape(-1), selected_target.reshape(-1)),
        "per_field": {
            name: _series_metrics(
                selected_prediction[..., field_slice].reshape(-1),
                selected_target[..., field_slice].reshape(-1),
            )
            for name, field_slice in _PROPRIO_FIELDS
            if field_slice.stop <= prediction.shape[-1]
        },
        "field_layout": {
            name: [field_slice.start, field_slice.stop]
            for name, field_slice in _PROPRIO_FIELDS
            if field_slice.stop <= prediction.shape[-1]
        },
        "interpretation": "Only fields emitted by the selected future decoder are evaluated.",
    }


def _proprio_field_stats(values: torch.Tensor) -> dict[str, dict[str, float | int | None]]:
    return {
        name: _finite_stats(values[..., field_slice])
        for name, field_slice in _PROPRIO_FIELDS
    }


def _task_goal_status(
    prediction: torch.Tensor,
    contact_probability: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    contact_target: torch.Tensor,
) -> tuple[str, ...]:
    status = []
    for limb_index in range(len(_LIMB_NAMES)):
        if bool(target_valid[limb_index] > 0.5):
            goal_error = float(torch.linalg.vector_norm(
                prediction[limb_index] - target[limb_index]
            ))
            goal_text = f"goal_err={goal_error:.3f}m"
        else:
            goal_text = "goal=n/a"
        status.append(
            f"{goal_text}, contact={float(contact_probability[limb_index]):.2f}/"
            f"{int(contact_target[limb_index] >= 0.5)}"
        )
    return tuple(status)


def _depth_ablation_summary(
    evaluations: dict[str, tuple[object, dict[str, torch.Tensor]]]
) -> dict[str, object]:
    available = evaluations["normal"][1]
    metric_names = tuple(name for name in (
        "velocity",
        "hand",
        "foot",
        "heightmap_rung",
        "heightmap_non_rung",
        "hand_rung",
        "hand_non_rung",
        "foot_rung",
        "foot_non_rung",
        "future",
        "limb_goal",
        "contact_bce",
        "motion_joint",
        "motion_root_orientation",
    ) if name in available)
    normal = {name: float(evaluations["normal"][1][name]) for name in metric_names}
    variants = {}
    for variant_name, (_, losses) in evaluations.items():
        absolute = {name: float(losses[name]) for name in metric_names}
        variants[variant_name] = {
            "mse": absolute,
            "delta_from_normal": {name: absolute[name] - normal[name] for name in metric_names},
            "relative_increase_from_normal": {
                name: (absolute[name] - normal[name]) / max(normal[name], 1.0e-12)
                for name in metric_names
            },
        }
    return {
        "variants": variants,
        "interpretation": (
            "Zero/shuffled depth should increase reconstruction MSE if vision is used. "
            "Near-zero relative increase means proprioception/GRU can explain the maps without useful depth dependence."
        ),
    }


def _colorize(values: np.ndarray, lower: float, upper: float, invalid: np.ndarray) -> np.ndarray:
    normalized = np.round(np.clip((values - lower) / (upper - lower), 0.0, 1.0) * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    image[invalid] = (45, 45, 45)
    return np.flipud(image)


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    invalid = ~np.isfinite(depth) | (depth <= 0.0)
    normalized = np.round(np.clip((depth - 0.1) / 1.9, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    image[invalid] = (20, 20, 20)
    return image


def _tile(image: np.ndarray, title: str, detail: str, *, width: int = 150, height: int = 250) -> np.ndarray:
    canvas = np.full((height + 60, width + 16, 3), 20, dtype=np.uint8)
    canvas[40 : 40 + height, 8 : 8 + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (242, 242, 242), 1)
    cv2.putText(canvas, detail, (8, height + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1)
    return canvas


def _dashboard(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    per_limb: dict[str, dict[str, float | int]],
    *,
    step: int,
    env_id: int,
    step_label: int | None = None,
    overlay_mode: str | None = None,
    depth: torch.Tensor | None = None,
    geometry_only: bool = False,
    task_goal_status: tuple[str, ...] | None = None,
) -> np.ndarray:
    target = targets[step, env_id].detach().cpu().numpy()
    prediction = predictions[step, env_id].detach().cpu().numpy()
    hit_valid = valid[step, env_id].detach().cpu().numpy().astype(bool)
    if not np.any(hit_valid):
        visible = np.zeros(1)
    elif geometry_only:
        visible = target[hit_valid]
    else:
        visible = np.concatenate((target[hit_valid], prediction[hit_valid]))
    range_limit = max(float(np.abs(visible).max()), 0.05)
    error = np.abs(prediction - target)
    error_limit = max(float(error[hit_valid].max()) if np.any(hit_valid) else 0.0, 0.01)
    rows = []
    for limb_index, limb_name in enumerate(_LIMB_NAMES):
        invalid = ~hit_valid[limb_index]
        target_tile = _tile(
                _colorize(target[limb_index], -range_limit, range_limit, invalid),
                "target (m)",
                f"range=[{-range_limit:+.3f},{range_limit:+.3f}]",
                width=482 if geometry_only else 150,
                height=120 if geometry_only else 250,
            )
        if geometry_only:
            tiles = (target_tile,)
        else:
            tiles = (
                target_tile,
                _tile(
                _colorize(prediction[limb_index], -range_limit, range_limit, invalid),
                "prediction (m)",
                f"valid={per_limb[limb_name]['valid_elements']}",
                ),
                _tile(
                _colorize(error[limb_index], 0.0, error_limit, invalid),
                "absolute error (m)",
                f"weighted MSE={per_limb[limb_name]['weighted_mse_2_to_1']:.6f}",
                ),
            )
        label = np.full((30, sum(tile.shape[1] for tile in tiles), 3), 28, dtype=np.uint8)
        status = "" if task_goal_status is None else f" | {task_goal_status[limb_index]}"
        cv2.putText(
            label,
            limb_name + status,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46 if task_goal_status is not None else 0.58,
            (255, 255, 255),
            1,
        )
        rows.append(np.concatenate((label, np.concatenate(tiles, axis=1)), axis=0))
    header = np.full((52, rows[0].shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(
        header,
        f"{'ThinRails geometry preflight' if geometry_only else 'TransformerVision no-update diagnostic'} "
        f"| step={step if step_label is None else step_label} "
        f"env={env_id} | rows: world -X to +X"
        + ("" if overlay_mode is None else f" | 3D={overlay_mode} (H cycles, R resets)"),
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (250, 250, 250),
        1,
    )
    cv2.putText(
        header,
        (
            "Processed Warp depth and true limb-relative targets; random predictions are hidden."
            if geometry_only
            else "Targets and predictions are limb-relative heights in meters. Display colors do not change training values."
        ),
        (8, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (205, 205, 205),
        1,
    )
    panels = [header]
    if depth is not None:
        depth_np = depth.detach().cpu().numpy()
        panels.append(
            _tile(
                _colorize_depth(depth_np),
                "processed Warp depth sent to CNN (m)",
                "fixed color range=[0.1,2.0], dark=invalid/dropout/masked",
                width=rows[0].shape[1] - 16,
                height=180,
            )
        )
    return np.concatenate((*panels, *rows), axis=0)


def _opencv_highgui_available() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", maxsplit=1)[1].strip().upper() != "NONE"
    return True


class _DashboardWindow:
    def __init__(self, overlay=None) -> None:
        self._backend = "opencv" if _opencv_highgui_available() else "omni"
        self._window = None
        self._provider = None
        self._omni_ui = None
        self._overlay = overlay
        if self._backend == "omni":
            import omni.ui

            self._omni_ui = omni.ui
            print("[TRANSFORMER VISION] OpenCV HighGUI unavailable; using an Isaac Sim image window.", flush=True)

    def show(self, image: np.ndarray) -> None:
        image = cv2.resize(image, dsize=None, fx=args_cli.display_scale, fy=args_cli.display_scale, interpolation=cv2.INTER_NEAREST)
        if self._backend == "opencv":
            cv2.imshow("CR1 TransformerVision diagnostic", image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("h"), ord("H")) and self._overlay is not None:
                self._overlay.cycle()
            elif key in (ord("r"), ord("R")) and self._overlay is not None:
                self._overlay.request_reset()
            return
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        height, width = rgba.shape[:2]
        if self._provider is None:
            self._provider = self._omni_ui.ByteImageProvider()
            self._window = self._omni_ui.Window(
                "CR1 TransformerVision diagnostic", width=width, height=height + 30, visible=True
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


def _make_height_point_markers() -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/CR1TransformerVision/limb_heightmaps",
            markers={
                limb_name: sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )
                for limb_name, color in _LIMB_MARKER_COLORS.items()
            },
        )
    )


class _HeightMapOverlay:
    def __init__(
        self, marker: VisualizationMarkers, env_id: int, *, prediction_enabled: bool = True
    ) -> None:
        self._marker = marker
        self._env_id = env_id
        self._mode_index = 0
        self._target_points_w = None
        self._prediction_points_w = None
        self._valid = None
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._reset_requested = False
        self._modes = ("target", "prediction", "off") if prediction_enabled else ("target", "off")

        from omni import appwindow

        app_window = appwindow.get_default_app_window()
        if app_window is not None:
            self._input = carb.input.acquire_input_interface()
            self._keyboard = app_window.get_keyboard()
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard, self._on_keyboard_event
            )
        print(
            "[TRANSFORMER VISION] 3D height-map overlay: target. "
            f"Press H to cycle {'target -> prediction -> off' if prediction_enabled else 'target -> off'}; "
            "press R to reset.",
            flush=True,
        )

    @property
    def mode(self) -> str:
        return self._modes[self._mode_index]

    def update(
        self,
        plane_points_w: torch.Tensor,
        target_relative_height_m: torch.Tensor,
        prediction_relative_height_m: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        self._target_points_w = relative_height_to_world_points(
            plane_points_w, target_relative_height_m
        )
        self._prediction_points_w = relative_height_to_world_points(
            plane_points_w, prediction_relative_height_m
        )
        self._valid = valid.to(dtype=torch.bool)
        self._render()

    def cycle(self) -> None:
        self._mode_index = (self._mode_index + 1) % len(self._modes)
        self._render()
        print(f"[TRANSFORMER VISION] 3D height-map overlay: {self.mode}", flush=True)

    def request_reset(self) -> None:
        self._reset_requested = True
        print("[TRANSFORMER VISION] Manual reset requested.", flush=True)

    def consume_reset_request(self) -> bool:
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def close(self) -> None:
        if self._keyboard_sub is not None:
            unsubscribe = getattr(self._input, "unsubscribe_to_keyboard_events", None)
            if unsubscribe is None:
                unsubscribe = self._input.unsubscribe_from_keyboard_events
            unsubscribe(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        self._marker.set_visibility(False)

    def _render(self) -> None:
        if self.mode == "off" or self._valid is None:
            self._marker.set_visibility(False)
            return
        source = self._target_points_w if self.mode == "target" else self._prediction_points_w
        positions = []
        marker_indices = []
        for limb_index in range(len(_LIMB_NAMES)):
            limb_valid = self._valid[self._env_id, limb_index]
            limb_points = source[self._env_id, limb_index][limb_valid].clone()
            limb_points[:, 2] += 0.01
            positions.append(limb_points)
            marker_indices.append(
                torch.full(
                    (limb_points.shape[0],),
                    limb_index,
                    dtype=torch.long,
                    device=limb_points.device,
                )
            )
        positions = torch.cat(positions, dim=0)
        marker_indices = torch.cat(marker_indices, dim=0)
        if positions.shape[0] == 0:
            self._marker.set_visibility(False)
            return
        self._marker.set_visibility(True)
        self._marker.visualize(positions, marker_indices=marker_indices)

    def _on_keyboard_event(self, event, *args) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "H":
                self.cycle()
            elif event.input.name == "R":
                self.request_reset()
        return True


def main() -> None:
    torch.manual_seed(args_cli.seed)
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen policy checkpoint was not found: {checkpoint}")
    vision_checkpoint = None
    if args_cli.vision_checkpoint is not None:
        vision_checkpoint = args_cli.vision_checkpoint.expanduser().resolve()
        if not vision_checkpoint.is_file():
            raise FileNotFoundError(f"TransformerVision checkpoint was not found: {vision_checkpoint}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    # The parent ladder task follows the robot every render. Keep this
    # diagnostic viewport under operator control while the frozen policy runs.
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = None
    env_cfg.viewer.body_name = None
    configure_depth_camera_timing(
        env_cfg, checkpoint=vision_checkpoint, update_hz=args_cli.depth_update_hz,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    window = None
    overlay = None
    try:
        vector_env = TransformerVisionVecEnvWrapper(env)
        encoder = (
            TaskGoalTransformerVisionEncoder()
            if args_cli.visual_architecture == "task_goal"
            else None
        )
        runner = FrozenPolicyVisionRunner(
            vector_env,
            checkpoint,
            rollout_steps=args_cli.steps,
            encoder=encoder,
        )
        if vision_checkpoint is not None:
            iteration = runner.load_visual_checkpoint(
                vision_checkpoint,
                load_optimizer=False,
                allow_legacy_variational=True,
            )
            print(
                f"[TRANSFORMER VISION] loaded visual checkpoint={vision_checkpoint.name} iteration={iteration}",
                flush=True,
            )
        initial = vector_env.get_observations()
        print(
            "[TRANSFORMER VISION] "
            f"checkpoint={checkpoint.name} policy={tuple(initial['policy'].shape)} "
            f"actions=[{args_cli.num_envs},25] proprio={tuple(initial['transformer_vision_proprio_history'].shape)} "
            f"depth={tuple(initial['transformer_vision_depth_history'].shape)} "
            f"heightmaps={tuple(initial['transformer_vision_limb_heightmaps'].shape)} "
            f"critic={tuple(initial['critic'].shape)}",
            flush=True,
        )
        if not args_cli.no_show:
            overlay = _HeightMapOverlay(
                _make_height_point_markers(),
                args_cli.display_env,
                prediction_enabled=not args_cli.geometry_only,
            )
            window = _DashboardWindow(overlay)

            def show_live_step(step, output, observations):
                live_predictions = _prediction_maps(output).unsqueeze(0)
                live_targets = observations["transformer_vision_limb_heightmaps"].unsqueeze(0)
                live_valid = observations["transformer_vision_heightmap_valid"].unsqueeze(0) > 0.5
                live_hit_class = observations["transformer_vision_heightmap_hit_class"].unsqueeze(0)
                target_cache = getattr(env.unwrapped, "_cr1_limb_heightmap_target_cache", None)
                if target_cache is None or "plane_points_w" not in target_cache:
                    raise RuntimeError("Current limb height-map world grid is unavailable")
                overlay.update(
                    target_cache["plane_points_w"],
                    live_targets.squeeze(0),
                    live_predictions.squeeze(0),
                    live_valid.squeeze(0),
                )
                env_slice = slice(args_cli.display_env, args_cli.display_env + 1)
                live_metrics = _per_limb_metrics(
                    live_predictions[:, env_slice],
                    live_targets[:, env_slice],
                    live_valid[:, env_slice],
                    live_hit_class[:, env_slice],
                )
                task_goal_status = None
                if output.limb_goal_b is not None:
                    task_goal_status = _task_goal_status(
                        output.limb_goal_b[args_cli.display_env],
                        output.contact_probability[args_cli.display_env],
                        observations["transformer_vision_limb_goal_target"][args_cli.display_env],
                        observations["transformer_vision_limb_goal_valid"][args_cli.display_env],
                        observations["transformer_vision_contact_indicator_target"][args_cli.display_env],
                    )
                live_dashboard = _dashboard(
                    live_predictions,
                    live_targets,
                    live_valid,
                    live_metrics,
                    step=0,
                    env_id=args_cli.display_env,
                    step_label=step,
                    overlay_mode=overlay.mode,
                    depth=observations["transformer_vision_depth_history"][
                        args_cli.display_env, -1
                    ],
                    geometry_only=args_cli.geometry_only,
                    task_goal_status=task_goal_status,
                )
                window.show(live_dashboard)

            print(
                "[TRANSFORMER VISION] Live dashboard will update once per controlled robot step.",
                flush=True,
            )
        else:
            show_live_step = None

        rollout = runner.collect_no_update(
            step_callback=show_live_step,
            reset_requested=None if overlay is None else overlay.consume_reset_request,
        )
        storage = runner.storage
        depth_evaluations = runner.evaluate_depth_ablation()
        normal_output, normal_losses = depth_evaluations["normal"]
        predictions = _prediction_maps(normal_output)
        valid = storage.heightmap_valid > 0.5
        per_limb = _per_limb_metrics(
            predictions,
            storage.limb_heightmaps,
            valid,
            storage.heightmap_hit_class,
        )
        depth_ablation = _depth_ablation_summary(depth_evaluations)
        velocity = _velocity_metrics(
            normal_output.velocity,
            storage.velocity_targets,
            normal_output.valid_mask.to(dtype=torch.bool),
        )
        proprio_normalization_stats = {
            "input_history": _proprio_field_stats(storage.proprio_history),
        }
        is_task_goal = normal_output.limb_goal_b is not None
        future_proprio = None
        if not is_task_goal:
            future_valid = normal_output.valid_mask.to(dtype=torch.bool) & storage.future_valid
            future_proprio = _future_proprio_metrics(
                normal_output.next_proprio,
                storage.next_proprio_targets,
                future_valid,
            )
            proprio_normalization_stats["future_target"] = _proprio_field_stats(
                storage.next_proprio_targets[storage.future_valid]
            )
        if not torch.isfinite(torch.tensor(rollout.vision_gradient_norm)) or rollout.vision_gradient_norm <= 0.0:
            raise RuntimeError(f"Visual gradient norm must be finite and nonzero, got {rollout.vision_gradient_norm}")

        selected_step = args_cli.display_step % args_cli.steps
        task_goal_status = None
        if is_task_goal:
            task_goal_status = _task_goal_status(
                normal_output.limb_goal_b[selected_step, args_cli.display_env],
                normal_output.contact_probability[selected_step, args_cli.display_env],
                storage.limb_goal_targets[selected_step, args_cli.display_env],
                storage.limb_goal_valid[selected_step, args_cli.display_env],
                storage.contact_indicator_targets[selected_step, args_cli.display_env],
            )
        dashboard = _dashboard(
            predictions,
            storage.limb_heightmaps,
            valid,
            per_limb,
            step=selected_step,
            env_id=args_cli.display_env,
            overlay_mode=None if overlay is None else overlay.mode,
            depth=storage.depth_history[selected_step, args_cli.display_env, -1],
            geometry_only=args_cli.geometry_only,
            task_goal_status=task_goal_status,
        )
        output_dir = args_cli.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"frozen_policy_env_{args_cli.display_env:03d}_step_{selected_step:03d}"
        cv2.imwrite(str(output_dir / f"{stem}_heightmap_dashboard.png"), dashboard)
        np.save(output_dir / f"{stem}_actions.npy", rollout.actions.detach().cpu().numpy())
        np.save(output_dir / f"{stem}_heightmap_targets_m.npy", storage.limb_heightmaps[:, args_cli.display_env].detach().cpu().numpy())
        np.save(output_dir / f"{stem}_heightmap_predictions_m.npy", predictions[:, args_cli.display_env].detach().cpu().numpy())
        np.save(output_dir / f"{stem}_heightmap_valid.npy", valid[:, args_cli.display_env].detach().cpu().numpy())
        np.save(
            output_dir / f"{stem}_heightmap_hit_class.npy",
            storage.heightmap_hit_class[:, args_cli.display_env].detach().cpu().numpy(),
        )
        summary = {
            "purpose": (
                "Frozen-policy ThinRails geometry preflight; processed Warp depth and true limb maps only. No optimizer step ran."
                if args_cli.geometry_only
                else "Frozen nonvisual actor rollout; TransformerVision diagnostic only. No PPO or optimizer step ran."
            ),
            "checkpoint": str(checkpoint),
            "vision_checkpoint": None if vision_checkpoint is None else str(vision_checkpoint),
            "input_shapes": {key: list(value.shape) for key, value in initial.items() if isinstance(value, torch.Tensor)},
            "actions_shape": list(rollout.actions.shape),
            "map_order": list(_LIMB_NAMES),
            "heightmap_units": "relative limb-reference height in meters",
            "heightmap_shape": [4, 10, 3],
            "per_limb": per_limb,
            "loss_metrics_deterministic": {name: float(value.detach()) for name, value in normal_losses.items()},
            "loss_weights": {
                "velocity": 2.0,
                "hand": 2.0,
                "foot": 2.0,
                "future": 0.0 if is_task_goal else 5.0,
                "limb_goal": 1.0 if is_task_goal else 0.0,
                "contact": 1.0 if is_task_goal else 0.0,
                "motion_joint": 1.0 if is_task_goal else 0.0,
                "motion_orientation": 1.0 if is_task_goal else 0.0,
                "kl": 1.0 if "kl" in normal_losses else 0.0,
            },
            "heightmap_point_weights": {"rung": 2.0, "non_rung": 1.0, "invalid": 0.0},
            "depth_ablation": depth_ablation,
            "velocity_diagnostic": velocity,
            "future_proprio_normalization": {
                "root_angular_velocity": f"multiply by {CR1_TRANSFORMER_VISION_ANGULAR_VELOCITY_SCALE}",
                "root_orientation_6d": "unchanged",
                "joint_position": "(position - randomized default position) / per-joint CR1 action scale",
                "joint_position_scales": CR1_ACTION_SCALE,
                "joint_velocity": f"multiply by {CR1_TRANSFORMER_VISION_JOINT_VELOCITY_SCALE}",
                "action": "unchanged",
                "climb_direction": "unchanged",
                "final_clip": [-CR1_TRANSFORMER_VISION_PROPRIO_CLIP, CR1_TRANSFORMER_VISION_PROPRIO_CLIP],
            },
            "proprio_normalization_stats": proprio_normalization_stats,
            "future_valid_transitions": int(storage.future_valid.sum()),
            "future_total_transitions": int(storage.future_valid.numel()),
            "done_transitions": int(rollout.dones.sum()),
            "frozen_actor_parameter_change_max_abs": rollout.actor_parameter_change_max_abs,
            "visual_parameter_change_max_abs": rollout.vision_parameter_change_max_abs,
            "visual_gradient_norm_without_step": rollout.vision_gradient_norm,
            "optimizer_steps": 0,
        }
        if future_proprio is not None:
            summary.update(
                {
                    "future_proprio_diagnostic": future_proprio,
                    "future_prediction_dim": int(normal_output.next_proprio.shape[-1]),
                    "future_supervised_fields": future_proprio["field_layout"],
                    "future_excluded_fields": (
                        {}
                        if normal_output.next_proprio.shape[-1] == 85
                        else {"action": [59, 84], "climb_direction": [84, 85]}
                    ),
                }
            )
        else:
            summary["task_goal_diagnostic"] = {
                "limb_goal_huber": float(normal_losses["limb_goal"]),
                "contact_bce": float(normal_losses["contact_bce"]),
                "contact_accuracy": float(normal_losses["contact_accuracy"]),
                "limb_goal_valid_fraction": float(normal_losses["limb_goal_valid_fraction"]),
                "motion_joint_huber": float(normal_losses["motion_joint"]),
                "motion_root_orientation_huber": float(
                    normal_losses["motion_root_orientation"]
                ),
                "motion_target_valid_fraction": float(
                    normal_losses["motion_target_valid_fraction"]
                ),
            }
        with (output_dir / f"{stem}_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
        print(
            "[TRANSFORMER VISION] "
            f"actions={tuple(rollout.actions.shape)} done_count={summary['done_transitions']} "
            f"future_valid={summary['future_valid_transitions']}/{summary['future_total_transitions']}",
            flush=True,
        )
        for limb_name, metrics in per_limb.items():
            print(
                f"[TRANSFORMER VISION] {limb_name} weighted_mse={metrics['weighted_mse_2_to_1']:.6f} "
                f"rung_mse={metrics['rung_mse']} non_rung_mse={metrics['non_rung_mse']} "
                f"rung_elements={metrics['rung_elements']} non_rung_elements={metrics['non_rung_elements']}",
                flush=True,
            )
        for variant_name, metrics in depth_ablation["variants"].items():
            relative = metrics["relative_increase_from_normal"]
            task_metric_text = (
                f"limb_goal_increase={relative['limb_goal']:+.3f} "
                f"contact_bce_increase={relative['contact_bce']:+.3f} "
                f"motion_joint_increase={relative['motion_joint']:+.3f} "
                "motion_orientation_increase="
                f"{relative['motion_root_orientation']:+.3f}"
                if is_task_goal
                else f"future_increase={relative['future']:+.3f}"
            )
            print(
                f"[TRANSFORMER VISION] depth={variant_name} "
                f"hand_mse={metrics['mse']['hand']:.6f} foot_mse={metrics['mse']['foot']:.6f} "
                f"rung_mse={metrics['mse']['heightmap_rung']:.6f} "
                f"hand_increase={relative['hand']:+.3f} foot_increase={relative['foot']:+.3f} "
                f"rung_increase={relative['heightmap_rung']:+.3f} "
                f"velocity_increase={relative['velocity']:+.3f} {task_metric_text}",
                flush=True,
            )
        velocity_overall = velocity["overall"]
        print(
            "[TRANSFORMER VISION] velocity "
            f"model_mse={velocity_overall['model_mse']:.6f} "
            f"zero_mse={velocity_overall['zero_baseline_mse']:.6f} "
            f"skill_vs_zero={velocity_overall['skill_score_vs_zero']} "
            f"target_std={velocity_overall['target_std']:.6f} "
            f"prediction_std={velocity_overall['prediction_std']:.6f} "
            f"correlation={velocity_overall['correlation']}",
            flush=True,
        )
        for axis, metrics in velocity["per_axis"].items():
            print(
                f"[TRANSFORMER VISION] velocity_{axis} model_mse={metrics['model_mse']:.6f} "
                f"zero_mse={metrics['zero_baseline_mse']:.6f} skill={metrics['skill_score_vs_zero']} "
                f"target_mean={metrics['target_mean']:+.6f} target_std={metrics['target_std']:.6f} "
                f"prediction_mean={metrics['prediction_mean']:+.6f} "
                f"prediction_std={metrics['prediction_std']:.6f} correlation={metrics['correlation']}",
                flush=True,
            )
        if future_proprio is not None:
            future_overall = future_proprio["overall"]
            print(
                "[TRANSFORMER VISION] future_proprio "
                f"dim={summary['future_prediction_dim']} "
                f"fields={list(summary['future_supervised_fields'])} "
                f"model_mse={future_overall['model_mse']:.6f} "
                f"model_rmse={future_overall['model_rmse']:.6f} "
                f"zero_mse={future_overall['zero_baseline_mse']:.6f}",
                flush=True,
            )
            for field_name, metrics in future_proprio["per_field"].items():
                print(
                    f"[TRANSFORMER VISION] future_{field_name} model_mse={metrics['model_mse']:.6f} "
                    f"rmse={metrics['model_rmse']:.6f} mae={metrics['model_mae']:.6f} "
                    f"target_std={metrics['target_std']:.6f} prediction_std={metrics['prediction_std']:.6f} "
                    f"correlation={metrics['correlation']}",
                    flush=True,
                )
        else:
            task_metrics = summary["task_goal_diagnostic"]
            print(
                "[TRANSFORMER VISION] task_goal "
                f"limb_goal_huber={task_metrics['limb_goal_huber']:.6f} "
                f"contact_bce={task_metrics['contact_bce']:.6f} "
                f"contact_accuracy={task_metrics['contact_accuracy']:.4f} "
                f"goal_valid_fraction={task_metrics['limb_goal_valid_fraction']:.4f} "
                f"motion_joint={task_metrics['motion_joint_huber']:.6f} "
                "motion_orientation="
                f"{task_metrics['motion_root_orientation_huber']:.6f} "
                f"motion_valid_fraction={task_metrics['motion_target_valid_fraction']:.4f}",
                flush=True,
            )
        for field_name, input_stats in proprio_normalization_stats["input_history"].items():
            target_stats = proprio_normalization_stats.get("future_target", {}).get(field_name)
            target_text = "" if target_stats is None else (
                f" target_range=({target_stats['min']:+.4f},{target_stats['max']:+.4f})"
                f" target_std={target_stats['std']:.4f}"
            )
            print(
                f"[TRANSFORMER VISION] normalized_{field_name} "
                f"input_range=({input_stats['min']:+.4f},{input_stats['max']:+.4f}) "
                f"input_std={input_stats['std']:.4f}{target_text}",
                flush=True,
            )
        print(
            "[TRANSFORMER VISION] "
            f"losses={summary['loss_metrics_deterministic']} gradient_norm={rollout.vision_gradient_norm:.6f} "
            f"frozen_actor_delta={rollout.actor_parameter_change_max_abs:.1e} "
            f"visual_delta={rollout.vision_parameter_change_max_abs:.1e} "
            f"optimizer_steps=0 saved={output_dir / f'{stem}_heightmap_dashboard.png'}",
            flush=True,
        )
        if not args_cli.no_show:
            # Every live UI update happened immediately before a controlled
            # env.step(action). Pause before keeping the final dashboard open.
            import omni.timeline

            omni.timeline.get_timeline_interface().pause()
            window.show(dashboard)
            print(
                "[TRANSFORMER VISION] Live rollout complete; physics is paused at the final controlled step. "
                "Close Isaac Sim after inspection.",
                flush=True,
            )
            while simulation_app.is_running():
                simulation_app.update()
    finally:
        if overlay is not None:
            overlay.close()
        if window is not None:
            window.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
