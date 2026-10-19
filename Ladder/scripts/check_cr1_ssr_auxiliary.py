"""Inspect CR1 SSR auxiliary targets and predictions without changing parameters.

This Gate-E tool intentionally collects a normal visual rollout, then replays
the exact stored temporal inputs through the auxiliary path.  It never calls a
PPO update or either optimizer's ``step`` method.
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "cr1_ssr_auxiliary"
_LIMB_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")


parser = argparse.ArgumentParser(
    description="Gate-E no-update diagnostic for CR1's recurrent Warp SSR visual auxiliary path."
)
parser.add_argument(
    "--task",
    type=str,
    default="Ladder-Climb-CR1-Geometry-DR-WarpSSRVisual-v0",
    help="Opt-in recurrent CR1 Warp SSR visual task.",
)
parser.add_argument("--num_envs", type=int, default=32, help="Parallel environments used for one rollout.")
parser.add_argument("--steps", type=int, default=24, help="Stored control steps used for the diagnostic.")
parser.add_argument("--display_env", type=int, default=0, help="Environment shown in the map dashboard.")
parser.add_argument(
    "--display_step",
    type=int,
    default=-1,
    help="Stored rollout step to display; -1 selects the final step.",
)
parser.add_argument(
    "--map_display_limit_m",
    type=float,
    default=0.0,
    help="Symmetric color limit in meters; zero automatically uses the selected target/prediction range.",
)
parser.add_argument("--display_scale", type=float, default=1.0, help="Dashboard enlargement factor.")
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="PNG, NPY, and JSON output directory.")
parser.add_argument("--no_show", action="store_true", help="Save the dashboard without opening an image window.")
parser.add_argument("--seed", type=int, default=42, help="PyTorch seed for the no-update rollout and VAE sample.")
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
if args_cli.map_display_limit_m < 0.0:
    parser.error("--map_display_limit_m must be non-negative")
if args_cli.display_scale <= 0.0:
    parser.error("--display_scale must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ladder.tasks  # noqa: F401
from ladder.algorithms import LadderSSRPPO
from ladder.models import LadderVisualLossWeights, LadderVisualTargets
from ladder.runners import LadderSSRVecEnvWrapper
from ladder.tasks.ladder_climb.config.cr1.agents.rsl_rl_ppo_cfg import CR1SSRVisualPPORunnerCfg


def _masked_mse_with_count(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Return map MSE and the exact number of valid map elements."""

    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError(
            "map prediction, target, and valid mask must have identical shapes, got "
            f"{tuple(prediction.shape)}, {tuple(target.shape)}, {tuple(valid.shape)}"
        )
    valid = valid.to(dtype=prediction.dtype)
    denominator = valid.sum()
    return ((prediction - target).square() * valid).sum() / denominator.clamp_min(1.0), int(denominator.item())


def _finite_stats(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "p01": None, "p50": None, "p99": None}
    quantiles = torch.quantile(values, torch.tensor((0.01, 0.50, 0.99), device=values.device))
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "p01": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p99": float(quantiles[2]),
    }


def _colorize(values: np.ndarray, lower: float, upper: float, invalid: np.ndarray) -> np.ndarray:
    normalized = np.round(np.clip((values - lower) / (upper - lower), 0.0, 1.0) * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    image[invalid] = (45, 45, 45)
    return np.flipud(image)


def _map_tile(image: np.ndarray, title: str, detail: str, *, width: int, height: int) -> np.ndarray:
    canvas = np.full((height + 62, width + 16, 3), 20, dtype=np.uint8)
    canvas[42 : 42 + height, 8 : 8 + width] = cv2.resize(
        image, (width, height), interpolation=cv2.INTER_NEAREST
    )
    cv2.putText(canvas, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (242, 242, 242), 1)
    cv2.putText(canvas, detail, (8, height + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1)
    return canvas


def _dashboard(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    per_limb: dict[str, dict[str, float | int]],
    *,
    display_limit_m: float,
    rollout_step: int,
    environment_index: int,
) -> np.ndarray:
    """Render all four target/prediction/error maps in shared physical units."""

    target = targets[rollout_step, environment_index].detach().cpu().numpy()
    prediction = predictions[rollout_step, environment_index].detach().cpu().numpy()
    hit_valid = valid[rollout_step, environment_index].detach().cpu().numpy().astype(bool)
    selected_values = np.concatenate((target[hit_valid], prediction[hit_valid])) if np.any(hit_valid) else np.zeros(1)
    limit = display_limit_m if display_limit_m > 0.0 else max(float(np.abs(selected_values).max()), 0.05)
    error = np.abs(prediction - target)
    error_limit = max(float(error[hit_valid].max()) if np.any(hit_valid) else 0.0, 0.01)
    rows = []
    for limb_index, limb_name in enumerate(_LIMB_NAMES):
        invalid = ~hit_valid[limb_index]
        target_tile = _map_tile(
            _colorize(target[limb_index], -limit, limit, invalid),
            "target relative height (m)",
            f"shared range=[{-limit:+.3f},{limit:+.3f}]",
            width=150,
            height=250,
        )
        prediction_tile = _map_tile(
            _colorize(prediction[limb_index], -limit, limit, invalid),
            "prediction relative height (m)",
            f"valid elements={per_limb[limb_name]['valid_elements']}",
            width=150,
            height=250,
        )
        error_tile = _map_tile(
            _colorize(error[limb_index], 0.0, error_limit, invalid),
            "absolute error (m)",
            f"MSE={per_limb[limb_name]['mse']:.6f}",
            width=150,
            height=250,
        )
        body = np.concatenate((target_tile, prediction_tile, error_tile), axis=1)
        label = np.full((31, body.shape[1], 3), 28, dtype=np.uint8)
        cv2.putText(label, limb_name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
        rows.append(np.concatenate((label, body), axis=0))
    header = np.full((52, rows[0].shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(
        header,
        f"Gate E no-update diagnostic | rollout step={rollout_step} env={environment_index} | rows: -world X to +world X",
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (250, 250, 250),
        1,
    )
    cv2.putText(
        header,
        "Target values are current relative-height labels in meters. Color scaling is display-only; no clipping or normalization is applied.",
        (8, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (205, 205, 205),
        1,
    )
    return np.concatenate((header, *rows), axis=0)


def _opencv_highgui_available() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", maxsplit=1)[1].strip().upper() != "NONE"
    return True


class _DashboardWindow:
    def __init__(self) -> None:
        self._backend = "opencv" if _opencv_highgui_available() else "omni"
        self._window = None
        self._provider = None
        self._omni_ui = None
        if self._backend == "omni":
            import omni.ui

            self._omni_ui = omni.ui
            print("[SSR AUX] OpenCV HighGUI unavailable; using an Isaac Sim image window.", flush=True)

    def show(self, image: np.ndarray) -> None:
        resized = cv2.resize(
            image,
            dsize=None,
            fx=args_cli.display_scale,
            fy=args_cli.display_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        if self._backend == "opencv":
            cv2.imshow("CR1 SSR auxiliary diagnostic", resized)
            cv2.waitKey(1)
            return
        rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
        height, width = rgba.shape[:2]
        if self._provider is None:
            self._provider = self._omni_ui.ByteImageProvider()
            self._window = self._omni_ui.Window(
                "CR1 SSR auxiliary diagnostic", width=width, height=height + 30, visible=True
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


def _build_targets(algorithm: LadderSSRPPO) -> LadderVisualTargets:
    storage = algorithm.storage
    return LadderVisualTargets(
        velocity=storage.velocity_targets,
        limb_heightmaps=storage.limb_heightmaps,
        heightmap_valid=storage.heightmap_valid,
        next_proprio=storage.next_proprio_targets,
        future_valid=storage.future_valid,
    )


def _auxiliary_parameters(algorithm: LadderSSRPPO) -> list[torch.nn.Parameter]:
    return [parameter for group in algorithm.auxiliary_optimizer.param_groups for parameter in group["params"]]


def main() -> None:
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    window = None
    try:
        vector_env = LadderSSRVecEnvWrapper(env)
        runner_cfg = CR1SSRVisualPPORunnerCfg()
        runner_cfg.device = args_cli.device
        runner_cfg.num_steps_per_env = args_cli.steps
        runner_cfg.algorithm.enable_auxiliary_update = False
        algorithm = LadderSSRPPO.construct_algorithm(
            vector_env.get_observations(), vector_env, runner_cfg.to_dict(), args_cli.device
        )
        algorithm.train_mode()
        observations = vector_env.get_observations()
        print(
            "[SSR AUX] "
            f"envs={args_cli.num_envs} steps={args_cli.steps} "
            f"proprio={tuple(observations['ssr_proprio_history'].shape)} "
            f"depth={tuple(observations['ssr_depth_history'].shape)} "
            f"maps={tuple(observations['ssr_limb_heightmaps'].shape)} "
            "optimization=disabled",
            flush=True,
        )
        for step in range(args_cli.steps):
            actions = algorithm.act(observations)
            observations, rewards, dones, extras = vector_env.step(actions)
            algorithm.process_env_step(observations, rewards, dones, extras)
            print(
                f"[SSR AUX] rollout={step + 1}/{args_cli.steps} "
                f"reward_mean={float(rewards.mean()):.6f} done_count={int(dones.sum())}",
                flush=True,
            )

        storage = algorithm.storage
        targets = _build_targets(algorithm)
        masks = torch.ones_like(storage.dones)
        parameters = _auxiliary_parameters(algorithm)
        parameter_snapshot = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
        algorithm.auxiliary_optimizer.zero_grad(set_to_none=True)
        output = algorithm.actor(
            storage.proprio_history,
            storage.depth_history,
            hidden_state=storage.initial_hidden_state,
            dones=storage.reset_before_mask(),
            masks=masks,
            deterministic=False,
        )
        weights = LadderVisualLossWeights()
        losses = algorithm.actor.auxiliary_losses(output, targets, weights)

        map_predictions = (
            output.hand_left_map,
            output.hand_right_map,
            output.foot_left_map,
            output.foot_right_map,
        )
        per_limb: dict[str, dict[str, float | int]] = {}
        map_losses = []
        for index, (limb_name, prediction) in enumerate(zip(_LIMB_NAMES, map_predictions, strict=True)):
            valid = targets.heightmap_valid[..., index, :, :] > 0.5
            target = targets.limb_heightmaps[..., index, :, :]
            map_loss, valid_elements = _masked_mse_with_count(prediction, target, valid)
            if valid_elements == 0:
                raise RuntimeError(f"{limb_name} has no valid target elements in this rollout")
            map_losses.append(map_loss)
            per_limb[limb_name] = {
                "mse": float(map_loss.detach()),
                "valid_elements": valid_elements,
                "target": _finite_stats(target[valid]),
                "prediction": _finite_stats(prediction.detach()[valid]),
            }

        if not torch.allclose(losses["hand"], 0.5 * (map_losses[0] + map_losses[1]), atol=1.0e-6, rtol=1.0e-5):
            raise RuntimeError("hand reconstruction MSE does not match its two independently masked side maps")
        if not torch.allclose(losses["foot"], 0.5 * (map_losses[2] + map_losses[3]), atol=1.0e-6, rtol=1.0e-5):
            raise RuntimeError("foot reconstruction MSE does not match its two independently masked side maps")

        losses["total"].backward()
        gradient_norm = torch.sqrt(
            sum(
                (parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None),
                torch.zeros((), device=algorithm.device),
            )
        )
        if not torch.isfinite(gradient_norm) or gradient_norm.item() <= 0.0:
            raise RuntimeError(f"Auxiliary encoder gradient norm must be finite and nonzero, got {gradient_norm.item()}")
        algorithm.auxiliary_optimizer.zero_grad(set_to_none=True)
        max_parameter_change = max(
            float((parameter.detach() - snapshot).abs().max())
            for parameter, snapshot in zip(algorithm.actor.parameters(), parameter_snapshot, strict=True)
        )
        if max_parameter_change != 0.0:
            raise RuntimeError(f"No-update diagnostic changed model parameters by {max_parameter_change}")

        selected_step = args_cli.display_step % args_cli.steps
        predictions = torch.stack(map_predictions, dim=2)
        target_values = targets.limb_heightmaps
        valid_values = targets.heightmap_valid > 0.5
        all_target_stats = _finite_stats(target_values[valid_values])
        display_limit = args_cli.map_display_limit_m
        beyond_display_fraction = 0.0
        if display_limit > 0.0:
            beyond_display_fraction = float((target_values[valid_values].abs() > display_limit).float().mean())
        dashboard = _dashboard(
            predictions,
            target_values,
            valid_values,
            per_limb,
            display_limit_m=display_limit,
            rollout_step=selected_step,
            environment_index=args_cli.display_env,
        )
        output_dir = args_cli.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"gate_e_env_{args_cli.display_env:03d}_step_{selected_step:03d}"
        cv2.imwrite(str(output_dir / f"{stem}_dashboard.png"), dashboard)
        np.save(output_dir / f"{stem}_target_relative_height_m.npy", target_values[:, args_cli.display_env].detach().cpu().numpy())
        np.save(output_dir / f"{stem}_prediction_relative_height_m.npy", predictions[:, args_cli.display_env].detach().cpu().numpy())
        np.save(output_dir / f"{stem}_hit_valid.npy", valid_values[:, args_cli.display_env].detach().cpu().numpy())
        summary = {
            "purpose": "Gate E diagnostic only; no PPO update and no optimizer step were executed.",
            "units": "relative limb-reference height in meters",
            "map_order": list(_LIMB_NAMES),
            "map_shape": [10, 3],
            "rollout": {"num_envs": args_cli.num_envs, "steps": args_cli.steps, "display_env": args_cli.display_env, "display_step": selected_step},
            "target_stats_all_valid": all_target_stats,
            "map_display_limit_m": display_limit if display_limit > 0.0 else "automatic",
            "target_fraction_beyond_explicit_display_limit": beyond_display_fraction,
            "per_limb": per_limb,
            "loss_unweighted": {name: float(value.detach()) for name, value in losses.items() if name != "total"},
            "loss_weighted": {
                "velocity": weights.velocity * float(losses["velocity"].detach()),
                "hand": weights.hand * float(losses["hand"].detach()),
                "foot": weights.foot * float(losses["foot"].detach()),
                "future": weights.future * float(losses["future"].detach()),
                "kl": weights.kl * float(losses["kl"].detach()),
                "total": float(losses["total"].detach()),
            },
            "loss_weights": vars(weights),
            "future_valid_transitions": int(targets.future_valid.sum()),
            "future_total_transitions": int(targets.future_valid.numel()),
            "auxiliary_gradient_norm": float(gradient_norm),
            "optimizer_steps": 0,
            "parameter_change_max_abs": max_parameter_change,
        }
        with (output_dir / f"{stem}_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
        print(f"[SSR AUX] target_stats_all_valid={all_target_stats}", flush=True)
        for limb_name, metrics in per_limb.items():
            print(
                f"[SSR AUX] {limb_name} mse={metrics['mse']:.6f} valid_elements={metrics['valid_elements']} "
                f"target={metrics['target']}",
                flush=True,
            )
        print(
            f"[SSR AUX] losses_unweighted={summary['loss_unweighted']} "
            f"losses_weighted={summary['loss_weighted']} "
            f"auxiliary_gradient_norm={float(gradient_norm):.6f}",
            flush=True,
        )
        print(
            f"[SSR AUX] optimizer_steps=0 parameter_change_max_abs={max_parameter_change:.1e} "
            f"saved={output_dir / f'{stem}_dashboard.png'}",
            flush=True,
        )
        if not args_cli.no_show and not args_cli.headless:
            window = _DashboardWindow()
            window.show(dashboard)
            print("[SSR AUX] Dashboard is open. Close the Isaac Sim application after inspection.", flush=True)
            while simulation_app.is_running():
                simulation_app.update()
    finally:
        if window is not None:
            window.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
