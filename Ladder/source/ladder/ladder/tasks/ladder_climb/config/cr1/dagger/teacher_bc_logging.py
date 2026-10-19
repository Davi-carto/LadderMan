"""Persistent diagnostics for the teacher-controlled visual BC experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import tarfile

import torch

from .pure_model import PURE_VISUAL_DAGGER_DATA_CONTRACT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            continue
    return None


class TeacherBCLog:
    """Write run provenance, scalar diagnostics, completed episodes, and samples.

    ``metadata`` must contain JSON-serializable values. Episode columns describe
    only the completed environments selected by the algorithm; tensor conversion
    happens once per column, outside the simulator's batched calculations.
    """

    def __init__(
        self,
        log_dir,
        *,
        env,
        runner_cfg,
        motion_file,
        teacher_checkpoint,
        logger,
        metadata=None,
    ) -> None:
        self.directory = Path(log_dir).expanduser().resolve() / "teacher_bc"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        unwrapped = env.unwrapped
        command = unwrapped.command_manager.get_term("motion")
        algorithm_cfg = runner_cfg["algorithm"]
        motion_path = Path(motion_file).expanduser().resolve()
        teacher_path = Path(teacher_checkpoint).expanduser().resolve()
        repo_root = next(
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "Ladder" / "source" / "ladder").is_dir()
        )
        source_archive = self.directory / "sources.tar.gz"
        source_paths = sorted((repo_root / "Ladder/source/ladder").rglob("*.py"))
        source_paths += sorted((repo_root / "Ladder/scripts/rsl_rl").glob("*.py"))
        with tarfile.open(source_archive, "w:gz") as archive:
            for path in source_paths:
                archive.add(path, arcname=str(path.relative_to(repo_root)))
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_contract": PURE_VISUAL_DAGGER_DATA_CONTRACT,
            "rollout_controller": "frozen_teacher_deterministic_action_mean_only",
            "environment_metrics_owner": "teacher, not student autonomous performance",
            "validation": {
                "num_envs": algorithm_cfg["validation_num_envs"],
                "selection": "fixed final environment IDs, excluded from gradient updates",
                "scope": "held-out teacher trajectories; shared adaptive reset sampler",
            },
            "loss": {
                "action": "KL(student || teacher), mean over samples and action dimensions",
                "feature": "raw feature MSE, mean over samples and feature dimensions",
                "action_weight": algorithm_cfg["action_kl_weight"],
                "feature_weight": algorithm_cfg["feature_weight"],
                "rl_loss": False,
                "visual_auxiliary_loss": False,
            },
            "motion": {"path": str(motion_path), "sha256": _sha256(motion_path)},
            "teacher_checkpoint": {
                "path": str(teacher_path),
                "sha256": _sha256(teacher_path),
            },
            "versions": {
                "torch": torch.__version__,
                "isaaclab": _package_version("isaaclab"),
                "rsl_rl": _package_version("rsl-rl-lib", "rsl-rl"),
            },
            "num_envs": int(unwrapped.num_envs),
            "motion_num_frames": int(command.motion.time_step_total),
            "step_dt": float(unwrapped.step_dt),
            "seed": runner_cfg.get("seed"),
            "num_steps_per_env": runner_cfg.get("num_steps_per_env"),
            "save_interval": runner_cfg.get("save_interval"),
            "diagnostic_snapshot_interval": algorithm_cfg.get("diagnostic_snapshot_interval"),
            "action_joint_names": {
                name: list(unwrapped.action_manager.get_term(name)._joint_names)
                for name in unwrapped.action_manager.active_terms
            },
            "previous_action_baseline": {
                "slice": "proprio_history[:, -1, 59:84]",
                "units": "raw policy action, whole proprio frame clipped to [-100, 100]",
                "source": "previous teacher action during rollout; student action during autonomous play",
            },
            "episode_metric_timing": "pre_action_metrics; existing command metrics may precede reference advancement",
            "source_archive": {
                "path": str(source_archive),
                "sha256": _sha256(source_archive),
                "num_files": len(source_paths),
            },
            "configuration_directory": str(self.directory.parent / "params"),
        }
        if metadata is not None:
            manifest.update(metadata)
        with (self.directory / "manifest.json").open("w", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, allow_nan=False)
            output.write("\n")

    def write_iteration(self, iteration: int, metrics: dict[str, float]) -> None:
        row = {"iteration": int(iteration), **metrics}
        encoded = json.dumps(row, allow_nan=False)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as output:
            output.write(encoded + "\n")
        writer = getattr(self.logger, "writer", None)
        if writer is not None:
            for name, value in metrics.items():
                writer.add_scalar(f"BC/{name}", value, iteration)

    def write_episodes(self, iteration: int, columns: dict[str, torch.Tensor]) -> None:
        if not columns:
            return
        values = {name: value.detach().cpu().tolist() for name, value in columns.items()}
        num_episodes = len(next(iter(values.values())))
        if any(len(value) != num_episodes for value in values.values()):
            raise ValueError("Teacher BC episode columns must have equal row counts")
        with (self.directory / "episodes.jsonl").open("a", encoding="utf-8") as output:
            for index in range(num_episodes):
                row = {"iteration": int(iteration)}
                row.update({name: value[index] for name, value in values.items()})
                output.write(json.dumps(row, allow_nan=False) + "\n")

    def save_snapshot(self, iteration: int, payload) -> None:
        directory = self.directory / "snapshots"
        directory.mkdir(exist_ok=True)
        torch.save(payload, directory / f"rollout_{iteration:06d}.pt")


__all__ = ["TeacherBCLog"]
