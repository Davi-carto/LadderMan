"""RSL-RL DAgger positive control with one shared observation tensor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict


SAME_INPUT_DAGGER_DATA_CONTRACT = (
    "cr1_same_input_dagger_shared_full_state780_mlp512_256_256_mse_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CR1SameInputDagger(Distillation):
    """Official action-MSE distillation with strict same-input checks.

    The student always supplies the environment action. The frozen teacher is
    evaluated on the very same ``policy`` tensor at the student's visited
    state and supplies only the regression label.
    """

    def __init__(
        self,
        *args,
        teacher_checkpoint: str,
        expected_observation_dim: int = 780,
        diagnostic_snapshot_interval: int = 500,
        diagnostic_sample_envs: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.gradient_length != 1:
            raise ValueError(
                "Feed-forward same-input DAgger requires gradient_length=1; "
                "rsl-rl does not flush a partial accumulated gradient."
            )
        if self.student.obs_groups != self.teacher.obs_groups:
            raise ValueError(
                "Student and teacher must select the same observation groups, got "
                f"{self.student.obs_groups} and {self.teacher.obs_groups}."
            )
        if self.student.obs_groups != ["policy"]:
            raise ValueError(
                "Same-input DAgger requires both models to consume only the shared "
                f"'policy' group, got {self.student.obs_groups}."
            )
        if self.student.obs_dim != expected_observation_dim:
            raise ValueError(
                "Unexpected shared observation dimension: "
                f"{self.student.obs_dim} != {expected_observation_dim}."
            )
        if self.teacher.obs_dim != expected_observation_dim:
            raise ValueError(
                "Unexpected teacher observation dimension: "
                f"{self.teacher.obs_dim} != {expected_observation_dim}."
            )
        if self.student.obs_normalization or self.teacher.obs_normalization:
            raise ValueError(
                "The accepted teacher was trained without observation normalization; "
                "both same-input normalizers must remain disabled."
            )

        checkpoint = Path(teacher_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Same-input DAgger teacher checkpoint was not found: {checkpoint}"
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "actor_state_dict" not in payload:
            raise KeyError(
                "Same-input DAgger requires an RSL-RL checkpoint containing "
                "actor_state_dict."
            )
        self.teacher.load_state_dict(payload["actor_state_dict"], strict=True)
        self.teacher.eval().requires_grad_(False)
        self.teacher_loaded = True
        self.teacher_checkpoint = str(checkpoint)
        self.teacher_checkpoint_sha256 = _sha256(checkpoint)

        if diagnostic_snapshot_interval <= 0:
            raise ValueError("diagnostic_snapshot_interval must be positive")
        if diagnostic_sample_envs <= 0:
            raise ValueError("diagnostic_sample_envs must be positive")
        self.diagnostic_snapshot_interval = diagnostic_snapshot_interval
        self.diagnostic_sample_envs = diagnostic_sample_envs
        self._diagnostic_dir: Path | None = None
        self._snapshot: dict[str, torch.Tensor | int] | None = None
        self._teacher_reference = {
            name: value.detach().clone()
            for name, value in self.teacher.state_dict().items()
        }
        self._reset_rollout_metrics()

    def configure_diagnostics(self, log_dir: str, manifest: dict) -> None:
        """Write the immutable experiment contract beside the TensorBoard log."""

        self._diagnostic_dir = Path(log_dir) / "same_input_diagnostics"
        self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_contract": SAME_INPUT_DAGGER_DATA_CONTRACT,
            "rollout_policy": "student_stochastic_only",
            "teacher_role": "online_label_only_at_student_visited_state",
            "loss": "mean_squared_error(student_mean, teacher_mean)",
            "shared_observation_group": "policy",
            "shared_observation_dim": self.student.obs_dim,
            "student_observation_groups": list(self.student.obs_groups),
            "teacher_observation_groups": list(self.teacher.obs_groups),
            "student_obs_normalization": self.student.obs_normalization,
            "teacher_obs_normalization": self.teacher.obs_normalization,
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            **manifest,
        }
        path = self._diagnostic_dir / "manifest.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Execute the student and record the teacher label on the same state."""

        student_actions = self.student(obs, stochastic_output=True).detach()
        student_mean = self.student.output_mean.detach()
        teacher_mean = self.teacher(obs).detach()

        student_input = self.student.get_latent(obs)
        teacher_input = self.teacher.get_latent(obs)
        if student_input.shape != teacher_input.shape:
            raise RuntimeError(
                "Student/teacher input shapes diverged during rollout: "
                f"{tuple(student_input.shape)} != {tuple(teacher_input.shape)}"
            )
        input_difference = (student_input - teacher_input).abs().max()
        if float(input_difference) != 0.0:
            raise RuntimeError(
                "Student and teacher did not receive byte-identical observation "
                f"values (max absolute difference {float(input_difference):.9g})."
            )

        error = student_mean - teacher_mean
        exploration = student_actions - student_mean
        self._rollout_squared_error += error.square().sum(dim=0)
        self._rollout_absolute_error += error.abs().sum(dim=0)
        self._rollout_max_error = torch.maximum(
            self._rollout_max_error, error.abs().amax(dim=0)
        )
        self._rollout_exploration_squared += exploration.square().sum()
        self._rollout_student_squared += student_mean.square().sum()
        self._rollout_teacher_squared += teacher_mean.square().sum()
        self._rollout_samples += error.shape[0]

        if (
            self._diagnostic_dir is not None
            and self._snapshot is None
            and self.num_updates % self.diagnostic_snapshot_interval == 0
        ):
            count = min(self.diagnostic_sample_envs, error.shape[0])
            self._snapshot = {
                "iteration": self.num_updates,
                "shared_observation": student_input[:count].detach().cpu().clone(),
                "student_mean": student_mean[:count].cpu().clone(),
                "teacher_mean": teacher_mean[:count].cpu().clone(),
                "executed_student_action": student_actions[:count].cpu().clone(),
            }

        self.transition.actions = student_actions
        self.transition.privileged_actions = teacher_mean
        self.transition.observations = obs
        return student_actions

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        if self._snapshot is not None and "reward" not in self._snapshot:
            count = self._snapshot["shared_observation"].shape[0]
            self._snapshot["reward"] = rewards[:count].detach().cpu().clone()
            self._snapshot["done"] = dones[:count].detach().cpu().clone()
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        student_before = {
            name: value.detach().clone()
            for name, value in self.student.mlp.state_dict().items()
        }
        losses = super().update()

        parameter_delta_sq = torch.zeros((), device=self.device)
        for name, value in self.student.mlp.state_dict().items():
            parameter_delta_sq += (value - student_before[name]).square().sum()
        losses["student_parameter_delta_l2"] = float(parameter_delta_sq.sqrt())
        gradient_sq = torch.zeros((), device=self.device)
        for parameter in self.student.parameters():
            if parameter.grad is not None:
                gradient_sq += parameter.grad.square().sum()
        losses["student_grad_l2_after_clip"] = float(gradient_sq.sqrt())

        sample_count = max(self._rollout_samples, 1)
        action_count = sample_count * self._rollout_squared_error.numel()
        losses["rollout_action_rmse"] = float(
            (self._rollout_squared_error.sum() / action_count).sqrt()
        )
        losses["rollout_action_mae"] = float(
            self._rollout_absolute_error.sum() / action_count
        )
        losses["rollout_action_max_abs"] = float(self._rollout_max_error.max())
        losses["rollout_exploration_rms"] = float(
            (self._rollout_exploration_squared / action_count).sqrt()
        )
        losses["rollout_student_action_rms"] = float(
            (self._rollout_student_squared / action_count).sqrt()
        )
        losses["rollout_teacher_action_rms"] = float(
            (self._rollout_teacher_squared / action_count).sqrt()
        )
        losses["rollout_student_fraction"] = 1.0
        losses["shared_input_max_abs_diff"] = 0.0
        for index, squared_error in enumerate(self._rollout_squared_error):
            losses[f"rollout_joint_{index:02d}_rmse"] = float(
                (squared_error / sample_count).sqrt()
            )

        if self._snapshot is not None:
            for name, value in self.teacher.state_dict().items():
                if not torch.equal(value, self._teacher_reference[name]):
                    raise RuntimeError(
                        f"Frozen teacher parameter changed during training: {name}"
                    )
            if self._diagnostic_dir is not None:
                snapshot_iteration = int(self._snapshot["iteration"])
                torch.save(
                    self._snapshot,
                    self._diagnostic_dir
                    / f"iteration_{snapshot_iteration:06d}_sample.pt",
                )

        self._snapshot = None
        self._reset_rollout_metrics()
        return losses

    def save(self) -> dict:
        payload = super().save()
        payload.update(
            {
                "data_contract": SAME_INPUT_DAGGER_DATA_CONTRACT,
                "teacher_checkpoint": self.teacher_checkpoint,
                "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
                "num_updates": self.num_updates,
            }
        )
        return payload

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        contract = loaded_dict.get("data_contract")
        if contract is not None and contract != SAME_INPUT_DAGGER_DATA_CONTRACT:
            raise RuntimeError(
                "Incompatible same-input DAgger checkpoint data contract: "
                f"{contract!r}"
            )
        source_teacher_digest = loaded_dict.get("teacher_checkpoint_sha256")
        if (
            source_teacher_digest is not None
            and source_teacher_digest != self.teacher_checkpoint_sha256
        ):
            raise RuntimeError(
                "The resume checkpoint was produced from a different teacher: "
                f"{source_teacher_digest} != {self.teacher_checkpoint_sha256}."
            )
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        self.num_updates = int(loaded_dict.get("num_updates", self.num_updates))
        self.teacher.eval().requires_grad_(False)
        self.teacher_loaded = True
        self._teacher_reference = {
            name: value.detach().clone()
            for name, value in self.teacher.state_dict().items()
        }
        return load_iteration

    def _reset_rollout_metrics(self) -> None:
        action_dim = int(self.teacher.distribution.output_dim)
        self._rollout_squared_error = torch.zeros(action_dim, device=self.device)
        self._rollout_absolute_error = torch.zeros(action_dim, device=self.device)
        self._rollout_max_error = torch.zeros(action_dim, device=self.device)
        self._rollout_exploration_squared = torch.zeros((), device=self.device)
        self._rollout_student_squared = torch.zeros((), device=self.device)
        self._rollout_teacher_squared = torch.zeros((), device=self.device)
        self._rollout_samples = 0


__all__ = ["CR1SameInputDagger", "SAME_INPUT_DAGGER_DATA_CONTRACT"]
