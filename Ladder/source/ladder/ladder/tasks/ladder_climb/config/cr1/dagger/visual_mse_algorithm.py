"""Student-rollout visual DAgger with deterministic action-mean MSE only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from .pure_model import CR1PureVisualDaggerStudent
from .student_teacher import TEACHER_KEY, load_frozen_teacher
from .visual_model import DEPTH_KEY, PROPRIO_KEY
from .visual_mse_storage import VisualMseDaggerStorage, VisualMseDaggerTransition


VISUAL_MSE_DAGGER_DATA_CONTRACT = (
    "cr1_visual_mse_dagger_proprio9x85_depth2x60x80_cnn_student_rollout_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CR1VisualMseDagger:
    """Use a frozen privileged teacher only to label student-visited states."""

    def __init__(
        self,
        policy: CR1PureVisualDaggerStudent,
        teacher: nn.Sequential,
        storage: VisualMseDaggerStorage,
        *,
        teacher_checkpoint: str,
        num_learning_epochs: int,
        num_mini_batches: int,
        learning_rate: float,
        max_grad_norm: float,
        diagnostic_snapshot_interval: int,
        diagnostic_sample_envs: int,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        # Action-mean MSE has no valid gradient for a learned action standard
        # deviation. Keep it as an explicit, fixed rollout-noise parameter.
        self.policy.log_std.requires_grad_(False)
        self.teacher = teacher.to(self.device).eval().requires_grad_(False)
        self.storage = storage
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = float(max_grad_norm)
        if self.num_learning_epochs <= 0 or self.num_mini_batches <= 0:
            raise ValueError("Learning epochs and mini-batches must be positive")
        if diagnostic_snapshot_interval <= 0 or diagnostic_sample_envs <= 0:
            raise ValueError("Diagnostic snapshot settings must be positive")
        self.diagnostic_snapshot_interval = int(diagnostic_snapshot_interval)
        self.diagnostic_sample_envs = int(diagnostic_sample_envs)
        self.optimizer = torch.optim.Adam(
            (parameter for parameter in self.policy.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
        )
        self.update_iteration = 0
        self.rnd = None
        self.intrinsic_rewards = None
        self._pending: VisualMseDaggerTransition | None = None
        checkpoint = Path(teacher_checkpoint).expanduser().resolve()
        self.teacher_checkpoint = str(checkpoint)
        self.teacher_checkpoint_sha256 = _sha256(checkpoint)
        self._teacher_reference = {
            name: value.detach().clone()
            for name, value in self.teacher.state_dict().items()
        }
        self._diagnostic_dir: Path | None = None
        self._snapshot: dict[str, torch.Tensor | int] | None = None
        self._reset_rollout_metrics()

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env, cfg: dict, device: str):
        policy_cfg = cfg["policy"]
        algorithm_cfg = dict(cfg["algorithm"])
        ignored_ppo_keys = (
            "class_name",
            "rnd_cfg",
            "symmetry_cfg",
            "optimizer",
            "normalize_advantage_per_mini_batch",
            "schedule",
            "desired_kl",
            "gamma",
            "lam",
            "clip_param",
            "value_loss_coef",
            "entropy_coef",
            "use_clipped_value_loss",
        )
        for key in ignored_ppo_keys:
            algorithm_cfg.pop(key, None)

        teacher_checkpoint = algorithm_cfg.pop("teacher_checkpoint")
        teacher_hidden_dims = list(algorithm_cfg.pop("teacher_hidden_dims"))
        expected_teacher_observation_dim = int(
            algorithm_cfg.pop("expected_teacher_observation_dim")
        )
        cls._validate_observations(obs, expected_teacher_observation_dim)
        if not teacher_checkpoint:
            raise ValueError("Visual MSE-DAgger requires teacher_checkpoint")
        teacher = load_frozen_teacher(
            teacher_checkpoint,
            observation_dim=expected_teacher_observation_dim,
            action_dim=env.num_actions,
            hidden_dims=teacher_hidden_dims,
            device=device,
        )
        init_noise_std = float(policy_cfg["init_noise_std"])
        if init_noise_std <= 0.0:
            raise ValueError("Visual MSE-DAgger rollout noise must be positive")
        policy = CR1PureVisualDaggerStudent(
            env.num_actions,
            list(policy_cfg["actor_hidden_dims"]),
            init_noise_std,
        )
        storage = VisualMseDaggerStorage(
            int(cfg["num_steps_per_env"]),
            env.num_envs,
            env.num_actions,
            device,
        )
        instance = cls(
            policy,
            teacher,
            storage,
            teacher_checkpoint=teacher_checkpoint,
            device=device,
            num_learning_epochs=int(algorithm_cfg.pop("num_learning_epochs")),
            num_mini_batches=int(algorithm_cfg.pop("num_mini_batches")),
            learning_rate=float(algorithm_cfg.pop("learning_rate")),
            max_grad_norm=float(algorithm_cfg.pop("max_grad_norm")),
            diagnostic_snapshot_interval=int(
                algorithm_cfg.pop("diagnostic_snapshot_interval")
            ),
            diagnostic_sample_envs=int(
                algorithm_cfg.pop("diagnostic_sample_envs")
            ),
        )
        if algorithm_cfg:
            raise ValueError(
                "Unused Visual MSE-DAgger algorithm settings: "
                f"{sorted(algorithm_cfg)}"
            )
        return instance

    def configure_diagnostics(self, log_dir: str, manifest: dict) -> None:
        self._diagnostic_dir = Path(log_dir) / "visual_mse_diagnostics"
        self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_contract": VISUAL_MSE_DAGGER_DATA_CONTRACT,
            "rollout_policy": "student_stochastic_only",
            "teacher_role": "online_action_mean_label_at_student_visited_state",
            "optimization_loss": "mean_squared_error(student_mean, teacher_mean)",
            "excluded_losses": [
                "ppo",
                "policy_kl",
                "teacher_feature_mse",
                "velocity_prediction",
                "heightmap_reconstruction",
                "decoder_reconstruction",
            ],
            "student_inputs": {
                "proprio_history_shape": [9, 85],
                "depth_history_shape": [2, 60, 80],
                "climb_command_index_per_proprio_frame": 84,
            },
            "teacher_observation_dim": int(self.teacher[0].in_features),
            "student_network": repr(self.policy),
            "fixed_student_rollout_std": self.policy.output_std.detach().cpu().tolist(),
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            **manifest,
        }
        (self._diagnostic_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self._pending is not None:
            raise RuntimeError("process_env_step must follow every act call")
        self._validate_observations(obs, int(self.teacher[0].in_features))
        actor_obs = TensorDict(
            {PROPRIO_KEY: obs[PROPRIO_KEY], DEPTH_KEY: obs[DEPTH_KEY]},
            batch_size=obs.batch_size,
            device=obs.device,
        )
        student_mean = self.policy(actor_obs).detach()
        teacher_mean = self.teacher(obs[TEACHER_KEY]).detach()
        if not torch.isfinite(student_mean).all():
            raise FloatingPointError("Student action mean contains non-finite values")
        if not torch.isfinite(teacher_mean).all():
            raise FloatingPointError("Teacher action mean contains non-finite values")
        exploration = torch.randn_like(student_mean) * self.policy.output_std
        student_actions = student_mean + exploration

        error = student_mean - teacher_mean
        self._accumulate_rollout_metrics(
            obs, student_mean, teacher_mean, student_actions, error, exploration
        )
        if (
            self._diagnostic_dir is not None
            and self._snapshot is None
            and self.update_iteration % self.diagnostic_snapshot_interval == 0
        ):
            count = min(self.diagnostic_sample_envs, student_mean.shape[0])
            self._snapshot = {
                "iteration": self.update_iteration,
                "proprio_history": obs[PROPRIO_KEY][:count].detach().cpu().clone(),
                "depth_history": obs[DEPTH_KEY][:count].detach().cpu().clone(),
                "teacher_observation": obs[TEACHER_KEY][:count].detach().cpu().clone(),
                "student_mean": student_mean[:count].cpu().clone(),
                "teacher_mean": teacher_mean[:count].cpu().clone(),
                "executed_student_action": student_actions[:count].cpu().clone(),
            }

        self._pending = VisualMseDaggerTransition(
            obs[PROPRIO_KEY].detach(),
            obs[DEPTH_KEY].detach(),
            teacher_mean,
        )
        return student_actions

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        del obs, extras
        if self._pending is None:
            raise RuntimeError("act must be called before process_env_step")
        self.storage.add(self._pending)
        self._pending = None
        self._rollout_reward_sum += rewards.sum()
        self._rollout_done_count += dones.to(torch.float32).sum()
        if self._snapshot is not None and "reward" not in self._snapshot:
            count = self._snapshot["student_mean"].shape[0]
            self._snapshot["reward"] = rewards[:count].detach().cpu().clone()
            self._snapshot["done"] = dones[:count].detach().cpu().clone()

    def compute_returns(self, obs: TensorDict) -> None:
        del obs

    def update(self) -> dict[str, float]:
        parameters_before = {
            name: value.detach().clone()
            for name, value in self.policy.named_parameters()
            if value.requires_grad
        }
        training_squared_error = torch.zeros(
            self.policy.action_dim, device=self.device
        )
        training_sample_count = 0
        gradient_norm_sum = {name: 0.0 for name in self._module_prefixes()}
        gradient_norm_max = {name: 0.0 for name in self._module_prefixes()}
        optimizer_steps = 0

        for batch in self.storage.batches(
            self.num_mini_batches, self.num_learning_epochs
        ):
            observations = TensorDict(
                {
                    PROPRIO_KEY: batch[PROPRIO_KEY],
                    DEPTH_KEY: batch[DEPTH_KEY],
                },
                batch_size=[batch[PROPRIO_KEY].shape[0]],
                device=self.device,
            )
            student_mean = self.policy(observations)
            per_element_error = (
                student_mean - batch["teacher_action_mean"]
            ).square()
            loss = per_element_error.mean()
            if not torch.isfinite(student_mean).all():
                raise FloatingPointError("Student action mean contains non-finite values")
            if not torch.isfinite(batch["teacher_action_mean"]).all():
                raise FloatingPointError(
                    "Stored teacher action mean contains non-finite values"
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Visual MSE loss is non-finite")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                (parameter for parameter in self.policy.parameters() if parameter.requires_grad),
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            batch_gradient_norms = self._gradient_norms()
            for name, value in batch_gradient_norms.items():
                gradient_norm_sum[name] += value
                gradient_norm_max[name] = max(gradient_norm_max[name], value)
            self.optimizer.step()
            training_squared_error += per_element_error.detach().sum(dim=0)
            training_sample_count += per_element_error.shape[0]
            optimizer_steps += 1

        self.storage.clear()
        self.update_iteration += 1
        if optimizer_steps == 0 or training_sample_count == 0:
            raise RuntimeError("Visual MSE-DAgger update produced no optimizer steps")

        losses = self._finalize_rollout_metrics()
        losses["behavior"] = float(
            training_squared_error.sum()
            / (training_sample_count * self.policy.action_dim)
        )
        losses["training_action_rmse"] = losses["behavior"] ** 0.5
        for index, squared_error in enumerate(training_squared_error):
            losses[f"training_joint_{index:02d}_rmse"] = float(
                (squared_error / training_sample_count).sqrt()
            )
        losses.update(self._parameter_delta_metrics(parameters_before))
        for name in self._module_prefixes():
            losses[f"student_{name}_grad_l2_mean_after_clip"] = (
                gradient_norm_sum[name] / optimizer_steps
            )
            losses[f"student_{name}_grad_l2_max_after_clip"] = gradient_norm_max[name]
        losses["optimizer_steps"] = float(optimizer_steps)

        self._assert_teacher_frozen()
        if self._snapshot is not None and self._diagnostic_dir is not None:
            snapshot_iteration = int(self._snapshot["iteration"])
            torch.save(
                self._snapshot,
                self._diagnostic_dir
                / f"iteration_{snapshot_iteration:06d}_sample.pt",
            )
        self._snapshot = None
        self._reset_rollout_metrics()
        return losses

    def act_inference(self, obs: TensorDict, dones=None) -> torch.Tensor:
        del dones
        with torch.no_grad():
            return self.policy(obs)

    def train_mode(self) -> None:
        self.policy.train()
        self.teacher.eval()

    def eval_mode(self) -> None:
        self.policy.eval()
        self.teacher.eval()

    def get_policy(self) -> CR1PureVisualDaggerStudent:
        return self.policy

    def save(self) -> dict:
        return {
            "data_contract": VISUAL_MSE_DAGGER_DATA_CONTRACT,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_iteration": self.update_iteration,
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if loaded_dict.get("data_contract") != VISUAL_MSE_DAGGER_DATA_CONTRACT:
            raise RuntimeError("Incompatible visual MSE-DAgger checkpoint data contract")
        source_teacher_digest = loaded_dict.get("teacher_checkpoint_sha256")
        if source_teacher_digest != self.teacher_checkpoint_sha256:
            raise RuntimeError(
                "The visual MSE-DAgger checkpoint used a different teacher: "
                f"{source_teacher_digest} != {self.teacher_checkpoint_sha256}."
            )
        load_cfg = load_cfg or {"actor": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor", True):
            self.policy.load_state_dict(
                loaded_dict["policy_state_dict"], strict=strict
            )
        if load_cfg.get("optimizer", True):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.update_iteration = int(loaded_dict.get("update_iteration", 0))
        self.teacher.eval().requires_grad_(False)
        self._teacher_reference = {
            name: value.detach().clone()
            for name, value in self.teacher.state_dict().items()
        }
        return bool(load_cfg.get("iteration", True))

    @staticmethod
    def _validate_observations(obs: TensorDict, teacher_observation_dim: int) -> None:
        missing = [
            key for key in (PROPRIO_KEY, DEPTH_KEY, TEACHER_KEY) if key not in obs
        ]
        if missing:
            raise KeyError(f"Missing visual MSE-DAgger observations: {missing}")
        batch = obs[PROPRIO_KEY].shape[0]
        expected_shapes = {
            PROPRIO_KEY: (batch, 9, 85),
            DEPTH_KEY: (batch, 2, 60, 80),
            TEACHER_KEY: (batch, teacher_observation_dim),
        }
        for key, expected in expected_shapes.items():
            if tuple(obs[key].shape) != expected:
                raise ValueError(
                    f"Visual MSE-DAgger {key} must have shape {expected}, "
                    f"got {tuple(obs[key].shape)}"
                )
            if not torch.isfinite(obs[key]).all():
                raise FloatingPointError(
                    f"Visual MSE-DAgger {key} contains non-finite values"
                )
        command = obs[PROPRIO_KEY][..., -1]
        valid_command = (
            torch.isclose(command, torch.full_like(command, -1.0))
            | torch.isclose(command, torch.zeros_like(command))
            | torch.isclose(command, torch.ones_like(command))
        )
        if not valid_command.all():
            raise ValueError("Proprio history command must contain only -1, 0, or +1")

    def _accumulate_rollout_metrics(
        self,
        obs: TensorDict,
        student_mean: torch.Tensor,
        teacher_mean: torch.Tensor,
        student_actions: torch.Tensor,
        error: torch.Tensor,
        exploration: torch.Tensor,
    ) -> None:
        sample_count = error.shape[0]
        self._rollout_squared_error += error.square().sum(dim=0)
        self._rollout_absolute_error += error.abs().sum(dim=0)
        self._rollout_max_error = torch.maximum(
            self._rollout_max_error, error.abs().amax(dim=0)
        )
        self._rollout_exploration_squared += exploration.square().sum()
        self._rollout_student_squared += student_mean.square().sum()
        self._rollout_teacher_squared += teacher_mean.square().sum()
        self._rollout_student_sum += student_mean.sum(dim=0)
        self._rollout_teacher_sum += teacher_mean.sum(dim=0)
        self._rollout_cross_sum += (student_mean * teacher_mean).sum(dim=0)
        self._rollout_student_square_sum += student_mean.square().sum(dim=0)
        self._rollout_teacher_square_sum += teacher_mean.square().sum(dim=0)
        self._rollout_cosine_sum += F.cosine_similarity(
            student_mean, teacher_mean, dim=-1, eps=1.0e-8
        ).sum()
        depth = obs[DEPTH_KEY]
        self._rollout_depth_sum += depth.sum()
        self._rollout_depth_square_sum += depth.square().sum()
        self._rollout_depth_value_count += depth.numel()
        self._rollout_depth_min = torch.minimum(self._rollout_depth_min, depth.min())
        self._rollout_depth_max = torch.maximum(self._rollout_depth_max, depth.max())
        self._rollout_distinct_depth_frames += (
            (depth[:, 1] - depth[:, 0]).abs().flatten(1).amax(dim=1) > 1.0e-8
        ).to(torch.float32).sum()
        proprio = obs[PROPRIO_KEY]
        self._rollout_repeated_proprio_history += (
            (proprio - proprio[:, -1:]).abs().flatten(1).amax(dim=1) <= 1.0e-8
        ).to(torch.float32).sum()
        command = proprio[:, -1, -1]
        self._rollout_command_down += (command < -0.5).to(torch.float32).sum()
        self._rollout_command_stop += (command.abs() <= 0.5).to(torch.float32).sum()
        self._rollout_command_up += (command > 0.5).to(torch.float32).sum()
        self._rollout_samples += sample_count

    def _finalize_rollout_metrics(self) -> dict[str, float]:
        sample_count = max(self._rollout_samples, 1)
        action_count = sample_count * self.policy.action_dim
        depth_count = max(self._rollout_depth_value_count, 1)
        depth_mean = self._rollout_depth_sum / depth_count
        depth_variance = (
            self._rollout_depth_square_sum / depth_count - depth_mean.square()
        ).clamp_min(0.0)
        covariance = (
            self._rollout_cross_sum
            - self._rollout_student_sum * self._rollout_teacher_sum / sample_count
        )
        student_variance_sum = (
            self._rollout_student_square_sum
            - self._rollout_student_sum.square() / sample_count
        ).clamp_min(0.0)
        teacher_variance_sum = (
            self._rollout_teacher_square_sum
            - self._rollout_teacher_sum.square() / sample_count
        ).clamp_min(0.0)
        denominator = (student_variance_sum * teacher_variance_sum).sqrt()
        correlation = torch.where(
            denominator > 1.0e-12,
            covariance / denominator.clamp_min(1.0e-12),
            torch.zeros_like(denominator),
        )
        result = {
            "rollout_action_rmse": float(
                (self._rollout_squared_error.sum() / action_count).sqrt()
            ),
            "rollout_action_mae": float(
                self._rollout_absolute_error.sum() / action_count
            ),
            "rollout_action_max_abs": float(self._rollout_max_error.max()),
            "rollout_action_correlation_mean": float(correlation.mean()),
            "rollout_action_cosine_similarity": float(
                self._rollout_cosine_sum / sample_count
            ),
            "rollout_exploration_rms": float(
                (self._rollout_exploration_squared / action_count).sqrt()
            ),
            "rollout_student_action_rms": float(
                (self._rollout_student_squared / action_count).sqrt()
            ),
            "rollout_teacher_action_rms": float(
                (self._rollout_teacher_squared / action_count).sqrt()
            ),
            "rollout_student_fraction": 1.0,
            "rollout_mean_reward": float(self._rollout_reward_sum / sample_count),
            "rollout_done_fraction": float(self._rollout_done_count / sample_count),
            "rollout_depth_mean": float(depth_mean),
            "rollout_depth_std": float(depth_variance.sqrt()),
            "rollout_depth_min": float(self._rollout_depth_min),
            "rollout_depth_max": float(self._rollout_depth_max),
            "rollout_distinct_depth_frame_fraction": float(
                self._rollout_distinct_depth_frames / sample_count
            ),
            "rollout_repeated_proprio_history_fraction": float(
                self._rollout_repeated_proprio_history / sample_count
            ),
            "rollout_command_down_fraction": float(
                self._rollout_command_down / sample_count
            ),
            "rollout_command_stop_fraction": float(
                self._rollout_command_stop / sample_count
            ),
            "rollout_command_up_fraction": float(
                self._rollout_command_up / sample_count
            ),
        }
        for index, squared_error in enumerate(self._rollout_squared_error):
            result[f"rollout_joint_{index:02d}_rmse"] = float(
                (squared_error / sample_count).sqrt()
            )
            result[f"rollout_joint_{index:02d}_correlation"] = float(
                correlation[index]
            )
        return result

    @staticmethod
    def _module_prefixes() -> tuple[str, ...]:
        return ("all", "depth_encoder", "feature_mlp", "action_head")

    def _gradient_norms(self) -> dict[str, float]:
        totals = {
            name: torch.zeros((), device=self.device)
            for name in self._module_prefixes()
        }
        for name, parameter in self.policy.named_parameters():
            if parameter.grad is None:
                continue
            square_sum = parameter.grad.square().sum()
            totals["all"] += square_sum
            for prefix in self._module_prefixes()[1:]:
                if name.startswith(prefix + "."):
                    totals[prefix] += square_sum
        return {name: float(value.sqrt()) for name, value in totals.items()}

    def _parameter_delta_metrics(
        self, parameters_before: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        totals = {
            name: torch.zeros((), device=self.device)
            for name in self._module_prefixes()
        }
        for name, parameter in self.policy.named_parameters():
            if name not in parameters_before:
                continue
            square_sum = (parameter - parameters_before[name]).square().sum()
            totals["all"] += square_sum
            for prefix in self._module_prefixes()[1:]:
                if name.startswith(prefix + "."):
                    totals[prefix] += square_sum
        return {
            f"student_{name}_parameter_delta_l2": float(value.sqrt())
            for name, value in totals.items()
        }

    def _assert_teacher_frozen(self) -> None:
        for name, value in self.teacher.state_dict().items():
            if not torch.equal(value, self._teacher_reference[name]):
                raise RuntimeError(f"Frozen teacher parameter changed: {name}")

    def _reset_rollout_metrics(self) -> None:
        action_dim = self.policy.action_dim
        vector = lambda: torch.zeros(action_dim, device=self.device)
        scalar = lambda: torch.zeros((), device=self.device)
        self._rollout_squared_error = vector()
        self._rollout_absolute_error = vector()
        self._rollout_max_error = vector()
        self._rollout_student_sum = vector()
        self._rollout_teacher_sum = vector()
        self._rollout_cross_sum = vector()
        self._rollout_student_square_sum = vector()
        self._rollout_teacher_square_sum = vector()
        self._rollout_exploration_squared = scalar()
        self._rollout_student_squared = scalar()
        self._rollout_teacher_squared = scalar()
        self._rollout_cosine_sum = scalar()
        self._rollout_reward_sum = scalar()
        self._rollout_done_count = scalar()
        self._rollout_depth_sum = scalar()
        self._rollout_depth_square_sum = scalar()
        self._rollout_depth_min = torch.full((), torch.inf, device=self.device)
        self._rollout_depth_max = torch.full((), -torch.inf, device=self.device)
        self._rollout_distinct_depth_frames = scalar()
        self._rollout_repeated_proprio_history = scalar()
        self._rollout_command_down = scalar()
        self._rollout_command_stop = scalar()
        self._rollout_command_up = scalar()
        self._rollout_samples = 0
        self._rollout_depth_value_count = 0


__all__ = ["CR1VisualMseDagger", "VISUAL_MSE_DAGGER_DATA_CONTRACT"]
