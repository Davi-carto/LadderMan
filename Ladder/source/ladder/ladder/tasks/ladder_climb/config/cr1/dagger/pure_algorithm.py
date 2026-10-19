"""Pure student-rollout DAgger for the reference-free visual CR1 policy."""

from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn

from .pure_model import CR1PureVisualDaggerStudent, PURE_VISUAL_DAGGER_DATA_CONTRACT
from .pure_storage import PureVisualDaggerStorage, PureVisualDaggerTransition
from .student_teacher import TEACHER_KEY, load_frozen_teacher
from .visual_model import DEPTH_KEY, PROPRIO_KEY


class CR1PureVisualDagger:
    """Update every student parameter using KL and penultimate-feature imitation only."""

    def __init__(
        self,
        policy: CR1PureVisualDaggerStudent,
        teacher: nn.Sequential,
        teacher_std: torch.Tensor,
        storage: PureVisualDaggerStorage,
        *,
        num_learning_epochs: int,
        num_mini_batches: int,
        learning_rate: float,
        max_grad_norm: float,
        action_kl_weight: float,
        feature_weight: float,
        bc_warmup_iterations: int,
        teacher_rollout_fraction: float,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.teacher = teacher.to(self.device).eval().requires_grad_(False)
        self.teacher_std = teacher_std.to(self.device).clamp_min(1.0e-6)
        self.storage = storage
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.action_kl_weight = action_kl_weight
        self.feature_weight = feature_weight
        if bc_warmup_iterations < 0:
            raise ValueError("bc_warmup_iterations must be non-negative")
        if not 0.0 <= teacher_rollout_fraction <= 1.0:
            raise ValueError("teacher_rollout_fraction must be in [0, 1]")
        self.bc_warmup_iterations = bc_warmup_iterations
        self.teacher_rollout_fraction = teacher_rollout_fraction
        self.update_iteration = 0
        self.rnd = None
        self.intrinsic_rewards = None
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self._pending: PureVisualDaggerTransition | None = None
        self._teacher_action_count = 0
        self._rollout_action_count = 0

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env, cfg: dict, device: str):
        CR1PureVisualDagger._validate_observations(obs)
        policy_cfg = cfg["policy"]
        algorithm_cfg = dict(cfg["algorithm"])
        for key in (
            "class_name", "rnd_cfg", "symmetry_cfg", "optimizer",
            "normalize_advantage_per_mini_batch", "schedule", "desired_kl",
            "gamma", "lam", "clip_param", "value_loss_coef", "entropy_coef",
            "use_clipped_value_loss",
        ):
            algorithm_cfg.pop(key, None)
        teacher_checkpoint = algorithm_cfg.pop("teacher_checkpoint")
        teacher_hidden_dims = list(algorithm_cfg.pop("teacher_hidden_dims"))
        if not teacher_checkpoint:
            raise ValueError("Pure visual DAgger requires teacher_checkpoint")
        teacher = load_frozen_teacher(
            teacher_checkpoint,
            observation_dim=int(obs[TEACHER_KEY].shape[-1]),
            action_dim=env.num_actions,
            hidden_dims=teacher_hidden_dims,
            device=device,
        )
        payload = torch.load(
            Path(teacher_checkpoint).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        try:
            teacher_std = payload["actor_state_dict"]["distribution.std_param"]
        except KeyError as error:
            raise KeyError("Teacher checkpoint must contain distribution.std_param for KL distillation") from error
        if teacher_hidden_dims[-1] != 256:
            raise ValueError("Pure visual DAgger expects a 256-D final teacher hidden layer")
        policy = CR1PureVisualDaggerStudent(
            env.num_actions,
            list(policy_cfg["actor_hidden_dims"]),
            float(policy_cfg["init_noise_std"]),
        )
        storage = PureVisualDaggerStorage(
            int(cfg["num_steps_per_env"]),
            env.num_envs,
            env.num_actions,
            teacher_hidden_dims[-1],
            device,
        )
        return cls(
            policy,
            teacher,
            teacher_std,
            storage,
            device=device,
            num_learning_epochs=int(algorithm_cfg.pop("num_learning_epochs")),
            num_mini_batches=int(algorithm_cfg.pop("num_mini_batches")),
            learning_rate=float(algorithm_cfg.pop("learning_rate")),
            max_grad_norm=float(algorithm_cfg.pop("max_grad_norm")),
            action_kl_weight=float(algorithm_cfg.pop("action_kl_weight")),
            feature_weight=float(algorithm_cfg.pop("feature_weight")),
            bc_warmup_iterations=int(algorithm_cfg.pop("bc_warmup_iterations")),
            teacher_rollout_fraction=float(algorithm_cfg.pop("teacher_rollout_fraction")),
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self._pending is not None:
            raise RuntimeError("process_env_step must follow every act call")
        self._validate_observations(obs)
        actor_obs = TensorDict(
            {PROPRIO_KEY: obs[PROPRIO_KEY], DEPTH_KEY: obs[DEPTH_KEY]},
            batch_size=obs.batch_size,
            device=obs.device,
        )
        with torch.no_grad():
            student_mean, _ = self.policy.action_mean_and_feature(actor_obs)
            teacher_feature = self.teacher[:-1](obs[TEACHER_KEY])
            teacher_mean = self.teacher[-1](teacher_feature)
            if self.update_iteration < self.bc_warmup_iterations:
                teacher_count = student_mean.shape[0]
            else:
                teacher_count = int(student_mean.shape[0] * self.teacher_rollout_fraction)
            actions = student_mean.clone()
            actions[:teacher_count] = teacher_mean[:teacher_count]
        self._teacher_action_count += teacher_count
        self._rollout_action_count += student_mean.shape[0]
        self._pending = PureVisualDaggerTransition(
            obs[PROPRIO_KEY].detach(),
            obs[DEPTH_KEY].detach(),
            teacher_mean.detach(),
            teacher_feature.detach(),
        )
        return actions

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        del obs, rewards, dones, extras
        if self._pending is None:
            raise RuntimeError("act must be called before process_env_step")
        self.storage.add(self._pending)
        self._pending = None

    def compute_returns(self, obs: TensorDict) -> None:
        del obs

    def update(self) -> dict[str, float]:
        totals = {
            "distillation_kl": 0.0,
            "distillation_feature_mse": 0.0,
            "weighted_action_kl": 0.0,
            "weighted_feature_mse": 0.0,
            "distillation_total": 0.0,
            "rollout_teacher_fraction": 0.0,
        }
        updates = 0
        for batch in self.storage.batches(self.num_mini_batches, self.num_learning_epochs):
            observations = TensorDict(
                {PROPRIO_KEY: batch[PROPRIO_KEY], DEPTH_KEY: batch[DEPTH_KEY]},
                batch_size=[batch[PROPRIO_KEY].shape[0]],
                device=self.device,
            )
            student_mean, student_feature = self.policy.action_mean_and_feature(observations)
            student_std = self.policy.output_std.expand_as(student_mean).clamp_min(1.0e-6)
            teacher_std = self.teacher_std.expand_as(student_mean)
            action_kl = (
                torch.log(teacher_std / student_std)
                + (student_std.square() + (student_mean - batch["teacher_action_mean"]).square())
                / (2.0 * teacher_std.square())
                - 0.5
            ).mean()
            feature_loss = (student_feature - batch["teacher_feature"]).square().mean()
            loss = self.action_kl_weight * action_kl + self.feature_weight * feature_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals["distillation_kl"] += float(action_kl.detach())
            totals["distillation_feature_mse"] += float(feature_loss.detach())
            totals["weighted_action_kl"] += float(
                (self.action_kl_weight * action_kl).detach()
            )
            totals["weighted_feature_mse"] += float(
                (self.feature_weight * feature_loss).detach()
            )
            totals["distillation_total"] += float(loss.detach())
            updates += 1
        self.storage.clear()
        rollout_teacher_fraction = (
            self._teacher_action_count / self._rollout_action_count
        )
        self._teacher_action_count = 0
        self._rollout_action_count = 0
        self.update_iteration += 1
        result = {name: value / updates for name, value in totals.items()}
        result["rollout_teacher_fraction"] = rollout_teacher_fraction
        return result

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
            "data_contract": PURE_VISUAL_DAGGER_DATA_CONTRACT,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_iteration": self.update_iteration,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if loaded_dict.get("data_contract") != PURE_VISUAL_DAGGER_DATA_CONTRACT:
            raise RuntimeError("Incompatible pure visual DAgger checkpoint data contract")
        load_cfg = load_cfg or {"actor": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor", True):
            self.policy.load_state_dict(loaded_dict["policy_state_dict"], strict=strict)
        if load_cfg.get("optimizer", True):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.update_iteration = int(loaded_dict.get("update_iteration", 0))
        return bool(load_cfg.get("iteration", True))

    @staticmethod
    def _validate_observations(obs: TensorDict) -> None:
        missing = [key for key in (PROPRIO_KEY, DEPTH_KEY, TEACHER_KEY) if key not in obs]
        if missing:
            raise KeyError(f"Missing pure visual DAgger observations: {missing}")
        batch = obs[PROPRIO_KEY].shape[0]
        if tuple(obs[PROPRIO_KEY].shape) != (batch, 9, 85):
            raise ValueError("Pure visual DAgger proprio history must have shape [B,9,85]")
        if tuple(obs[DEPTH_KEY].shape) != (batch, 2, 60, 80):
            raise ValueError("Pure visual DAgger depth history must have shape [B,2,60,80]")


__all__ = ["CR1PureVisualDagger"]
