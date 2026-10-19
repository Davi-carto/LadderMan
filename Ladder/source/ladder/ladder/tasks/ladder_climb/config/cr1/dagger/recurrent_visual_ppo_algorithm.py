"""End-to-end recurrent PPO on student trajectories, MSE on all trajectories."""

from __future__ import annotations

import json
from itertools import chain
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from .recurrent_visual_model import DEPTH_UPDATE_KEY
from .recurrent_visual_mse_algorithm import CR1RecurrentVisualMseDagger
from .recurrent_visual_mse_storage import RecurrentVisualMseDaggerTransition
from .recurrent_visual_ppo_model import (
    CR1RecurrentVisualPpoStudent,
    RECURRENT_VISUAL_PPO_DAGGER_DATA_CONTRACT,
)
from .recurrent_visual_ppo_storage import RecurrentVisualPpoDaggerStorage
from .student_teacher import TEACHER_KEY, elu_mlp, load_frozen_teacher
from .visual_model import DEPTH_KEY, PROPRIO_KEY


class CR1RecurrentVisualPpoDagger(CR1RecurrentVisualMseDagger):
    def __init__(
        self, *args, critic, gamma, lam, clip_param, value_loss_coef,
        entropy_coef, use_clipped_value_loss, schedule, desired_kl,
        min_learning_rate, max_learning_rate, ppo_initial_coeff,
        ppo_warmup_iterations, dagger_initial_coeff,
        dagger_final_coeff, dagger_anneal_start, dagger_anneal_iterations, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.bc_warmup_iterations != 0 or self.teacher_rollout_fraction >= 1.0:
            raise ValueError("Hybrid PPO requires student trajectories from iteration zero")
        if schedule not in ("fixed", "adaptive"):
            raise ValueError("PPO schedule must be fixed or adaptive")
        if schedule == "adaptive" and (desired_kl is None or desired_kl <= 0):
            raise ValueError("Adaptive PPO requires a positive desired_kl")
        if not 0 < min_learning_rate <= self.learning_rate <= max_learning_rate:
            raise ValueError("PPO learning rate must lie within its configured bounds")
        if not 0 <= ppo_initial_coeff <= 1 or ppo_warmup_iterations < 0:
            raise ValueError("Invalid PPO actor warmup coefficient or interval")
        if not 0 <= dagger_final_coeff <= dagger_initial_coeff:
            raise ValueError("Distillation coefficients must be non-negative and decay")
        if dagger_anneal_start < 0 or dagger_anneal_iterations <= 0:
            raise ValueError("Invalid distillation annealing interval")
        self.critic = critic.to(self.device)
        self.policy.log_std.requires_grad_(True)
        self.optimizer = torch.optim.Adam(
            chain(self.policy.parameters(), self.critic.parameters()),
            lr=self.learning_rate,
        )
        self.gamma, self.lam = float(gamma), float(lam)
        self.clip_param = float(clip_param)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.schedule, self.desired_kl = schedule, desired_kl
        self.min_learning_rate = float(min_learning_rate)
        self.max_learning_rate = float(max_learning_rate)
        self.ppo_initial_coeff = float(ppo_initial_coeff)
        self.ppo_warmup_iterations = int(ppo_warmup_iterations)
        self.dagger_initial_coeff = float(dagger_initial_coeff)
        self.dagger_final_coeff = float(dagger_final_coeff)
        self.dagger_anneal_start = int(dagger_anneal_start)
        self.dagger_anneal_iterations = int(dagger_anneal_iterations)
        self._ppo_pending = None
        self._returns_ready = False

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        settings = dict(cfg["algorithm"])
        settings.pop("class_name", None)
        for name in ("rnd_cfg", "symmetry_cfg"):
            if settings.pop(name, None):
                raise ValueError(f"Hybrid recurrent PPO does not support {name}")
        if cfg.get("multi_gpu"):
            raise ValueError("Hybrid recurrent PPO currently supports one device")
        if settings.pop("optimizer", "adam").lower() != "adam":
            raise ValueError("Hybrid recurrent PPO uses Adam")
        if settings.pop("normalize_advantage_per_mini_batch", False):
            raise ValueError("Advantages are normalized over student rollout samples")
        teacher_checkpoint = settings.pop("teacher_checkpoint")
        teacher_hidden_dims = list(settings.pop("teacher_hidden_dims"))
        observation_dim = int(settings.pop("expected_teacher_observation_dim"))
        cls._validate_observations(obs, observation_dim)
        teacher = load_frozen_teacher(
            teacher_checkpoint, observation_dim=observation_dim,
            action_dim=env.num_actions, hidden_dims=teacher_hidden_dims, device=device,
        )
        noise = float(cfg["policy"]["init_noise_std"])
        if noise <= 0:
            raise ValueError("Student exploration std must be positive")
        policy = CR1RecurrentVisualPpoStudent(
            env.num_actions, list(cfg["policy"]["actor_hidden_dims"]), noise,
        )
        critic = elu_mlp(observation_dim, list(cfg["policy"]["critic_hidden_dims"]), 1)
        storage = RecurrentVisualPpoDaggerStorage(
            cfg["num_steps_per_env"], env.num_envs, env.num_actions, observation_dim, device,
        )
        return cls(
            policy, teacher, storage, critic=critic,
            teacher_checkpoint=teacher_checkpoint, device=device,
            termination_manager=getattr(getattr(env, "unwrapped", None), "termination_manager", None),
            **settings,
        )

    @property
    def ppo_coeff(self):
        if self.ppo_warmup_iterations == 0:
            return 1.0
        progress = min(max(self.update_iteration / self.ppo_warmup_iterations, 0.0), 1.0)
        return self.ppo_initial_coeff + progress * (1.0 - self.ppo_initial_coeff)

    @property
    def dagger_coeff(self):
        progress = min(max(
            (self.update_iteration - self.dagger_anneal_start)
            / self.dagger_anneal_iterations, 0.0,
        ), 1.0)
        return self.dagger_initial_coeff + progress * (
            self.dagger_final_coeff - self.dagger_initial_coeff
        )

    @torch.no_grad()
    def act(self, obs):
        if self._pending is not None:
            raise RuntimeError("process_env_step must follow every act call")
        self._validate_observations(obs, int(self.teacher[0].in_features))
        hidden = self._rollout_hidden
        if hidden is None:
            hidden = self.policy.initial_hidden(
                obs.batch_size[0], device=obs.device, dtype=obs[PROPRIO_KEY].dtype,
            )
        mean, _, next_hidden, activations = self.policy.action_mean_and_feature_step(
            self._actor_observations(obs), hidden,
        )
        std = self.policy.output_std.expand_as(mean)
        distribution = Normal(mean, std)
        student_action = distribution.sample()
        teacher_mean = self.teacher(obs[TEACHER_KEY])
        teacher_mask = self._teacher_controlled_mask(mean.shape[0])
        actions = torch.where(teacher_mask[:, None], teacher_mean, student_action)
        exploration = torch.where(
            teacher_mask[:, None], torch.zeros_like(mean), student_action - mean,
        )
        self._accumulate_rollout_metrics(
            obs, mean, teacher_mean, student_action, mean - teacher_mean,
            exploration, hidden, next_hidden, activations,
        )
        self._accumulate_controller_action_metrics(mean - teacher_mean, teacher_mask)
        self._rollout_hidden = next_hidden.detach()
        self._pending = RecurrentVisualMseDaggerTransition(
            proprio_history=obs[PROPRIO_KEY].detach(), depth=obs[DEPTH_KEY].detach(),
            depth_updated=obs[DEPTH_UPDATE_KEY].detach(),
            teacher_action_mean=teacher_mean.detach(), hidden_before=hidden.detach(),
            teacher_controlled=teacher_mask,
        )
        # Teacher slots are retained for diagnostics but excluded from every PPO term.
        self._ppo_pending = {
            "critic_observation": obs[TEACHER_KEY].detach().clone(),
            "actions": actions.detach(), "old_action_mean": mean.detach(),
            "old_action_std": std.detach(),
            "old_action_log_prob": distribution.log_prob(actions).sum(-1).detach(),
            "values": self.critic(obs[TEACHER_KEY]).squeeze(-1).detach(),
        }
        if (self._diagnostic_dir is not None and self._snapshot is None
                and self.update_iteration % self.diagnostic_snapshot_interval == 0):
            ids = self._diagnostic_env_ids(teacher_mask, min(
                self.diagnostic_sample_envs, mean.shape[0],
            ))
            tensors = {
                "teacher_controlled": teacher_mask, "proprio_history": obs[PROPRIO_KEY],
                "depth": obs[DEPTH_KEY], "depth_updated": obs[DEPTH_UPDATE_KEY],
                "teacher_observation": obs[TEACHER_KEY], "student_mean": mean,
                "teacher_mean": teacher_mean, "candidate_student_action": student_action,
                "executed_action": actions, **self._ppo_pending,
            }
            self._snapshot = {
                "iteration": self.update_iteration, "environment_ids": ids.cpu().clone(),
                "parameter_timing": "before_optimizer_update",
                # model_N.pt is saved AFTER update N; retain matching pre-update
                # weights here so the sampled action can be replayed exactly.
                "policy_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in self.policy.state_dict().items()
                },
                **{name: value[ids].cpu().clone() for name, value in tensors.items()},
                "hidden_before": hidden[:, ids].cpu().clone(),
                "hidden_after": next_hidden[:, ids].cpu().clone(),
                **{name: (value[:, ids] if name == "gru_candidate" else value[ids])
                   .detach().cpu().clone() for name, value in activations.items()},
            }
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        if self._ppo_pending is None:
            raise RuntimeError("act must precede process_env_step")
        time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool().clone()
        # An actual failure on a timeout step must remain terminal.
        if self._termination_manager is not None:
            time_outs &= ~self._termination_manager.terminated.bool()
        self.storage.add_rl(**self._ppo_pending, rewards=rewards, time_outs=time_outs)
        if self._snapshot is not None and "time_outs" not in self._snapshot:
            ids = self._snapshot["environment_ids"].to(self.device)
            self._snapshot["time_outs"] = time_outs[ids].detach().cpu().clone()
        self._ppo_pending = None
        super().process_env_step(obs, rewards, dones, extras)

    @torch.no_grad()
    def compute_returns(self, obs):
        self.storage.compute_returns(
            self.critic(obs[TEACHER_KEY]).squeeze(-1), self.gamma, self.lam,
        )
        self._returns_ready = True
        if self._snapshot is not None:
            ids = self._snapshot["environment_ids"].to(self.device)
            for name in ("returns", "advantages"):
                self._snapshot[name] = getattr(self.storage, name)[0, ids].cpu().clone()

    def _batch_losses(self, batch):
        mean, _, _ = self.policy.action_mean_and_feature_sequence(
            batch[PROPRIO_KEY], batch["depth"], batch[DEPTH_UPDATE_KEY],
            batch["dones"], batch["initial_hidden"],
        )
        mse = (mean - batch["teacher_action_mean"]).square().mean()
        valid = ~batch["teacher_controlled"]
        # Index BEFORE computing losses: teacher outliers/NaNs cannot contaminate PPO.
        if not valid.any():
            zero = mean.sum() * 0.0
            return dict(surrogate=zero, value=zero, entropy=zero, behavior=mse,
                        policy_kl=zero.detach(), ratio_error=zero.detach(),
                        clip_fraction=zero.detach(), mean=mean)
        new_mean = mean[valid]
        std = self.policy.output_std.expand_as(new_mean)
        distribution = Normal(new_mean, std)
        log_prob = distribution.log_prob(batch["actions"][valid]).sum(-1)
        ratio = (log_prob - batch["old_action_log_prob"][valid]).exp()
        advantage = batch["advantages"][valid]
        surrogate = torch.maximum(
            -advantage * ratio,
            -advantage * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param),
        ).mean()
        value = self.critic(batch["critic_observations"][valid]).squeeze(-1)
        target, old_value = batch["returns"][valid], batch["values"][valid]
        value_error = (value - target).square()
        if self.use_clipped_value_loss:
            clipped = old_value + (value - old_value).clamp(-self.clip_param, self.clip_param)
            value_error = torch.maximum(value_error, (clipped - target).square())
        with torch.no_grad():
            old_std, old_mean = batch["old_action_std"][valid], batch["old_action_mean"][valid]
            kl = (torch.log(std / old_std) + (
                old_std.square() + (old_mean - new_mean).square()
            ) / (2.0 * std.square()) - 0.5).sum(-1).mean()
        return {
            "surrogate": surrogate, "value": value_error.mean(),
            "entropy": distribution.entropy().sum(-1).mean(), "behavior": mse,
            "policy_kl": kl, "ratio_error": (ratio.detach() - 1.0).abs().max(),
            "clip_fraction": ((ratio.detach() - 1.0).abs() > self.clip_param).float().mean(),
            "mean": mean,
        }

    def _objective_gradient_metrics(self, terms):
        named = [(name, value) for name, value in self.policy.named_parameters()
                 if value.requires_grad]
        parameters = [value for _, value in named]
        ppo = torch.autograd.grad(terms["surrogate"], parameters, retain_graph=True, allow_unused=True)
        bc = torch.autograd.grad(self.dagger_coeff * terms["behavior"], parameters,
                                 retain_graph=True, allow_unused=True)
        result = {}
        dot = torch.zeros((), device=self.device)
        for prefix in self._module_prefixes():
            for objective, grads in (("ppo", ppo), ("weighted_dagger", bc)):
                total = torch.zeros((), device=self.device)
                for (name, _), grad in zip(named, grads):
                    if grad is not None and (prefix == "all" or name.startswith(prefix + ".")):
                        total += grad.detach().square().sum()
                result[f"{objective}_{prefix}_grad_l2_probe"] = float(total.sqrt())
            result[f"weighted_ppo_{prefix}_grad_l2_probe"] = (
                self.ppo_coeff * result[f"ppo_{prefix}_grad_l2_probe"]
            )
        for a, b in zip(ppo, bc):
            if a is not None and b is not None:
                dot += (a.detach() * b.detach()).sum()
        denominator = (result["ppo_all_grad_l2_probe"]
                       * result["weighted_dagger_all_grad_l2_probe"])
        result["ppo_dagger_gradient_cosine_probe"] = float(dot) / max(denominator, 1e-12)
        return result

    def update(self):
        if not self._returns_ready:
            raise RuntimeError("compute_returns must precede hybrid PPO update")
        before = {name: value.detach().clone() for name, value in self.policy.named_parameters()}
        critic_before = [p.detach().clone() for p in self.critic.parameters()]
        losses = self._finalize_rollout_metrics()
        coefficient = self.dagger_coeff
        ppo_coefficient = self.ppo_coeff
        totals = {name: 0.0 for name in ("surrogate", "value", "entropy", "behavior",
                                        "policy_kl", "clip_fraction", "total")}
        grad_sums = {name: 0.0 for name in self._module_prefixes()}
        grad_max = dict(grad_sums)
        group_errors = {name: [0.0, 0] for name in ("teacher", "student")}
        joint_squared_error = torch.zeros(self.policy.action_dim, device=self.device)
        training_samples = 0
        steps = 0
        for batch in self.storage.batches(self.num_mini_batches, self.num_learning_epochs):
            terms = self._batch_losses(batch)
            if steps == 0:
                losses["ppo_preupdate_ratio_max_abs_error_probe"] = float(terms["ratio_error"])
                losses.update(self._objective_gradient_metrics(terms))
            if self.schedule == "adaptive" and (~batch["teacher_controlled"]).any():
                kl = float(terms["policy_kl"])
                if kl > 2.0 * self.desired_kl:
                    self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
                elif 0.0 < kl < self.desired_kl / 2.0:
                    self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate
            # Warm up the policy objective (including entropy), not the critic.
            loss = (ppo_coefficient * (terms["surrogate"] - self.entropy_coef * terms["entropy"])
                    + self.value_loss_coef * terms["value"] + coefficient * terms["behavior"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite recurrent PPO/DAgger loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            critic_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True,
            )
            grads = self._gradient_norms()
            for name in grads:
                grad_sums[name] += grads[name]
                grad_max[name] = max(grad_max[name], grads[name])
            self.optimizer.step()
            for name in totals:
                totals[name] += float(loss.detach() if name == "total" else terms[name].detach())
            squared_error = (terms["mean"].detach() - batch["teacher_action_mean"]).square()
            joint_squared_error += squared_error.sum(dim=(0, 1))
            training_samples += squared_error.shape[0] * squared_error.shape[1]
            for name, mask in (("teacher", batch["teacher_controlled"]),
                               ("student", ~batch["teacher_controlled"])):
                group_errors[name][0] += float(squared_error[mask].sum())
                group_errors[name][1] += int(mask.sum()) * self.policy.action_dim
            steps += 1
        for name, total in totals.items():
            losses[name] = total / steps
        for name, (error, count) in group_errors.items():
            losses[f"training_{name}_controlled_action_rmse"] = (error / max(count, 1)) ** 0.5
        losses["training_action_rmse"] = losses["behavior"] ** 0.5
        for index, error in enumerate(joint_squared_error):
            losses[f"training_joint_{index:02d}_rmse"] = float((error / training_samples).sqrt())
        losses["dagger_coeff"] = coefficient
        losses["ppo_coeff"] = ppo_coefficient
        losses["weighted_ppo_surrogate_loss"] = ppo_coefficient * losses["surrogate"]
        losses["weighted_dagger_mse"] = coefficient * losses["behavior"]
        losses["weighted_value_loss"] = self.value_loss_coef * losses["value"]
        losses["effective_entropy_coef"] = ppo_coefficient * self.entropy_coef
        losses["weighted_entropy_loss"] = -losses["effective_entropy_coef"] * losses["entropy"]
        losses["weighted_ppo_actor_loss"] = (
            losses["weighted_ppo_surrogate_loss"] + losses["weighted_entropy_loss"]
        )
        losses["optimizer_steps"] = float(steps)
        losses["recurrent_sequence_length"] = float(self.storage.num_steps)
        losses["ppo_student_samples"] = float((~self.storage.teacher_controlled).sum())
        losses["ppo_teacher_samples"] = 0.0
        losses["critic_grad_l2_before_clip_last_batch"] = float(critic_norm)
        losses["critic_parameter_delta_l2"] = float(sum(
            (p.detach() - old).square().sum()
            for p, old in zip(self.critic.parameters(), critic_before)
        ).sqrt())
        valid = ~self.storage.teacher_controlled
        returns, values = self.storage.returns[valid], self.storage.values[valid]
        losses["student_return_mean"] = float(returns.mean())
        losses["student_value_mean"] = float(values.mean())
        losses["critic_explained_variance"] = float(
            1.0 - (returns - values).var(unbiased=False) / returns.var(unbiased=False).clamp_min(1e-8)
        )
        losses.update(self._parameter_delta_metrics(before))
        for name in grad_sums:
            losses[f"student_{name}_grad_l2_mean_after_clip"] = grad_sums[name] / steps
            losses[f"student_{name}_grad_l2_max_after_clip"] = grad_max[name]
        losses["student_action_std_min"] = float(self.policy.output_std.min().detach())
        losses["student_action_std_max"] = float(self.policy.output_std.max().detach())
        self._assert_teacher_frozen()
        if self._snapshot is not None and self._diagnostic_dir is not None:
            self._snapshot["update_metrics"] = losses
            torch.save(self._snapshot, self._diagnostic_dir /
                       f"iteration_{int(self._snapshot['iteration']):06d}_sample.pt")
        self._snapshot = None
        self.storage.clear()
        self._returns_ready = False
        self.update_iteration += 1
        self._reset_rollout_metrics()
        return losses

    def _training_contract(self):
        return {"num_steps_per_env": self.storage.num_steps, **{
            name: getattr(self, name) for name in (
            "bc_warmup_iterations", "teacher_rollout_fraction", "gamma", "lam",
            "clip_param", "value_loss_coef", "entropy_coef", "use_clipped_value_loss",
            "schedule", "desired_kl", "min_learning_rate", "max_learning_rate",
            "ppo_initial_coeff", "ppo_warmup_iterations",
            "dagger_initial_coeff", "dagger_final_coeff", "dagger_anneal_start",
            "dagger_anneal_iterations", "num_learning_epochs", "num_mini_batches",
            "max_grad_norm",
        )}}

    def configure_diagnostics(self, log_dir, manifest):
        self._diagnostic_dir = Path(log_dir) / "recurrent_visual_ppo_diagnostics"
        self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            **manifest, "data_contract": RECURRENT_VISUAL_PPO_DAGGER_DATA_CONTRACT,
            "training_contract": self._training_contract(),
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "student_network": repr(self.policy), "critic_network": repr(self.critic),
            "critic_input": "teacher privileged observation, training only",
            "critic_initialization": "scratch", "feature_normalization": "LayerNorm(128) per branch",
            "ppo_samples": "student controlled trajectories only",
            "dagger_samples": "all teacher and student trajectories, elementwise mean MSE",
            "advantage_normalization": "student samples only, rollout mean/std",
            "rollout_std": "learnable log_std, initial std from policy config",
            "timeout_bootstrap": "RSL current value bootstrap; exclude simultaneous true termination",
            "ppo_gradient_path": "raw observations -> encoders -> GRU -> actor (no detached features)",
            "objective_gradient_probe": (
                "first minibatch before each update: raw and alpha-weighted PPO surrogate "
                "(excluding entropy), lambda-weighted MSE; cosine uses raw PPO vs weighted MSE"
            ),
            "loss_schedule": (
                "alpha ramps PPO surrogate and entropy from ppo_initial_coeff to 1; "
                "critic remains unscaled; lambda anneals MSE independently; "
                "both use restored update_iteration, with no automatic performance gating"
            ),
            "gru_saturation_definition": "fraction of sampled elements with abs(value) >= 0.9999",
            "recurrent_training": {"time_axis_shuffled": False,
                                   "initial_hidden": "rollout_start_hidden",
                                   "done_mask_timing": "reset_before_t_plus_1",
                                   "gru_update_timing": "fresh_depth_frames_only"},
            "excluded_losses": ["teacher_feature_mse", "velocity", "reconstruction", "teacher_student_kl"],
        }
        (self._diagnostic_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def save(self):
        return {
            "data_contract": RECURRENT_VISUAL_PPO_DAGGER_DATA_CONTRACT,
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_iteration": self.update_iteration, "learning_rate": self.learning_rate,
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "training_contract": self._training_contract(),
        }

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if loaded_dict.get("data_contract") != RECURRENT_VISUAL_PPO_DAGGER_DATA_CONTRACT:
            raise RuntimeError("Incompatible recurrent visual PPO-DAgger checkpoint data contract")
        if loaded_dict.get("teacher_checkpoint_sha256") != self.teacher_checkpoint_sha256:
            raise RuntimeError("The recurrent PPO checkpoint used a different teacher")
        options = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg is not None:
            options.update(load_cfg)
        if options["optimizer"] and loaded_dict.get("training_contract") != self._training_contract():
            raise RuntimeError("The recurrent PPO checkpoint used a different training contract")
        if options["actor"]:
            self.policy.load_state_dict(loaded_dict["policy_state_dict"], strict=strict)
        if options["critic"]:
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if options["optimizer"]:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = float(loaded_dict["learning_rate"])
        if options["iteration"]:
            self.update_iteration = int(loaded_dict["update_iteration"])
        self._rollout_hidden = None
        self._pending = self._ppo_pending = self._snapshot = None
        self._returns_ready = False
        self.storage.clear()
        self.policy.reset()
        self._reset_rollout_metrics()
        self.teacher.eval().requires_grad_(False)
        return options["iteration"]

    def train_mode(self):
        super().train_mode()
        self.critic.train()

    def eval_mode(self):
        super().eval_mode()
        self.critic.eval()


__all__ = ["CR1RecurrentVisualPpoDagger"]
