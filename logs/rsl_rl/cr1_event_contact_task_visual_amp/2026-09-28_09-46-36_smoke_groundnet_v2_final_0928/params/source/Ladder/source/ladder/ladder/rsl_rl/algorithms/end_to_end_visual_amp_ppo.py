"""End-to-end visual PPO with a phase-free adversarial motion prior."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ladder.rsl_rl.algorithms.end_to_end_visual_ppo import EndToEndVisualPPO
from ladder.rsl_rl.amp import (
    AMPDiscriminator,
    AMPHistory,
    AMPReplayBuffer,
    AMPRunningMeanStd,
    AMP_STYLE_FEATURE_LAYOUT,
    AMP_STYLE_OBSERVATION_DIM,
)
from ladder.rsl_rl.runners.end_to_end_visual_amp_wrapper import AMP_STYLE_KEY


class EndToEndVisualAMPPPO(EndToEndVisualPPO):
    """PPO task learning plus least-squares AMP style rewards.

    The discriminator is optimized separately.  As in the AMP paper, the
    policy never receives discriminator gradients directly; PPO learns from
    the scalar style reward assigned to policy state transitions.
    """

    def __init__(
        self,
        *args,
        amp_task_reward_weight=1.0,
        amp_style_reward_weight=0.2,
        amp_reward_warmup_updates=500,
        amp_discriminator_learning_rate=1.0e-4,
        amp_discriminator_batch_size=4096,
        amp_discriminator_updates=4,
        amp_replay_capacity=131072,
        amp_gradient_penalty_coeff=10.0,
        amp_logit_regularization_coeff=0.0,
        amp_discriminator_weight_decay=1.0e-4,
        amp_discriminator_max_grad_norm=1.0,
        amp_anchor_hold_probability=0.20,
        amp_balance_directions=True,
        amp_num_frames=5,
        amp_history_stride=2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if amp_task_reward_weight <= 0.0 or amp_style_reward_weight < 0.0:
            raise ValueError("AMP task weight must be positive and style weight non-negative")
        if amp_reward_warmup_updates < 0:
            raise ValueError("AMP reward warmup updates cannot be negative")
        if amp_discriminator_batch_size < 1 or amp_discriminator_updates < 1:
            raise ValueError("AMP discriminator batch and update counts must be positive")
        if amp_discriminator_learning_rate <= 0.0:
            raise ValueError("AMP discriminator learning rate must be positive")
        if amp_gradient_penalty_coeff < 0.0 or amp_logit_regularization_coeff < 0.0:
            raise ValueError("AMP discriminator regularizers cannot be negative")
        if amp_discriminator_max_grad_norm <= 0.0:
            raise ValueError("AMP discriminator max grad norm must be positive")
        if not 0.0 <= amp_anchor_hold_probability <= 1.0:
            raise ValueError("AMP anchor hold probability must be in [0, 1]")

        if amp_num_frames < 2 or amp_history_stride < 1:
            raise ValueError("Invalid AMP history window")
        self.amp_num_frames = int(amp_num_frames)
        self.amp_history_stride = int(amp_history_stride)
        self.amp_task_reward_weight = float(amp_task_reward_weight)
        self.amp_style_reward_weight = float(amp_style_reward_weight)
        self.amp_reward_warmup_updates = int(amp_reward_warmup_updates)
        self.amp_discriminator_batch_size = int(amp_discriminator_batch_size)
        self.amp_discriminator_updates = int(amp_discriminator_updates)
        self.amp_gradient_penalty_coeff = float(amp_gradient_penalty_coeff)
        self.amp_logit_regularization_coeff = float(amp_logit_regularization_coeff)
        self.amp_discriminator_max_grad_norm = float(amp_discriminator_max_grad_norm)
        self.amp_anchor_hold_probability = float(amp_anchor_hold_probability)
        self.amp_balance_directions = bool(amp_balance_directions)
        self.amp_step_dt = None
        self.amp_training_updates = 0
        self._sample_amp_expert = None

        self.amp_discriminator = AMPDiscriminator(AMP_STYLE_OBSERVATION_DIM, num_frames=self.amp_num_frames).to(
            self.device
        )
        self.amp_normalizer = AMPRunningMeanStd(AMP_STYLE_OBSERVATION_DIM).to(
            self.device
        )
        self.amp_discriminator_optimizer = torch.optim.Adam(
            self.amp_discriminator.parameters(),
            lr=amp_discriminator_learning_rate,
            weight_decay=amp_discriminator_weight_decay,
        )
        transition_dim = self.amp_num_frames * AMP_STYLE_OBSERVATION_DIM
        self.amp_history = AMPHistory(self.storage.num_envs, AMP_STYLE_OBSERVATION_DIM,
                                      self.amp_num_frames, self.amp_history_stride, self.device)
        self.amp_replay = AMPReplayBuffer(
            amp_replay_capacity, transition_dim, self.device
        )
        shape = (self.storage.num_steps, self.storage.num_envs)
        self.amp_policy_transitions = torch.zeros(
            *shape, transition_dim, device=self.device
        )
        self.amp_transition_valid = torch.zeros(
            *shape, dtype=torch.bool, device=self.device
        )
        self.amp_style_rewards = torch.zeros(*shape, device=self.device)
        self.amp_task_rewards = torch.zeros(*shape, device=self.device)
        self._pending_amp_current = None
        self._amp_update_metrics = {}

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        algorithm = super().construct_algorithm(obs, env, cfg, device)
        if not hasattr(env, "sample_amp_expert_transitions"):
            raise TypeError("AMP PPO requires EndToEndVisualAMPPPOVecEnvWrapper")
        algorithm._sample_amp_expert = env.sample_amp_expert_transitions
        algorithm.amp_step_dt = float(env.unwrapped.step_dt)
        dataset = env.amp_reference_dataset
        if (dataset.num_frames, dataset.history_stride) != (algorithm.amp_num_frames, algorithm.amp_history_stride):
            raise RuntimeError("AMP expert/policy temporal windows differ")
        if dataset.observation_dim != AMP_STYLE_OBSERVATION_DIM:
            raise RuntimeError("Policy and expert AMP observation dimensions differ")
        if dataset.anchor_hold_probability != algorithm.amp_anchor_hold_probability:
            raise RuntimeError("AMP wrapper/config anchor hold probabilities differ")
        if dataset.balance_directions != algorithm.amp_balance_directions:
            raise RuntimeError("AMP wrapper/config direction balancing differs")
        return algorithm

    @classmethod
    def _validate_observations(cls, obs):
        super()._validate_observations(obs)
        count = obs["critic"].shape[0]
        if AMP_STYLE_KEY not in obs:
            raise KeyError(f"Missing AMP observation {AMP_STYLE_KEY!r}")
        expected = (count, AMP_STYLE_OBSERVATION_DIM)
        if tuple(obs[AMP_STYLE_KEY].shape) != expected:
            raise ValueError(
                f"{AMP_STYLE_KEY} must have shape {expected}, got "
                f"{tuple(obs[AMP_STYLE_KEY].shape)}"
            )

    def _effective_amp_style_weight(self) -> float:
        if self.amp_reward_warmup_updates == 0:
            return self.amp_style_reward_weight
        fraction = min(
            self.amp_training_updates / self.amp_reward_warmup_updates, 1.0
        )
        return self.amp_style_reward_weight * fraction

    @torch.no_grad()
    def act(self, obs):
        self._validate_observations(obs)
        actions = super().act(obs)
        self.amp_history.seed(obs[AMP_STYLE_KEY])
        self._pending_amp_current = obs[AMP_STYLE_KEY].detach().clone()
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        if self._pending_amp_current is None:
            raise RuntimeError("AMP act must precede process_env_step")
        index = self.storage.step
        done = dones.to(device=self.device, dtype=torch.bool)
        next_state = obs[AMP_STYLE_KEY].to(self.device)
        transition, valid = self.amp_history.append(next_state, done)
        self.amp_policy_transitions[index].copy_(transition)
        self.amp_transition_valid[index].copy_(valid)
        with torch.no_grad():
            normalized = self.amp_normalizer.normalize_transitions(transition)
            style_reward = self.amp_discriminator.reward(normalized)
            style_reward = torch.where(valid, style_reward, torch.zeros_like(style_reward))
        task_reward = rewards.to(self.device)
        self.amp_task_rewards[index].copy_(task_reward)
        self.amp_style_rewards[index].copy_(style_reward)
        combined_reward = (
            self.amp_task_reward_weight * task_reward
            + self._effective_amp_style_weight() * self.amp_step_dt * style_reward
        )
        super().process_env_step(obs, combined_reward, dones, extras)
        self._pending_amp_current = None

    def _update_amp_discriminator(self) -> dict[str, float]:
        policy = self.amp_policy_transitions[self.amp_transition_valid]
        if policy.shape[0] == 0:
            return {
                "amp_discriminator_updates": 0.0,
                "amp_valid_transition_fraction": 0.0,
            }
        self.amp_replay.add(policy)
        totals = {
            "amp_discriminator_loss": 0.0,
            "amp_expert_loss": 0.0,
            "amp_policy_loss": 0.0,
            "amp_gradient_penalty": 0.0,
            "amp_logit_regularization": 0.0,
            "amp_expert_prediction": 0.0,
            "amp_policy_prediction": 0.0,
            "amp_expert_accuracy": 0.0,
            "amp_policy_accuracy": 0.0,
        }
        max_gradient_norm = 0.0
        for _ in range(self.amp_discriminator_updates):
            policy_batch = self.amp_replay.sample(self.amp_discriminator_batch_size)
            expert_batch = self._sample_amp_expert(self.amp_discriminator_batch_size)
            with torch.no_grad():
                expert_frame = torch.randint(self.amp_num_frames, (1,), device=self.device)
                self.amp_normalizer.update_transitions(
                    torch.cat((policy_batch.reshape(-1, self.amp_num_frames, AMP_STYLE_OBSERVATION_DIM)[:, -1],
                               expert_batch.reshape(-1, self.amp_num_frames, AMP_STYLE_OBSERVATION_DIM)[:, expert_frame].squeeze(1)), dim=0)
                )
                policy_normalized = self.amp_normalizer.normalize_transitions(
                    policy_batch
                )
                expert_normalized = self.amp_normalizer.normalize_transitions(
                    expert_batch
                )
            policy_prediction = self.amp_discriminator(policy_normalized)
            expert_prediction = self.amp_discriminator(expert_normalized)
            policy_loss = F.mse_loss(
                policy_prediction, -torch.ones_like(policy_prediction)
            )
            expert_loss = F.mse_loss(
                expert_prediction, torch.ones_like(expert_prediction)
            )
            classification_loss = 0.5 * (policy_loss + expert_loss)
            gradient_penalty = self.amp_discriminator.gradient_penalty(
                expert_normalized
            )
            logit_regularization = 0.5 * (
                policy_prediction.square().mean()
                + expert_prediction.square().mean()
            )
            loss = (
                classification_loss
                + self.amp_gradient_penalty_coeff * gradient_penalty
                + self.amp_logit_regularization_coeff * logit_regularization
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite AMP discriminator loss")
            self.amp_discriminator_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                self.amp_discriminator.parameters(),
                self.amp_discriminator_max_grad_norm,
                error_if_nonfinite=True,
            )
            self.amp_discriminator_optimizer.step()
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            values = {
                "amp_discriminator_loss": loss,
                "amp_expert_loss": expert_loss,
                "amp_policy_loss": policy_loss,
                "amp_gradient_penalty": gradient_penalty,
                "amp_logit_regularization": logit_regularization,
                "amp_expert_prediction": expert_prediction.mean(),
                "amp_policy_prediction": policy_prediction.mean(),
                "amp_expert_accuracy": (expert_prediction > 0.0).float().mean(),
                "amp_policy_accuracy": (policy_prediction < 0.0).float().mean(),
            }
            for name, value in values.items():
                totals[name] += float(value.detach())
        updates = float(self.amp_discriminator_updates)
        return {
            **{name: value / updates for name, value in totals.items()},
            "amp_discriminator_updates": updates,
            "amp_discriminator_grad_norm_before_clip_max": max_gradient_norm,
            "amp_replay_size": float(self.amp_replay.size),
            "amp_valid_transition_fraction": float(
                self.amp_transition_valid.float().mean()
            ),
        }

    def update(self):
        if not self._returns_ready:
            raise RuntimeError("compute_returns must precede update")
        self._amp_update_metrics = self._update_amp_discriminator()
        self._amp_update_metrics.update(
            {
                "amp_task_reward_mean": float(self.amp_task_rewards.mean()),
                "amp_style_reward_mean": float(self.amp_style_rewards.mean()),
                "amp_style_reward_nonterminal_mean": float(
                    self.amp_style_rewards.sum() / self.amp_transition_valid.sum().clamp_min(1)
                ),
                "amp_weighted_style_reward_mean": float(self.amp_style_rewards.mean())
                    * self._effective_amp_style_weight() * self.amp_step_dt,
                "amp_effective_style_reward_weight": self._effective_amp_style_weight(),
                "amp_normalizer_count": float(self.amp_normalizer.count),
            }
        )
        result = super().update()
        self.amp_training_updates += 1
        self.amp_transition_valid.zero_()
        self.amp_style_rewards.zero_()
        self.amp_task_rewards.zero_()
        return result

    def _additional_update_metrics(self) -> dict[str, float]:
        return dict(self._amp_update_metrics)

    def _diagnostic_snapshot(self):
        snapshot = super()._diagnostic_snapshot()
        if snapshot is not None:
            count = min(16, self.storage.num_envs)
            snapshot.update({
                "amp_contract": self._amp_contract(),
                "amp_policy_windows": self.amp_policy_transitions[:, :count].detach().cpu().clone(),
                "amp_valid": self.amp_transition_valid[:, :count].cpu().clone(),
                "amp_task_rewards": self.amp_task_rewards[:, :count].cpu().clone(),
                "amp_style_rewards": self.amp_style_rewards[:, :count].cpu().clone(),
                "amp_state_timing": "after discriminator update; rewards computed before update",
                "amp_discriminator": {k: v.detach().cpu().clone() for k, v in self.amp_discriminator.state_dict().items()},
                "amp_normalizer": {k: v.detach().cpu().clone() for k, v in self.amp_normalizer.state_dict().items()},
            })
        return snapshot

    def _amp_contract(self) -> dict:
        return {
            "algorithm": "ssr_least_squares_amp_history_v2",
            "num_frames": self.amp_num_frames,
            "history_stride": self.amp_history_stride,
            "normalization": "ssr_rolling100_unbiased_var_epsilon1e-4_clip10",
            "step_dt": self.amp_step_dt,
            "feature_layout": AMP_STYLE_FEATURE_LAYOUT,
            "feature_dim": AMP_STYLE_OBSERVATION_DIM,
            "task_reward_weight": self.amp_task_reward_weight,
            "style_reward_weight": self.amp_style_reward_weight,
            "reward_warmup_updates": self.amp_reward_warmup_updates,
            "discriminator_batch_size": self.amp_discriminator_batch_size,
            "discriminator_updates": self.amp_discriminator_updates,
            "gradient_penalty_coeff": self.amp_gradient_penalty_coeff,
            "logit_regularization_coeff": self.amp_logit_regularization_coeff,
            "anchor_hold_probability": self.amp_anchor_hold_probability,
            "balance_directions": self.amp_balance_directions,
        }

    def save(self):
        payload = super().save()
        payload.update(
            {
                "amp_contract": self._amp_contract(),
                "amp_discriminator_state_dict": self.amp_discriminator.state_dict(),
                "amp_discriminator_optimizer_state_dict": (
                    self.amp_discriminator_optimizer.state_dict()
                ),
                "amp_normalizer_state_dict": self.amp_normalizer.state_dict(),
                "amp_training_updates": self.amp_training_updates,
            }
        )
        return payload

    def load(self, loaded_dict, load_cfg=None, strict=True):
        self.amp_history = AMPHistory(self.storage.num_envs, AMP_STYLE_OBSERVATION_DIM,
                                      self.amp_num_frames, self.amp_history_stride, self.device)
        iteration_loaded = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is not None and not any(load_cfg.get(key, True) for key in ("critic", "optimizer", "iteration")):
            # Deployment uses only the actor; training-only AMP settings need
            # not match a small diagnostic run or a later tuning experiment.
            return iteration_loaded
        amp_contract = loaded_dict.get("amp_contract")
        if amp_contract is None:
            print(
                "[AMP] Initializing a fresh discriminator from a non-AMP PPO checkpoint.",
                flush=True,
            )
            self.amp_training_updates = 0
            return iteration_loaded
        if amp_contract != self._amp_contract():
            raise RuntimeError("AMP checkpoint training contract differs")
        self.amp_discriminator.load_state_dict(
            loaded_dict["amp_discriminator_state_dict"], strict=strict
        )
        self.amp_normalizer.load_state_dict(
            loaded_dict["amp_normalizer_state_dict"], strict=strict
        )
        load_optimizer = load_cfg is None or load_cfg.get("optimizer", True)
        if load_optimizer:
            self.amp_discriminator_optimizer.load_state_dict(
                loaded_dict["amp_discriminator_optimizer_state_dict"]
            )
        load_iteration = load_cfg is None or load_cfg.get("iteration", True)
        if load_iteration:
            self.amp_training_updates = int(loaded_dict["amp_training_updates"])
        return iteration_loaded

    def train_mode(self):
        super().train_mode()
        self.amp_discriminator.train()

    def eval_mode(self):
        super().eval_mode()
        self.amp_discriminator.eval()


__all__ = ["EndToEndVisualAMPPPO"]
