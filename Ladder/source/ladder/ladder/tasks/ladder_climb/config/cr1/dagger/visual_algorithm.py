"""Visual Hybrid DAgger + PPO with reference-free perception heads."""

from __future__ import annotations

from itertools import chain

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from ladder.rsl_rl.algorithms.transformer_vision_ppo import TransformerVisionPPO
from ladder.rsl_rl.modules import TransformerVisionTargets
from ladder.rsl_rl.storage import (
    TransformerVisionPPORolloutStorage,
    TransformerVisionPPOTransition,
)
from ladder.rsl_rl.storage.transformer_vision_ppo_storage import (
    CRITIC_KEY,
    HEIGHTMAP_HIT_CLASS_KEY,
    HEIGHTMAP_KEY,
    HEIGHTMAP_VALID_KEY,
    VELOCITY_KEY,
)

from .student_teacher import TEACHER_KEY, elu_mlp, load_frozen_teacher
from .visual_model import (
    DEPTH_KEY,
    PROPRIO_KEY,
    VISUAL_DATA_CONTRACT,
    CR1DaggerVisualStudent,
)


class CR1VisualHybridDaggerPPO(TransformerVisionPPO):
    """Student-rollout PPO plus teacher action supervision for head and encoder."""

    def __init__(self, *args, teacher: nn.Module, dagger_initial_coeff: float,
                 dagger_final_coeff: float, dagger_anneal_start: int,
                 dagger_anneal_iterations: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval().requires_grad_(False)
        self.dagger_initial_coeff = dagger_initial_coeff
        self.dagger_final_coeff = dagger_final_coeff
        self.dagger_anneal_start = dagger_anneal_start
        self.dagger_anneal_iterations = dagger_anneal_iterations
        self.update_iteration = 0

    @property
    def dagger_coeff(self) -> float:
        if self.update_iteration <= self.dagger_anneal_start:
            return self.dagger_initial_coeff
        if self.update_iteration >= self.dagger_anneal_iterations:
            return self.dagger_final_coeff
        alpha = (self.update_iteration - self.dagger_anneal_start) / (
            self.dagger_anneal_iterations - self.dagger_anneal_start
        )
        return self.dagger_initial_coeff + alpha * (
            self.dagger_final_coeff - self.dagger_initial_coeff
        )

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str):
        CR1VisualHybridDaggerPPO._validate_visual_dagger_observations(obs)
        policy_cfg = cfg["policy"]
        algorithm_cfg = dict(cfg["algorithm"])
        for key in ("class_name", "rnd_cfg", "symmetry_cfg", "optimizer"):
            algorithm_cfg.pop(key, None)
        teacher_checkpoint = algorithm_cfg.pop("teacher_checkpoint")
        if not teacher_checkpoint:
            raise ValueError("Visual DAgger requires teacher_checkpoint")
        policy = CR1DaggerVisualStudent(
            env.num_actions,
            list(policy_cfg["actor_hidden_dims"]),
            float(policy_cfg["init_noise_std"]),
        )
        teacher = load_frozen_teacher(
            teacher_checkpoint,
            observation_dim=int(obs[TEACHER_KEY].shape[-1]),
            action_dim=env.num_actions,
            hidden_dims=list(algorithm_cfg.pop("teacher_hidden_dims")),
            device=device,
        )
        critic = elu_mlp(
            int(obs[CRITIC_KEY].shape[-1]),
            list(policy_cfg["critic_hidden_dims"]),
            1,
        )
        storage = TransformerVisionPPORolloutStorage(
            int(cfg["num_steps_per_env"]),
            env.num_envs,
            env.num_actions,
            int(obs[CRITIC_KEY].shape[-1]),
            device,
            actor_input_dim=460,
            proprio_history_frames=5,
            teacher_supervision=True,
        )
        dagger_args = {
            name: algorithm_cfg.pop(name)
            for name in (
                "dagger_initial_coeff",
                "dagger_final_coeff",
                "dagger_anneal_start",
                "dagger_anneal_iterations",
            )
        }
        algorithm_cfg.pop("visual_architecture", None)
        algorithm_cfg.pop("visual_initialization", None)
        algorithm_cfg.pop("visual_checkpoint", None)
        return CR1VisualHybridDaggerPPO(
            policy,
            critic,
            storage,
            teacher=teacher,
            device=device,
            visual_checkpoint=None,
            **dagger_args,
            **algorithm_cfg,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self._pending is not None:
            raise RuntimeError("process_env_step must follow every act call")
        self._validate_visual_dagger_observations(obs)
        if self.storage.step == 0:
            self.storage.begin_rollout(self.visual_hidden_state)
        self.policy.encoder.eval()
        with torch.no_grad():
            output = self.policy.encoder(
                obs[PROPRIO_KEY],
                obs[DEPTH_KEY],
                hidden_state=self.visual_hidden_state,
            )
            actor_input = self.policy.actor_input(obs[PROPRIO_KEY], output)
            mean = self.policy.action_mean(actor_input)
            std = self.policy.output_std.expand_as(mean)
            distribution = Normal(mean, std)
            actions = distribution.sample()
            value = self.critic(obs[CRITIC_KEY])
            teacher_mean = self.teacher(obs[TEACHER_KEY])
        self.visual_hidden_state = output.next_hidden_state.detach()
        self._pending = TransformerVisionPPOTransition(
            proprio_history=obs[PROPRIO_KEY].detach(),
            depth_history=obs[DEPTH_KEY].detach(),
            critic_observation=obs[CRITIC_KEY].detach(),
            actor_input=actor_input.detach(),
            actions=actions.detach(),
            old_action_mean=mean.detach(),
            old_action_std=std.detach(),
            old_action_log_prob=distribution.log_prob(actions).sum(-1, keepdim=True).detach(),
            value=value.detach(),
            velocity_target=obs[VELOCITY_KEY].detach(),
            limb_heightmaps=obs[HEIGHTMAP_KEY].detach(),
            heightmap_valid=obs[HEIGHTMAP_VALID_KEY].detach(),
            heightmap_hit_class=obs[HEIGHTMAP_HIT_CLASS_KEY].detach(),
            teacher_action_mean=teacher_mean.detach(),
        )
        return actions

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        self._validate_visual_dagger_observations(obs)
        if self._pending is None:
            raise RuntimeError("act must be called before process_env_step")
        reward = rewards.detach().clone().to(self.device)
        done = dones.to(device=self.device, dtype=torch.bool)
        if "time_outs" in extras:
            reward += (
                self.gamma
                * self._pending.value.squeeze(-1)
                * extras["time_outs"].to(device=self.device, dtype=reward.dtype)
            )
        self.storage.add(
            self._pending,
            reward=reward,
            done=done,
            next_proprio=None,
        )
        self.visual_hidden_state = self.policy.encoder.reset_hidden_state(
            self.visual_hidden_state, done
        )
        self._pending = None

    def compute_returns(self, obs: TensorDict) -> None:
        self._validate_visual_dagger_observations(obs)
        with torch.no_grad():
            last_value = self.critic(obs[CRITIC_KEY])
        self.storage.compute_returns(last_value, self.gamma, self.lam)

    def act_inference(self, obs: TensorDict, dones=None) -> torch.Tensor:
        if dones is not None:
            self.policy.reset(dones)
        with torch.no_grad():
            return self.policy(obs)

    def _ppo_update(self) -> dict[str, float]:
        totals = {
            "value": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "policy_kl": 0.0,
            "dagger_action_mse": 0.0,
        }
        updates = 0
        coefficient = self.dagger_coeff
        for batch in self.storage.ppo_batches(self.num_mini_batches, self.num_learning_epochs):
            mean = self.policy.action_mean(batch["actor_inputs"])
            std = self.policy.output_std.expand_as(mean)
            distribution = Normal(mean, std)
            log_prob = distribution.log_prob(batch["actions"]).sum(-1, keepdim=True)
            entropy = distribution.entropy().sum(-1).mean()
            value = self.critic(batch["critic_observations"])
            ratio = (log_prob - batch["old_action_log_prob"]).exp()
            surrogate = -batch["advantages"] * ratio
            surrogate_clipped = -batch["advantages"] * ratio.clamp(
                1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch["values"] + (value - batch["values"]).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (value - batch["returns"]).square(),
                    (value_clipped - batch["returns"]).square(),
                ).mean()
            else:
                value_loss = (value - batch["returns"]).square().mean()
            dagger_loss = (mean - batch["teacher_action_mean"]).square().mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
                + coefficient * dagger_loss
            )
            self.ppo_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                chain(
                    self.policy.action_mlp.parameters(),
                    (self.policy.log_std,),
                    self.critic.parameters(),
                ),
                self.max_grad_norm,
            )
            self.ppo_optimizer.step()
            with torch.no_grad():
                self.policy.log_std.clamp_(min=torch.tensor(0.2).log().item())
                kl = self._normal_kl(
                    batch["old_action_mean"], batch["old_action_std"], mean, std
                ).mean()
            self._adapt_learning_rate(
                batch["old_action_mean"], batch["old_action_std"], mean, std
            )
            for name, value_item in (
                ("value", value_loss),
                ("surrogate", surrogate_loss),
                ("entropy", entropy),
                ("policy_kl", kl),
                ("dagger_action_mse", dagger_loss),
            ):
                totals[name] += float(value_item.detach())
            updates += 1
        result = {name: value / updates for name, value in totals.items()}
        result["dagger_coeff"] = coefficient
        result["student_teacher_action_rmse"] = result["dagger_action_mse"] ** 0.5
        return result

    def _auxiliary_update(self) -> dict[str, float]:
        result = super()._auxiliary_update()
        coefficient = self.dagger_coeff
        reset_before = self.storage.reset_before_mask()
        total = 0.0
        batches = 0
        for env_ids in self.storage.auxiliary_logical_env_batches(
            self.num_mini_batches, 1
        ):
            self.auxiliary_optimizer.zero_grad(set_to_none=True)
            self.policy.action_mlp.zero_grad(set_to_none=True)
            output = self.policy.encoder(
                self.storage.proprio_history[:, env_ids],
                self.storage.depth_history[:, env_ids],
                hidden_state=self.storage.initial_hidden_state[:, env_ids],
                dones=reset_before[:, env_ids],
                masks=torch.ones_like(self.storage.dones[:, env_ids]),
            )
            actor_input = self.policy.actor_input(
                self.storage.proprio_history[:, env_ids], output
            )
            mean = self.policy.action_mean(actor_input)
            imitation = (
                mean - self.storage.teacher_action_mean[:, env_ids]
            ).square().mean()
            (coefficient * imitation).backward()
            nn.utils.clip_grad_norm_(self.policy.encoder.parameters(), self.max_grad_norm)
            self.auxiliary_optimizer.step()
            total += float(imitation.detach())
            batches += 1
        self.policy.action_mlp.zero_grad(set_to_none=True)
        result["aux_dagger_encoder_mse"] = total / batches
        return result

    def update(self) -> dict[str, float]:
        result = super().update()
        self.update_iteration += 1
        return result

    def save(self) -> dict:
        result = super().save()
        result["data_contract"] = VISUAL_DATA_CONTRACT
        result["update_iteration"] = self.update_iteration
        return result

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        loaded = super().load(loaded_dict, load_cfg, strict)
        self.update_iteration = int(loaded_dict.get("update_iteration", 0))
        return loaded

    @staticmethod
    def _validate_visual_dagger_observations(obs: TensorDict) -> None:
        required = (
            PROPRIO_KEY,
            DEPTH_KEY,
            TEACHER_KEY,
            CRITIC_KEY,
            HEIGHTMAP_KEY,
            HEIGHTMAP_VALID_KEY,
            HEIGHTMAP_HIT_CLASS_KEY,
            VELOCITY_KEY,
        )
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f"Missing visual DAgger observations: {missing}")
        batch = obs[PROPRIO_KEY].shape[0]
        if tuple(obs[PROPRIO_KEY].shape) != (batch, 5, 85):
            raise ValueError("Visual DAgger proprio history must have shape [B,5,85]")
        if tuple(obs[DEPTH_KEY].shape) != (batch, 2, 60, 80):
            raise ValueError("Visual DAgger depth history must have shape [B,2,60,80]")


__all__ = ["CR1VisualHybridDaggerPPO"]
