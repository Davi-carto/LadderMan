"""Asymmetric PPO with a separately optimized recurrent TransformerVision encoder."""

from __future__ import annotations

import math
from itertools import chain
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from ladder.rsl_rl.modules import (
    TaskGoalTransformerVisionLossWeights,
    TransformerVisionActor,
    TransformerVisionLossWeights,
    TransformerVisionTargets,
)
from ladder.rsl_rl.modules.transformer_vision_actor import (
    CURRENT_PROPRIO_KEY,
    DEPTH_HISTORY_KEY,
    PROPRIO_HISTORY_KEY,
)
from ladder.rsl_rl.storage.transformer_vision_ppo_storage import (
    CONTACT_INDICATOR_KEY,
    CRITIC_KEY,
    HEIGHTMAP_HIT_CLASS_KEY,
    HEIGHTMAP_KEY,
    HEIGHTMAP_VALID_KEY,
    LIMB_GOAL_KEY,
    LIMB_GOAL_VALID_KEY,
    MOTION_JOINT_POS_KEY,
    MOTION_ROOT_ORIENTATION_KEY,
    MOTION_TARGET_VALID_KEY,
    NEXT_PROPRIO_KEY,
    VELOCITY_KEY,
    TransformerVisionPPORolloutStorage,
    TransformerVisionPPOTransition,
)

def _elu_mlp(input_dim: int, hidden_dims: tuple[int, ...] | list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class TransformerVisionPPO:
    """PPO over fixed rollout latents plus temporal visual auxiliary updates.

    The action PPO optimizer never owns TransformerVision parameters. During
    rollout the exact ``[o_t, latent_t]`` is stored, so PPO ratios do not rerun
    the GRU. The visual
    optimizer replays full per-environment sequences with the stored rollout
    start state and done-aligned resets.
    """

    def __init__(
        self,
        policy: TransformerVisionActor,
        critic: nn.Module,
        storage: TransformerVisionPPORolloutStorage,
        *,
        num_learning_epochs: int,
        num_mini_batches: int,
        clip_param: float,
        gamma: float,
        lam: float,
        value_loss_coef: float,
        entropy_coef: float,
        learning_rate: float,
        min_learning_rate: float,
        max_learning_rate: float,
        auxiliary_learning_rate: float,
        auxiliary_mini_batch_envs: int,
        auxiliary_num_epochs: int,
        max_grad_norm: float,
        use_clipped_value_loss: bool,
        schedule: str,
        desired_kl: float | None,
        use_grad_scaler: bool,
        normalize_advantage_per_mini_batch: bool,
        visual_checkpoint: str | None,
        device: str | torch.device,
    ) -> None:
        if schedule not in ("adaptive", "fixed"):
            raise ValueError(f"Unsupported PPO schedule {schedule!r}")
        if auxiliary_mini_batch_envs <= 0 or auxiliary_num_epochs <= 0:
            raise ValueError("Auxiliary batch size and epoch count must be positive")
        if min_learning_rate <= 0.0 or max_learning_rate < min_learning_rate:
            raise ValueError("PPO learning-rate bounds must satisfy 0 < min <= max")
        if not min_learning_rate <= learning_rate <= max_learning_rate:
            raise ValueError("Initial PPO learning rate must lie inside its configured bounds")
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.critic = critic.to(self.device)
        self.storage = storage
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.auxiliary_learning_rate = auxiliary_learning_rate
        self.auxiliary_mini_batch_envs = auxiliary_mini_batch_envs
        self.auxiliary_num_epochs = auxiliary_num_epochs
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.use_grad_scaler = use_grad_scaler
        self._auxiliary_amp_enabled = use_grad_scaler and self.device.type == "cuda"
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.visual_checkpoint = visual_checkpoint
        self.data_contract = policy.encoder.data_contract
        self.rnd = None
        self.intrinsic_rewards = None
        self.visual_hidden_state = torch.zeros(1, storage.num_envs, 256, device=self.device)
        self.inference_hidden_state = torch.zeros_like(self.visual_hidden_state)
        self._pending: TransformerVisionPPOTransition | None = None

        ppo_parameters = [
            *self.policy.action_mlp.parameters(),
            self.policy.log_std,
            *self.critic.parameters(),
        ]
        auxiliary_parameters = list(self.policy.encoder.parameters())
        if {id(parameter) for parameter in ppo_parameters}.intersection(
            id(parameter) for parameter in auxiliary_parameters
        ):
            raise RuntimeError("PPO and visual auxiliary optimizers must not share parameters")
        self.ppo_optimizer = torch.optim.Adam(ppo_parameters, lr=learning_rate)
        self.auxiliary_optimizer = torch.optim.Adam(auxiliary_parameters, lr=auxiliary_learning_rate)
        self.auxiliary_grad_scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=self._auxiliary_amp_enabled,
        )

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> "TransformerVisionPPO":
        if not isinstance(obs, TensorDict):
            raise TypeError("TransformerVisionPPO requires a TensorDict vector environment")
        algorithm_cfg = dict(cfg["algorithm"])
        visual_architecture = str(
            algorithm_cfg.get("visual_architecture", "deterministic")
        )
        TransformerVisionPPO._validate_observations(
            obs, require_task_goals=visual_architecture.startswith("task_goal")
        )
        if cfg.get("multi_gpu") is not None:
            raise NotImplementedError("TransformerVisionPPO currently supports one GPU")
        algorithm_cfg.pop("class_name", None)
        if algorithm_cfg.pop("rnd_cfg", None) is not None:
            raise NotImplementedError("RND is not implemented for TransformerVisionPPO")
        if algorithm_cfg.pop("symmetry_cfg", None) is not None:
            raise NotImplementedError("Symmetry augmentation is not implemented for TransformerVisionPPO")
        algorithm_cfg.pop("optimizer", None)
        policy_cfg = cfg["policy"]
        algorithm_cfg.pop("visual_architecture", None)
        policy = TransformerVisionActor(
            action_dim=env.num_actions,
            actor_hidden_dims=policy_cfg["actor_hidden_dims"],
            init_noise_std=float(policy_cfg["init_noise_std"]),
            encoder_variant=visual_architecture,
        )
        visual_initialization = str(
            algorithm_cfg.pop("visual_initialization", "checkpoint")
        )
        if visual_initialization not in ("checkpoint", "scratch"):
            raise ValueError(
                "visual_initialization must be 'checkpoint' or 'scratch', got "
                f"{visual_initialization!r}"
            )
        visual_checkpoint = algorithm_cfg.pop("visual_checkpoint", None)
        if visual_initialization == "scratch" and visual_checkpoint:
            raise ValueError(
                "Scratch TransformerVision PPO must not load a visual pretraining checkpoint"
            )
        if visual_checkpoint:
            TransformerVisionPPO._load_visual_pretrain(policy, visual_checkpoint)
        elif visual_initialization == "scratch":
            print(
                "[TRANSFORMER VISION PPO] actor, critic, and visual encoder initialized from scratch",
                flush=True,
            )
        critic = _elu_mlp(
            int(obs[CRITIC_KEY].shape[-1]),
            policy_cfg["critic_hidden_dims"],
            1,
        )
        storage = TransformerVisionPPORolloutStorage(
            num_steps=int(cfg["num_steps_per_env"]),
            num_envs=env.num_envs,
            action_dim=env.num_actions,
            critic_dim=int(obs[CRITIC_KEY].shape[-1]),
            device=device,
            task_goal_supervision=visual_architecture.startswith("task_goal"),
            actor_input_dim=policy.actor_input_dim,
        )
        return TransformerVisionPPO(
            policy,
            critic,
            storage,
            device=device,
            visual_checkpoint=visual_checkpoint,
            num_learning_epochs=int(algorithm_cfg.pop("num_learning_epochs")),
            num_mini_batches=int(algorithm_cfg.pop("num_mini_batches")),
            clip_param=float(algorithm_cfg.pop("clip_param")),
            gamma=float(algorithm_cfg.pop("gamma")),
            lam=float(algorithm_cfg.pop("lam")),
            value_loss_coef=float(algorithm_cfg.pop("value_loss_coef")),
            entropy_coef=float(algorithm_cfg.pop("entropy_coef")),
            learning_rate=float(algorithm_cfg.pop("learning_rate")),
            min_learning_rate=float(algorithm_cfg.pop("min_learning_rate", 1.0e-5)),
            max_learning_rate=float(algorithm_cfg.pop("max_learning_rate", 1.0e-3)),
            auxiliary_learning_rate=float(algorithm_cfg.pop("auxiliary_learning_rate")),
            auxiliary_mini_batch_envs=int(algorithm_cfg.pop("auxiliary_mini_batch_envs")),
            auxiliary_num_epochs=int(algorithm_cfg.pop("auxiliary_num_epochs")),
            max_grad_norm=float(algorithm_cfg.pop("max_grad_norm")),
            use_clipped_value_loss=bool(algorithm_cfg.pop("use_clipped_value_loss")),
            schedule=str(algorithm_cfg.pop("schedule")),
            desired_kl=algorithm_cfg.pop("desired_kl"),
            use_grad_scaler=bool(algorithm_cfg.pop("use_grad_scaler", True)),
            normalize_advantage_per_mini_batch=bool(
                algorithm_cfg.pop("normalize_advantage_per_mini_batch", False)
            ),
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Compute one latent, sample one action, and retain the exact rollout values."""

        if self._pending is not None:
            raise RuntimeError("process_env_step must follow every act call")
        self._validate_observations(
            obs, require_task_goals=self.policy.encoder_variant == "task_goal"
        )
        if self.storage.step == 0:
            self.storage.begin_rollout(self.visual_hidden_state)
        self.policy.encoder.eval()
        with torch.no_grad():
            visual_output = self.policy.encoder(
                obs[PROPRIO_HISTORY_KEY],
                obs[DEPTH_HISTORY_KEY],
                hidden_state=self.visual_hidden_state,
                deterministic=False,
            )
            actor_input = torch.cat((obs[CURRENT_PROPRIO_KEY], visual_output.latent), dim=-1)
            action_mean = self.policy.action_mean(actor_input)
            action_std = self.policy.output_std.expand_as(action_mean)
            distribution = Normal(action_mean, action_std)
            actions = distribution.sample()
            action_log_prob = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
            value = self.critic(obs[CRITIC_KEY])
        self.visual_hidden_state = visual_output.next_hidden_state.detach()
        self._pending = TransformerVisionPPOTransition(
            proprio_history=obs[PROPRIO_HISTORY_KEY].detach(),
            depth_history=obs[DEPTH_HISTORY_KEY].detach(),
            critic_observation=obs[CRITIC_KEY].detach(),
            actor_input=actor_input.detach(),
            actions=actions.detach(),
            old_action_mean=action_mean.detach(),
            old_action_std=action_std.detach(),
            old_action_log_prob=action_log_prob.detach(),
            value=value.detach(),
            velocity_target=obs[VELOCITY_KEY].detach(),
            limb_heightmaps=obs[HEIGHTMAP_KEY].detach(),
            heightmap_valid=obs[HEIGHTMAP_VALID_KEY].detach(),
            heightmap_hit_class=obs[HEIGHTMAP_HIT_CLASS_KEY].detach(),
            limb_goal_target=(
                obs[LIMB_GOAL_KEY].detach() if LIMB_GOAL_KEY in obs else None
            ),
            limb_goal_valid=(
                obs[LIMB_GOAL_VALID_KEY].detach()
                if LIMB_GOAL_VALID_KEY in obs
                else None
            ),
            contact_indicator_target=(
                obs[CONTACT_INDICATOR_KEY].detach()
                if CONTACT_INDICATOR_KEY in obs
                else None
            ),
            motion_joint_pos_target=(
                obs[MOTION_JOINT_POS_KEY].detach()
                if MOTION_JOINT_POS_KEY in obs
                else None
            ),
            motion_root_orientation_target=(
                obs[MOTION_ROOT_ORIENTATION_KEY].detach()
                if MOTION_ROOT_ORIENTATION_KEY in obs
                else None
            ),
            motion_target_valid=(
                obs[MOTION_TARGET_VALID_KEY].detach()
                if MOTION_TARGET_VALID_KEY in obs
                else None
            ),
        )
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Commit the transition and reset only done environments' GRU columns."""

        if self._pending is None:
            raise RuntimeError("act must be called before process_env_step")
        self._validate_observations(
            obs, require_task_goals=self.policy.encoder_variant == "task_goal"
        )
        reward = rewards.detach().clone().to(self.device)
        done = dones.to(device=self.device, dtype=torch.bool)
        if "time_outs" in extras:
            timeout = extras["time_outs"].to(device=self.device, dtype=reward.dtype)
            reward += self.gamma * self._pending.value.squeeze(-1) * timeout
        self.storage.add(
            self._pending,
            reward=reward,
            done=done,
            next_proprio=obs[NEXT_PROPRIO_KEY].detach(),
        )
        self.visual_hidden_state = self.policy.encoder.reset_hidden_state(self.visual_hidden_state, done)
        self._pending = None

    def compute_returns(self, obs: TensorDict) -> None:
        self._validate_observations(
            obs, require_task_goals=self.policy.encoder_variant == "task_goal"
        )
        with torch.no_grad():
            last_value = self.critic(obs[CRITIC_KEY])
        self.storage.compute_returns(last_value, self.gamma, self.lam)

    def update(self) -> dict[str, float]:
        if self.storage.step != self.storage.num_steps:
            raise RuntimeError("A complete rollout is required before update")
        fixed_log_prob_error = float(self.rollout_log_prob_max_error())
        ppo_losses = self._ppo_update()
        auxiliary_losses = self._auxiliary_update()
        self.storage.clear()
        return {
            **ppo_losses,
            **auxiliary_losses,
            "rollout_fixed_log_prob_error": fixed_log_prob_error,
            "aux_grad_scaler_scale": float(self.auxiliary_grad_scaler.get_scale()),
        }

    def _ppo_update(self) -> dict[str, float]:
        totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "policy_kl": 0.0}
        updates = 0
        max_gradient_norm = 0.0
        nonfinite_gradient_steps = 0
        self.policy.action_mlp.train()
        self.critic.train()
        for batch in self.storage.ppo_batches(self.num_mini_batches, self.num_learning_epochs):
            advantages = batch["advantages"]
            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1.0e-8)
            action_mean = self.policy.action_mean(batch["actor_inputs"])
            action_std = self.policy.output_std.expand_as(action_mean)
            distribution = Normal(action_mean, action_std)
            log_prob = distribution.log_prob(batch["actions"]).sum(dim=-1, keepdim=True)
            entropy = distribution.entropy().sum(dim=-1).mean()
            value = self.critic(batch["critic_observations"])
            self._adapt_learning_rate(
                batch["old_action_mean"], batch["old_action_std"], action_mean, action_std
            )
            ratio = torch.exp(log_prob - batch["old_action_log_prob"])
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
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
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            self.ppo_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                chain(self.policy.action_mlp.parameters(), (self.policy.log_std,), self.critic.parameters()),
                self.max_grad_norm,
            )
            if torch.isfinite(gradient_norm):
                max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            else:
                nonfinite_gradient_steps += 1
                raise RuntimeError(f"Non-finite PPO gradient norm: {float(gradient_norm)}")
            self.ppo_optimizer.step()
            # The humanoid visual PPO keeps a nonzero exploration floor after
            # every policy update. With log-standard-deviation parameters this
            # is the equivalent lower bound.
            with torch.no_grad():
                self.policy.log_std.clamp_(min=math.log(0.2))
            with torch.no_grad():
                kl = self._normal_kl(
                    batch["old_action_mean"], batch["old_action_std"], action_mean, action_std
                ).mean()
            totals["value"] += float(value_loss.detach())
            totals["surrogate"] += float(surrogate_loss.detach())
            totals["entropy"] += float(entropy.detach())
            totals["policy_kl"] += float(kl.detach())
            updates += 1
        result = {name: value / updates for name, value in totals.items()}
        result["ppo_gradient_norm"] = max_gradient_norm
        result["ppo_nonfinite_gradient_steps"] = float(nonfinite_gradient_steps)
        return result

    def _auxiliary_update(self) -> dict[str, float]:
        totals: dict[str, torch.Tensor] | None = None
        total_env_exposures = 0
        optimizer_steps = 0
        max_gradient_norm = 0.0
        nonfinite_gradient_steps = 0
        reset_before = self.storage.reset_before_mask()
        self.policy.encoder.train()
        for logical_env_ids in self.storage.auxiliary_logical_env_batches(
            self.num_mini_batches, self.auxiliary_num_epochs
        ):
            self.auxiliary_optimizer.zero_grad(set_to_none=True)
            logical_batch_envs = int(logical_env_ids.numel())
            for start in range(0, logical_batch_envs, self.auxiliary_mini_batch_envs):
                env_ids = logical_env_ids[start : start + self.auxiliary_mini_batch_envs]
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self._auxiliary_amp_enabled,
                ):
                    output = self.policy.encoder(
                        self.storage.proprio_history[:, env_ids],
                        self.storage.depth_history[:, env_ids],
                        hidden_state=self.storage.initial_hidden_state[:, env_ids],
                        dones=reset_before[:, env_ids],
                        masks=torch.ones_like(self.storage.dones[:, env_ids]),
                        deterministic=False,
                    )
                    targets = TransformerVisionTargets(
                        velocity=self.storage.velocity_targets[:, env_ids],
                        limb_heightmaps=self.storage.limb_heightmaps[:, env_ids],
                        heightmap_valid=self.storage.heightmap_valid[:, env_ids],
                        heightmap_hit_class=self.storage.heightmap_hit_class[:, env_ids],
                        next_proprio=self.storage.next_proprio_targets[:, env_ids],
                        future_valid=self.storage.future_valid[:, env_ids],
                        limb_goal_b=self.storage.limb_goal_targets[:, env_ids],
                        limb_goal_valid=self.storage.limb_goal_valid[:, env_ids],
                        contact_indicator=self.storage.contact_indicator_targets[:, env_ids],
                        motion_joint_pos=self.storage.motion_joint_pos_targets[:, env_ids],
                        motion_root_orientation_6d=(
                            self.storage.motion_root_orientation_targets[:, env_ids]
                        ),
                        motion_valid=self.storage.motion_target_valid[:, env_ids],
                    )
                    loss_weights = (
                        TaskGoalTransformerVisionLossWeights()
                        if self.policy.encoder_variant.startswith("task_goal")
                        else TransformerVisionLossWeights()
                    )
                    losses = self.policy.encoder.losses(output, targets, loss_weights)
                    micro_batch_envs = int(env_ids.numel())
                    micro_batch_weight = micro_batch_envs / logical_batch_envs
                self.auxiliary_grad_scaler.scale(
                    losses["total"] * micro_batch_weight
                ).backward()
                if totals is None:
                    totals = {
                        name: value.detach() * micro_batch_envs
                        for name, value in losses.items()
                    }
                else:
                    for name, value in losses.items():
                        totals[name].add_(value.detach() * micro_batch_envs)
                total_env_exposures += micro_batch_envs
            self.auxiliary_grad_scaler.unscale_(self.auxiliary_optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(self.policy.encoder.parameters(), self.max_grad_norm)
            if torch.isfinite(gradient_norm):
                max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            else:
                nonfinite_gradient_steps += 1
                if not self._auxiliary_amp_enabled:
                    raise RuntimeError(f"Non-finite TransformerVision gradient norm: {float(gradient_norm)}")
            self.auxiliary_grad_scaler.step(self.auxiliary_optimizer)
            self.auxiliary_grad_scaler.update()
            optimizer_steps += 1
        if totals is None or total_env_exposures == 0:
            raise RuntimeError("TransformerVision auxiliary update received no batches")
        result = {f"aux_{name}": float(value / total_env_exposures) for name, value in totals.items()}
        result["aux_gradient_norm"] = max_gradient_norm
        result["aux_nonfinite_gradient_steps"] = float(nonfinite_gradient_steps)
        result["aux_optimizer_steps"] = float(optimizer_steps)
        return result

    def rollout_log_prob_max_error(self) -> torch.Tensor:
        if self.storage.step == 0:
            raise RuntimeError("No rollout is available")
        count = self.storage.step * self.storage.num_envs
        actor_inputs = self.storage.actor_inputs[: self.storage.step].reshape(
            count, self.storage.actor_input_dim
        )
        actions = self.storage.actions[: self.storage.step].reshape(count, self.storage.action_dim)
        old_log_prob = self.storage.old_action_log_prob[: self.storage.step].reshape(count, 1)
        with torch.no_grad():
            mean = self.policy.action_mean(actor_inputs)
            distribution = Normal(mean, self.policy.output_std.expand_as(mean))
            current_log_prob = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        return (current_log_prob - old_log_prob).abs().max()

    def act_inference(self, obs: TensorDict, dones: torch.Tensor | None = None) -> torch.Tensor:
        self._validate_observations(
            obs, require_task_goals=self.policy.encoder_variant == "task_goal"
        )
        if dones is not None:
            self.inference_hidden_state = self.policy.encoder.reset_hidden_state(
                self.inference_hidden_state, dones.to(device=self.device, dtype=torch.bool)
            )
        self.policy.encoder.eval()
        with torch.no_grad():
            output = self.policy.encoder(
                obs[PROPRIO_HISTORY_KEY],
                obs[DEPTH_HISTORY_KEY],
                hidden_state=self.inference_hidden_state,
                deterministic=True,
            )
            actor_input = torch.cat((obs[CURRENT_PROPRIO_KEY], output.latent), dim=-1)
            actions = self.policy.action_mean(actor_input)
        self.inference_hidden_state = output.next_hidden_state.detach()
        return actions

    def reset_inference(self) -> None:
        self.inference_hidden_state.zero_()
        self.policy.reset()

    def train_mode(self) -> None:
        self.policy.action_mlp.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.policy.eval()
        self.critic.eval()

    def get_policy(self) -> TransformerVisionActor:
        return self.policy

    def save(self) -> dict:
        return {
            "data_contract": self.data_contract,
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "ppo_optimizer_state_dict": self.ppo_optimizer.state_dict(),
            "auxiliary_optimizer_state_dict": self.auxiliary_optimizer.state_dict(),
            "grad_scaler_state_dict": self.auxiliary_grad_scaler.state_dict(),
            "visual_pretrain_checkpoint": self.visual_checkpoint,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if loaded_dict.get("data_contract") != self.data_contract:
            raise RuntimeError(
                "TransformerVision PPO checkpoint has data contract "
                f"{loaded_dict.get('data_contract')!r}, but the selected encoder requires "
                f"{self.data_contract!r}"
            )
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor", True):
            self.policy.load_state_dict(loaded_dict["policy_state_dict"], strict=strict)
        if load_cfg.get("critic", True):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer", True):
            self.ppo_optimizer.load_state_dict(loaded_dict["ppo_optimizer_state_dict"])
            self.auxiliary_optimizer.load_state_dict(loaded_dict["auxiliary_optimizer_state_dict"])
            if "grad_scaler_state_dict" in loaded_dict:
                self.auxiliary_grad_scaler.load_state_dict(
                    loaded_dict["grad_scaler_state_dict"]
                )
            restored_learning_rate = float(self.ppo_optimizer.param_groups[0]["lr"])
            self.learning_rate = min(
                self.max_learning_rate,
                max(self.min_learning_rate, restored_learning_rate),
            )
            for parameter_group in self.ppo_optimizer.param_groups:
                parameter_group["lr"] = self.learning_rate
        return bool(load_cfg.get("iteration", True))

    def _adapt_learning_rate(
        self,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        if self.schedule != "adaptive" or self.desired_kl is None:
            return
        with torch.no_grad():
            kl_mean = self._normal_kl(old_mean, old_std, mean, std).mean()
        if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
        elif 0.0 < kl_mean < self.desired_kl / 2.0:
            self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
        for parameter_group in self.ppo_optimizer.param_groups:
            parameter_group["lr"] = self.learning_rate

    @staticmethod
    def _normal_kl(
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        return torch.log(std / old_std) + (
            old_std.square() + (old_mean - mean).square()
        ) / (2.0 * std.square()) - 0.5

    @staticmethod
    def _load_visual_pretrain(policy: TransformerVisionActor, checkpoint: str) -> None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TransformerVision checkpoint was not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"{path} is not a TransformerVision pretraining checkpoint")
        if payload.get("data_contract") != policy.encoder.data_contract:
            raise RuntimeError(f"TransformerVision checkpoint has an incompatible data contract: {path}")
        policy.encoder.load_state_dict(payload["model_state_dict"], strict=True)
        print(f"[TRANSFORMER VISION PPO] initialized visual encoder from {path}", flush=True)

    @staticmethod
    def _validate_observations(
        obs: TensorDict, *, require_task_goals: bool = False
    ) -> None:
        required = (
            CURRENT_PROPRIO_KEY,
            PROPRIO_HISTORY_KEY,
            DEPTH_HISTORY_KEY,
            HEIGHTMAP_KEY,
            HEIGHTMAP_VALID_KEY,
            HEIGHTMAP_HIT_CLASS_KEY,
            VELOCITY_KEY,
            NEXT_PROPRIO_KEY,
            CRITIC_KEY,
        )
        missing = [name for name in required if name not in obs]
        if require_task_goals:
            missing.extend(
                name
                for name in (
                    LIMB_GOAL_KEY,
                    LIMB_GOAL_VALID_KEY,
                    CONTACT_INDICATOR_KEY,
                    MOTION_JOINT_POS_KEY,
                    MOTION_ROOT_ORIENTATION_KEY,
                    MOTION_TARGET_VALID_KEY,
                )
                if name not in obs
            )
        if missing:
            raise KeyError(f"Missing TransformerVision PPO observations: {missing}")
        batch_size = obs[CRITIC_KEY].shape[0]
        expected = {
            CURRENT_PROPRIO_KEY: (batch_size, 85),
            PROPRIO_HISTORY_KEY: (batch_size, 10, 85),
            DEPTH_HISTORY_KEY: (batch_size, 2, 60, 80),
            HEIGHTMAP_KEY: (batch_size, 4, 10, 3),
            HEIGHTMAP_VALID_KEY: (batch_size, 4, 10, 3),
            HEIGHTMAP_HIT_CLASS_KEY: (batch_size, 4, 10, 3),
            VELOCITY_KEY: (batch_size, 3),
            NEXT_PROPRIO_KEY: (batch_size, 85),
        }
        if require_task_goals:
            expected.update(
                {
                    LIMB_GOAL_KEY: (batch_size, 4, 3),
                    LIMB_GOAL_VALID_KEY: (batch_size, 4),
                    CONTACT_INDICATOR_KEY: (batch_size, 4),
                    MOTION_JOINT_POS_KEY: (batch_size, 25),
                    MOTION_ROOT_ORIENTATION_KEY: (batch_size, 6),
                    MOTION_TARGET_VALID_KEY: (batch_size,),
                }
            )
        for name, shape in expected.items():
            if tuple(obs[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(obs[name].shape)}")
        if obs[CRITIC_KEY].ndim != 2:
            raise ValueError(f"critic must have shape [B,C], got {tuple(obs[CRITIC_KEY].shape)}")


__all__ = ["TransformerVisionPPO"]
