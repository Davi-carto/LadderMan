"""Recurrent asymmetric PPO mechanics for the CR1 visual actor."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Iterator

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from ladder.models import LadderSSRVisualActor, LadderVisualLossWeights, LadderVisualTargets


_PROPRIO_DIM = 85
_HISTORY_FRAMES = 10
_DEPTH_FRAMES = 2
_DEPTH_HEIGHT = 60
_DEPTH_WIDTH = 80
_MAP_SHAPE = (4, 10, 3)
_PROPRIO_HISTORY_KEY = "ssr_proprio_history"
_DEPTH_HISTORY_KEY = "ssr_depth_history"
_HEIGHTMAP_KEY = "ssr_limb_heightmaps"
_HEIGHTMAP_VALID_KEY = "ssr_heightmap_valid"
_VELOCITY_KEY = "ssr_base_lin_vel"
_NEXT_PROPRIO_KEY = "ssr_next_proprio"


def _elu_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class _PendingTransition:
    proprio_history: torch.Tensor
    depth_history: torch.Tensor
    critic_observation: torch.Tensor
    actor_input: torch.Tensor
    actions: torch.Tensor
    old_action_mean: torch.Tensor
    old_action_std: torch.Tensor
    old_action_log_prob: torch.Tensor
    value: torch.Tensor
    velocity_target: torch.Tensor
    limb_heightmaps: torch.Tensor
    heightmap_valid: torch.Tensor


class LadderSSRRolloutStorage:
    """Fixed-shape rollout data required by PPO and visual auxiliary updates."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        action_dim: int,
        critic_dim: int,
        device: str | torch.device,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step = 0
        self.proprio_history = torch.zeros(
            num_steps, num_envs, _HISTORY_FRAMES, _PROPRIO_DIM, device=self.device
        )
        self.depth_history = torch.zeros(
            num_steps, num_envs, _DEPTH_FRAMES, _DEPTH_HEIGHT, _DEPTH_WIDTH, device=self.device
        )
        self.critic_observations = torch.zeros(num_steps, num_envs, critic_dim, device=self.device)
        self.actor_inputs = torch.zeros(num_steps, num_envs, _PROPRIO_DIM + 51, device=self.device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=self.device)
        self.old_action_mean = torch.zeros(num_steps, num_envs, action_dim, device=self.device)
        self.old_action_std = torch.zeros(num_steps, num_envs, action_dim, device=self.device)
        self.old_action_log_prob = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)
        self.returns = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.velocity_targets = torch.zeros(num_steps, num_envs, 3, device=self.device)
        self.limb_heightmaps = torch.zeros(num_steps, num_envs, *_MAP_SHAPE, device=self.device)
        self.heightmap_valid = torch.zeros(num_steps, num_envs, *_MAP_SHAPE, device=self.device)
        self.next_proprio_targets = torch.zeros(num_steps, num_envs, _PROPRIO_DIM, device=self.device)
        self.future_valid = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)
        self.initial_hidden_state = torch.zeros(1, num_envs, 256, device=self.device)

    def begin_rollout(self, hidden_state: torch.Tensor) -> None:
        if self.step != 0:
            raise RuntimeError("A rollout can only begin with an empty storage")
        expected_shape = (1, self.num_envs, 256)
        if tuple(hidden_state.shape) != expected_shape:
            raise ValueError(f"initial hidden state must have shape {expected_shape}, got {tuple(hidden_state.shape)}")
        self.initial_hidden_state.copy_(hidden_state)

    def add(
        self,
        transition: _PendingTransition,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_proprio: torch.Tensor,
    ) -> None:
        if self.step >= self.num_steps:
            raise OverflowError("Rollout buffer overflow")
        index = self.step
        self.proprio_history[index].copy_(transition.proprio_history)
        self.depth_history[index].copy_(transition.depth_history)
        self.critic_observations[index].copy_(transition.critic_observation)
        self.actor_inputs[index].copy_(transition.actor_input)
        self.actions[index].copy_(transition.actions)
        self.old_action_mean[index].copy_(transition.old_action_mean)
        self.old_action_std[index].copy_(transition.old_action_std)
        self.old_action_log_prob[index].copy_(transition.old_action_log_prob)
        self.values[index].copy_(transition.value)
        self.rewards[index].copy_(reward.reshape(self.num_envs, 1))
        self.dones[index].copy_(done.to(dtype=torch.bool).reshape(self.num_envs))
        self.velocity_targets[index].copy_(transition.velocity_target)
        self.limb_heightmaps[index].copy_(transition.limb_heightmaps)
        self.heightmap_valid[index].copy_(transition.heightmap_valid)
        self.next_proprio_targets[index].copy_(next_proprio)
        self.future_valid[index].copy_(~self.dones[index])
        self.step += 1

    def compute_returns(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        if self.step != self.num_steps:
            raise RuntimeError(f"Expected {self.num_steps} transitions, stored {self.step}")
        advantage = torch.zeros_like(last_value)
        for index in reversed(range(self.num_steps)):
            next_value = last_value if index == self.num_steps - 1 else self.values[index + 1]
            not_done = (~self.dones[index]).to(dtype=self.values.dtype).unsqueeze(-1)
            delta = self.rewards[index] + gamma * not_done * next_value - self.values[index]
            advantage = delta + gamma * lam * not_done * advantage
            self.advantages[index] = advantage
            self.returns[index] = advantage + self.values[index]
        self.advantages.sub_(self.advantages.mean()).div_(self.advantages.std().clamp_min(1.0e-8))

    def ppo_batches(self, num_mini_batches: int, num_epochs: int) -> Iterator[dict[str, torch.Tensor]]:
        total_samples = self.num_steps * self.num_envs
        mini_batch_size = max(1, total_samples // num_mini_batches)
        flat = {
            "actor_inputs": self.actor_inputs.reshape(total_samples, -1),
            "actions": self.actions.reshape(total_samples, -1),
            "old_action_mean": self.old_action_mean.reshape(total_samples, -1),
            "old_action_std": self.old_action_std.reshape(total_samples, -1),
            "old_action_log_prob": self.old_action_log_prob.reshape(total_samples, 1),
            "critic_observations": self.critic_observations.reshape(total_samples, -1),
            "values": self.values.reshape(total_samples, 1),
            "advantages": self.advantages.reshape(total_samples, 1),
            "returns": self.returns.reshape(total_samples, 1),
        }
        for _ in range(num_epochs):
            indices = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, mini_batch_size):
                selected = indices[start : start + mini_batch_size]
                yield {name: value[selected] for name, value in flat.items()}

    def reset_before_mask(self) -> torch.Tensor:
        """Return done flags aligned with the input at each recurrent step."""

        reset_before = torch.zeros_like(self.dones)
        reset_before[1:] = self.dones[:-1]
        return reset_before

    def clear(self) -> None:
        self.step = 0


class LadderSSRPPO:
    """PPO with a fixed rollout latent and a separate visual auxiliary phase."""

    def __init__(
        self,
        actor: LadderSSRVisualActor,
        critic: nn.Module,
        storage: LadderSSRRolloutStorage,
        *,
        num_learning_epochs: int,
        num_mini_batches: int,
        clip_param: float,
        gamma: float,
        lam: float,
        value_loss_coef: float,
        entropy_coef: float,
        learning_rate: float,
        auxiliary_learning_rate: float,
        max_grad_norm: float,
        use_clipped_value_loss: bool,
        schedule: str,
        desired_kl: float | None,
        enable_auxiliary_update: bool,
        device: str | torch.device,
    ) -> None:
        if schedule not in ("adaptive", "fixed"):
            raise ValueError(f"Unsupported PPO learning-rate schedule {schedule!r}")
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.storage = storage
        self.device = torch.device(device)
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.auxiliary_learning_rate = auxiliary_learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.enable_auxiliary_update = enable_auxiliary_update
        self.rnd = None
        self.intrinsic_rewards = None
        self.hidden_state = torch.zeros(1, storage.num_envs, 256, device=self.device)
        self.inference_hidden_state = torch.zeros_like(self.hidden_state)
        self._pending: _PendingTransition | None = None

        ppo_params = [*self.actor.actor.parameters(), self.actor.action_std, *self.critic.parameters()]
        ppo_ids = {id(parameter) for parameter in ppo_params}
        auxiliary_params = [
            parameter
            for parameter in self.actor.parameters()
            if id(parameter) not in ppo_ids
        ]
        if ppo_ids.intersection(id(parameter) for parameter in auxiliary_params):
            raise RuntimeError("PPO and auxiliary parameter sets must not overlap")
        self.ppo_optimizer = torch.optim.Adam(ppo_params, lr=learning_rate)
        self.auxiliary_optimizer = torch.optim.Adam(auxiliary_params, lr=auxiliary_learning_rate)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> "LadderSSRPPO":
        """Build this local algorithm through RSL-RL's extension hook."""

        if not isinstance(obs, TensorDict):
            raise TypeError("LadderSSRPPO requires the Ladder TensorDict vector wrapper")
        LadderSSRPPO._validate_observations(obs)
        if cfg.get("multi_gpu") is not None:
            raise NotImplementedError("LadderSSRPPO's first implementation supports one GPU only")
        policy_cfg = cfg["policy"]
        algorithm_cfg = dict(cfg["algorithm"])
        algorithm_cfg.pop("class_name", None)
        algorithm_cfg.pop("rnd_cfg", None)
        algorithm_cfg.pop("symmetry_cfg", None)
        algorithm_cfg.pop("normalize_advantage_per_mini_batch", None)
        critic_dim = int(obs["critic"].shape[-1])
        actor = LadderSSRVisualActor(proprio_dim=_PROPRIO_DIM, action_dim=env.num_actions)
        with torch.no_grad():
            actor.action_std.fill_(float(policy_cfg["init_noise_std"]))
        critic = _elu_mlp(critic_dim, list(policy_cfg["critic_hidden_dims"]), 1)
        storage = LadderSSRRolloutStorage(
            num_steps=int(cfg["num_steps_per_env"]),
            num_envs=env.num_envs,
            action_dim=env.num_actions,
            critic_dim=critic_dim,
            device=device,
        )
        return LadderSSRPPO(
            actor,
            critic,
            storage,
            device=device,
            auxiliary_learning_rate=float(algorithm_cfg.pop("auxiliary_learning_rate")),
            enable_auxiliary_update=bool(algorithm_cfg.pop("enable_auxiliary_update")),
            num_learning_epochs=int(algorithm_cfg.pop("num_learning_epochs")),
            num_mini_batches=int(algorithm_cfg.pop("num_mini_batches")),
            clip_param=float(algorithm_cfg.pop("clip_param")),
            gamma=float(algorithm_cfg.pop("gamma")),
            lam=float(algorithm_cfg.pop("lam")),
            value_loss_coef=float(algorithm_cfg.pop("value_loss_coef")),
            entropy_coef=float(algorithm_cfg.pop("entropy_coef")),
            learning_rate=float(algorithm_cfg.pop("learning_rate")),
            max_grad_norm=float(algorithm_cfg.pop("max_grad_norm")),
            use_clipped_value_loss=bool(algorithm_cfg.pop("use_clipped_value_loss")),
            schedule=str(algorithm_cfg.pop("schedule")),
            desired_kl=algorithm_cfg.pop("desired_kl"),
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample an action and retain only the rollout values needed later."""

        if self._pending is not None:
            raise RuntimeError("process_env_step must follow every act call")
        self._validate_observations(obs)
        if self.storage.step == 0:
            self.storage.begin_rollout(self.hidden_state)
        proprio_history, depth_history = self._visual_inputs(obs)
        with torch.no_grad():
            output = self.actor(proprio_history, depth_history, hidden_state=self.hidden_state, deterministic=False)
            distribution = self.actor.action_distribution(output)
            actions = distribution.sample()
            action_log_prob = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
            value = self.critic(obs["critic"])
        self.hidden_state = output.next_hidden_state.detach()
        self._pending = _PendingTransition(
            proprio_history=proprio_history.detach(),
            depth_history=depth_history.detach(),
            critic_observation=obs["critic"].detach(),
            actor_input=output.actor_input.detach(),
            actions=actions.detach(),
            old_action_mean=output.actions.detach(),
            old_action_std=self.actor.output_std.detach().expand_as(actions),
            old_action_log_prob=action_log_prob.detach(),
            value=value.detach(),
            velocity_target=obs[_VELOCITY_KEY].detach(),
            limb_heightmaps=obs[_HEIGHTMAP_KEY].detach(),
            heightmap_valid=obs[_HEIGHTMAP_VALID_KEY].detach(),
        )
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Attach post-action ``o_(t+1)`` and commit one complete transition."""

        if self._pending is None:
            raise RuntimeError("act must be called before process_env_step")
        reward = rewards.detach().clone().to(self.device)
        if "time_outs" in extras:
            timeout = extras["time_outs"].to(device=self.device, dtype=reward.dtype)
            reward += self.gamma * self._pending.value.squeeze(-1) * timeout
        next_proprio = obs[_NEXT_PROPRIO_KEY].detach()
        self.storage.add(self._pending, reward, dones, next_proprio)
        self.hidden_state = self.actor.reset_hidden_state(self.hidden_state, dones.to(device=self.device, dtype=torch.bool))
        self._pending = None

    def compute_returns(self, obs: TensorDict) -> None:
        self._validate_observations(obs)
        with torch.no_grad():
            last_value = self.critic(obs["critic"])
        self.storage.compute_returns(last_value, self.gamma, self.lam)

    def update(self) -> dict[str, float]:
        """Run PPO on frozen actor inputs, then optionally run the auxiliary phase."""

        if self.storage.step != self.storage.num_steps:
            raise RuntimeError("compute an entire rollout before update")
        value_loss_sum = 0.0
        surrogate_loss_sum = 0.0
        entropy_sum = 0.0
        kl_sum = 0.0
        updates = 0
        for batch in self.storage.ppo_batches(self.num_mini_batches, self.num_learning_epochs):
            mean = self.actor.actor(batch["actor_inputs"])
            std = self.actor.output_std.expand_as(mean)
            distribution = Normal(mean, std)
            log_prob = distribution.log_prob(batch["actions"]).sum(dim=-1, keepdim=True)
            entropy = distribution.entropy().sum(dim=-1).mean()
            values = self.critic(batch["critic_observations"])
            self._adapt_learning_rate(batch["old_action_mean"], batch["old_action_std"], mean, std)
            ratio = torch.exp(log_prob - batch["old_action_log_prob"])
            surrogate = -batch["advantages"] * ratio
            surrogate_clipped = -batch["advantages"] * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch["values"] + (values - batch["values"]).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum(
                    (values - batch["returns"]).square(), (value_clipped - batch["returns"]).square()
                ).mean()
            else:
                value_loss = (values - batch["returns"]).square().mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            self.ppo_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                chain(self.actor.actor.parameters(), (self.actor.action_std,), self.critic.parameters()),
                self.max_grad_norm,
            )
            self.ppo_optimizer.step()
            with torch.no_grad():
                kl = self._normal_kl(batch["old_action_mean"], batch["old_action_std"], mean, std).mean()
            value_loss_sum += value_loss.item()
            surrogate_loss_sum += surrogate_loss.item()
            entropy_sum += entropy.item()
            kl_sum += kl.item()
            updates += 1

        losses = {
            "value": value_loss_sum / updates,
            "surrogate": surrogate_loss_sum / updates,
            "entropy": entropy_sum / updates,
            "policy_kl": kl_sum / updates,
        }
        if self.enable_auxiliary_update:
            losses.update(self._auxiliary_update())
        else:
            losses.update({"aux_total": 0.0, "aux_gradient_norm": 0.0})
        self.storage.clear()
        return losses

    def rollout_log_prob_max_error(self) -> torch.Tensor:
        """Measure the fixed-latent PPO invariant before a PPO parameter update."""

        if self.storage.step == 0:
            raise RuntimeError("No rollout is available")
        actor_inputs = self.storage.actor_inputs[: self.storage.step].reshape(-1, _PROPRIO_DIM + 51)
        actions = self.storage.actions[: self.storage.step].reshape(-1, self.actor.action_dim)
        old_log_prob = self.storage.old_action_log_prob[: self.storage.step].reshape(-1, 1)
        with torch.no_grad():
            distribution = Normal(self.actor.actor(actor_inputs), self.actor.output_std.expand_as(actions))
            current_log_prob = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        return (current_log_prob - old_log_prob).abs().max()

    def act_inference(self, obs: TensorDict, dones: torch.Tensor | None = None) -> torch.Tensor:
        """Return deterministic actions while retaining the deployment GRU state."""

        self._validate_observations(obs)
        proprio_history, depth_history = self._visual_inputs(obs)
        if dones is not None:
            self.inference_hidden_state = self.actor.reset_hidden_state(
                self.inference_hidden_state, dones.to(device=self.device, dtype=torch.bool)
            )
        with torch.no_grad():
            output = self.actor(
                proprio_history, depth_history, hidden_state=self.inference_hidden_state, deterministic=True
            )
        self.inference_hidden_state = output.next_hidden_state.detach()
        return output.actions

    def reset_inference(self) -> None:
        self.inference_hidden_state.zero_()

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def get_policy(self) -> LadderSSRVisualActor:
        return self.actor

    def save(self) -> dict:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "ppo_optimizer_state_dict": self.ppo_optimizer.state_dict(),
            "auxiliary_optimizer_state_dict": self.auxiliary_optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor", True):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic", True):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer", True):
            self.ppo_optimizer.load_state_dict(loaded_dict["ppo_optimizer_state_dict"])
            self.auxiliary_optimizer.load_state_dict(loaded_dict["auxiliary_optimizer_state_dict"])
        return bool(load_cfg.get("iteration", True))

    def _auxiliary_update(self) -> dict[str, float]:
        masks = torch.ones_like(self.storage.dones)
        output = self.actor(
            self.storage.proprio_history,
            self.storage.depth_history,
            hidden_state=self.storage.initial_hidden_state,
            dones=self.storage.reset_before_mask(),
            masks=masks,
            deterministic=False,
        )
        targets = LadderVisualTargets(
            velocity=self.storage.velocity_targets,
            limb_heightmaps=self.storage.limb_heightmaps,
            heightmap_valid=self.storage.heightmap_valid,
            next_proprio=self.storage.next_proprio_targets,
            future_valid=self.storage.future_valid,
        )
        losses = self.actor.auxiliary_losses(output, targets, LadderVisualLossWeights())
        self.auxiliary_optimizer.zero_grad()
        losses["total"].backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            (parameter for group in self.auxiliary_optimizer.param_groups for parameter in group["params"]),
            self.max_grad_norm,
        )
        self.auxiliary_optimizer.step()
        return {
            "aux_velocity": losses["velocity"].item(),
            "aux_hand": losses["hand"].item(),
            "aux_foot": losses["foot"].item(),
            "aux_future": losses["future"].item(),
            "aux_kl": losses["kl"].item(),
            "aux_total": losses["total"].item(),
            "aux_gradient_norm": float(gradient_norm),
        }

    def _adapt_learning_rate(
        self, old_mean: torch.Tensor, old_std: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        if self.schedule != "adaptive" or self.desired_kl is None:
            return
        with torch.no_grad():
            kl_mean = self._normal_kl(old_mean, old_std, mean, std).mean()
        if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
        elif 0.0 < kl_mean < self.desired_kl / 2.0:
            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
        for parameter_group in self.ppo_optimizer.param_groups:
            parameter_group["lr"] = self.learning_rate

    @staticmethod
    def _normal_kl(
        old_mean: torch.Tensor, old_std: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return torch.log(std / old_std) + (old_std.square() + (old_mean - mean).square()) / (2.0 * std.square()) - 0.5

    @staticmethod
    def _visual_inputs(obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        return obs[_PROPRIO_HISTORY_KEY], obs[_DEPTH_HISTORY_KEY]

    @staticmethod
    def _validate_observations(obs: TensorDict) -> None:
        required = (
            _PROPRIO_HISTORY_KEY,
            _DEPTH_HISTORY_KEY,
            _HEIGHTMAP_KEY,
            _HEIGHTMAP_VALID_KEY,
            _VELOCITY_KEY,
            _NEXT_PROPRIO_KEY,
            "critic",
        )
        missing = [name for name in required if name not in obs]
        if missing:
            raise KeyError(f"Missing SSR observation terms: {missing}")
        batch_size = obs["critic"].shape[0]
        expected_shapes = {
            _PROPRIO_HISTORY_KEY: (batch_size, _HISTORY_FRAMES, _PROPRIO_DIM),
            _DEPTH_HISTORY_KEY: (batch_size, _DEPTH_FRAMES, _DEPTH_HEIGHT, _DEPTH_WIDTH),
            _HEIGHTMAP_KEY: (batch_size, *_MAP_SHAPE),
            _HEIGHTMAP_VALID_KEY: (batch_size, *_MAP_SHAPE),
            _VELOCITY_KEY: (batch_size, 3),
            _NEXT_PROPRIO_KEY: (batch_size, _PROPRIO_DIM),
        }
        for name, expected_shape in expected_shapes.items():
            if tuple(obs[name].shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(obs[name].shape)}")


__all__ = ["LadderSSRPPO", "LadderSSRRolloutStorage"]
