"""Rollout storage for TransformerVision asymmetric PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


HEIGHTMAP_KEY = "transformer_vision_limb_heightmaps"
HEIGHTMAP_VALID_KEY = "transformer_vision_heightmap_valid"
HEIGHTMAP_HIT_CLASS_KEY = "transformer_vision_heightmap_hit_class"
VELOCITY_KEY = "transformer_vision_velocity_target"
NEXT_PROPRIO_KEY = "transformer_vision_next_proprio_target"
LIMB_GOAL_KEY = "transformer_vision_limb_goal_target"
LIMB_GOAL_VALID_KEY = "transformer_vision_limb_goal_valid"
CONTACT_INDICATOR_KEY = "transformer_vision_contact_indicator_target"
MOTION_JOINT_POS_KEY = "transformer_vision_motion_joint_pos_target"
MOTION_ROOT_ORIENTATION_KEY = "transformer_vision_motion_root_orientation_6d_target"
MOTION_TARGET_VALID_KEY = "transformer_vision_motion_target_valid"
CRITIC_KEY = "critic"


@dataclass
class TransformerVisionPPOTransition:
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
    heightmap_hit_class: torch.Tensor
    limb_goal_target: torch.Tensor | None = None
    limb_goal_valid: torch.Tensor | None = None
    contact_indicator_target: torch.Tensor | None = None
    motion_joint_pos_target: torch.Tensor | None = None
    motion_root_orientation_target: torch.Tensor | None = None
    motion_target_valid: torch.Tensor | None = None
    teacher_action_mean: torch.Tensor | None = None


class TransformerVisionPPORolloutStorage:
    """Store fixed PPO inputs and complete recurrent auxiliary sequences."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        action_dim: int,
        critic_dim: int,
        device: str | torch.device,
        task_goal_supervision: bool = False,
        actor_input_dim: int = 136,
        proprio_history_frames: int = 10,
        teacher_supervision: bool = False,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.critic_dim = critic_dim
        self.device = torch.device(device)
        self.task_goal_supervision = task_goal_supervision
        self.actor_input_dim = actor_input_dim
        self.proprio_history_frames = proprio_history_frames
        self.teacher_supervision = teacher_supervision
        self.step = 0
        self.proprio_history = torch.zeros(
            num_steps, num_envs, proprio_history_frames, 85, device=self.device
        )
        self.depth_history = torch.zeros(num_steps, num_envs, 2, 60, 80, device=self.device)
        self.critic_observations = torch.zeros(num_steps, num_envs, critic_dim, device=self.device)
        self.actor_inputs = torch.zeros(
            num_steps, num_envs, actor_input_dim, device=self.device
        )
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=self.device)
        self.old_action_mean = torch.zeros_like(self.actions)
        self.old_action_std = torch.zeros_like(self.actions)
        self.old_action_log_prob = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)
        self.returns = torch.zeros_like(self.values)
        self.advantages = torch.zeros_like(self.values)
        self.velocity_targets = torch.zeros(num_steps, num_envs, 3, device=self.device)
        self.limb_heightmaps = torch.zeros(num_steps, num_envs, 4, 10, 3, device=self.device)
        self.heightmap_valid = torch.zeros_like(self.limb_heightmaps)
        self.heightmap_hit_class = torch.zeros(
            num_steps, num_envs, 4, 10, 3, dtype=torch.long, device=self.device
        )
        self.next_proprio_targets = torch.zeros(num_steps, num_envs, 85, device=self.device)
        self.future_valid = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)
        self.limb_goal_targets = torch.zeros(num_steps, num_envs, 4, 3, device=self.device)
        self.limb_goal_valid = torch.zeros(num_steps, num_envs, 4, dtype=torch.bool, device=self.device)
        self.contact_indicator_targets = torch.zeros(num_steps, num_envs, 4, device=self.device)
        self.motion_joint_pos_targets = torch.zeros(
            num_steps, num_envs, 25, device=self.device
        )
        self.motion_root_orientation_targets = torch.zeros(
            num_steps, num_envs, 6, device=self.device
        )
        self.motion_target_valid = torch.zeros(
            num_steps, num_envs, dtype=torch.bool, device=self.device
        )
        self.teacher_action_mean = (
            torch.zeros_like(self.actions) if teacher_supervision else None
        )
        self.initial_hidden_state = torch.zeros(1, num_envs, 256, device=self.device)

    def begin_rollout(self, hidden_state: torch.Tensor) -> None:
        if self.step != 0:
            raise RuntimeError("A rollout can only begin with empty storage")
        expected = (1, self.num_envs, 256)
        if tuple(hidden_state.shape) != expected:
            raise ValueError(f"hidden_state must have shape {expected}, got {tuple(hidden_state.shape)}")
        self.initial_hidden_state.copy_(hidden_state)

    def add(
        self,
        transition: TransformerVisionPPOTransition,
        *,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_proprio: torch.Tensor | None,
    ) -> None:
        if self.step >= self.num_steps:
            raise OverflowError("TransformerVision PPO rollout storage is full")
        index = self.step
        expected = {
            "proprio_history": (self.num_envs, self.proprio_history_frames, 85),
            "depth_history": (self.num_envs, 2, 60, 80),
            "critic_observation": (self.num_envs, self.critic_dim),
            "actor_input": (self.num_envs, self.actor_input_dim),
            "actions": (self.num_envs, self.action_dim),
            "old_action_mean": (self.num_envs, self.action_dim),
            "old_action_std": (self.num_envs, self.action_dim),
            "old_action_log_prob": (self.num_envs, 1),
            "value": (self.num_envs, 1),
            "velocity_target": (self.num_envs, 3),
            "limb_heightmaps": (self.num_envs, 4, 10, 3),
            "heightmap_valid": (self.num_envs, 4, 10, 3),
            "heightmap_hit_class": (self.num_envs, 4, 10, 3),
        }
        for name in expected:
            value = getattr(transition, name)
            if tuple(value.shape) != expected[name]:
                raise ValueError(f"{name} must have shape {expected[name]}, got {tuple(value.shape)}")
        task_values = {
            "limb_goal_target": transition.limb_goal_target,
            "limb_goal_valid": transition.limb_goal_valid,
            "contact_indicator_target": transition.contact_indicator_target,
            "motion_joint_pos_target": transition.motion_joint_pos_target,
            "motion_root_orientation_target": transition.motion_root_orientation_target,
            "motion_target_valid": transition.motion_target_valid,
        }
        if self.task_goal_supervision:
            task_expected = {
                "limb_goal_target": (self.num_envs, 4, 3),
                "limb_goal_valid": (self.num_envs, 4),
                "contact_indicator_target": (self.num_envs, 4),
                "motion_joint_pos_target": (self.num_envs, 25),
                "motion_root_orientation_target": (self.num_envs, 6),
                "motion_target_valid": (self.num_envs,),
            }
            for name, shape in task_expected.items():
                value = task_values[name]
                if value is None or tuple(value.shape) != shape:
                    actual = None if value is None else tuple(value.shape)
                    raise ValueError(f"{name} must have shape {shape}, got {actual}")
        if self.teacher_supervision:
            if transition.teacher_action_mean is None or tuple(
                transition.teacher_action_mean.shape
            ) != (self.num_envs, self.action_dim):
                actual = (
                    None
                    if transition.teacher_action_mean is None
                    else tuple(transition.teacher_action_mean.shape)
                )
                raise ValueError(
                    "teacher_action_mean must have shape "
                    f"{(self.num_envs, self.action_dim)}, got {actual}"
                )
        if tuple(reward.shape) != (self.num_envs,):
            raise ValueError(f"reward must have shape {(self.num_envs,)}, got {tuple(reward.shape)}")
        if tuple(done.shape) != (self.num_envs,):
            raise ValueError(f"done must have shape {(self.num_envs,)}, got {tuple(done.shape)}")
        if next_proprio is not None and tuple(next_proprio.shape) != (self.num_envs, 85):
            raise ValueError(
                f"next_proprio must have shape {(self.num_envs, 85)}, got {tuple(next_proprio.shape)}"
            )

        self.proprio_history[index].copy_(transition.proprio_history)
        self.depth_history[index].copy_(transition.depth_history)
        self.critic_observations[index].copy_(transition.critic_observation)
        self.actor_inputs[index].copy_(transition.actor_input)
        self.actions[index].copy_(transition.actions)
        self.old_action_mean[index].copy_(transition.old_action_mean)
        self.old_action_std[index].copy_(transition.old_action_std)
        self.old_action_log_prob[index].copy_(transition.old_action_log_prob)
        self.values[index].copy_(transition.value)
        self.rewards[index].copy_(reward.unsqueeze(-1))
        self.dones[index].copy_(done.to(dtype=torch.bool))
        self.velocity_targets[index].copy_(transition.velocity_target)
        self.limb_heightmaps[index].copy_(transition.limb_heightmaps)
        self.heightmap_valid[index].copy_(transition.heightmap_valid)
        self.heightmap_hit_class[index].copy_(transition.heightmap_hit_class.to(dtype=torch.long))
        if next_proprio is not None:
            self.next_proprio_targets[index].copy_(next_proprio)
            self.future_valid[index].copy_(~self.dones[index])
        else:
            self.next_proprio_targets[index].zero_()
            self.future_valid[index].zero_()
        if self.task_goal_supervision:
            self.limb_goal_targets[index].copy_(transition.limb_goal_target)
            self.limb_goal_valid[index].copy_(transition.limb_goal_valid.to(dtype=torch.bool))
            self.contact_indicator_targets[index].copy_(transition.contact_indicator_target)
            self.motion_joint_pos_targets[index].copy_(
                transition.motion_joint_pos_target
            )
            self.motion_root_orientation_targets[index].copy_(
                transition.motion_root_orientation_target
            )
            self.motion_target_valid[index].copy_(
                transition.motion_target_valid.to(dtype=torch.bool)
            )
        if self.teacher_supervision:
            self.teacher_action_mean[index].copy_(transition.teacher_action_mean)
        self.step += 1

    def compute_returns(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        if self.step != self.num_steps:
            raise RuntimeError(f"Expected {self.num_steps} transitions, stored {self.step}")
        if tuple(last_value.shape) != (self.num_envs, 1):
            raise ValueError(f"last_value must have shape {(self.num_envs, 1)}")
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
            "actor_inputs": self.actor_inputs.reshape(
                total_samples, self.actor_input_dim
            ),
            "actions": self.actions.reshape(total_samples, self.action_dim),
            "old_action_mean": self.old_action_mean.reshape(total_samples, self.action_dim),
            "old_action_std": self.old_action_std.reshape(total_samples, self.action_dim),
            "old_action_log_prob": self.old_action_log_prob.reshape(total_samples, 1),
            "critic_observations": self.critic_observations.reshape(total_samples, self.critic_dim),
            "values": self.values.reshape(total_samples, 1),
            "advantages": self.advantages.reshape(total_samples, 1),
            "returns": self.returns.reshape(total_samples, 1),
        }
        if self.teacher_supervision:
            flat["teacher_action_mean"] = self.teacher_action_mean.reshape(
                total_samples, self.action_dim
            )
        for _ in range(num_epochs):
            indices = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, mini_batch_size):
                selected = indices[start : start + mini_batch_size]
                yield {name: value[selected] for name, value in flat.items()}

    def auxiliary_env_batches(self, mini_batch_envs: int, num_epochs: int) -> Iterator[torch.Tensor]:
        """Yield environment indexes while preserving every environment's full time sequence."""

        if mini_batch_envs <= 0 or num_epochs <= 0:
            raise ValueError("Auxiliary batch size and epoch count must be positive")
        for _ in range(num_epochs):
            permutation = torch.randperm(self.num_envs, device=self.device)
            for start in range(0, self.num_envs, mini_batch_envs):
                yield permutation[start : start + mini_batch_envs]

    def auxiliary_logical_env_batches(
        self, num_mini_batches: int, num_epochs: int
    ) -> Iterator[torch.Tensor]:
        """Yield exactly ``num_mini_batches`` recurrent environment groups per epoch."""

        if num_mini_batches <= 0 or num_epochs <= 0:
            raise ValueError("Auxiliary mini-batch and epoch counts must be positive")
        if num_mini_batches > self.num_envs:
            raise ValueError(
                f"Auxiliary num_mini_batches {num_mini_batches} exceeds num_envs {self.num_envs}"
            )
        for _ in range(num_epochs):
            permutation = torch.randperm(self.num_envs, device=self.device)
            yield from torch.tensor_split(permutation, num_mini_batches)

    def reset_before_mask(self) -> torch.Tensor:
        reset_before = torch.zeros_like(self.dones)
        reset_before[1:] = self.dones[:-1]
        return reset_before

    def clear(self) -> None:
        self.step = 0


__all__ = [
    "CRITIC_KEY",
    "HEIGHTMAP_HIT_CLASS_KEY",
    "HEIGHTMAP_KEY",
    "HEIGHTMAP_VALID_KEY",
    "LIMB_GOAL_KEY",
    "LIMB_GOAL_VALID_KEY",
    "CONTACT_INDICATOR_KEY",
    "MOTION_JOINT_POS_KEY",
    "MOTION_ROOT_ORIENTATION_KEY",
    "MOTION_TARGET_VALID_KEY",
    "NEXT_PROPRIO_KEY",
    "TransformerVisionPPORolloutStorage",
    "TransformerVisionPPOTransition",
    "VELOCITY_KEY",
]
