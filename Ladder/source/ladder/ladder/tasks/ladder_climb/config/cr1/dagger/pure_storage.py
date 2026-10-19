"""Rollout storage containing only data required by pure visual DAgger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class PureVisualDaggerTransition:
    proprio_history: torch.Tensor
    depth_history: torch.Tensor
    teacher_action_mean: torch.Tensor
    teacher_feature: torch.Tensor


class PureVisualDaggerStorage:
    def __init__(self, num_steps, num_envs, action_dim, teacher_feature_dim, device) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        # Teacher-only BC reserves a fixed suffix of environments for validation.
        # The original mixed DAgger path continues to train on every environment.
        self.num_training_envs = num_envs
        self.device = torch.device(device)
        self.step = 0
        self.proprio_history = torch.zeros(num_steps, num_envs, 9, 85, device=self.device)
        self.depth_history = torch.zeros(num_steps, num_envs, 2, 60, 80, device=self.device)
        self.teacher_action_mean = torch.zeros(num_steps, num_envs, action_dim, device=self.device)
        self.teacher_feature = torch.zeros(
            num_steps, num_envs, teacher_feature_dim, device=self.device
        )

    def add(self, transition: PureVisualDaggerTransition) -> None:
        if self.step >= self.num_steps:
            raise OverflowError("Pure visual DAgger rollout storage is full")
        index = self.step
        self.proprio_history[index].copy_(transition.proprio_history)
        self.depth_history[index].copy_(transition.depth_history)
        self.teacher_action_mean[index].copy_(transition.teacher_action_mean)
        self.teacher_feature[index].copy_(transition.teacher_feature)
        self.step += 1

    def batches(self, num_mini_batches: int, num_epochs: int) -> Iterator[dict[str, torch.Tensor]]:
        if self.step != self.num_steps:
            raise RuntimeError(f"Expected {self.num_steps} transitions, stored {self.step}")
        total = self.num_steps * self.num_training_envs
        if not 1 <= self.num_training_envs <= self.num_envs:
            raise ValueError("num_training_envs must be in [1, num_envs]")
        if not 1 <= num_mini_batches <= total:
            raise ValueError("num_mini_batches must not exceed the training sample count")
        flat = {
            "proprio_history": self.proprio_history.flatten(0, 1),
            "depth_history": self.depth_history.flatten(0, 1),
            "teacher_action_mean": self.teacher_action_mean.flatten(0, 1),
            "teacher_feature": self.teacher_feature.flatten(0, 1),
        }
        for _ in range(num_epochs):
            for selected in torch.tensor_split(
                torch.randperm(total, device=self.device), num_mini_batches
            ):
                selected = (
                    selected // self.num_training_envs * self.num_envs
                    + selected % self.num_training_envs
                )
                yield {name: value[selected] for name, value in flat.items()}

    def clear(self) -> None:
        self.step = 0


__all__ = ["PureVisualDaggerStorage", "PureVisualDaggerTransition"]
