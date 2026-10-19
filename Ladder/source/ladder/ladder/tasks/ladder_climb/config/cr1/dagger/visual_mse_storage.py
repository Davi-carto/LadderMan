"""Rollout storage for the visual action-MSE DAgger control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class VisualMseDaggerTransition:
    proprio_history: torch.Tensor
    depth_history: torch.Tensor
    teacher_action_mean: torch.Tensor


class VisualMseDaggerStorage:
    """Store only student inputs and deterministic teacher action labels."""

    def __init__(self, num_steps, num_envs, action_dim, device) -> None:
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.step = 0
        self.proprio_history = torch.zeros(
            self.num_steps, self.num_envs, 9, 85, device=self.device
        )
        self.depth_history = torch.zeros(
            self.num_steps, self.num_envs, 2, 60, 80, device=self.device
        )
        self.teacher_action_mean = torch.zeros(
            self.num_steps, self.num_envs, action_dim, device=self.device
        )

    def add(self, transition: VisualMseDaggerTransition) -> None:
        if self.step >= self.num_steps:
            raise OverflowError("Visual MSE-DAgger rollout storage is full")
        self.proprio_history[self.step].copy_(transition.proprio_history)
        self.depth_history[self.step].copy_(transition.depth_history)
        self.teacher_action_mean[self.step].copy_(transition.teacher_action_mean)
        self.step += 1

    def batches(
        self, num_mini_batches: int, num_epochs: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        if self.step != self.num_steps:
            raise RuntimeError(
                f"Expected {self.num_steps} transitions, stored {self.step}"
            )
        total = self.num_steps * self.num_envs
        if not 1 <= num_mini_batches <= total:
            raise ValueError("num_mini_batches must not exceed the sample count")
        flat = {
            "proprio_history": self.proprio_history.flatten(0, 1),
            "depth_history": self.depth_history.flatten(0, 1),
            "teacher_action_mean": self.teacher_action_mean.flatten(0, 1),
        }
        for _ in range(num_epochs):
            permutation = torch.randperm(total, device=self.device)
            for selected in torch.tensor_split(permutation, num_mini_batches):
                yield {name: value[selected] for name, value in flat.items()}

    def clear(self) -> None:
        self.step = 0


__all__ = ["VisualMseDaggerStorage", "VisualMseDaggerTransition"]
