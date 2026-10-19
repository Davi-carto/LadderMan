"""Sequence-preserving rollout storage for recurrent visual DAgger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from .recurrent_visual_model import (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    PROPRIO_DIM,
    PROPRIO_HISTORY_FRAMES,
    RECURRENT_HIDDEN_DIM,
)


@dataclass
class RecurrentVisualMseDaggerTransition:
    proprio_history: torch.Tensor
    depth: torch.Tensor
    depth_updated: torch.Tensor
    teacher_action_mean: torch.Tensor
    hidden_before: torch.Tensor
    teacher_controlled: torch.Tensor
    privileged: torch.Tensor | None = None


class RecurrentVisualMseDaggerStorage:
    """Store complete per-environment sequences without flattening time."""

    def __init__(self, num_steps, num_envs, action_dim, device, privileged_dim=0) -> None:
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.privileged_dim = int(privileged_dim)
        if self.privileged_dim < 0:
            raise ValueError("privileged_dim must be non-negative")
        self.step = 0
        self.initial_proprio_history = torch.zeros(
            self.num_envs,
            PROPRIO_HISTORY_FRAMES,
            PROPRIO_DIM,
            device=self.device,
        )
        self.initial_hidden = torch.zeros(
            1, self.num_envs, RECURRENT_HIDDEN_DIM, device=self.device
        )
        self.proprio_frame = torch.zeros(
            self.num_steps, self.num_envs, PROPRIO_DIM, device=self.device
        )
        self.depth = torch.zeros(
            self.num_steps,
            self.num_envs,
            1,
            DEPTH_HEIGHT,
            DEPTH_WIDTH,
            device=self.device,
        )
        self.depth_updated = torch.zeros(
            self.num_steps, self.num_envs, dtype=torch.bool, device=self.device
        )
        self.teacher_action_mean = torch.zeros(
            self.num_steps, self.num_envs, action_dim, device=self.device
        )
        self.dones = torch.zeros(
            self.num_steps, self.num_envs, dtype=torch.bool, device=self.device
        )
        self.teacher_controlled = torch.zeros(
            self.num_steps, self.num_envs, dtype=torch.bool, device=self.device
        )
        self.privileged = (
            torch.zeros(
                self.num_steps,
                self.num_envs,
                self.privileged_dim,
                device=self.device,
            )
            if self.privileged_dim
            else None
        )

    def add(
        self,
        transition: RecurrentVisualMseDaggerTransition,
        dones: torch.Tensor,
    ) -> None:
        if self.step >= self.num_steps:
            raise OverflowError("Recurrent visual MSE-DAgger rollout storage is full")
        if self.step == 0:
            self.initial_proprio_history.copy_(transition.proprio_history)
            self.initial_hidden.copy_(transition.hidden_before)
        self.proprio_frame[self.step].copy_(transition.proprio_history[:, -1])
        self.depth[self.step].copy_(transition.depth)
        self.depth_updated[self.step].copy_(
            transition.depth_updated.reshape(self.num_envs).to(dtype=torch.bool)
        )
        self.teacher_action_mean[self.step].copy_(transition.teacher_action_mean)
        self.dones[self.step].copy_(dones.to(dtype=torch.bool))
        self.teacher_controlled[self.step].copy_(
            transition.teacher_controlled.reshape(self.num_envs).to(
                dtype=torch.bool
            )
        )
        if self.privileged_dim:
            if transition.privileged is None:
                raise KeyError("Missing privileged transition for privileged student")
            expected_shape = (self.num_envs, self.privileged_dim)
            if tuple(transition.privileged.shape) != expected_shape:
                raise ValueError(
                    f"privileged transition must have shape {expected_shape}, "
                    f"got {tuple(transition.privileged.shape)}"
                )
            self.privileged[self.step].copy_(transition.privileged)
        elif transition.privileged is not None:
            raise ValueError("Privileged transition supplied to non-privileged storage")
        self.step += 1

    def batches(
        self, num_mini_batches: int, num_epochs: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        if self.step != self.num_steps:
            raise RuntimeError(
                f"Expected {self.num_steps} transitions, stored {self.step}"
            )
        if not 1 <= num_mini_batches <= self.num_envs:
            raise ValueError("num_mini_batches must not exceed num_envs")
        for _ in range(num_epochs):
            permutation = torch.randperm(self.num_envs, device=self.device)
            for env_ids in torch.tensor_split(permutation, num_mini_batches):
                batch = {
                    "proprio_history": self._reconstruct_proprio_history(env_ids),
                    "depth": self.depth[:, env_ids],
                    "depth_updated": self.depth_updated[:, env_ids],
                    "teacher_action_mean": self.teacher_action_mean[:, env_ids],
                    "dones": self.dones[:, env_ids],
                    "teacher_controlled": self.teacher_controlled[:, env_ids],
                    "initial_hidden": self.initial_hidden[:, env_ids],
                }
                if self.privileged_dim:
                    batch["privileged_actor"] = self.privileged[:, env_ids]
                yield batch

    def _reconstruct_proprio_history(
        self, env_ids: torch.Tensor
    ) -> torch.Tensor:
        history = self.initial_proprio_history[env_ids]
        frames = self.proprio_frame[:, env_ids]
        sequences = [history]
        for step in range(1, self.num_steps):
            shifted = torch.cat((history[:, 1:], frames[step, :, None]), dim=1)
            reset = frames[step, :, None].expand(
                -1, PROPRIO_HISTORY_FRAMES, -1
            )
            history = torch.where(
                self.dones[step - 1, env_ids, None, None], reset, shifted
            )
            sequences.append(history)
        return torch.stack(sequences)

    def clear(self) -> None:
        self.step = 0


__all__ = [
    "RecurrentVisualMseDaggerStorage",
    "RecurrentVisualMseDaggerTransition",
]
