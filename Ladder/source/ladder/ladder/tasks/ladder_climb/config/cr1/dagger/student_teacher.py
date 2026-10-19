"""Networks and frozen-teacher loading for CR1 DAgger."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn


TEACHER_KEY = "teacher"


def elu_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


def load_frozen_teacher(
    checkpoint: str,
    *,
    observation_dim: int,
    action_dim: int,
    hidden_dims: list[int],
    device: str | torch.device,
) -> nn.Sequential:
    """Load only the deterministic actor MLP from an RSL-RL checkpoint."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint was not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Teacher checkpoint payload must be a dictionary")

    if "actor_state_dict" in payload:
        source = payload["actor_state_dict"]
        state = OrderedDict(
            (key.removeprefix("mlp."), value)
            for key, value in source.items()
            if key.startswith("mlp.")
        )
    elif "model_state_dict" in payload:
        source = payload["model_state_dict"]
        state = OrderedDict(
            (key.removeprefix("actor."), value)
            for key, value in source.items()
            if key.startswith("actor.")
        )
    else:
        raise KeyError(
            "Teacher checkpoint must contain actor_state_dict or model_state_dict"
        )
    if not state:
        raise KeyError("Teacher checkpoint contains no supported actor MLP parameters")

    teacher = elu_mlp(observation_dim, hidden_dims, action_dim)
    teacher.load_state_dict(state, strict=True)
    teacher.to(device).eval().requires_grad_(False)
    return teacher


__all__ = [
    "TEACHER_KEY",
    "elu_mlp",
    "load_frozen_teacher",
]
