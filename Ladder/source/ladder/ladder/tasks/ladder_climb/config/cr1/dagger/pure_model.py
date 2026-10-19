"""Reference-free visual student for pure LadderMan-style DAgger distillation."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from .visual_model import DEPTH_KEY, PROPRIO_KEY


PURE_VISUAL_DAGGER_DATA_CONTRACT = "cr1_pure_visual_dagger_9x85_depth2_cnn_v1"


class CR1PaperDepthEncoder(nn.Module):
    """Three 8-channel, kernel-5 convolutions adapted to 2x60x80 depth history."""

    output_dim = 8 * 4 * 7

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Flatten(),
        )

    def forward(self, depth_history: torch.Tensor) -> torch.Tensor:
        return self.network(depth_history)


class CR1PureVisualDaggerStudent(nn.Module):
    """Nine-frame proprioception plus direct CNN features, trained only by distillation."""

    def __init__(
        self,
        action_dim: int,
        hidden_dims: list[int],
        init_noise_std: float,
    ) -> None:
        super().__init__()
        if not hidden_dims or hidden_dims[-1] != 256:
            raise ValueError("The final student policy hidden layer must be 256-D")
        self.action_dim = action_dim
        self.depth_encoder = CR1PaperDepthEncoder()
        input_dim = 9 * 85 + self.depth_encoder.output_dim
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
            previous_dim = hidden_dim
        self.feature_mlp = nn.Sequential(*layers)
        self.action_head = nn.Linear(previous_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_noise_std)).log())

    @property
    def output_std(self) -> torch.Tensor:
        return self.log_std.exp()

    def action_mean_and_feature(self, observations: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        proprio = observations[PROPRIO_KEY]
        depth_feature = self.depth_encoder(observations[DEPTH_KEY])
        actor_input = torch.cat((proprio.flatten(-2), depth_feature), dim=-1)
        feature = self.feature_mlp(actor_input)
        return self.action_head(feature), feature

    def forward(self, observations: TensorDict) -> torch.Tensor:
        mean, _ = self.action_mean_and_feature(observations)
        return mean

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones


__all__ = [
    "CR1PaperDepthEncoder",
    "CR1PureVisualDaggerStudent",
    "PURE_VISUAL_DAGGER_DATA_CONTRACT",
]
