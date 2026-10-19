"""Depth encoder and visual actor model for LadderMan."""

from __future__ import annotations

import torch
from rsl_rl.models import CNNModel
from rsl_rl.modules import CNN
from tensordict import TensorDict


class LadderDepthEncoder(CNN):
    """Three-layer depth CNN from LadderMan Appendix A.2.

    The paper specifies channels and kernel size but not strides. Stride two is
    used for all layers to keep the 60 x 80 visual latent compact.
    """

    def __init__(self, input_dim: tuple[int, int], input_channels: int = 1) -> None:
        super().__init__(
            input_dim=input_dim,
            input_channels=input_channels,
            output_channels=[8, 8, 8],
            kernel_size=5,
            stride=2,
            padding="none",
            activation="elu",
            flatten=True,
        )


class LadderVisualActorModel(CNNModel):
    """RSL-RL actor that fuses proprioception with encoded depth."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (2048, 1024, 512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        depth_groups = [
            name for name in obs_groups[obs_set] if isinstance(obs[name], torch.Tensor) and obs[name].ndim == 4
        ]
        if len(depth_groups) != 1:
            raise ValueError(f"Ladder visual actor requires one depth group, got {depth_groups}")
        depth_group = depth_groups[0]
        depth = obs[depth_group]
        if depth.shape[1] != 1:
            raise ValueError(f"Ladder depth must have one channel, got shape {tuple(depth.shape)}")

        encoder = LadderDepthEncoder(
            input_dim=(int(depth.shape[2]), int(depth.shape[3])),
            input_channels=int(depth.shape[1]),
        )
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            cnns={depth_group: encoder},
        )


__all__ = ["LadderDepthEncoder", "LadderVisualActorModel"]
