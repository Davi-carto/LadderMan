"""Actor that consumes the current proprioceptive frame and TransformerVision latent."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from .transformer_vision import (
    LegacyTaskGoalTransformerVisionEncoder,
    TaskGoalTransformerVisionEncoder,
    TransformerVisionEncoder,
    TransformerVisionOutput,
    VariationalTransformerVisionEncoder,
)


CURRENT_PROPRIO_KEY = "transformer_vision_current_proprio"
PROPRIO_HISTORY_KEY = "transformer_vision_proprio_history"
DEPTH_HISTORY_KEY = "transformer_vision_depth_history"


def _elu_mlp(input_dim: int, hidden_dims: tuple[int, ...] | list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class TransformerVisionActor(nn.Module):
    """Deployable visual actor with a persistent per-environment GRU state.

    PPO training uses :meth:`action_mean` with the exact actor input
    stored during rollout.  :meth:`forward` is the deterministic deployment
    path. The active encoder's dynamic latent is deterministic; the retained
    legacy VAE variant uses its mean during deployment.
    """

    def __init__(
        self,
        action_dim: int = 25,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        init_noise_std: float = 1.0,
        encoder_variant: str = "deterministic",
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if init_noise_std <= 0.0:
            raise ValueError("init_noise_std must be positive")
        if encoder_variant not in (
            "deterministic", "variational_legacy", "task_goal", "task_goal_legacy"
        ):
            raise ValueError(
                "encoder_variant must be 'deterministic', 'variational_legacy', or 'task_goal'"
            )
        self.action_dim = action_dim
        self.encoder_variant = encoder_variant
        encoder_types = {
            "deterministic": TransformerVisionEncoder,
            "variational_legacy": VariationalTransformerVisionEncoder,
            "task_goal": TaskGoalTransformerVisionEncoder,
            "task_goal_legacy": LegacyTaskGoalTransformerVisionEncoder,
        }
        encoder_type = encoder_types[encoder_variant]
        self.encoder = encoder_type(proprio_dim=85)
        self.actor_input_dim = 85 + self.encoder.actor_latent_dim
        self.action_mlp = _elu_mlp(
            self.actor_input_dim, actor_hidden_dims, action_dim
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_noise_std)))
        self.register_buffer("_inference_hidden_state", torch.zeros(1, 0, 256), persistent=False)
        Normal.set_default_validate_args(False)

    @property
    def output_std(self) -> torch.Tensor:
        return self.log_std.exp()

    def action_mean(self, actor_input: torch.Tensor) -> torch.Tensor:
        if actor_input.shape[-1] != self.actor_input_dim:
            raise ValueError(
                f"actor_input must end in {self.actor_input_dim} values, "
                f"got {tuple(actor_input.shape)}"
            )
        return self.action_mlp(actor_input)

    def forward(self, observations: TensorDict, stochastic_output: bool = False) -> torch.Tensor:
        """Run one deployment step while retaining GRU state between calls."""

        actions, _ = self.forward_with_output(observations, stochastic_output=stochastic_output)
        return actions

    def forward_with_output(
        self, observations: TensorDict, stochastic_output: bool = False
    ) -> tuple[torch.Tensor, TransformerVisionOutput]:
        """Run one deployment step and expose the matching visual estimates for diagnostics."""

        current, proprio_history, depth_history = self._deployment_inputs(observations)
        batch_size = current.shape[0]
        if self._inference_hidden_state.shape[1] != batch_size:
            self._inference_hidden_state = torch.zeros(
                1, batch_size, 256, device=current.device, dtype=current.dtype
            )
        output = self.encoder(
            proprio_history,
            depth_history,
            hidden_state=self._inference_hidden_state,
            deterministic=True,
        )
        self._inference_hidden_state = output.next_hidden_state.detach()
        mean = self.action_mean(torch.cat((current, output.latent), dim=-1))
        if stochastic_output:
            return Normal(mean, self.output_std.expand_as(mean)).sample(), output
        return mean, output

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Clear every state, or only the environments selected by ``dones``."""

        if self._inference_hidden_state.numel() == 0:
            return
        if dones is None:
            self._inference_hidden_state.zero_()
            return
        if tuple(dones.shape) != (self._inference_hidden_state.shape[1],):
            raise ValueError(
                "dones must have one value per inference environment, got "
                f"{tuple(dones.shape)}"
            )
        self._inference_hidden_state = TransformerVisionEncoder.reset_hidden_state(
            self._inference_hidden_state, dones.to(device=self._inference_hidden_state.device, dtype=torch.bool)
        )

    @staticmethod
    def _deployment_inputs(observations: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = (CURRENT_PROPRIO_KEY, PROPRIO_HISTORY_KEY, DEPTH_HISTORY_KEY)
        missing = [name for name in required if name not in observations]
        if missing:
            raise KeyError(f"Missing TransformerVision actor observations: {missing}")
        current = observations[CURRENT_PROPRIO_KEY]
        proprio_history = observations[PROPRIO_HISTORY_KEY]
        depth_history = observations[DEPTH_HISTORY_KEY]
        batch_size = current.shape[0]
        expected = {
            CURRENT_PROPRIO_KEY: (batch_size, 85),
            PROPRIO_HISTORY_KEY: (batch_size, 10, 85),
            DEPTH_HISTORY_KEY: (batch_size, 2, 60, 80),
        }
        for name, value in (
            (CURRENT_PROPRIO_KEY, current),
            (PROPRIO_HISTORY_KEY, proprio_history),
            (DEPTH_HISTORY_KEY, depth_history),
        ):
            if tuple(value.shape) != expected[name]:
                raise ValueError(f"{name} must have shape {expected[name]}, got {tuple(value.shape)}")
        return current, proprio_history, depth_history


__all__ = [
    "CURRENT_PROPRIO_KEY",
    "DEPTH_HISTORY_KEY",
    "PROPRIO_HISTORY_KEY",
    "TransformerVisionActor",
]
