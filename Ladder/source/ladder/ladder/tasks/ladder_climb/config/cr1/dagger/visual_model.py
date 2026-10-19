"""Command-aware visual student without reference-motion estimation heads."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from ladder.rsl_rl.modules import TransformerVisionEncoder, TransformerVisionOutput
from ladder.rsl_rl.modules.transformer_vision import TransformerVisionLossWeights

from .student_teacher import elu_mlp


DEPTH_KEY = "depth_history"
PROPRIO_KEY = "proprio_history"
VISUAL_DATA_CONTRACT = "cr1_dagger_visual_5x85_depth2_velocity_heightmap35_v2"


class CR1DaggerVisionEncoder(TransformerVisionEncoder):
    """Deterministic five-frame encoder with no task-goal/motion head."""

    data_contract = VISUAL_DATA_CONTRACT

    def __init__(self) -> None:
        super().__init__(proprio_dim=85, history_frames=5, include_dynamic=False)
        self.dynamic_head = None
        self.future_decoder = None
        self.future_decoder_activation = None
        self.future_decoder_final = None

    def losses(self, output, targets, weights=TransformerVisionLossWeights()):
        """Train only velocity and current-limb height maps."""

        result, _ = self._common_losses(output, targets)
        result["total"] = (
            weights.velocity * result["velocity"]
            + weights.hand * result["hand"]
            + weights.foot * result["foot"]
        )
        return result


class CR1DaggerVisualStudent(nn.Module):
    """Deployable visual actor over 5x85 proprio and a 35-D visual latent."""

    def __init__(
        self,
        action_dim: int,
        hidden_dims: list[int],
        init_noise_std: float,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.encoder_variant = "deterministic"
        self.encoder = CR1DaggerVisionEncoder()
        self.action_mlp = elu_mlp(5 * 85 + 35, hidden_dims, action_dim)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(init_noise_std)).log()
        )
        self._inference_hidden_state: torch.Tensor | None = None

    @property
    def output_std(self) -> torch.Tensor:
        return self.log_std.exp()

    def actor_input(
        self, proprio_history: torch.Tensor, output: TransformerVisionOutput
    ) -> torch.Tensor:
        return torch.cat((proprio_history.flatten(-2), output.latent), dim=-1)

    def action_mean(self, actor_input: torch.Tensor) -> torch.Tensor:
        return self.action_mlp(actor_input)

    def forward(self, observations: TensorDict) -> torch.Tensor:
        proprio = observations[PROPRIO_KEY]
        batch_size = proprio.shape[0]
        if self._inference_hidden_state is None or self._inference_hidden_state.shape[1] != batch_size:
            self._inference_hidden_state = torch.zeros(
                1, batch_size, 256, device=proprio.device, dtype=proprio.dtype
            )
        output = self.encoder(
            proprio,
            observations[DEPTH_KEY],
            hidden_state=self._inference_hidden_state,
            deterministic=True,
        )
        self._inference_hidden_state = output.next_hidden_state.detach()
        return self.action_mean(self.actor_input(proprio, output))

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if self._inference_hidden_state is None:
            return
        if dones is None:
            self._inference_hidden_state.zero_()
        else:
            self._inference_hidden_state = self.encoder.reset_hidden_state(
                self._inference_hidden_state, dones.to(dtype=torch.bool)
            )


__all__ = [
    "DEPTH_KEY",
    "PROPRIO_KEY",
    "VISUAL_DATA_CONTRACT",
    "CR1DaggerVisualStudent",
    "CR1DaggerVisionEncoder",
]
