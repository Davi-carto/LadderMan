"""Deterministic visual reconstruction actor for raw-sequence end-to-end PPO."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from .transformer_vision import TransformerVisionEncoder, TransformerVisionOutput
from .transformer_vision_actor import (
    CURRENT_PROPRIO_KEY,
    DEPTH_HISTORY_KEY,
    PROPRIO_HISTORY_KEY,
    TransformerVisionActor,
    _elu_mlp,
)


END_TO_END_VISUAL_DATA_CONTRACT = (
    "cr1_end_to_end_visual_reconstruction_proprio25x85_depth2x60x80_30hz_"
    "gru50hz_velocity3_limb_latent32_no_dynamic_no_reference_dropout0_v1"
)
ORACLE_CONTACT_GOAL_KEY = "oracle_contact_goal"
ORACLE_CONTACT_GOAL_DATA_CONTRACT = (
    "cr1_end_to_end_visual_oracle_proprio25x85_depth2x60x80_30hz_gru50hz_"
    "gru_feature256_next_stable_goal12_actual_limb_pos12_actual_contact4_"
    "desired_contact4_v2"
)


@dataclass
class EndToEndVisualTargets:
    velocity: torch.Tensor
    limb_heightmaps: torch.Tensor
    heightmap_valid: torch.Tensor
    heightmap_hit_class: torch.Tensor


@dataclass(frozen=True)
class EndToEndVisualLossWeights:
    velocity: float = 2.0
    hand: float = 2.0
    foot: float = 2.0


class EndToEndVisualEncoder(TransformerVisionEncoder):
    """Estimate velocity and four limb maps; update GRU at every policy step."""

    data_contract = END_TO_END_VISUAL_DATA_CONTRACT

    def __init__(self) -> None:
        super().__init__(proprio_dim=85, history_frames=25, include_dynamic=False)
        # The base creates these modules even when include_dynamic is false.
        # Remove them so the new checkpoint owns only the active estimators.
        del self.dynamic_head
        del self.future_decoder
        del self.future_decoder_activation
        del self.future_decoder_final
        # PPO must reproduce its rollout distribution before any update.
        # Disable dropout in both module layers and attention probability masks.
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            elif isinstance(module, nn.MultiheadAttention):
                module.dropout = 0.0

    def losses(
        self,
        output: TransformerVisionOutput,
        targets: EndToEndVisualTargets,
        weights: EndToEndVisualLossWeights = EndToEndVisualLossWeights(),
    ) -> dict[str, torch.Tensor]:
        # Preserve the established field-wise masked MSE: rung pixels have
        # weight 2 and other valid pixels weight 1, including in the denominator.
        result, _ = self._common_losses(output, targets)
        result["total"] = (
            weights.velocity * result["velocity"]
            + weights.hand * result["hand"]
            + weights.foot * result["foot"]
        )
        return result


class EndToEndVisualActor(TransformerVisionActor):
    """Visual actor, optionally given privileged task intent for an oracle test."""

    data_contract = END_TO_END_VISUAL_DATA_CONTRACT

    def __init__(
        self,
        action_dim: int = 25,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        init_noise_std: float = 1.0,
        use_oracle_contact_goal: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if init_noise_std <= 0.0:
            raise ValueError("init_noise_std must be positive")
        self.action_dim = action_dim
        self.encoder_variant = "end_to_end_reconstruction"
        self.encoder = EndToEndVisualEncoder()
        self.use_oracle_contact_goal = use_oracle_contact_goal
        if use_oracle_contact_goal:
            self.data_contract = ORACLE_CONTACT_GOAL_DATA_CONTRACT
            self.encoder.data_contract = self.data_contract
        self.actor_visual_dim = 256 if use_oracle_contact_goal else self.encoder.actor_latent_dim
        self.oracle_input_dim = 32 if use_oracle_contact_goal else 0
        self.actor_input_dim = 85 + self.actor_visual_dim + self.oracle_input_dim
        self.action_mlp = _elu_mlp(self.actor_input_dim, actor_hidden_dims, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_noise_std)))
        self.register_buffer("_inference_hidden_state", torch.zeros(1, 0, 256), persistent=False)

    def actor_visual_feature(self, output: TransformerVisionOutput) -> torch.Tensor:
        return output.recurrent_feature if self.use_oracle_contact_goal else output.latent

    def make_actor_input(self, current, visual_feature, oracle_contact_goal=None):
        expected_visual = (*current.shape[:-1], self.actor_visual_dim)
        if tuple(visual_feature.shape) != expected_visual:
            raise ValueError(f"visual_feature must have shape {expected_visual}")
        if not self.use_oracle_contact_goal:
            return torch.cat((current, visual_feature), dim=-1)
        if oracle_contact_goal is None:
            raise KeyError("Oracle actor requires oracle_contact_goal observations")
        expected = (*current.shape[:-1], self.oracle_input_dim)
        if tuple(oracle_contact_goal.shape) != expected:
            raise ValueError(f"oracle_contact_goal must have shape {expected}")
        return torch.cat((current, visual_feature, oracle_contact_goal), dim=-1)

    def forward_with_output(self, observations: TensorDict, stochastic_output: bool = False):
        current, history, depth = self._deployment_inputs(observations)
        if self._inference_hidden_state.shape[1] != current.shape[0]:
            self._inference_hidden_state = current.new_zeros(1, current.shape[0], 256)
        output = self.encoder(history, depth, hidden_state=self._inference_hidden_state,
                              deterministic=True)
        actor_input = self.make_actor_input(current, self.actor_visual_feature(output),
                                            observations.get(ORACLE_CONTACT_GOAL_KEY))
        self._inference_hidden_state = output.next_hidden_state.detach()
        mean = self.action_mean(actor_input)
        if stochastic_output:
            mean = Normal(mean, self.output_std.expand_as(mean)).sample()
        return mean, output

    def forward_sequence(
        self,
        current_proprio: torch.Tensor,
        proprio_history: torch.Tensor,
        depth_history: torch.Tensor,
        hidden_state: torch.Tensor,
        dones: torch.Tensor,
        oracle_contact_goal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TransformerVisionOutput]:
        """Recompute every latent with ``dones[t]`` resetting before step t.

        Raw depth already follows the camera's 30 Hz sample-and-hold schedule.
        The GRU still consumes new proprioception at every 50 Hz policy step.
        """
        if proprio_history.ndim != 4:
            raise ValueError("Sequence proprio_history must have shape [T,B,25,85]")
        if tuple(current_proprio.shape) != (*proprio_history.shape[:2], 85):
            raise ValueError("Sequence current_proprio must have shape [T,B,85]")
        output = self.encoder(
            proprio_history, depth_history, hidden_state=hidden_state,
            dones=dones, deterministic=True,
        )
        actor_input = self.make_actor_input(
            current_proprio, self.actor_visual_feature(output), oracle_contact_goal
        )
        return self.action_mean(actor_input), output

    @staticmethod
    def _deployment_inputs(observations: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = (CURRENT_PROPRIO_KEY, PROPRIO_HISTORY_KEY, DEPTH_HISTORY_KEY)
        missing = [name for name in required if name not in observations]
        if missing:
            raise KeyError(f"Missing end-to-end visual actor observations: {missing}")
        current, proprio_history, depth_history = (observations[name] for name in required)
        batch_size = current.shape[0]
        expected = ((batch_size, 85), (batch_size, 25, 85), (batch_size, 2, 60, 80))
        for name, value, shape in zip(required, (current, proprio_history, depth_history), expected):
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        return current, proprio_history, depth_history


__all__ = [
    "END_TO_END_VISUAL_DATA_CONTRACT",
    "ORACLE_CONTACT_GOAL_KEY",
    "ORACLE_CONTACT_GOAL_DATA_CONTRACT",
    "EndToEndVisualActor",
    "EndToEndVisualEncoder",
    "EndToEndVisualTargets",
    "EndToEndVisualLossWeights",
]
