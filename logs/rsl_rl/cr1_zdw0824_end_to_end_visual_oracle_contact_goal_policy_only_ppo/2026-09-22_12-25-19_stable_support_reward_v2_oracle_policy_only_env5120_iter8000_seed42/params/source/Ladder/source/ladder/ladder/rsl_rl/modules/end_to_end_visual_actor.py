"""Lightweight recurrent visual actor for raw-sequence end-to-end PPO."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from .transformer_vision import TransformerVisionEncoder
from .transformer_vision_actor import (
    CURRENT_PROPRIO_KEY,
    DEPTH_HISTORY_KEY,
    PROPRIO_HISTORY_KEY,
    TransformerVisionActor,
    _elu_mlp,
)


END_TO_END_VISUAL_DATA_CONTRACT = (
    "cr1_end_to_end_visual_policy_only_proprio25x85_depth2x60x80_30hz_"
    "cnn16_32_64_prop128_depth128_gru256_v2"
)
ORACLE_CONTACT_GOAL_KEY = "oracle_contact_goal"
ORACLE_CONTACT_GOAL_DATA_CONTRACT = (
    "cr1_end_to_end_visual_oracle_policy_only_proprio25x85_depth2x60x80_30hz_"
    "cnn16_32_64_prop128_depth128_gru256_next_stable_goal12_actual_limb_pos12_"
    "ground_or_rung_contact4_desired_contact4_v4"
)


@dataclass
class EndToEndVisualOutput:
    """Only recurrent policy features; there are no reconstruction outputs."""

    recurrent_feature: torch.Tensor
    valid_mask: torch.Tensor
    next_hidden_state: torch.Tensor

    @property
    def latent(self) -> torch.Tensor:
        """Compatibility alias used by policy diagnostics and playback."""

        return self.recurrent_feature


class EndToEndVisualEncoder(TransformerVisionEncoder):
    """Encode proprioceptive history and depth directly into a GRU-256 feature."""

    data_contract = END_TO_END_VISUAL_DATA_CONTRACT

    def __init__(self) -> None:
        # Reuse only the validated reset-aware GRU utilities from the established
        # encoder. Do not instantiate its Transformer or reconstruction heads.
        nn.Module.__init__(self)
        self.proprio_dim = 85
        self.history_frames = 25
        self.include_dynamic = False
        self.actor_latent_dim = 256
        self.state_encoder = _elu_mlp(25 * 85, (256,), 128)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        ).to(memory_format=torch.channels_last)
        self.depth_pool = nn.AdaptiveAvgPool2d((2, 4))
        self.depth_projection = nn.Sequential(nn.Linear(64 * 2 * 4, 128), nn.ELU())
        self.gru = nn.GRU(256, 256, num_layers=1)

    def forward(
        self,
        proprio_history: torch.Tensor,
        depth_history: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        deterministic: bool = False,
        dynamic_noise: torch.Tensor | None = None,
    ) -> EndToEndVisualOutput:
        del deterministic
        if dynamic_noise is not None:
            raise ValueError("The policy-only visual encoder has no stochastic latent")
        leading_shape, depth_history = self._validate_inputs(proprio_history, depth_history)
        batch_size = math.prod(leading_shape)
        proprio_feature = self.state_encoder(
            proprio_history.reshape(batch_size, -1)
        ).reshape(*leading_shape, 128)
        depth_batch = depth_history.reshape(batch_size, 2, 60, 80).contiguous(
            memory_format=torch.channels_last
        )
        depth_feature = self.depth_projection(
            self.depth_pool(self.depth_encoder(depth_batch)).flatten(start_dim=1)
        ).reshape(*leading_shape, 128)
        fused = torch.cat((proprio_feature, depth_feature), dim=-1)
        recurrent, next_hidden_state, valid_mask = self._run_gru(
            fused, leading_shape, hidden_state, dones, masks
        )
        return EndToEndVisualOutput(
            recurrent_feature=recurrent,
            valid_mask=valid_mask,
            next_hidden_state=next_hidden_state,
        )


class EndToEndVisualActor(TransformerVisionActor):
    """Policy-only recurrent visual actor, optionally with privileged task intent."""

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
        self.encoder_variant = "end_to_end_policy_only"
        self.encoder = EndToEndVisualEncoder()
        self.use_oracle_contact_goal = use_oracle_contact_goal
        if use_oracle_contact_goal:
            self.data_contract = ORACLE_CONTACT_GOAL_DATA_CONTRACT
            self.encoder.data_contract = self.data_contract
        self.actor_visual_dim = 256
        self.oracle_input_dim = 32 if use_oracle_contact_goal else 0
        self.actor_input_dim = 85 + self.actor_visual_dim + self.oracle_input_dim
        # Keep the requested actor architecture unchanged.
        self.action_mlp = _elu_mlp(self.actor_input_dim, actor_hidden_dims, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_noise_std)))
        self.register_buffer("_inference_hidden_state", torch.zeros(1, 0, 256), persistent=False)

    @staticmethod
    def actor_visual_feature(output: EndToEndVisualOutput) -> torch.Tensor:
        return output.recurrent_feature

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
        output = self.encoder(
            history, depth, hidden_state=self._inference_hidden_state, deterministic=True
        )
        actor_input = self.make_actor_input(
            current, self.actor_visual_feature(output), observations.get(ORACLE_CONTACT_GOAL_KEY)
        )
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
    ) -> tuple[torch.Tensor, EndToEndVisualOutput]:
        """Recompute every policy feature with ``dones[t]`` resetting before step t."""

        if proprio_history.ndim != 4:
            raise ValueError("Sequence proprio_history must have shape [T,B,25,85]")
        if tuple(current_proprio.shape) != (*proprio_history.shape[:2], 85):
            raise ValueError("Sequence current_proprio must have shape [T,B,85]")
        output = self.encoder(
            proprio_history,
            depth_history,
            hidden_state=hidden_state,
            dones=dones,
            deterministic=True,
        )
        actor_input = self.make_actor_input(
            current_proprio, self.actor_visual_feature(output), oracle_contact_goal
        )
        return self.action_mean(actor_input), output

    @staticmethod
    def _deployment_inputs(
        observations: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    "EndToEndVisualOutput",
]
