"""Pure PyTorch recurrent visual actor for the CR1 SSR-style policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


_HISTORY_FRAMES = 10
_DEPTH_FRAMES = 2
_DEPTH_HEIGHT = 60
_DEPTH_WIDTH = 80
_TOKEN_DIM = 64
_VISUAL_TOKEN_COUNT = 10
_GRU_HIDDEN_DIM = 256
_MAP_SHAPE = (10, 3)


@dataclass
class LadderVisualOutput:
    """Named deployable estimates and training-only reconstructions."""

    state_token: torch.Tensor
    visual_tokens: torch.Tensor
    transformer_tokens: torch.Tensor
    fused_features: torch.Tensor
    recurrent_features: torch.Tensor
    velocity: torch.Tensor
    hand_left_latent: torch.Tensor
    hand_right_latent: torch.Tensor
    foot_left_latent: torch.Tensor
    foot_right_latent: torch.Tensor
    dynamic_mu: torch.Tensor
    dynamic_logvar: torch.Tensor
    dynamic_latent: torch.Tensor
    actor_latent: torch.Tensor
    actor_input: torch.Tensor
    actions: torch.Tensor
    hand_left_map: torch.Tensor
    hand_right_map: torch.Tensor
    foot_left_map: torch.Tensor
    foot_right_map: torch.Tensor
    next_proprio: torch.Tensor
    valid_mask: torch.Tensor
    next_hidden_state: torch.Tensor


@dataclass
class LadderVisualTargets:
    """Privileged targets consumed only by :meth:`auxiliary_losses`."""

    velocity: torch.Tensor
    limb_heightmaps: torch.Tensor
    heightmap_valid: torch.Tensor
    next_proprio: torch.Tensor
    future_valid: torch.Tensor | None = None


@dataclass(frozen=True)
class LadderVisualLossWeights:
    """Initial auxiliary-loss coefficients accepted for the Ladder task."""

    velocity: float = 2.0
    hand: float = 2.0
    foot: float = 2.0
    future: float = 5.0
    kl: float = 1.0


def _elu_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class LadderSSRVisualActor(nn.Module):
    """Transformer-GRU visual actor with SSR-style auxiliary estimators.

    The model intentionally has no IsaacLab, environment, or PPO dependency.
    ``forward`` accepts either one rollout step ``[B,...]`` or a padded time
    batch ``[T,B,...]``.  A true value in ``dones`` resets that environment's
    GRU state immediately before processing the corresponding observation.
    """

    def __init__(self, proprio_dim: int, action_dim: int = 25) -> None:
        super().__init__()
        if proprio_dim <= 0:
            raise ValueError(f"proprio_dim must be positive, got {proprio_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.state_encoder = _elu_mlp(proprio_dim * _HISTORY_FRAMES, (256, 128), _TOKEN_DIM)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(_DEPTH_FRAMES, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )
        self.depth_pool = nn.AdaptiveAvgPool2d((2, 5))
        self.visual_token_encoder = nn.Sequential(nn.Linear(128, _TOKEN_DIM), nn.ELU())
        self.transformer = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=_TOKEN_DIM,
                    nhead=1,
                    dim_feedforward=256,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(2)
            ]
        )
        self.fusion = nn.Sequential(nn.Linear(11 * _TOKEN_DIM, _GRU_HIDDEN_DIM), nn.ELU())
        self.gru = nn.GRU(input_size=_GRU_HIDDEN_DIM, hidden_size=_GRU_HIDDEN_DIM, num_layers=1)

        self.velocity_head = nn.Linear(_GRU_HIDDEN_DIM, 3)
        self.hand_left_head = nn.Linear(_GRU_HIDDEN_DIM, 8)
        self.hand_right_head = nn.Linear(_GRU_HIDDEN_DIM, 8)
        self.foot_left_head = nn.Linear(_GRU_HIDDEN_DIM, 8)
        self.foot_right_head = nn.Linear(_GRU_HIDDEN_DIM, 8)
        self.dynamic_mu_head = nn.Linear(_GRU_HIDDEN_DIM, 16)
        self.dynamic_logvar_head = nn.Linear(_GRU_HIDDEN_DIM, 16)

        map_size = _MAP_SHAPE[0] * _MAP_SHAPE[1]
        self.hand_left_decoder = _elu_mlp(8, (64, 128), map_size)
        self.hand_right_decoder = _elu_mlp(8, (64, 128), map_size)
        self.foot_left_decoder = _elu_mlp(8, (64, 128), map_size)
        self.foot_right_decoder = _elu_mlp(8, (64, 128), map_size)
        self.future_decoder = _elu_mlp(51, (64, 128), proprio_dim)
        self.future_decoder_final = nn.Linear(proprio_dim, proprio_dim)

        # The width follows the primary SSR reference's default actor MLP.
        self.actor = _elu_mlp(proprio_dim + 51, (512, 256, 128), action_dim)
        self.action_std = nn.Parameter(torch.ones(action_dim))

    @staticmethod
    def reset_hidden_state(hidden_state: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Return a hidden state with exactly the selected environments reset."""

        if hidden_state.ndim != 3 or hidden_state.shape[0] != 1:
            raise ValueError(
                f"hidden_state must have shape [1,B,256], got {tuple(hidden_state.shape)}"
            )
        if dones.ndim != 1 or dones.shape[0] != hidden_state.shape[1]:
            raise ValueError(
                f"dones must have shape [{hidden_state.shape[1]}], got {tuple(dones.shape)}"
            )
        return hidden_state * (~dones.to(dtype=torch.bool))[None, :, None].to(hidden_state.dtype)

    def action_distribution(self, output: LadderVisualOutput) -> Normal:
        """Return the diagonal Gaussian used by the later PPO integration."""

        return Normal(output.actions, self.output_std.expand_as(output.actions))

    @property
    def output_std(self) -> torch.Tensor:
        """Keep the learned standard deviation valid for a Normal distribution."""

        return self.action_std.clamp_min(1.0e-6)

    def forward(
        self,
        obs_history: torch.Tensor,
        depth: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        deterministic: bool = False,
        dynamic_noise: torch.Tensor | None = None,
    ) -> LadderVisualOutput:
        """Encode rollout or padded-sequence visual observations.

        ``masks`` is only relevant to padded sequences.  Invalid entries do
        not advance hidden state and are excluded by ``auxiliary_losses``.
        """

        leading_shape, depth = self._validate_inputs(obs_history, depth)
        state_token, visual_tokens, transformer_tokens, fused_features = self._encode(
            obs_history, depth, leading_shape
        )
        recurrent_features, next_hidden_state, valid_mask = self._run_gru(
            fused_features, leading_shape, hidden_state, dones, masks
        )

        velocity = self.velocity_head(recurrent_features)
        hand_left_latent = self.hand_left_head(recurrent_features)
        hand_right_latent = self.hand_right_head(recurrent_features)
        foot_left_latent = self.foot_left_head(recurrent_features)
        foot_right_latent = self.foot_right_head(recurrent_features)
        dynamic_mu = self.dynamic_mu_head(recurrent_features)
        dynamic_logvar = self.dynamic_logvar_head(recurrent_features)
        if deterministic:
            dynamic_latent = dynamic_mu
        else:
            if dynamic_noise is None:
                dynamic_noise = torch.randn_like(dynamic_mu)
            if dynamic_noise.shape != dynamic_mu.shape:
                raise ValueError(
                    f"dynamic_noise must have shape {tuple(dynamic_mu.shape)}, got {tuple(dynamic_noise.shape)}"
                )
            dynamic_latent = dynamic_mu + torch.exp(0.5 * dynamic_logvar) * dynamic_noise

        actor_latent = torch.cat(
            (
                velocity,
                hand_left_latent,
                hand_right_latent,
                foot_left_latent,
                foot_right_latent,
                dynamic_latent,
            ),
            dim=-1,
        )
        current_proprio = obs_history[..., -1, :]
        actor_input = torch.cat((current_proprio, actor_latent), dim=-1)
        return LadderVisualOutput(
            state_token=state_token,
            visual_tokens=visual_tokens,
            transformer_tokens=transformer_tokens,
            fused_features=fused_features,
            recurrent_features=recurrent_features,
            velocity=velocity,
            hand_left_latent=hand_left_latent,
            hand_right_latent=hand_right_latent,
            foot_left_latent=foot_left_latent,
            foot_right_latent=foot_right_latent,
            dynamic_mu=dynamic_mu,
            dynamic_logvar=dynamic_logvar,
            dynamic_latent=dynamic_latent,
            actor_latent=actor_latent,
            actor_input=actor_input,
            actions=self.actor(actor_input),
            hand_left_map=self.hand_left_decoder(hand_left_latent).reshape(*leading_shape, *_MAP_SHAPE),
            hand_right_map=self.hand_right_decoder(hand_right_latent).reshape(*leading_shape, *_MAP_SHAPE),
            foot_left_map=self.foot_left_decoder(foot_left_latent).reshape(*leading_shape, *_MAP_SHAPE),
            foot_right_map=self.foot_right_decoder(foot_right_latent).reshape(*leading_shape, *_MAP_SHAPE),
            next_proprio=self.future_decoder_final(self.future_decoder(actor_latent)),
            valid_mask=valid_mask,
            next_hidden_state=next_hidden_state,
        )

    def auxiliary_losses(
        self,
        output: LadderVisualOutput,
        targets: LadderVisualTargets,
        weights: LadderVisualLossWeights = LadderVisualLossWeights(),
    ) -> dict[str, torch.Tensor]:
        """Compute the five training-only losses without exposing targets to the actor."""

        expected_leading = output.velocity.shape[:-1]
        expected_maps = (*expected_leading, 4, *_MAP_SHAPE)
        if targets.velocity.shape != output.velocity.shape:
            raise ValueError(
                f"velocity target must have shape {tuple(output.velocity.shape)}, got {tuple(targets.velocity.shape)}"
            )
        if targets.limb_heightmaps.shape != expected_maps:
            raise ValueError(
                f"limb_heightmaps must have shape {expected_maps}, got {tuple(targets.limb_heightmaps.shape)}"
            )
        if targets.heightmap_valid.shape != expected_maps:
            raise ValueError(
                f"heightmap_valid must have shape {expected_maps}, got {tuple(targets.heightmap_valid.shape)}"
            )
        if targets.next_proprio.shape != output.next_proprio.shape:
            raise ValueError(
                f"next_proprio target must have shape {tuple(output.next_proprio.shape)}, got {tuple(targets.next_proprio.shape)}"
            )

        valid = output.valid_mask.to(dtype=output.velocity.dtype)
        map_valid = targets.heightmap_valid.to(dtype=output.velocity.dtype) * valid[..., None, None, None]
        future_valid = valid if targets.future_valid is None else valid * targets.future_valid.to(valid.dtype)

        velocity_loss = self._masked_mse(output.velocity, targets.velocity, valid)
        hand_loss = 0.5 * (
            self._masked_mse(output.hand_left_map, targets.limb_heightmaps[..., 0, :, :], map_valid[..., 0, :, :])
            + self._masked_mse(output.hand_right_map, targets.limb_heightmaps[..., 1, :, :], map_valid[..., 1, :, :])
        )
        foot_loss = 0.5 * (
            self._masked_mse(output.foot_left_map, targets.limb_heightmaps[..., 2, :, :], map_valid[..., 2, :, :])
            + self._masked_mse(output.foot_right_map, targets.limb_heightmaps[..., 3, :, :], map_valid[..., 3, :, :])
        )
        future_loss = self._masked_mse(output.next_proprio, targets.next_proprio, future_valid)
        kl_per_sample = -0.5 * torch.sum(
            1.0 + output.dynamic_logvar - output.dynamic_mu.square() - output.dynamic_logvar.exp(), dim=-1
        )
        kl_loss = self._masked_mean(kl_per_sample, valid)
        total = (
            weights.velocity * velocity_loss
            + weights.hand * hand_loss
            + weights.foot * foot_loss
            + weights.future * future_loss
            + weights.kl * kl_loss
        )
        return {
            "velocity": velocity_loss,
            "hand": hand_loss,
            "foot": foot_loss,
            "future": future_loss,
            "kl": kl_loss,
            "total": total,
        }

    def _validate_inputs(
        self, obs_history: torch.Tensor, depth: torch.Tensor
    ) -> tuple[tuple[int, ...], torch.Tensor]:
        if obs_history.ndim not in (3, 4):
            raise ValueError(
                "obs_history must have rollout shape [B,10,D] or sequence shape [T,B,10,D], "
                f"got {tuple(obs_history.shape)}"
            )
        if tuple(obs_history.shape[-2:]) != (_HISTORY_FRAMES, self.proprio_dim):
            raise ValueError(
                f"obs_history must end in ({_HISTORY_FRAMES},{self.proprio_dim}), got {tuple(obs_history.shape)}"
            )
        leading_shape = tuple(obs_history.shape[:-2])
        no_channel_shape = (*leading_shape, _DEPTH_FRAMES, _DEPTH_HEIGHT, _DEPTH_WIDTH)
        singleton_channel_shape = (*leading_shape, _DEPTH_FRAMES, 1, _DEPTH_HEIGHT, _DEPTH_WIDTH)
        if tuple(depth.shape) == no_channel_shape:
            canonical_depth = depth
        elif tuple(depth.shape) == singleton_channel_shape:
            canonical_depth = depth.squeeze(-3)
        else:
            raise ValueError(
                "depth must have shape "
                f"{no_channel_shape} or {singleton_channel_shape}, got {tuple(depth.shape)}"
            )
        return leading_shape, canonical_depth

    def _encode(
        self, obs_history: torch.Tensor, depth: torch.Tensor, leading_shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(torch.tensor(leading_shape).prod().item())
        state_token = self.state_encoder(obs_history.reshape(batch_size, -1)).reshape(
            *leading_shape, 1, _TOKEN_DIM
        )
        depth_features = self.depth_encoder(depth.reshape(batch_size, _DEPTH_FRAMES, _DEPTH_HEIGHT, _DEPTH_WIDTH))
        pooled_features = self.depth_pool(depth_features)
        visual_tokens = self.visual_token_encoder(
            pooled_features.permute(0, 2, 3, 1).reshape(batch_size, _VISUAL_TOKEN_COUNT, 128)
        ).reshape(*leading_shape, _VISUAL_TOKEN_COUNT, _TOKEN_DIM)
        tokens = torch.cat((state_token, visual_tokens), dim=-2)
        transformer_tokens = self.transformer(tokens.reshape(batch_size, 11, _TOKEN_DIM)).reshape(
            *leading_shape, 11, _TOKEN_DIM
        )
        fused_features = self.fusion(transformer_tokens.reshape(batch_size, -1)).reshape(
            *leading_shape, _GRU_HIDDEN_DIM
        )
        return state_token, visual_tokens, transformer_tokens, fused_features

    def _run_gru(
        self,
        fused_features: torch.Tensor,
        leading_shape: tuple[int, ...],
        hidden_state: torch.Tensor | None,
        dones: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(leading_shape) == 1:
            batch_size = leading_shape[0]
            done_mask = self._optional_mask(dones, (batch_size,), "dones", fused_features.device, default=False)
            valid_mask = self._optional_mask(masks, (batch_size,), "masks", fused_features.device, default=True)
            hidden_state = self._prepare_hidden_state(hidden_state, batch_size, fused_features)
            hidden_state = self.reset_hidden_state(hidden_state, done_mask & valid_mask)
            recurrent, candidate_hidden = self.gru(fused_features.unsqueeze(0), hidden_state)
            next_hidden = torch.where(valid_mask[None, :, None], candidate_hidden, hidden_state)
            return recurrent.squeeze(0), next_hidden, valid_mask

        steps, batch_size = leading_shape
        done_mask = self._optional_mask(dones, (steps, batch_size), "dones", fused_features.device, default=False)
        valid_mask = self._optional_mask(masks, (steps, batch_size), "masks", fused_features.device, default=True)
        hidden = self._prepare_hidden_state(hidden_state, batch_size, fused_features)
        recurrent_steps = []
        for step in range(steps):
            active = valid_mask[step]
            hidden = self.reset_hidden_state(hidden, done_mask[step] & active)
            recurrent, candidate_hidden = self.gru(fused_features[step].unsqueeze(0), hidden)
            hidden = torch.where(active[None, :, None], candidate_hidden, hidden)
            recurrent_steps.append(recurrent.squeeze(0))
        return torch.stack(recurrent_steps), hidden, valid_mask

    def _prepare_hidden_state(
        self, hidden_state: torch.Tensor | None, batch_size: int, features: torch.Tensor
    ) -> torch.Tensor:
        if hidden_state is None:
            return torch.zeros((1, batch_size, _GRU_HIDDEN_DIM), dtype=features.dtype, device=features.device)
        expected_shape = (1, batch_size, _GRU_HIDDEN_DIM)
        if tuple(hidden_state.shape) != expected_shape:
            raise ValueError(f"hidden_state must have shape {expected_shape}, got {tuple(hidden_state.shape)}")
        return hidden_state

    @staticmethod
    def _optional_mask(
        value: torch.Tensor | None,
        shape: tuple[int, ...],
        name: str,
        device: torch.device,
        default: bool,
    ) -> torch.Tensor:
        if value is None:
            return torch.full(shape, default, dtype=torch.bool, device=device)
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        return value.to(device=device, dtype=torch.bool)

    @staticmethod
    def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        while valid.ndim < prediction.ndim:
            valid = valid.unsqueeze(-1)
        return ((prediction - target).square() * valid).sum() / valid.expand_as(prediction).sum().clamp_min(1.0)

    @staticmethod
    def _masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return (value * valid).sum() / valid.sum().clamp_min(1.0)


__all__ = [
    "LadderSSRVisualActor",
    "LadderVisualLossWeights",
    "LadderVisualOutput",
    "LadderVisualTargets",
]
