"""Transformer-GRU visual estimators used by visual pretraining and PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ladder.utils.limb_heightmaps import RUNG


_HISTORY_FRAMES = 10
_DEPTH_FRAMES = 2
_DEPTH_HEIGHT = 60
_DEPTH_WIDTH = 80
_TOKEN_DIM = 64
_VISUAL_TOKEN_COUNT = 10
_HIDDEN_DIM = 256
_MAP_SHAPE = (10, 3)
_EXPLICIT_VELOCITY_LIMIT = 10.0
_DYNAMIC_DIM = 16
_MOTION_DIM = 16
_MOTION_TARGET_DIM = 31

# The 85-D proprioceptive frame remains the encoder input.  Only these first
# 59 physical-state values are reconstructed at t+1; policy action and climb
# command are deliberately excluded from the dynamic supervision target.
FUTURE_PHYSICAL_STATE_FIELDS = (
    ("root_angular_velocity", slice(0, 3)),
    ("root_orientation_6d", slice(3, 9)),
    ("joint_position", slice(9, 34)),
    ("joint_velocity", slice(34, 59)),
)
FUTURE_PHYSICAL_STATE_DIM = 59
TRANSFORMER_VISION_DATA_CONTRACT = (
    "cr1_field_normalized_proprio_rung_weighted_dynamic16_deterministic_future_physical59_v3"
)
VARIATIONAL_TRANSFORMER_VISION_DATA_CONTRACT = (
    "cr1_field_normalized_proprio_rung_weighted_dynamic16_future_elu_v2"
)
TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT = (
    "cr1_field_normalized_proprio_rung_weighted_task_goal12_contact4_"
    "next_stable_motion_joint25_root_orientation6_v2"
)
LEGACY_TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT = (
    "cr1_field_normalized_proprio_rung_weighted_task_goal12_contact4_v1"
)


@dataclass
class TransformerVisionOutput:
    """Deployable estimates and training-only reconstruction outputs."""

    velocity: torch.Tensor
    hand_left_map: torch.Tensor
    hand_right_map: torch.Tensor
    foot_left_map: torch.Tensor
    foot_right_map: torch.Tensor
    next_proprio: torch.Tensor | None
    limb_goal_b: torch.Tensor | None
    contact_logits: torch.Tensor | None
    contact_probability: torch.Tensor | None
    motion_latent: torch.Tensor | None
    motion_joint_pos: torch.Tensor | None
    motion_root_orientation_6d: torch.Tensor | None
    dynamic_latent: torch.Tensor
    dynamic_mu: torch.Tensor | None
    dynamic_logvar: torch.Tensor | None
    latent: torch.Tensor
    valid_mask: torch.Tensor
    next_hidden_state: torch.Tensor


@dataclass
class TransformerVisionTargets:
    """Simulator-only supervision that never enters the frozen actor."""

    velocity: torch.Tensor
    limb_heightmaps: torch.Tensor
    heightmap_valid: torch.Tensor
    heightmap_hit_class: torch.Tensor
    next_proprio: torch.Tensor
    future_valid: torch.Tensor | None = None
    limb_goal_b: torch.Tensor | None = None
    limb_goal_valid: torch.Tensor | None = None
    contact_indicator: torch.Tensor | None = None
    motion_joint_pos: torch.Tensor | None = None
    motion_root_orientation_6d: torch.Tensor | None = None
    motion_valid: torch.Tensor | None = None


@dataclass(frozen=True)
class TransformerVisionLossWeights:
    velocity: float = 2.0
    hand: float = 2.0
    foot: float = 2.0
    future: float = 5.0
    kl: float = 1.0


@dataclass(frozen=True)
class TaskGoalTransformerVisionLossWeights:
    velocity: float = 2.0
    hand: float = 2.0
    foot: float = 2.0
    limb_goal: float = 1.0
    contact: float = 1.0
    motion_joint: float = 1.0
    motion_orientation: float = 1.0


def _elu_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class TransformerVisionEncoder(nn.Module):
    """The deterministic 11-token Transformer and GRU visual estimator.

    The 16-D dynamic latent is a direct GRU projection.  Its decoder predicts
    only the 59-D physical-state portion of the next 85-D proprioceptive frame.
    There is no sampling distribution and no KL loss in this active model.
    """

    data_contract = TRANSFORMER_VISION_DATA_CONTRACT
    future_target_dim = FUTURE_PHYSICAL_STATE_DIM

    def __init__(
        self,
        proprio_dim: int = 85,
        history_frames: int = _HISTORY_FRAMES,
        include_dynamic: bool = True,
    ) -> None:
        super().__init__()
        if history_frames <= 0:
            raise ValueError("history_frames must be positive")
        self.proprio_dim = proprio_dim
        self.history_frames = history_frames
        self.include_dynamic = include_dynamic
        self.actor_latent_dim = 51 if include_dynamic else 35
        self.state_encoder = _elu_mlp(proprio_dim * history_frames, (256, 128), _TOKEN_DIM)
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
        self.fusion = nn.Sequential(nn.Linear(11 * _TOKEN_DIM, _HIDDEN_DIM), nn.ELU())
        self.gru = nn.GRU(_HIDDEN_DIM, _HIDDEN_DIM, num_layers=1)

        self.velocity_head = nn.Linear(_HIDDEN_DIM, 3)
        self.hand_left_head = nn.Linear(_HIDDEN_DIM, 8)
        self.hand_right_head = nn.Linear(_HIDDEN_DIM, 8)
        self.foot_left_head = nn.Linear(_HIDDEN_DIM, 8)
        self.foot_right_head = nn.Linear(_HIDDEN_DIM, 8)
        self.dynamic_head = nn.Linear(_HIDDEN_DIM, _DYNAMIC_DIM)

        map_size = _MAP_SHAPE[0] * _MAP_SHAPE[1]
        self.hand_left_decoder = _elu_mlp(8, (64, 128), map_size)
        self.hand_right_decoder = _elu_mlp(8, (64, 128), map_size)
        self.foot_left_decoder = _elu_mlp(8, (64, 128), map_size)
        self.foot_right_decoder = _elu_mlp(8, (64, 128), map_size)
        self.future_decoder = _elu_mlp(
            _DYNAMIC_DIM, (64, 128), self.future_target_dim
        )
        self.future_decoder_activation = nn.ELU()
        self.future_decoder_final = nn.Linear(
            self.future_target_dim, self.future_target_dim
        )

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
    ) -> TransformerVisionOutput:
        leading_shape, depth_history = self._validate_inputs(proprio_history, depth_history)
        state_token, visual_tokens = self._tokens(proprio_history, depth_history, leading_shape)
        batch_size = int(torch.tensor(leading_shape).prod().item())
        tokens = torch.cat((state_token, visual_tokens), dim=-2)
        transformed = self.transformer(tokens.reshape(batch_size, 11, _TOKEN_DIM))
        fused = self.fusion(transformed.reshape(batch_size, -1)).reshape(*leading_shape, _HIDDEN_DIM)
        recurrent, next_hidden_state, valid_mask = self._run_gru(
            fused, leading_shape, hidden_state, dones, masks
        )

        # Match the humanoid visual estimator: explicit estimates entering the
        # actor are bounded independently from the reconstruction-only latents.
        velocity = torch.clamp(
            self.velocity_head(recurrent),
            min=-_EXPLICIT_VELOCITY_LIMIT,
            max=_EXPLICIT_VELOCITY_LIMIT,
        )
        hand_left = self.hand_left_head(recurrent)
        hand_right = self.hand_right_head(recurrent)
        foot_left = self.foot_left_head(recurrent)
        foot_right = self.foot_right_head(recurrent)
        if self.include_dynamic:
            (
                dynamic,
                dynamic_mu,
                dynamic_logvar,
                limb_goal_b,
                contact_logits,
                contact_probability,
            ) = self._dynamic_features(
                recurrent,
                deterministic=deterministic,
                dynamic_noise=dynamic_noise,
            )
        else:
            dynamic = recurrent.new_empty(*recurrent.shape[:-1], 0)
            dynamic_mu = None
            dynamic_logvar = None
            limb_goal_b = None
            contact_logits = None
            contact_probability = None
        (
            motion_latent,
            motion_joint_pos,
            motion_root_orientation_6d,
        ) = self._motion_features(recurrent)
        latent_parts = [velocity, hand_left, hand_right, foot_left, foot_right]
        if self.include_dynamic:
            latent_parts.append(dynamic)
        if motion_latent is not None:
            latent_parts.append(motion_latent)
        latent = torch.cat(latent_parts, dim=-1)
        next_proprio = None
        if self.include_dynamic and limb_goal_b is None:
            next_proprio = self.future_decoder_final(
                self.future_decoder_activation(self.future_decoder(dynamic))
            )
        return TransformerVisionOutput(
            velocity=velocity,
            hand_left_map=self.hand_left_decoder(hand_left).reshape(*leading_shape, *_MAP_SHAPE),
            hand_right_map=self.hand_right_decoder(hand_right).reshape(*leading_shape, *_MAP_SHAPE),
            foot_left_map=self.foot_left_decoder(foot_left).reshape(*leading_shape, *_MAP_SHAPE),
            foot_right_map=self.foot_right_decoder(foot_right).reshape(*leading_shape, *_MAP_SHAPE),
            next_proprio=next_proprio,
            limb_goal_b=limb_goal_b,
            contact_logits=contact_logits,
            contact_probability=contact_probability,
            motion_latent=motion_latent,
            motion_joint_pos=motion_joint_pos,
            motion_root_orientation_6d=motion_root_orientation_6d,
            dynamic_latent=dynamic,
            dynamic_mu=dynamic_mu,
            dynamic_logvar=dynamic_logvar,
            latent=latent,
            valid_mask=valid_mask,
            next_hidden_state=next_hidden_state,
        )

    def losses(
        self,
        output: TransformerVisionOutput,
        targets: TransformerVisionTargets,
        weights: TransformerVisionLossWeights = TransformerVisionLossWeights(),
    ) -> dict[str, torch.Tensor]:
        result, valid = self._common_losses(output, targets)
        if output.next_proprio is None:
            raise RuntimeError("Future-state encoder output is missing next_proprio")
        future_target = self.select_future_target(targets.next_proprio)
        if future_target.shape != output.next_proprio.shape:
            raise ValueError("future proprio target shape does not match the prediction")
        future_valid = valid if targets.future_valid is None else valid * targets.future_valid.to(valid.dtype)
        future = self.masked_mse(output.next_proprio, future_target, future_valid)
        result.update(
            {
                "future": future,
                "dynamic_abs_max": output.dynamic_latent.detach().float().abs().amax(),
                "total": (
                    weights.velocity * result["velocity"]
                    + weights.hand * result["hand"]
                    + weights.foot * result["foot"]
                    + weights.future * future
                ),
            }
        )
        for name, field_slice in FUTURE_PHYSICAL_STATE_FIELDS:
            result[f"future_{name}"] = self.masked_mse(
                output.next_proprio[..., field_slice],
                future_target[..., field_slice],
                future_valid,
            )
        return result

    def _common_losses(
        self,
        output: TransformerVisionOutput,
        targets: TransformerVisionTargets,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        expected_leading = output.velocity.shape[:-1]
        expected_maps = (*expected_leading, 4, *_MAP_SHAPE)
        if targets.velocity.shape != output.velocity.shape:
            raise ValueError("velocity target shape does not match the velocity estimate")
        if (
            targets.limb_heightmaps.shape != expected_maps
            or targets.heightmap_valid.shape != expected_maps
            or targets.heightmap_hit_class.shape != expected_maps
        ):
            raise ValueError(f"height-map targets must have shape {expected_maps}")

        valid = output.valid_mask.to(dtype=output.velocity.dtype)
        map_valid = targets.heightmap_valid.to(dtype=output.velocity.dtype) * valid[..., None, None, None]
        rung_mask = targets.heightmap_hit_class == RUNG
        map_weight = torch.where(rung_mask, 2.0, 1.0).to(dtype=output.velocity.dtype) * map_valid
        map_predictions = (output.hand_left_map, output.hand_right_map, output.foot_left_map, output.foot_right_map)
        map_losses = tuple(
            self.masked_mse(prediction, targets.limb_heightmaps[..., index, :, :], map_weight[..., index, :, :])
            for index, prediction in enumerate(map_predictions)
        )
        velocity = self.masked_mse(output.velocity, targets.velocity, valid)
        velocity_zero = self.masked_mse(torch.zeros_like(targets.velocity), targets.velocity, valid)
        velocity_axis = tuple(
            self.masked_mse(output.velocity[..., index], targets.velocity[..., index], valid)
            for index in range(3)
        )
        velocity_zero_axis = tuple(
            self.masked_mse(torch.zeros_like(targets.velocity[..., index]), targets.velocity[..., index], valid)
            for index in range(3)
        )
        velocity_skill_axis = tuple(
            self.relative_skill(model_error, zero_error)
            for model_error, zero_error in zip(velocity_axis, velocity_zero_axis, strict=True)
        )
        hand = 0.5 * (map_losses[0] + map_losses[1])
        foot = 0.5 * (map_losses[2] + map_losses[3])
        stacked_predictions = torch.stack(map_predictions, dim=-3)
        result = {
            "velocity": velocity,
            "velocity_zero_baseline": velocity_zero,
            "velocity_skill_score": self.relative_skill(velocity, velocity_zero),
            "velocity_x": velocity_axis[0],
            "velocity_y": velocity_axis[1],
            "velocity_z": velocity_axis[2],
            "velocity_zero_x": velocity_zero_axis[0],
            "velocity_zero_y": velocity_zero_axis[1],
            "velocity_zero_z": velocity_zero_axis[2],
            "velocity_skill_x": velocity_skill_axis[0],
            "velocity_skill_y": velocity_skill_axis[1],
            "velocity_skill_z": velocity_skill_axis[2],
            "hand_left": map_losses[0],
            "hand_right": map_losses[1],
            "foot_left": map_losses[2],
            "foot_right": map_losses[3],
            "hand": hand,
            "foot": foot,
            "heightmap_rung": self.masked_mse(
                stacked_predictions,
                targets.limb_heightmaps,
                map_valid * rung_mask.to(dtype=map_valid.dtype),
            ),
            "heightmap_non_rung": self.masked_mse(
                stacked_predictions,
                targets.limb_heightmaps,
                map_valid * (~rung_mask).to(dtype=map_valid.dtype),
            ),
            "hand_rung": self.masked_mse(
                stacked_predictions[..., :2, :, :],
                targets.limb_heightmaps[..., :2, :, :],
                map_valid[..., :2, :, :] * rung_mask[..., :2, :, :].to(dtype=map_valid.dtype),
            ),
            "hand_non_rung": self.masked_mse(
                stacked_predictions[..., :2, :, :],
                targets.limb_heightmaps[..., :2, :, :],
                map_valid[..., :2, :, :] * (~rung_mask[..., :2, :, :]).to(dtype=map_valid.dtype),
            ),
            "foot_rung": self.masked_mse(
                stacked_predictions[..., 2:, :, :],
                targets.limb_heightmaps[..., 2:, :, :],
                map_valid[..., 2:, :, :] * rung_mask[..., 2:, :, :].to(dtype=map_valid.dtype),
            ),
            "foot_non_rung": self.masked_mse(
                stacked_predictions[..., 2:, :, :],
                targets.limb_heightmaps[..., 2:, :, :],
                map_valid[..., 2:, :, :] * (~rung_mask[..., 2:, :, :]).to(dtype=map_valid.dtype),
            ),
        }
        return result, valid

    def _dynamic_features(
        self,
        recurrent: torch.Tensor,
        *,
        deterministic: bool,
        dynamic_noise: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
    ]:
        del deterministic
        if dynamic_noise is not None:
            raise ValueError(
                "dynamic_noise is only supported by VariationalTransformerVisionEncoder"
            )
        return self.dynamic_head(recurrent), None, None, None, None, None

    def _motion_features(
        self, recurrent: torch.Tensor
    ) -> tuple[None, None, None]:
        del recurrent
        return None, None, None

    def select_future_target(self, next_proprio: torch.Tensor) -> torch.Tensor:
        """Select the physical-state fields supervised by the active decoder."""

        if next_proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"next_proprio must end in {self.proprio_dim} values, got {tuple(next_proprio.shape)}"
            )
        if self.proprio_dim != 85:
            raise ValueError(
                "The CR1 physical-state target layout is defined only for the 85-D proprioceptive frame"
            )
        return next_proprio[..., :FUTURE_PHYSICAL_STATE_DIM]

    @staticmethod
    def masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        while valid.ndim < prediction.ndim:
            valid = valid.unsqueeze(-1)
        valid = valid.to(dtype=prediction.dtype)
        return ((prediction - target).square() * valid).sum() / valid.expand_as(prediction).sum().clamp_min(1.0)

    @staticmethod
    def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid = valid.to(dtype=value.dtype)
        return (value * valid).sum() / valid.sum().clamp_min(1.0)

    @staticmethod
    def relative_skill(model_error: torch.Tensor, zero_baseline_error: torch.Tensor) -> torch.Tensor:
        """Return 1 - model MSE / zero-prediction MSE without dividing by zero."""

        return 1.0 - model_error / zero_baseline_error.clamp_min(1.0e-12)

    def _tokens(
        self, proprio_history: torch.Tensor, depth_history: torch.Tensor, leading_shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(torch.tensor(leading_shape).prod().item())
        state = self.state_encoder(proprio_history.reshape(batch_size, -1)).reshape(*leading_shape, 1, _TOKEN_DIM)
        depth = self.depth_encoder(depth_history.reshape(batch_size, _DEPTH_FRAMES, _DEPTH_HEIGHT, _DEPTH_WIDTH))
        visual = self.visual_token_encoder(
            self.depth_pool(depth).permute(0, 2, 3, 1).reshape(batch_size, _VISUAL_TOKEN_COUNT, 128)
        ).reshape(*leading_shape, _VISUAL_TOKEN_COUNT, _TOKEN_DIM)
        return state, visual

    def _validate_inputs(
        self, proprio_history: torch.Tensor, depth_history: torch.Tensor
    ) -> tuple[tuple[int, ...], torch.Tensor]:
        if proprio_history.ndim not in (3, 4) or tuple(proprio_history.shape[-2:]) != (
            self.history_frames,
            self.proprio_dim,
        ):
            raise ValueError(
                "proprio_history must have shape "
                f"[B,{self.history_frames},{self.proprio_dim}] or "
                f"[T,B,{self.history_frames},{self.proprio_dim}]"
            )
        leading_shape = tuple(proprio_history.shape[:-2])
        expected = (*leading_shape, _DEPTH_FRAMES, _DEPTH_HEIGHT, _DEPTH_WIDTH)
        singleton_channel = (*leading_shape, _DEPTH_FRAMES, 1, _DEPTH_HEIGHT, _DEPTH_WIDTH)
        if tuple(depth_history.shape) == singleton_channel:
            depth_history = depth_history.squeeze(-3)
        elif tuple(depth_history.shape) != expected:
            raise ValueError(f"depth_history must have shape {expected} or {singleton_channel}")
        return leading_shape, depth_history

    def _run_gru(
        self,
        fused: torch.Tensor,
        leading_shape: tuple[int, ...],
        hidden_state: torch.Tensor | None,
        dones: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(leading_shape) == 1:
            batch_size = leading_shape[0]
            valid = self._optional_mask(masks, (batch_size,), fused.device, True)
            done = self._optional_mask(dones, (batch_size,), fused.device, False)
            hidden = self._prepare_hidden(hidden_state, batch_size, fused)
            hidden = self.reset_hidden_state(hidden, done & valid)
            recurrent, candidate = self.gru(fused.unsqueeze(0), hidden)
            return recurrent.squeeze(0), torch.where(valid[None, :, None], candidate, hidden), valid

        steps, batch_size = leading_shape
        valid = self._optional_mask(masks, (steps, batch_size), fused.device, True)
        done = self._optional_mask(dones, (steps, batch_size), fused.device, False)
        hidden = self._prepare_hidden(hidden_state, batch_size, fused)
        outputs = []
        for step in range(steps):
            active = valid[step]
            hidden = self.reset_hidden_state(hidden, done[step] & active)
            recurrent, candidate = self.gru(fused[step].unsqueeze(0), hidden)
            hidden = torch.where(active[None, :, None], candidate, hidden)
            outputs.append(recurrent.squeeze(0))
        return torch.stack(outputs), hidden, valid

    @staticmethod
    def reset_hidden_state(hidden_state: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        if tuple(hidden_state.shape[:1]) != (1,) or hidden_state.shape[-1] != _HIDDEN_DIM:
            raise ValueError(f"hidden_state must have shape [1,B,{_HIDDEN_DIM}]")
        return hidden_state * (~dones.to(dtype=torch.bool))[None, :, None].to(hidden_state.dtype)

    @staticmethod
    def _optional_mask(value: torch.Tensor | None, shape: tuple[int, ...], device: torch.device, default: bool) -> torch.Tensor:
        if value is None:
            return torch.full(shape, default, dtype=torch.bool, device=device)
        if tuple(value.shape) != shape:
            raise ValueError(f"mask must have shape {shape}, got {tuple(value.shape)}")
        return value.to(device=device, dtype=torch.bool)

    @staticmethod
    def _prepare_hidden(hidden: torch.Tensor | None, batch_size: int, features: torch.Tensor) -> torch.Tensor:
        if hidden is None:
            return torch.zeros((1, batch_size, _HIDDEN_DIM), device=features.device, dtype=features.dtype)
        if tuple(hidden.shape) != (1, batch_size, _HIDDEN_DIM):
            raise ValueError(f"hidden_state must have shape {(1, batch_size, _HIDDEN_DIM)}")
        return hidden


class TaskGoalTransformerVisionEncoder(TransformerVisionEncoder):
    """TransformerVision encoder supervised by deployable limb task semantics.

    The 16-D task head is an ELU MLP. Its first 12 outputs are four 3-D
    current/next limb goals in the robot base frame. Its final four outputs are
    contact logits; sigmoid probabilities, rather than logits, enter the actor.
    A separate 16-D motion MLP and 31-D decoder learn the observable,
    state-locked next stable whole-body pose. Both compact latents enter the
    actor, while decoded targets remain training-only.
    """

    data_contract = TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT
    limb_goal_huber_delta = 0.05
    motion_joint_huber_delta = 0.05
    motion_orientation_huber_delta = 0.05

    def __init__(self, proprio_dim: int = 85) -> None:
        super().__init__(proprio_dim=proprio_dim)
        del self.dynamic_head
        del self.future_decoder
        del self.future_decoder_activation
        del self.future_decoder_final
        self.task_head = _elu_mlp(_HIDDEN_DIM, (128, 64), _DYNAMIC_DIM)
        self.motion_head = _elu_mlp(_HIDDEN_DIM, (128, 64), _MOTION_DIM)
        self.motion_decoder = _elu_mlp(
            _MOTION_DIM, (64, 128), _MOTION_TARGET_DIM
        )
        self.actor_latent_dim += _MOTION_DIM

    def _dynamic_features(
        self,
        recurrent: torch.Tensor,
        *,
        deterministic: bool,
        dynamic_noise: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del deterministic
        if dynamic_noise is not None:
            raise ValueError("dynamic_noise is not supported by TaskGoalTransformerVisionEncoder")
        task_prediction = self.task_head(recurrent)
        limb_goal_b = task_prediction[..., :12].reshape(*recurrent.shape[:-1], 4, 3)
        contact_logits = task_prediction[..., 12:]
        contact_probability = torch.sigmoid(contact_logits)
        actor_task_latent = torch.cat(
            (limb_goal_b.flatten(start_dim=-2), contact_probability), dim=-1
        )
        return (
            actor_task_latent,
            None,
            None,
            limb_goal_b,
            contact_logits,
            contact_probability,
        )

    def _motion_features(
        self, recurrent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_latent = self.motion_head(recurrent)
        decoded = self.motion_decoder(motion_latent)
        return motion_latent, decoded[..., :25], decoded[..., 25:]

    def losses(
        self,
        output: TransformerVisionOutput,
        targets: TransformerVisionTargets,
        weights: TaskGoalTransformerVisionLossWeights = TaskGoalTransformerVisionLossWeights(),
    ) -> dict[str, torch.Tensor]:
        result, valid = self._common_losses(output, targets)
        if (
            output.limb_goal_b is None
            or output.contact_logits is None
            or output.contact_probability is None
        ):
            raise RuntimeError("Task-goal encoder output is missing task predictions")
        if (
            targets.limb_goal_b is None
            or targets.limb_goal_valid is None
            or targets.contact_indicator is None
            or targets.motion_joint_pos is None
            or targets.motion_root_orientation_6d is None
            or targets.motion_valid is None
        ):
            raise RuntimeError(
                "Task-goal encoder requires limb-goal, contact, and stable-motion targets"
            )
        if (
            output.motion_latent is None
            or output.motion_joint_pos is None
            or output.motion_root_orientation_6d is None
        ):
            raise RuntimeError("Task-goal encoder output is missing stable-motion predictions")

        expected_leading = output.velocity.shape[:-1]
        expected_goal = (*expected_leading, 4, 3)
        expected_limb = (*expected_leading, 4)
        if targets.limb_goal_b.shape != expected_goal:
            raise ValueError(f"limb_goal_b target must have shape {expected_goal}")
        if targets.limb_goal_valid.shape != expected_limb:
            raise ValueError(f"limb_goal_valid target must have shape {expected_limb}")
        if targets.contact_indicator.shape != expected_limb:
            raise ValueError(f"contact_indicator target must have shape {expected_limb}")
        expected_motion_joint = (*expected_leading, 25)
        expected_motion_orientation = (*expected_leading, 6)
        if targets.motion_joint_pos.shape != expected_motion_joint:
            raise ValueError(
                f"motion_joint_pos target must have shape {expected_motion_joint}"
            )
        if targets.motion_root_orientation_6d.shape != expected_motion_orientation:
            raise ValueError(
                "motion_root_orientation_6d target must have shape "
                f"{expected_motion_orientation}"
            )
        if targets.motion_valid.shape != expected_leading:
            raise ValueError(f"motion_valid target must have shape {expected_leading}")

        goal_valid = targets.limb_goal_valid.to(dtype=valid.dtype) * valid[..., None]
        contact_valid = valid[..., None].expand_as(targets.contact_indicator)
        goal_losses = tuple(
            self.masked_huber(
                output.limb_goal_b[..., limb_index, :],
                targets.limb_goal_b[..., limb_index, :],
                goal_valid[..., limb_index],
                beta=self.limb_goal_huber_delta,
            )
            for limb_index in range(4)
        )
        limb_goal = sum(goal_losses) / len(goal_losses)
        contact = self.masked_bce_with_logits(
            output.contact_logits,
            targets.contact_indicator,
            contact_valid,
        )
        contact_correct = (
            (output.contact_probability >= 0.5)
            == (targets.contact_indicator >= 0.5)
        ).to(dtype=valid.dtype)
        contact_accuracy = self.masked_mean(contact_correct, contact_valid)
        motion_valid = targets.motion_valid.to(dtype=valid.dtype) * valid
        motion_joint = self.masked_huber(
            output.motion_joint_pos,
            targets.motion_joint_pos,
            motion_valid,
            beta=self.motion_joint_huber_delta,
        )
        motion_orientation = self.masked_huber(
            output.motion_root_orientation_6d,
            targets.motion_root_orientation_6d,
            motion_valid,
            beta=self.motion_orientation_huber_delta,
        )
        result.update(
            {
                "limb_goal_left_hand": goal_losses[0],
                "limb_goal_right_hand": goal_losses[1],
                "limb_goal_left_foot": goal_losses[2],
                "limb_goal_right_foot": goal_losses[3],
                "limb_goal": limb_goal,
                "limb_goal_valid_fraction": goal_valid.mean(),
                "contact_bce": contact,
                "contact_accuracy": contact_accuracy,
                "motion_joint": motion_joint,
                "motion_root_orientation": motion_orientation,
                "motion_target_valid_fraction": motion_valid.mean(),
                "motion_latent_abs_max": output.motion_latent.detach().float().abs().amax(),
                "dynamic_abs_max": output.dynamic_latent.detach().float().abs().amax(),
                "total": (
                    weights.velocity * result["velocity"]
                    + weights.hand * result["hand"]
                    + weights.foot * result["foot"]
                    + weights.limb_goal * limb_goal
                    + weights.contact * contact
                    + weights.motion_joint * motion_joint
                    + weights.motion_orientation * motion_orientation
                ),
            }
        )
        return result

    @staticmethod
    def masked_huber(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        *,
        beta: float,
    ) -> torch.Tensor:
        loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
        while valid.ndim < loss.ndim:
            valid = valid.unsqueeze(-1)
        valid = valid.to(dtype=loss.dtype)
        return (loss * valid).sum() / valid.expand_as(loss).sum().clamp_min(1.0)

    @staticmethod
    def masked_bce_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, target.to(dtype=logits.dtype), reduction="none"
        )
        valid = valid.to(dtype=loss.dtype)
        return (loss * valid).sum() / valid.sum().clamp_min(1.0)


class LegacyTaskGoalTransformerVisionEncoder(TaskGoalTransformerVisionEncoder):
    """Exact TaskGoal v1 topology and losses retained for resume/playback."""

    data_contract = LEGACY_TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT

    def __init__(self, proprio_dim: int = 85) -> None:
        super().__init__(proprio_dim=proprio_dim)
        del self.motion_head
        del self.motion_decoder
        self.actor_latent_dim -= _MOTION_DIM

    def _motion_features(
        self, recurrent: torch.Tensor
    ) -> tuple[None, None, None]:
        del recurrent
        return None, None, None

    def losses(
        self,
        output: TransformerVisionOutput,
        targets: TransformerVisionTargets,
        weights: TaskGoalTransformerVisionLossWeights = TaskGoalTransformerVisionLossWeights(),
    ) -> dict[str, torch.Tensor]:
        result, valid = self._common_losses(output, targets)
        if (
            output.limb_goal_b is None
            or output.contact_logits is None
            or output.contact_probability is None
        ):
            raise RuntimeError("Legacy task-goal encoder output is missing task predictions")
        if (
            targets.limb_goal_b is None
            or targets.limb_goal_valid is None
            or targets.contact_indicator is None
        ):
            raise RuntimeError(
                "Legacy task-goal encoder requires limb-goal and contact targets"
            )

        expected_leading = output.velocity.shape[:-1]
        expected_goal = (*expected_leading, 4, 3)
        expected_limb = (*expected_leading, 4)
        if targets.limb_goal_b.shape != expected_goal:
            raise ValueError(f"limb_goal_b target must have shape {expected_goal}")
        if targets.limb_goal_valid.shape != expected_limb:
            raise ValueError(f"limb_goal_valid target must have shape {expected_limb}")
        if targets.contact_indicator.shape != expected_limb:
            raise ValueError(f"contact_indicator target must have shape {expected_limb}")

        goal_valid = targets.limb_goal_valid.to(dtype=valid.dtype) * valid[..., None]
        contact_valid = valid[..., None].expand_as(targets.contact_indicator)
        goal_losses = tuple(
            self.masked_huber(
                output.limb_goal_b[..., limb_index, :],
                targets.limb_goal_b[..., limb_index, :],
                goal_valid[..., limb_index],
                beta=self.limb_goal_huber_delta,
            )
            for limb_index in range(4)
        )
        limb_goal = sum(goal_losses) / len(goal_losses)
        contact = self.masked_bce_with_logits(
            output.contact_logits,
            targets.contact_indicator,
            contact_valid,
        )
        contact_correct = (
            (output.contact_probability >= 0.5)
            == (targets.contact_indicator >= 0.5)
        ).to(dtype=valid.dtype)
        result.update(
            {
                "limb_goal_left_hand": goal_losses[0],
                "limb_goal_right_hand": goal_losses[1],
                "limb_goal_left_foot": goal_losses[2],
                "limb_goal_right_foot": goal_losses[3],
                "limb_goal": limb_goal,
                "limb_goal_valid_fraction": goal_valid.mean(),
                "contact_bce": contact,
                "contact_accuracy": self.masked_mean(contact_correct, contact_valid),
                "dynamic_abs_max": output.dynamic_latent.detach().float().abs().amax(),
                "total": (
                    weights.velocity * result["velocity"]
                    + weights.hand * result["hand"]
                    + weights.foot * result["foot"]
                    + weights.limb_goal * limb_goal
                    + weights.contact * contact
                ),
            }
        )
        return result


class VariationalTransformerVisionEncoder(TransformerVisionEncoder):
    """Retained VAE estimator used by the accepted v2 checkpoints.

    This class preserves the original 16-D ``mu/logvar`` heads, stochastic
    reparameterization, full 85-D future decoder, and KL loss.  New training
    uses :class:`TransformerVisionEncoder`; this class exists for exact legacy
    checkpoint inspection and controlled comparisons.
    """

    data_contract = VARIATIONAL_TRANSFORMER_VISION_DATA_CONTRACT
    future_target_dim = 85

    def __init__(self, proprio_dim: int = 85) -> None:
        super().__init__(proprio_dim=proprio_dim)
        del self.dynamic_head
        self.dynamic_mu_head = nn.Linear(_HIDDEN_DIM, _DYNAMIC_DIM)
        self.dynamic_logvar_head = nn.Linear(_HIDDEN_DIM, _DYNAMIC_DIM)
        self.future_decoder = _elu_mlp(
            _DYNAMIC_DIM, (64, 128), self.future_target_dim
        )
        self.future_decoder_final = nn.Linear(
            self.future_target_dim, self.future_target_dim
        )

    def _dynamic_features(
        self,
        recurrent: torch.Tensor,
        *,
        deterministic: bool,
        dynamic_noise: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
    ]:
        dynamic_mu = self.dynamic_mu_head(recurrent)
        dynamic_logvar = self.dynamic_logvar_head(recurrent)
        if deterministic:
            dynamic = dynamic_mu
        else:
            noise = torch.randn_like(dynamic_mu) if dynamic_noise is None else dynamic_noise
            if noise.shape != dynamic_mu.shape:
                raise ValueError(
                    f"dynamic_noise must have shape {tuple(dynamic_mu.shape)}, got {tuple(noise.shape)}"
                )
            dynamic = dynamic_mu + torch.exp(0.5 * dynamic_logvar) * noise
        return dynamic, dynamic_mu, dynamic_logvar, None, None, None

    def select_future_target(self, next_proprio: torch.Tensor) -> torch.Tensor:
        if next_proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"next_proprio must end in {self.proprio_dim} values, got {tuple(next_proprio.shape)}"
            )
        return next_proprio

    def losses(
        self,
        output: TransformerVisionOutput,
        targets: TransformerVisionTargets,
        weights: TransformerVisionLossWeights = TransformerVisionLossWeights(),
    ) -> dict[str, torch.Tensor]:
        losses = super().losses(output, targets, weights)
        if output.dynamic_mu is None or output.dynamic_logvar is None:
            raise RuntimeError("Variational output is missing mu/logvar")
        valid = output.valid_mask.to(dtype=output.velocity.dtype)
        kl = self.masked_mean(
            -0.5
            * torch.sum(
                1.0
                + output.dynamic_logvar
                - output.dynamic_mu.square()
                - output.dynamic_logvar.exp(),
                dim=-1,
            ),
            valid,
        )
        dynamic_standard_deviation = torch.exp(0.5 * output.dynamic_logvar.detach())
        losses.update(
            {
                "kl": kl,
                "dynamic_logvar_mean": output.dynamic_logvar.detach().mean(),
                "dynamic_logvar_min": output.dynamic_logvar.detach().amin(),
                "dynamic_logvar_max": output.dynamic_logvar.detach().amax(),
                "dynamic_std_max": dynamic_standard_deviation.amax(),
                "dynamic_mu_abs_max": output.dynamic_mu.detach().abs().amax(),
                "total": losses["total"] + weights.kl * kl,
            }
        )
        return losses
