"""Recurrent deployable student for visual action-MSE DAgger."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from .visual_model import DEPTH_KEY, PROPRIO_KEY


DEPTH_UPDATE_KEY = "depth_updated"
PRIVILEGED_KEY = "privileged_actor"
PROPRIO_HISTORY_FRAMES = 25
PROPRIO_DIM = 85
DEPTH_HEIGHT = 60
DEPTH_WIDTH = 80
# The current Warp depth contract is metric depth in [0.1, 2.0] m,
# centered at 1.0 m with unit scale.  Invalid/dropout pixels become -1.0.
DEPTH_INPUT_MIN = -1.0
DEPTH_INPUT_MAX = 1.0
RECURRENT_HIDDEN_DIM = 256
RECURRENT_VISUAL_DAGGER_DATA_CONTRACT = (
    "cr1_visual_recurrent_mse_dagger_proprio25x85_command_each_"
    "depth1x60x80_30hz_fresh_gru256_student_rollout_v1"
)


def _elu_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.extend((nn.Linear(previous_dim, output_dim), nn.ELU()))
    return nn.Sequential(*layers)


class CR1RecurrentDepthEncoder(nn.Module):
    """Depth CNN adapted from the recurrent X2 visual policy to 60x80 input."""

    output_dim = 128

    def __init__(self) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(nn.Linear(128 * 2 * 3, 128), nn.ELU())

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolutions(depth))


class CR1RecurrentVisualDaggerStudent(nn.Module):
    """25-frame command-aware proprioception plus fresh-frame visual GRU."""

    def __init__(
        self,
        action_dim: int,
        hidden_dims: list[int],
        init_noise_std: float,
        privileged_dim: int = 0,
    ) -> None:
        super().__init__()
        if not hidden_dims or hidden_dims[-1] != 256:
            raise ValueError("The final student policy hidden layer must be 256-D")
        self.action_dim = int(action_dim)
        self.privileged_dim = int(privileged_dim)
        if self.privileged_dim < 0:
            raise ValueError("privileged_dim must be non-negative")
        self.proprio_encoder = _elu_mlp(
            PROPRIO_HISTORY_FRAMES * PROPRIO_DIM, [512, 256], 128
        )
        self.depth_encoder = CR1RecurrentDepthEncoder()
        self.gru = nn.GRU(256, RECURRENT_HIDDEN_DIM, num_layers=1)
        self.recurrent_projection = nn.Sequential(
            nn.Linear(RECURRENT_HIDDEN_DIM, 256), nn.ELU()
        )
        layers: list[nn.Module] = []
        previous_dim = PROPRIO_DIM + 256 + self.privileged_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
            previous_dim = hidden_dim
        self.feature_mlp = nn.Sequential(*layers)
        self.action_head = nn.Linear(previous_dim, self.action_dim)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(init_noise_std)).log()
        )
        self._inference_hidden_state: torch.Tensor | None = None

    @property
    def output_std(self) -> torch.Tensor:
        return self.log_std.exp()

    def initial_hidden(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(
            1, batch_size, RECURRENT_HIDDEN_DIM, device=device, dtype=dtype
        )

    @staticmethod
    def reset_hidden_state(
        hidden_state: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        if hidden_state.ndim != 3 or tuple(hidden_state.shape[:1]) != (1,):
            raise ValueError("GRU hidden_state must have shape [1,B,H]")
        if tuple(dones.shape) != (hidden_state.shape[1],):
            raise ValueError(
                f"dones must have shape {(hidden_state.shape[1],)}, got {tuple(dones.shape)}"
            )
        keep = (~dones.to(device=hidden_state.device, dtype=torch.bool))[None, :, None]
        return hidden_state * keep.to(hidden_state.dtype)

    def action_mean_and_feature_step(
        self,
        observations: TensorDict,
        hidden_state: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        proprio, depth, depth_updated = self._validate_step_inputs(observations)
        privileged = self._privileged_input(observations, proprio.shape[0])
        if hidden_state is None:
            hidden_state = self.initial_hidden(
                proprio.shape[0], device=proprio.device, dtype=proprio.dtype
            )
        self._validate_hidden(hidden_state, proprio.shape[0])
        proprio_feature = self.proprio_encoder(proprio.flatten(-2))
        depth_feature = self.depth_encoder(depth)
        recurrent_input = torch.cat((proprio_feature, depth_feature), dim=-1)
        _, candidate = self.gru(recurrent_input.unsqueeze(0), hidden_state)
        next_hidden = torch.where(
            depth_updated[None, :, None], candidate, hidden_state
        )
        recurrent_feature = self.recurrent_projection(next_hidden.squeeze(0))
        feature_inputs = (proprio[:, -1], recurrent_feature)
        if privileged is not None:
            feature_inputs = (*feature_inputs, privileged)
        feature = self.feature_mlp(torch.cat(feature_inputs, dim=-1))
        mean = self.action_head(feature)
        diagnostics = {
            "proprio_feature": proprio_feature,
            "depth_feature": depth_feature,
            "recurrent_input": recurrent_input,
            "gru_candidate": candidate,
            "recurrent_feature": recurrent_feature,
            "actor_feature": feature,
        }
        if privileged is not None:
            diagnostics[PRIVILEGED_KEY] = privileged
        return mean, feature, next_hidden, diagnostics

    def action_mean_and_feature_sequence(
        self,
        proprio_history: torch.Tensor,
        depth: torch.Tensor,
        depth_updated: torch.Tensor,
        dones_after: torch.Tensor,
        initial_hidden: torch.Tensor,
        privileged: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_sequence_inputs(
            proprio_history, depth, depth_updated, dones_after, initial_hidden
        )
        steps, batch_size = proprio_history.shape[:2]
        recurrent_inputs = self._encode(
            proprio_history.flatten(0, 1), depth.flatten(0, 1)
        ).reshape(steps, batch_size, -1)
        if self.privileged_dim:
            if privileged is None:
                raise KeyError(f"Missing recurrent visual student observation: {PRIVILEGED_KEY}")
            if tuple(privileged.shape) != (steps, batch_size, self.privileged_dim):
                raise ValueError(
                    f"privileged sequence must have shape {(steps, batch_size, self.privileged_dim)}, "
                    f"got {tuple(privileged.shape)}"
                )
            if not torch.isfinite(privileged).all():
                raise FloatingPointError("privileged sequence contains non-finite values")
        elif privileged is not None:
            raise ValueError("privileged sequence supplied to a non-privileged student")
        hidden = initial_hidden
        means = []
        features = []
        for step in range(steps):
            if step > 0:
                hidden = self.reset_hidden_state(hidden, dones_after[step - 1])
            _, candidate = self.gru(recurrent_inputs[step : step + 1], hidden)
            hidden = torch.where(
                depth_updated[step][None, :, None], candidate, hidden
            )
            mean, feature = self._actor(
                proprio_history[step, :, -1],
                hidden.squeeze(0),
                None if privileged is None else privileged[step],
            )
            means.append(mean)
            features.append(feature)
        return torch.stack(means), torch.stack(features), hidden

    def forward(self, observations: TensorDict) -> torch.Tensor:
        proprio, _, _ = self._validate_step_inputs(observations)
        if (
            self._inference_hidden_state is None
            or self._inference_hidden_state.shape[1] != proprio.shape[0]
            or self._inference_hidden_state.device != proprio.device
            or self._inference_hidden_state.dtype != proprio.dtype
        ):
            self._inference_hidden_state = self.initial_hidden(
                proprio.shape[0], device=proprio.device, dtype=proprio.dtype
            )
        mean, _, self._inference_hidden_state, _ = (
            self.action_mean_and_feature_step(
                observations, self._inference_hidden_state
            )
        )
        self._inference_hidden_state = self._inference_hidden_state.detach()
        return mean

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if self._inference_hidden_state is None:
            return
        if dones is None:
            self._inference_hidden_state.zero_()
        else:
            self._inference_hidden_state = self.reset_hidden_state(
                self._inference_hidden_state, dones
            )

    def _encode(
        self, proprio_history: torch.Tensor, depth: torch.Tensor
    ) -> torch.Tensor:
        proprio_feature = self.proprio_encoder(proprio_history.flatten(-2))
        depth_feature = self.depth_encoder(depth)
        return torch.cat((proprio_feature, depth_feature), dim=-1)

    def _actor(
        self,
        current_proprio: torch.Tensor,
        recurrent_hidden: torch.Tensor,
        privileged: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent_feature = self.recurrent_projection(recurrent_hidden)
        feature_inputs = (current_proprio, recurrent_feature)
        if self.privileged_dim:
            if privileged is None:
                raise KeyError(f"Missing recurrent visual student observation: {PRIVILEGED_KEY}")
            feature_inputs = (*feature_inputs, privileged)
        elif privileged is not None:
            raise ValueError("privileged input supplied to a non-privileged student")
        feature = self.feature_mlp(torch.cat(feature_inputs, dim=-1))
        return self.action_head(feature), feature

    def _privileged_input(
        self, observations: TensorDict, batch_size: int
    ) -> torch.Tensor | None:
        if self.privileged_dim == 0:
            if PRIVILEGED_KEY in observations:
                raise ValueError("Privileged input supplied to a non-privileged student")
            return None
        if PRIVILEGED_KEY not in observations:
            raise KeyError(f"Missing recurrent visual student observation: {PRIVILEGED_KEY}")
        privileged = observations[PRIVILEGED_KEY]
        if tuple(privileged.shape) != (batch_size, self.privileged_dim):
            raise ValueError(
                f"privileged_actor must have shape {(batch_size, self.privileged_dim)}, "
                f"got {tuple(privileged.shape)}"
            )
        if not torch.isfinite(privileged).all():
            raise FloatingPointError("privileged_actor contains non-finite values")
        if float(privileged.abs().amax()) > 100.0 + 1.0e-5:
            raise ValueError("privileged_actor exceeds the source clip range [-100, 100]")
        return privileged

    @staticmethod
    def _validate_step_inputs(
        observations: TensorDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for key in (PROPRIO_KEY, DEPTH_KEY, DEPTH_UPDATE_KEY):
            if key not in observations:
                raise KeyError(f"Missing recurrent visual student observation: {key}")
        proprio = observations[PROPRIO_KEY]
        depth = observations[DEPTH_KEY]
        depth_updated = observations[DEPTH_UPDATE_KEY]
        batch_size = proprio.shape[0]
        if tuple(proprio.shape) != (
            batch_size,
            PROPRIO_HISTORY_FRAMES,
            PROPRIO_DIM,
        ):
            raise ValueError(
                "proprio_history must have shape "
                f"[B,{PROPRIO_HISTORY_FRAMES},{PROPRIO_DIM}], got {tuple(proprio.shape)}"
            )
        if tuple(depth.shape) != (batch_size, 1, DEPTH_HEIGHT, DEPTH_WIDTH):
            raise ValueError(
                f"depth_history must have shape [B,1,{DEPTH_HEIGHT},{DEPTH_WIDTH}], "
                f"got {tuple(depth.shape)}"
            )
        if tuple(depth_updated.shape) == (batch_size, 1):
            depth_updated = depth_updated[:, 0]
        elif tuple(depth_updated.shape) != (batch_size,):
            raise ValueError(
                f"depth_updated must have shape [B] or [B,1], got {tuple(depth_updated.shape)}"
            )
        CR1RecurrentVisualDaggerStudent._validate_numeric_contract(
            proprio, depth, depth_updated
        )
        return proprio, depth, depth_updated.to(dtype=torch.bool)

    @staticmethod
    def _validate_numeric_contract(
        proprio: torch.Tensor,
        depth: torch.Tensor,
        depth_updated: torch.Tensor,
    ) -> None:
        """Fail fast when an input bypasses the fixed field-wise scaling contract."""

        for name, value in (
            ("proprio_history", proprio),
            ("depth_history", depth),
            ("depth_updated", depth_updated),
        ):
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"Recurrent visual student {name} contains non-finite values"
                )
        proprio_max = float(proprio.abs().amax())
        if proprio_max > 100.0 + 1.0e-5:
            raise ValueError(
                "proprio_history exceeds the source clip range [-100, 100]: "
                f"max_abs={proprio_max:.6g}"
            )
        depth_min = float(depth.amin())
        depth_max = float(depth.amax())
        if depth_min < DEPTH_INPUT_MIN - 1.0e-5 or depth_max > DEPTH_INPUT_MAX + 1.0e-5:
            raise ValueError(
                "depth_history is outside the centered metric-depth contract "
                f"[{DEPTH_INPUT_MIN}, {DEPTH_INPUT_MAX}]: "
                f"range=[{depth_min:.6g}, {depth_max:.6g}]"
            )

    @staticmethod
    def _validate_sequence_inputs(
        proprio_history: torch.Tensor,
        depth: torch.Tensor,
        depth_updated: torch.Tensor,
        dones_after: torch.Tensor,
        initial_hidden: torch.Tensor,
    ) -> None:
        if proprio_history.ndim != 4 or tuple(proprio_history.shape[-2:]) != (
            PROPRIO_HISTORY_FRAMES,
            PROPRIO_DIM,
        ):
            raise ValueError(
                "Sequence proprio_history must have shape "
                f"[T,B,{PROPRIO_HISTORY_FRAMES},{PROPRIO_DIM}]"
            )
        steps, batch_size = proprio_history.shape[:2]
        if tuple(depth.shape) != (steps, batch_size, 1, DEPTH_HEIGHT, DEPTH_WIDTH):
            raise ValueError("Sequence depth must have shape [T,B,1,60,80]")
        for name, value in (
            ("depth_updated", depth_updated),
            ("dones_after", dones_after),
        ):
            if tuple(value.shape) != (steps, batch_size):
                raise ValueError(
                    f"{name} must have shape {(steps, batch_size)}, got {tuple(value.shape)}"
                )
        CR1RecurrentVisualDaggerStudent._validate_numeric_contract(
            proprio_history,
            depth,
            depth_updated,
        )
        if not torch.isfinite(dones_after).all():
            raise FloatingPointError("dones_after contains non-finite values")
        CR1RecurrentVisualDaggerStudent._validate_hidden(initial_hidden, batch_size)

    @staticmethod
    def _validate_hidden(hidden_state: torch.Tensor, batch_size: int) -> None:
        expected = (1, batch_size, RECURRENT_HIDDEN_DIM)
        if tuple(hidden_state.shape) != expected:
            raise ValueError(
                f"GRU hidden_state must have shape {expected}, got {tuple(hidden_state.shape)}"
            )
        if not torch.isfinite(hidden_state).all():
            raise FloatingPointError("GRU hidden_state contains non-finite values")


__all__ = [
    "CR1RecurrentDepthEncoder",
    "CR1RecurrentVisualDaggerStudent",
    "DEPTH_UPDATE_KEY",
    "PROPRIO_HISTORY_FRAMES",
    "PROPRIO_DIM",
    "RECURRENT_HIDDEN_DIM",
    "RECURRENT_VISUAL_DAGGER_DATA_CONTRACT",
]
