"""Adversarial motion-prior components for LadderMan.

The discriminator deliberately receives only task-agnostic robot motion
features.  Stable-support commands remain the source of task intent; AMP only
specifies how a transition should look.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # Isaac Lab 2.1 legacy spelling.
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


AMP_STYLE_FEATURE_LAYOUT = (
    ("projected_gravity", 3),
    ("anchor_linear_velocity_local", 3),
    ("anchor_angular_velocity_local", 3),
    ("joint_position", 25),
    ("joint_velocity", 25),
    ("limb_position_local", 12),
)
AMP_STYLE_OBSERVATION_DIM = sum(width for _, width in AMP_STYLE_FEATURE_LAYOUT)


def build_amp_style_observation(
    *,
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
    anchor_lin_vel_w: torch.Tensor,
    anchor_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    limb_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Build the local, task-agnostic robot state used by AMP.

    Absolute world position, ladder geometry, reference phase, commands, and
    target rung IDs are intentionally excluded.  This follows AMP's separation
    between a task reward (what to do) and a motion prior (how to do it).
    """

    batch = anchor_pos_w.shape[0]
    expected = {
        "anchor_pos_w": (batch, 3),
        "anchor_quat_w": (batch, 4),
        "anchor_lin_vel_w": (batch, 3),
        "anchor_ang_vel_w": (batch, 3),
        "joint_pos": (batch, 25),
        "joint_vel": (batch, 25),
        "limb_pos_w": (batch, 4, 3),
    }
    values = locals()
    for name, shape in expected.items():
        if tuple(values[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(values[name].shape)}")

    gravity_w = anchor_pos_w.new_zeros(batch, 3)
    gravity_w[:, 2] = -1.0
    projected_gravity = quat_apply_inverse(anchor_quat_w, gravity_w)
    anchor_lin_vel_local = quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w)
    anchor_ang_vel_local = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
    relative_limb_pos_w = limb_pos_w - anchor_pos_w[:, None, :]
    repeated_quat = anchor_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
    limb_pos_local = quat_apply_inverse(
        repeated_quat, relative_limb_pos_w.reshape(-1, 3)
    ).reshape(batch, 12)
    observation = torch.cat(
        (
            projected_gravity,
            anchor_lin_vel_local,
            anchor_ang_vel_local,
            joint_pos,
            joint_vel,
            limb_pos_local,
        ),
        dim=-1,
    )
    if observation.shape != (batch, AMP_STYLE_OBSERVATION_DIM):
        raise RuntimeError(
            "AMP observation layout changed unexpectedly: "
            f"{tuple(observation.shape)}"
        )
    return observation


class LadderAMPReferenceDataset:
    """Sample phase-free expert transitions from the loaded ladder motion."""

    def __init__(
        self,
        command,
        *,
        anchor_hold_probability: float = 0.20,
        balance_directions: bool = True,
        num_frames: int = 5,
        history_stride: int = 2,
    ) -> None:
        if not 0.0 <= anchor_hold_probability <= 1.0:
            raise ValueError("anchor_hold_probability must be in [0, 1]")
        motion = command.motion
        if num_frames < 2 or history_stride < 1:
            raise ValueError("AMP requires at least two frames and positive stride")
        self.num_frames, self.history_stride = num_frames, history_stride
        if motion._stable_support_limb_body_indexes is None:
            raise ValueError("Ladder AMP requires ground-aware stable-support limb bodies")
        anchor = command.motion_anchor_body_index
        limb_indexes = motion._stable_support_limb_body_indexes
        self.features = build_amp_style_observation(
            anchor_pos_w=motion.body_pos_w[:, anchor],
            anchor_quat_w=motion.body_quat_w[:, anchor],
            anchor_lin_vel_w=motion.body_lin_vel_w[:, anchor],
            anchor_ang_vel_w=motion.body_ang_vel_w[:, anchor],
            joint_pos=motion.joint_pos,
            joint_vel=motion.joint_vel,
            limb_pos_w=motion._body_pos_w[:, limb_indexes],
        ).detach()
        self.history_span = (num_frames - 1) * history_stride + 1
        if self.features.shape[0] < self.history_span:
            raise ValueError("AMP reference is shorter than its history window")
        if not torch.isfinite(self.features).all():
            raise ValueError("AMP reference features contain non-finite values")

        self.anchor_hold_probability = float(anchor_hold_probability)
        self.balance_directions = bool(balance_directions)
        transition_indexes = torch.arange(
            self.features.shape[0] - self.history_span + 1, device=self.features.device
        )
        direction = motion.climb_direction[transition_indexes]
        self._all_transition_indexes = transition_indexes
        self._up_transition_indexes = transition_indexes[direction > 0]
        self._down_transition_indexes = transition_indexes[direction < 0]
        anchors = motion.stable_motion_anchor_frames
        self._anchor_indexes = anchors[
            (anchors >= 0) & (anchors < self.features.shape[0])
        ]
        if self._anchor_indexes.numel() == 0:
            raise ValueError("AMP stable-hold augmentation requires stable support anchors")
        self.hold_features = self.features[self._anchor_indexes].clone()
        # A held pose has zero linear/angular/joint velocity. Repeating a
        # moving reference frame without this correction is inconsistent.
        self.hold_features[:, 3:9] = 0.0
        self.hold_features[:, 34:59] = 0.0

    @property
    def observation_dim(self) -> int:
        return AMP_STYLE_OBSERVATION_DIM

    def _sample_indexes(self, source: torch.Tensor, count: int) -> torch.Tensor:
        if count == 0:
            return source.new_empty((0,))
        if source.numel() == 0:
            source = self._all_transition_indexes
        selected = torch.randint(source.numel(), (count,), device=source.device)
        return source[selected]

    def sample(self, batch_size: int) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("AMP expert batch_size must be positive")
        hold_count = int(round(batch_size * self.anchor_hold_probability))
        motion_count = batch_size - hold_count
        if (
            self.balance_directions
            and self._up_transition_indexes.numel() > 0
            and self._down_transition_indexes.numel() > 0
        ):
            up_count = motion_count // 2
            indexes = torch.cat(
                (
                    self._sample_indexes(self._up_transition_indexes, up_count),
                    self._sample_indexes(
                        self._down_transition_indexes, motion_count - up_count
                    ),
                )
            )
        else:
            indexes = self._sample_indexes(self._all_transition_indexes, motion_count)
        offsets = torch.arange(self.num_frames, device=self.features.device) * self.history_stride
        transitions = self.features[indexes[:, None] + offsets].flatten(1)
        if hold_count:
            selection = torch.randint(len(self.hold_features), (hold_count,), device=self.features.device)
            holds = self.hold_features[selection, None].expand(-1, self.num_frames, -1).flatten(1)
            transitions = torch.cat((transitions, holds), dim=0)
        permutation = torch.randperm(batch_size, device=self.features.device)
        return transitions[permutation]


class AMPHistory:
    """SSR-style strided state windows, excluding cross-reset/padded samples."""

    def __init__(self, num_envs, feature_dim, num_frames, stride, device):
        self.span = (num_frames - 1) * stride + 1
        self.stride = stride
        self.frames = torch.zeros(num_envs, self.span, feature_dim, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def seed(self, state):
        empty = self.count == 0
        self.frames[empty] = state[empty, None]
        self.count[empty] = 1

    def append(self, state, done):
        self.frames[:, :-1] = self.frames[:, 1:].clone()
        self.frames[:, -1] = state
        self.count = (self.count + 1).clamp_max(self.span)
        self.frames[done] = state[done, None]
        self.count[done] = 1
        return self.frames[:, ::self.stride].flatten(1), (self.count >= self.span) & ~done


class AMPRunningMeanStd(nn.Module):
    """SSR's rolling 100-update normalizer, with checkpointed torch buffers."""

    def __init__(self, feature_dim: int, epsilon: float = 1.0e-4) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.epsilon = float(epsilon)
        self.register_buffer("mean", torch.zeros(feature_dim, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(feature_dim, dtype=torch.float64))
        self.register_buffer("mean_history", torch.zeros(100, feature_dim, dtype=torch.float64))
        self.register_buffer("variance_history", torch.zeros(100, feature_dim, dtype=torch.float64))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update_transitions(self, transitions: torch.Tensor) -> None:
        frames = transitions.reshape(-1, self.feature_dim).float()
        if frames.shape[0] < 2:
            return
        index = int(self.updates) % 100
        self.mean_history[index] = frames.mean(dim=0)
        self.variance_history[index] = frames.var(dim=0, unbiased=True)
        self.updates.add_(1)
        size = min(int(self.updates), 100)
        self.mean.copy_(self.mean_history[:size].mean(dim=0))
        self.variance.copy_(self.variance_history[:size].mean(dim=0)
                            + self.mean_history[:size].var(dim=0, unbiased=False))
        self.count.add_(frames.shape[0])

    def normalize_transitions(self, transitions: torch.Tensor) -> torch.Tensor:
        shape = transitions.shape
        frames = transitions.reshape(-1, self.feature_dim)
        normalized = (frames - self.mean) / torch.sqrt(self.variance + self.epsilon)
        return normalized.clamp(-10.0, 10.0).reshape(shape).float()


class AMPDiscriminator(nn.Module):
    """SSR 512/256/128 ReLU discriminator over normalized robot histories."""

    def __init__(self, feature_dim: int, hidden_dims=(512, 256, 128), num_frames=2) -> None:
        super().__init__()
        input_dim = num_frames * int(feature_dim)
        layers = []
        for output_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
            input_dim = output_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, 1)

    def forward(self, transitions: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(transitions))

    @torch.no_grad()
    def reward(self, normalized_transitions: torch.Tensor) -> torch.Tensor:
        prediction = self(normalized_transitions).squeeze(-1)
        return torch.clamp(1.0 - 0.25 * (prediction - 1.0).square(), min=0.0)

    def gradient_penalty(self, expert_transitions: torch.Tensor) -> torch.Tensor:
        expert = expert_transitions.detach().requires_grad_(True)
        prediction = self(expert)
        gradient = torch.autograd.grad(
            prediction,
            expert,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradient.square().sum(dim=-1).mean()


class AMPReplayBuffer:
    """Fixed-size device ring buffer for recent policy state transitions."""

    def __init__(self, capacity: int, transition_dim: int, device) -> None:
        if capacity < 1:
            raise ValueError("AMP replay capacity must be positive")
        self.capacity = int(capacity)
        self.data = torch.empty(self.capacity, transition_dim, device=device)
        self.position = 0
        self.size = 0

    @torch.no_grad()
    def add(self, transitions: torch.Tensor) -> None:
        values = transitions.detach()
        if values.shape[0] >= self.capacity:
            self.data.copy_(values[-self.capacity :])
            self.position = 0
            self.size = self.capacity
            return
        end = self.position + values.shape[0]
        if end <= self.capacity:
            self.data[self.position : end].copy_(values)
        else:
            first = self.capacity - self.position
            self.data[self.position :].copy_(values[:first])
            self.data[: end - self.capacity].copy_(values[first:])
        self.position = end % self.capacity
        self.size = min(self.capacity, self.size + values.shape[0])

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty AMP replay buffer")
        indexes = torch.randint(self.size, (batch_size,), device=self.data.device)
        return self.data[indexes]


__all__ = [
    "AMPDiscriminator",
    "AMPReplayBuffer",
    "AMPRunningMeanStd",
    "AMP_STYLE_FEATURE_LAYOUT",
    "AMP_STYLE_OBSERVATION_DIM",
    "LadderAMPReferenceDataset",
    "build_amp_style_observation",
]
