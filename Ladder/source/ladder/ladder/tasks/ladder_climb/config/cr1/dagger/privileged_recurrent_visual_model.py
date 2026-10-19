"""Privileged-information recurrent visual student for an oracle ablation."""

from __future__ import annotations

from torch import nn

from .recurrent_visual_model import (
    CR1RecurrentVisualDaggerStudent,
    PRIVILEGED_KEY,
)


# 4 limbs x 3-D next-stable target position, 4 contact flags, and 4 limbs x
# 3-D body-frame linear velocity.
PRIVILEGED_TARGET_POS_DIM = 4 * 3
PRIVILEGED_CONTACT_DIM = 4
PRIVILEGED_LIMB_VELOCITY_DIM = 4 * 3
PRIVILEGED_LIMB_NAMES = (
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
)
PRIVILEGED_ACTOR_DIM = (
    PRIVILEGED_TARGET_POS_DIM
    + PRIVILEGED_CONTACT_DIM
    + PRIVILEGED_LIMB_VELOCITY_DIM
)
PRIVILEGED_RECURRENT_VISUAL_LEGACY_DAGGER_DATA_CONTRACT = (
    "cr1_privileged_visual_recurrent_mse_dagger_proprio25x85_"
    "depth1x60x80_30hz_gru256_target12_contact4_limbvel12_v1"
)
PRIVILEGED_RECURRENT_VISUAL_DAGGER_DATA_CONTRACT = (
    "cr1_privileged_visual_recurrent_mse_dagger_proprio25x85_"
    "depth1x60x80_30hz_layernorm_gru256_target12_contact4_limbvel12_v2"
)


class CR1PrivilegedRecurrentVisualLegacyDaggerStudent(
    CR1RecurrentVisualDaggerStudent
):
    """Legacy unnormalized privileged student kept for checkpoint inspection."""

    def __init__(self, action_dim: int, hidden_dims: list[int], init_noise_std: float) -> None:
        super().__init__(
            action_dim,
            hidden_dims,
            init_noise_std,
            privileged_dim=PRIVILEGED_ACTOR_DIM,
        )


class CR1PrivilegedRecurrentVisualDaggerStudent(
    CR1PrivilegedRecurrentVisualLegacyDaggerStudent
):
    """Privileged student with normalized proprioception and depth features."""

    def __init__(self, action_dim: int, hidden_dims: list[int], init_noise_std: float) -> None:
        super().__init__(action_dim, hidden_dims, init_noise_std)
        # Both step inference and sequence training use these same encoders.
        self.proprio_encoder.add_module("layer_norm", nn.LayerNorm(128))
        self.depth_encoder.projection.add_module("layer_norm", nn.LayerNorm(128))


__all__ = [
    "PRIVILEGED_KEY",
    "PRIVILEGED_ACTOR_DIM",
    "PRIVILEGED_LIMB_NAMES",
    "PRIVILEGED_RECURRENT_VISUAL_LEGACY_DAGGER_DATA_CONTRACT",
    "PRIVILEGED_RECURRENT_VISUAL_DAGGER_DATA_CONTRACT",
    "CR1PrivilegedRecurrentVisualLegacyDaggerStudent",
    "CR1PrivilegedRecurrentVisualDaggerStudent",
]
