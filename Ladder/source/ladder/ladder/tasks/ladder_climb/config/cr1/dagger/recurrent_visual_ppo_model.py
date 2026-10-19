"""Normalized recurrent student for end-to-end PPO and action imitation."""

from torch import nn

from .recurrent_visual_model import CR1RecurrentVisualDaggerStudent


RECURRENT_VISUAL_PPO_DAGGER_DATA_CONTRACT = (
    "cr1_visual_ppo_dagger_proprio25x85_depth1x60x80_30hz_"
    "layernorm_gru256_privileged_critic_mixed_v1"
)


class CR1RecurrentVisualPpoStudent(CR1RecurrentVisualDaggerStudent):
    """Normalize each 128-D feature branch before the shared visual GRU."""

    def __init__(self, action_dim, hidden_dims, init_noise_std):
        super().__init__(action_dim, hidden_dims, init_noise_std)
        # Both step inference and sequence training use these same encoders.
        self.proprio_encoder.add_module("layer_norm", nn.LayerNorm(128))
        self.depth_encoder.projection.add_module("layer_norm", nn.LayerNorm(128))


__all__ = [
    "CR1RecurrentVisualPpoStudent",
    "RECURRENT_VISUAL_PPO_DAGGER_DATA_CONTRACT",
]
