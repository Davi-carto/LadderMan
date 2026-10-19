"""LayerNorm-only ablation of fixed 25/75 mixed recurrent action-MSE DAgger."""

from .recurrent_visual_mse_algorithm import CR1RecurrentVisualMseDagger
from .recurrent_visual_ppo_model import CR1RecurrentVisualPpoStudent


RECURRENT_VISUAL_LN_MSE_DAGGER_DATA_CONTRACT = (
    "cr1_visual_ln_mse_dagger_proprio25x85_command_each_"
    "depth1x60x80_30hz_layernorm_gru256_mixed_v1"
)


class CR1RecurrentVisualLnMseDagger(CR1RecurrentVisualMseDagger):
    """Reuse the normalized network with the original MSE-only optimizer."""

    # This imported class is only a network; the MSE base keeps its std fixed
    # and does not construct a critic, returns, or any PPO objective.
    student_class = CR1RecurrentVisualPpoStudent
    data_contract = RECURRENT_VISUAL_LN_MSE_DAGGER_DATA_CONTRACT

    def configure_diagnostics(self, log_dir: str, manifest: dict) -> None:
        super().configure_diagnostics(log_dir, {
            **manifest,
            "data_contract": self.data_contract,
            "feature_normalization": "LayerNorm(128) per proprio/depth branch before GRU",
            "ablation": "original no-warmup mixed action-MSE DAgger plus branch LayerNorm only",
        })


__all__ = [
    "CR1RecurrentVisualLnMseDagger",
    "RECURRENT_VISUAL_LN_MSE_DAGGER_DATA_CONTRACT",
]
