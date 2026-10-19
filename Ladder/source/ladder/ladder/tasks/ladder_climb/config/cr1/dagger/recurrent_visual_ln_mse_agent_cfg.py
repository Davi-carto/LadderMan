"""Keep the no-warmup MSE baseline settings for a LayerNorm-only control."""

from isaaclab.utils import configclass

from .recurrent_visual_mse_agent_cfg import (
    CR1RecurrentVisualNoWarmupMixedMseDaggerAlgorithmCfg,
    CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg,
)


@configclass
class CR1RecurrentVisualLnMseDaggerAlgorithmCfg(
    CR1RecurrentVisualNoWarmupMixedMseDaggerAlgorithmCfg
):
    class_name: str = (
        "ladder.tasks.ladder_climb.config.cr1.dagger."
        "recurrent_visual_ln_mse_algorithm:CR1RecurrentVisualLnMseDagger"
    )


@configclass
class CR1RecurrentVisualLnMseDaggerRunnerCfg(
    CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg
):
    """From-scratch LayerNorm ablation; 25% teacher and 75% student rollout."""

    use_recurrent_visual_ln_mse_dagger_runner: bool = True
    experiment_name: str = "cr1_zdw0824_recurrent_visual_ln_mse_dagger_model9999"
    max_iterations: int = 2000
    save_interval: int = 500
    algorithm: CR1RecurrentVisualLnMseDaggerAlgorithmCfg = (
        CR1RecurrentVisualLnMseDaggerAlgorithmCfg(
            value_loss_coef=0.0,
            use_clipped_value_loss=False,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=1,
            num_mini_batches=24,
            learning_rate=1.0e-3,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=None,
            max_grad_norm=1.0,
        )
    )


__all__ = [
    "CR1RecurrentVisualLnMseDaggerAlgorithmCfg",
    "CR1RecurrentVisualLnMseDaggerRunnerCfg",
]
