"""End-to-end recurrent visual PPO with mixed-rollout action-MSE DAgger."""

from isaaclab.utils import configclass

from .recurrent_visual_mse_agent_cfg import (
    CR1RecurrentVisualNoWarmupMixedMseDaggerAlgorithmCfg,
    CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg,
)


@configclass
class CR1RecurrentVisualPpoDaggerAlgorithmCfg(
    CR1RecurrentVisualNoWarmupMixedMseDaggerAlgorithmCfg
):
    class_name: str = (
        "ladder.tasks.ladder_climb.config.cr1.dagger."
        "recurrent_visual_ppo_algorithm:CR1RecurrentVisualPpoDagger"
    )
    min_learning_rate: float = 1.0e-5
    max_learning_rate: float = 3.0e-4
    ppo_initial_coeff: float = 0.1
    ppo_warmup_iterations: int = 500
    dagger_initial_coeff: float = 1.0
    dagger_final_coeff: float = 0.1
    dagger_anneal_start: int = 2000
    dagger_anneal_iterations: int = 5000


@configclass
class CR1RecurrentVisualPpoDaggerRunnerCfg(
    CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg
):
    """25% teacher for imitation; 75% student for PPO and imitation."""

    use_recurrent_visual_ppo_dagger_runner: bool = True
    experiment_name: str = "cr1_zdw0824_recurrent_visual_ppo_dagger_model9999"
    algorithm: CR1RecurrentVisualPpoDaggerAlgorithmCfg = (
        CR1RecurrentVisualPpoDaggerAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=3,
            num_mini_batches=24,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    )


__all__ = [
    "CR1RecurrentVisualPpoDaggerAlgorithmCfg",
    "CR1RecurrentVisualPpoDaggerRunnerCfg",
]
