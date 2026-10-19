"""Configuration for the privileged-information recurrent visual DAgger ablation."""

from isaaclab.utils import configclass

from .recurrent_visual_mse_agent_cfg import (
    CR1RecurrentVisualScheduledMseDaggerAlgorithmCfg,
    CR1RecurrentVisualScheduledMseDaggerRunnerCfg,
)


@configclass
class CR1PrivilegedRecurrentVisualMseDaggerAlgorithmCfg(
    CR1RecurrentVisualScheduledMseDaggerAlgorithmCfg
):
    bc_warmup_iterations: int = 1000
    teacher_rollout_schedule_start: int = 1000
    teacher_rollout_schedule_end: int = 5000
    class_name: str = (
        "ladder.tasks.ladder_climb.config.cr1.dagger."
        "privileged_recurrent_visual_mse_algorithm:"
        "CR1PrivilegedRecurrentVisualMseDagger"
    )


@configclass
class CR1PrivilegedRecurrentVisualMseDaggerRunnerCfg(
    CR1RecurrentVisualScheduledMseDaggerRunnerCfg
):
    """Scheduled action-MSE DAgger with a 28-D privileged oracle input."""

    use_privileged_recurrent_visual_mse_dagger_runner: bool = True
    experiment_name: str = (
        "cr1_zdw0824_privileged_recurrent_visual_ln_mse_dagger_model9999"
    )
    algorithm: CR1PrivilegedRecurrentVisualMseDaggerAlgorithmCfg = (
        CR1PrivilegedRecurrentVisualMseDaggerAlgorithmCfg(
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
    "CR1PrivilegedRecurrentVisualMseDaggerAlgorithmCfg",
    "CR1PrivilegedRecurrentVisualMseDaggerRunnerCfg",
]
