"""RSL-RL configuration for recurrent visual action-MSE DAgger."""

from isaaclab.utils import configclass

from .visual_mse_agent_cfg import (
    CR1VisualMseDaggerAlgorithmCfg,
    CR1VisualMseDaggerRunnerCfg,
)


_LATEST_BLIND_TEACHER = (
    "logs/rsl_rl/cr1_nonvisual_zdw0824_no_command_full_state_middle_stand_smooth_output/"
    "2026-09-09_13-56-37_train10000_scratch_delay0or5ms_phys5ms_seed42/model_9999.pt"
)


@configclass
class CR1RecurrentVisualMseDaggerAlgorithmCfg(CR1VisualMseDaggerAlgorithmCfg):
    class_name: str = (
        "ladder.tasks.ladder_climb.config.cr1.dagger."
        "recurrent_visual_mse_algorithm:CR1RecurrentVisualMseDagger"
    )
    teacher_checkpoint: str = _LATEST_BLIND_TEACHER
    bc_warmup_iterations: int = 0
    teacher_rollout_fraction: float = 0.0


@configclass
class CR1RecurrentVisualMseDaggerRunnerCfg(CR1VisualMseDaggerRunnerCfg):
    """Sequence-trained GRU student with action-mean MSE only."""

    use_recurrent_visual_mse_dagger_runner: bool = True
    experiment_name: str = "cr1_zdw0824_recurrent_visual_mse_dagger_model9999"
    algorithm: CR1RecurrentVisualMseDaggerAlgorithmCfg = (
        CR1RecurrentVisualMseDaggerAlgorithmCfg(
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


@configclass
class CR1RecurrentVisualMixedMseDaggerAlgorithmCfg(
    CR1RecurrentVisualMseDaggerAlgorithmCfg
):
    bc_warmup_iterations: int = 500
    teacher_rollout_fraction: float = 0.25


@configclass
class CR1RecurrentVisualMixedMseDaggerRunnerCfg(
    CR1RecurrentVisualMseDaggerRunnerCfg
):
    """Teacher BC warmup followed by fixed 25/75 mixed DAgger rollout."""

    use_recurrent_visual_mixed_mse_dagger_runner: bool = True
    experiment_name: str = (
        "cr1_zdw0824_recurrent_visual_mixed_mse_dagger_model9999"
    )
    algorithm: CR1RecurrentVisualMixedMseDaggerAlgorithmCfg = (
        CR1RecurrentVisualMixedMseDaggerAlgorithmCfg(
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


@configclass
class CR1RecurrentVisualNoWarmupMixedMseDaggerAlgorithmCfg(
    CR1RecurrentVisualMixedMseDaggerAlgorithmCfg
):
    bc_warmup_iterations: int = 0


@configclass
class CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg(
    CR1RecurrentVisualMixedMseDaggerRunnerCfg
):
    """Fixed 25/75 mixed DAgger rollout from the first iteration."""

    experiment_name: str = (
        "cr1_zdw0824_recurrent_visual_mixed_no_warmup_mse_dagger_model9999"
    )
    algorithm: CR1RecurrentVisualNoWarmupMixedMseDaggerAlgorithmCfg = (
        CR1RecurrentVisualNoWarmupMixedMseDaggerAlgorithmCfg(
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


@configclass
class CR1RecurrentVisualScheduledMseDaggerAlgorithmCfg(
    CR1RecurrentVisualMseDaggerAlgorithmCfg
):
    """Pure action-MSE DAgger with a linear teacher-controller schedule."""

    bc_warmup_iterations: int = 500
    teacher_rollout_fraction: float = 0.1
    teacher_rollout_schedule_start: int = 500
    teacher_rollout_schedule_end: int = 4000
    teacher_rollout_schedule_initial_fraction: float = 1.0
    teacher_rollout_schedule_final_fraction: float = 0.1


@configclass
class CR1RecurrentVisualScheduledMseDaggerRunnerCfg(
    CR1RecurrentVisualMseDaggerRunnerCfg
):
    """All-teacher warmup, then linear teacher-to-student rollout mixing."""

    use_recurrent_visual_scheduled_mse_dagger_runner: bool = True
    experiment_name: str = (
        "cr1_zdw0824_recurrent_visual_scheduled_mse_dagger_model9999"
    )
    algorithm: CR1RecurrentVisualScheduledMseDaggerAlgorithmCfg = (
        CR1RecurrentVisualScheduledMseDaggerAlgorithmCfg(
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
    "CR1RecurrentVisualMseDaggerAlgorithmCfg",
    "CR1RecurrentVisualMseDaggerRunnerCfg",
    "CR1RecurrentVisualMixedMseDaggerAlgorithmCfg",
    "CR1RecurrentVisualMixedMseDaggerRunnerCfg",
    "CR1RecurrentVisualNoWarmupMixedMseDaggerAlgorithmCfg",
    "CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg",
    "CR1RecurrentVisualScheduledMseDaggerAlgorithmCfg",
    "CR1RecurrentVisualScheduledMseDaggerRunnerCfg",
]
