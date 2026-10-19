"""RSL-RL configuration for the visual action-MSE DAgger control."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from ..agents.rsl_rl_ppo_cfg import (
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputPPORunnerCfg,
)


@configclass
class CR1VisualMseDaggerAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = (
        "ladder.tasks.ladder_climb.config.cr1.dagger.visual_mse_algorithm:"
        "CR1VisualMseDagger"
    )
    teacher_checkpoint: str = ""
    teacher_hidden_dims: list[int] = [512, 256, 256]
    expected_teacher_observation_dim: int = 780
    diagnostic_snapshot_interval: int = 500
    diagnostic_sample_envs: int = 16


@configclass
class CR1VisualMseDaggerRunnerCfg(
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputPPORunnerCfg
):
    """100% student-rollout visual DAgger with action-mean MSE only."""

    use_pure_dagger_visual_runner: bool = True
    use_visual_mse_dagger_runner: bool = True
    seed: int = 42
    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 500
    experiment_name: str = "cr1_zdw0824_visual_mse_dagger_synced_model9999"
    empirical_normalization: bool = False
    clip_actions: float | None = None
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,
        actor_hidden_dims=[2048, 1024, 512, 256],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm: CR1VisualMseDaggerAlgorithmCfg = CR1VisualMseDaggerAlgorithmCfg(
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


__all__ = ["CR1VisualMseDaggerAlgorithmCfg", "CR1VisualMseDaggerRunnerCfg"]
