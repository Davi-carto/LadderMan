"""RSL-RL configurations for visual CR1 DAgger training."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from ..agents.rsl_rl_ppo_cfg import (
    CR1TransformerVisionPPOAlgorithmCfg,
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputPPORunnerCfg,
)


@configclass
class CR1VisualDaggerAlgorithmCfg(CR1TransformerVisionPPOAlgorithmCfg):
    class_name: str = (
        "ladder.tasks.ladder_climb.config.cr1.dagger.visual_algorithm:"
        "CR1VisualHybridDaggerPPO"
    )
    teacher_checkpoint: str = ""
    teacher_hidden_dims: list[int] = [512, 256, 256]
    dagger_initial_coeff: float = 1.0
    dagger_final_coeff: float = 0.05
    dagger_anneal_start: int = 500
    dagger_anneal_iterations: int = 4000
    teacher_checkpoint: str = ""
    teacher_hidden_dims: list[int] = [512, 256, 256]
    dagger_initial_coeff: float = 1.0
    dagger_final_coeff: float = 0.05
    dagger_anneal_start: int = 500
    dagger_anneal_iterations: int = 4000


@configclass
class CR1VisualDaggerPPORunnerCfg(
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputPPORunnerCfg
):
    use_dagger_visual_runner: bool = True
    experiment_name: str = "cr1_zdw0824_up_down_visual_dagger"
    save_interval: int = 500
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 1024, 512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm: CR1VisualDaggerAlgorithmCfg = CR1VisualDaggerAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
        visual_architecture="deterministic",
        visual_initialization="scratch",
        auxiliary_learning_rate=5.0e-4,
        auxiliary_mini_batch_envs=256,
        auxiliary_num_epochs=5,
        use_grad_scaler=True,
    )


@configclass
class CR1PureVisualDaggerAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = (
        "ladder.tasks.ladder_climb.config.cr1.dagger.pure_algorithm:"
        "CR1PureVisualDagger"
    )
    teacher_checkpoint: str = ""
    teacher_hidden_dims: list[int] = [512, 256, 256]
    action_kl_weight: float = 2.0
    feature_weight: float = 1.0
    bc_warmup_iterations: int = 500
    teacher_rollout_fraction: float = 0.25


@configclass
class CR1PureVisualDaggerRunnerCfg(
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputPPORunnerCfg
):
    use_pure_dagger_visual_runner: bool = True
    experiment_name: str = "cr1_zdw0824_up_down_pure_visual_dagger"
    save_interval: int = 500
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 1024, 512, 256],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm: CR1PureVisualDaggerAlgorithmCfg = CR1PureVisualDaggerAlgorithmCfg(
        value_loss_coef=0.0,
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=0.5,
    )


@configclass
class CR1SyncedVisualDaggerPPORunnerCfg(CR1VisualDaggerPPORunnerCfg):
    """Separate logs for the visual task synchronized to the latest blind MDP."""

    experiment_name: str = "cr1_zdw0824_visual_dagger_synced_latest_blind"


@configclass
class CR1SyncedPureVisualDaggerRunnerCfg(CR1PureVisualDaggerRunnerCfg):
    """Separate logs for pure visual DAgger on the synchronized MDP."""

    experiment_name: str = "cr1_zdw0824_pure_visual_dagger_synced_latest_blind"


@configclass
class CR1TeacherOnlyVisualBCAlgorithmCfg(CR1PureVisualDaggerAlgorithmCfg):
    class_name: str = (
        "ladder.tasks.ladder_climb.config.cr1.dagger.teacher_only_algorithm:"
        "CR1TeacherOnlyVisualBC"
    )
    bc_warmup_iterations: int = 0
    teacher_rollout_fraction: float = 1.0
    validation_num_envs: int = 64
    diagnostic_snapshot_interval: int = 500


@configclass
class CR1TeacherOnlyVisualBCRunnerCfg(CR1PureVisualDaggerRunnerCfg):
    use_teacher_only_visual_bc: bool = True
    experiment_name: str = "cr1_zdw0824_up_down_teacher_only_visual_bc"
    algorithm: CR1TeacherOnlyVisualBCAlgorithmCfg = CR1TeacherOnlyVisualBCAlgorithmCfg(
        value_loss_coef=0.0,
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=0.5,
    )


__all__ = [
    "CR1VisualDaggerAlgorithmCfg",
    "CR1VisualDaggerPPORunnerCfg",
    "CR1PureVisualDaggerAlgorithmCfg",
    "CR1PureVisualDaggerRunnerCfg",
    "CR1SyncedVisualDaggerPPORunnerCfg",
    "CR1SyncedPureVisualDaggerRunnerCfg",
    "CR1TeacherOnlyVisualBCAlgorithmCfg",
    "CR1TeacherOnlyVisualBCRunnerCfg",
]
