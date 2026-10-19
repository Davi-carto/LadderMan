"""Scratch PPO with a jointly optimized policy-only visual encoder."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CR1EndToEndVisualPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "ladder.rsl_rl.algorithms.end_to_end_visual_ppo:EndToEndVisualPPO"
    min_learning_rate: float = 1.0e-5
    max_learning_rate: float = 1.0e-3
    sequence_micro_batch_envs: int = 64


@configclass
class CR1EndToEndVisualPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    use_end_to_end_visual_ppo: bool = True
    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "cr1_zdw0824_end_to_end_visual_policy_only_ppo"
    empirical_normalization = False
    clip_actions = None
    clip_observations: float | None = None
    obs_groups = {
        "actor": [
            "transformer_vision_current_proprio",
            "transformer_vision_proprio_history",
            "transformer_vision_depth_history",
        ],
        "critic": ["critic"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = CR1EndToEndVisualPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        for name in ("actor_obs_normalization", "critic_obs_normalization"):
            if hasattr(self.policy, name):
                setattr(self.policy, name, False)


@configclass
class CR1EndToEndVisualOraclePolicyCfg(RslRlPpoActorCriticCfg):
    """End-to-end actor configuration with the oracle 32-D intent enabled."""

    use_oracle_contact_goal: bool = True


@configclass
class CR1EndToEndVisualOracleContactGoalPPORunnerCfg(
    CR1EndToEndVisualPPORunnerCfg
):
    """Independent oracle contact-goal diagnostic run."""

    experiment_name = "cr1_zdw0824_end_to_end_visual_oracle_contact_goal_policy_only_ppo"
    # Match the accepted full-state PPO safety envelope.  This bounds the
    # action stored by IsaacLab (and therefore the previous-action observation)
    # before the existing joint-target rate limiter converts it to a command.
    clip_actions = 10.0
    obs_groups = {
        "actor": [
            "transformer_vision_current_proprio",
            "transformer_vision_proprio_history",
            "transformer_vision_depth_history",
            "oracle_contact_goal",
        ],
        "critic": ["critic"],
    }
    policy = CR1EndToEndVisualOraclePolicyCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


__all__ = [
    "CR1EndToEndVisualPPORunnerCfg",
    "CR1EndToEndVisualPPOAlgorithmCfg",
    "CR1EndToEndVisualOraclePolicyCfg",
    "CR1EndToEndVisualOracleContactGoalPPORunnerCfg",
]
