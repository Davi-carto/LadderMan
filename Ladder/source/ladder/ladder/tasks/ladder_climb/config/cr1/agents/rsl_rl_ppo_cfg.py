from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CR1LadderVisualActorCriticCfg:
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    state_dependent_std: bool = False
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = [2048, 1024, 512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    actor_class_name: str = "ladder.models:LadderVisualActorModel"


@configclass
class CR1SSRVisualPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO settings for Ladder's recurrent SSR visual task."""

    class_name: str = "ladder.algorithms:LadderSSRPPO"
    auxiliary_learning_rate: float = 1.0e-3
    # Gate D keeps this disabled for real jobs until Gate E accepts target
    # normalization and visual auxiliary-loss diagnostics.
    enable_auxiliary_update: bool = False


@configclass
class CR1TransformerVisionPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Separate PPO and temporal visual-update settings for visual actors."""

    class_name: str = "ladder.rsl_rl.algorithms:TransformerVisionPPO"
    min_learning_rate: float = 1.0e-5
    max_learning_rate: float = 1.0e-3
    use_grad_scaler: bool = True
    visual_architecture: str = "deterministic"
    visual_checkpoint: str = ""
    visual_initialization: str = "checkpoint"
    auxiliary_learning_rate: float = 5.0e-4
    auxiliary_mini_batch_envs: int = 256
    auxiliary_num_epochs: int = 1


@configclass
class CR1LadderPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 250
    experiment_name = "cr1_wood_step_ladder"
    empirical_normalization = True
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 256],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        # IsaacLab releases that expose separate normalization flags require
        # them explicitly. Older releases keep empirical_normalization above.
        if hasattr(self.policy, "actor_obs_normalization"):
            self.policy.actor_obs_normalization = False
        if hasattr(self.policy, "critic_obs_normalization"):
            self.policy.critic_obs_normalization = False


@configclass
class CR1SSRVisualPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Opt-in recurrent asymmetric PPO configuration for Warp SSR vision."""

    use_ladder_ssr_runner: bool = True
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 250
    experiment_name = "cr1_wood_step_warp_ssr_visual"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = CR1SSRVisualPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CR1TransformerVisionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Upward-climbing TransformerVision asymmetric PPO runner."""

    use_transformer_vision_runner: bool = True
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 100
    experiment_name = "cr1_transformer_vision_ppo_up"
    empirical_normalization = False
    clip_observations: float | None = None
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = CR1TransformerVisionPPOAlgorithmCfg(
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
        max_grad_norm=0.1,
    )


@configclass
class CR1ThinRailsTransformerVisionPPORunnerCfg(
    CR1TransformerVisionPPORunnerCfg
):
    """Scratch actor/critic/visual training on the independent ThinRails task."""

    experiment_name = "cr1_transformer_vision_ppo_thin_rails_scratch"

    def __post_init__(self):
        super().__post_init__()
        self.save_interval = 400
        self.algorithm.visual_initialization = "scratch"
        self.algorithm.visual_checkpoint = ""
        # Match the humanoid visual PPO's KL adaptation, while retaining a
        # Ladder-specific upper bound verified against the earlier divergence.
        self.algorithm.schedule = "adaptive"
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.min_learning_rate = 1.0e-5
        self.algorithm.max_learning_rate = 1.0e-3
        self.algorithm.use_grad_scaler = True
        # Match the humanoid CE schedule: five passes over four recurrent
        # environment groups. The 256-env setting is the physical micro-batch
        # size used for gradient accumulation, not an extra optimizer step.
        self.algorithm.auxiliary_num_epochs = 5
        self.algorithm.auxiliary_mini_batch_envs = 256
        self.clip_actions = 10.0
        self.clip_observations = 100.0


@configclass
class CR1StopCommandTransformerVisionPPORunnerCfg(
    CR1ThinRailsTransformerVisionPPORunnerCfg
):
    """Separate run directory for the up/down/stop visual policy."""

    experiment_name = "cr1_transformer_vision_stop_command"


@configclass
class CR1StopCommandValidatedTaskTransformerVisionPPORunnerCfg(
    CR1StopCommandTransformerVisionPPORunnerCfg
):
    """Separate run directory for the validated-task visual policy."""

    experiment_name = "cr1_transformer_vision_zdw0824_stop_command_validated_task"


@configclass
class CR1UpDownSmoothOutputTransformerVisionPPORunnerCfg(
    CR1StopCommandValidatedTaskTransformerVisionPPORunnerCfg
):
    """Separate run directory for the up/down-only smooth-output policy."""

    experiment_name = "cr1_transformer_vision_zdw0824_up_down_smooth_output"


@configclass
class CR1UpDownTaskGoalTransformerVisionPPORunnerCfg(
    CR1UpDownSmoothOutputTransformerVisionPPORunnerCfg
):
    """Independent run for the supervised four-limb task-goal latent."""

    experiment_name = "cr1_transformer_vision_zdw0824_up_down_task_goal"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.visual_architecture = "task_goal"


@configclass
class CR1V4NoCommandTransformerVisionPPORunnerCfg(
    CR1ThinRailsTransformerVisionPPORunnerCfg
):
    """Separate scratch run directory for the v4 no-command policy."""

    experiment_name = "cr1_transformer_vision_v4_no_command"


@configclass
class CR1V4NoCommandNonVisualPPORunnerCfg(CR1LadderPPORunnerCfg):
    """Standard RSL-RL PPO baseline for the v4 no-command environment."""

    experiment_name = "cr1_nonvisual_v4_no_command"
    save_interval = 400


@configclass
class CR1ZDW0824NoCommandFullStateMiddleStandPPORunnerCfg(
    CR1LadderPPORunnerCfg
):
    """Separate run directory for the Stage-36 middle-standing reference."""

    experiment_name = "cr1_nonvisual_zdw0824_no_command_full_state_middle_stand"
    save_interval = 400


@configclass
class CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputPPORunnerCfg(
    CR1ZDW0824NoCommandFullStateMiddleStandPPORunnerCfg
):
    """Separate experiment for joint-acceleration-regularized Stage-36 training."""

    experiment_name = (
        "cr1_nonvisual_zdw0824_no_command_full_state_middle_stand_smooth_output"
    )


@configclass
class CR1ReferenceRateLimitedPPORunnerCfg(CR1LadderPPORunnerCfg):
    experiment_name = "cr1_wood_step_reference_rate_limited"


@configclass
class CR1ReferenceForceDiagnosticPPORunnerCfg(CR1LadderPPORunnerCfg):
    experiment_name = "cr1_wood_step_reference_force_diagnostic"


@configclass
class CR1ReferenceForceLimitedPPORunnerCfg(CR1LadderPPORunnerCfg):
    experiment_name = "cr1_wood_step_reference_force_limited"


@configclass
class CR1ReferenceForceLimitedStrictFrontSidePPORunnerCfg(
    CR1LadderPPORunnerCfg
):
    experiment_name = "cr1_wood_step_reference_force_limited_strict_front_side"


@configclass
class CR1ReferenceForceLimitedStrictFrontSideDeployObsPPORunnerCfg(
    CR1LadderPPORunnerCfg
):
    experiment_name = "cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs"


@configclass
class CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsPPORunnerCfg(
    CR1LadderPPORunnerCfg
):
    experiment_name = (
        "cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards"
    )


@configclass
class CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsNearTargetsPPORunnerCfg(
    CR1LadderPPORunnerCfg
):
    experiment_name = (
        "cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs_"
        "no_side_boards_near_targets"
    )


@configclass
class CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsPPORunnerCfg(
    CR1LadderPPORunnerCfg
):
    experiment_name = (
        "cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs_"
        "no_side_boards_original_targets"
    )


@configclass
class CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsSafeFrame0PPORunnerCfg(
    CR1LadderPPORunnerCfg
):
    experiment_name = (
        "cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs_"
        "no_side_boards_original_targets_safe_frame0"
    )


@configclass
class CR1ReferenceRailsProtectedSafeFrame0PPORunnerCfg(CR1LadderPPORunnerCfg):
    experiment_name = "cr1_wood_step_reference_rails_protected_safe_frame0"


@configclass
class CR1ReferenceRailsProtectedPerBodySafeFrame0PPORunnerCfg(
    CR1LadderPPORunnerCfg
):
    experiment_name = "cr1_wood_step_reference_rails_protected_per_body_safe_frame0"


@configclass
class CR1ReferenceRailsProtectedPerBodyThinRailsSafeFrame0PPORunnerCfg(
    CR1LadderPPORunnerCfg
):
    experiment_name = (
        "cr1_wood_step_reference_rails_protected_per_body_thin_rails_safe_frame0"
    )


@configclass
class CR1LadderGeometryDRPPORunnerCfg(CR1LadderPPORunnerCfg):
    experiment_name = "cr1_wood_step_ladder_geometry_dr"


@configclass
class CR1LadderGeometryDRRelaxedResetPPORunnerCfg(CR1LadderPPORunnerCfg):
    experiment_name = "cr1_wood_step_ladder_geometry_dr_relaxed_reset"


@configclass
class CR1LadderGeometryDRRelaxedResetNominal30PPORunnerCfg(CR1LadderPPORunnerCfg):
    experiment_name = "cr1_wood_step_ladder_geometry_dr_relaxed_reset_nominal30"


@configclass
class CR1LadderVisualPPORunnerCfg(CR1LadderPPORunnerCfg):
    experiment_name = "cr1_wood_step_ladder_visual"
    obs_groups = {
        "actor": ["visual_proprio", "visual_depth"],
        "critic": ["critic"],
    }
    policy = CR1LadderVisualActorCriticCfg()
