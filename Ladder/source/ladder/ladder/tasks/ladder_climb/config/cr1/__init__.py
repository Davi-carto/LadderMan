import gymnasium as gym

from . import (
    agents,
    dagger,
    wood_step_env_cfg,
    wood_step_geometry_dr_env_cfg,
    wood_step_geometry_dr_relaxed_reset_env_cfg,
    wood_step_geometry_dr_relaxed_reset_nominal30_env_cfg,
    wood_step_visual_env_cfg,
    wood_step_warp_visual_env_cfg,
    wood_step_transformer_vision_env_cfg,
    wood_step_transformer_vision_ppo_env_cfg,
    wood_step_transformer_vision_thin_rails_ppo_env_cfg,
    wood_step_transformer_vision_stop_command_env_cfg,
    wood_step_transformer_vision_stop_command_validated_task_env_cfg,
    wood_step_transformer_vision_up_down_smooth_output_env_cfg,
    wood_step_transformer_vision_up_down_task_goal_env_cfg,
    wood_step_end_to_end_visual_ppo_env_cfg,
    wood_step_transformer_vision_v4_no_command_env_cfg,
    wood_step_v4_no_command_nonvisual_env_cfg,
    wood_step_zdw0824_no_command_full_state_env_cfg,
    wood_step_reference_rate_limited_env_cfg,
    wood_step_reference_force_diagnostic_env_cfg,
    wood_step_reference_force_limited_env_cfg,
    wood_step_reference_force_limited_strict_front_side_env_cfg,
    wood_step_reference_force_limited_strict_front_side_deploy_obs_env_cfg,
    wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_env_cfg,
    wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_near_targets_env_cfg,
    wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_original_targets_env_cfg,
    wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_original_targets_safe_frame0_env_cfg,
    wood_step_reference_rails_protected_per_body_thin_rails_safe_frame0_env_cfg,
    wood_step_reference_rails_protected_per_body_safe_frame0_env_cfg,
    wood_step_reference_rails_protected_safe_frame0_env_cfg,
)


for task_suffix, env_class, agent_class in (
    ("ContactTask", "CR1ContactTaskAMPEnvCfg", "CR1ContactTaskAMPRunnerCfg"),
    ("ContactSequence", "CR1ContactSequenceAMPEnvCfg", "CR1ContactSequenceAMPRunnerCfg"),
    ("ContactSequence-Phased", "CR1PhasedContactSequenceAMPEnvCfg", "CR1PhasedContactSequenceAMPRunnerCfg"),
    ("ContactSequence-Phased-BodyGoal", "CR1PhasedBodyGoalContactSequenceAMPEnvCfg", "CR1PhasedBodyGoalContactSequenceAMPRunnerCfg"),
    ("ContactSequence-Phased-BodyGoal-FootSafety", "CR1FootSafetyContactSequenceAMPEnvCfg", "CR1FootSafetyContactSequenceAMPRunnerCfg"),
    ("ContactSequence-Phased-BodyGoal-FootSafety-Reference", "CR1ReferenceGuidedContactSequenceAMPEnvCfg", "CR1ReferenceGuidedContactSequenceAMPRunnerCfg"),
    ("ContactSequence-Phased-BodyGoal-FootSafety-Reference-PauseDecay", "CR1PauseDecayReferenceContactSequenceAMPEnvCfg", "CR1PauseDecayReferenceContactSequenceAMPRunnerCfg"),
    ("ContactSequence-Phased-BodyGoal-FootSafety-Reference-PauseDecay-Weight2x", "CR1DoubleReferencePauseDecayContactSequenceAMPEnvCfg", "CR1DoubleReferencePauseDecayContactSequenceAMPRunnerCfg"),
    ("ContactSequence-Phased-BodyGoal-FootSafety-Reference-PauseDecay-Reg2x-Energy", "CR1Reg2xEnergyPauseDecayContactSequenceAMPEnvCfg", "CR1Reg2xEnergyPauseDecayContactSequenceAMPRunnerCfg"),
    ("ContactSequence-Phased-BodyGoal-FootSafety-Reference-PauseDecay-RegRamp-Energy", "CR1RegRampEnergyPauseDecayContactSequenceAMPEnvCfg", "CR1RegRampEnergyPauseDecayContactSequenceAMPRunnerCfg"),
):
    gym.register(
        id=f"Ladder-Climb-CR1-ZDW0824-Visual-{task_suffix}-AMP-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.contact_task_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.end_to_end_visual_ppo_cfg:{agent_class}",
        },
    )


gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "ZDW0824-WarpTransformerVision-EndToEnd-Reconstruction-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wood_step_end_to_end_visual_ppo_env_cfg.CR1EndToEndVisualPPOEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.end_to_end_visual_ppo_cfg:CR1EndToEndVisualPPORunnerCfg"
        ),
    },
)


gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "ZDW0824-WarpTransformerVision-EndToEnd-OracleContactGoal-AMP-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_end_to_end_visual_ppo_env_cfg
            .CR1EndToEndVisualOracleContactGoalAMPPPOEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.end_to_end_visual_ppo_cfg:"
            "CR1EndToEndVisualOracleContactGoalAMPPPORunnerCfg"
        ),
    },
)


gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "ZDW0824-WarpTransformerVision-EndToEnd-OracleContactGoal-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_end_to_end_visual_ppo_env_cfg
            .CR1EndToEndVisualOracleContactGoalPPOEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.end_to_end_visual_ppo_cfg:"
            "CR1EndToEndVisualOracleContactGoalPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Ladder-Climb-CR1-ZDW0824-UpDown-Visual-Dagger-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1DaggerVisualEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.agent_cfg:CR1VisualDaggerPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-ZDW0824-UpDown-Pure-Visual-Dagger-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1PureVisualDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.agent_cfg:CR1PureVisualDaggerRunnerCfg"
        ),
    },
)


gym.register(
    id="Ladder-Climb-CR1-ZDW0824-UpDown-Visual-Dagger-Synced-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SyncedVisualDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.agent_cfg:"
            "CR1SyncedVisualDaggerPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-ZDW0824-UpDown-Pure-Visual-Dagger-Synced-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SyncedPureVisualDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.agent_cfg:"
            "CR1SyncedPureVisualDaggerRunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-ZDW0824-SameInput-Dagger-Control-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SameInputDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.same_input_agent_cfg:"
            "CR1SameInputDaggerRunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-ZDW0824-UpDown-Visual-MSE-Dagger-Synced-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SyncedVisualMseDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.visual_mse_agent_cfg:"
            "CR1VisualMseDaggerRunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-MSE-Dagger-Synced-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.recurrent_visual_mse_agent_cfg:"
            "CR1RecurrentVisualMseDaggerRunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-"
        "Privileged-MSE-Dagger-Synced-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SyncedPrivilegedRecurrentVisualMseDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.privileged_recurrent_visual_mse_agent_cfg:"
            "CR1PrivilegedRecurrentVisualMseDaggerRunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-Mixed-"
        "MSE-Dagger-Synced-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.recurrent_visual_mse_agent_cfg:"
            "CR1RecurrentVisualMixedMseDaggerRunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-Scheduled-"
        "MSE-Dagger-Synced-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.recurrent_visual_mse_agent_cfg:"
            "CR1RecurrentVisualScheduledMseDaggerRunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-Mixed-"
        "NoWarmup-MSE-Dagger-Synced-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.recurrent_visual_mse_agent_cfg:"
            "CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-LN-MSE-Dagger-Synced-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.recurrent_visual_ln_mse_agent_cfg:"
            "CR1RecurrentVisualLnMseDaggerRunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-PPO-Dagger-Synced-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.recurrent_visual_ppo_agent_cfg:"
            "CR1RecurrentVisualPpoDaggerRunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-ZDW0824-UpDown-TeacherOnly-Visual-BC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": dagger.CR1PureVisualDaggerEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{dagger.__name__}.agent_cfg:CR1TeacherOnlyVisualBCRunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wood_step_env_cfg.CR1WoodStepEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CR1LadderPPORunnerCfg",
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "ZDW0824-WarpTransformerVision-UpDown-TaskGoal-FrozenPolicy-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_transformer_vision_up_down_task_goal_env_cfg
            .CR1UpDownTaskGoalFrozenPolicyEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1UpDownTaskGoalTransformerVisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "ZDW0824-WarpTransformerVision-UpDown-TaskGoal-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_transformer_vision_up_down_task_goal_env_cfg
            .CR1UpDownTaskGoalTransformerVisionEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1UpDownTaskGoalTransformerVisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "WarpTransformerVision-V4-NoCommand-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_transformer_vision_v4_no_command_env_cfg
            .CR1V4NoCommandTransformerVisionEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1V4NoCommandTransformerVisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "V4-NoCommand-NonVisual-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_v4_no_command_nonvisual_env_cfg
            .CR1V4NoCommandNonVisualEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1V4NoCommandNonVisualPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "ZDW0824-NoCommand-FullState-MiddleStand-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_zdw0824_no_command_full_state_env_cfg
            .CR1ZDW0824NoCommandFullStateMiddleStandEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ZDW0824NoCommandFullStateMiddleStandPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "ZDW0824-NoCommand-FullState-MiddleStand-SmoothOutput-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_zdw0824_no_command_full_state_env_cfg
            .CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Reference-RateLimited-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wood_step_reference_rate_limited_env_cfg.CR1ReferenceRateLimitedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CR1ReferenceRateLimitedPPORunnerCfg",
    },
)

gym.register(
    id="Ladder-Climb-CR1-Reference-Force-Diagnostic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_force_diagnostic_env_cfg.CR1ReferenceForceDiagnosticEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceForceDiagnosticPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Reference-ForceLimited-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_force_limited_env_cfg.CR1ReferenceForceLimitedEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CR1ReferenceForceLimitedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_force_limited_strict_front_side_env_cfg
            .CR1ReferenceForceLimitedStrictFrontSideEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceForceLimitedStrictFrontSidePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_force_limited_strict_front_side_deploy_obs_env_cfg
            .CR1ReferenceForceLimitedStrictFrontSideDeployObsEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceForceLimitedStrictFrontSideDeployObsPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-NoSideBoards-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_env_cfg
            .CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-"
        "NoSideBoards-NearTargets-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_near_targets_env_cfg
            .CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsNearTargetsEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsNearTargetsPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-"
        "NoSideBoards-OriginalTargets-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_original_targets_env_cfg
            .CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-"
        "NoSideBoards-OriginalTargets-SafeFrame0-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards_original_targets_safe_frame0_env_cfg
            .CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsSafeFrame0EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceForceLimitedStrictFrontSideDeployObsNoSideBoardsOriginalTargetsSafeFrame0PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Reference-RailsProtected-SafeFrame0-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_rails_protected_safe_frame0_env_cfg
            .CR1ReferenceRailsProtectedSafeFrame0EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceRailsProtectedSafeFrame0PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Reference-RailsProtected-PerBody-SafeFrame0-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_rails_protected_per_body_safe_frame0_env_cfg
            .CR1ReferenceRailsProtectedPerBodySafeFrame0EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceRailsProtectedPerBodySafeFrame0PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-"
        "ThinRails-SafeFrame0-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_reference_rails_protected_per_body_thin_rails_safe_frame0_env_cfg
            .CR1ReferenceRailsProtectedPerBodyThinRailsSafeFrame0EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceRailsProtectedPerBodyThinRailsSafeFrame0PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Geometry-DR-Relaxed-Reset-Nominal30-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_geometry_dr_relaxed_reset_nominal30_env_cfg.CR1WoodStepGeometryDRRelaxedResetNominal30EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1LadderGeometryDRRelaxedResetNominal30PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Geometry-DR-Relaxed-Reset-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_geometry_dr_relaxed_reset_env_cfg.CR1WoodStepGeometryDRRelaxedResetEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CR1LadderGeometryDRRelaxedResetPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Geometry-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wood_step_geometry_dr_env_cfg.CR1WoodStepGeometryDREnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CR1LadderGeometryDRPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Geometry-DR-Visual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wood_step_visual_env_cfg.CR1WoodStepVisualEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CR1LadderVisualPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Geometry-DR-WarpVisual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wood_step_warp_visual_env_cfg.CR1WoodStepWarpVisualEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CR1LadderVisualPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-WarpTransformerVision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_transformer_vision_env_cfg.CR1FrozenPolicyTransformerVisionEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceForceLimitedStrictFrontSideDeployObsPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-WarpTransformerVision-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_transformer_vision_ppo_env_cfg.CR1TransformerVisionPPOEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CR1TransformerVisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "SafeFrame0-WarpTransformerVision-PPO-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_transformer_vision_thin_rails_ppo_env_cfg
            .CR1ThinRailsTransformerVisionPPOEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ThinRailsTransformerVisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "WarpTransformerVision-StopCommand-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_transformer_vision_stop_command_env_cfg
            .CR1StopCommandTransformerVisionEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1StopCommandTransformerVisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "ZDW0824-WarpTransformerVision-StopCommand-ValidatedTask-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_transformer_vision_stop_command_validated_task_env_cfg
            .CR1StopCommandValidatedTaskTransformerVisionEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1StopCommandValidatedTaskTransformerVisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "ZDW0824-WarpTransformerVision-UpDown-SmoothOutput-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_transformer_vision_up_down_smooth_output_env_cfg
            .CR1UpDownSmoothOutputTransformerVisionEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1UpDownSmoothOutputTransformerVisionPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
        "SafeFrame0-WarpTransformerVision-Preflight-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            wood_step_transformer_vision_thin_rails_ppo_env_cfg
            .CR1ThinRailsTransformerVisionPreflightEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "CR1ReferenceRailsProtectedPerBodyThinRailsSafeFrame0PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Ladder-Climb-CR1-Geometry-DR-WarpSSRVisual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wood_step_warp_visual_env_cfg.CR1WoodStepWarpVisualEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CR1SSRVisualPPORunnerCfg"
        ),
    },
)
