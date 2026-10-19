"""The new DAgger tasks inherit the current blind MDP without changing old tasks."""

import pytest

pytest.importorskip("isaaclab.sim", reason="Isaac Sim runtime is required for task config imports")

import gymnasium as gym
from isaaclab.managers import RewardTermCfg

from ladder.tasks.ladder_climb.config.cr1.dagger.pure_env_cfg import CR1PureVisualDaggerEnvCfg
from ladder.tasks.ladder_climb.config.cr1.dagger.recurrent_visual_mse_agent_cfg import (
    CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg,
    CR1RecurrentVisualScheduledMseDaggerRunnerCfg,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.recurrent_visual_ppo_agent_cfg import (
    CR1RecurrentVisualPpoDaggerRunnerCfg,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.recurrent_visual_ln_mse_agent_cfg import (
    CR1RecurrentVisualLnMseDaggerRunnerCfg,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.same_input_env_cfg import (
    CR1SameInputDaggerEnvCfg,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.synced_env_cfg import (
    CR1SyncedPureVisualDaggerEnvCfg,
    CR1SyncedRecurrentVisualMseDaggerEnvCfg,
    CR1SyncedVisualDaggerEnvCfg,
    CR1SyncedVisualMseDaggerEnvCfg,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.privileged_env_cfg import (
    CR1SyncedPrivilegedRecurrentVisualMseDaggerEnvCfg,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.privileged_recurrent_visual_mse_agent_cfg import (
    CR1PrivilegedRecurrentVisualMseDaggerRunnerCfg,
)
from ladder.tasks.ladder_climb.config.cr1.dagger.visual_env_cfg import CR1DaggerVisualEnvCfg
from ladder.tasks.ladder_climb.config.cr1.wood_step_transformer_vision_stop_command_validated_task_env_cfg import (
    ZDW0824_VISUAL_STOP_COMMAND_MOTION,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg,
)


@pytest.mark.parametrize("env_cls", [
    CR1SyncedVisualDaggerEnvCfg,
    CR1SyncedPureVisualDaggerEnvCfg,
    CR1SyncedVisualMseDaggerEnvCfg,
    CR1SyncedRecurrentVisualMseDaggerEnvCfg,
])
def test_synced_task_only_changes_observations_motion_annotation_and_optional_rewards(env_cls):
    blind = CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg()
    visual = env_cls()
    assert isinstance(visual, CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg)
    assert visual.commands.motion.motion_file == str(ZDW0824_VISUAL_STOP_COMMAND_MOTION)

    expected = blind.to_dict()
    actual = visual.to_dict()
    actual["observations"] = expected["observations"]
    actual["commands"]["motion"]["motion_file"] = expected["commands"]["motion"]["motion_file"]
    if env_cls is CR1SyncedPureVisualDaggerEnvCfg:
        assert not any(isinstance(term, RewardTermCfg) for term in vars(visual.rewards).values())
        actual["rewards"] = expected["rewards"]
    assert actual == expected


@pytest.mark.parametrize("synced_cls,legacy_cls", [
    (CR1SyncedVisualDaggerEnvCfg, CR1DaggerVisualEnvCfg),
    (CR1SyncedPureVisualDaggerEnvCfg, CR1PureVisualDaggerEnvCfg),
    (CR1SyncedVisualMseDaggerEnvCfg, CR1PureVisualDaggerEnvCfg),
])
def test_synced_task_preserves_visual_contract_and_leaves_legacy_reset_unchanged(synced_cls, legacy_cls):
    legacy = legacy_cls()
    synced = synced_cls()
    assert synced.observations.to_dict() == legacy.observations.to_dict()
    assert synced.commands.motion.use_reset_valid_mask is False
    assert synced.commands.motion.fixed_start_probability == pytest.approx(0.1)
    assert synced.commands.motion.uniform_start_probability == pytest.approx(0.4)
    assert synced.commands.motion.fixed_start_pose_randomization_probability == pytest.approx(0.7)
    assert synced.terminations.root_pose.params["orientation_threshold"] == pytest.approx(0.8)
    assert legacy.commands.motion.use_reset_valid_mask is True
    assert legacy.commands.motion.fixed_start_probability == pytest.approx(0.2)
    assert legacy.commands.motion.uniform_start_probability == pytest.approx(0.35)
    assert legacy.commands.motion.fixed_start_pose_randomization_probability == 0.0
    assert legacy.terminations.root_pose.params["orientation_threshold"] == pytest.approx(1.2)


def test_same_input_control_only_replaces_the_blind_observation_container():
    blind = CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg()
    control = CR1SameInputDaggerEnvCfg()

    assert tuple(vars(control.observations)) == ("policy",)
    assert control.observations.policy.to_dict() == blind.observations.policy.to_dict()
    expected = blind.to_dict()
    actual = control.to_dict()
    actual["observations"] = expected["observations"]
    assert actual == expected

    spec = gym.spec("Ladder-Climb-CR1-ZDW0824-SameInput-Dagger-Control-v0")
    assert spec.kwargs["env_cfg_entry_point"] is CR1SameInputDaggerEnvCfg
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        ":CR1SameInputDaggerRunnerCfg"
    )


def test_recurrent_visual_contract_uses_command_in_25_frames_and_one_depth_frame():
    env = CR1SyncedRecurrentVisualMseDaggerEnvCfg()
    inputs = env.observations.transformer_vision_inputs
    params = inputs.proprio_history.params
    assert params["num_proprio_frames"] == 25
    assert params["num_depth_frames"] == 1
    assert params["include_climb_command"] is True
    assert params["climb_command_name"] == "motion"
    assert inputs.depth_updated.func.__name__ == "cr1_transformer_vision_depth_updated"


def test_privileged_recurrent_visual_contract_adds_oracle_actor_group():
    env = CR1SyncedPrivilegedRecurrentVisualMseDaggerEnvCfg()
    privileged = env.observations.privileged_actor
    assert tuple(vars(privileged)) == (
        "next_stable_target_pos_b",
        "actual_ladder_contact",
        "limb_linear_velocity_b",
    )
    runner = CR1PrivilegedRecurrentVisualMseDaggerRunnerCfg()
    assert runner.use_privileged_recurrent_visual_mse_dagger_runner
    assert runner.algorithm.bc_warmup_iterations == 1000
    assert runner.algorithm.teacher_rollout_schedule_start == 1000
    assert runner.algorithm.teacher_rollout_schedule_end == 5000
    assert runner.algorithm.teacher_rollout_schedule_initial_fraction == pytest.approx(1.0)
    assert runner.algorithm.teacher_rollout_schedule_final_fraction == pytest.approx(0.1)
    assert runner.algorithm.learning_rate == pytest.approx(1.0e-3)
    assert runner.save_interval == 500
    assert runner.algorithm.class_name.endswith(
        ":CR1PrivilegedRecurrentVisualMseDagger"
    )


def test_no_warmup_mixed_dagger_starts_with_fixed_25_percent_teacher_rollout():
    runner = CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg()
    assert runner.algorithm.bc_warmup_iterations == 0
    assert runner.algorithm.teacher_rollout_fraction == pytest.approx(0.25)
    assert runner.algorithm.learning_rate == pytest.approx(1.0e-3)


def test_scheduled_mse_dagger_uses_teacher_warmup_and_linear_rollout_schedule():
    runner = CR1RecurrentVisualScheduledMseDaggerRunnerCfg()
    assert runner.use_recurrent_visual_scheduled_mse_dagger_runner
    assert runner.use_recurrent_visual_mse_dagger_runner
    assert runner.algorithm.bc_warmup_iterations == 500
    assert runner.algorithm.teacher_rollout_schedule_start == 500
    assert runner.algorithm.teacher_rollout_schedule_end == 4000
    assert runner.algorithm.teacher_rollout_schedule_initial_fraction == pytest.approx(1.0)
    assert runner.algorithm.teacher_rollout_schedule_final_fraction == pytest.approx(0.1)
    assert runner.algorithm.teacher_rollout_fraction == pytest.approx(0.1)
    assert runner.save_interval == 500


def test_ln_mse_ablation_preserves_baseline_optimization():
    runner = CR1RecurrentVisualLnMseDaggerRunnerCfg()
    baseline = CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg()
    assert runner.use_recurrent_visual_ln_mse_dagger_runner
    assert not getattr(runner, "use_recurrent_visual_ppo_dagger_runner", False)
    actual = runner.algorithm.to_dict()
    expected = baseline.algorithm.to_dict()
    assert actual.pop("class_name").endswith(":CR1RecurrentVisualLnMseDagger")
    expected.pop("class_name")
    assert actual == expected
    assert runner.policy.to_dict() == baseline.policy.to_dict()
    assert runner.num_steps_per_env == baseline.num_steps_per_env
    assert runner.max_iterations == 2000
    assert runner.save_interval == 500


def test_ppo_dagger_enables_student_rl_without_changing_mse_baseline():
    runner = CR1RecurrentVisualPpoDaggerRunnerCfg()
    baseline = CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg()
    assert runner.use_recurrent_visual_ppo_dagger_runner
    assert runner.use_pure_dagger_visual_runner
    assert runner.use_recurrent_visual_mse_dagger_runner
    assert runner.algorithm.bc_warmup_iterations == 0
    assert runner.algorithm.teacher_rollout_fraction == pytest.approx(0.25)
    assert runner.algorithm.teacher_checkpoint == baseline.algorithm.teacher_checkpoint
    assert runner.policy.to_dict() == baseline.policy.to_dict()
    assert runner.algorithm.class_name.endswith(":CR1RecurrentVisualPpoDagger")
    assert runner.algorithm.value_loss_coef == 1.0
    assert runner.algorithm.use_clipped_value_loss
    assert runner.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert runner.algorithm.schedule == "adaptive"
    assert runner.algorithm.ppo_initial_coeff == pytest.approx(0.1)
    assert runner.algorithm.ppo_warmup_iterations == 500
    assert runner.algorithm.dagger_initial_coeff == 1.0
    assert runner.algorithm.dagger_final_coeff == pytest.approx(0.1)
    assert runner.algorithm.dagger_anneal_start == 2000
    assert runner.algorithm.dagger_anneal_iterations == 5000
    assert runner.max_iterations == 10000
    assert runner.save_interval == 500
    assert baseline.algorithm.value_loss_coef == 0.0
    assert baseline.algorithm.learning_rate == pytest.approx(1.0e-3)


@pytest.mark.parametrize("suffix,env_cls,runner_name", [
    ("Visual-Dagger-Synced", CR1SyncedVisualDaggerEnvCfg, "CR1SyncedVisualDaggerPPORunnerCfg"),
    ("Pure-Visual-Dagger-Synced", CR1SyncedPureVisualDaggerEnvCfg, "CR1SyncedPureVisualDaggerRunnerCfg"),
    ("Visual-Dagger", CR1DaggerVisualEnvCfg, "CR1VisualDaggerPPORunnerCfg"),
    ("Pure-Visual-Dagger", CR1PureVisualDaggerEnvCfg, "CR1PureVisualDaggerRunnerCfg"),
    (
        "Visual-MSE-Dagger-Synced",
        CR1SyncedVisualMseDaggerEnvCfg,
        "CR1VisualMseDaggerRunnerCfg",
    ),
    (
        "Recurrent-Visual-MSE-Dagger-Synced",
        CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "CR1RecurrentVisualMseDaggerRunnerCfg",
    ),
    (
        "Recurrent-Visual-Mixed-MSE-Dagger-Synced",
        CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "CR1RecurrentVisualMixedMseDaggerRunnerCfg",
    ),
    (
        "Recurrent-Visual-Mixed-NoWarmup-MSE-Dagger-Synced",
        CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg",
    ),
    (
        "Recurrent-Visual-Scheduled-MSE-Dagger-Synced",
        CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "CR1RecurrentVisualScheduledMseDaggerRunnerCfg",
    ),
    (
        "Recurrent-Visual-LN-MSE-Dagger-Synced",
        CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "CR1RecurrentVisualLnMseDaggerRunnerCfg",
    ),
    (
        "Recurrent-Visual-PPO-Dagger-Synced",
        CR1SyncedRecurrentVisualMseDaggerEnvCfg,
        "CR1RecurrentVisualPpoDaggerRunnerCfg",
    ),
    (
        "Recurrent-Visual-Privileged-MSE-Dagger-Synced",
        CR1SyncedPrivilegedRecurrentVisualMseDaggerEnvCfg,
        "CR1PrivilegedRecurrentVisualMseDaggerRunnerCfg",
    ),
])
def test_new_and_legacy_registrations_select_their_own_environment(suffix, env_cls, runner_name):
    spec = gym.spec(f"Ladder-Climb-CR1-ZDW0824-UpDown-{suffix}-v0")
    assert spec.kwargs["env_cfg_entry_point"] is env_cls
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(f":{runner_name}")
