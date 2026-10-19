"""The independent visual PPO task preserves the retained blind-task MDP."""

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("isaaclab.sim", reason="Isaac Sim runtime is required for task config imports")

import gymnasium as gym

from ladder.tasks.ladder_climb.config.cr1.agents.end_to_end_visual_ppo_cfg import (
    CR1EndToEndVisualOracleContactGoalPPORunnerCfg,
    CR1EndToEndVisualPPORunnerCfg,
    CR1PhasedContactSequenceAMPRunnerCfg,
    CR1PhasedBodyGoalContactSequenceAMPRunnerCfg,
    CR1FootSafetyContactSequenceAMPRunnerCfg,
    CR1ReferenceGuidedContactSequenceAMPRunnerCfg,
    CR1PauseDecayReferenceContactSequenceAMPRunnerCfg,
    CR1DoubleReferencePauseDecayContactSequenceAMPRunnerCfg,
    CR1Reg2xEnergyPauseDecayContactSequenceAMPRunnerCfg,
    CR1RegRampEnergyPauseDecayContactSequenceAMPRunnerCfg,
)
from ladder.tasks.ladder_climb.config.cr1.contact_task_env_cfg import (
    CR1PhasedContactSequenceAMPEnvCfg,
    CR1PhasedBodyGoalContactSequenceAMPEnvCfg,
    CR1FootSafetyContactSequenceAMPEnvCfg,
    CR1ReferenceGuidedContactSequenceAMPEnvCfg,
    CR1PauseDecayReferenceContactSequenceAMPEnvCfg,
    CR1DoubleReferencePauseDecayContactSequenceAMPEnvCfg,
    CR1Reg2xEnergyPauseDecayContactSequenceAMPEnvCfg,
    CR1RegRampEnergyPauseDecayContactSequenceAMPEnvCfg,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_end_to_end_visual_ppo_env_cfg import (
    CR1EndToEndVisualOracleContactGoalPPOEnvCfg,
    CR1EndToEndVisualPPOEnvCfg,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg,
)


TASK = (
    "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
    "ZDW0824-WarpTransformerVision-EndToEnd-Reconstruction-v0"
)
ORACLE_TASK = (
    "Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-"
    "ZDW0824-WarpTransformerVision-EndToEnd-OracleContactGoal-v0"
)


def test_body_goal_experiment_only_adds_goal_and_shaping_to_phased_task():
    baseline = CR1PhasedContactSequenceAMPEnvCfg().to_dict()
    actual = CR1PhasedBodyGoalContactSequenceAMPEnvCfg().to_dict()
    assert not baseline["commands"]["motion"]["contact_task_body_goal_enabled"]
    assert actual["commands"]["motion"]["contact_task_body_goal_enabled"]
    actual["commands"]["motion"]["contact_task_body_goal_enabled"] = False
    body_reward = actual["rewards"].pop("body_progress")
    assert body_reward["weight"] == pytest.approx(0.5)
    assert body_reward["params"] == {"term": "body_progress"}
    assert actual == baseline
    old_runner = CR1PhasedContactSequenceAMPRunnerCfg()
    runner = CR1PhasedBodyGoalContactSequenceAMPRunnerCfg()
    assert runner.policy.use_contact_task_body_goal
    assert runner.algorithm.to_dict() == old_runner.algorithm.to_dict()
    task = gym.spec("Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-AMP-v0")
    assert task.kwargs["env_cfg_entry_point"].endswith(":CR1PhasedBodyGoalContactSequenceAMPEnvCfg")
    assert task.kwargs["rsl_rl_cfg_entry_point"].endswith(":CR1PhasedBodyGoalContactSequenceAMPRunnerCfg")


def test_foot_safety_only_adds_three_reward_rates_and_keeps_actor_contract():
    baseline = CR1PhasedBodyGoalContactSequenceAMPEnvCfg().to_dict()
    actual = CR1FootSafetyContactSequenceAMPEnvCfg().to_dict()
    assert actual["commands"]["motion"]["contact_task_foot_safety_enabled"]
    actual["commands"]["motion"]["contact_task_foot_safety_enabled"] = False
    for name, weight in (("foot_front_contact", -3.), ("foot_underside_contact", -3.), ("foot_swing_pose", -1.)):
        term = actual["rewards"].pop(name)
        assert term["weight"] == weight and term["params"] == {"term": name}
    assert actual == baseline
    old = CR1PhasedBodyGoalContactSequenceAMPRunnerCfg().to_dict()
    new = CR1FootSafetyContactSequenceAMPRunnerCfg().to_dict()
    new["experiment_name"] = old["experiment_name"]
    assert new == old
    task = gym.spec("Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-AMP-v0")
    assert task.kwargs["env_cfg_entry_point"].endswith(":CR1FootSafetyContactSequenceAMPEnvCfg")
    assert task.kwargs["rsl_rl_cfg_entry_point"].endswith(":CR1FootSafetyContactSequenceAMPRunnerCfg")


def test_reference_guidance_preserves_policy_amp_contacts_and_safety():
    baseline = CR1FootSafetyContactSequenceAMPEnvCfg().to_dict()
    actual = CR1ReferenceGuidedContactSequenceAMPEnvCfg().to_dict()
    assert actual["episode_length_s"] == 320.
    actual["episode_length_s"] = baseline["episode_length_s"]
    motion, old = actual["commands"]["motion"], baseline["commands"]["motion"]
    assert motion["contact_task_reference_guidance_enabled"]
    assert (motion["stable_anchor_failure_after_s"], motion["task_reference_duration_scale"],
            motion["task_deadline_extra_s"]) == (10., 2., 1.)
    for name in motion:
        if name.startswith("guidance_") or name in ("contact_task_reference_guidance_enabled",
                "stable_anchor_failure_after_s", "task_reference_duration_scale", "task_deadline_extra_s"):
            motion[name] = old[name]
    for name, weight in (("body_lower", .15), ("body_upper", .15), ("joint_lower", .25),
                         ("joint_upper", .15), ("swing_foot", .30), ("anchor_orientation", .10)):
        term = actual["rewards"].pop("reference_" + name)
        assert term["weight"] == weight
    assert actual == baseline
    before = CR1FootSafetyContactSequenceAMPRunnerCfg().to_dict()
    after = CR1ReferenceGuidedContactSequenceAMPRunnerCfg().to_dict()
    after["experiment_name"] = before["experiment_name"]
    assert after == before
    task = gym.spec("Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-Reference-AMP-v0")
    assert task.kwargs["env_cfg_entry_point"].endswith(":CR1ReferenceGuidedContactSequenceAMPEnvCfg")


def test_pause_decay_variant_preserves_all_other_environment_and_runner_settings():
    baseline = CR1ReferenceGuidedContactSequenceAMPEnvCfg().to_dict()
    actual = CR1PauseDecayReferenceContactSequenceAMPEnvCfg().to_dict()
    motion = actual["commands"]["motion"]
    assert motion["guidance_pause_decay_enabled"]
    assert (motion["guidance_pause_grace_s"], motion["guidance_pause_fade_s"]) == (.2, .5)
    motion["guidance_pause_decay_enabled"] = False
    assert actual == baseline
    before = CR1ReferenceGuidedContactSequenceAMPRunnerCfg().to_dict()
    after = CR1PauseDecayReferenceContactSequenceAMPRunnerCfg().to_dict()
    after["experiment_name"] = before["experiment_name"]
    assert after == before
    task = gym.spec("Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-Reference-PauseDecay-AMP-v0")
    assert task.kwargs["env_cfg_entry_point"].endswith(":CR1PauseDecayReferenceContactSequenceAMPEnvCfg")
    assert task.kwargs["rsl_rl_cfg_entry_point"].endswith(":CR1PauseDecayReferenceContactSequenceAMPRunnerCfg")


def test_double_reference_variant_only_scales_six_weights_and_keeps_resume_contract():
    baseline = CR1PauseDecayReferenceContactSequenceAMPEnvCfg().to_dict()
    actual = CR1DoubleReferencePauseDecayContactSequenceAMPEnvCfg().to_dict()
    for name, weight in (("body_lower", .30), ("body_upper", .30), ("joint_lower", .50),
                         ("joint_upper", .30), ("swing_foot", .60), ("anchor_orientation", .20)):
        key = "reference_" + name
        assert actual["rewards"][key]["weight"] == pytest.approx(weight)
        actual["rewards"][key]["weight"] /= 2.0
    assert actual == baseline
    assert CR1PauseDecayReferenceContactSequenceAMPEnvCfg().to_dict() == baseline
    before = CR1PauseDecayReferenceContactSequenceAMPRunnerCfg().to_dict()
    after = CR1DoubleReferencePauseDecayContactSequenceAMPRunnerCfg().to_dict()
    assert after["experiment_name"] != before["experiment_name"]
    after["experiment_name"] = before["experiment_name"]
    assert after == before
    task = gym.spec("Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-Reference-PauseDecay-Weight2x-AMP-v0")
    assert task.kwargs["env_cfg_entry_point"].endswith(":CR1DoubleReferencePauseDecayContactSequenceAMPEnvCfg")
    assert task.kwargs["rsl_rl_cfg_entry_point"].endswith(":CR1DoubleReferencePauseDecayContactSequenceAMPRunnerCfg")


def test_reg2x_energy_variant_only_changes_control_costs_and_keeps_resume_contract():
    baseline = CR1PauseDecayReferenceContactSequenceAMPEnvCfg().to_dict()
    actual = CR1Reg2xEnergyPauseDecayContactSequenceAMPEnvCfg().to_dict()
    for name, weight in (("action_rate_l2", -.2), ("action_second_difference_l2_lower", -.1),
                         ("joint_acc_l2_upper", -5e-7), ("joint_acc_l2_lower", -2e-6),
                         ("joint_torque_l2", -2e-5)):
        assert actual["rewards"][name]["weight"] == pytest.approx(weight)
        actual["rewards"][name]["weight"] /= 2.0
    power = actual["rewards"].pop("joint_mechanical_power")
    assert power["weight"] == pytest.approx(-5e-5)
    assert power["func"].endswith(":joint_mechanical_power_l1")
    assert power["params"]["asset_cfg"]["name"] == "robot"
    assert power["params"]["asset_cfg"]["joint_ids"] == slice(None)
    assert actual == baseline  # Includes reference, safety, contact, reset, and observations.
    assert CR1PauseDecayReferenceContactSequenceAMPEnvCfg().to_dict() == baseline
    before = CR1PauseDecayReferenceContactSequenceAMPRunnerCfg().to_dict()
    after = CR1Reg2xEnergyPauseDecayContactSequenceAMPRunnerCfg().to_dict()
    assert after["experiment_name"] != before["experiment_name"]
    after["experiment_name"] = before["experiment_name"]
    assert after == before  # AMP, optimizer, actor/critic, 500-step checkpoint interval.
    task = gym.spec("Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-Reference-PauseDecay-Reg2x-Energy-AMP-v0")
    assert task.kwargs["env_cfg_entry_point"].endswith(":CR1Reg2xEnergyPauseDecayContactSequenceAMPEnvCfg")
    assert task.kwargs["rsl_rl_cfg_entry_point"].endswith(":CR1Reg2xEnergyPauseDecayContactSequenceAMPRunnerCfg")


def test_reg_ramp_energy_variant_preserves_baseline_weights_and_only_enables_schedule():
    baseline = CR1PauseDecayReferenceContactSequenceAMPEnvCfg().to_dict()
    actual = CR1RegRampEnergyPauseDecayContactSequenceAMPEnvCfg().to_dict()
    power = actual["rewards"].pop("joint_mechanical_power")
    assert power["weight"] == -5e-5
    assert actual == baseline
    before = CR1PauseDecayReferenceContactSequenceAMPRunnerCfg().to_dict()
    after = CR1RegRampEnergyPauseDecayContactSequenceAMPRunnerCfg().to_dict()
    schedule = after["algorithm"]["regulation_reward_schedule"]
    assert (schedule["ramp_updates"], schedule["final_multiplier"]) == (2500, 2.5)
    assert len(schedule["base_weights"]) == 5
    for name, weight in schedule["base_weights"].items():
        assert weight == baseline["rewards"][name]["weight"]
    assert after["max_iterations"] == 4500
    after["max_iterations"] = before["max_iterations"]
    after["experiment_name"] = before["experiment_name"]
    after["algorithm"]["regulation_reward_schedule"] = None
    assert after == before
    assert CR1PauseDecayReferenceContactSequenceAMPRunnerCfg().to_dict() == before
    task = gym.spec("Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-Reference-PauseDecay-RegRamp-Energy-AMP-v0")
    assert task.kwargs["env_cfg_entry_point"].endswith(":CR1RegRampEnergyPauseDecayContactSequenceAMPEnvCfg")
    assert task.kwargs["rsl_rl_cfg_entry_point"].endswith(":CR1RegRampEnergyPauseDecayContactSequenceAMPRunnerCfg")


def _expand_scene_namespace(value):
    """Match InteractiveScene's in-place prim-path expansion before YAML saving."""

    if isinstance(value, dict):
        return {key: _expand_scene_namespace(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_scene_namespace(item) for item in value]
    if isinstance(value, str):
        return value.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    return value


def test_new_visual_environment_only_changes_observations_and_motion_annotation():
    blind = CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg()
    visual = CR1EndToEndVisualPPOEnvCfg()
    expected, actual = blind.to_dict(), visual.to_dict()
    assert actual["commands"]["motion"]["motion_file"] != expected["commands"]["motion"]["motion_file"]
    actual["observations"] = expected["observations"]
    actual["commands"]["motion"]["motion_file"] = expected["commands"]["motion"]["motion_file"]
    assert actual == expected
    assert not visual.commands.motion.motion_semantic_progress_enabled


def test_new_visual_environment_matches_the_retained_blind_training_yaml():
    snapshot = (
        Path(__file__).resolve().parents[2]
        / "logs/rsl_rl/cr1_nonvisual_zdw0824_no_command_full_state_middle_stand_smooth_output"
        / "2026-09-09_13-56-37_train10000_scratch_delay0or5ms_phys5ms_seed42/params/env.yaml"
    )
    if not snapshot.exists():
        pytest.skip("Retained local training snapshot is not distributed with the source")
    # Match train.py's launch-only visualization overrides. The saved snapshot
    # is emitted after InteractiveScene expands asset/sensor prim namespaces.
    cfg = CR1EndToEndVisualPPOEnvCfg()
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    # Like IsaacLab dump_yaml, serialize class_to_dict via yaml.dump. BaseLoader
    # then compares values without instantiating YAML Python tags.
    expected = yaml.load(snapshot.read_text(), Loader=yaml.BaseLoader)
    actual = _expand_scene_namespace(yaml.load(yaml.dump(cfg.to_dict()), Loader=yaml.BaseLoader))
    actual["observations"] = expected["observations"]
    actual["commands"]["motion"]["motion_file"] = expected["commands"]["motion"]["motion_file"]
    # These two launch-time choices do not alter per-environment MDP settings.
    actual["seed"] = expected["seed"]
    actual["scene"]["num_envs"] = expected["scene"]["num_envs"]
    assert actual == expected


def test_policy_only_history_has_no_reconstruction_target_groups():
    cfg = CR1EndToEndVisualPPOEnvCfg()
    obs = cfg.observations
    assert set(vars(obs)) == {
        "critic", "transformer_vision_depth", "transformer_vision_inputs",
    }
    assert obs.critic.to_dict() == CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg().observations.critic.to_dict()
    inputs = obs.transformer_vision_inputs
    params = inputs.proprio_history.params
    assert params["num_proprio_frames"] == 10
    assert params["num_depth_frames"] == 3
    assert params["proprio_interval_s"] == pytest.approx(0.1)
    assert params["depth_history_offsets_s"] == (0.0, 0.2, 0.4)
    assert params["history_buffer_capacity"] == 64
    assert params["include_climb_command"] is True
    assert params["climb_command_name"] == "motion"
    assert inputs.depth_updated.func.__name__ == "cr1_transformer_vision_depth_updated"
    assert not hasattr(obs, "transformer_vision_auxiliary_targets")
    assert not hasattr(obs, "transformer_vision_targets")
    depth = obs.transformer_vision_depth.image.params
    assert (depth["height"], depth["width"]) == (60, 80)
    assert depth["camera_body_name"] == "d435_head_link"
    assert depth["update_period"] == pytest.approx(1.0 / 30.0)
    assert depth["ladder_box_spec_set"] == "thin_rails_grounded"


def test_annotated_motion_preserves_all_blind_reference_arrays():
    blind = CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg()
    visual = CR1EndToEndVisualPPOEnvCfg()
    paths = [Path(cfg.commands.motion.motion_file) for cfg in (blind, visual)]
    if not all(path.exists() for path in paths):
        pytest.skip("Local reference motion files are required")
    with np.load(paths[0], allow_pickle=False) as original, np.load(paths[1], allow_pickle=False) as annotated:
        assert set(original.files) <= set(annotated.files)
        for key in original.files:
            np.testing.assert_array_equal(original[key], annotated[key], err_msg=key)


def test_runner_is_scratch_joint_ppo_and_uses_the_new_registration():
    cfg = CR1EndToEndVisualPPORunnerCfg()
    assert cfg.use_end_to_end_visual_ppo
    assert not getattr(cfg, "use_transformer_vision_runner", False)
    assert not getattr(cfg, "use_recurrent_visual_ppo_dagger_runner", False)
    assert not cfg.resume
    assert cfg.seed == 42
    assert (cfg.num_steps_per_env, cfg.max_iterations, cfg.save_interval) == (24, 10000, 500)
    assert cfg.policy.actor_hidden_dims == [512, 256, 128]
    assert cfg.policy.critic_hidden_dims == [512, 256, 128]
    assert cfg.policy.init_noise_std == 1.0
    assert cfg.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algorithm.sequence_micro_batch_envs == 64
    assert not hasattr(cfg.algorithm, "auxiliary_loss_coef")
    assert not hasattr(cfg.algorithm, "velocity_loss_coef")
    assert not hasattr(cfg.algorithm, "hand_loss_coef")
    assert not hasattr(cfg.algorithm, "foot_loss_coef")
    assert not hasattr(cfg.algorithm, "teacher_checkpoint")
    assert not hasattr(cfg.algorithm, "auxiliary_learning_rate")
    assert cfg.algorithm.class_name == "ladder.rsl_rl.algorithms.end_to_end_visual_ppo:EndToEndVisualPPO"
    assert gym.spec(TASK).kwargs["env_cfg_entry_point"] is CR1EndToEndVisualPPOEnvCfg
    assert gym.spec(TASK).kwargs["rsl_rl_cfg_entry_point"].endswith(":CR1EndToEndVisualPPORunnerCfg")


def test_oracle_environment_adds_diagnostic_observations_and_stable_anchor_gate():
    baseline = CR1EndToEndVisualPPOEnvCfg().to_dict()
    oracle_cfg = CR1EndToEndVisualOracleContactGoalPPOEnvCfg()
    oracle = oracle_cfg.to_dict()
    oracle_group = oracle["observations"].pop("oracle_contact_goal")
    oracle["terminations"] = baseline["terminations"]
    oracle_rewards = oracle["rewards"]
    oracle["rewards"] = baseline["rewards"]
    oracle["episode_length_s"] = baseline["episode_length_s"]
    oracle_ground_sensors = {
        name: oracle["scene"].pop(name)
        for name in (
            "left_foot_ground_contacts",
            "right_foot_ground_contacts",
        )
    }
    oracle_motion = oracle["commands"]["motion"]
    baseline_motion = baseline["commands"]["motion"]
    gate_fields = {
        name: oracle_motion[name]
        for name in (
            "stable_anchor_gating_enabled",
            "stable_anchor_confirmation_steps",
            "stable_anchor_relaxed_after_s",
            "stable_anchor_failure_after_s",
            "stable_anchor_root_height_error_threshold",
            "stable_anchor_root_orientation_threshold",
            "stable_anchor_root_linear_speed_threshold",
            "stable_anchor_root_angular_speed_threshold",
            "stable_anchor_danger_contact_force_threshold",
            "stable_anchor_limb_contact_sensor_names",
            "stable_anchor_ground_contact_sensor_names",
            "stable_anchor_rail_contact_sensor_names",
            "reset_by_stable_support_segments",
        )
    }
    for name in gate_fields:
        oracle_motion[name] = baseline_motion[name]
    assert oracle == baseline
    assert oracle_group["concatenate_terms"] is False
    assert oracle_group["enable_corruption"] is False
    for name, function_name in {
        "next_stable_goal_b": "motion_semantic_target_contact_pos_b",
        "actual_support_relation": "actual_support_relation",
        "current_expected_support_relation": "current_expected_support_relation",
        "next_stable_support_relation": "next_stable_support_relation",
    }.items():
        assert getattr(oracle_cfg.observations.oracle_contact_goal, name).func.__name__ == function_name
        assert oracle_group[name]["params"] == {"command_name": "motion"}
    assert gate_fields["stable_anchor_gating_enabled"] is True
    assert gate_fields["stable_anchor_confirmation_steps"] == 4
    assert gate_fields["stable_anchor_relaxed_after_s"] == pytest.approx(2.0)
    assert gate_fields["stable_anchor_failure_after_s"] == pytest.approx(5.5)
    assert len(gate_fields["stable_anchor_limb_contact_sensor_names"]) == 4
    assert len(gate_fields["stable_anchor_ground_contact_sensor_names"]) == 4
    assert gate_fields["stable_anchor_rail_contact_sensor_names"]
    assert gate_fields["reset_by_stable_support_segments"] is True
    assert set(oracle_ground_sensors) == {
        "left_foot_ground_contacts",
        "right_foot_ground_contacts",
    }
    assert oracle_cfg.terminations.body_local_pose is None
    assert oracle_cfg.terminations.required_contact is None
    assert oracle_cfg.terminations.stable_anchor_timeout.func.__name__ == "stable_anchor_timeout"
    assert oracle_cfg.episode_length_s == pytest.approx(210.0)
    for name in (
        "left_hand_contact_position",
        "right_hand_contact_position",
        "foot_contact_position",
    ):
        assert oracle_rewards[name]["func"].__name__ == (
            "stable_support_phase_conditioned_limb_tracking_exp"
        )
    assert oracle_rewards["next_stable_target_distance_progress"]["func"].__name__ == (
        "next_stable_target_distance_progress"
    )
    assert oracle_rewards["next_stable_target_distance_progress"]["weight"] == pytest.approx(2.0)
    assert oracle_rewards["next_stable_support_completion"]["func"].__name__ == (
        "stable_anchor_completion_bonus"
    )
    assert oracle_rewards["next_stable_support_completion"]["weight"] == pytest.approx(1.0)


def test_oracle_runner_flag_and_registration_are_isolated_from_baseline():
    baseline = CR1EndToEndVisualPPORunnerCfg()
    oracle = CR1EndToEndVisualOracleContactGoalPPORunnerCfg()
    assert oracle.use_end_to_end_visual_ppo
    assert oracle.policy.use_oracle_contact_goal is True
    assert not hasattr(baseline.policy, "use_oracle_contact_goal")
    assert oracle.experiment_name == (
        "cr1_zdw0824_end_to_end_visual_oracle_contact_goal_policy_only_ppo"
    )
    assert baseline.clip_actions is None
    assert oracle.clip_actions == 10.0
    assert oracle.obs_groups["actor"] == [
        *baseline.obs_groups["actor"],
        "oracle_contact_goal",
    ]
    assert oracle.obs_groups["critic"] == baseline.obs_groups["critic"] == ["critic"]
    assert oracle.algorithm.to_dict() == baseline.algorithm.to_dict()
    spec = gym.spec(ORACLE_TASK)
    assert spec.kwargs["env_cfg_entry_point"] is CR1EndToEndVisualOracleContactGoalPPOEnvCfg
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        ":CR1EndToEndVisualOracleContactGoalPPORunnerCfg"
    )
