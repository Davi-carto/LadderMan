import numpy as np
import pytest

pytest.importorskip(
    "isaaclab.sim", reason="Isaac Sim runtime is required for task config imports"
)

from ladder.tasks.ladder_climb.config.cr1.agents.rsl_rl_ppo_cfg import (
    CR1ThinRailsTransformerVisionPPORunnerCfg,
    CR1UpDownTaskGoalTransformerVisionPPORunnerCfg,
    CR1UpDownSmoothOutputTransformerVisionPPORunnerCfg,
    CR1StopCommandValidatedTaskTransformerVisionPPORunnerCfg,
    CR1TransformerVisionPPORunnerCfg,
    CR1V4NoCommandNonVisualPPORunnerCfg,
    CR1V4NoCommandTransformerVisionPPORunnerCfg,
    CR1ZDW0824NoCommandFullStateMiddleStandPPORunnerCfg,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_geometry_dr_env_cfg import (
    STOP_COMMAND_GROUNDED_THIN_RAIL_BOX_SPECS,
    THIN_RAIL_WOOD_STEP_BOX_SPECS,
    WOOD_STEP_BOX_SPECS,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_transformer_vision_ppo_env_cfg import (
    CR1TransformerVisionPPOEnvCfg,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_transformer_vision_thin_rails_ppo_env_cfg import (
    CR1ThinRailsTransformerVisionPreflightEnvCfg,
    CR1ThinRailsTransformerVisionPPOEnvCfg,
    THIN_RAIL_BOX_SPEC_SET,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_transformer_vision_stop_command_env_cfg import (
    CR1StopCommandTransformerVisionEnvCfg,
    V4_CONTACT_TARGET_OFFSETS,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_transformer_vision_stop_command_validated_task_env_cfg import (
    CR1StopCommandValidatedTaskTransformerVisionEnvCfg,
    ZDW0824_TOP_DEFAULT_STAND_START_FRAME,
    ZDW0824_VISUAL_STOP_COMMAND_MOTION,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_transformer_vision_up_down_smooth_output_env_cfg import (
    CR1UpDownSmoothOutputTransformerVisionEnvCfg,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_transformer_vision_up_down_task_goal_env_cfg import (
    CR1UpDownTaskGoalTransformerVisionEnvCfg,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_transformer_vision_v4_no_command_env_cfg import (
    CR1V4NoCommandTransformerVisionEnvCfg,
    V4_NO_COMMAND_FILTERED_MOTION,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_v4_no_command_nonvisual_env_cfg import (
    CR1V4NoCommandNonVisualEnvCfg,
)
from ladder.tasks.ladder_climb.config.cr1.wood_step_zdw0824_no_command_full_state_env_cfg import (
    CR1ZDW0824NoCommandFullStateMiddleStandEnvCfg,
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg,
    FOOT_CONTACT_WEIGHT,
    FOOT_REFERENCE_TEMPERATURE,
    FOOT_TARGET_TEMPERATURE,
    FOOT_WRONG_RUNG_FORCE_SCALE_N,
    FOOT_WRONG_RUNG_FORCE_SOFT_N,
    FOOT_WRONG_RUNG_FORCE_WEIGHT,
    FOOT_WRONG_RUNG_MAX_EXCESS,
    HAND_CONTACT_WEIGHT,
    HAND_REFERENCE_TEMPERATURE,
    HAND_TARGET_TEMPERATURE,
    LOWER_JOINT_BROAD_TEMPERATURE,
    MOTION_BODY_LIN_VEL_STD,
    MOTION_BODY_POS_LOWER_STD,
    MOTION_BODY_POS_UPPER_STD,
    UPPER_JOINT_BROAD_TEMPERATURE,
    ZDW0824_CONTACT_TARGET_OFFSETS,
    ZDW0824_LADDER_BOX_SPEC_SET,
    ZDW0824_MIDDLE_STAND_MOTION,
)
from ladder.tasks.ladder_climb.mdp.observations import _cr1_wood_step_box_specs


def test_thin_rails_visual_task_uses_one_geometry_spec_set_everywhere():
    cfg = CR1ThinRailsTransformerVisionPPOEnvCfg()
    expected = {spec.name: spec for spec in THIN_RAIL_WOOD_STEP_BOX_SPECS}
    physical = cfg.scene.ladder.rigid_objects

    assert set(physical) == set(expected)
    for name, spec in expected.items():
        assert physical[name].spawn.size == spec.size
        assert physical[name].init_state.pos == spec.position
        assert physical[name].init_state.rot == spec.quaternion

    assert (
        cfg.observations.transformer_vision_depth.image.params[
            "ladder_box_spec_set"
        ]
        == THIN_RAIL_BOX_SPEC_SET
    )
    for term_name in ("relative_height", "hit_valid", "hit_class"):
        term = getattr(cfg.observations.transformer_vision_targets, term_name)
        assert term.params["ladder_box_spec_set"] == THIN_RAIL_BOX_SPEC_SET
    assert _cr1_wood_step_box_specs(THIN_RAIL_BOX_SPEC_SET) is THIN_RAIL_WOOD_STEP_BOX_SPECS


def test_new_critic_adds_only_requested_120_plus_25_plus_4_fields():
    cfg = CR1ThinRailsTransformerVisionPPOEnvCfg()
    critic = cfg.observations.critic

    assert critic.joint_vel.func.__name__ == "joint_vel"
    assert critic.joint_vel.scale == pytest.approx(0.1)
    assert critic.ladder_params.params == {"command_name": "motion"}
    assert critic.limb_heightmaps.params["output"] == "relative_height_flat"
    assert critic.limb_heightmaps.params["rows"] == 10
    assert critic.limb_heightmaps.params["columns"] == 3
    assert critic.limb_heightmaps.params["ladder_box_spec_set"] == THIN_RAIL_BOX_SPEC_SET
    assert 631 + 25 + 4 + 4 * 10 * 3 == 780


def test_new_task_preserves_latest_thin_rail_rewards_and_scratch_initialization():
    env_cfg = CR1ThinRailsTransformerVisionPPOEnvCfg()
    runner_cfg = CR1ThinRailsTransformerVisionPPORunnerCfg()

    assert env_cfg.rewards.left_rail_any_body_contact.weight == pytest.approx(-1.0)
    assert env_cfg.rewards.right_rail_any_body_contact.weight == pytest.approx(-1.0)
    assert env_cfg.rewards.left_rail_force_excess.weight == pytest.approx(-0.25)
    assert env_cfg.rewards.right_rail_force_excess.weight == pytest.approx(-0.25)
    assert env_cfg.rewards.left_hand_missing_required_step_contact.weight == pytest.approx(-0.4)
    assert env_cfg.rewards.left_hand_step_underside_contact.weight == pytest.approx(-0.6)
    assert runner_cfg.algorithm.visual_initialization == "scratch"
    assert runner_cfg.algorithm.visual_checkpoint == ""
    assert runner_cfg.save_interval == 400
    assert runner_cfg.algorithm.schedule == "adaptive"
    assert runner_cfg.algorithm.learning_rate == pytest.approx(5.0e-4)
    assert runner_cfg.algorithm.min_learning_rate == pytest.approx(1.0e-5)
    assert runner_cfg.algorithm.max_learning_rate == pytest.approx(1.0e-3)
    assert runner_cfg.algorithm.auxiliary_learning_rate == pytest.approx(5.0e-4)
    assert runner_cfg.algorithm.auxiliary_num_epochs == 5
    assert runner_cfg.algorithm.auxiliary_mini_batch_envs == 256
    assert runner_cfg.algorithm.max_grad_norm == pytest.approx(0.1)
    assert runner_cfg.algorithm.use_grad_scaler is True
    assert runner_cfg.clip_actions == pytest.approx(10.0)
    assert runner_cfg.clip_observations == pytest.approx(100.0)
    assert runner_cfg.policy.actor_hidden_dims == [512, 256, 128]


def test_new_task_enables_only_thin_rails_input_and_dynamics_randomization():
    cfg = CR1ThinRailsTransformerVisionPPOEnvCfg()
    history = cfg.observations.transformer_vision_inputs.proprio_history.params
    depth = cfg.observations.transformer_vision_depth.image.params

    assert history["depth_mean"] == pytest.approx(1.0)
    assert history["depth_scale"] == pytest.approx(1.0)
    assert history["add_proprio_noise"] is True
    assert history["root_angular_velocity_noise"] == pytest.approx(0.5)
    assert history["orientation_noise_rad"] == pytest.approx(0.05)
    assert history["joint_position_noise_normalized"] == pytest.approx(0.01)
    assert history["joint_velocity_noise"] == pytest.approx(1.5)
    assert depth["horizontal_fov_randomization_range_deg"] == (-1.0, 1.0)
    assert depth["vertical_fov_randomization_range_deg"] == (-1.0, 1.0)
    assert cfg.actions.joint_pos.min_control_delay == 0
    assert cfg.actions.joint_pos.max_control_delay == 1
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.decimation == cfg.sim.render_interval == 4
    assert cfg.sim.dt * cfg.decimation == pytest.approx(0.02)
    assert not hasattr(cfg.actions.joint_pos, "filter_alpha")
    assert not hasattr(cfg.actions.joint_pos, "startup_ramp_s")
    for term in (cfg.events.motor_strength, cfg.events.motor_strength_interval):
        assert term.func.__name__ == "randomize_actuator_strength_and_gains"
        assert term.params["motor_strength_range"] == (0.8, 1.2)
        assert term.params["stiffness_factor_range"] == (0.9, 1.1)
        assert term.params["damping_factor_range"] == (0.9, 1.1)
    assert cfg.events.motor_strength.mode == "reset"
    assert cfg.events.motor_strength_interval.mode == "interval"
    assert cfg.events.motor_strength_interval.interval_range_s == (8.0, 8.0)
    assert cfg.events.motor_strength_interval.is_global_time is False


def test_existing_visual_task_keeps_standard_geometry_and_checkpoint_initialization():
    old_cfg = CR1TransformerVisionPPOEnvCfg()
    old_runner_cfg = CR1TransformerVisionPPORunnerCfg()

    assert "ladder_box_spec_set" not in (
        old_cfg.observations.transformer_vision_depth.image.params
    )
    assert _cr1_wood_step_box_specs("standard") is WOOD_STEP_BOX_SPECS
    for term_name in ("relative_height", "hit_valid", "hit_class"):
        term = getattr(old_cfg.observations.transformer_vision_targets, term_name)
        assert "ladder_box_spec_set" not in term.params
    assert old_runner_cfg.policy.actor_hidden_dims == [512, 256, 128]
    assert old_runner_cfg.algorithm.visual_initialization == "checkpoint"
    assert old_runner_cfg.algorithm.schedule == "adaptive"
    old_history = old_cfg.observations.transformer_vision_inputs.proprio_history.params
    assert "depth_mean" not in old_history
    assert "add_proprio_noise" not in old_history
    assert "horizontal_fov_randomization_range_deg" not in (
        old_cfg.observations.transformer_vision_depth.image.params
    )
    assert old_cfg.actions.joint_pos.min_control_delay == 0
    assert old_cfg.actions.joint_pos.max_control_delay == 0
    assert old_cfg.sim.dt == pytest.approx(0.005)
    assert old_cfg.decimation == 4


def test_preflight_adds_frozen_policy_observation_without_changing_training_task():
    training_cfg = CR1ThinRailsTransformerVisionPPOEnvCfg()
    preflight_cfg = CR1ThinRailsTransformerVisionPreflightEnvCfg()

    assert not hasattr(training_cfg.observations, "policy")
    assert preflight_cfg.observations.policy.concatenate_terms is True
    assert preflight_cfg.observations.critic.limb_heightmaps.params[
        "ladder_box_spec_set"
    ] == THIN_RAIL_BOX_SPEC_SET


def test_v4_tasks_use_the_contact_targets_embedded_in_the_reference():
    for cfg in (
        CR1StopCommandTransformerVisionEnvCfg(),
        CR1V4NoCommandTransformerVisionEnvCfg(),
    ):
        assert cfg.commands.motion.limb_target_offsets == V4_CONTACT_TARGET_OFFSETS


def test_v4_no_command_task_removes_every_command_signal_and_stop_episode():
    cfg = CR1V4NoCommandTransformerVisionEnvCfg()
    runner_cfg = CR1V4NoCommandTransformerVisionPPORunnerCfg()
    history = cfg.observations.transformer_vision_inputs.proprio_history.params
    next_proprio = (
        cfg.observations.transformer_vision_auxiliary_targets.next_proprio.params
    )

    assert history["include_climb_command"] is False
    assert "climb_command_name" not in history
    assert next_proprio["include_climb_command"] is False
    assert "climb_command_name" not in next_proprio
    assert not hasattr(cfg.observations.critic, "climb_command")
    assert cfg.commands.motion.stop_command_training_enabled is False
    assert cfg.commands.motion.stop_hold_reset_probability == pytest.approx(0.0)
    assert cfg.commands.motion.stop_seek_episode_probability == pytest.approx(0.0)
    assert cfg.commands.motion.motion_file == str(V4_NO_COMMAND_FILTERED_MOTION)
    assert runner_cfg.experiment_name == "cr1_transformer_vision_v4_no_command"


def test_v4_no_command_nonvisual_task_keeps_environment_without_visual_groups():
    cfg = CR1V4NoCommandNonVisualEnvCfg()
    visual_cfg = CR1V4NoCommandTransformerVisionEnvCfg()
    runner_cfg = CR1V4NoCommandNonVisualPPORunnerCfg()

    assert cfg.commands.motion.motion_file == str(V4_NO_COMMAND_FILTERED_MOTION)
    assert cfg.commands.motion.ladder_geometry == visual_cfg.commands.motion.ladder_geometry
    assert cfg.commands.motion.limb_target_offsets == V4_CONTACT_TARGET_OFFSETS
    assert cfg.commands.motion.stop_command_training_enabled is False
    assert cfg.commands.motion.stop_hold_reset_probability == 0.0
    assert cfg.commands.motion.stop_seek_episode_probability == 0.0
    assert hasattr(cfg.observations, "policy")
    assert hasattr(cfg.observations, "critic")
    assert not hasattr(cfg.observations, "transformer_vision_depth")
    assert not hasattr(cfg.observations, "transformer_vision_inputs")
    assert set(cfg.scene.ladder.rigid_objects) == set(
        visual_cfg.scene.ladder.rigid_objects
    )
    assert cfg.actions.joint_pos.min_control_delay == 0
    assert cfg.actions.joint_pos.max_control_delay == 1
    assert runner_cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert not hasattr(runner_cfg, "use_transformer_vision_runner")
    assert runner_cfg.experiment_name == "cr1_nonvisual_v4_no_command"
    assert runner_cfg.save_interval == 400


@pytest.mark.parametrize("env_cls", [
    CR1ZDW0824NoCommandFullStateMiddleStandEnvCfg,
    CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg,
    CR1UpDownSmoothOutputTransformerVisionEnvCfg,
    CR1UpDownTaskGoalTransformerVisionEnvCfg,
])
def test_current_blind_and_visual_tasks_use_two_point_physics_delay_at_50hz(env_cls):
    cfg = env_cls()
    assert cfg.sim.dt == pytest.approx(0.005)
    assert cfg.decimation == cfg.sim.render_interval == 4
    assert cfg.sim.dt * cfg.decimation == pytest.approx(0.02)
    assert cfg.actions.joint_pos.min_control_delay == 0
    assert cfg.actions.joint_pos.max_control_delay == 1
    assert cfg.actions.joint_pos.max_target_velocity == pytest.approx(6.0)


def test_zdw0824_middle_stand_tasks_keep_full_state_and_phase_conditioned_limb_rewards():
    cfg = CR1ZDW0824NoCommandFullStateMiddleStandEnvCfg()
    smooth_cfg = CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg()
    runner_cfg = CR1ZDW0824NoCommandFullStateMiddleStandPPORunnerCfg()

    assert type(cfg.observations.policy) is type(cfg.observations.critic)
    assert cfg.observations.policy.to_dict() == cfg.observations.critic.to_dict()
    assert cfg.observations.policy.limb_heightmaps.params[
        "ladder_box_spec_set"
    ] == ZDW0824_LADDER_BOX_SPEC_SET
    assert cfg.commands.motion.motion_file == str(ZDW0824_MIDDLE_STAND_MOTION)
    assert cfg.commands.motion.limb_target_offsets == ZDW0824_CONTACT_TARGET_OFFSETS
    assert cfg.commands.motion.stop_command_training_enabled is False
    assert cfg.commands.motion.stop_hold_reset_probability == 0.0
    assert cfg.commands.motion.stop_seek_episode_probability == 0.0
    for task_cfg in (cfg, smooth_cfg):
        assert task_cfg.rewards.foot_contact_position.weight == pytest.approx(
            FOOT_CONTACT_WEIGHT
        )
        assert "active_weight" not in task_cfg.rewards.foot_contact_position.params
        assert task_cfg.rewards.foot_contact_position.params[
            "target_temperature"
        ] == pytest.approx(FOOT_TARGET_TEMPERATURE)
        assert task_cfg.rewards.foot_contact_position.params[
            "reference_temperature"
        ] == pytest.approx(FOOT_REFERENCE_TEMPERATURE)
        assert task_cfg.rewards.foot_contact_position.func.__name__ == (
            "ladder_phase_conditioned_limb_tracking_exp"
        )
        assert task_cfg.rewards.foot_contact_position.params[
            "reference_body_names"
        ] == ["left_ankle_x_link", "right_ankle_x_link"]
        assert not hasattr(task_cfg.rewards, "reference_motion_history_huber")
    assert cfg.rewards.motion_body_lin_vel.params["std"] == pytest.approx(
        MOTION_BODY_LIN_VEL_STD
    )
    assert MOTION_BODY_LIN_VEL_STD == pytest.approx(0.75)
    assert cfg.rewards.motion_body_pos_asym.params[
        "lower_body_std"
    ] == pytest.approx(MOTION_BODY_POS_LOWER_STD)
    assert cfg.rewards.motion_body_pos_asym.params[
        "upper_body_std"
    ] == pytest.approx(MOTION_BODY_POS_UPPER_STD)
    assert cfg.rewards.motion_joint_pos_upper.weight == pytest.approx(0.25)
    assert cfg.rewards.motion_joint_pos_upper_broad.weight == pytest.approx(0.25)
    assert cfg.rewards.motion_joint_pos_upper_broad.params[
        "temperature"
    ] == pytest.approx(UPPER_JOINT_BROAD_TEMPERATURE)
    assert cfg.rewards.motion_joint_pos_lower.weight == pytest.approx(0.5)
    assert cfg.rewards.motion_joint_pos_lower_broad.weight == pytest.approx(0.5)
    assert cfg.rewards.motion_joint_pos_lower_broad.params[
        "temperature"
    ] == pytest.approx(LOWER_JOINT_BROAD_TEMPERATURE)
    for side in ("left", "right"):
        hand_contact = getattr(
            cfg.rewards, f"{side}_hand_contact_position"
        )
        assert hand_contact.weight == pytest.approx(HAND_CONTACT_WEIGHT)
        assert "active_weight" not in hand_contact.params
        assert hand_contact.params["target_temperature"] == pytest.approx(
            HAND_TARGET_TEMPERATURE
        )
        assert hand_contact.params["reference_temperature"] == pytest.approx(
            HAND_REFERENCE_TEMPERATURE
        )
        assert hand_contact.func.__name__ == (
            "ladder_phase_conditioned_limb_tracking_exp"
        )
        smooth_hand_contact = getattr(
            smooth_cfg.rewards, f"{side}_hand_contact_position"
        )
        assert smooth_hand_contact.func.__name__ == (
            "ladder_phase_conditioned_limb_tracking_exp"
        )
        wrong_rung = getattr(
            cfg.rewards, f"{side}_foot_wrong_rung_force_excess"
        )
        assert wrong_rung.weight == pytest.approx(FOOT_WRONG_RUNG_FORCE_WEIGHT)
        assert wrong_rung.params["force_soft"] == pytest.approx(
            FOOT_WRONG_RUNG_FORCE_SOFT_N
        )
        assert wrong_rung.params["force_scale"] == pytest.approx(
            FOOT_WRONG_RUNG_FORCE_SCALE_N
        )
        assert wrong_rung.params["surface"] == "all"
        assert wrong_rung.params["limb_name"] == f"{side}_foot"
        assert wrong_rung.params["rung_scope"] == "non_target"
        assert wrong_rung.params["max_excess"] == pytest.approx(
            FOOT_WRONG_RUNG_MAX_EXCESS
        )
    assert not hasattr(cfg.observations, "transformer_vision_depth")
    assert not hasattr(cfg.observations, "transformer_vision_inputs")
    assert smooth_cfg.rewards.joint_acc_l2_upper.weight == pytest.approx(-2.5e-7)
    assert smooth_cfg.rewards.joint_acc_l2_lower.weight == pytest.approx(-1.0e-6)
    assert smooth_cfg.rewards.motion_joint_vel_lower_robust.weight == pytest.approx(-0.05)
    assert smooth_cfg.rewards.motion_joint_vel_lower_robust.params["deadband"] == pytest.approx(0.15)
    assert smooth_cfg.rewards.motion_joint_vel_lower_robust.params["delta"] == pytest.approx(0.5)
    assert smooth_cfg.rewards.action_second_difference_l2_lower.weight == pytest.approx(-0.05)
    assert runner_cfg.experiment_name == (
        "cr1_nonvisual_zdw0824_no_command_full_state_middle_stand"
    )


def test_zdw0824_middle_stand_task_uses_two_anchor_all_frame_reset_distribution():
    cfg = CR1ZDW0824NoCommandFullStateMiddleStandEnvCfg()
    runner_cfg = CR1ZDW0824NoCommandFullStateMiddleStandPPORunnerCfg()

    assert cfg.commands.motion.motion_file == str(ZDW0824_MIDDLE_STAND_MOTION)
    assert cfg.commands.motion.use_reset_valid_mask is False
    assert cfg.commands.motion.fixed_start_frame == 0
    assert cfg.commands.motion.fixed_start_probability == pytest.approx(0.10)
    assert cfg.commands.motion.fixed_start_pose_randomization_probability == pytest.approx(0.70)
    assert cfg.commands.motion.fixed_start_pose_range == {
        "x": (-0.03, 0.03),
        "y": (-0.03, 0.03),
        "z": (0.0, 0.005),
        "roll": (-0.01, 0.01),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.05, 0.05),
    }
    assert cfg.commands.motion.secondary_fixed_start_frame == 1500
    assert cfg.commands.motion.secondary_fixed_start_probability == pytest.approx(0.10)
    assert cfg.commands.motion.secondary_fixed_start_requires_nominal_ladder is True
    assert cfg.commands.motion.uniform_start_probability == pytest.approx(0.40)
    assert (
        1.0
        - cfg.commands.motion.fixed_start_probability
        - cfg.commands.motion.secondary_fixed_start_probability
        - cfg.commands.motion.uniform_start_probability
    ) == pytest.approx(0.40)
    assert cfg.terminations.root_pose.params["orientation_threshold"] == pytest.approx(
        0.8
    )
    assert set(cfg.scene.ladder.rigid_objects) == {
        spec.name for spec in STOP_COMMAND_GROUNDED_THIN_RAIL_BOX_SPECS
    }
    assert runner_cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert runner_cfg.experiment_name == (
        "cr1_nonvisual_zdw0824_no_command_full_state_middle_stand"
    )


def test_stop_command_validated_task_preserves_visual_observations_and_commands():
    base_cfg = CR1StopCommandTransformerVisionEnvCfg()
    verified_cfg = CR1ZDW0824NoCommandFullStateMiddleStandEnvCfg()
    cfg = CR1StopCommandValidatedTaskTransformerVisionEnvCfg()

    assert type(cfg.observations) is type(base_cfg.observations)
    assert cfg.observations.to_dict() == base_cfg.observations.to_dict()
    assert cfg.commands.motion.stop_command_training_enabled is True
    assert (
        cfg.commands.motion.stop_hold_reset_probability
        == base_cfg.commands.motion.stop_hold_reset_probability
    )
    assert (
        cfg.commands.motion.stop_seek_episode_probability
        == base_cfg.commands.motion.stop_seek_episode_probability
    )
    assert (
        cfg.commands.motion.stop_hold_time_range_s
        == base_cfg.commands.motion.stop_hold_time_range_s
    )
    assert (
        cfg.commands.motion.ladder_geometry == verified_cfg.commands.motion.ladder_geometry
    )
    assert cfg.commands.motion.limb_target_offsets == ZDW0824_CONTACT_TARGET_OFFSETS


def test_stop_command_validated_task_uses_validated_mdp_and_reset_mixture():
    verified_cfg = CR1ZDW0824NoCommandFullStateMiddleStandEnvCfg()
    cfg = CR1StopCommandValidatedTaskTransformerVisionEnvCfg()
    runner_cfg = CR1StopCommandValidatedTaskTransformerVisionPPORunnerCfg()

    assert cfg.episode_length_s == pytest.approx(90.0)
    assert cfg.rewards.to_dict() == verified_cfg.rewards.to_dict()
    assert cfg.actions.to_dict() == verified_cfg.actions.to_dict()
    assert cfg.events.to_dict() == verified_cfg.events.to_dict()
    for term_name in (
        "time_out",
        "motion_complete",
        "body_local_pose",
        "required_contact",
    ):
        assert getattr(cfg.terminations, term_name).to_dict() == getattr(
            verified_cfg.terminations, term_name
        ).to_dict()
    assert cfg.terminations.root_pose.params["orientation_threshold"] == pytest.approx(
        1.2
    )
    assert verified_cfg.terminations.root_pose.params[
        "orientation_threshold"
    ] == pytest.approx(0.8)
    assert cfg.commands.motion.motion_file == str(ZDW0824_VISUAL_STOP_COMMAND_MOTION)
    assert cfg.commands.motion.fixed_start_frame == 0
    assert cfg.commands.motion.fixed_start_probability == pytest.approx(0.20)
    assert (
        cfg.commands.motion.fixed_start_pose_randomization_probability
        == pytest.approx(0.0)
    )
    assert (
        cfg.commands.motion.secondary_fixed_start_frame
        == ZDW0824_TOP_DEFAULT_STAND_START_FRAME
    )
    assert cfg.commands.motion.secondary_fixed_start_probability == pytest.approx(0.10)
    assert cfg.commands.motion.secondary_fixed_start_requires_nominal_ladder is True
    assert cfg.commands.motion.uniform_start_probability == pytest.approx(0.35)
    assert (
        1.0
        - cfg.commands.motion.fixed_start_probability
        - cfg.commands.motion.secondary_fixed_start_probability
        - cfg.commands.motion.uniform_start_probability
    ) == pytest.approx(0.35)
    assert runner_cfg.experiment_name == (
        "cr1_transformer_vision_zdw0824_stop_command_validated_task"
    )


def test_zdw0824_visual_command_motion_only_adds_command_fields():
    with np.load(ZDW0824_MIDDLE_STAND_MOTION, allow_pickle=False) as source, np.load(
        ZDW0824_VISUAL_STOP_COMMAND_MOTION, allow_pickle=False
    ) as filtered:
        assert set(source.files) < set(filtered.files)
        for key in source.files:
            np.testing.assert_array_equal(filtered[key], source[key])
        assert filtered["joint_pos"].shape == (3061, 25)
        assert bool(filtered["reset_valid_mask"][0])
        assert bool(
            filtered["reset_valid_mask"][
                ZDW0824_TOP_DEFAULT_STAND_START_FRAME
            ]
        )
        assert filtered["climb_direction"][:1550].tolist() == [[1.0]] * 1550
        assert filtered["climb_direction"][1550:].tolist() == [[-1.0]] * 1511
        direct_frames = filtered["zero_command_direct_hold_frames"]
        assert np.all(filtered["zero_command_valid_mask"][direct_frames])
        geometry_masks = filtered["reset_ladder_geometry_cell_valid_mask"]
        assert np.all(np.any(geometry_masks[..., direct_frames], axis=-1))


def test_zdw0824_visual_up_down_smooth_output_disables_stop_training():
    base_cfg = CR1StopCommandValidatedTaskTransformerVisionEnvCfg()
    smooth_cfg = CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg()
    cfg = CR1UpDownSmoothOutputTransformerVisionEnvCfg()
    runner_cfg = CR1UpDownSmoothOutputTransformerVisionPPORunnerCfg()

    assert type(cfg.observations) is type(base_cfg.observations)
    assert cfg.observations.to_dict() == base_cfg.observations.to_dict()
    assert cfg.rewards.to_dict() == smooth_cfg.rewards.to_dict()
    assert cfg.rewards.joint_acc_l2_upper.weight == pytest.approx(-2.5e-7)
    assert cfg.rewards.joint_acc_l2_lower.weight == pytest.approx(-1.0e-6)
    assert cfg.rewards.motion_joint_vel_lower_robust.weight == pytest.approx(-0.05)
    assert cfg.rewards.action_second_difference_l2_lower.weight == pytest.approx(-0.05)
    assert cfg.commands.motion.stop_command_training_enabled is False
    assert cfg.commands.motion.stop_hold_reset_probability == pytest.approx(0.0)
    assert cfg.commands.motion.stop_seek_episode_probability == pytest.approx(0.0)
    assert cfg.commands.motion.fixed_start_probability == pytest.approx(0.20)
    assert cfg.commands.motion.secondary_fixed_start_probability == pytest.approx(0.10)
    assert cfg.commands.motion.uniform_start_probability == pytest.approx(0.35)
    assert runner_cfg.experiment_name == (
        "cr1_transformer_vision_zdw0824_up_down_smooth_output"
    )

    with np.load(cfg.commands.motion.motion_file, allow_pickle=False) as motion:
        assert set(np.unique(motion["climb_direction"]).tolist()) == {-1.0, 1.0}


def test_task_goal_visual_task_enables_state_locked_stable_motion_labels():
    cfg = CR1UpDownTaskGoalTransformerVisionEnvCfg()
    runner_cfg = CR1UpDownTaskGoalTransformerVisionPPORunnerCfg()
    auxiliary = cfg.observations.transformer_vision_auxiliary_targets

    assert cfg.commands.motion.motion_semantic_progress_enabled is True
    assert auxiliary.task_motion_joint_pos.func.__name__ == (
        "motion_semantic_target_joint_pos"
    )
    assert auxiliary.task_motion_root_orientation_6d.func.__name__ == (
        "motion_semantic_target_root_orientation_6d"
    )
    assert auxiliary.task_motion_valid.func.__name__ == (
        "motion_semantic_target_valid"
    )
    assert runner_cfg.algorithm.visual_architecture == "task_goal"
