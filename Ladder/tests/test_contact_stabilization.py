import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stabilize_cr1_contact_motion import (
    _add_ground_support_contacts,
    _contact_anchors_and_weights,
    _ground_clearance_activity_weights,
    _ground_clearance_anchors_and_weights,
    _ground_root_blend,
    _refine_initial_root_xy,
    _slope_limited_clearance,
    _solve_root_translation,
)
from blend_cr1_motion_collision_aware import (
    CollisionPairDetail,
    IntendedSupportAllowance,
    _interpolate,
    _interpolate_joint_group,
    _is_box_top_face,
    _ladder_pair_slack,
    _regularize_alpha,
    _resolve_joint_groups,
    _within_joint_envelope,
    _within_pair_limits,
)
from optimize_cr1_standing_transition import _rotation_error, _transition_weights


def test_contact_anchor_weights_blend_and_use_run_median():
    positions = np.zeros((20, 2, 3), dtype=np.float64)
    positions[2:18, 0, 0] = np.linspace(0.0, 0.03, 16)
    contact = np.zeros((20, 2), dtype=bool)
    contact[2:18, 0] = True

    anchors, weights = _contact_anchors_and_weights(positions, contact)

    assert np.allclose(anchors[2:18, 0, 0], 0.015)
    assert weights[2, 0] == 0.0
    assert weights[17, 0] == 0.0
    assert weights[:, 1].sum() == 0.0
    assert weights[:, 0].max() == 1.0


def test_ground_support_is_added_without_removing_ladder_contacts():
    contact = np.zeros((6, 2), dtype=bool)
    contact[4:, 0] = True
    feet = np.zeros((6, 2, 3), dtype=np.float64)
    feet[..., 2] = 0.2
    feet[:3, 1, 2] = -0.01

    support = _add_ground_support_contacts(
        contact,
        feet,
        ground_z=-0.03,
        maximum_height=0.07,
    )

    assert np.all(support[4:, 0])
    assert np.all(support[:3, 1])
    assert not np.any(support[:4, 0])
    assert not np.any(support[3:, 1])


def test_root_translation_reduces_weighted_stance_error():
    positions = np.zeros((40, 2, 3), dtype=np.float64)
    positions[5:35, 0, 0] = np.linspace(-0.02, 0.02, 30)
    contact = np.zeros((40, 2), dtype=bool)
    contact[5:35, 0] = True
    anchors, weights = _contact_anchors_and_weights(positions, contact)

    correction = _solve_root_translation(positions, anchors, weights)

    before = weights[..., None] * (positions - anchors)
    after = weights[..., None] * (positions + correction[:, None, :] - anchors)
    assert correction.shape == (40, 3)
    assert np.isfinite(correction).all()
    assert np.linalg.norm(after) < np.linalg.norm(before)


def test_contact_at_trajectory_boundary_does_not_blend_in():
    positions = np.zeros((100, 2, 3), dtype=np.float64)
    contact = np.zeros((100, 2), dtype=bool)
    contact[:85, 1] = True

    _, weights = _contact_anchors_and_weights(positions, contact)

    assert weights[0, 1] == 1.0
    assert weights[50, 1] == 1.0
    assert weights[84, 1] == 0.0


def test_initial_root_xy_refinement_preserves_boundaries_and_z():
    frame_count = 120
    time = np.arange(frame_count, dtype=np.float64) / 50.0
    root_pos = np.zeros((frame_count, 3), dtype=np.float64)
    root_pos[:, 1] = 0.08 * np.sin(2.0 * np.pi * 0.8 * time)
    root_pos[:, 2] = np.linspace(0.8, 1.0, frame_count)

    refined = _refine_initial_root_xy(root_pos, 50.0)

    assert np.allclose(refined[0], root_pos[0])
    assert np.allclose(refined[:, 2], root_pos[:, 2])
    assert np.allclose(refined[81:], root_pos[81:])
    assert np.linalg.norm(np.diff(refined[:51, :2], axis=0), axis=1).sum() < np.linalg.norm(
        np.diff(root_pos[:51, :2], axis=0), axis=1
    ).sum()


def test_slope_limited_clearance_is_an_upper_envelope():
    required = np.zeros(11, dtype=np.float64)
    required[5] = 0.004

    clearance = _slope_limited_clearance(required, 0.001)

    assert np.all(clearance >= required)
    assert np.max(np.abs(np.diff(clearance))) <= 0.001 + 1.0e-12
    assert np.allclose(clearance[1:10], [0.0, 0.001, 0.002, 0.003, 0.004, 0.003, 0.002, 0.001, 0.0])


def test_ground_clearance_targets_only_change_active_sole_z():
    feet = np.zeros((8, 2, 3), dtype=np.float64)
    feet[..., 0] = 0.4
    feet[..., 1] = np.asarray([0.1, -0.1])
    minimum_z = np.full((8, 2), -0.02, dtype=np.float64)
    minimum_z[3, 0] = -0.035

    anchors, weights, clearance = _ground_clearance_anchors_and_weights(
        feet, minimum_z, ground_z=-0.03, clearance_margin=0.001
    )

    assert np.allclose(anchors[..., :2], feet[..., :2])
    assert np.allclose(anchors[..., 2] - feet[..., 2], clearance)
    assert np.all(weights[:, 1] == 0.0)
    assert np.isclose(clearance[3, 0], 0.006)


def test_ground_root_blend_has_disclosed_boundaries():
    blend = _ground_root_blend(30, 4, 9, 19, 24)

    assert np.all(blend[:5] == 1.0)
    assert blend[9] == 0.0
    assert np.all(blend[9:20] == 0.0)
    assert blend[24] == 1.0
    assert np.all(blend[24:] == 1.0)
    assert np.all(np.diff(blend[4:10]) <= 0.0)
    assert np.all(np.diff(blend[19:25]) >= 0.0)


def test_ground_activity_window_is_smooth_and_bounded():
    minimum_z = np.zeros((60, 2), dtype=np.float64)
    minimum_z[20:31, 0] = -0.01

    weights = _ground_clearance_activity_weights(minimum_z, ground_z=0.0, blend_frames=5)

    assert weights[14, 0] == 0.0
    assert weights[15, 0] == 0.0
    assert weights[20, 0] == 1.0
    assert np.all(weights[20:31, 0] == 1.0)
    assert weights[36, 0] == 0.0
    assert np.all(np.diff(weights[15:21, 0]) >= 0.0)
    assert np.all(np.diff(weights[31:37, 0]) <= 0.0)
    assert np.all(weights[:, 1] == 0.0)


def test_standing_transition_weights_have_quintic_core_and_zero_boundaries():
    weights = _transition_weights(80, 100, 190, 218)

    assert weights.shape == (139,)
    assert weights[0] == 0.0
    assert weights[20] == 1.0
    assert np.all(weights[20:111] == 1.0)
    assert weights[-1] == 0.0
    assert np.all(np.diff(weights[:21]) >= 0.0)
    assert np.all(np.diff(weights[110:]) <= 0.0)


def test_standing_transition_rotation_error_is_world_frame_target_delta():
    from scipy.spatial.transform import Rotation

    target = Rotation.from_euler("x", 10.0, degrees=True).as_matrix()[None]
    current_xyzw = Rotation.identity().as_quat()
    current_wxyz = current_xyzw[[3, 0, 1, 2]][None, None]

    error = _rotation_error(target, current_wxyz)

    assert error.shape == (1, 1, 3)
    assert np.allclose(error[0, 0], np.deg2rad([10.0, 0.0, 0.0]))


def test_collision_pair_limits_do_not_allow_new_or_deeper_penetration():
    source = {(1, 10): 0.004}

    assert _within_pair_limits({(1, 10): 0.004, (2, 10): 0.001}, source, 0.001)
    assert not _within_pair_limits({(1, 10): 0.0042}, source, 0.001)
    assert not _within_pair_limits({(2, 10): 0.0012}, source, 0.001)


def test_collision_pair_limits_add_support_slack_relative_to_source():
    source = {(1, 10): 0.004}
    slack = {(1, 10): 0.003, (2, 10): 0.003}

    assert _within_pair_limits({(1, 10): 0.0069, (2, 10): 0.0029}, source, 0.001, slack)
    assert not _within_pair_limits({(1, 10): 0.0072}, source, 0.001, slack)
    assert not _within_pair_limits({(2, 10): 0.0032}, source, 0.001, slack)


def test_ladder_pair_slack_excludes_ground_and_unrelated_environment_bodies():
    point = np.zeros(3)
    axis = np.asarray([0.0, 0.0, 1.0])
    details = {
        (1, 10): CollisionPairDetail("left_foot", "env_ladder_step_03", point, axis),
        (2, 11): CollisionPairDetail("right_foot", "env_ladder_side_board_left", point, axis),
        (3, 12): CollisionPairDetail("left_foot", "env_ground", point, axis),
        (4, 13): CollisionPairDetail("right_foot", "environment_fixture", point, axis),
    }

    assert _ladder_pair_slack(details, 0.002) == {(1, 10): 0.002, (2, 11): 0.002}


def test_joint_envelope_rejects_only_values_outside_source_envelope():
    lower = np.asarray([-0.4, -1.0])
    upper = np.asarray([0.5, 1.0])

    assert _within_joint_envelope(np.asarray([-0.4, 0.9]), lower, upper)
    assert not _within_joint_envelope(np.asarray([-0.401, 0.9]), lower, upper)
    assert not _within_joint_envelope(np.asarray([0.0, 1.001]), lower, upper)


def test_intended_support_allowance_requires_target_rung_top_and_collision_body():
    from types import SimpleNamespace

    spec = SimpleNamespace(
        name="step_03",
        size=(0.64, 0.13, 0.03),
        position=(0.7, 0.04, 0.6),
        quaternion=(1.0, 0.0, 0.0, 0.0),
    )
    indicator = np.asarray([[0.0, 1.0, 0.0, 0.0]])
    rung_id = np.asarray([[-1, 2, -1, -1]])
    allowance = IntendedSupportAllowance(indicator, rung_id, (spec,), 0.001)
    top_point = np.asarray([0.7, 0.04, 0.615])
    vertical_axis = np.asarray([0.0, 0.0, 1.0])
    details = {
        (1, 10): CollisionPairDetail(
            "right_sphere_hand_link", "env_ladder_step_03", top_point, vertical_axis
        ),
        (2, 10): CollisionPairDetail(
            "right_wrist_y_link", "env_ladder_step_03", top_point, vertical_axis
        ),
        (3, 11): CollisionPairDetail(
            "right_sphere_hand_link", "env_ladder_step_02", top_point, vertical_axis
        ),
        (4, 10): CollisionPairDetail(
            "right_sphere_hand_link",
            "env_ladder_step_03",
            np.asarray([0.7, 0.04, 0.585]),
            vertical_axis,
        ),
    }

    assert allowance.pair_slack(0, details, 0.003) == {(1, 10): 0.003}
    assert _is_box_top_face(top_point, vertical_axis, spec, 0.001)
    assert not _is_box_top_face(top_point, np.asarray([1.0, 0.0, 0.0]), spec, 0.001)


def test_collision_alpha_regularization_respects_framewise_upper_bounds():
    upper = np.ones(60, dtype=np.float64)
    upper[25:35] = 0.2

    alpha = _regularize_alpha(upper, velocity_weight=20.0, acceleration_weight=200.0)

    assert np.all(alpha >= 0.0)
    assert np.all(alpha <= upper + 1.0e-10)
    assert np.all(alpha[25:35] <= 0.2 + 1.0e-10)
    assert alpha[0] > alpha[24] > alpha[29]
    assert alpha[-1] > alpha[35] > alpha[30]


def test_collision_alpha_regularization_handles_all_zero_upper_bounds():
    upper = np.zeros(40, dtype=np.float64)

    alpha = _regularize_alpha(upper, velocity_weight=20.0, acceleration_weight=200.0)

    assert np.array_equal(alpha, upper)


def test_collision_interpolation_uses_output_float32_precision():
    source_pos = np.asarray([0.123456789, 0.0, 1.0])
    candidate_pos = np.asarray([0.223456789, 0.1, 1.1])
    source_quat = np.asarray([1.0, 0.0, 0.0, 0.0])
    candidate_quat = np.asarray([0.999, 0.0, 0.0, 0.04])
    source_joint = np.asarray([0.123456789, -0.4])
    candidate_joint = np.asarray([0.223456789, -0.2])

    state = _interpolate(
        0.37,
        source_pos,
        source_quat,
        source_joint,
        candidate_pos,
        candidate_quat,
        candidate_joint,
    )

    for value in state:
        assert np.array_equal(value, value.astype(np.float32).astype(np.float64))


def test_groupwise_interpolation_changes_only_selected_joints():
    root_pos = np.asarray([0.1, 0.2, 1.0])
    root_quat = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_joint = np.asarray([0.1, 0.2, 0.3, 0.4])
    source_joint = np.asarray([0.0, 0.0, 0.0, 0.0])
    candidate_joint = np.asarray([1.0, 1.0, 1.0, 1.0])

    state = _interpolate_joint_group(
        0.25,
        root_pos,
        root_quat,
        current_joint,
        source_joint,
        candidate_joint,
        np.asarray([1, 3]),
    )

    assert np.array_equal(state[0], root_pos.astype(np.float32).astype(np.float64))
    assert np.array_equal(state[1], root_quat.astype(np.float32).astype(np.float64))
    assert np.allclose(state[2], [0.1, 0.25, 0.3, 0.25])


def test_cr1_joint_groups_are_disjoint_and_cover_all_motion_joints():
    joint_names = [
        "waist_z_joint",
        "waist_x_joint",
        "waist_y_joint",
        "left_shoulder_y_joint",
        "left_shoulder_x_joint",
        "left_shoulder_z_joint",
        "left_elbow_joint",
        "right_shoulder_y_joint",
        "right_shoulder_x_joint",
        "right_shoulder_z_joint",
        "right_elbow_joint",
        "neck_z_joint",
        "neck_y_joint",
        "left_hip_y_joint",
        "left_hip_x_joint",
        "left_hip_z_joint",
        "left_knee_joint",
        "left_ankle_y_joint",
        "left_ankle_x_joint",
        "right_hip_y_joint",
        "right_hip_x_joint",
        "right_hip_z_joint",
        "right_knee_joint",
        "right_ankle_y_joint",
        "right_ankle_x_joint",
    ]

    groups = _resolve_joint_groups(joint_names)
    indexes = np.concatenate([group_indexes for _, group_indexes in groups])

    assert [name for name, _ in groups] == [
        "left_arm",
        "right_arm",
        "torso",
        "right_leg",
        "left_leg",
    ]
    assert sorted(indexes.tolist()) == list(range(len(joint_names)))
