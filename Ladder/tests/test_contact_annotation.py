from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from ladder.tasks.ladder_climb.ladder_geometry import LADDER_A, get_rung_pose
from ladder.utils.contact_annotation import (
    ContactAnnotationCfg,
    ContactPhaseAnnotationCfg,
    DirectionalContactPhaseCfg,
    annotate_directional_ladder_contact_phases,
    annotate_ladder_contact_phases,
    annotate_ladder_contacts,
    annotate_motion_file,
    make_contact_annotation_cfg,
)
from ladder.utils.contact_schema import (
    CR1_COMMAND_CONTACT_TARGET_OFFSETS,
    CR1_CONTACT_DISTANCE_THRESHOLDS,
    CR1_CONTACT_TARGET_OFFSETS,
    CR1_LIMB_BODY_NAMES,
    DEFAULT_G1_BODY_NAMES,
    LIMB_NAMES,
)


def _motion_with_limb_positions(left_hand_positions):
    body_pos_w = np.zeros((len(left_hand_positions), len(DEFAULT_G1_BODY_NAMES), 3), dtype=np.float32)
    left_hand_index = DEFAULT_G1_BODY_NAMES.index("left_wrist_yaw_link")
    right_hand_index = DEFAULT_G1_BODY_NAMES.index("right_wrist_yaw_link")
    left_foot_index = DEFAULT_G1_BODY_NAMES.index("left_ankle_roll_link")
    right_foot_index = DEFAULT_G1_BODY_NAMES.index("right_ankle_roll_link")
    body_pos_w[:, left_hand_index, :] = np.asarray(left_hand_positions, dtype=np.float32)
    body_pos_w[:, right_hand_index, :] = np.asarray(get_rung_pose(0, LADDER_A)[0], dtype=np.float32)
    body_pos_w[:, left_foot_index, :] = np.asarray(get_rung_pose(0, LADDER_A)[0], dtype=np.float32)
    body_pos_w[:, right_foot_index, :] = np.asarray(get_rung_pose(0, LADDER_A)[0], dtype=np.float32)
    return {"fps": np.asarray([50]), "body_pos_w": body_pos_w, "body_names": np.asarray(DEFAULT_G1_BODY_NAMES)}


def test_contact_annotation_labels_stable_nearest_rung_contact():
    rung_id = 2
    rung_pos = np.asarray(get_rung_pose(rung_id, LADDER_A)[0], dtype=np.float32)
    left_target = rung_pos + np.asarray([0.0, LADDER_A.lateral_contact_offset, 0.0], dtype=np.float32)
    motion = _motion_with_limb_positions(np.repeat(left_target[None, :], 8, axis=0))
    cfg = ContactAnnotationCfg(distance_threshold=0.03, speed_threshold=0.05, min_contact_frames=2)

    labels = annotate_ladder_contacts(motion, cfg)

    left_hand = LIMB_NAMES.index("left_hand")
    assert labels["contact_indicator"].shape == (8, len(LIMB_NAMES))
    assert np.all(labels["contact_indicator"][:, left_hand] == 1.0)
    assert np.all(labels["contact_rung_id"][:, left_hand] == rung_id)


def test_contact_annotation_removes_short_contact_runs():
    rung_pos = np.asarray(get_rung_pose(1, LADDER_A)[0], dtype=np.float32)
    left_target = rung_pos + np.asarray([0.0, LADDER_A.lateral_contact_offset, 0.0], dtype=np.float32)
    far_pos = left_target + np.asarray([0.5, 0.0, 0.0], dtype=np.float32)
    left_hand_positions = np.stack([far_pos, far_pos, left_target, far_pos, far_pos], axis=0)
    motion = _motion_with_limb_positions(left_hand_positions)
    cfg = ContactAnnotationCfg(distance_threshold=0.03, speed_threshold=100.0, min_contact_frames=2)

    labels = annotate_ladder_contacts(motion, cfg)

    left_hand = LIMB_NAMES.index("left_hand")
    assert np.all(labels["contact_indicator"][:, left_hand] == 0.0)


def test_contact_annotation_supports_explicit_limb_indexes():
    rung_pos = np.asarray(get_rung_pose(3, LADDER_A)[0], dtype=np.float32)
    left_target = rung_pos + np.asarray([0.0, LADDER_A.lateral_contact_offset, 0.0], dtype=np.float32)
    body_pos_w = np.zeros((6, 4, 3), dtype=np.float32)
    body_pos_w[:, 0, :] = left_target
    motion = {"fps": np.asarray([50]), "body_pos_w": body_pos_w}
    cfg = ContactAnnotationCfg(distance_threshold=0.03, speed_threshold=0.05, min_contact_frames=2)

    labels = annotate_ladder_contacts(motion, cfg, limb_body_indexes=[0, 1, 2, 3])

    left_hand = LIMB_NAMES.index("left_hand")
    assert np.all(labels["contact_indicator"][:, left_hand] == 1.0)
    assert np.all(labels["contact_rung_id"][:, left_hand] == 3)


def test_contact_annotation_uses_per_limb_target_offsets():
    offsets = ((0.0, 0.10, 0.0), (0.0, -0.11, 0.0), (0.0, 0.12, 0.0), (0.0, -0.13, 0.0))
    rung_pos = np.asarray(get_rung_pose(2, LADDER_A)[0], dtype=np.float32)
    limb_positions = rung_pos[None, :] + np.asarray(offsets, dtype=np.float32)
    body_pos_w = np.repeat(limb_positions[None, :, :], 4, axis=0)
    motion = {"fps": np.asarray([30]), "body_pos_w": body_pos_w}
    cfg = ContactAnnotationCfg(
        limb_target_offsets=offsets,
        distance_threshold=0.01,
        speed_threshold=0.01,
        min_contact_frames=1,
    )

    labels = annotate_ladder_contacts(motion, cfg, limb_body_indexes=[0, 1, 2, 3])

    assert np.allclose(labels["contact_nearest_distance"], 0.0, atol=1.0e-6)
    assert np.allclose(labels["contact_annotation_limb_target_offsets"], offsets)


def test_cr1_defaults_use_sphere_hands_and_sole_contact_points():
    assert CR1_LIMB_BODY_NAMES["left_hand"] == "left_sphere_hand_link"
    assert CR1_LIMB_BODY_NAMES["right_hand"] == "right_sphere_hand_link"
    assert CR1_LIMB_BODY_NAMES["left_foot"] == "left_sole_contact_point"
    assert CR1_LIMB_BODY_NAMES["right_foot"] == "right_sole_contact_point"
    assert CR1_CONTACT_TARGET_OFFSETS["left_hand"] == (-0.015, 0.25, 0.045)
    assert CR1_CONTACT_TARGET_OFFSETS["right_hand"] == (-0.01, -0.25, 0.045)
    assert CR1_CONTACT_TARGET_OFFSETS["left_foot"] == (0.0, 0.13, 0.0)
    assert CR1_CONTACT_TARGET_OFFSETS["right_foot"] == (0.0, -0.18, 0.0)
    assert CR1_CONTACT_DISTANCE_THRESHOLDS == {
        "left_hand": 0.052,
        "right_hand": 0.05,
        "left_foot": 0.145,
        "right_foot": 0.145,
    }
    assert CR1_COMMAND_CONTACT_TARGET_OFFSETS["left_foot"] == (0.0, 0.18, 0.0)


def test_contact_annotation_supports_per_limb_distance_thresholds():
    body_pos_w = np.zeros((4, 4, 3), dtype=np.float32)
    rung_pos = np.asarray(get_rung_pose(0, LADDER_A)[0], dtype=np.float32)
    offsets = np.asarray(
        [
            [0.02, LADDER_A.lateral_contact_offset, 0.0],
            [0.02, -LADDER_A.lateral_contact_offset, 0.0],
            [0.02, LADDER_A.lateral_contact_offset, 0.0],
            [0.02, -LADDER_A.lateral_contact_offset, 0.0],
        ],
        dtype=np.float32,
    )
    body_pos_w[:] = rung_pos + offsets
    motion = {"fps": np.asarray([30]), "body_pos_w": body_pos_w}
    cfg = ContactAnnotationCfg(
        limb_distance_thresholds=(0.01, 0.03, 0.01, 0.03),
        speed_threshold=1.0,
        min_contact_frames=1,
    )

    labels = annotate_ladder_contacts(motion, cfg, limb_body_indexes=[0, 1, 2, 3])

    assert labels["contact_indicator"][0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert np.allclose(labels["contact_annotation_limb_distance_thresholds"], [0.01, 0.03, 0.01, 0.03])


def test_contact_annotation_hysteresis_fills_only_near_same_rung_gaps():
    rung_pos = np.asarray(get_rung_pose(2, LADDER_A)[0], dtype=np.float32)
    target = rung_pos + np.asarray(
        [0.0, LADDER_A.lateral_contact_offset, 0.0], dtype=np.float32
    )
    near_excursion = target + np.asarray([0.06, 0.0, 0.0], dtype=np.float32)
    true_separation = target + np.asarray([0.12, 0.0, 0.0], dtype=np.float32)
    positions = np.concatenate(
        (
            np.repeat(target[None], 4, axis=0),
            np.repeat(near_excursion[None], 3, axis=0),
            np.repeat(target[None], 4, axis=0),
            np.repeat(true_separation[None], 3, axis=0),
            np.repeat(target[None], 4, axis=0),
        ),
        axis=0,
    )
    motion = _motion_with_limb_positions(positions)
    motion["fps"] = np.asarray([1.0])
    cfg = ContactAnnotationCfg(
        distance_threshold=0.03,
        limb_release_distance_thresholds=(0.08, 0.03, 0.03, 0.03),
        speed_threshold=1.0,
        min_contact_frames=2,
        max_gap_frames=0,
    )

    labels = annotate_ladder_contacts(motion, cfg)

    left_hand = LIMB_NAMES.index("left_hand")
    assert np.all(labels["contact_indicator"][4:7, left_hand] == 1.0)
    assert np.all(labels["contact_hysteresis_filled_indicator"][4:7, left_hand] == 1.0)
    assert np.all(labels["contact_indicator"][11:14, left_hand] == 0.0)
    assert np.allclose(
        labels["contact_annotation_limb_release_distance_thresholds"],
        [0.08, 0.03, 0.03, 0.03],
    )


def test_contact_annotation_generic_default_thresholds():
    assert ContactAnnotationCfg().distance_threshold == 0.125
    assert ContactAnnotationCfg().speed_threshold == 0.35
    assert ContactAnnotationCfg().ground_initial_feet is False
    assert make_contact_annotation_cfg().distance_threshold == 0.125
    assert make_contact_annotation_cfg().speed_threshold == 0.35
    assert make_contact_annotation_cfg().ground_initial_feet is False


def test_surface_phase_accepts_finite_tread_area_away_from_fixed_target():
    rung_id = 2
    rung_center = np.asarray(get_rung_pose(rung_id, LADDER_A)[0], dtype=np.float32)
    rung_top_z = rung_center[2]
    sphere_center = rung_center.copy()
    sphere_center[1] = -0.25
    sphere_center[2] = rung_top_z + 0.045 + 0.02
    body_pos_w = np.repeat(sphere_center[None, None, :], 5, axis=0)
    motion = {"body_pos_w": body_pos_w}
    phase_cfg = ContactPhaseAnnotationCfg(
        limb_names=("left_hand",),
        limb_body_names=("left_sphere_hand_link",),
        entry_distance_thresholds=(0.06,),
        release_distance_thresholds=(0.08,),
        entry_min_clearances=(-0.01,),
        release_min_clearances=(-0.015,),
        contact_point_z_offsets=(-0.045,),
        min_contact_frames=1,
    )

    labels = annotate_ladder_contact_phases(motion, phase_cfg, limb_body_indexes=[0])

    assert np.all(labels["contact_phase_indicator"] == 1.0)
    assert np.all(labels["contact_phase_rung_id"] == rung_id)
    assert np.allclose(labels["contact_surface_nearest_distance"], 0.02, atol=1.0e-6)
    fixed_target = rung_center + np.asarray([0.0, LADDER_A.lateral_contact_offset, 0.0])
    assert np.linalg.norm(sphere_center - fixed_target) > 0.4


def test_surface_phase_uses_sphere_bottom_for_hand_clearance():
    rung_center = np.asarray(get_rung_pose(1, LADDER_A)[0], dtype=np.float32)
    rung_top_z = rung_center[2]
    sphere_center = rung_center.copy()
    sphere_center[2] = rung_top_z + 0.045 + 0.055
    motion = {"body_pos_w": np.repeat(sphere_center[None, None, :], 3, axis=0)}
    cfg = ContactPhaseAnnotationCfg(
        limb_names=("left_hand",),
        limb_body_names=("left_sphere_hand_link",),
        entry_distance_thresholds=(0.06,),
        release_distance_thresholds=(0.08,),
        entry_min_clearances=(-0.01,),
        release_min_clearances=(-0.015,),
        contact_point_z_offsets=(-0.045,),
        min_contact_frames=1,
    )

    labels = annotate_ladder_contact_phases(motion, cfg, limb_body_indexes=[0])

    assert np.all(labels["contact_phase_indicator"] == 1.0)
    assert np.allclose(labels["contact_surface_nearest_clearance"], 0.055, atol=1.0e-6)


def test_surface_phase_holds_same_rung_until_release_threshold():
    rung_id = 3
    rung_center = np.asarray(get_rung_pose(rung_id, LADDER_A)[0], dtype=np.float32)
    rung_top_z = rung_center[2]
    clearances = np.asarray([0.02, 0.02, 0.02, 0.07, 0.07, 0.09], dtype=np.float32)
    positions = np.repeat(rung_center[None, :], len(clearances), axis=0)
    positions[:, 2] = rung_top_z + 0.045 + clearances
    motion = {"body_pos_w": positions[:, None, :]}
    cfg = ContactPhaseAnnotationCfg(
        limb_names=("left_hand",),
        limb_body_names=("left_sphere_hand_link",),
        entry_distance_thresholds=(0.06,),
        release_distance_thresholds=(0.08,),
        entry_min_clearances=(-0.01,),
        release_min_clearances=(-0.015,),
        contact_point_z_offsets=(-0.045,),
        min_contact_frames=1,
        max_gap_frames=0,
    )

    labels = annotate_ladder_contact_phases(motion, cfg, limb_body_indexes=[0])

    assert labels["contact_phase_raw_indicator"][:, 0].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert labels["contact_phase_indicator"][:, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert labels["contact_phase_rung_id"][:, 0].tolist() == [rung_id] * 5 + [-1]
    assert labels["contact_phase_hysteresis_filled_indicator"][:, 0].tolist() == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
    ]


def test_directional_surface_phase_locks_departed_rung_and_resets_for_descent():
    geometry = make_contact_annotation_cfg(num_rungs=2).ladder_geometry
    rung_0 = np.asarray(get_rung_pose(0, geometry)[0], dtype=np.float32)
    rung_1 = np.asarray(get_rung_pose(1, geometry)[0], dtype=np.float32)

    def sphere_center(rung, clearance):
        point = rung.copy()
        point[2] += 0.045 + clearance
        return point

    positions = np.stack(
        [
            *[sphere_center(rung_0, 0.0) for _ in range(3)],
            sphere_center(rung_0, 0.02),
            sphere_center(rung_0, 0.04),
            sphere_center(rung_0, 0.0),
            sphere_center(rung_0, 0.0),
            *[sphere_center(rung_1, 0.0) for _ in range(5)],
            *[sphere_center(rung_1, 0.0) for _ in range(3)],
            sphere_center(rung_1, 0.02),
            sphere_center(rung_1, 0.04),
            *[sphere_center(rung_0, 0.0) for _ in range(3)],
        ]
    )
    motion = {
        "fps": np.asarray([50.0], dtype=np.float32),
        "body_pos_w": positions[:, None, :],
        "contact_indicator": np.zeros((len(positions), 1), dtype=np.float32),
        "contact_rung_id": np.zeros((len(positions), 1), dtype=np.int64),
    }
    motion["contact_indicator"][10, 0] = 1.0
    motion["contact_rung_id"][10, 0] = 1
    phase_cfg = ContactPhaseAnnotationCfg(
        ladder_geometry=geometry,
        limb_names=("left_hand",),
        limb_body_names=("left_sphere_hand_link",),
        entry_distance_thresholds=(0.06,),
        release_distance_thresholds=(0.08,),
        entry_min_clearances=(-0.01,),
        release_min_clearances=(-0.015,),
        contact_point_z_offsets=(-0.045,),
        min_contact_frames=1,
    )
    directional_cfg = DirectionalContactPhaseCfg(
        limb_names=("left_hand",),
        top_neutral_start_frame=10,
        descent_start_frame=12,
        core_distance_thresholds=(0.01,),
        departure_distance_thresholds=(0.01,),
        departure_rate_thresholds=(0.10,),
        departure_normal_velocity_thresholds=(0.20,),
        confirmation_frames=2,
    )

    labels = annotate_directional_ladder_contact_phases(
        motion,
        phase_cfg,
        directional_cfg,
        limb_body_indexes=[0],
    )

    phase = labels["contact_phase_indicator"][:, 0]
    rung_ids = labels["contact_phase_rung_id"][:, 0]
    assert np.all(phase[:10] == 1.0)
    assert np.all(rung_ids[:7] == 0)
    assert np.all(rung_ids[7:10] == 1)
    assert phase[10] == 1.0
    assert rung_ids[10] == 1
    assert phase[11] == 0.0
    assert np.all(rung_ids[12:16] == 1)
    assert np.all(rung_ids[17:20] == 0)
    assert labels["contact_phase_travel_direction"].tolist() == [1] * 10 + [0] * 2 + [-1] * 8


def test_surface_phase_fills_short_same_rung_gap_outside_release_window():
    geometry = make_contact_annotation_cfg(num_rungs=1).ladder_geometry
    rung = np.asarray(get_rung_pose(0, geometry)[0], dtype=np.float32)

    def sphere_center(clearance):
        point = rung.copy()
        point[2] += 0.045 + clearance
        return point

    positions = np.stack(
        [
            sphere_center(0.0),
            sphere_center(0.0),
            sphere_center(0.10),
            sphere_center(0.10),
            sphere_center(0.0),
            sphere_center(0.0),
        ]
    )
    motion = {"body_pos_w": positions[:, None, :]}
    cfg = ContactPhaseAnnotationCfg(
        ladder_geometry=geometry,
        limb_names=("left_hand",),
        limb_body_names=("left_sphere_hand_link",),
        entry_distance_thresholds=(0.06,),
        release_distance_thresholds=(0.08,),
        entry_min_clearances=(-0.01,),
        release_min_clearances=(-0.015,),
        contact_point_z_offsets=(-0.045,),
        min_contact_frames=1,
        max_gap_frames=2,
    )

    labels = annotate_ladder_contact_phases(motion, cfg, limb_body_indexes=[0])

    assert labels["contact_phase_indicator"][:, 0].tolist() == [1.0] * 6
    assert labels["contact_phase_rung_id"][:, 0].tolist() == [0] * 6


def test_directional_surface_phase_bridges_same_rung_recontact_episode():
    geometry = make_contact_annotation_cfg(num_rungs=1).ladder_geometry
    rung = np.asarray(get_rung_pose(0, geometry)[0], dtype=np.float32)

    def sphere_center(clearance):
        point = rung.copy()
        point[2] += 0.045 + clearance
        return point

    clearances = [0.0] * 3 + [0.011, 0.02, 0.03, 0.04, 0.05, 0.055] + [0.0] * 6 + [0.20] * 5
    positions = np.stack([sphere_center(clearance) for clearance in clearances])
    support = np.zeros((len(positions), 1), dtype=np.float32)
    support[12:14, 0] = 1.0
    motion = {
        "fps": np.asarray([50.0], dtype=np.float32),
        "body_pos_w": positions[:, None, :],
        "contact_indicator": support,
        "contact_rung_id": np.zeros((len(positions), 1), dtype=np.int64),
    }
    phase_cfg = ContactPhaseAnnotationCfg(
        ladder_geometry=geometry,
        limb_names=("left_hand",),
        limb_body_names=("left_sphere_hand_link",),
        entry_distance_thresholds=(0.06,),
        release_distance_thresholds=(0.08,),
        entry_min_clearances=(-0.01,),
        release_min_clearances=(-0.015,),
        contact_point_z_offsets=(-0.045,),
        min_contact_frames=1,
        max_gap_frames=5,
    )
    directional_cfg = DirectionalContactPhaseCfg(
        limb_names=("left_hand",),
        top_neutral_start_frame=15,
        descent_start_frame=18,
        core_distance_thresholds=(0.01,),
        departure_distance_thresholds=(0.01,),
        departure_rate_thresholds=(0.10,),
        departure_normal_velocity_thresholds=(0.20,),
        confirmation_frames=2,
    )

    labels = annotate_directional_ladder_contact_phases(
        motion,
        phase_cfg,
        directional_cfg,
        limb_body_indexes=[0],
    )

    assert labels["contact_phase_indicator"][:15, 0].tolist() == [1.0] * 15
    assert labels["contact_phase_rung_id"][:15, 0].tolist() == [0] * 15
    assert labels["contact_phase_directional_recontact_bridge_indicator"][:, 0].sum() > 0


def test_cr1_virtual_sole_points_are_materialized_from_ankle_pose(tmp_path):
    frames = 5
    body_names = (
        "left_sphere_hand_link",
        "right_sphere_hand_link",
        "left_ankle_x_link",
        "right_ankle_x_link",
    )
    body_pos_w = np.zeros((frames, len(body_names), 3), dtype=np.float32)
    body_quat_w = np.zeros((frames, len(body_names), 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.zeros_like(body_pos_w)
    body_ang_vel_w = np.zeros_like(body_pos_w)

    rung_pos = np.asarray(get_rung_pose(2, LADDER_A)[0], dtype=np.float32)
    left_target = rung_pos + np.asarray([0.0, LADDER_A.lateral_contact_offset, 0.0], dtype=np.float32)
    right_target = rung_pos + np.asarray([0.0, -LADDER_A.lateral_contact_offset, 0.0], dtype=np.float32)
    sole_offset = np.asarray([0.045, 0.0, -0.075], dtype=np.float32)
    body_pos_w[:, 2, :] = left_target - sole_offset

    # A 180-degree Y rotation maps the local sole offset to (-x, 0, -z).
    body_quat_w[:, 3, :] = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    rotated_sole_offset = np.asarray([-0.045, 0.0, 0.075], dtype=np.float32)
    body_pos_w[:, 3, :] = right_target - rotated_sole_offset

    input_file = tmp_path / "motion.npz"
    output_file = tmp_path / "annotated.npz"
    np.savez(
        input_file,
        fps=np.asarray([30.0], dtype=np.float32),
        body_names=np.asarray(body_names),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    cfg = make_contact_annotation_cfg(
        limb_body_names=tuple(CR1_LIMB_BODY_NAMES[name] for name in LIMB_NAMES),
        speed_threshold=1.0,
        min_contact_frames=1,
        ladder_asset_translation=(0.38, 0.04, 0.08),
        ladder_asset_yaw_deg=90.0,
    )

    labels = annotate_motion_file(input_file, output_file, cfg)

    assert labels["limb_body_indexes"].tolist() == [0, 1, 4, 5]
    assert np.allclose(labels["contact_nearest_distance"][:, 2:], 0.0, atol=1.0e-6)
    with np.load(output_file, allow_pickle=False) as annotated:
        assert annotated["body_names"].tolist()[-2:] == [
            "left_sole_contact_point",
            "right_sole_contact_point",
        ]
        assert annotated["body_pos_w"].shape == (frames, 6, 3)
        assert annotated["body_quat_w"].shape == (frames, 6, 4)
        assert annotated["body_lin_vel_w"].shape == (frames, 6, 3)
        assert annotated["body_ang_vel_w"].shape == (frames, 6, 3)
        assert annotated["contact_annotation_ladder_num_rungs"].tolist() == [12]
        assert np.allclose(annotated["contact_annotation_ladder_asset_translation"], [0.38, 0.04, 0.08])
        assert annotated["contact_annotation_ladder_asset_yaw_deg"].tolist() == [90.0]


def test_cr1_urdf_sole_points_match_foot_bottom_center():
    urdf_path = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "robots"
        / "CR1-B2-PRO-260608"
        / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
    )
    root = ElementTree.parse(urdf_path).getroot()

    for side in ("left", "right"):
        joint = root.find(f"./joint[@name='{side}_sole_contact_point_joint']")
        assert joint is not None
        assert joint.find("parent").attrib["link"] == f"{side}_ankle_x_link"
        assert joint.find("child").attrib["link"] == f"{side}_sole_contact_point"
        assert joint.find("origin").attrib["xyz"] == "0.045 0 -0.075"


def test_initial_foot_grounding_raises_entire_body_trajectory(tmp_path):
    body_names = (
        "left_sphere_hand_link",
        "right_sphere_hand_link",
        "left_sole_contact_point",
        "right_sole_contact_point",
    )
    body_pos_w = np.zeros((3, 4, 3), dtype=np.float32)
    body_pos_w[0, 2, 2] = -0.03
    body_pos_w[0, 3, 2] = -0.05
    body_pos_w[1:, 2:, 2] = 0.02
    input_file = tmp_path / "motion.npz"
    output_file = tmp_path / "annotated.npz"
    np.savez(
        input_file,
        fps=np.asarray([30.0], dtype=np.float32),
        body_names=np.asarray(body_names),
        body_pos_w=body_pos_w,
    )
    cfg = make_contact_annotation_cfg(
        limb_body_names=tuple(CR1_LIMB_BODY_NAMES[name] for name in LIMB_NAMES),
        speed_threshold=100.0,
        min_contact_frames=1,
        ground_initial_feet=True,
        initial_foot_target_z=-0.03,
    )

    labels = annotate_motion_file(input_file, output_file, cfg)

    assert np.allclose(labels["contact_annotation_robot_grounding_z_offset"], [0.02])
    assert np.allclose(labels["contact_annotation_robot_initial_foot_target_z"], [-0.03])
    with np.load(output_file, allow_pickle=False) as annotated:
        assert np.isclose(annotated["body_pos_w"][0, 3, 2], -0.03)
        assert np.allclose(annotated["body_pos_w"][..., 2], body_pos_w[..., 2] + 0.02)
