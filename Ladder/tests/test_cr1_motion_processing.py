import importlib.util
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from ladder.utils.urdf_kinematics import forward_kinematics


REPO_ROOT = Path(__file__).resolve().parents[2]
CR1_URDF = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)


def _load_smoothing_script():
    script_path = REPO_ROOT / "Ladder" / "scripts" / "smooth_resample_cr1_motion.py"
    spec = importlib.util.spec_from_file_location("smooth_resample_cr1_motion", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cr1_urdf_has_25_actuated_joints_and_current_contact_bodies():
    root = ElementTree.parse(CR1_URDF).getroot()
    actuated = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") not in ("fixed", "floating")
    ]
    links = {link.attrib["name"] for link in root.findall("link")}

    assert len(actuated) == 25
    assert {
        "left_wrist_x_link",
        "right_wrist_x_link",
        "left_sphere_hand_link",
        "right_sphere_hand_link",
        "left_sole_contact_point",
        "right_sole_contact_point",
    } <= links


def test_outlier_interpolation_uses_neighbor_midpoint_and_quaternion_slerp():
    smoothing = _load_smoothing_script()
    joint_pos = np.zeros((5, 2), dtype=np.float64)
    root_pos = np.zeros((5, 3), dtype=np.float64)
    root_quat = np.zeros((5, 4), dtype=np.float64)
    root_quat[:, 0] = 1.0
    joint_pos[1] = [1.0, 3.0]
    joint_pos[2] = [99.0, -99.0]
    joint_pos[3] = [5.0, 7.0]
    root_pos[1] = [1.0, 2.0, 3.0]
    root_pos[2] = [99.0, 99.0, 99.0]
    root_pos[3] = [5.0, 6.0, 7.0]

    repaired_joint, repaired_root, repaired_quat = smoothing._interpolate_outlier_frames(
        joint_pos, root_pos, root_quat, [2]
    )

    assert np.allclose(repaired_joint[2], [3.0, 5.0])
    assert np.allclose(repaired_root[2], [3.0, 4.0, 5.0])
    assert np.allclose(repaired_quat[2], [1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(repaired_joint[[0, 1, 3, 4]], joint_pos[[0, 1, 3, 4]])


def test_leg_only_smoothing_preserves_non_leg_joints_and_root():
    smoothing = _load_smoothing_script()
    root = ElementTree.parse(CR1_URDF).getroot()
    joint_names = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") in ("revolute", "continuous")
    ]
    frames = 101
    times = np.arange(frames, dtype=np.float64) / 50.0
    joint_pos = np.zeros((frames, len(joint_names)), dtype=np.float64)
    non_leg_index = joint_names.index("waist_z_joint")
    ankle_index = joint_names.index("left_ankle_x_joint")
    joint_pos[:, non_leg_index] = np.sin(2.0 * np.pi * 10.0 * times)
    joint_pos[:, ankle_index] = np.sin(2.0 * np.pi * 10.0 * times)
    root_pos = np.stack([0.1 * times, np.zeros(frames), np.ones(frames)], axis=-1)
    root_angles = 0.2 * np.sin(2.0 * np.pi * 0.5 * times)
    root_quat = np.zeros((frames, 1, 4), dtype=np.float64)
    root_quat[:, 0, 0] = np.cos(0.5 * root_angles)
    root_quat[:, 0, 3] = np.sin(0.5 * root_angles)
    motion = {
        "fps": np.asarray([50.0]),
        "joint_pos": joint_pos,
        "joint_names": np.asarray(joint_names),
        "source_joint_names": np.asarray(joint_names),
        "body_pos_w": root_pos[:, None, :],
        "body_quat_w": root_quat,
        "body_names": np.asarray(["base_link"]),
    }

    qpos, _ = smoothing.smooth_resample_motion(
        motion,
        output_fps=50.0,
        joint_cutoff_hz=5.0,
        leg_joint_cutoff_hz=3.0,
        ankle_joint_cutoff_hz=2.0,
        root_position_cutoff_hz=4.0,
        root_orientation_cutoff_hz=4.0,
        preserve_non_leg_joints=True,
        preserve_root_trajectory=True,
    )

    assert np.allclose(qpos[:, :3], root_pos, atol=1.0e-7)
    assert np.array_equal(qpos[:, 3:7], root_quat[:, 0].astype(np.float32))
    assert np.allclose(qpos[:, 7 + non_leg_index], joint_pos[:, non_leg_index], atol=1.0e-7)
    assert not np.allclose(qpos[:, 7 + ankle_index], joint_pos[:, ankle_index], atol=1.0e-3)


def test_non_leg_smoothing_preserves_leg_joints_and_root():
    smoothing = _load_smoothing_script()
    root = ElementTree.parse(CR1_URDF).getroot()
    joint_names = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") in ("revolute", "continuous")
    ]
    frames = 101
    times = np.arange(frames, dtype=np.float64) / 50.0
    joint_pos = np.zeros((frames, len(joint_names)), dtype=np.float64)
    waist_index = joint_names.index("waist_z_joint")
    ankle_index = joint_names.index("left_ankle_x_joint")
    joint_pos[:, waist_index] = np.sin(2.0 * np.pi * 10.0 * times)
    joint_pos[:, ankle_index] = np.sin(2.0 * np.pi * 10.0 * times)
    root_pos = np.stack([0.1 * times, np.zeros(frames), np.ones(frames)], axis=-1)
    root_quat = np.zeros((frames, 1, 4), dtype=np.float64)
    root_quat[:, 0, 0] = 1.0
    motion = {
        "fps": np.asarray([50.0]),
        "joint_pos": joint_pos,
        "joint_names": np.asarray(joint_names),
        "source_joint_names": np.asarray(joint_names),
        "body_pos_w": root_pos[:, None, :],
        "body_quat_w": root_quat,
        "body_names": np.asarray(["base_link"]),
    }

    qpos, _ = smoothing.smooth_resample_motion(
        motion,
        output_fps=50.0,
        joint_cutoff_hz=5.0,
        leg_joint_cutoff_hz=3.0,
        ankle_joint_cutoff_hz=2.0,
        root_position_cutoff_hz=4.0,
        root_orientation_cutoff_hz=4.0,
        preserve_leg_joints=True,
        preserve_root_trajectory=True,
    )

    assert np.allclose(qpos[:, :3], root_pos, atol=1.0e-7)
    assert np.array_equal(qpos[:, 3:7], root_quat[:, 0].astype(np.float32))
    assert not np.allclose(qpos[:, 7 + waist_index], joint_pos[:, waist_index], atol=1.0e-3)
    assert np.allclose(qpos[:, 7 + ankle_index], joint_pos[:, ankle_index], atol=1.0e-7)


def test_offline_urdf_fk_matches_existing_isaaclab_export():
    motion_file = (
        REPO_ROOT
        / "Ladder"
        / "motions"
        / "climing_ladder_cr1_downsample4_motion_annotated_wrist_d0145_v035.npz"
    )
    with np.load(motion_file, allow_pickle=False) as motion:
        frame_ids = np.asarray([0, 200, 400, 727])
        joint_names = motion["joint_names"].tolist()
        body_names = motion["body_names"].tolist()
        root_index = body_names.index("base_link")
        body_pos_w, body_quat_w = forward_kinematics(
            CR1_URDF,
            motion["body_pos_w"][frame_ids, root_index],
            motion["body_quat_w"][frame_ids, root_index],
            motion["joint_pos"][frame_ids],
            joint_names,
            body_names,
        )

        assert np.allclose(body_pos_w, motion["body_pos_w"][frame_ids], atol=2.0e-6)
        alignment = np.abs(np.sum(body_quat_w * motion["body_quat_w"][frame_ids], axis=-1))
        assert np.all(alignment > 1.0 - 5.0e-7)


def test_group_smoothed_50hz_motion_is_training_shaped_and_finite():
    motion_file = (
        REPO_ROOT
        / "Ladder"
        / "motions"
        / "climing_ladder_cr1_leg3hz_other5hz_50fps_annotated_wrist_d0145_v035.npz"
    )
    with np.load(motion_file, allow_pickle=False) as motion:
        assert motion["fps"].tolist() == [50.0]
        assert motion["joint_pos"].shape == (1213, 25)
        assert motion["body_pos_w"].shape == (1213, 42, 3)
        assert motion["contact_indicator"].shape == (1213, 4)
        assert motion["contact_rung_id"].shape == (1213, 4)
        for key in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "contact_indicator",
        ):
            assert np.isfinite(motion[key]).all()
        assert motion["contact_annotation_ladder_num_rungs"].tolist() == [7]
        assert np.allclose(motion["contact_annotation_ladder_origin"], [0.445, 0.04, 0.19])
        assert motion["motion_processing_joint_cutoff_hz"].tolist() == [5.0]
        assert motion["motion_processing_leg_joint_cutoff_hz"].tolist() == [3.0]


def test_contact_transition_frames_are_stable_for_fixed_frame_smoke_checks():
    motion_file = (
        REPO_ROOT
        / "Ladder"
        / "motions"
        / "climing_ladder_cr1_leg3hz_other5hz_50fps_annotated_wrist_d0145_v035.npz"
    )
    expected = {
        "left_hand": [228, 368, 389, 631, 661, 910, 932, 1079],
        "right_hand": [235, 244, 282, 499, 524, 776, 799, 1082],
        "left_foot": [344, 561, 597, 844, 881],
        "right_foot": [470, 701, 744, 976, 1021],
    }
    with np.load(motion_file, allow_pickle=False) as motion:
        indicator = motion["contact_indicator"] > 0.5
        for limb_index, limb_name in enumerate(motion["limb_names"].tolist()):
            transitions = np.flatnonzero(indicator[1:, limb_index] != indicator[:-1, limb_index]) + 1
            assert transitions.tolist() == expected[limb_name]


def test_arc_length_retimed_preview_distributes_hand_motion_and_preserves_start():
    motion_file = (
        REPO_ROOT
        / "Ladder"
        / "motions"
        / "restored_prefix_v300_stride2_slow50"
        / "smooth_start"
        / "reference_repaired"
        / "climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_arc_uniform_contact_safe_preview.npz"
    )
    segments = (
        (440, 555, "right_sphere_hand_link"),
        (830, 1000, "right_sphere_hand_link"),
        (1121, 1193, "left_sphere_hand_link"),
    )
    with np.load(motion_file, allow_pickle=False) as motion:
        assert motion["joint_pos"].shape == (1386, 25)
        assert np.all(np.diff(motion["arc_retime_source_time"]) > 0.0)
        assert np.all(motion["joint_pos"][:25] == motion["joint_pos"][0])
        assert np.all(motion["contact_rung_id"][motion["contact_indicator"] <= 0.5] == -1)

        body_names = motion["body_names"].tolist()
        for start, end, body_name in segments:
            body_pos = motion["body_pos_w"][start : end + 1, body_names.index(body_name)]
            step_distance = np.linalg.norm(np.diff(body_pos, axis=0), axis=-1)
            assert step_distance.max() / step_distance.mean() < 2.0
