import numpy as np
from scipy.spatial.transform import Rotation

from ladder.utils.motion_world_transform import align_motion_ladder_frame


def test_align_motion_ladder_frame_rotates_world_fields_and_preserves_joints():
    identity_wxyz = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    motion = {
        "joint_pos": np.asarray([[0.25, -0.5]], dtype=np.float32),
        "body_pos_w": np.asarray([[[0.0, 0.3, 0.2], [0.0, 1.3, 0.4]]], dtype=np.float32),
        "body_quat_w": np.asarray([[identity_wxyz, identity_wxyz]], dtype=np.float32),
        "body_lin_vel_w": np.asarray([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32),
        "body_ang_vel_w": np.asarray([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32),
    }

    output = align_motion_ladder_frame(
        motion,
        source_ladder_position=(0.0, 0.3, 0.0),
        source_ladder_yaw_deg=180.0,
        target_ladder_position=(0.38, 0.04, 0.0),
        target_ladder_yaw_deg=90.0,
        robot_translation=(0.0, 0.0, -0.005),
    )

    assert np.array_equal(output["joint_pos"], motion["joint_pos"])
    assert np.allclose(output["body_pos_w"][0, 0], [0.38, 0.04, 0.195], atol=1.0e-7)
    assert np.allclose(output["body_pos_w"][0, 1], [1.38, 0.04, 0.395], atol=1.0e-7)
    assert np.allclose(output["body_lin_vel_w"][0, 0], [1.0, 0.0, 0.0], atol=1.0e-7)
    assert np.allclose(output["body_lin_vel_w"][0, 1], [0.0, -1.0, 0.0], atol=1.0e-7)
    expected_xyzw = Rotation.from_euler("z", -90.0, degrees=True).as_quat()
    expected_wxyz = expected_xyzw[[3, 0, 1, 2]]
    alignment = np.abs(np.sum(output["body_quat_w"] * expected_wxyz, axis=-1))
    assert np.allclose(alignment, 1.0, atol=1.0e-7)
