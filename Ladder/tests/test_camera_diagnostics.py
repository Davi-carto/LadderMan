import numpy as np

from ladder.utils.camera_diagnostics import (
    _bgr_or_gray_to_rgba,
    _mask_excluded_pixels,
    _opencv_highgui_available,
    rfm_keep_mask_from_semantic_info,
    robot_ids_from_semantic_info,
    rung_ids_from_semantic_info,
    rung_mask_from_semantic_info,
    semantic_label_mapping,
)
from ladder.utils.robot_semantics import robot_link_prim_paths


def test_rung_mask_uses_runtime_semantic_labels_not_nonzero_ids():
    info = {
        "idToLabels": {
            "0": {"class": "BACKGROUND"},
            "3": {"class": "ladder"},
            "7": {"class": "rung"},
            "11": {"class": "ladder_rung_edge"},
        }
    }
    semantic = np.array([[[0], [3], [7]], [[11], [3], [0]]], dtype=np.uint32)

    assert rung_ids_from_semantic_info(info) == {7, 11}
    assert semantic_label_mapping(info) == {
        0: {"class": "BACKGROUND"},
        3: {"class": "ladder"},
        7: {"class": "rung"},
        11: {"class": "ladder_rung_edge"},
    }
    assert rung_mask_from_semantic_info(semantic, info).tolist() == [
        [False, False, True],
        [True, False, False],
    ]


def test_rung_mask_is_empty_when_mapping_has_no_rung_label():
    info = {"idToLabels": {"4": {"class": "ladder"}}}
    semantic = np.array([[4, 9]], dtype=np.uint32)

    assert not rung_mask_from_semantic_info(semantic, info).any()


def test_rfm_keep_mask_retains_rungs_and_robot_only():
    info = {
        "idToLabels": {
            "0": {"class": "BACKGROUND"},
            "1": {"class": "UNLABELLED"},
            "2": {"class": "rung"},
            "3": {"class": "ladder"},
            "4": {"class": "robot"},
        }
    }
    semantic = np.array([[[0], [1], [2]], [[3], [4], [2]]], dtype=np.uint32)

    assert robot_ids_from_semantic_info(info) == {4}
    assert rfm_keep_mask_from_semantic_info(semantic, info).tolist() == [
        [False, False, True],
        [False, True, True],
    ]


def test_robot_link_paths_include_every_link_in_every_environment():
    assert robot_link_prim_paths(
        ["/World/envs/env_0/Robot", "/World/envs/env_1/Robot"],
        ("body", "left_arm"),
    ) == (
        "/World/envs/env_0/Robot/body",
        "/World/envs/env_0/Robot/left_arm",
        "/World/envs/env_1/Robot/body",
        "/World/envs/env_1/Robot/left_arm",
    )


def test_opencv_highgui_detection_handles_headless_and_qt_builds():
    assert not _opencv_highgui_available("General configuration\n  GUI: NONE\n")
    assert _opencv_highgui_available("General configuration\n  GUI: QT5\n")


def test_omni_rgba_conversion_preserves_display_colors():
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    gray = np.array([[40]], dtype=np.uint8)

    assert _bgr_or_gray_to_rgba(bgr).tolist() == [[[30, 20, 10, 255]]]
    assert _bgr_or_gray_to_rgba(gray).tolist() == [[[40, 40, 40, 255]]]


def test_masked_depth_display_keeps_selected_pixels():
    depth_color = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8,
    )
    keep_mask = np.array([[True, False], [False, True]])

    assert _mask_excluded_pixels(depth_color, keep_mask).tolist() == [
        [[1, 2, 3], [0, 0, 0]],
        [[0, 0, 0], [10, 11, 12]],
    ]
