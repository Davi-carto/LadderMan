import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "zdw0824_capture1"
    / "pitchminus3_yawminus4_xminus030"
    / "36_middle_default_standing_100_zplus5mm"
    / (
        "zdw0824_capture1_stage36_train_ready_middle_default100_zplus5mm_"
        "reset5mm_50hz_motion.npz"
    )
)
COMMAND_MOTION = (
    SOURCE_MOTION.parents[1]
    / "37_visual_up_down_stop_command"
    / (
        "zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_"
        "reset5mm_visual_up_down_stop_50hz_motion.npz"
    )
)


def _load_annotation_script():
    path = REPO_ROOT / "Ladder" / "scripts" / "annotate_cr1_climb_commands.py"
    spec = importlib.util.spec_from_file_location("annotate_cr1_climb_commands", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage37_only_appends_command_arrays_to_stage36():
    with np.load(SOURCE_MOTION, allow_pickle=False) as source, np.load(
        COMMAND_MOTION, allow_pickle=False
    ) as command:
        assert len(source.files) == 187
        assert len(command.files) == 240
        assert set(source.files) < set(command.files)
        for name in source.files:
            np.testing.assert_array_equal(command[name], source[name])


def test_stage37_command_direction_and_stop_candidates_are_valid():
    with np.load(COMMAND_MOTION, allow_pickle=False) as motion:
        assert motion["climb_direction"].shape == (3061, 1)
        assert np.all(motion["climb_direction"][:1550] == 1.0)
        assert np.all(motion["climb_direction"][1550:] == -1.0)
        assert motion[
            "zero_command_all_limb_contact_segment_selected_frames"
        ].tolist() == [
            617,
            656,
            780,
            798,
            909,
            971,
            1094,
            1142,
            1233,
            1819,
            1963,
            1987,
            2125,
            2159,
            2306,
            2339,
            2442,
            2516,
        ]
        assert np.all(motion["zero_command_valid_mask"][:25])
        assert np.all(motion["zero_command_valid_mask"][1500:1600])
        assert np.all(motion["zero_command_valid_mask"][2991:3061])
        direct_frames = motion["zero_command_direct_hold_frames"]
        assert np.all(motion["zero_command_valid_mask"][direct_frames])
        geometry_masks = motion["reset_ladder_geometry_cell_valid_mask"]
        assert np.all(np.any(geometry_masks[..., direct_frames], axis=-1))

        stable_frames = motion["stable_support_anchor_frames"]
        assert stable_frames.tolist() == [
            0,
            285,
            373,
            456,
            514,
            607,
            686,
            765,
            824,
            916,
            996,
            1077,
            1142,
            1237,
            1500,
            1837,
            1935,
            2001,
            2092,
            2165,
            2287,
            2357,
            2434,
            2510,
            2597,
            2664,
            2729,
            2794,
            3060,
        ]
        stable_contact = motion["stable_support_contact_indicator"] > 0.5
        stable_rung = motion["stable_support_rung_id"]
        full_support = motion["stable_support_anchor_is_full_support"]
        assert np.all(stable_contact[stable_frames[full_support]])
        assert motion["stable_support_anchor_kind"].tolist() == [
            0,
            *([1] * 13),
            2,
            *([1] * 13),
            3,
        ]
        assert motion["stable_support_reset_segment_starts"].tolist() == stable_frames[:-1].tolist()
        assert motion["stable_support_reset_segment_ends"].tolist() == (
            (stable_frames[1:-1] - 1).tolist() + [3060]
        )
        np.testing.assert_array_equal(
            motion["stable_support_reset_segment_ids"],
            np.searchsorted(stable_frames[:-1], np.arange(3061), side="right") - 1,
        )
        assert motion["stable_support_contact_indicator"][0].tolist() == [0.0, 0.0, 1.0, 1.0]
        assert motion["stable_support_contact_indicator"][1500].tolist() == [0.0, 0.0, 1.0, 1.0]
        assert motion["stable_support_contact_indicator"][3060].tolist() == [0.0, 0.0, 1.0, 1.0]
        assert stable_rung[285].tolist() == [4, 4, 0, 0]
        assert stable_rung[607].tolist() == [5, 5, 1, 1]
        assert stable_rung[1237].tolist() == [7, 7, 3, 3]
        assert stable_rung[2794].tolist() == [4, 4, 0, 0]
        assert np.all(
            motion["stable_support_anchor_max_limb_speeds"][full_support] <= 0.10
        )
        assert motion["stable_support_anchor_surface_distances"].shape == (29, 4)
        assert np.isclose(
            motion["stable_support_ground_surface_distance_threshold"][0], 0.015
        )
        frame_514_index = stable_frames.tolist().index(514)
        assert motion["stable_support_anchor_surface_distances"][
            frame_514_index, 0
        ] < 0.005
        np.testing.assert_array_equal(
            stable_rung[motion["contact_indicator"] > 0.5],
            motion["contact_rung_id"][motion["contact_indicator"] > 0.5] + 1,
        )


def test_stage37_artifact_matches_reusable_annotation_function():
    annotation = _load_annotation_script()
    with np.load(SOURCE_MOTION, allow_pickle=False) as source:
        source_dict = {name: source[name] for name in source.files}
    rebuilt = annotation.build_command_annotations(
        source_dict,
        source_file=SOURCE_MOTION.resolve(),
    )
    with np.load(COMMAND_MOTION, allow_pickle=False) as command:
        assert set(rebuilt) == set(command.files)
        for name, values in rebuilt.items():
            np.testing.assert_array_equal(values, command[name])
