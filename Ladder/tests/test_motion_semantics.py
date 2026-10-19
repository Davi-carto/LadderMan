from pathlib import Path

import numpy as np

from ladder.utils.motion_semantics import (
    direction_bounded_limb_goals,
    stable_motion_target_frames,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "zdw0824_capture1"
    / "pitchminus3_yawminus4_xminus030"
    / "37_visual_up_down_stop_command"
    / (
        "zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_"
        "reset5mm_visual_up_down_stop_50hz_motion.npz"
    )
)


def test_limb_goals_do_not_cross_up_down_boundary():
    with np.load(MOTION, allow_pickle=False) as motion:
        rung_id, valid = direction_bounded_limb_goals(
            motion["contact_indicator"],
            motion["contact_rung_id"],
            motion["climb_direction"],
        )

    assert not np.any(valid[1291:1550, 0])
    assert not np.any(valid[1265:1550, 1])
    assert np.all(valid[1550:1600, :2])
    assert np.all(rung_id[1550:1600, :2] == 6)
    assert np.all(valid[1500:1600, 2:])
    assert np.all(rung_id[1500:1600, 2:] == 2)


def test_stable_targets_are_constant_until_completion_and_direction_local():
    with np.load(MOTION, allow_pickle=False) as motion:
        target, successor = stable_motion_target_frames(
            motion["climb_direction"], motion["zero_command_direct_hold_frames"]
        )

    assert target[0] == 0
    assert np.all(target[1:618] == 617)
    assert successor[0] == 617
    assert successor[617] == 656
    assert np.all(target[1500:1550] == 1500)
    assert successor[1500] == 1500
    assert np.all(target[1550:1820] == 1819)
    assert successor[1819] == 1963
    assert np.all(target[3008:] == 3008)
    assert successor[3008] == 3008
