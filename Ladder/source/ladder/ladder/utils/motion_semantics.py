"""Direction-safe labels for recurrent ladder motion semantics."""

from __future__ import annotations

import numpy as np


def direction_bounded_limb_goals(
    contact_indicator: np.ndarray,
    contact_rung_id: np.ndarray,
    climb_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return current/next rung labels without crossing a command boundary."""

    contact = np.asarray(contact_indicator) > 0.5
    rung_id = np.asarray(contact_rung_id, dtype=np.int64)
    direction = np.asarray(climb_direction, dtype=np.float32).reshape(-1)
    if contact.ndim != 2 or rung_id.shape != contact.shape:
        raise ValueError("contact_indicator and contact_rung_id must have matching [frame, limb] shapes")
    if direction.shape != (contact.shape[0],):
        raise ValueError("climb_direction must have one value per frame")

    goal_rung_id = np.full(contact.shape, -1, dtype=np.int64)
    goal_valid = np.zeros(contact.shape, dtype=np.bool_)
    for start, end in contiguous_direction_runs(direction):
        for limb_index in range(contact.shape[1]):
            next_rung_id = -1
            for frame_index in range(end - 1, start - 1, -1):
                if contact[frame_index, limb_index]:
                    next_rung_id = int(rung_id[frame_index, limb_index])
                if next_rung_id >= 0:
                    goal_rung_id[frame_index, limb_index] = next_rung_id
                    goal_valid[frame_index, limb_index] = True
    return goal_rung_id, goal_valid


def stable_motion_target_frames(
    climb_direction: np.ndarray,
    stable_anchor_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each frame to a direction-local stable target and each anchor to its successor.

    A reference frame targets the first stable anchor at or after that frame in
    the same contiguous command run. Frames after the run's final anchor keep
    that final stable pose. An anchor's successor is the next anchor in the
    same run, or itself for a terminal anchor.
    """

    direction = np.asarray(climb_direction, dtype=np.float32).reshape(-1)
    anchors = np.unique(np.asarray(stable_anchor_frames, dtype=np.int64).reshape(-1))
    if direction.size == 0:
        raise ValueError("climb_direction must not be empty")
    if anchors.size == 0:
        raise ValueError("stable_anchor_frames must not be empty")
    if np.any(anchors < 0) or np.any(anchors >= direction.size):
        raise ValueError("stable anchor frame lies outside the motion")

    initial_target = np.full(direction.shape, -1, dtype=np.int64)
    successor = np.arange(direction.size, dtype=np.int64)
    for start, end in contiguous_direction_runs(direction):
        run_anchors = anchors[(anchors >= start) & (anchors < end)]
        if run_anchors.size == 0:
            raise ValueError(
                f"command run [{start}, {end}) has no stable motion anchor"
            )
        frames = np.arange(start, end, dtype=np.int64)
        target_indexes = np.searchsorted(run_anchors, frames, side="left")
        target_indexes = np.minimum(target_indexes, run_anchors.size - 1)
        initial_target[start:end] = run_anchors[target_indexes]
        if run_anchors.size > 1:
            successor[run_anchors[:-1]] = run_anchors[1:]
        successor[run_anchors[-1]] = run_anchors[-1]

    if np.any(initial_target < 0):
        raise RuntimeError("not every reference frame received a stable motion target")
    return initial_target, successor


def contiguous_direction_runs(climb_direction: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs over an ordered one-dimensional command array."""

    direction = np.asarray(climb_direction).reshape(-1)
    if direction.size == 0:
        return []
    boundaries = np.flatnonzero(direction[1:] != direction[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
    ends = np.concatenate((boundaries, np.array([direction.size], dtype=np.int64)))
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


__all__ = [
    "contiguous_direction_runs",
    "direction_bounded_limb_goals",
    "stable_motion_target_frames",
]
