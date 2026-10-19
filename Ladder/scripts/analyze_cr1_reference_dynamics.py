"""Read-only dynamics audit for a CR1 reference NPZ.

This tool locates joint-space velocity, acceleration, and jerk peaks.  It does not
modify the reference.  Use repeated ``--segment START:END`` arguments to compare
visually suspicious regions before deciding whether to repair them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_BODY_NAMES = (
    "base_link",
    "left_sphere_hand_link",
    "right_sphere_hand_link",
    "left_sole_contact_point",
    "right_sole_contact_point",
)


def _segment(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split(":")
        start, end = int(start_text), int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected START:END") from exc
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("require 0 <= START <= END")
    return start, end


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_file", type=Path)
    parser.add_argument("--segment", action="append", type=_segment, default=[])
    parser.add_argument(
        "--body",
        action="append",
        default=[],
        help="Body to audit; repeat as needed. Defaults to base, sphere hands, and sole points.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def _names(values: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(values).reshape(-1).tolist()]


def _top_events(
    values: np.ndarray,
    names: list[str],
    frame_offset: int,
    top_k: int,
) -> list[str]:
    if values.size == 0:
        return []
    flat_indexes = np.argpartition(np.abs(values).reshape(-1), -min(top_k, values.size))[
        -min(top_k, values.size) :
    ]
    flat_indexes = flat_indexes[np.argsort(np.abs(values).reshape(-1)[flat_indexes])[::-1]]
    events = []
    for flat_index in flat_indexes:
        row, column = np.unravel_index(flat_index, values.shape)
        events.append(
            f"frame={row + frame_offset} joint={names[column]} value={values[row, column]:+.3f}"
        )
    return events


def _report(
    joint_pos: np.ndarray,
    joint_names: list[str],
    body_pos: np.ndarray,
    body_names: list[str],
    fps: float,
    start: int,
    end: int,
    top_k: int,
) -> None:
    # Include preceding samples so derivatives reported at start are not silently lost.
    clip_start = max(0, start - 3)
    positions = joint_pos[clip_start : end + 1]
    velocity = np.diff(positions, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    derivative_data = (
        ("velocity rad/s", velocity, clip_start + 1),
        ("acceleration rad/s^2", acceleration, clip_start + 2),
        ("jerk rad/s^3", jerk, clip_start + 3),
    )
    print(f"[SEGMENT] {start}:{end}")
    for label, values, frame_offset in derivative_data:
        first_row = max(0, start - frame_offset)
        selected = values[first_row : end - frame_offset + 1]
        selected_offset = frame_offset + first_row
        peak = float(np.max(np.abs(selected))) if selected.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(selected)))) if selected.size else 0.0
        print(f"  {label}: peak={peak:.3f}, rms={rms:.3f}")
        for event in _top_events(selected, joint_names, selected_offset, top_k):
            print(f"    {event}")

    body_velocity = np.diff(body_pos[clip_start : end + 1], axis=0) * fps
    body_acceleration = np.diff(body_velocity, axis=0) * fps
    body_jerk = np.diff(body_acceleration, axis=0) * fps
    body_derivative_data = (
        ("body speed m/s", body_velocity, clip_start + 1),
        ("body acceleration m/s^2", body_acceleration, clip_start + 2),
        ("body jerk m/s^3", body_jerk, clip_start + 3),
    )
    for label, values, frame_offset in body_derivative_data:
        first_row = max(0, start - frame_offset)
        selected = values[first_row : end - frame_offset + 1]
        selected_offset = frame_offset + first_row
        magnitudes = np.linalg.norm(selected, axis=-1)
        if not magnitudes.size:
            continue
        flat_indexes = np.argpartition(
            magnitudes.reshape(-1), -min(top_k, magnitudes.size)
        )[-min(top_k, magnitudes.size) :]
        flat_indexes = flat_indexes[
            np.argsort(magnitudes.reshape(-1)[flat_indexes])[::-1]
        ]
        print(
            f"  {label}: peak={float(magnitudes.max()):.3f}, "
            f"rms={float(np.sqrt(np.mean(np.square(magnitudes)))):.3f}"
        )
        for flat_index in flat_indexes:
            row, column = np.unravel_index(flat_index, magnitudes.shape)
            print(
                f"    frame={row + selected_offset} body={body_names[column]} "
                f"value={magnitudes[row, column]:.3f}"
            )


def main() -> None:
    args = _parse_args()
    if args.top_k < 1:
        raise ValueError("top_k must be at least 1")
    motion_file = args.motion_file.expanduser().resolve()
    with np.load(motion_file, allow_pickle=False) as data:
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        joint_names = _names(data["joint_names"])
        all_body_names = _names(data["body_names"])
        requested_body_names = args.body or list(DEFAULT_BODY_NAMES)
        missing_body_names = [name for name in requested_body_names if name not in all_body_names]
        if missing_body_names:
            raise ValueError(f"motion is missing requested bodies: {missing_body_names}")
        body_indexes = [all_body_names.index(name) for name in requested_body_names]
        body_pos = np.asarray(data["body_pos_w"][:, body_indexes], dtype=np.float64)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    frame_count = joint_pos.shape[0]
    segments = args.segment or [(0, frame_count - 1)]
    print(f"[MOTION] {motion_file}")
    print(f"[MOTION] frames={frame_count}, fps={fps:g}, duration={(frame_count - 1) / fps:.3f} s")
    for start, end in segments:
        if end >= frame_count:
            raise ValueError(f"segment {start}:{end} exceeds {frame_count} frames")
        _report(
            joint_pos,
            joint_names,
            body_pos,
            requested_body_names,
            fps,
            start,
            end,
            args.top_k,
        )


if __name__ == "__main__":
    main()
