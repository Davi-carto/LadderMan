"""Redistribute localized CR1 motion uniformly along an end-effector path.

Unlike uniform time stretching, this tool allocates output samples according to
spatial path progress.  It is intended for intervals that contain a long dwell and
a short burst of motion.  The robot/ladder geometry and source joint-space path are
not changed; only the mapping from output time to source time is changed.

Segment syntax is ``START:END:OUTPUT_INTERVALS:BODY``.  Bounds are inclusive and
OUTPUT_INTERVALS must be at least ``END - START``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_cr1_reference_jumps import (  # noqa: E402
    DEFAULT_INPUT,
    DEFAULT_ROBOT_ASSET,
    _interpolate_root_quaternion,
    _names,
    _pchip_expand,
)

from ladder.utils.urdf_kinematics import (  # noqa: E402
    finite_difference,
    forward_kinematics,
    quaternion_angular_velocity,
)


COORDINATION_BODY_NAMES = (
    "base_link",
    "left_sphere_hand_link",
    "right_sphere_hand_link",
    "left_sole_contact_point",
    "right_sole_contact_point",
)


def _parse_segment(value: str) -> tuple[int, int, int, str]:
    try:
        start_text, end_text, intervals_text, body_name = value.split(":")
        start, end, output_intervals = int(start_text), int(end_text), int(intervals_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid segment {value!r}; expected START:END:OUTPUT_INTERVALS:BODY"
        ) from exc
    if start < 0 or end <= start or output_intervals < end - start or not body_name:
        raise argparse.ArgumentTypeError(
            "require 0 <= START < END and OUTPUT_INTERVALS >= END - START"
        )
    return start, end, output_intervals, body_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT_ASSET)
    parser.add_argument(
        "--segment",
        action="append",
        type=_parse_segment,
        required=True,
        metavar="START:END:OUTPUT_INTERVALS:BODY",
    )
    parser.add_argument(
        "--joint_weight_m_per_rad",
        type=float,
        default=0.005,
        help="Small joint-path contribution that preserves coordination during end-effector dwells.",
    )
    parser.add_argument(
        "--coordination_body_weight",
        type=float,
        default=1.0,
        help="Weight for the combined base, sphere-hand, and sole-point path length.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _segment_source_time(
    body_pos: np.ndarray,
    coordination_body_pos: np.ndarray,
    joint_pos: np.ndarray,
    start: int,
    end: int,
    output_intervals: int,
    joint_weight: float,
    coordination_body_weight: float,
) -> np.ndarray:
    body_distance = np.linalg.norm(np.diff(body_pos[start : end + 1], axis=0), axis=-1)
    coordination_delta = np.diff(coordination_body_pos[start : end + 1], axis=0)
    coordination_distance = np.linalg.norm(
        coordination_delta.reshape(coordination_delta.shape[0], -1), axis=-1
    )
    joint_distance = np.linalg.norm(np.diff(joint_pos[start : end + 1], axis=0), axis=-1)
    interval_distance = (
        body_distance
        + coordination_body_weight * coordination_distance
        + joint_weight * joint_distance
    )
    cumulative_distance = np.concatenate(([0.0], np.cumsum(interval_distance)))
    if cumulative_distance[-1] <= 1.0e-9:
        raise ValueError(f"segment {start}:{end} has no measurable motion")

    unique_distance, first_indexes = np.unique(cumulative_distance, return_index=True)
    source_at_distance = start + first_indexes.astype(np.float64)
    # Exact segment endpoints must remain exact even when a terminal dwell exists.
    source_at_distance[-1] = float(end)
    target_distance = np.linspace(0.0, cumulative_distance[-1], output_intervals + 1)
    source_time = np.interp(target_distance, unique_distance, source_at_distance)
    source_time[0] = float(start)
    source_time[-1] = float(end)
    return source_time


def _build_source_time(
    motion: dict[str, np.ndarray],
    segments: list[tuple[int, int, int, str]],
    joint_weight: float,
    coordination_body_weight: float,
) -> tuple[np.ndarray, list[str]]:
    frame_count = int(np.asarray(motion["joint_pos"]).shape[0])
    body_names = _names(motion["body_names"])
    missing_coordination_bodies = [
        name for name in COORDINATION_BODY_NAMES if name not in body_names
    ]
    if missing_coordination_bodies:
        raise ValueError(
            f"motion is missing coordination bodies: {missing_coordination_bodies}"
        )
    coordination_body_indexes = [body_names.index(name) for name in COORDINATION_BODY_NAMES]
    coordination_body_pos = np.asarray(
        motion["body_pos_w"][:, coordination_body_indexes], dtype=np.float64
    )
    joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    chunks: list[np.ndarray] = []
    specs: list[str] = []
    cursor = 0
    for start, end, output_intervals, body_name in sorted(segments):
        if start < cursor:
            raise ValueError("segments must not overlap")
        if end >= frame_count:
            raise ValueError(f"segment {start}:{end} exceeds {frame_count} frames")
        if body_name not in body_names:
            raise ValueError(f"body {body_name!r} is missing from the motion")
        chunks.append(np.arange(cursor, start, dtype=np.float64))
        body_pos = np.asarray(
            motion["body_pos_w"][:, body_names.index(body_name)], dtype=np.float64
        )
        chunks.append(
            _segment_source_time(
                body_pos,
                coordination_body_pos,
                joint_pos,
                start,
                end,
                output_intervals,
                joint_weight,
                coordination_body_weight,
            )
        )
        specs.append(f"{start}:{end}:{output_intervals}:{body_name}")
        cursor = end + 1
    chunks.append(np.arange(cursor, frame_count, dtype=np.float64))
    source_time = np.concatenate(chunks)
    if np.any(np.diff(source_time) <= 0.0):
        raise RuntimeError("constructed source timeline is not strictly increasing")
    return source_time, specs


def _retime_motion(
    motion: dict[str, np.ndarray],
    source_time: np.ndarray,
    segment_specs: list[str],
    joint_weight: float,
    coordination_body_weight: float,
    robot_asset: Path,
) -> dict[str, np.ndarray]:
    frame_count = int(np.asarray(motion["joint_pos"]).shape[0])
    source_indices = np.floor(source_time).astype(np.int64)
    fractions = source_time - source_indices
    source_indices[-1] = frame_count - 2
    fractions[-1] = 1.0

    joint_pos = _pchip_expand(motion["joint_pos"], source_indices, fractions).astype(np.float32)
    body_names = _names(motion["body_names"])
    joint_names = _names(motion["joint_names"])
    root_index = body_names.index("base_link")
    root_pos = _pchip_expand(
        np.asarray(motion["body_pos_w"])[:, root_index], source_indices, fractions
    ).astype(np.float32)
    root_quat = _interpolate_root_quaternion(
        np.asarray(motion["body_quat_w"])[:, root_index], source_indices, fractions
    )
    source_joint_names = _names(motion.get("source_joint_names", motion["joint_names"]))
    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset, root_pos, root_quat, joint_pos, source_joint_names, body_names
    )
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    categorical_indices = source_indices + np.isclose(fractions, 1.0).astype(np.int64)
    categorical_indices = np.minimum(categorical_indices, frame_count - 1)

    output: dict[str, np.ndarray] = {}
    for key, value in motion.items():
        if key.startswith("reset_") or key in {
            "contact_nearest_distance",
            "contact_speed",
            "contact_raw_indicator",
        }:
            continue
        array = np.asarray(value)
        output[key] = (
            array[categorical_indices]
            if array.ndim > 0 and array.shape[0] == frame_count
            else array.copy()
        )
    if "contact_indicator" in motion:
        contact_indicator = np.asarray(motion["contact_indicator"])[categorical_indices].copy()
        interior = (fractions > 1.0e-8) & (fractions < 1.0 - 1.0e-8)
        contact_indicator[interior] = np.minimum(
            np.asarray(motion["contact_indicator"])[source_indices[interior]],
            np.asarray(motion["contact_indicator"])[source_indices[interior] + 1],
        )
        output["contact_indicator"] = contact_indicator
        if "contact_rung_id" in motion:
            contact_rung_id = np.asarray(motion["contact_rung_id"])[categorical_indices].copy()
            contact_rung_id[contact_indicator <= 0.5] = -1
            output["contact_rung_id"] = contact_rung_id
    limb_indexes = np.asarray(output["limb_body_indexes"], dtype=np.int64)
    output.update(
        {
            "joint_pos": joint_pos,
            "joint_vel": finite_difference(joint_pos, fps),
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": finite_difference(body_pos_w, fps),
            "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
            "contact_speed": np.linalg.norm(
                finite_difference(body_pos_w[:, limb_indexes], fps), axis=-1
            ).astype(np.float32),
            "arc_retime_source_time": source_time.astype(np.float32),
            "arc_retime_source_frame": source_indices,
            "arc_retime_source_fraction": fractions.astype(np.float32),
            "arc_retime_segment_specs": np.asarray(segment_specs),
            "arc_retime_joint_weight_m_per_rad": np.asarray([joint_weight], dtype=np.float32),
            "arc_retime_coordination_body_weight": np.asarray(
                [coordination_body_weight], dtype=np.float32
            ),
            "arc_retime_coordination_body_names": np.asarray(COORDINATION_BODY_NAMES),
            "arc_retime_method": np.asarray(
                ["coordinated_end_effector_arc_length_pchip_fk_contact_safe"]
            ),
            "arc_retime_robot_asset": np.asarray([str(robot_asset)]),
            "arc_retime_source_frame_count": np.asarray([frame_count], dtype=np.int64),
        }
    )
    return output


def main() -> None:
    args = _parse_args()
    if args.joint_weight_m_per_rad < 0.0:
        raise ValueError("joint_weight_m_per_rad must be non-negative")
    if args.coordination_body_weight < 0.0:
        raise ValueError("coordination_body_weight must be non-negative")
    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    robot_asset = args.robot_asset.expanduser().resolve()
    for path in (input_file, robot_asset):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"output file exists: {output_file}; pass --overwrite")
    with np.load(input_file, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}
    source_time, segment_specs = _build_source_time(
        motion,
        args.segment,
        args.joint_weight_m_per_rad,
        args.coordination_body_weight,
    )
    output = _retime_motion(
        motion,
        source_time,
        segment_specs,
        args.joint_weight_m_per_rad,
        args.coordination_body_weight,
        robot_asset,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **output)
    print(f"[INFO] Wrote arc-length-retimed reference: {output_file}")
    print(
        f"[INFO] frames {len(source_time)} (source {len(motion['joint_pos'])}), "
        f"added {len(source_time) - len(motion['joint_pos'])}, "
        f"fps={float(np.asarray(motion['fps']).reshape(-1)[0]):g}"
    )
    print(f"[INFO] segments: {', '.join(segment_specs)}")


if __name__ == "__main__":
    main()
