"""Repair localized CR1 reference jumps by stretching selected frame intervals.

The tool deliberately changes only time parametrization in selected intervals.  It
does not move the ladder, change the robot model, or invent a new IK solution.  This
makes it a conservative first pass for retargeted references: the source path is
interpolated without changing the output rate, while large frame-to-frame commands
are spread over more samples.  A smooth transition at segment boundaries avoids
introducing a new acceleration discontinuity.  Contact phases are time-warped with
their source frames.  Stale reset and derived contact diagnostics are removed and
must be regenerated afterwards.

Segment syntax is ``START:END:FACTOR`` (both bounds inclusive, factor >= 2).  For
example ``775:835:3`` turns every source interval in [775, 835] into three output
intervals.  The output remains at the input fps, so the selected motion takes longer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ladder.utils.urdf_kinematics import finite_difference, forward_kinematics, quaternion_angular_velocity


DEFAULT_INPUT = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "restored_prefix_v300_stride2_slow50"
    / "smooth_start"
    / "ladder_geometry_dr"
    / "climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_train_ready_ladder_dr.npz"
)
DEFAULT_ROBOT_ASSET = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)


def _parse_segment(value: str) -> tuple[int, int, int]:
    try:
        start_text, end_text, factor_text = value.split(":")
        start, end, factor = int(start_text), int(end_text), int(factor_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid segment {value!r}; expected START:END:FACTOR"
        ) from exc
    if start < 0 or end < start or factor < 2:
        raise argparse.ArgumentTypeError(
            f"invalid segment {value!r}; require 0 <= START <= END and FACTOR >= 2"
        )
    return start, end, factor


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
        metavar="START:END:FACTOR",
        help="Selected interval to stretch; repeat for multiple non-overlapping intervals.",
    )
    parser.add_argument(
        "--transition_frames",
        type=int,
        default=5,
        help="Source-frame width of the smooth speed transition at each segment boundary.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _names(values: np.ndarray) -> list[str]:
    result = []
    for value in np.asarray(values).reshape(-1).tolist():
        result.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return result


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _expanded_indices(
    frame_count: int,
    segments: list[tuple[int, int, int]],
    transition_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source indices and fractions for a smoothly stretched timeline."""

    if transition_frames < 0:
        raise ValueError("transition_frames must be non-negative")
    interval_scales = np.ones(frame_count - 1, dtype=np.float64)
    occupied = np.zeros(frame_count - 1, dtype=np.bool_)
    for start, end, factor in segments:
        if end >= frame_count:
            raise ValueError(f"segment {start}:{end}:{factor} exceeds {frame_count} frames")
        if np.any(occupied[start:end]):
            raise ValueError("segments must not overlap")
        occupied[start:end] = True
        interval_count = end - start
        if transition_frames == 0:
            weight = np.ones(interval_count, dtype=np.float64)
        else:
            centers = np.arange(interval_count, dtype=np.float64) + 0.5
            transition = min(float(transition_frames), 0.5 * interval_count)
            edge_progress = np.minimum(centers, interval_count - centers) / transition
            weight = _smoothstep(edge_progress)
        interval_scales[start:end] = 1.0 + (factor - 1.0) * weight

    source_output_time = np.concatenate(([0.0], np.cumsum(interval_scales)))
    output_intervals = int(round(source_output_time[-1]))
    output_time = np.linspace(0.0, source_output_time[-1], output_intervals + 1)
    source_time = PchipInterpolator(
        source_output_time,
        np.arange(frame_count, dtype=np.float64),
    )(output_time)
    source_time = np.clip(source_time, 0.0, frame_count - 1.0)
    source_indices = np.floor(source_time).astype(np.int64)
    fractions = source_time - source_indices
    # The final source frame is represented as the end of the last interval.
    source_indices[-1] = frame_count - 2
    fractions[-1] = 1.0
    return source_indices, fractions


def _interpolate_root_quaternion(quaternions_wxyz: np.ndarray, source_indices: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    quaternions_wxyz = np.asarray(quaternions_wxyz, dtype=np.float64)
    quaternions_wxyz = quaternions_wxyz / np.linalg.norm(quaternions_wxyz, axis=-1, keepdims=True)
    # Keep adjacent source rotations on the same hemisphere before Slerp.
    signs = np.ones(quaternions_wxyz.shape[0], dtype=np.float64)
    signs[1:] = np.where(
        np.sum(quaternions_wxyz[:-1] * quaternions_wxyz[1:], axis=-1) < 0.0,
        -1.0,
        1.0,
    )
    quaternions_wxyz = quaternions_wxyz * np.cumprod(signs)[:, None]
    source_rotations = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]])
    result = np.empty((len(source_indices), 4), dtype=np.float64)
    for index, (source_index, fraction) in enumerate(zip(source_indices, fractions, strict=True)):
        if fraction == 0.0:
            result[index] = quaternions_wxyz[source_index]
            continue
        rotation = Slerp(
            [0.0, 1.0], Rotation.concatenate([source_rotations[source_index], source_rotations[source_index + 1]])
        )([fraction])[0]
        xyzw = rotation.as_quat()
        result[index] = xyzw[[3, 0, 1, 2]]
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return result.astype(np.float32)


def _pchip_expand(values: np.ndarray, source_indices: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    source_time = np.arange(values.shape[0], dtype=np.float64)
    output_time = source_indices.astype(np.float64) + fractions
    return PchipInterpolator(source_time, np.asarray(values, dtype=np.float64), axis=0)(output_time)


def expand_motion_timeline(
    motion: dict[str, np.ndarray],
    segments: list[tuple[int, int, int]],
    transition_frames: int,
    robot_asset: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Expand selected intervals and rebuild all kinematic fields from root/joints."""

    required = ("fps", "joint_pos", "body_pos_w", "body_quat_w", "joint_names", "body_names")
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"motion is missing required keys: {missing}")

    frame_count = int(np.asarray(motion["joint_pos"]).shape[0])
    source_indices, fractions = _expanded_indices(frame_count, segments, transition_frames)
    joint_pos_source = np.asarray(motion["joint_pos"], dtype=np.float64)
    joint_pos = _pchip_expand(joint_pos_source, source_indices, fractions).astype(np.float32)

    body_names = _names(motion["body_names"])
    joint_names = _names(motion["joint_names"])
    root_index = body_names.index("base_link")
    source_root_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)[:, root_index]
    root_pos = _pchip_expand(source_root_pos, source_indices, fractions).astype(np.float32)
    root_quat = _interpolate_root_quaternion(
        np.asarray(motion["body_quat_w"], dtype=np.float64)[:, root_index], source_indices, fractions
    )

    source_joint_names = _names(motion.get("source_joint_names", motion["joint_names"]))
    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset, root_pos, root_quat, joint_pos, source_joint_names, body_names
    )
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    categorical_indices = source_indices + np.isclose(fractions, 1.0).astype(np.int64)
    categorical_indices = np.minimum(categorical_indices, frame_count - 1)
    output = {}
    for key, value in motion.items():
        if key.startswith("reset_") or key in {
            "contact_nearest_distance",
            "contact_speed",
            "contact_raw_indicator",
        }:
            continue
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == frame_count:
            output[key] = array[categorical_indices]
        else:
            output[key] = array.copy()
    if "limb_body_indexes" in output:
        limb_indexes = np.asarray(output["limb_body_indexes"], dtype=np.int64)
        limb_velocity = finite_difference(body_pos_w[:, limb_indexes], fps)
        output["contact_speed"] = np.linalg.norm(limb_velocity, axis=-1).astype(np.float32)
    output.update(
        {
            "joint_pos": joint_pos,
            "joint_vel": finite_difference(joint_pos, fps),
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": finite_difference(body_pos_w, fps),
            "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
            "reference_repair_source_frame": source_indices,
            "reference_repair_source_fraction": fractions.astype(np.float32),
            "reference_repair_source_time": (source_indices + fractions).astype(np.float32),
            "reference_repair_segment_specs": np.asarray(
                [f"{start}:{end}:{factor}" for start, end, factor in segments]
            ),
            "reference_repair_method": np.asarray(["tapered_pchip_time_stretch_with_fk"]),
            "reference_repair_transition_frames": np.asarray([transition_frames], dtype=np.int64),
            "reference_repair_robot_asset": np.asarray([str(robot_asset)]),
            "reference_repair_source_frame_count": np.asarray([frame_count], dtype=np.int64),
        }
    )
    return output, source_indices


def main() -> None:
    args = _parse_args()
    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    robot_asset = args.robot_asset.expanduser().resolve()
    if not robot_asset.is_file():
        raise FileNotFoundError(robot_asset)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"output file exists: {output_file}; pass --overwrite")
    with np.load(input_file, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}
    expanded, source_indices = expand_motion_timeline(
        motion,
        args.segment,
        args.transition_frames,
        robot_asset,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **expanded)
    print(f"[INFO] Wrote repaired reference: {output_file}")
    print(
        f"[INFO] frames {len(source_indices)} (source {len(motion['joint_pos'])}), "
        f"added {len(source_indices) - len(motion['joint_pos'])}, fps={float(np.asarray(motion['fps']).reshape(-1)[0]):g}"
    )
    print(f"[INFO] segments: {', '.join(f'{a}:{b}:{f}' for a, b, f in args.segment)}")
    print(f"[INFO] boundary transition: {args.transition_frames} source frames")


if __name__ == "__main__":
    main()
