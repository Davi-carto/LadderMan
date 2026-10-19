"""Smooth an exported CR1 motion and resample its root/joints to a uniform output rate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.signal import butter, sosfiltfilt
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
    / "climing_ladder_cr1_downsample4_motion_annotated_wrist_d0145_v035.npz"
)
DEFAULT_CR1_URDF = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)
CR1_LEG_JOINT_NAMES = (
    "left_hip_y_joint",
    "left_hip_x_joint",
    "left_hip_z_joint",
    "left_knee_joint",
    "left_ankle_y_joint",
    "left_ankle_x_joint",
    "right_hip_y_joint",
    "right_hip_x_joint",
    "right_hip_z_joint",
    "right_knee_joint",
    "right_ankle_y_joint",
    "right_ankle_x_joint",
)
CR1_ANKLE_JOINT_NAMES = (
    "left_ankle_y_joint",
    "left_ankle_x_joint",
    "right_ankle_y_joint",
    "right_ankle_x_joint",
)
JOINT_LIMIT_PROJECTION_WIDTH = 0.008


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth and resample CR1 root/joint trajectories.")
    parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT, help="Input CR1 motion npz.")
    parser.add_argument("--output_file", type=Path, required=True, help="Output full Ladder motion npz.")
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_CR1_URDF, help="CR1 URDF for offline FK.")
    parser.add_argument("--output_fps", type=float, default=50.0, help="Uniform output frequency.")
    parser.add_argument("--joint_cutoff_hz", type=float, default=5.0, help="Non-leg joint low-pass cutoff.")
    parser.add_argument("--leg_joint_cutoff_hz", type=float, default=3.0, help="Leg joint low-pass cutoff.")
    parser.add_argument(
        "--ankle_joint_cutoff_hz",
        type=float,
        default=None,
        help="Ankle joint low-pass cutoff; defaults to --leg_joint_cutoff_hz.",
    )
    parser.add_argument(
        "--root_position_cutoff_hz", type=float, default=3.0, help="Root position low-pass cutoff."
    )
    parser.add_argument(
        "--root_orientation_cutoff_hz", type=float, default=3.0, help="Root orientation low-pass cutoff."
    )
    parser.add_argument(
        "--preserve_non_leg_joints",
        action="store_true",
        help="Keep non-leg joint positions unchanged instead of applying --joint_cutoff_hz.",
    )
    parser.add_argument(
        "--preserve_leg_joints",
        action="store_true",
        help=(
            "Keep hip, knee, and ankle joint positions unchanged instead of applying the grouped leg filters."
        ),
    )
    parser.add_argument(
        "--preserve_root_trajectory",
        action="store_true",
        help="Keep the root position/orientation unchanged instead of applying root low-pass filters.",
    )
    parser.add_argument(
        "--interpolate_outlier_frames",
        type=int,
        nargs="*",
        default=(),
        help="Zero-based source frames to replace by the midpoint of their immediate neighbors before filtering.",
    )
    parser.add_argument(
        "--trim_start_frames",
        type=int,
        default=0,
        help="Remove this many frames after smoothing/resampling and before FK/velocity reconstruction.",
    )
    parser.add_argument(
        "--clip_to_joint_limits",
        action="store_true",
        help="Smoothly project filtered joint positions into the URDF hard limits before rebuilding states.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_file if it exists.")
    return parser.parse_args()


def _as_names(values: np.ndarray) -> list[str]:
    items = np.asarray(values).reshape(-1).tolist()
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items]


def _derivative_rms(values: np.ndarray, fps: float, order: int) -> float:
    derivative = values.astype(np.float64)
    for _ in range(order):
        derivative = np.gradient(derivative, 1.0 / fps, axis=0)
    return float(np.sqrt(np.mean(np.square(derivative))))


def _clip_to_urdf_joint_limits(
    joint_pos: np.ndarray,
    joint_names: list[str],
    urdf_path: Path,
) -> tuple[np.ndarray, int, float]:
    joints = {joint.attrib["name"]: joint for joint in ElementTree.parse(urdf_path).getroot().findall("joint")}
    clipped = np.asarray(joint_pos, dtype=np.float64).copy()
    original = clipped.copy()
    for index, name in enumerate(joint_names):
        if name not in joints:
            raise ValueError(f"joint '{name}' is missing from {urdf_path}")
        limit = joints[name].find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            continue
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        width = JOINT_LIMIT_PROJECTION_WIDTH
        above_lower = lower + width * np.logaddexp(0.0, (clipped[:, index] - lower) / width)
        clipped[:, index] = upper - width * np.logaddexp(0.0, (upper - above_lower) / width)
    correction = clipped - original
    changed = int(np.count_nonzero(np.abs(correction) > 1.0e-12))
    maximum = float(np.max(np.abs(correction))) if changed else 0.0
    return clipped, changed, maximum


def _interpolate_outlier_frames(
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    frame_indices: tuple[int, ...] | list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = sorted(set(int(frame) for frame in frame_indices))
    if not frames:
        return joint_pos.copy(), root_pos.copy(), root_quat_wxyz.copy()
    invalid = [frame for frame in frames if frame <= 0 or frame >= joint_pos.shape[0] - 1]
    if invalid:
        raise ValueError(f"interpolate_outlier_frames must be interior frame indices, got {invalid}")
    adjacent = [(left, right) for left, right in zip(frames, frames[1:]) if right == left + 1]
    if adjacent:
        raise ValueError(f"adjacent interpolate_outlier_frames are ambiguous: {adjacent}")

    source_joint_pos = np.asarray(joint_pos, dtype=np.float64)
    source_root_pos = np.asarray(root_pos, dtype=np.float64)
    source_root_quat = np.asarray(root_quat_wxyz, dtype=np.float64)
    repaired_joint_pos = source_joint_pos.copy()
    repaired_root_pos = source_root_pos.copy()
    repaired_root_quat = source_root_quat.copy()
    for frame in frames:
        repaired_joint_pos[frame] = 0.5 * (source_joint_pos[frame - 1] + source_joint_pos[frame + 1])
        repaired_root_pos[frame] = 0.5 * (source_root_pos[frame - 1] + source_root_pos[frame + 1])
        neighbor_quats_xyzw = source_root_quat[[frame - 1, frame + 1]][:, [1, 2, 3, 0]]
        midpoint_xyzw = Slerp(
            [0.0, 1.0], Rotation.from_quat(neighbor_quats_xyzw)
        )([0.5]).as_quat()[0]
        repaired_root_quat[frame] = midpoint_xyzw[[3, 0, 1, 2]]
    return repaired_joint_pos, repaired_root_pos, repaired_root_quat


def smooth_resample_motion(
    motion: dict[str, np.ndarray],
    output_fps: float,
    joint_cutoff_hz: float,
    leg_joint_cutoff_hz: float,
    root_position_cutoff_hz: float,
    root_orientation_cutoff_hz: float,
    ankle_joint_cutoff_hz: float | None = None,
    preserve_non_leg_joints: bool = False,
    preserve_leg_joints: bool = False,
    preserve_root_trajectory: bool = False,
    interpolate_outlier_frames: tuple[int, ...] | list[int] = (),
) -> tuple[np.ndarray, dict[str, float | int]]:
    required = ("fps", "joint_pos", "joint_names", "body_pos_w", "body_quat_w", "body_names")
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"motion is missing required keys: {missing}")

    input_fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    if input_fps <= 0.0 or output_fps <= 0.0:
        raise ValueError(f"input/output fps must be positive, got {input_fps}/{output_fps}")
    effective_ankle_cutoff_hz = (
        leg_joint_cutoff_hz if ankle_joint_cutoff_hz is None else ankle_joint_cutoff_hz
    )
    for name, cutoff in (
        ("joint_cutoff_hz", joint_cutoff_hz),
        ("leg_joint_cutoff_hz", leg_joint_cutoff_hz),
        ("ankle_joint_cutoff_hz", effective_ankle_cutoff_hz),
        ("root_position_cutoff_hz", root_position_cutoff_hz),
        ("root_orientation_cutoff_hz", root_orientation_cutoff_hz),
    ):
        if not 0.0 < cutoff < 0.5 * input_fps:
            raise ValueError(f"{name} must be in (0, {0.5 * input_fps:g}), got {cutoff}")

    joint_names = _as_names(motion["joint_names"])
    source_joint_names = _as_names(motion.get("source_joint_names", motion["joint_names"]))
    missing_source_joints = [name for name in source_joint_names if name not in joint_names]
    if missing_source_joints:
        raise ValueError(f"source_joint_names are missing from joint_names: {missing_source_joints}")
    source_joint_indexes = [joint_names.index(name) for name in source_joint_names]
    joint_pos_original = np.asarray(motion["joint_pos"], dtype=np.float64)[:, source_joint_indexes]
    joint_pos = joint_pos_original.copy()
    if joint_pos.shape[0] <= 15:
        raise ValueError("motion must contain more than 15 frames for fourth-order zero-phase filtering")

    missing_leg_joints = [name for name in CR1_LEG_JOINT_NAMES if name not in source_joint_names]
    if missing_leg_joints:
        raise ValueError(f"CR1 leg joints are missing from source_joint_names: {missing_leg_joints}")
    leg_indexes = [source_joint_names.index(name) for name in CR1_LEG_JOINT_NAMES]
    missing_ankle_joints = [name for name in CR1_ANKLE_JOINT_NAMES if name not in source_joint_names]
    if missing_ankle_joints:
        raise ValueError(f"CR1 ankle joints are missing from source_joint_names: {missing_ankle_joints}")
    ankle_indexes = [source_joint_names.index(name) for name in CR1_ANKLE_JOINT_NAMES]

    body_names = _as_names(motion["body_names"])
    root_index = body_names.index("base_link") if "base_link" in body_names else 0
    root_pos_original = np.asarray(motion["body_pos_w"], dtype=np.float64)[:, root_index]
    root_quat_original = np.asarray(motion["body_quat_w"], dtype=np.float64)[:, root_index]
    root_quat_original /= np.linalg.norm(root_quat_original, axis=-1, keepdims=True)
    joint_pos, root_pos, root_quat_wxyz = _interpolate_outlier_frames(
        joint_pos,
        root_pos_original,
        root_quat_original,
        interpolate_outlier_frames,
    )
    root_quat_wxyz /= np.linalg.norm(root_quat_wxyz, axis=-1, keepdims=True)

    leg_sos = butter(4, leg_joint_cutoff_hz, fs=input_fps, output="sos")
    ankle_sos = butter(4, effective_ankle_cutoff_hz, fs=input_fps, output="sos")
    if preserve_non_leg_joints:
        joint_pos_smooth = joint_pos.copy()
    else:
        joint_sos = butter(4, joint_cutoff_hz, fs=input_fps, output="sos")
        joint_pos_smooth = sosfiltfilt(joint_sos, joint_pos, axis=0)
    if preserve_leg_joints:
        joint_pos_smooth[:, leg_indexes] = joint_pos[:, leg_indexes]
    else:
        joint_pos_smooth[:, leg_indexes] = sosfiltfilt(leg_sos, joint_pos[:, leg_indexes], axis=0)
        joint_pos_smooth[:, ankle_indexes] = sosfiltfilt(ankle_sos, joint_pos[:, ankle_indexes], axis=0)

    root_rotations = Rotation.from_quat(root_quat_wxyz[:, [1, 2, 3, 0]])
    if preserve_root_trajectory:
        root_pos_smooth = root_pos.copy()
        root_rotations_smooth = root_rotations
    else:
        root_position_sos = butter(4, root_position_cutoff_hz, fs=input_fps, output="sos")
        root_pos_smooth = sosfiltfilt(root_position_sos, root_pos, axis=0)
        first_root_rotation = root_rotations[0]
        relative_root_rotvec = (first_root_rotation.inv() * root_rotations).as_rotvec()
        root_orientation_sos = butter(4, root_orientation_cutoff_hz, fs=input_fps, output="sos")
        relative_root_rotvec_smooth = sosfiltfilt(root_orientation_sos, relative_root_rotvec, axis=0)
        root_rotations_smooth = first_root_rotation * Rotation.from_rotvec(relative_root_rotvec_smooth)

    input_times = np.arange(joint_pos.shape[0], dtype=np.float64) / input_fps
    duration = input_times[-1]
    output_frames = int(round(duration * output_fps)) + 1
    output_times = np.arange(output_frames, dtype=np.float64) / output_fps
    interpolation_times = np.minimum(output_times, duration)

    joint_pos_out = np.stack(
        [
            np.interp(interpolation_times, input_times, joint_pos_smooth[:, index])
            for index in range(joint_pos_smooth.shape[1])
        ],
        axis=1,
    )
    preserve_root_samples = (
        preserve_root_trajectory
        and output_frames == root_pos.shape[0]
        and np.array_equal(interpolation_times, input_times)
    )
    if preserve_root_samples:
        root_pos_out = root_pos.copy()
        root_quat_wxyz_out = root_quat_wxyz.copy()
    else:
        root_pos_out = np.stack(
            [np.interp(interpolation_times, input_times, root_pos_smooth[:, index]) for index in range(3)],
            axis=1,
        )
        root_quat_xyzw_out = Slerp(input_times, root_rotations_smooth)(interpolation_times).as_quat()
        root_quat_wxyz_out = root_quat_xyzw_out[:, [3, 0, 1, 2]]

    root_angular_velocity = quaternion_angular_velocity(root_quat_original, input_fps)
    root_angular_velocity_out = quaternion_angular_velocity(root_quat_wxyz_out, output_fps)
    original_root_rotations = Rotation.from_quat(root_quat_original[:, [1, 2, 3, 0]])
    root_orientation_delta = (original_root_rotations.inv() * root_rotations_smooth).magnitude()

    qpos = np.concatenate([root_pos_out, root_quat_wxyz_out, joint_pos_out], axis=1).astype(np.float32)
    diagnostics: dict[str, float | int] = {
        "input_frames": joint_pos.shape[0],
        "output_frames": output_frames,
        "input_fps": input_fps,
        "output_fps": output_fps,
        "input_duration_s": duration,
        "output_duration_s": (output_frames - 1) / output_fps,
        "joint_acceleration_rms_before": _derivative_rms(joint_pos_original, input_fps, 2),
        "joint_acceleration_rms_after": _derivative_rms(joint_pos_out, output_fps, 2),
        "leg_acceleration_rms_before": _derivative_rms(joint_pos_original[:, leg_indexes], input_fps, 2),
        "leg_acceleration_rms_after": _derivative_rms(joint_pos_out[:, leg_indexes], output_fps, 2),
        "ankle_acceleration_rms_before": _derivative_rms(
            joint_pos_original[:, ankle_indexes], input_fps, 2
        ),
        "ankle_acceleration_rms_after": _derivative_rms(
            joint_pos_out[:, ankle_indexes], output_fps, 2
        ),
        "ankle_joint_cutoff_hz": effective_ankle_cutoff_hz,
        "root_acceleration_rms_before": _derivative_rms(root_pos_original, input_fps, 2),
        "root_acceleration_rms_after": _derivative_rms(root_pos_out, output_fps, 2),
        "root_angular_acceleration_rms_before": _derivative_rms(root_angular_velocity, input_fps, 1),
        "root_angular_acceleration_rms_after": _derivative_rms(root_angular_velocity_out, output_fps, 1),
        "joint_position_delta_rms": float(
            np.sqrt(np.mean(np.square(joint_pos_smooth - joint_pos_original)))
        ),
        "joint_position_delta_max": float(np.max(np.abs(joint_pos_smooth - joint_pos_original))),
        "root_position_delta_rms": float(
            np.sqrt(np.mean(np.square(root_pos_smooth - root_pos_original)))
        ),
        "root_position_delta_max": float(np.max(np.abs(root_pos_smooth - root_pos_original))),
        "root_orientation_delta_rms": float(np.sqrt(np.mean(np.square(root_orientation_delta)))),
        "root_orientation_delta_max": float(np.max(root_orientation_delta)),
    }
    return qpos, diagnostics


def main() -> None:
    args = _parse_args()
    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input motion not found: {input_file}")
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_file}. Pass --overwrite to replace it.")

    with np.load(input_file, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}
    qpos, diagnostics = smooth_resample_motion(
        motion,
        args.output_fps,
        args.joint_cutoff_hz,
        args.leg_joint_cutoff_hz,
        args.root_position_cutoff_hz,
        args.root_orientation_cutoff_hz,
        args.ankle_joint_cutoff_hz,
        args.preserve_non_leg_joints,
        args.preserve_leg_joints,
        args.preserve_root_trajectory,
        args.interpolate_outlier_frames,
    )
    if args.trim_start_frames < 0 or args.trim_start_frames >= qpos.shape[0]:
        raise ValueError(
            f"trim_start_frames must be in [0, {qpos.shape[0] - 1}], got {args.trim_start_frames}"
        )
    pretrim_frames = qpos.shape[0]
    qpos = qpos[args.trim_start_frames :]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    robot_asset = args.robot_asset.expanduser().resolve()
    if not robot_asset.is_file():
        raise FileNotFoundError(f"Robot URDF not found: {robot_asset}")
    source_joint_names = _as_names(motion.get("source_joint_names", motion["joint_names"]))
    joint_limit_clipped_samples = 0
    joint_limit_correction_max = 0.0
    if args.clip_to_joint_limits:
        qpos[:, 7:], joint_limit_clipped_samples, joint_limit_correction_max = _clip_to_urdf_joint_limits(
            qpos[:, 7:], source_joint_names, robot_asset
        )
    body_names = _as_names(motion["body_names"])
    root_pos = qpos[:, :3]
    root_quat_wxyz = qpos[:, 3:7]
    joint_pos_source = qpos[:, 7:]
    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset, root_pos, root_quat_wxyz, joint_pos_source, source_joint_names, body_names
    )
    joint_names = _as_names(motion["joint_names"])
    source_indexes = [source_joint_names.index(name) for name in joint_names]
    joint_pos = joint_pos_source[:, source_indexes].astype(np.float32)
    output = {
        "fps": np.asarray([args.output_fps], dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": finite_difference(joint_pos, args.output_fps),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": finite_difference(body_pos_w, args.output_fps),
        "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, args.output_fps),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "source_joint_names": np.asarray(source_joint_names),
        "motion_processing_source_file": np.asarray([str(input_file)]),
        "motion_processing_source_fps": np.asarray([diagnostics["input_fps"]], dtype=np.float32),
        "motion_processing_joint_filter": np.asarray(["butterworth_zero_phase_order4_grouped"]),
        "motion_processing_joint_cutoff_hz": np.asarray([args.joint_cutoff_hz], dtype=np.float32),
        "motion_processing_leg_joint_cutoff_hz": np.asarray(
            [args.leg_joint_cutoff_hz], dtype=np.float32
        ),
        "motion_processing_ankle_joint_cutoff_hz": np.asarray(
            [diagnostics["ankle_joint_cutoff_hz"]], dtype=np.float32
        ),
        "motion_processing_root_position_cutoff_hz": np.asarray(
            [args.root_position_cutoff_hz], dtype=np.float32
        ),
        "motion_processing_root_orientation_cutoff_hz": np.asarray(
            [args.root_orientation_cutoff_hz], dtype=np.float32
        ),
        "motion_processing_preserve_non_leg_joints": np.asarray(
            [args.preserve_non_leg_joints]
        ),
        "motion_processing_preserve_leg_joints": np.asarray([args.preserve_leg_joints]),
        "motion_processing_preserve_root_trajectory": np.asarray(
            [args.preserve_root_trajectory]
        ),
        "motion_processing_interpolated_outlier_frames": np.asarray(
            args.interpolate_outlier_frames, dtype=np.int64
        ),
        "motion_processing_pretrim_frames": np.asarray([pretrim_frames], dtype=np.int64),
        "motion_processing_trim_start_frames": np.asarray([args.trim_start_frames], dtype=np.int64),
        "motion_processing_joint_limit_projection": np.asarray([args.clip_to_joint_limits]),
        "motion_processing_joint_limit_clipped_samples": np.asarray(
            [joint_limit_clipped_samples], dtype=np.int64
        ),
        "motion_processing_joint_limit_correction_max": np.asarray(
            [joint_limit_correction_max], dtype=np.float32
        ),
        "motion_processing_joint_limit_projection_width": np.asarray(
            [JOINT_LIMIT_PROJECTION_WIDTH], dtype=np.float32
        ),
    }
    for key, value in motion.items():
        if key.startswith(("world_alignment_", "motion_processing_ground_clearance_")):
            output[key] = np.asarray(value).copy()
    np.savez(output_file, **output)
    print(f"[INFO] Wrote resampled motion: {output_file}")
    print(
        f"[INFO] {diagnostics['input_frames']} frames @ {diagnostics['input_fps']:g} Hz "
        f"({diagnostics['input_duration_s']:.3f} s) -> "
        f"{diagnostics['output_frames']} frames @ {diagnostics['output_fps']:g} Hz "
        f"({diagnostics['output_duration_s']:.3f} s)"
    )
    if args.trim_start_frames:
        print(
            f"[INFO] Trimmed first {args.trim_start_frames} output frames; "
            f"retained {qpos.shape[0]} frames ({(qpos.shape[0] - 1) / args.output_fps:.3f} s)"
        )
    if args.clip_to_joint_limits:
        print(
            f"[INFO] Smooth URDF joint-limit projection changed {joint_limit_clipped_samples} samples; "
            f"maximum correction={joint_limit_correction_max:.6f} rad"
        )
    print(
        f"[INFO] Joint acceleration RMS: {diagnostics['joint_acceleration_rms_before']:.3f} -> "
        f"{diagnostics['joint_acceleration_rms_after']:.3f} rad/s^2"
    )
    print(
        f"[INFO] Leg acceleration RMS: {diagnostics['leg_acceleration_rms_before']:.3f} -> "
        f"{diagnostics['leg_acceleration_rms_after']:.3f} rad/s^2 "
        f"(leg cutoff={args.leg_joint_cutoff_hz:g} Hz, other cutoff={args.joint_cutoff_hz:g} Hz)"
    )
    print(
        f"[INFO] Ankle acceleration RMS: {diagnostics['ankle_acceleration_rms_before']:.3f} -> "
        f"{diagnostics['ankle_acceleration_rms_after']:.3f} rad/s^2 "
        f"(ankle cutoff={diagnostics['ankle_joint_cutoff_hz']:g} Hz)"
    )
    print(
        f"[INFO] Joint filter delta: rms={diagnostics['joint_position_delta_rms']:.5f} rad, "
        f"max={diagnostics['joint_position_delta_max']:.5f} rad"
    )
    print(
        f"[INFO] Root acceleration RMS: {diagnostics['root_acceleration_rms_before']:.3f} -> "
        f"{diagnostics['root_acceleration_rms_after']:.3f} m/s^2"
    )
    print(
        f"[INFO] Root angular acceleration RMS: "
        f"{diagnostics['root_angular_acceleration_rms_before']:.3f} -> "
        f"{diagnostics['root_angular_acceleration_rms_after']:.3f} rad/s^2"
    )
    print(
        f"[INFO] Root filter delta: position rms={diagnostics['root_position_delta_rms']:.5f} m, "
        f"max={diagnostics['root_position_delta_max']:.5f} m; orientation "
        f"rms={diagnostics['root_orientation_delta_rms']:.5f} rad, "
        f"max={diagnostics['root_orientation_delta_max']:.5f} rad"
    )


if __name__ == "__main__":
    main()
