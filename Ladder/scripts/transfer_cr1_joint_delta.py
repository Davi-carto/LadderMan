"""Transfer a recorded CR1 joint-processing delta onto another motion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ladder.utils.urdf_kinematics import (
    finite_difference,
    forward_kinematics,
    quaternion_angular_velocity,
)


DEFAULT_ROBOT = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)
LEG_JOINT_NAMES = {
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
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_file", type=Path, required=True)
    parser.add_argument("--delta_before_file", type=Path, required=True)
    parser.add_argument("--delta_after_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--joint_group", choices=("legs", "non_legs", "all"), required=True)
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _names(values: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(values).reshape(-1).tolist()]


def _load(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    args = _parse_args()
    paths = [
        args.base_file.expanduser().resolve(),
        args.delta_before_file.expanduser().resolve(),
        args.delta_after_file.expanduser().resolve(),
    ]
    output_file = args.output_file.expanduser().resolve()
    robot_asset = args.robot_asset.expanduser().resolve()
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_file}")
    if not robot_asset.is_file():
        raise FileNotFoundError(robot_asset)

    base, before, after = (_load(path) for path in paths)
    joint_names = _names(base["joint_names"])
    body_names = _names(base["body_names"])
    source_joint_names = _names(base["source_joint_names"])
    frame_count = base["joint_pos"].shape[0]
    fps = float(np.asarray(base["fps"]).reshape(-1)[0])
    for label, motion in (("before", before), ("after", after)):
        if _names(motion["joint_names"]) != joint_names:
            raise ValueError(f"{label} joint names do not match the base motion")
        if motion["joint_pos"].shape != base["joint_pos"].shape:
            raise ValueError(f"{label} joint positions do not match the base motion shape")
        if not np.isclose(float(np.asarray(motion["fps"]).reshape(-1)[0]), fps):
            raise ValueError(f"{label} fps does not match the base motion")

    if args.joint_group == "legs":
        selected = [index for index, name in enumerate(joint_names) if name in LEG_JOINT_NAMES]
    elif args.joint_group == "non_legs":
        selected = [index for index, name in enumerate(joint_names) if name not in LEG_JOINT_NAMES]
    else:
        selected = list(range(len(joint_names)))
    if not selected:
        raise ValueError(f"joint group '{args.joint_group}' is empty")

    joint_pos = np.asarray(base["joint_pos"], dtype=np.float64).copy()
    delta = (
        np.asarray(after["joint_pos"], dtype=np.float64)
        - np.asarray(before["joint_pos"], dtype=np.float64)
    )
    joint_pos[:, selected] += delta[:, selected]
    root_index = body_names.index("base_link")
    root_pos = np.asarray(base["body_pos_w"][:, root_index], dtype=np.float64)
    root_quat = np.asarray(base["body_quat_w"][:, root_index], dtype=np.float64)
    source_indexes = [joint_names.index(name) for name in source_joint_names]
    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset,
        root_pos,
        root_quat,
        joint_pos[:, source_indexes],
        source_joint_names,
        body_names,
    )

    output = {key: np.asarray(value).copy() for key, value in base.items()}
    output.update(
        {
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": finite_difference(joint_pos, fps),
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": finite_difference(body_pos_w, fps),
            "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
            "joint_delta_transfer_base_file": np.asarray([str(paths[0])]),
            "joint_delta_transfer_before_file": np.asarray([str(paths[1])]),
            "joint_delta_transfer_after_file": np.asarray([str(paths[2])]),
            "joint_delta_transfer_group": np.asarray([args.joint_group]),
            "joint_delta_transfer_rms": np.asarray(
                [np.sqrt(np.mean(np.square(delta[:, selected])))], dtype=np.float32
            ),
            "joint_delta_transfer_max": np.asarray(
                [np.max(np.abs(delta[:, selected]))], dtype=np.float32
            ),
        }
    )
    if output["joint_pos"].shape != (frame_count, len(joint_names)):
        raise ValueError("unexpected output joint shape")
    for key, value in output.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite output: {key}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **output)
    print(f"[INFO] Wrote joint-delta transfer: {output_file}")
    print(
        f"[INFO] Applied {args.joint_group} delta: "
        f"rms={np.sqrt(np.mean(np.square(delta[:, selected]))):.6f} rad, "
        f"max={np.max(np.abs(delta[:, selected])):.6f} rad"
    )


if __name__ == "__main__":
    main()
