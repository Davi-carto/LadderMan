"""Export Holosoma CR1 qpos to Ladder motion fields with offline URDF kinematics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ladder.utils.urdf_kinematics import finite_difference, forward_kinematics, quaternion_angular_velocity


DEFAULT_CR1_URDF = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline export of Holosoma CR1 qpos to Ladder motion npz.")
    parser.add_argument("--input_file", type=Path, required=True, help="Holosoma qpos npz.")
    parser.add_argument("--output_file", type=Path, required=True, help="Output Ladder motion npz.")
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_CR1_URDF, help="CR1 training URDF.")
    parser.add_argument("--fps_override", type=float, default=None, help="Override the source fps metadata.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def _urdf_names(urdf_path: Path) -> tuple[list[str], list[str]]:
    root = ElementTree.parse(urdf_path).getroot()
    joint_names = [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") in ("revolute", "continuous")
    ]
    body_names = [link.attrib["name"] for link in root.findall("link")]
    return joint_names, body_names


def main() -> None:
    args = _parse_args()
    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    robot_asset = args.robot_asset.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input motion not found: {input_file}")
    if not robot_asset.is_file():
        raise FileNotFoundError(f"Robot URDF not found: {robot_asset}")
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_file}. Pass --overwrite to replace it.")

    with np.load(input_file, allow_pickle=False) as data:
        if "qpos" not in data.files:
            raise KeyError(f"{input_file} is missing required key 'qpos'")
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps = float(
            args.fps_override
            if args.fps_override is not None
            else np.asarray(data["fps"]).reshape(-1)[0]
        )

    joint_names, body_names = _urdf_names(robot_asset)
    expected_width = 7 + len(joint_names)
    if qpos.ndim != 2 or qpos.shape[1] != expected_width:
        raise ValueError(f"expected qpos shape (T, {expected_width}), got {qpos.shape}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos contains NaN or Inf")

    root_pos = qpos[:, :3]
    root_quat = qpos[:, 3:7]
    root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
    joint_pos = qpos[:, 7:]
    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset,
        root_pos,
        root_quat,
        joint_pos,
        joint_names,
        body_names,
    )
    motion = {
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": finite_difference(joint_pos, fps),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": finite_difference(body_pos_w, fps),
        "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "source_joint_names": np.asarray(joint_names),
        "motion_export_source_file": np.asarray([str(input_file)]),
        "motion_export_method": np.asarray(["offline_urdf_forward_kinematics"]),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **motion)
    print(f"[INFO] Wrote motion: {output_file}")
    print(f"[INFO] Frames: {qpos.shape[0]}, fps: {fps:g}, duration: {(qpos.shape[0] - 1) / fps:.3f}s")
    print(f"[INFO] joint_pos: {motion['joint_pos'].shape}, body_pos_w: {motion['body_pos_w'].shape}")


if __name__ == "__main__":
    main()
