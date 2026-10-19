"""Add a grounded CR1 default-standing hold and smooth transition to a motion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ladder.utils.urdf_kinematics import (
    finite_difference,
    forward_kinematics,
    quaternion_angular_velocity,
)


DEFAULT_CR1_URDF = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)
SOLE_BODY_NAMES = ("left_sole_contact_point", "right_sole_contact_point")
FOOT_COLLISION_BODY_NAMES = ("left_ankle_x_link", "right_ankle_x_link")
GROUND_CLEARANCE_MAX_STEP = 0.001
FRAME_FIELDS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepend a zero-joint grounded stand and quintic transition to a CR1 motion."
    )
    parser.add_argument("--input_file", type=Path, required=True, help="Unannotated CR1 motion npz.")
    parser.add_argument("--output_file", type=Path, required=True, help="Output motion npz.")
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_CR1_URDF, help="CR1 URDF for FK.")
    parser.add_argument("--hold_frames", type=int, default=25, help="Default-pose hold length.")
    parser.add_argument("--transition_frames", type=int, default=50, help="Quintic transition length.")
    parser.add_argument("--ground_z", type=float, default=-0.03, help="World ground-plane Z.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_file if it exists.")
    return parser.parse_args()


def _as_names(values: np.ndarray) -> list[str]:
    items = np.asarray(values).reshape(-1).tolist()
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items]


def _yaw_only_quaternion(quat_wxyz: np.ndarray) -> np.ndarray:
    source = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
    yaw = source.as_euler("xyz")[2]
    quat_xyzw = Rotation.from_euler("z", yaw).as_quat()
    return quat_xyzw[[3, 0, 1, 2]]


def _quintic_weights(frame_count: int) -> np.ndarray:
    if frame_count < 2:
        raise ValueError("transition_frames must be at least 2")
    t = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def _slerp_pairs(start_wxyz: np.ndarray, targets_wxyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.empty_like(targets_wxyz, dtype=np.float64)
    for index, (target, weight) in enumerate(zip(targets_wxyz, weights, strict=True)):
        target = np.asarray(target, dtype=np.float64)
        if np.dot(start_wxyz, target) < 0.0:
            target = -target
        rotations = Rotation.from_quat(
            np.stack((start_wxyz, target))[:, [1, 2, 3, 0]]
        )
        quat_xyzw = Slerp([0.0, 1.0], rotations)([weight]).as_quat()[0]
        output[index] = quat_xyzw[[3, 0, 1, 2]]
    return output


def _foot_collision_vertices(robot_asset: Path) -> dict[str, np.ndarray]:
    try:
        import trimesh
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("default-start ground clearance requires trimesh") from exc

    root = ElementTree.parse(robot_asset).getroot()
    vertices: dict[str, np.ndarray] = {}
    for body_name in FOOT_COLLISION_BODY_NAMES:
        link = root.find(f"./link[@name='{body_name}']")
        if link is None:
            raise ValueError(f"robot URDF is missing link '{body_name}'")
        mesh = link.find("./collision/geometry/mesh")
        if mesh is None or "filename" not in mesh.attrib:
            raise ValueError(f"link '{body_name}' does not have a collision mesh")
        mesh_path = (robot_asset.parent / mesh.attrib["filename"]).resolve()
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        vertices[body_name] = np.asarray(loaded.vertices, dtype=np.float64)
    return vertices


def _minimum_foot_collision_z(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_names: list[str],
    collision_vertices: dict[str, np.ndarray],
) -> np.ndarray:
    minimum_z = np.full(body_pos_w.shape[0], np.inf, dtype=np.float64)
    for body_name, vertices in collision_vertices.items():
        body_index = body_names.index(body_name)
        quat_wxyz = np.asarray(body_quat_w[:, body_index], dtype=np.float64)
        rotations = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
        local_world_z_axis = rotations[:, 2, :]
        body_minimum_z = np.empty(body_pos_w.shape[0], dtype=np.float64)
        for frame, axis in enumerate(local_world_z_axis):
            body_minimum_z[frame] = (
                body_pos_w[frame, body_index, 2] + np.min(vertices @ axis)
            )
        minimum_z = np.minimum(minimum_z, body_minimum_z)
    return minimum_z


def _slope_limited_upper_envelope(
    required: np.ndarray, hold_frames: int, max_step: float
) -> np.ndarray:
    """Return a conservative clearance envelope with bounded per-frame change."""

    envelope = np.asarray(required, dtype=np.float64).copy()
    for frame in range(1, len(envelope)):
        envelope[frame] = max(envelope[frame], envelope[frame - 1] - max_step)
    for frame in range(len(envelope) - 2, -1, -1):
        envelope[frame] = max(envelope[frame], envelope[frame + 1] - max_step)
    static_end = hold_frames + 1  # include transition frame 0, whose quintic weight is zero
    envelope[:static_end] = np.max(envelope[:static_end])
    for frame in range(static_end, len(envelope)):
        envelope[frame] = max(envelope[frame], envelope[frame - 1] - max_step)
    return envelope


def add_default_standing_start(
    motion: dict[str, np.ndarray],
    *,
    robot_asset: Path,
    hold_frames: int,
    transition_frames: int,
    ground_z: float,
    source_file: Path,
) -> dict[str, np.ndarray]:
    required = ("fps", "joint_pos", "joint_names", "body_pos_w", "body_quat_w", "body_names")
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"motion is missing required keys: {missing}")
    if hold_frames <= 0:
        raise ValueError(f"hold_frames must be positive, got {hold_frames}")

    joint_pos_source = np.asarray(motion["joint_pos"], dtype=np.float64)
    frame_count, joint_count = joint_pos_source.shape
    if transition_frames >= frame_count:
        raise ValueError(
            f"transition_frames must be smaller than the {frame_count}-frame motion"
        )
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    joint_names = _as_names(motion["joint_names"])
    body_names = _as_names(motion["body_names"])
    root_index = body_names.index("base_link")
    sole_indexes = [body_names.index(name) for name in SOLE_BODY_NAMES]
    root_pos_source = np.asarray(motion["body_pos_w"][:, root_index], dtype=np.float64)
    root_quat_source = np.asarray(motion["body_quat_w"][:, root_index], dtype=np.float64)

    standing_joint_pos = np.zeros(joint_count, dtype=np.float64)
    standing_root_quat = _yaw_only_quaternion(root_quat_source[0])
    zero_root_pos = np.zeros((1, 3), dtype=np.float64)
    standing_body_pos, _ = forward_kinematics(
        robot_asset,
        zero_root_pos,
        standing_root_quat[None],
        standing_joint_pos[None],
        joint_names,
        body_names,
    )
    standing_sole_offsets = standing_body_pos[0, sole_indexes]
    source_sole_center_xy = np.mean(
        np.asarray(motion["body_pos_w"][0, sole_indexes, :2], dtype=np.float64), axis=0
    )
    standing_root_xy = source_sole_center_xy - np.mean(standing_sole_offsets[:, :2], axis=0)
    standing_root_z = ground_z - float(np.min(standing_sole_offsets[:, 2]))
    standing_root_pos = np.asarray(
        [standing_root_xy[0], standing_root_xy[1], standing_root_z], dtype=np.float64
    )

    weights = _quintic_weights(transition_frames)
    transition_joint_pos = (
        (1.0 - weights[:, None]) * standing_joint_pos[None]
        + weights[:, None] * joint_pos_source[:transition_frames]
    )
    transition_root_pos = (
        (1.0 - weights[:, None]) * standing_root_pos[None]
        + weights[:, None] * root_pos_source[:transition_frames]
    )
    transition_root_quat = _slerp_pairs(
        standing_root_quat, root_quat_source[:transition_frames], weights
    )

    joint_pos = np.concatenate(
        (
            np.repeat(standing_joint_pos[None], hold_frames, axis=0),
            transition_joint_pos,
            joint_pos_source[transition_frames:],
        ),
        axis=0,
    )
    root_pos = np.concatenate(
        (
            np.repeat(standing_root_pos[None], hold_frames, axis=0),
            transition_root_pos,
            root_pos_source[transition_frames:],
        ),
        axis=0,
    )
    root_quat = np.concatenate(
        (
            np.repeat(standing_root_quat[None], hold_frames, axis=0),
            transition_root_quat,
            root_quat_source[transition_frames:],
        ),
        axis=0,
    )

    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset, root_pos, root_quat, joint_pos, joint_names, body_names
    )
    collision_vertices = _foot_collision_vertices(robot_asset)
    minimum_foot_collision_z = _minimum_foot_collision_z(
        body_pos_w, body_quat_w, body_names, collision_vertices
    )
    required_clearance = np.maximum(
        ground_z
        - np.minimum(
            minimum_foot_collision_z,
            np.min(body_pos_w[:, sole_indexes, 2], axis=1),
        ),
        0.0,
    )
    clearance_envelope = _slope_limited_upper_envelope(
        required_clearance, hold_frames, GROUND_CLEARANCE_MAX_STEP
    )
    root_pos[:, 2] += clearance_envelope
    body_pos_w[:, :, 2] += clearance_envelope[:, None]

    output = {}
    for key, value in motion.items():
        if key in FRAME_FIELDS:
            continue
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == frame_count:
            # The transition replaces source frames in place, while the static
            # hold is the only newly inserted time span. Keep every derived
            # frame field aligned; collision/contact fields are expected to be
            # recomputed before promoting the result to train-ready.
            array = np.concatenate(
                (np.repeat(array[:1], hold_frames, axis=0), array), axis=0
            )
        output[key] = array.copy()
    output.update(
        {
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": finite_difference(joint_pos, fps),
            "body_pos_w": body_pos_w.astype(np.float32),
            "body_quat_w": body_quat_w.astype(np.float32),
            "body_lin_vel_w": finite_difference(body_pos_w, fps),
            "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
            "motion_processing_default_standing_source_file": np.asarray([str(source_file)]),
            "motion_processing_default_standing_hold_frames": np.asarray(
                [hold_frames], dtype=np.int64
            ),
            "motion_processing_default_standing_transition_frames": np.asarray(
                [transition_frames], dtype=np.int64
            ),
            "motion_processing_default_standing_ground_z": np.asarray(
                [ground_z], dtype=np.float32
            ),
            "motion_processing_default_standing_root_pos": root_pos[:1].astype(np.float32),
            "motion_processing_default_standing_root_quat_wxyz": standing_root_quat[
                None
            ].astype(np.float32),
            "motion_processing_default_standing_frame_field_policy": np.asarray(
                ["prepend_source_frame0_then_shift"]
            ),
            "motion_processing_ground_clearance_required_z": required_clearance.astype(
                np.float32
            ),
            "motion_processing_ground_clearance_z": clearance_envelope.astype(np.float32),
            "motion_processing_ground_clearance_max": np.asarray(
                [np.max(clearance_envelope)], dtype=np.float32
            ),
            "motion_processing_ground_clearance_max_step": np.asarray(
                [GROUND_CLEARANCE_MAX_STEP], dtype=np.float32
            ),
            "motion_processing_ground_clearance_geometry": np.asarray(
                ["ankle_x_collision_mesh_and_virtual_sole"]
            ),
        }
    )
    return output


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
        motion = {key: data[key] for key in data.files}
    output = add_default_standing_start(
        motion,
        robot_asset=robot_asset,
        hold_frames=args.hold_frames,
        transition_frames=args.transition_frames,
        ground_z=args.ground_z,
        source_file=input_file,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **output)

    frame_count = int(output["joint_pos"].shape[0])
    fps = float(np.asarray(output["fps"]).reshape(-1)[0])
    print(f"[INFO] Wrote default-standing-start motion: {output_file}")
    print(
        f"[INFO] Frames={frame_count}, fps={fps:g}, hold={args.hold_frames} "
        f"({args.hold_frames / fps:.3f} s), transition={args.transition_frames} "
        f"({args.transition_frames / fps:.3f} s)"
    )
    print(
        f"[INFO] Standing root pos={output['motion_processing_default_standing_root_pos'][0].tolist()}, "
        f"max ground-clearance correction="
        f"{float(output['motion_processing_ground_clearance_max'][0]):.6f} m"
    )
    print(
        f"[INFO] Hold maximum joint/root linear speed="
        f"{np.max(np.abs(output['joint_vel'][:args.hold_frames])):.6f} rad/s / "
        f"{np.max(np.abs(output['body_lin_vel_w'][:args.hold_frames, 0])):.6f} m/s"
    )


if __name__ == "__main__":
    main()
