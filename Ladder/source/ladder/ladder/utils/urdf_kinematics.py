"""Small batched URDF forward-kinematics helper for offline reference processing."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(text: str | None) -> np.ndarray:
    return np.fromstring(text, sep=" ") if text else np.zeros(3, dtype=np.float64)


def _joint_origin(joint: ElementTree.Element) -> tuple[np.ndarray, Rotation]:
    origin = joint.find("origin")
    if origin is None:
        return np.zeros(3, dtype=np.float64), Rotation.identity()
    return (
        _vector(origin.attrib.get("xyz")),
        Rotation.from_euler("xyz", _vector(origin.attrib.get("rpy"))),
    )


def forward_kinematics(
    urdf_path: str | Path,
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    joint_names: list[str],
    body_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute body poses in ``body_names`` order for a standard URDF tree."""

    root = ElementTree.parse(urdf_path).getroot()
    child_joints: dict[str, list[ElementTree.Element]] = {}
    child_links = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        child_joints.setdefault(parent, []).append(joint)
        child_links.add(child)
    link_names = [link.attrib["name"] for link in root.findall("link")]
    root_links = [name for name in link_names if name not in child_links]
    if len(root_links) != 1:
        raise ValueError(f"expected one URDF root link, got {root_links}")
    if root_links[0] not in body_names:
        raise ValueError(f"URDF root body '{root_links[0]}' is missing from body_names")

    root_pos_w = np.asarray(root_pos_w, dtype=np.float64)
    root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64)
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    if root_pos_w.shape != (joint_pos.shape[0], 3) or root_quat_wxyz.shape != (joint_pos.shape[0], 4):
        raise ValueError("root poses and joint positions must have the same frame count")
    if joint_pos.shape[1] != len(joint_names):
        raise ValueError(f"joint_pos has {joint_pos.shape[1]} columns for {len(joint_names)} joint names")

    frame_count = joint_pos.shape[0]
    positions = {root_links[0]: root_pos_w}
    rotations = {root_links[0]: Rotation.from_quat(root_quat_wxyz[:, [1, 2, 3, 0]])}
    stack = [root_links[0]]
    while stack:
        parent = stack.pop()
        for joint in child_joints.get(parent, []):
            child = joint.find("child").attrib["link"]
            origin_pos, origin_rot = _joint_origin(joint)
            parent_rot = rotations[parent]
            positions[child] = positions[parent] + parent_rot.apply(
                np.broadcast_to(origin_pos, (frame_count, 3)).copy()
            )
            child_rot = parent_rot * origin_rot
            if joint.attrib["type"] in ("revolute", "continuous"):
                name = joint.attrib["name"]
                if name not in joint_names:
                    raise ValueError(f"joint '{name}' is missing from joint_names")
                axis_element = joint.find("axis")
                axis = _vector(axis_element.attrib.get("xyz")) if axis_element is not None else np.asarray([1, 0, 0])
                angle = joint_pos[:, joint_names.index(name)]
                child_rot = child_rot * Rotation.from_rotvec(angle[:, None] * axis[None, :])
            rotations[child] = child_rot
            stack.append(child)

    missing_bodies = [name for name in body_names if name not in positions]
    if missing_bodies:
        raise ValueError(f"URDF forward kinematics did not resolve bodies: {missing_bodies}")
    body_pos_w = np.stack([positions[name] for name in body_names], axis=1).astype(np.float32)
    body_quat_w = np.stack(
        [rotations[name].as_quat()[:, [3, 0, 1, 2]] for name in body_names], axis=1
    ).astype(np.float32)
    return body_pos_w, body_quat_w


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(np.asarray(values, dtype=np.float64), 1.0 / fps, axis=0).astype(np.float32)


def quaternion_angular_velocity(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    """Compute world-frame angular velocity with centered quaternion differences."""

    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    flat = quat_wxyz.reshape(quat_wxyz.shape[0], -1, 4)
    rotations = Rotation.from_quat(flat[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()
    rotations = rotations.reshape(flat.shape[0], flat.shape[1], 3, 3)
    angular_velocity = np.zeros(flat.shape[:2] + (3,), dtype=np.float64)
    if flat.shape[0] > 1:
        for frame in range(flat.shape[0]):
            previous = max(frame - 1, 0)
            following = min(frame + 1, flat.shape[0] - 1)
            dt = (following - previous) / fps
            relative = np.swapaxes(rotations[previous], -1, -2) @ rotations[following]
            local_rotvec = Rotation.from_matrix(relative).as_rotvec() / dt
            angular_velocity[frame] = np.einsum("bij,bj->bi", rotations[frame], local_rotvec)
    return angular_velocity.reshape(quat_wxyz.shape[:-1] + (3,)).astype(np.float32)
