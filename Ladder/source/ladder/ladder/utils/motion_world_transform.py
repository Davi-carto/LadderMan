"""Rigid world-frame transforms for Ladder motion files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


_POSITION_KEYS = ("body_pos_w",)
_ORIENTATION_KEYS = ("body_quat_w",)
_VECTOR_KEYS = ("body_lin_vel_w", "body_ang_vel_w")


def align_motion_ladder_frame(
    motion: Mapping[str, np.ndarray],
    *,
    source_ladder_position: Sequence[float],
    source_ladder_yaw_deg: float,
    target_ladder_position: Sequence[float],
    target_ladder_yaw_deg: float,
    robot_translation: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Map a motion from one ladder world pose to another.

    Joint-space fields are preserved. World positions, orientations, and
    linear/angular velocity vectors receive the same yaw-frame transform.
    ``robot_translation`` is applied after ladder-frame alignment and is
    intended for an explicitly approved grounding correction.
    """

    required = _POSITION_KEYS + _ORIENTATION_KEYS + _VECTOR_KEYS
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"motion is missing required world-frame fields: {missing}")

    source_position = _vector3(source_ladder_position, "source_ladder_position")
    target_position = _vector3(target_ladder_position, "target_ladder_position")
    robot_offset = _vector3(robot_translation, "robot_translation")
    delta_yaw_deg = float(target_ladder_yaw_deg) - float(source_ladder_yaw_deg)
    delta_rotation = Rotation.from_euler("z", delta_yaw_deg, degrees=True)

    output = {key: np.asarray(value).copy() for key, value in motion.items()}
    for key in _POSITION_KEYS:
        values = _vectors(output[key], key)
        output[key] = (
            delta_rotation.apply((values - source_position).reshape(-1, 3)).reshape(values.shape)
            + target_position
            + robot_offset
        ).astype(np.float32)
    for key in _VECTOR_KEYS:
        values = _vectors(output[key], key)
        output[key] = delta_rotation.apply(values.reshape(-1, 3)).reshape(values.shape).astype(np.float32)
    for key in _ORIENTATION_KEYS:
        quaternions = np.asarray(output[key], dtype=np.float64)
        if quaternions.ndim < 2 or quaternions.shape[-1] != 4:
            raise ValueError(f"{key} must end in quaternion width 4, got {quaternions.shape}")
        norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
        if np.any(norms <= 0.0) or not np.isfinite(norms).all():
            raise ValueError(f"{key} contains an invalid quaternion")
        rotations = Rotation.from_quat((quaternions / norms)[..., [1, 2, 3, 0]].reshape(-1, 4))
        transformed = (delta_rotation * rotations).as_quat().reshape(quaternions.shape)
        output[key] = transformed[..., [3, 0, 1, 2]].astype(np.float32)

    output.update(
        {
            "world_alignment_source_ladder_position": source_position.astype(np.float32),
            "world_alignment_source_ladder_yaw_deg": np.asarray(
                [source_ladder_yaw_deg], dtype=np.float32
            ),
            "world_alignment_target_ladder_position": target_position.astype(np.float32),
            "world_alignment_target_ladder_yaw_deg": np.asarray(
                [target_ladder_yaw_deg], dtype=np.float32
            ),
            "world_alignment_delta_yaw_deg": np.asarray([delta_yaw_deg], dtype=np.float32),
            "world_alignment_robot_translation": robot_offset.astype(np.float32),
        }
    )
    return output


def _vector3(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values, got {vector}")
    return vector


def _vectors(values: np.ndarray, name: str) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim < 2 or vectors.shape[-1] != 3 or not np.isfinite(vectors).all():
        raise ValueError(f"{name} must end in three finite coordinates, got {vectors.shape}")
    return vectors
