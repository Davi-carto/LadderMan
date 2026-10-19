"""Primitive-box description and geometry warping for the CR1 wood-step ladder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


WOOD_STEP_FIXED_BOX_NAMES = ("base_platform",)
WOOD_STEP_RUNG_BOX_NAMES = tuple(f"step_{index:02d}" for index in range(1, 8))
WOOD_STEP_INCLINED_BOX_NAMES = (
    "front_zigzag_side_board_left",
    "front_zigzag_side_board_right",
    "rear_support_leg_left",
    "rear_support_leg_right",
)


@dataclass(frozen=True)
class WoodStepBoxSpec:
    name: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


def _numbers(value: str, count: int) -> np.ndarray:
    result = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    if result.shape != (count,):
        raise ValueError(f"expected {count} values, got {value!r}")
    return result


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    pure_vector = np.concatenate((np.zeros(1, dtype=np.float64), vector))
    conjugate = quaternion * np.asarray((1.0, -1.0, -1.0, -1.0))
    return _quat_mul(_quat_mul(quaternion, pure_vector), conjugate)[1:]


def thin_front_side_board_specs(
    specs: tuple[WoodStepBoxSpec, ...],
    *,
    remaining_depth_fraction: float,
) -> tuple[WoodStepBoxSpec, ...]:
    """Thin both front side boards while preserving their positive local-y faces."""

    if not 0.0 < remaining_depth_fraction <= 1.0:
        raise ValueError(
            "remaining_depth_fraction must be in (0, 1], got "
            f"{remaining_depth_fraction}"
        )

    side_board_names = {
        "front_zigzag_side_board_left",
        "front_zigzag_side_board_right",
    }
    found_names: set[str] = set()
    thinned_specs: list[WoodStepBoxSpec] = []
    for spec in specs:
        if spec.name not in side_board_names:
            thinned_specs.append(spec)
            continue

        found_names.add(spec.name)
        old_depth = spec.size[1]
        new_depth = old_depth * remaining_depth_fraction
        center_shift = 0.5 * (old_depth - new_depth)
        positive_depth_axis = _quat_rotate(
            np.asarray(spec.quaternion), np.asarray((0.0, 1.0, 0.0))
        )
        new_position = np.asarray(spec.position) + center_shift * positive_depth_axis
        thinned_specs.append(
            WoodStepBoxSpec(
                name=spec.name,
                size=(spec.size[0], new_depth, spec.size[2]),
                position=tuple(new_position.tolist()),
                quaternion=spec.quaternion,
            )
        )

    missing_names = sorted(side_board_names - found_names)
    if missing_names:
        raise ValueError(f"missing front side-board specs: {missing_names}")
    return tuple(thinned_specs)


def _rpy_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = 0.5 * rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=np.float64,
    )


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.asarray((math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)))


def _rotate_yaw(vector: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    result = vector.copy()
    result[..., 0] = cosine * vector[..., 0] - sine * vector[..., 1]
    result[..., 1] = sine * vector[..., 0] + cosine * vector[..., 1]
    return result


def load_wood_step_box_specs(
    urdf_path: str | Path,
    *,
    asset_translation: tuple[float, float, float],
    asset_yaw_deg: float,
) -> tuple[WoodStepBoxSpec, ...]:
    """Load the 18 collision boxes and express them in the environment-local world frame."""

    path = Path(urdf_path)
    root = ElementTree.parse(path).getroot()
    yaw = math.radians(asset_yaw_deg)
    root_quat = _yaw_quat_wxyz(yaw)
    translation = np.asarray(asset_translation, dtype=np.float64)
    specs: list[WoodStepBoxSpec] = []
    for collision in root.findall(".//collision"):
        box = collision.find("./geometry/box")
        if box is None:
            raise ValueError(f"wood-step ladder collision must be a box: {collision.attrib.get('name')}")
        origin = collision.find("origin")
        local_position = _numbers(origin.attrib.get("xyz", "0 0 0"), 3)
        local_rpy = _numbers(origin.attrib.get("rpy", "0 0 0"), 3)
        position = translation + _rotate_yaw(local_position, yaw)
        quaternion = _quat_mul(root_quat, _rpy_quat_wxyz(local_rpy))
        collision_name = collision.attrib.get("name", "")
        name = collision_name.removesuffix("_collision")
        specs.append(
            WoodStepBoxSpec(
                name=name,
                size=tuple(_numbers(box.attrib["size"], 3).tolist()),
                position=tuple(position.tolist()),
                quaternion=tuple((quaternion / np.linalg.norm(quaternion)).tolist()),
            )
        )
    if len(specs) != 18 or len({spec.name for spec in specs}) != len(specs):
        raise ValueError(f"expected 18 uniquely named wood-step boxes, got {len(specs)}")
    return tuple(specs)


def ladder_delta_quat_wxyz(source_phi: float, target_phi: float) -> np.ndarray:
    """Quaternion rotating the source climbing direction onto the target direction."""

    half_angle = 0.5 * (source_phi - target_phi)
    return np.asarray((math.cos(half_angle), 0.0, math.sin(half_angle), 0.0))


def warp_ladder_points(
    points: np.ndarray,
    *,
    source_phi: float,
    source_spacing: float,
    target_phi: float,
    target_spacing: float,
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Map points through the ladder axial/normal frame while keeping the first rung fixed."""

    points = np.asarray(points, dtype=np.float64)
    origin_array = np.asarray(origin, dtype=np.float64)
    relative = points - origin_array
    source_cos, source_sin = math.cos(source_phi), math.sin(source_phi)
    target_cos, target_sin = math.cos(target_phi), math.sin(target_phi)
    axial = relative[..., 0] * source_cos + relative[..., 2] * source_sin
    normal = -relative[..., 0] * source_sin + relative[..., 2] * source_cos
    axial = axial * (target_spacing / source_spacing)
    result = points.copy()
    result[..., 0] = origin_array[0] + axial * target_cos - normal * target_sin
    result[..., 1] = points[..., 1]
    result[..., 2] = origin_array[2] + axial * target_sin + normal * target_cos
    return result


def randomized_wood_step_box_poses(
    specs: tuple[WoodStepBoxSpec, ...],
    *,
    source_phi: float,
    source_spacing: float,
    target_phi: float,
    target_spacing: float,
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return environment-local poses for every physical/visual ladder box."""

    positions = np.asarray([spec.position for spec in specs], dtype=np.float64)
    quaternions = np.asarray([spec.quaternion for spec in specs], dtype=np.float64)
    warped_positions = warp_ladder_points(
        positions,
        source_phi=source_phi,
        source_spacing=source_spacing,
        target_phi=target_phi,
        target_spacing=target_spacing,
        origin=origin,
    )
    fixed_mask = np.asarray([spec.name in WOOD_STEP_FIXED_BOX_NAMES for spec in specs])
    warped_positions[fixed_mask] = positions[fixed_mask]
    spec_indexes = {spec.name: index for index, spec in enumerate(specs)}
    source_direction = np.asarray((math.cos(source_phi), 0.0, math.sin(source_phi)))
    target_direction = np.asarray((math.cos(target_phi), 0.0, math.sin(target_phi)))
    origin_array = np.asarray(origin, dtype=np.float64)
    for rung_index, name in enumerate(WOOD_STEP_RUNG_BOX_NAMES):
        box_index = spec_indexes[name]
        source_rung_top = origin_array + rung_index * source_spacing * source_direction
        center_offset = positions[box_index] - source_rung_top
        target_rung_top = origin_array + rung_index * target_spacing * target_direction
        warped_positions[box_index] = target_rung_top + center_offset
    inclined_mask = np.asarray([spec.name in WOOD_STEP_INCLINED_BOX_NAMES for spec in specs])
    delta_quat = ladder_delta_quat_wxyz(source_phi, target_phi)
    quaternions[inclined_mask] = _quat_mul(delta_quat, quaternions[inclined_mask])
    return warped_positions, quaternions
