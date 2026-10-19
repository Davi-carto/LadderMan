"""Warp depth and hit-class rendering for the CR1 wood-step ladder scene."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
import trimesh
import warp as wp


BACKGROUND = 0
RUNG = 1
ROBOT = 2
LADDER_OTHER = 3
GROUND = 4

HIT_CLASS_NAMES = {
    BACKGROUND: "BACKGROUND",
    RUNG: "RUNG",
    ROBOT: "ROBOT",
    LADDER_OTHER: "LADDER_OTHER",
    GROUND: "GROUND",
}

WARP_DIAGNOSTIC_COUNT_NAMES = (
    "rays_total",
    "unit_direction_rays",
    "rays_hitting_any_ladder_box",
    "rays_hitting_rung_box",
    "rays_hitting_other_ladder_box",
    "rays_hitting_robot_mesh",
    "rays_hitting_robot_sphere",
    "rays_hitting_ground_plane",
    "final_hits_in_depth_range",
    "final_background_pixels",
    "final_rung_pixels",
    "final_robot_pixels",
    "final_other_ladder_pixels",
    "final_ground_pixels",
)
WARP_CENTER_RAY_VALUE_NAMES = (
    "camera_position_x",
    "camera_position_y",
    "camera_position_z",
    "camera_quaternion_x",
    "camera_quaternion_y",
    "camera_quaternion_z",
    "camera_quaternion_w",
    "camera_ray_x",
    "camera_ray_y",
    "camera_ray_z",
    "world_ray_x",
    "world_ray_y",
    "world_ray_z",
    "forward_cosine",
    "nearest_ray_distance",
    "image_plane_depth",
    "nearest_class",
    "ground_ray_distance",
    "ladder_box_count",
    "robot_mesh_count",
    "robot_sphere_count",
    "first_ladder_position_x",
    "first_ladder_position_y",
    "first_ladder_position_z",
    "first_ladder_quaternion_x",
    "first_ladder_quaternion_y",
    "first_ladder_quaternion_z",
    "first_ladder_quaternion_w",
    "first_ladder_half_size_x",
    "first_ladder_half_size_y",
    "first_ladder_half_size_z",
    "first_ladder_class",
    "near_ray_distance",
)
_CENTER_RAY_LADDER_DISTANCE_OFFSET = len(WARP_CENTER_RAY_VALUE_NAMES)

_MAX_ROBOT_MESHES = 64
_MAX_ROBOT_SPHERES = 8
_MAX_LADDER_BOXES = 32
_RAY_EPSILON = 1.0e-6


@dataclass(frozen=True)
class WarpCameraCfg:
    """Pinhole projection and output crop used by the Warp renderer."""

    width: int = 80
    height: int = 60
    horizontal_fov_deg: float = 87.0
    vertical_fov_deg: float = 58.0
    horizontal_fov_margin_deg: float = 0.0
    vertical_fov_margin_deg: float = 0.0
    min_depth: float = 0.1
    max_depth: float = 2.0
    crop_top: int = 0
    crop_bottom: int = 0
    crop_left: int = 0
    crop_right: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Camera dimensions must be positive, got {self.width}x{self.height}")
        if not 0.0 < self.horizontal_fov_deg < 180.0:
            raise ValueError(f"horizontal_fov_deg must be in (0, 180), got {self.horizontal_fov_deg}")
        if not 0.0 < self.vertical_fov_deg < 180.0:
            raise ValueError(f"vertical_fov_deg must be in (0, 180), got {self.vertical_fov_deg}")
        if self.horizontal_fov_margin_deg < 0.0 or self.vertical_fov_margin_deg < 0.0:
            raise ValueError("FOV margins must be non-negative")
        if self.horizontal_fov_deg + self.horizontal_fov_margin_deg >= 180.0:
            raise ValueError("horizontal FOV plus margin must be less than 180 degrees")
        if self.vertical_fov_deg + self.vertical_fov_margin_deg >= 180.0:
            raise ValueError("vertical FOV plus margin must be less than 180 degrees")
        if not 0.0 <= self.min_depth < self.max_depth:
            raise ValueError(
                f"Depth bounds must satisfy 0 <= min < max, got [{self.min_depth}, {self.max_depth}]"
            )
        crops = (self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)
        if any(value < 0 for value in crops):
            raise ValueError(f"Crop values must be non-negative, got {crops}")
        if self.crop_top + self.crop_bottom >= self.height:
            raise ValueError("Top and bottom crop remove the entire image")
        if self.crop_left + self.crop_right >= self.width:
            raise ValueError("Left and right crop remove the entire image")

    @property
    def output_height(self) -> int:
        return self.height - self.crop_top - self.crop_bottom

    @property
    def output_width(self) -> int:
        return self.width - self.crop_left - self.crop_right

    @property
    def horizontal_tangent(self) -> float:
        return math.tan(math.radians(self.horizontal_fov_deg) / 2.0)

    @property
    def vertical_tangent(self) -> float:
        return math.tan(math.radians(self.vertical_fov_deg) / 2.0)

    @property
    def maximum_ray_distance(self) -> float:
        horizontal_tangent = math.tan(
            math.radians(self.horizontal_fov_deg + self.horizontal_fov_margin_deg) / 2.0
        )
        vertical_tangent = math.tan(
            math.radians(self.vertical_fov_deg + self.vertical_fov_margin_deg) / 2.0
        )
        corner_scale = math.sqrt(
            1.0 + horizontal_tangent**2 + vertical_tangent**2
        )
        return self.max_depth * corner_scale + 1.0e-3

    def crop(self, image: torch.Tensor) -> torch.Tensor:
        """Crop the final two image dimensions without changing camera FOV."""

        row_end = self.height - self.crop_bottom if self.crop_bottom else self.height
        column_end = self.width - self.crop_right if self.crop_right else self.width
        return image[..., self.crop_top : row_end, self.crop_left : column_end]


@dataclass(frozen=True)
class RobotCollisionGeometry:
    """One URDF collision geometry expressed in its owning link frame."""

    link_name: str
    kind: str
    origin_position: tuple[float, float, float]
    origin_quaternion_wxyz: tuple[float, float, float, float]
    mesh_path: Path | None = None
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sphere_radius: float | None = None


@dataclass(frozen=True)
class WarpCameraOutput:
    """Cropped metric image-plane depth and per-pixel hit class."""

    depth: torch.Tensor
    hit_class: torch.Tensor
    diagnostic_counts: torch.Tensor | None = None
    center_ray_values: torch.Tensor | None = None

    @property
    def rung_mask(self) -> torch.Tensor:
        return self.hit_class == RUNG

    @property
    def robot_mask(self) -> torch.Tensor:
        return self.hit_class == ROBOT

    @property
    def rfm_keep_mask(self) -> torch.Tensor:
        return self.rung_mask | self.robot_mask

    @property
    def rfm_depth(self) -> torch.Tensor:
        return torch.where(self.rfm_keep_mask, self.depth, torch.zeros_like(self.depth))


def aperture_from_fov(fov_deg: float, focal_length: float) -> float:
    """Return the USD camera aperture matching one pinhole field of view."""

    if not 0.0 < fov_deg < 180.0:
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")
    if focal_length <= 0.0:
        raise ValueError(f"focal_length must be positive, got {focal_length}")
    return 2.0 * focal_length * math.tan(math.radians(fov_deg) / 2.0)


def pinhole_fov_remap_grid(
    source_intrinsics: np.ndarray,
    *,
    output_width: int,
    output_height: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV maps from a target pinhole FOV into a source image."""

    intrinsics = np.asarray(source_intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"source_intrinsics must have shape (3, 3), got {intrinsics.shape}")
    if output_width <= 0 or output_height <= 0:
        raise ValueError(
            f"Output dimensions must be positive, got {output_width}x{output_height}"
        )
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError(
            f"horizontal_fov_deg must be in (0, 180), got {horizontal_fov_deg}"
        )
    if not 0.0 < vertical_fov_deg < 180.0:
        raise ValueError(f"vertical_fov_deg must be in (0, 180), got {vertical_fov_deg}")

    focal_x = float(intrinsics[0, 0])
    focal_y = float(intrinsics[1, 1])
    center_x = float(intrinsics[0, 2])
    center_y = float(intrinsics[1, 2])
    if focal_x <= 0.0 or focal_y <= 0.0:
        raise ValueError(f"Source focal lengths must be positive, got ({focal_x}, {focal_y})")

    columns = np.arange(output_width, dtype=np.float64) + 0.5
    rows = np.arange(output_height, dtype=np.float64) + 0.5
    target_x = (
        2.0 * columns / output_width - 1.0
    ) * math.tan(math.radians(horizontal_fov_deg) / 2.0)
    target_y_down = (
        2.0 * rows / output_height - 1.0
    ) * math.tan(math.radians(vertical_fov_deg) / 2.0)
    map_x = focal_x * target_x[None, :] + center_x - 0.5
    map_y = focal_y * target_y_down[:, None] + center_y - 0.5
    return (
        np.broadcast_to(map_x, (output_height, output_width)).astype(np.float32).copy(),
        np.broadcast_to(map_y, (output_height, output_width)).astype(np.float32).copy(),
    )


def load_robot_collision_geometries(urdf_path: str | Path) -> tuple[RobotCollisionGeometry, ...]:
    """Parse CR1 mesh and sphere collisions from a URDF."""

    path = Path(urdf_path).expanduser().resolve()
    root = ElementTree.parse(path).getroot()
    geometries: list[RobotCollisionGeometry] = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            position = _numbers(origin.attrib.get("xyz", "0 0 0"), 3) if origin is not None else np.zeros(3)
            rpy = _numbers(origin.attrib.get("rpy", "0 0 0"), 3) if origin is not None else np.zeros(3)
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError(f"Expected one collision shape for CR1 link {link_name!r}")
            shape = geometry[0]
            if shape.tag == "mesh":
                filename = shape.attrib.get("filename", "")
                if not filename:
                    raise ValueError(f"Missing collision mesh filename for CR1 link {link_name!r}")
                mesh_path = _resolve_mesh_path(path.parent, filename)
                scale = _numbers(shape.attrib.get("scale", "1 1 1"), 3)
                geometries.append(
                    RobotCollisionGeometry(
                        link_name=link_name,
                        kind="mesh",
                        origin_position=tuple(position.tolist()),
                        origin_quaternion_wxyz=tuple(_rpy_quaternion_wxyz(rpy).tolist()),
                        mesh_path=mesh_path,
                        mesh_scale=tuple(scale.tolist()),
                    )
                )
            elif shape.tag == "sphere":
                geometries.append(
                    RobotCollisionGeometry(
                        link_name=link_name,
                        kind="sphere",
                        origin_position=tuple(position.tolist()),
                        origin_quaternion_wxyz=tuple(_rpy_quaternion_wxyz(rpy).tolist()),
                        sphere_radius=float(shape.attrib["radius"]),
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported CR1 collision shape {shape.tag!r} on link {link_name!r}"
                )
    return tuple(geometries)


def _numbers(value: str, count: int) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split()], dtype=np.float32)
    if values.shape != (count,):
        raise ValueError(f"Expected {count} values, got {value!r}")
    return values


def _resolve_mesh_path(urdf_directory: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        filename = filename.removeprefix("package://")
    path = Path(filename)
    if not path.is_absolute():
        path = urdf_directory / path
    return path.resolve()


def _rpy_quaternion_wxyz(rpy: np.ndarray) -> np.ndarray:
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
        dtype=np.float32,
    )


def _torch_quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion_vector = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(quaternion_vector, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(
        quaternion_vector, twice_cross, dim=-1
    )


def _wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., 1:], quaternion[..., :1]), dim=-1).contiguous()


@wp.func
def _ray_box_intersection(
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    center: wp.vec3,
    half_size: wp.vec3,
    rotation: wp.quat,
) -> float:
    inverse_rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], rotation[3])
    origin = wp.quat_rotate(inverse_rotation, ray_origin - center)
    direction = wp.quat_rotate(inverse_rotation, ray_direction)
    near = -1.0e10
    far = 1.0e10

    for axis in range(3):
        if wp.abs(direction[axis]) < 1.0e-12:
            if origin[axis] < -half_size[axis] or origin[axis] > half_size[axis]:
                return -1.0
        else:
            inverse_direction = 1.0 / direction[axis]
            first = (-half_size[axis] - origin[axis]) * inverse_direction
            second = (half_size[axis] - origin[axis]) * inverse_direction
            if first > second:
                temporary = first
                first = second
                second = temporary
            near = wp.max(near, first)
            far = wp.min(far, second)
            if near > far:
                return -1.0

    if near > _RAY_EPSILON:
        return near
    if far > _RAY_EPSILON:
        return far
    return -1.0


@wp.func
def _ray_sphere_intersection(
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    center: wp.vec3,
    radius: float,
) -> float:
    relative_origin = ray_origin - center
    projection = wp.dot(relative_origin, ray_direction)
    constant = wp.dot(relative_origin, relative_origin) - radius * radius
    discriminant = projection * projection - constant
    if discriminant < 0.0:
        return -1.0
    root = wp.sqrt(discriminant)
    near = -projection - root
    if near > _RAY_EPSILON:
        return near
    far = -projection + root
    if far > _RAY_EPSILON:
        return far
    return -1.0


@wp.kernel
def _render_cr1_ladder_kernel(
    width: int,
    height: int,
    tan_half_fov_x: float,
    tan_half_fov_y: float,
    min_depth: float,
    max_depth: float,
    maximum_ray_distance: float,
    camera_position: wp.array(dtype=wp.vec3),
    camera_rotation: wp.array(dtype=wp.quat),
    robot_mesh_positions: wp.array(dtype=wp.vec3),
    robot_mesh_rotations: wp.array(dtype=wp.quat),
    robot_mesh_ids: wp.array(dtype=wp.uint64),
    robot_mesh_count: int,
    robot_sphere_centers: wp.array(dtype=wp.vec3),
    robot_sphere_radii: wp.array(dtype=wp.float32),
    robot_sphere_count: int,
    ladder_positions: wp.array(dtype=wp.vec3),
    ladder_rotations: wp.array(dtype=wp.quat),
    ladder_half_sizes: wp.array(dtype=wp.vec3),
    ladder_classes: wp.array(dtype=wp.int32),
    ladder_box_count: int,
    ground_height: float,
    ground_enabled: int,
    diagnostics_enabled: int,
    depth: wp.array(dtype=wp.float32, ndim=2),
    hit_class: wp.array(dtype=wp.int32, ndim=2),
    diagnostic_counts: wp.array(dtype=wp.int32),
    center_ray_values: wp.array(dtype=wp.float32),
):
    pixel = wp.tid()
    row = pixel // width
    column = pixel - row * width

    screen_x = 2.0 * (float(column) + 0.5) / float(width) - 1.0
    screen_y = 1.0 - 2.0 * (float(row) + 0.5) / float(height)
    camera_ray_x = screen_x * tan_half_fov_x
    camera_ray_y = screen_y * tan_half_fov_y
    camera_ray = wp.vec3(camera_ray_x, camera_ray_y, -1.0)
    forward_cosine = 1.0 / wp.length(camera_ray)

    ray_origin = camera_position[0]
    ray_direction = wp.normalize(wp.quat_rotate(camera_rotation[0], camera_ray))
    near_ray_distance = min_depth / forward_cosine
    query_origin = ray_origin + near_ray_distance * ray_direction
    nearest = maximum_ray_distance
    nearest_class = int(BACKGROUND)
    any_ladder_hit = int(0)
    any_rung_hit = int(0)
    any_other_ladder_hit = int(0)
    any_robot_mesh_hit = int(0)
    any_robot_sphere_hit = int(0)
    ground_distance = float(-1.0)

    if diagnostics_enabled != 0:
        wp.atomic_add(diagnostic_counts, 0, 1)
        direction_length = wp.length(ray_direction)
        if direction_length > 0.999 and direction_length < 1.001:
            wp.atomic_add(diagnostic_counts, 1, 1)
        if row == height // 2 and column == width // 2:
            center_ray_values[0] = ray_origin[0]
            center_ray_values[1] = ray_origin[1]
            center_ray_values[2] = ray_origin[2]
            center_ray_values[3] = camera_rotation[0][0]
            center_ray_values[4] = camera_rotation[0][1]
            center_ray_values[5] = camera_rotation[0][2]
            center_ray_values[6] = camera_rotation[0][3]
            center_ray_values[7] = camera_ray[0]
            center_ray_values[8] = camera_ray[1]
            center_ray_values[9] = camera_ray[2]
            center_ray_values[10] = ray_direction[0]
            center_ray_values[11] = ray_direction[1]
            center_ray_values[12] = ray_direction[2]
            center_ray_values[13] = forward_cosine
            center_ray_values[18] = float(ladder_box_count)
            center_ray_values[19] = float(robot_mesh_count)
            center_ray_values[20] = float(robot_sphere_count)
            if ladder_box_count > 0:
                center_ray_values[21] = ladder_positions[0][0]
                center_ray_values[22] = ladder_positions[0][1]
                center_ray_values[23] = ladder_positions[0][2]
                center_ray_values[24] = ladder_rotations[0][0]
                center_ray_values[25] = ladder_rotations[0][1]
                center_ray_values[26] = ladder_rotations[0][2]
                center_ray_values[27] = ladder_rotations[0][3]
                center_ray_values[28] = ladder_half_sizes[0][0]
                center_ray_values[29] = ladder_half_sizes[0][1]
                center_ray_values[30] = ladder_half_sizes[0][2]
                center_ray_values[31] = float(ladder_classes[0])
            center_ray_values[32] = near_ray_distance

    for box_index in range(_MAX_LADDER_BOXES):
        if box_index >= ladder_box_count:
            break
        distance = _ray_box_intersection(
            query_origin,
            ray_direction,
            ladder_positions[box_index],
            ladder_half_sizes[box_index],
            ladder_rotations[box_index],
        )
        if distance > _RAY_EPSILON:
            distance = distance + near_ray_distance
        if diagnostics_enabled != 0 and row == height // 2 and column == width // 2:
            center_ray_values[_CENTER_RAY_LADDER_DISTANCE_OFFSET + box_index] = distance
        if distance > _RAY_EPSILON:
            any_ladder_hit = 1
            if int(ladder_classes[box_index]) == int(RUNG):
                any_rung_hit = 1
            else:
                any_other_ladder_hit = 1
        if distance > _RAY_EPSILON and distance < nearest:
            nearest = distance
            nearest_class = int(ladder_classes[box_index])

    for mesh_index in range(_MAX_ROBOT_MESHES):
        if mesh_index >= robot_mesh_count:
            break
        link_rotation = robot_mesh_rotations[mesh_index]
        inverse_rotation = wp.quat(
            -link_rotation[0], -link_rotation[1], -link_rotation[2], link_rotation[3]
        )
        local_origin = wp.quat_rotate(
            inverse_rotation, query_origin - robot_mesh_positions[mesh_index]
        )
        local_direction = wp.quat_rotate(inverse_rotation, ray_direction)
        distance = float(0.0)
        barycentric_u = float(0.0)
        barycentric_v = float(0.0)
        sign = float(0.0)
        normal = wp.vec3()
        face = int(0)
        if wp.mesh_query_ray(
            robot_mesh_ids[mesh_index],
            local_origin,
            local_direction,
            maximum_ray_distance,
            distance,
            barycentric_u,
            barycentric_v,
            sign,
            normal,
            face,
        ):
            if distance > _RAY_EPSILON:
                distance = distance + near_ray_distance
                any_robot_mesh_hit = 1
                if distance < nearest:
                    nearest = distance
                    nearest_class = int(ROBOT)

    for sphere_index in range(_MAX_ROBOT_SPHERES):
        if sphere_index >= robot_sphere_count:
            break
        distance = _ray_sphere_intersection(
            query_origin,
            ray_direction,
            robot_sphere_centers[sphere_index],
            robot_sphere_radii[sphere_index],
        )
        if distance > _RAY_EPSILON:
            distance = distance + near_ray_distance
            any_robot_sphere_hit = 1
        if distance > _RAY_EPSILON and distance < nearest:
            nearest = distance
            nearest_class = int(ROBOT)

    if ground_enabled != 0 and wp.abs(ray_direction[2]) > 1.0e-12:
        ground_distance = (ground_height - query_origin[2]) / ray_direction[2]
        if ground_distance > _RAY_EPSILON:
            ground_distance = ground_distance + near_ray_distance
        if ground_distance > _RAY_EPSILON and ground_distance < nearest:
            nearest = ground_distance
            nearest_class = int(GROUND)

    image_plane_depth = nearest * forward_cosine
    if image_plane_depth < min_depth or image_plane_depth > max_depth:
        depth[row, column] = max_depth
        hit_class[row, column] = int(BACKGROUND)
    else:
        depth[row, column] = image_plane_depth
        hit_class[row, column] = nearest_class

    if diagnostics_enabled != 0:
        if any_ladder_hit != 0:
            wp.atomic_add(diagnostic_counts, 2, 1)
        if any_rung_hit != 0:
            wp.atomic_add(diagnostic_counts, 3, 1)
        if any_other_ladder_hit != 0:
            wp.atomic_add(diagnostic_counts, 4, 1)
        if any_robot_mesh_hit != 0:
            wp.atomic_add(diagnostic_counts, 5, 1)
        if any_robot_sphere_hit != 0:
            wp.atomic_add(diagnostic_counts, 6, 1)
        if ground_distance > _RAY_EPSILON:
            wp.atomic_add(diagnostic_counts, 7, 1)
        if image_plane_depth >= min_depth and image_plane_depth <= max_depth:
            wp.atomic_add(diagnostic_counts, 8, 1)
        final_class = int(hit_class[row, column])
        if final_class == int(BACKGROUND):
            wp.atomic_add(diagnostic_counts, 9, 1)
        elif final_class == int(RUNG):
            wp.atomic_add(diagnostic_counts, 10, 1)
        elif final_class == int(ROBOT):
            wp.atomic_add(diagnostic_counts, 11, 1)
        elif final_class == int(LADDER_OTHER):
            wp.atomic_add(diagnostic_counts, 12, 1)
        elif final_class == int(GROUND):
            wp.atomic_add(diagnostic_counts, 13, 1)
        if row == height // 2 and column == width // 2:
            center_ray_values[14] = nearest
            center_ray_values[15] = image_plane_depth
            center_ray_values[16] = float(nearest_class)
            center_ray_values[17] = ground_distance


class CR1WarpCamera:
    """Render one CR1/wood-step scene from dynamic Isaac Lab poses."""

    def __init__(
        self,
        cfg: WarpCameraCfg,
        *,
        robot_urdf: str | Path,
        robot_body_names: list[str] | tuple[str, ...],
        ladder_box_specs: tuple[object, ...],
        ladder_object_names: list[str] | tuple[str, ...],
        rung_box_names: set[str] | frozenset[str] | tuple[str, ...],
        device: str | torch.device,
    ):
        self.cfg = cfg
        self.device = str(device)
        self._torch_device = torch.device(device)
        geometries = load_robot_collision_geometries(robot_urdf)
        mesh_geometries = [geometry for geometry in geometries if geometry.kind == "mesh"]
        sphere_geometries = [geometry for geometry in geometries if geometry.kind == "sphere"]
        if len(mesh_geometries) > _MAX_ROBOT_MESHES:
            raise ValueError(f"CR1 has {len(mesh_geometries)} meshes; maximum is {_MAX_ROBOT_MESHES}")
        if len(sphere_geometries) > _MAX_ROBOT_SPHERES:
            raise ValueError(f"CR1 has {len(sphere_geometries)} spheres; maximum is {_MAX_ROBOT_SPHERES}")
        if len(ladder_object_names) > _MAX_LADDER_BOXES:
            raise ValueError(
                f"Ladder has {len(ladder_object_names)} boxes; maximum is {_MAX_LADDER_BOXES}"
            )

        body_indexes = {name: index for index, name in enumerate(robot_body_names)}
        missing_links = sorted(
            {geometry.link_name for geometry in geometries} - body_indexes.keys()
        )
        if missing_links:
            raise ValueError(f"CR1 collision links missing from Isaac Lab body_names: {missing_links}")
        self._mesh_body_indexes = torch.as_tensor(
            [body_indexes[geometry.link_name] for geometry in mesh_geometries],
            dtype=torch.long,
            device=self._torch_device,
        )
        self._sphere_body_indexes = torch.as_tensor(
            [body_indexes[geometry.link_name] for geometry in sphere_geometries],
            dtype=torch.long,
            device=self._torch_device,
        )
        self._sphere_local_centers = torch.as_tensor(
            [geometry.origin_position for geometry in sphere_geometries],
            dtype=torch.float32,
            device=self._torch_device,
        ).reshape(-1, 3)
        self._sphere_radii = torch.as_tensor(
            [geometry.sphere_radius for geometry in sphere_geometries],
            dtype=torch.float32,
            device=self._torch_device,
        )

        self._warp_meshes: list[wp.Mesh] = []
        mesh_ids: list[int] = []
        with wp.ScopedDevice(self.device):
            for geometry in mesh_geometries:
                vertices, faces = _load_collision_mesh(geometry)
                mesh = wp.Mesh(
                    points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                    indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=self.device),
                )
                self._warp_meshes.append(mesh)
                mesh_ids.append(int(mesh.id))
        # Mesh BVHs are created on Warp's stream.  Rendering CUDA frames on the
        # current Torch stream below requires the static acceleration structures
        # to be complete first.
        wp.synchronize_device(self.device)
        self._mesh_ids = torch.as_tensor(
            mesh_ids, dtype=torch.int64, device=self._torch_device
        )
        self._mesh_ids_wp = wp.from_torch(self._mesh_ids, dtype=wp.uint64)

        specs_by_name = {spec.name: spec for spec in ladder_box_specs}
        missing_boxes = sorted(set(ladder_object_names) - specs_by_name.keys())
        if missing_boxes:
            raise ValueError(f"Ladder object names missing box specs: {missing_boxes}")
        rung_names = set(rung_box_names)
        self._ladder_half_sizes = torch.as_tensor(
            np.asarray(
                [np.asarray(specs_by_name[name].size) * 0.5 for name in ladder_object_names]
            ),
            dtype=torch.float32,
            device=self._torch_device,
        )
        self._ladder_classes = torch.as_tensor(
            [RUNG if name in rung_names else LADDER_OTHER for name in ladder_object_names],
            dtype=torch.int32,
            device=self._torch_device,
        )
        self._ladder_half_sizes_wp = wp.from_torch(self._ladder_half_sizes, dtype=wp.vec3)
        self._ladder_classes_wp = wp.from_torch(self._ladder_classes, dtype=wp.int32)
        self._robot_mesh_count = len(mesh_geometries)
        self._robot_sphere_count = len(sphere_geometries)
        self._ladder_box_count = len(ladder_object_names)
        self._depth = torch.empty(
            (cfg.height, cfg.width), dtype=torch.float32, device=self._torch_device
        )
        self._hit_class = torch.empty(
            (cfg.height, cfg.width), dtype=torch.int32, device=self._torch_device
        )
        self._diagnostic_counts = torch.zeros(
            len(WARP_DIAGNOSTIC_COUNT_NAMES), dtype=torch.int32, device=self._torch_device
        )
        self._center_ray_values = torch.full(
            (_CENTER_RAY_LADDER_DISTANCE_OFFSET + _MAX_LADDER_BOXES,),
            float("nan"),
            dtype=torch.float32,
            device=self._torch_device,
        )
        self._frame_tensors: tuple[torch.Tensor, ...] = ()

    def render(
        self,
        *,
        camera_position_w: torch.Tensor,
        camera_quaternion_wxyz_opengl: torch.Tensor,
        robot_body_positions_w: torch.Tensor,
        robot_body_quaternions_wxyz: torch.Tensor,
        ladder_positions_w: torch.Tensor,
        ladder_quaternions_wxyz: torch.Tensor,
        ground_height: float,
        include_robot: bool = True,
        include_ladder: bool = True,
        include_ground: bool = True,
        collect_diagnostics: bool = False,
    ) -> WarpCameraOutput:
        """Render one frame from current Isaac Lab actor-frame poses."""

        camera_position = _single_pose_tensor(camera_position_w, 3, "camera_position_w")
        camera_quaternion = _single_pose_tensor(
            camera_quaternion_wxyz_opengl, 4, "camera_quaternion_wxyz_opengl"
        )
        robot_positions = _single_environment_tensor(
            robot_body_positions_w, 3, "robot_body_positions_w"
        )
        robot_quaternions = _single_environment_tensor(
            robot_body_quaternions_wxyz, 4, "robot_body_quaternions_wxyz"
        )
        ladder_positions = _single_environment_tensor(
            ladder_positions_w, 3, "ladder_positions_w"
        )
        ladder_quaternions = _single_environment_tensor(
            ladder_quaternions_wxyz, 4, "ladder_quaternions_wxyz"
        )
        if ladder_positions.shape[0] != self._ladder_box_count:
            raise ValueError(
                f"Expected {self._ladder_box_count} ladder poses, got {ladder_positions.shape[0]}"
            )

        mesh_positions = robot_positions[self._mesh_body_indexes].contiguous()
        mesh_rotations = _wxyz_to_xyzw(
            robot_quaternions[self._mesh_body_indexes]
        )
        sphere_link_positions = robot_positions[self._sphere_body_indexes]
        sphere_link_quaternions = robot_quaternions[self._sphere_body_indexes]
        sphere_centers = (
            sphere_link_positions
            + _torch_quat_apply_wxyz(sphere_link_quaternions, self._sphere_local_centers)
        ).contiguous()
        camera_rotation = _wxyz_to_xyzw(camera_quaternion)
        ladder_rotations = _wxyz_to_xyzw(ladder_quaternions)

        frame_tensors = (
            camera_position,
            camera_rotation,
            mesh_positions,
            mesh_rotations,
            sphere_centers,
            ladder_positions,
            ladder_rotations,
        )
        self._frame_tensors = frame_tensors
        robot_mesh_count = self._robot_mesh_count if include_robot else 0
        robot_sphere_count = self._robot_sphere_count if include_robot else 0
        ladder_box_count = self._ladder_box_count if include_ladder else 0
        if collect_diagnostics:
            self._diagnostic_counts.zero_()
            self._center_ray_values.fill_(float("nan"))
        launch_stream = (
            wp.stream_from_torch(self._torch_device)
            if self._torch_device.type == "cuda"
            else None
        )
        wp.launch(
            kernel=_render_cr1_ladder_kernel,
            dim=self.cfg.width * self.cfg.height,
            inputs=[
                self.cfg.width,
                self.cfg.height,
                self.cfg.horizontal_tangent,
                self.cfg.vertical_tangent,
                self.cfg.min_depth,
                self.cfg.max_depth,
                self.cfg.maximum_ray_distance,
                wp.from_torch(camera_position, dtype=wp.vec3),
                wp.from_torch(camera_rotation, dtype=wp.quat),
                wp.from_torch(mesh_positions, dtype=wp.vec3),
                wp.from_torch(mesh_rotations, dtype=wp.quat),
                self._mesh_ids_wp,
                robot_mesh_count,
                wp.from_torch(sphere_centers, dtype=wp.vec3),
                wp.from_torch(self._sphere_radii, dtype=wp.float32),
                robot_sphere_count,
                wp.from_torch(ladder_positions.contiguous(), dtype=wp.vec3),
                wp.from_torch(ladder_rotations, dtype=wp.quat),
                self._ladder_half_sizes_wp,
                self._ladder_classes_wp,
                ladder_box_count,
                float(ground_height),
                int(include_ground),
                int(collect_diagnostics),
            ],
            outputs=[
                wp.from_torch(self._depth, dtype=wp.float32),
                wp.from_torch(self._hit_class, dtype=wp.int32),
                wp.from_torch(self._diagnostic_counts, dtype=wp.int32),
                wp.from_torch(self._center_ray_values, dtype=wp.float32),
            ],
            device=self.device,
            stream=launch_stream,
        )
        # Warp launches asynchronously on CUDA.  The output is consumed through
        # zero-copy Torch aliases immediately below, so finish this device's work
        # before handing the tensors to the caller.
        wp.synchronize_device(self.device)
        return WarpCameraOutput(
            depth=self.cfg.crop(self._depth),
            hit_class=self.cfg.crop(self._hit_class),
            diagnostic_counts=(self._diagnostic_counts if collect_diagnostics else None),
            center_ray_values=(self._center_ray_values if collect_diagnostics else None),
        )


@wp.kernel
def _render_cr1_ladder_batch_kernel(
    width: int,
    height: int,
    tan_half_fov_x: wp.array(dtype=wp.float32),
    tan_half_fov_y: wp.array(dtype=wp.float32),
    min_depth: float,
    max_depth: float,
    maximum_ray_distance: float,
    render_env_ids: wp.array(dtype=wp.int32),
    camera_positions: wp.array(dtype=wp.vec3),
    camera_rotations: wp.array(dtype=wp.quat),
    robot_mesh_positions: wp.array(dtype=wp.vec3, ndim=2),
    robot_mesh_rotations: wp.array(dtype=wp.quat, ndim=2),
    robot_mesh_ids: wp.array(dtype=wp.uint64),
    robot_mesh_count: int,
    robot_sphere_centers: wp.array(dtype=wp.vec3, ndim=2),
    robot_sphere_radii: wp.array(dtype=wp.float32),
    robot_sphere_count: int,
    ladder_positions: wp.array(dtype=wp.vec3, ndim=2),
    ladder_rotations: wp.array(dtype=wp.quat, ndim=2),
    ladder_half_sizes: wp.array(dtype=wp.vec3),
    ladder_classes: wp.array(dtype=wp.int32),
    ladder_box_count: int,
    ground_height: float,
    ground_enabled: int,
    depth: wp.array(dtype=wp.float32, ndim=3),
    hit_class: wp.array(dtype=wp.int32, ndim=3),
):
    linear_index = wp.tid()
    pixels_per_image = width * height
    render_slot = linear_index // pixels_per_image
    pixel = linear_index - render_slot * pixels_per_image
    environment_index = render_env_ids[render_slot]
    row = pixel // width
    column = pixel - row * width

    screen_x = 2.0 * (float(column) + 0.5) / float(width) - 1.0
    screen_y = 1.0 - 2.0 * (float(row) + 0.5) / float(height)
    camera_ray = wp.vec3(
        screen_x * tan_half_fov_x[render_slot],
        screen_y * tan_half_fov_y[render_slot],
        -1.0,
    )
    forward_cosine = 1.0 / wp.length(camera_ray)
    ray_origin = camera_positions[render_slot]
    ray_direction = wp.normalize(wp.quat_rotate(camera_rotations[render_slot], camera_ray))

    # Start ray queries on the near plane. Otherwise collision geometry enclosing
    # the optical center can hide valid ladder intersections behind it.
    near_ray_distance = min_depth / forward_cosine
    query_origin = ray_origin + near_ray_distance * ray_direction
    nearest = maximum_ray_distance
    nearest_class = int(BACKGROUND)

    for box_index in range(_MAX_LADDER_BOXES):
        if box_index >= ladder_box_count:
            break
        distance = _ray_box_intersection(
            query_origin,
            ray_direction,
            ladder_positions[render_slot, box_index],
            ladder_half_sizes[box_index],
            ladder_rotations[render_slot, box_index],
        )
        if distance > _RAY_EPSILON:
            distance = distance + near_ray_distance
        if distance > _RAY_EPSILON and distance < nearest:
            nearest = distance
            nearest_class = int(ladder_classes[box_index])

    for mesh_index in range(_MAX_ROBOT_MESHES):
        if mesh_index >= robot_mesh_count:
            break
        link_rotation = robot_mesh_rotations[render_slot, mesh_index]
        inverse_rotation = wp.quat(
            -link_rotation[0], -link_rotation[1], -link_rotation[2], link_rotation[3]
        )
        local_origin = wp.quat_rotate(
            inverse_rotation, query_origin - robot_mesh_positions[render_slot, mesh_index]
        )
        local_direction = wp.quat_rotate(inverse_rotation, ray_direction)
        distance = float(0.0)
        barycentric_u = float(0.0)
        barycentric_v = float(0.0)
        sign = float(0.0)
        normal = wp.vec3()
        face = int(0)
        if wp.mesh_query_ray(
            robot_mesh_ids[mesh_index],
            local_origin,
            local_direction,
            maximum_ray_distance,
            distance,
            barycentric_u,
            barycentric_v,
            sign,
            normal,
            face,
        ):
            if distance > _RAY_EPSILON:
                distance = distance + near_ray_distance
                if distance < nearest:
                    nearest = distance
                    nearest_class = int(ROBOT)

    for sphere_index in range(_MAX_ROBOT_SPHERES):
        if sphere_index >= robot_sphere_count:
            break
        distance = _ray_sphere_intersection(
            query_origin,
            ray_direction,
            robot_sphere_centers[render_slot, sphere_index],
            robot_sphere_radii[sphere_index],
        )
        if distance > _RAY_EPSILON:
            distance = distance + near_ray_distance
        if distance > _RAY_EPSILON and distance < nearest:
            nearest = distance
            nearest_class = int(ROBOT)

    if ground_enabled != 0 and wp.abs(ray_direction[2]) > 1.0e-12:
        ground_distance = (ground_height - query_origin[2]) / ray_direction[2]
        if ground_distance > _RAY_EPSILON:
            ground_distance = ground_distance + near_ray_distance
        if ground_distance > _RAY_EPSILON and ground_distance < nearest:
            nearest = ground_distance
            nearest_class = int(GROUND)

    image_plane_depth = nearest * forward_cosine
    if image_plane_depth < min_depth or image_plane_depth > max_depth:
        depth[environment_index, row, column] = max_depth
        hit_class[environment_index, row, column] = int(BACKGROUND)
    else:
        depth[environment_index, row, column] = image_plane_depth
        hit_class[environment_index, row, column] = nearest_class


class CR1WarpBatchCamera:
    """Batched CR1 depth renderer for Warp training-path validation.

    Inputs are GPU tensors from the current IsaacLab scene. A render call
    launches one kernel over all requested environments and never synchronizes
    the device, so subsequent GPU augmentation and policy code can consume the
    depth buffers on the same stream.
    """

    def __init__(
        self,
        cfg: WarpCameraCfg,
        *,
        num_envs: int,
        robot_urdf: str | Path,
        robot_body_names: list[str] | tuple[str, ...],
        ladder_box_specs: tuple[object, ...],
        ladder_object_names: list[str] | tuple[str, ...],
        rung_box_names: set[str] | frozenset[str] | tuple[str, ...],
        device: str | torch.device,
    ):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = str(device)
        self._torch_device = torch.device(device)
        geometries = load_robot_collision_geometries(robot_urdf)
        mesh_geometries = [geometry for geometry in geometries if geometry.kind == "mesh"]
        sphere_geometries = [geometry for geometry in geometries if geometry.kind == "sphere"]
        if len(mesh_geometries) > _MAX_ROBOT_MESHES:
            raise ValueError(f"CR1 has {len(mesh_geometries)} meshes; maximum is {_MAX_ROBOT_MESHES}")
        if len(sphere_geometries) > _MAX_ROBOT_SPHERES:
            raise ValueError(f"CR1 has {len(sphere_geometries)} spheres; maximum is {_MAX_ROBOT_SPHERES}")
        if len(ladder_object_names) > _MAX_LADDER_BOXES:
            raise ValueError(
                f"Ladder has {len(ladder_object_names)} boxes; maximum is {_MAX_LADDER_BOXES}"
            )

        body_indexes = {name: index for index, name in enumerate(robot_body_names)}
        missing_links = sorted(
            {geometry.link_name for geometry in geometries} - body_indexes.keys()
        )
        if missing_links:
            raise ValueError(f"CR1 collision links missing from Isaac Lab body_names: {missing_links}")
        self._mesh_body_indexes = torch.as_tensor(
            [body_indexes[geometry.link_name] for geometry in mesh_geometries],
            dtype=torch.long,
            device=self._torch_device,
        )
        self._sphere_body_indexes = torch.as_tensor(
            [body_indexes[geometry.link_name] for geometry in sphere_geometries],
            dtype=torch.long,
            device=self._torch_device,
        )
        self._sphere_local_centers = torch.as_tensor(
            [geometry.origin_position for geometry in sphere_geometries],
            dtype=torch.float32,
            device=self._torch_device,
        ).reshape(-1, 3)
        self._sphere_radii = torch.as_tensor(
            [geometry.sphere_radius for geometry in sphere_geometries],
            dtype=torch.float32,
            device=self._torch_device,
        )

        self._warp_meshes: list[wp.Mesh] = []
        mesh_ids: list[int] = []
        with wp.ScopedDevice(self.device):
            for geometry in mesh_geometries:
                vertices, faces = _load_collision_mesh(geometry)
                mesh = wp.Mesh(
                    points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                    indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=self.device),
                )
                self._warp_meshes.append(mesh)
                mesh_ids.append(int(mesh.id))
        # Mesh BVHs are static and constructed once. This initialization barrier
        # is intentionally not part of the per-frame rendering path.
        wp.synchronize_device(self.device)
        self._mesh_ids = torch.as_tensor(mesh_ids, dtype=torch.int64, device=self._torch_device)
        self._mesh_ids_wp = wp.from_torch(self._mesh_ids, dtype=wp.uint64)

        specs_by_name = {spec.name: spec for spec in ladder_box_specs}
        missing_boxes = sorted(set(ladder_object_names) - specs_by_name.keys())
        if missing_boxes:
            raise ValueError(f"Ladder object names missing box specs: {missing_boxes}")
        rung_names = set(rung_box_names)
        self._ladder_half_sizes = torch.as_tensor(
            np.asarray([np.asarray(specs_by_name[name].size) * 0.5 for name in ladder_object_names]),
            dtype=torch.float32,
            device=self._torch_device,
        )
        self._ladder_classes = torch.as_tensor(
            [RUNG if name in rung_names else LADDER_OTHER for name in ladder_object_names],
            dtype=torch.int32,
            device=self._torch_device,
        )
        self._ladder_half_sizes_wp = wp.from_torch(self._ladder_half_sizes, dtype=wp.vec3)
        self._ladder_classes_wp = wp.from_torch(self._ladder_classes, dtype=wp.int32)
        self._robot_mesh_count = len(mesh_geometries)
        self._robot_sphere_count = len(sphere_geometries)
        self._ladder_box_count = len(ladder_object_names)
        self._depth = torch.empty(
            (num_envs, cfg.height, cfg.width), dtype=torch.float32, device=self._torch_device
        )
        self._hit_class = torch.empty(
            (num_envs, cfg.height, cfg.width), dtype=torch.int32, device=self._torch_device
        )
        self._all_env_ids = torch.arange(
            num_envs, dtype=torch.int32, device=self._torch_device
        )
        self._default_horizontal_tangents = torch.full(
            (num_envs,), cfg.horizontal_tangent, dtype=torch.float32, device=self._torch_device
        )
        self._default_vertical_tangents = torch.full(
            (num_envs,), cfg.vertical_tangent, dtype=torch.float32, device=self._torch_device
        )
        self._frame_tensors: tuple[torch.Tensor, ...] = ()

    def render(
        self,
        *,
        camera_positions_w: torch.Tensor,
        camera_quaternions_wxyz_opengl: torch.Tensor,
        robot_body_positions_w: torch.Tensor,
        robot_body_quaternions_wxyz: torch.Tensor,
        ladder_positions_w: torch.Tensor,
        ladder_quaternions_wxyz: torch.Tensor,
        ground_height: float,
        env_ids: torch.Tensor | None = None,
        horizontal_tangents: torch.Tensor | None = None,
        vertical_tangents: torch.Tensor | None = None,
        include_robot: bool = True,
        include_ladder: bool = True,
        include_ground: bool = True,
    ) -> WarpCameraOutput:
        """Render one native depth image for each input environment.

        ``env_ids`` maps the leading input dimension into this renderer's
        preallocated buffers. It is useful for a future lower-rate visual
        update; the common all-environment case returns buffer views directly.
        """

        camera_positions = _batch_tensor(camera_positions_w, 2, 3, "camera_positions_w", self._torch_device)
        camera_quaternions = _batch_tensor(
            camera_quaternions_wxyz_opengl,
            2,
            4,
            "camera_quaternions_wxyz_opengl",
            self._torch_device,
        )
        robot_positions = _batch_tensor(
            robot_body_positions_w, 3, 3, "robot_body_positions_w", self._torch_device
        )
        robot_quaternions = _batch_tensor(
            robot_body_quaternions_wxyz,
            3,
            4,
            "robot_body_quaternions_wxyz",
            self._torch_device,
        )
        ladder_positions = _batch_tensor(
            ladder_positions_w, 3, 3, "ladder_positions_w", self._torch_device
        )
        ladder_quaternions = _batch_tensor(
            ladder_quaternions_wxyz,
            3,
            4,
            "ladder_quaternions_wxyz",
            self._torch_device,
        )
        batch_size = camera_positions.shape[0]
        if batch_size == 0:
            raise ValueError("Batch renderer requires at least one environment")
        if camera_quaternions.shape[0] != batch_size:
            raise ValueError("camera position and quaternion batch sizes must match")
        for name, tensor in (
            ("robot_body_positions_w", robot_positions),
            ("robot_body_quaternions_wxyz", robot_quaternions),
            ("ladder_positions_w", ladder_positions),
            ("ladder_quaternions_wxyz", ladder_quaternions),
        ):
            if tensor.shape[0] != batch_size:
                raise ValueError(f"{name} batch size must match camera positions")
        if robot_positions.shape[1] != robot_quaternions.shape[1]:
            raise ValueError("robot position and quaternion body counts must match")
        if ladder_positions.shape[1] != self._ladder_box_count:
            raise ValueError(
                f"Expected {self._ladder_box_count} ladder poses, got {ladder_positions.shape[1]}"
            )
        if ladder_quaternions.shape[1] != self._ladder_box_count:
            raise ValueError(
                f"Expected {self._ladder_box_count} ladder quaternions, got {ladder_quaternions.shape[1]}"
            )

        if env_ids is None:
            if batch_size > self.num_envs:
                raise ValueError(
                    f"Batch size {batch_size} exceeds configured num_envs {self.num_envs}"
                )
            render_env_ids = self._all_env_ids[:batch_size]
        else:
            render_env_ids = torch.as_tensor(env_ids, dtype=torch.int32, device=self._torch_device)
            if render_env_ids.ndim != 1 or render_env_ids.numel() != batch_size:
                raise ValueError(
                    f"env_ids must have shape ({batch_size},), got {tuple(render_env_ids.shape)}"
                )
            if bool((render_env_ids < 0).any()) or bool((render_env_ids >= self.num_envs).any()):
                raise ValueError(f"env_ids must be in [0, {self.num_envs})")

        if horizontal_tangents is None:
            horizontal_tangents = self._default_horizontal_tangents[:batch_size]
        else:
            horizontal_tangents = _batch_scalar_tensor(
                horizontal_tangents,
                batch_size,
                "horizontal_tangents",
                self._torch_device,
            )
        if vertical_tangents is None:
            vertical_tangents = self._default_vertical_tangents[:batch_size]
        else:
            vertical_tangents = _batch_scalar_tensor(
                vertical_tangents,
                batch_size,
                "vertical_tangents",
                self._torch_device,
            )
        if bool((horizontal_tangents <= 0.0).any()) or bool((vertical_tangents <= 0.0).any()):
            raise ValueError("Camera FOV tangents must be positive")

        mesh_positions = robot_positions[:, self._mesh_body_indexes].contiguous()
        mesh_rotations = _wxyz_to_xyzw(robot_quaternions[:, self._mesh_body_indexes])
        sphere_link_positions = robot_positions[:, self._sphere_body_indexes]
        sphere_link_quaternions = robot_quaternions[:, self._sphere_body_indexes]
        sphere_local_centers = self._sphere_local_centers.unsqueeze(0).expand(batch_size, -1, -1)
        sphere_centers = (
            sphere_link_positions
            + _torch_quat_apply_wxyz(sphere_link_quaternions, sphere_local_centers)
        ).contiguous()
        camera_rotations = _wxyz_to_xyzw(camera_quaternions)
        ladder_rotations = _wxyz_to_xyzw(ladder_quaternions)
        self._frame_tensors = (
            render_env_ids,
            horizontal_tangents,
            vertical_tangents,
            camera_positions,
            camera_rotations,
            mesh_positions,
            mesh_rotations,
            sphere_centers,
            ladder_positions,
            ladder_rotations,
        )

        launch_stream = (
            wp.stream_from_torch(self._torch_device) if self._torch_device.type == "cuda" else None
        )
        wp.launch(
            kernel=_render_cr1_ladder_batch_kernel,
            dim=batch_size * self.cfg.width * self.cfg.height,
            inputs=[
                self.cfg.width,
                self.cfg.height,
                wp.from_torch(horizontal_tangents, dtype=wp.float32),
                wp.from_torch(vertical_tangents, dtype=wp.float32),
                self.cfg.min_depth,
                self.cfg.max_depth,
                self.cfg.maximum_ray_distance,
                wp.from_torch(render_env_ids, dtype=wp.int32),
                wp.from_torch(camera_positions, dtype=wp.vec3),
                wp.from_torch(camera_rotations, dtype=wp.quat),
                wp.from_torch(mesh_positions, dtype=wp.vec3),
                wp.from_torch(mesh_rotations, dtype=wp.quat),
                self._mesh_ids_wp,
                self._robot_mesh_count if include_robot else 0,
                wp.from_torch(sphere_centers, dtype=wp.vec3),
                wp.from_torch(self._sphere_radii, dtype=wp.float32),
                self._robot_sphere_count if include_robot else 0,
                wp.from_torch(ladder_positions, dtype=wp.vec3),
                wp.from_torch(ladder_rotations, dtype=wp.quat),
                self._ladder_half_sizes_wp,
                self._ladder_classes_wp,
                self._ladder_box_count if include_ladder else 0,
                float(ground_height),
                int(include_ground),
            ],
            outputs=[
                wp.from_torch(self._depth, dtype=wp.float32),
                wp.from_torch(self._hit_class, dtype=wp.int32),
            ],
            device=self.device,
            stream=launch_stream,
        )

        full_batch = env_ids is None and batch_size == self.num_envs
        if full_batch:
            depth = self._depth
            hit_class = self._hit_class
        else:
            depth = self._depth.index_select(0, render_env_ids.to(dtype=torch.long))
            hit_class = self._hit_class.index_select(0, render_env_ids.to(dtype=torch.long))
        return WarpCameraOutput(depth=self.cfg.crop(depth), hit_class=self.cfg.crop(hit_class))


def _load_collision_mesh(
    geometry: RobotCollisionGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    if geometry.mesh_path is None or not geometry.mesh_path.is_file():
        raise FileNotFoundError(f"CR1 collision mesh not found: {geometry.mesh_path}")
    mesh = trimesh.load(geometry.mesh_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"CR1 collision mesh is empty: {geometry.mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * np.asarray(
        geometry.mesh_scale, dtype=np.float32
    )
    rotation = _quaternion_rotation_matrix(
        np.asarray(geometry.origin_quaternion_wxyz, dtype=np.float32)
    )
    vertices = (rotation @ vertices.T).T + np.asarray(
        geometry.origin_position, dtype=np.float32
    )
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"CR1 collision mesh has no triangles: {geometry.mesh_path}")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def _quaternion_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _single_pose_tensor(tensor: torch.Tensor, size: int, name: str) -> torch.Tensor:
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},) or (1, {size}), got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32).reshape(1, size).contiguous()


def _single_environment_tensor(tensor: torch.Tensor, size: int, name: str) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[-1] != size:
        raise ValueError(
            f"{name} must have shape (N, {size}) or (1, N, {size}), got {tuple(tensor.shape)}"
        )
    return tensor.to(dtype=torch.float32).contiguous()


def _batch_tensor(
    tensor: torch.Tensor,
    dimensions: int,
    last_dimension: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if tensor.ndim != dimensions or tensor.shape[-1] != last_dimension:
        expected = "x".join(["N"] * (dimensions - 1) + [str(last_dimension)])
        raise ValueError(f"{name} must have shape ({expected}), got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    return tensor.to(dtype=torch.float32).contiguous()


def _batch_scalar_tensor(
    tensor: torch.Tensor,
    batch_size: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if tensor.shape != (batch_size,):
        raise ValueError(f"{name} must have shape ({batch_size},), got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    return tensor.to(dtype=torch.float32).contiguous()


__all__ = [
    "BACKGROUND",
    "CR1WarpBatchCamera",
    "CR1WarpCamera",
    "GROUND",
    "HIT_CLASS_NAMES",
    "LADDER_OTHER",
    "ROBOT",
    "RUNG",
    "RobotCollisionGeometry",
    "WarpCameraCfg",
    "WarpCameraOutput",
    "WARP_CENTER_RAY_VALUE_NAMES",
    "WARP_DIAGNOSTIC_COUNT_NAMES",
    "aperture_from_fov",
    "load_robot_collision_geometries",
    "pinhole_fov_remap_grid",
]
