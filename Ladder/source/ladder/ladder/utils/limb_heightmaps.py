"""Batched gravity-aligned limb-centered height maps for CR1."""

from __future__ import annotations

from dataclasses import dataclass

import torch


NO_HIT = 0
RUNG = 1
RAIL = 2
LADDER_OTHER = 3
GROUND = 4
INVALID = 5

HEIGHTMAP_CLASS_NAMES = {
    NO_HIT: "no-hit",
    RUNG: "rung",
    RAIL: "rail",
    LADDER_OTHER: "ladder-other",
    GROUND: "ground",
    INVALID: "invalid",
}


def heightmap_cache_is_current(
    cache: dict | None, *, common_step_counter: int, output: str
) -> bool:
    """Return whether a shared manager-term result belongs to this control step."""

    return (
        cache is not None
        and cache.get("common_step_counter") == common_step_counter
        and output in cache
    )


@dataclass(frozen=True)
class LimbHeightMapCfg:
    """Regular horizontal grid dimensions in the world XY frame."""

    rows: int
    columns: int
    x_extent: float
    y_extent: float
    ray_start_height: float = 2.0
    max_ray_distance: float = 4.0

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("height-map rows and columns must be positive")
        if self.x_extent <= 0.0 or self.y_extent <= 0.0:
            raise ValueError("height-map extents must be positive")
        if self.ray_start_height <= 0.0 or self.max_ray_distance <= 0.0:
            raise ValueError("height-map ray distances must be positive")


@dataclass
class LimbHeightMap:
    """One batched map with tensors shaped ``[num_envs, rows, columns]``.

    ``height`` is retained as the legacy name for the raw world hit height.
    ``relative_height_m`` is the training label in meters; it is zero at
    no-hit samples and must be filtered with ``hit_valid``.
    """

    height: torch.Tensor
    raw_hit_z_w: torch.Tensor
    reference_z_w: torch.Tensor
    relative_height_m: torch.Tensor
    hit_class: torch.Tensor
    valid: torch.Tensor
    hit_valid: torch.Tensor
    points_w: torch.Tensor
    plane_points_w: torch.Tensor
    ray_origins_w: torch.Tensor
    center_xy: torch.Tensor
    x_axis_w: torch.Tensor
    y_axis_w: torch.Tensor
    x_offsets: torch.Tensor
    y_offsets: torch.Tensor


def relative_height_to_world_points(
    plane_points_w: torch.Tensor,
    relative_height_m: torch.Tensor,
) -> torch.Tensor:
    """Place a limb-relative height map on its world-XY sampling grid."""

    if plane_points_w.shape[-1] != 3 or plane_points_w.shape[:-1] != relative_height_m.shape:
        raise ValueError(
            "plane_points_w must have shape [..., rows, columns, 3] matching relative_height_m"
        )
    points_w = plane_points_w.clone()
    points_w[..., 2] = plane_points_w[..., 2] - relative_height_m
    return points_w


def sample_limb_heightmap(
    *,
    limb_centers_w: torch.Tensor,
    box_positions_w: torch.Tensor,
    box_quaternions_wxyz: torch.Tensor,
    box_sizes: torch.Tensor,
    box_classes: torch.Tensor,
    cfg: LimbHeightMapCfg,
    ground_height_w: torch.Tensor | None = None,
) -> LimbHeightMap:
    """Cast world-down rays from a limb-centered horizontal grid onto terrain.

    The limb supplies only the grid origin. Grid rows always follow world X,
    columns always follow world Y, and all rays follow world negative Z.
    ``height`` is the world-space z coordinate of a hit. A no-hit sample remains
    geometry-valid and stores zero only as a diagnostic placeholder. Its
    ``hit_valid`` flag is false, so its relative value is not a training label.
    """

    _validate_inputs(
        limb_centers_w,
        box_positions_w,
        box_quaternions_wxyz,
        box_sizes,
        box_classes,
        ground_height_w,
    )
    device = limb_centers_w.device
    dtype = limb_centers_w.dtype
    num_envs = limb_centers_w.shape[0]
    selected = (
        (box_classes == RUNG)
        | (box_classes == RAIL)
        | (box_classes == LADDER_OTHER)
    )
    if not torch.any(selected):
        raise ValueError("box_classes must contain at least one ladder surface")

    box_positions_w = box_positions_w[:, selected]
    box_quaternions_wxyz = box_quaternions_wxyz[:, selected]
    box_sizes = box_sizes[selected]
    box_classes = box_classes[selected]

    center_xy = limb_centers_w[:, :2]
    x_axis_w = limb_centers_w.new_tensor((1.0, 0.0, 0.0)).expand(num_envs, -1)
    y_axis_w = limb_centers_w.new_tensor((0.0, 1.0, 0.0)).expand(num_envs, -1)
    x_offsets = torch.linspace(
        -0.5 * cfg.x_extent,
        0.5 * cfg.x_extent,
        cfg.rows,
        device=device,
        dtype=dtype,
    )
    y_offsets = torch.linspace(
        -0.5 * cfg.y_extent,
        0.5 * cfg.y_extent,
        cfg.columns,
        device=device,
        dtype=dtype,
    )
    plane_points_w = (
        limb_centers_w[:, None, None, :]
        + x_offsets[None, :, None, None] * x_axis_w[:, None, None, :]
        + y_offsets[None, None, :, None] * y_axis_w[:, None, None, :]
    )
    ray_origins_w = plane_points_w.clone()
    ray_origins_w[..., 2] += cfg.ray_start_height

    num_points = cfg.rows * cfg.columns
    ray_origins_flat_w = ray_origins_w.reshape(num_envs, num_points, 1, 3)
    ray_directions_w = torch.zeros_like(ray_origins_flat_w)
    ray_directions_w[..., 2] = -1.0
    relative_w = ray_origins_flat_w - box_positions_w[:, None, :, :]

    quaternion_norm = torch.linalg.vector_norm(box_quaternions_wxyz, dim=-1)
    safe_quaternions = box_quaternions_wxyz / quaternion_norm.clamp_min(1.0e-12)[..., None]
    ray_origins_l = _quat_apply_inverse(
        safe_quaternions[:, None, :, :], relative_w
    )
    ray_directions_l = _quat_apply_inverse(
        safe_quaternions[:, None, :, :], ray_directions_w
    )

    half_sizes = 0.5 * box_sizes[None, None, :, :]
    parallel = torch.abs(ray_directions_l) < 1.0e-8
    parallel_inside = (~parallel) | (torch.abs(ray_origins_l) <= half_sizes + 1.0e-6)
    safe_directions = torch.where(
        parallel, torch.ones_like(ray_directions_l), ray_directions_l
    )
    first = (-half_sizes - ray_origins_l) / safe_directions
    second = (half_sizes - ray_origins_l) / safe_directions
    negative_infinity = torch.full_like(first, -torch.inf)
    positive_infinity = torch.full_like(first, torch.inf)
    axis_low = torch.where(parallel, negative_infinity, torch.minimum(first, second))
    axis_high = torch.where(parallel, positive_infinity, torch.maximum(first, second))
    ray_enter = torch.amax(axis_low, dim=-1)
    ray_exit = torch.amin(axis_high, dim=-1)
    ray_distance = torch.where(ray_enter >= 0.0, ray_enter, ray_exit)
    box_hit = (
        torch.all(parallel_inside, dim=-1)
        & (ray_enter <= ray_exit + 1.0e-6)
        & (ray_distance >= 0.0)
        & (ray_distance <= cfg.max_ray_distance)
    )
    nearest_distance, box_index = torch.min(
        torch.where(box_hit, ray_distance, torch.full_like(ray_distance, torch.inf)),
        dim=-1,
    )
    ray_origin_z = ray_origins_w[..., 2].reshape(num_envs, num_points)
    hit_class = box_classes[box_index]
    if ground_height_w is not None:
        ground_distance = ray_origin_z - ground_height_w[:, None]
        ground_hit = (ground_distance >= 0.0) & (
            ground_distance <= cfg.max_ray_distance
        )
        ground_is_nearest = ground_hit & (ground_distance < nearest_distance)
        nearest_distance = torch.where(
            ground_is_nearest, ground_distance, nearest_distance
        )
        hit_class = torch.where(
            ground_is_nearest, torch.full_like(hit_class, GROUND), hit_class
        )
    has_hit = torch.isfinite(nearest_distance)
    height = torch.where(has_hit, ray_origin_z - nearest_distance, torch.zeros_like(nearest_distance))
    hit_class = torch.where(
        has_hit, hit_class, torch.full_like(hit_class, NO_HIT)
    )

    environment_valid = (
        torch.all(torch.isfinite(limb_centers_w), dim=-1)
        & torch.all(torch.isfinite(box_positions_w), dim=(-1, -2))
        & torch.all(torch.isfinite(box_quaternions_wxyz), dim=(-1, -2))
        & torch.all(quaternion_norm > 1.0e-8, dim=-1)
    )
    shared_geometry_valid = torch.all(torch.isfinite(box_sizes)) & torch.all(box_sizes > 0.0)
    environment_valid = environment_valid & shared_geometry_valid
    if ground_height_w is not None:
        environment_valid = environment_valid & torch.isfinite(ground_height_w)
    valid = environment_valid[:, None].expand(-1, num_points)
    height = torch.where(valid, height, torch.zeros_like(height))
    hit_class = torch.where(
        valid, hit_class, torch.full_like(hit_class, INVALID)
    )
    hit_valid = valid & has_hit

    height = height.reshape(num_envs, cfg.rows, cfg.columns)
    hit_class = hit_class.reshape(num_envs, cfg.rows, cfg.columns)
    valid = valid.reshape(num_envs, cfg.rows, cfg.columns)
    hit_valid = hit_valid.reshape(num_envs, cfg.rows, cfg.columns)
    reference_z_w = plane_points_w[..., 2]
    relative_height_m = torch.where(
        hit_valid,
        reference_z_w - height,
        torch.zeros_like(height),
    )
    points_w = plane_points_w.clone()
    points_w[..., 2] = height
    return LimbHeightMap(
        height=height,
        raw_hit_z_w=height,
        reference_z_w=reference_z_w,
        relative_height_m=relative_height_m,
        hit_class=hit_class,
        valid=valid,
        hit_valid=hit_valid,
        points_w=points_w,
        plane_points_w=plane_points_w,
        ray_origins_w=ray_origins_w,
        center_xy=center_xy,
        x_axis_w=x_axis_w,
        y_axis_w=y_axis_w,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
    )


def _quat_apply_inverse(quaternion_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply inverse unit-quaternion rotation with broadcast dimensions."""

    vector_part = -quaternion_wxyz[..., 1:]
    scalar_part = quaternion_wxyz[..., :1]
    first_cross = torch.linalg.cross(vector_part, vector, dim=-1)
    return vector + 2.0 * (
        scalar_part * first_cross
        + torch.linalg.cross(vector_part, first_cross, dim=-1)
    )


def _validate_inputs(
    limb_centers_w: torch.Tensor,
    box_positions_w: torch.Tensor,
    box_quaternions_wxyz: torch.Tensor,
    box_sizes: torch.Tensor,
    box_classes: torch.Tensor,
    ground_height_w: torch.Tensor | None,
) -> None:
    if not limb_centers_w.is_floating_point():
        raise TypeError("height-map geometry tensors must use a floating dtype")
    num_envs = limb_centers_w.shape[0] if limb_centers_w.ndim == 2 else -1
    if limb_centers_w.shape != (num_envs, 3):
        raise ValueError(
            f"limb_centers_w must have shape [B, 3], got {tuple(limb_centers_w.shape)}"
        )
    if box_positions_w.ndim != 3 or box_positions_w.shape[:1] != (num_envs,):
        raise ValueError("box_positions_w must have shape [B, K, 3]")
    num_boxes = box_positions_w.shape[1]
    expected_shapes = {
        "box_positions_w": (num_envs, num_boxes, 3),
        "box_quaternions_wxyz": (num_envs, num_boxes, 4),
        "box_sizes": (num_boxes, 3),
        "box_classes": (num_boxes,),
    }
    values = {
        "box_positions_w": box_positions_w,
        "box_quaternions_wxyz": box_quaternions_wxyz,
        "box_sizes": box_sizes,
        "box_classes": box_classes,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(values[name].shape)}")
    if ground_height_w is not None and ground_height_w.shape != (num_envs,):
        raise ValueError(
            f"ground_height_w must have shape ({num_envs},), got {tuple(ground_height_w.shape)}"
        )
    floating_tensors = (
        box_positions_w,
        box_quaternions_wxyz,
        box_sizes,
    )
    device_tensors = (*floating_tensors, box_classes)
    if ground_height_w is not None:
        device_tensors = (*device_tensors, ground_height_w)
    if any(value.device != limb_centers_w.device for value in device_tensors):
        raise ValueError("all height-map tensors must be on the same device")
    if any(value.dtype != limb_centers_w.dtype for value in floating_tensors):
        raise ValueError("all floating height-map tensors must use the same dtype")
    if ground_height_w is not None:
        if not ground_height_w.is_floating_point():
            raise TypeError("ground_height_w must use a floating dtype")
        if ground_height_w.dtype != limb_centers_w.dtype:
            raise ValueError("ground_height_w must use the same floating dtype")
    if box_classes.dtype != torch.long:
        raise TypeError("box_classes must use torch.long")
    if torch.any((box_classes < NO_HIT) | (box_classes > LADDER_OTHER)):
        raise ValueError(
            "box_classes may contain only no-hit, rung, rail, or other-ladder IDs"
        )
