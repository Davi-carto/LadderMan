from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - IsaacLab runtime dependency.
    torch = None


LadderShape = Literal["cuboid", "cylinder"]


@dataclass
class LadderGeometry:
    """Procedural ladder parameters shared by scene generation and contact targets.

    This is mutable because IsaacLab's Hydra config loader recursively applies
    serialized config fields in place on newer IsaacLab releases.
    """

    phi: float = math.radians(65.5)
    rung_spacing: float = 0.248
    rung_length: float = 0.60
    rung_width: float = 0.08
    num_rungs: int = 12
    shape: LadderShape = "cuboid"
    origin: tuple[float, float, float] = (0.55, 0.0, 0.08)
    rail_width: float = 0.06
    rail_depth: float = 0.06
    rail_extra_length: float = 0.40
    lateral_contact_offset: float = 0.18

    @property
    def climb_direction(self) -> tuple[float, float, float]:
        return (math.cos(self.phi), 0.0, math.sin(self.phi))

    @property
    def plane_normal(self) -> tuple[float, float, float]:
        return (-math.sin(self.phi), 0.0, math.cos(self.phi))

    @property
    def rung_quat_wxyz(self) -> tuple[float, float, float, float]:
        # Rungs run along Y, so cuboid/cylinder local axes stay aligned with world axes.
        return (1.0, 0.0, 0.0, 0.0)

    @property
    def rail_quat_wxyz(self) -> tuple[float, float, float, float]:
        # Rotate local Z to the climbing direction in the X-Z plane.
        half_angle = 0.5 * (math.pi / 2.0 - self.phi)
        return (math.cos(half_angle), 0.0, math.sin(half_angle), 0.0)

    @property
    def rail_length(self) -> float:
        return max(0.0, (self.num_rungs - 1) * self.rung_spacing + self.rail_extra_length)


def generate_ladder(
    phi: float = math.radians(65.5),
    rung_spacing: float = 0.248,
    rung_length: float = 0.60,
    rung_radius_or_width: float = 0.08,
    num_rungs: int = 12,
    shape: LadderShape = "cuboid",
    origin: tuple[float, float, float] = (0.55, 0.0, 0.08),
) -> LadderGeometry:
    """Create a LadderGeometry object.

    `phi` may be passed in radians or degrees. Values larger than pi are treated as degrees.
    """

    phi_rad = math.radians(phi) if abs(phi) > math.pi else phi
    return LadderGeometry(
        phi=phi_rad,
        rung_spacing=rung_spacing,
        rung_length=rung_length,
        rung_width=rung_radius_or_width,
        num_rungs=num_rungs,
        shape=shape,
        origin=origin,
    )


LADDER_A = generate_ladder(phi=65.5, rung_spacing=0.248, rung_length=0.60, rung_radius_or_width=0.08)


def get_rung_pose(rung_id: int, ladder_geometry: LadderGeometry = LADDER_A) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return the rung center pose in the environment-local world frame."""

    if rung_id < 0 or rung_id >= ladder_geometry.num_rungs:
        raise IndexError(f"rung_id {rung_id} outside [0, {ladder_geometry.num_rungs})")
    direction = ladder_geometry.climb_direction
    origin = ladder_geometry.origin
    distance = rung_id * ladder_geometry.rung_spacing
    pos = (
        origin[0] + distance * direction[0],
        origin[1] + distance * direction[1],
        origin[2] + distance * direction[2],
    )
    return pos, ladder_geometry.rung_quat_wxyz


def rung_positions_tensor(
    ladder_geometry: LadderGeometry,
    *,
    device,
    dtype=None,
):
    """Return all rung centers as a `(num_rungs, 3)` tensor."""

    if torch is None:
        raise ModuleNotFoundError("torch is required for tensor ladder geometry helpers")
    dtype = torch.float32 if dtype is None else dtype
    positions = [get_rung_pose(i, ladder_geometry)[0] for i in range(ladder_geometry.num_rungs)]
    return torch.tensor(positions, device=device, dtype=dtype)


def limb_lateral_offsets_tensor(
    ladder_geometry: LadderGeometry,
    limb_names: tuple[str, ...] | list[str],
    *,
    device,
    dtype=None,
):
    """Return per-limb target offsets along the rung."""

    if torch is None:
        raise ModuleNotFoundError("torch is required for tensor ladder geometry helpers")
    dtype = torch.float32 if dtype is None else dtype
    offsets = []
    for limb_name in limb_names:
        if limb_name.startswith("left_"):
            offsets.append((0.0, ladder_geometry.lateral_contact_offset, 0.0))
        elif limb_name.startswith("right_"):
            offsets.append((0.0, -ladder_geometry.lateral_contact_offset, 0.0))
        else:
            offsets.append((0.0, 0.0, 0.0))
    return torch.tensor(offsets, device=device, dtype=dtype)
