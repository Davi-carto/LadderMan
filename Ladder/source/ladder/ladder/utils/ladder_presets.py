"""Shared ladder target geometry and display poses for motion tools."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LadderToolPreset:
    name: str
    phi_deg: float
    rung_spacing: float
    rung_length: float
    rung_width: float
    num_rungs: int
    target_origin: tuple[float, float, float]
    asset_translation: tuple[float, float, float] | None = None
    asset_yaw_deg: float | None = None


LADDER_A_CONTACT_PRESET = LadderToolPreset(
    name="ladder_a",
    phi_deg=65.5,
    rung_spacing=0.248,
    rung_length=0.60,
    rung_width=0.08,
    num_rungs=12,
    target_origin=(0.55, 0.0, 0.08),
)

# The base_platform cube spans local z=[0, 0.16], so asset z=0 places the
# requested large rectangular base exactly on the ground plane.
_WOOD_STEP_BASE_PLATFORM_GROUNDED_Z = 0.0

# Targets are on the seven step top surfaces. X/Y/yaw preserve the Holosoma
# alignment; the ladder and all targets share the same vertical adjustment.
HOLOSOMA_CR1_WOOD_STEP = LadderToolPreset(
    name="holosoma_cr1_wood_step",
    phi_deg=math.degrees(math.atan2(0.23, 0.135)),
    rung_spacing=math.hypot(0.23, 0.135),
    rung_length=0.64,
    rung_width=0.13,
    num_rungs=7,
    target_origin=(0.445, 0.04, 0.19 + _WOOD_STEP_BASE_PLATFORM_GROUNDED_Z),
    asset_translation=(0.38, 0.04, _WOOD_STEP_BASE_PLATFORM_GROUNDED_Z),
    asset_yaw_deg=90.0,
)
