import math

import pytest

from ladder.tasks.ladder_climb.ladder_geometry import LADDER_A, generate_ladder, get_rung_pose, limb_lateral_offsets_tensor
from ladder.utils.ladder_presets import HOLOSOMA_CR1_WOOD_STEP


def test_ladder_a_parameters_are_in_meters_and_radians():
    assert math.isclose(LADDER_A.phi, math.radians(65.5))
    assert math.isclose(LADDER_A.rung_spacing, 0.248)
    assert math.isclose(LADDER_A.rung_length, 0.60)
    assert math.isclose(LADDER_A.rung_width, 0.08)


def test_get_rung_pose_advances_along_ladder_direction():
    geom = generate_ladder(phi=60.0, rung_spacing=0.2, num_rungs=3)
    p0, _ = get_rung_pose(0, geom)
    p1, _ = get_rung_pose(1, geom)
    assert math.isclose(p1[0] - p0[0], 0.2 * math.cos(math.radians(60.0)))
    assert math.isclose(p1[2] - p0[2], 0.2 * math.sin(math.radians(60.0)))


def test_holosoma_cr1_preset_matches_wood_step_top_centers():
    preset = HOLOSOMA_CR1_WOOD_STEP
    geom = generate_ladder(
        phi=preset.phi_deg,
        rung_spacing=preset.rung_spacing,
        rung_length=preset.rung_length,
        rung_radius_or_width=preset.rung_width,
        num_rungs=preset.num_rungs,
        origin=preset.target_origin,
    )
    first, _ = get_rung_pose(0, geom)
    second, _ = get_rung_pose(1, geom)

    assert math.isclose(first[0], 0.445)
    assert math.isclose(first[1], 0.04)
    assert math.isclose(first[2], 0.19)
    assert math.isclose(second[0] - first[0], 0.135)
    assert math.isclose(second[2] - first[2], 0.23)
    assert math.isclose(preset.asset_translation[0], 0.38)
    assert math.isclose(preset.asset_translation[1], 0.04)
    assert math.isclose(preset.asset_translation[2], 0.0)
    assert math.isclose(0.0 + preset.asset_translation[2], 0.0, abs_tol=1.0e-9)
    assert preset.asset_yaw_deg == 90.0


def test_get_rung_pose_rejects_invalid_id():
    with pytest.raises(IndexError):
        get_rung_pose(LADDER_A.num_rungs, LADDER_A)


def test_limb_offsets_are_left_right_symmetric():
    pytest.importorskip("torch")
    import torch

    offsets = limb_lateral_offsets_tensor(LADDER_A, ("left_hand", "right_hand", "left_foot", "right_foot"), device="cpu")
    expected = LADDER_A.lateral_contact_offset
    assert torch.allclose(offsets[:, 0], torch.zeros(4))
    assert torch.allclose(offsets[[0, 2], 1], torch.full((2,), expected))
    assert torch.allclose(offsets[[1, 3], 1], torch.full((2,), -expected))
