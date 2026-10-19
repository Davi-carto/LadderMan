import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ladder.utils.limb_heightmaps import (
    GROUND,
    INVALID,
    NO_HIT,
    RAIL,
    RUNG,
    LimbHeightMapCfg,
    relative_height_to_world_points,
    sample_limb_heightmap,
)
from ladder.utils.ladder_presets import HOLOSOMA_CR1_WOOD_STEP
from ladder.utils.wood_step_ladder import (
    WOOD_STEP_RUNG_BOX_NAMES,
    load_wood_step_box_specs,
    randomized_wood_step_box_poses,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LADDER_URDF = REPO_ROOT / "assets/ladders/ladder_A_aframe/wood_step_ladder_sim.urdf"
HEIGHTMAP_VIEWER = REPO_ROOT / "Ladder/scripts/view_cr1_limb_heightmaps.py"


def _sample(
    *,
    limb_centers=torch.tensor([[0.0, 0.0, 0.0]]),
    positions=torch.tensor([[[0.0, 0.0, 0.0]]]),
    quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
    sizes=torch.tensor([[0.20, 0.20, 0.20]]),
    classes=torch.tensor([RUNG], dtype=torch.long),
    cfg=LimbHeightMapCfg(rows=3, columns=3, x_extent=0.4, y_extent=0.4),
    ground_height=None,
):
    return sample_limb_heightmap(
        limb_centers_w=limb_centers,
        box_positions_w=positions,
        box_quaternions_wxyz=quaternions,
        box_sizes=sizes,
        box_classes=classes,
        cfg=cfg,
        ground_height_w=ground_height,
    )


def test_centered_rung_reports_front_surface_and_valid_gaps():
    result = _sample()

    assert result.height.shape == (1, 3, 3)
    assert result.hit_class[0, 1, 1].item() == RUNG
    assert result.height[0, 1, 1].item() == pytest.approx(0.10)
    assert result.hit_class[0, 0, 0].item() == NO_HIT
    assert result.height[0, 0, 0].item() == 0.0
    assert torch.all(result.valid)
    assert result.hit_valid[0, 1, 1].item()
    assert not result.hit_valid[0, 0, 0].item()
    assert result.relative_height_m[0, 1, 1].item() == pytest.approx(-0.10)
    assert result.relative_height_m[0, 0, 0].item() == 0.0
    torch.testing.assert_close(result.raw_hit_z_w, result.height)
    torch.testing.assert_close(result.points_w[0, 1, 1], torch.tensor([0.0, 0.0, 0.10]))


def test_relative_height_uses_limb_reference_height_not_limb_orientation():
    result = _sample(
        limb_centers=torch.tensor([[0.0, 0.0, 0.50]]),
        positions=torch.tensor([[[0.0, 0.0, 0.20]]]),
        quaternions=torch.tensor([[[0.7071068, 0.0, 0.7071068, 0.0]]]),
    )

    assert result.raw_hit_z_w[0, 1, 1].item() == pytest.approx(0.30, abs=1.0e-6)
    assert result.reference_z_w[0, 1, 1].item() == pytest.approx(0.50)
    assert result.relative_height_m[0, 1, 1].item() == pytest.approx(0.20, abs=1.0e-6)


def test_relative_height_reconstructs_world_hit_points():
    result = _sample(
        limb_centers=torch.tensor([[0.0, 0.0, 0.50]]),
        positions=torch.tensor([[[0.0, 0.0, 0.20]]]),
    )

    points_w = relative_height_to_world_points(
        result.plane_points_w, result.relative_height_m
    )

    torch.testing.assert_close(points_w[0, 1, 1], result.points_w[0, 1, 1])
    torch.testing.assert_close(points_w[..., :2], result.plane_points_w[..., :2])


def test_relative_height_world_point_shape_must_match_grid():
    with pytest.raises(ValueError):
        relative_height_to_world_points(torch.zeros(1, 2, 3), torch.zeros(1, 3))


def test_first_world_down_hit_wins_and_preserves_class():
    result = _sample(
        positions=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.30]]]),
        quaternions=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
        ),
        sizes=torch.tensor([[0.20, 0.20, 0.20], [0.20, 0.20, 0.20]]),
        classes=torch.tensor([RUNG, RAIL], dtype=torch.long),
    )

    assert result.hit_class[0, 1, 1].item() == RAIL
    assert result.height[0, 1, 1].item() == pytest.approx(0.40)


def test_ground_plane_fills_gaps_but_does_not_hide_nearer_ladder():
    result = _sample(
        positions=torch.tensor([[[0.0, 0.0, 0.20]]]),
        sizes=torch.tensor([[0.20, 0.20, 0.20]]),
        ground_height=torch.tensor([-0.03]),
    )

    assert result.hit_class[0, 1, 1].item() == RUNG
    assert result.height[0, 1, 1].item() == pytest.approx(0.30)
    assert result.hit_class[0, 0, 0].item() == GROUND
    assert result.height[0, 0, 0].item() == pytest.approx(-0.03)


def test_rotated_obb_intersection_uses_runtime_quaternion():
    half_angle = math.pi / 4.0
    result = _sample(
        positions=torch.tensor([[[0.0, 0.0, 0.0]]]),
        quaternions=torch.tensor(
            [[[math.cos(half_angle), 0.0, math.sin(half_angle), 0.0]]]
        ),
        sizes=torch.tensor([[0.20, 0.20, 0.40]]),
    )

    # After a 90-degree Y rotation, world Z sees the 0.20-m local X extent.
    assert result.height[0, 1, 1].item() == pytest.approx(0.10, abs=1.0e-6)


def test_grid_uses_world_axes_and_world_gravity_ray():
    result = _sample(
        positions=torch.tensor([[[0.0, 0.0, 0.0]]]),
        sizes=torch.tensor([[0.20, 0.20, 0.20]]),
    )

    torch.testing.assert_close(result.x_axis_w[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(result.y_axis_w[0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(result.ray_origins_w[0, 1, 1], torch.tensor([0.0, 0.0, 2.0]))


def test_each_environment_uses_its_own_limb_and_box_pose():
    result = _sample(
        limb_centers=torch.zeros(2, 3),
        positions=torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.30]]]),
        quaternions=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]
        ),
    )

    assert result.height[0, 1, 1].item() == pytest.approx(0.10)
    assert result.height[1, 1, 1].item() == pytest.approx(0.40)


def test_nonfinite_runtime_geometry_marks_map_invalid():
    result = _sample(positions=torch.tensor([[[float("nan"), 0.0, 0.0]]]))

    assert torch.all(~result.valid)
    assert torch.all(result.hit_class == INVALID)
    assert torch.count_nonzero(result.height) == 0


def test_grid_is_centered_on_limb_projection_in_world_coordinates():
    result = _sample(
        limb_centers=torch.tensor([[0.0, 0.30, 0.50]]),
        positions=torch.tensor([[[0.0, 0.30, 0.50]]]),
    )

    torch.testing.assert_close(result.center_xy, torch.tensor([[0.0, 0.30]]))
    torch.testing.assert_close(result.x_offsets, torch.tensor([-0.20, 0.0, 0.20]))
    torch.testing.assert_close(result.y_offsets, torch.tensor([-0.20, 0.0, 0.20]))


def test_configuration_rejects_nonpositive_grid_geometry():
    with pytest.raises(ValueError):
        LimbHeightMapCfg(rows=0, columns=3, x_extent=0.4, y_extent=0.4)


def test_diagnostic_viewport_does_not_follow_robot_asset():
    source = HEIGHTMAP_VIEWER.read_text(encoding="utf-8")

    assert 'env_cfg.viewer.origin_type = "world"' in source
    assert "env_cfg.viewer.asset_name = None" in source


def test_diagnostic_uses_requested_sparse_limb_windows():
    source = HEIGHTMAP_VIEWER.read_text(encoding="utf-8")

    assert source.count("LimbHeightMapCfg(10, 3, 0.315, 0.160)") == 4


def _real_wood_step_map(inclination_delta: float, spacing_delta: float):
    preset = HOLOSOMA_CR1_WOOD_STEP
    specs = load_wood_step_box_specs(
        LADDER_URDF,
        asset_translation=preset.asset_translation,
        asset_yaw_deg=preset.asset_yaw_deg,
    )
    source_phi = math.radians(preset.phi_deg)
    target_phi = source_phi + inclination_delta
    target_spacing = preset.rung_spacing + spacing_delta
    positions, quaternions = randomized_wood_step_box_poses(
        specs,
        source_phi=source_phi,
        source_spacing=preset.rung_spacing,
        target_phi=target_phi,
        target_spacing=target_spacing,
        origin=preset.target_origin,
    )
    result = sample_limb_heightmap(
        limb_centers_w=torch.tensor([[0.445, -0.50, 0.90]], dtype=torch.float32),
        box_positions_w=torch.from_numpy(positions.astype(np.float32)).unsqueeze(0),
        box_quaternions_wxyz=torch.from_numpy(quaternions.astype(np.float32)).unsqueeze(0),
        box_sizes=torch.tensor([spec.size for spec in specs], dtype=torch.float32),
        box_classes=torch.tensor(
            [
                RUNG
                if spec.name in WOOD_STEP_RUNG_BOX_NAMES
                else RAIL
                if spec.name.startswith("front_zigzag_side_board_")
                else NO_HIT
                for spec in specs
            ],
            dtype=torch.long,
        ),
        cfg=LimbHeightMapCfg(15, 9, 0.70, 0.64),
    )
    return result


@pytest.mark.parametrize(
    ("inclination_delta", "spacing_delta"),
    ((0.0, 0.0), (math.radians(1.0), 0.01)),
)
def test_real_wood_step_hand_grid_resolves_rung_hits(
    inclination_delta, spacing_delta
):
    result = _real_wood_step_map(inclination_delta, spacing_delta)

    assert torch.count_nonzero(result.hit_class == RUNG) > 0
    assert torch.all(result.valid)


def test_real_wood_step_map_labels_front_side_board_as_rail():
    result = _real_wood_step_map(0.0, 0.0)

    assert torch.count_nonzero(result.hit_class == RAIL) > 0
    assert torch.count_nonzero(result.hit_class == RUNG) > 0
