import math
from pathlib import Path

import numpy as np

from ladder.utils.ladder_presets import HOLOSOMA_CR1_WOOD_STEP
from ladder.utils.wood_step_ladder import (
    WOOD_STEP_RUNG_BOX_NAMES,
    ladder_delta_quat_wxyz,
    load_wood_step_box_specs,
    randomized_wood_step_box_poses,
    thin_front_side_board_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LADDER_URDF = REPO_ROOT / "assets/ladders/ladder_A_aframe/wood_step_ladder_sim.urdf"


def _specs():
    preset = HOLOSOMA_CR1_WOOD_STEP
    return load_wood_step_box_specs(
        LADDER_URDF,
        asset_translation=preset.asset_translation,
        asset_yaw_deg=preset.asset_yaw_deg,
    )


def test_wood_step_parser_loads_exactly_18_unique_boxes():
    specs = _specs()

    assert len(specs) == 18
    assert len({spec.name for spec in specs}) == 18
    assert set(WOOD_STEP_RUNG_BOX_NAMES) <= {spec.name for spec in specs}


def test_zero_randomization_reproduces_nominal_box_poses():
    specs = _specs()
    preset = HOLOSOMA_CR1_WOOD_STEP
    phi = math.radians(preset.phi_deg)

    positions, quaternions = randomized_wood_step_box_poses(
        specs,
        source_phi=phi,
        source_spacing=preset.rung_spacing,
        target_phi=phi,
        target_spacing=preset.rung_spacing,
        origin=preset.target_origin,
    )

    np.testing.assert_allclose(positions, [spec.position for spec in specs], atol=1.0e-12)
    np.testing.assert_allclose(quaternions, [spec.quaternion for spec in specs], atol=1.0e-12)


def test_thin_front_side_boards_preserve_positive_depth_face():
    specs = _specs()
    thinned_specs = thin_front_side_board_specs(
        specs,
        remaining_depth_fraction=1.0 / 3.0,
    )
    thinned_by_name = {spec.name: spec for spec in thinned_specs}

    for original in specs:
        thinned = thinned_by_name[original.name]
        if not original.name.startswith("front_zigzag_side_board_"):
            assert thinned == original
            continue

        quaternion = np.asarray(original.quaternion)
        w, x, y, z = quaternion
        rotation = np.asarray(
            (
                (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
                (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
                (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
            )
        )
        positive_depth_axis = rotation[:, 1]
        original_face = (
            np.asarray(original.position)
            + 0.5 * original.size[1] * positive_depth_axis
        )
        thinned_face = (
            np.asarray(thinned.position)
            + 0.5 * thinned.size[1] * positive_depth_axis
        )

        assert thinned.size[0] == original.size[0]
        assert thinned.size[1] == original.size[1] / 3.0
        assert thinned.size[2] == original.size[2]
        assert thinned.position[1] == original.position[1]
        np.testing.assert_allclose(thinned_face, original_face, atol=1.0e-12)
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(thinned.position) - np.asarray(original.position)),
            original.size[1] / 3.0,
            atol=1.0e-12,
        )


def test_randomized_step_top_centers_match_sampled_rung_centers():
    specs = _specs()
    preset = HOLOSOMA_CR1_WOOD_STEP
    source_phi = math.radians(preset.phi_deg)
    target_phi = source_phi + math.radians(1.0)
    target_spacing = preset.rung_spacing + 0.01
    positions, _ = randomized_wood_step_box_poses(
        specs,
        source_phi=source_phi,
        source_spacing=preset.rung_spacing,
        target_phi=target_phi,
        target_spacing=target_spacing,
        origin=preset.target_origin,
    )
    spec_indexes = {spec.name: index for index, spec in enumerate(specs)}
    direction = np.asarray((math.cos(target_phi), 0.0, math.sin(target_phi)))

    for rung_index, name in enumerate(WOOD_STEP_RUNG_BOX_NAMES):
        box_index = spec_indexes[name]
        top_center = positions[box_index] + np.asarray((0.0, 0.0, specs[box_index].size[2] / 2.0))
        expected = np.asarray(preset.target_origin) + rung_index * target_spacing * direction
        np.testing.assert_allclose(top_center, expected, atol=1.0e-12)


def test_inclined_structural_boxes_use_the_sampled_ladder_rotation():
    preset = HOLOSOMA_CR1_WOOD_STEP
    source_phi = math.radians(preset.phi_deg)
    target_phi = source_phi - math.radians(1.0)
    specs = _specs()
    _, sampled_quaternions = randomized_wood_step_box_poses(
        specs,
        source_phi=source_phi,
        source_spacing=preset.rung_spacing,
        target_phi=target_phi,
        target_spacing=preset.rung_spacing - 0.01,
        origin=preset.target_origin,
    )
    delta_quat = ladder_delta_quat_wxyz(source_phi, target_phi)
    spec_indexes = {spec.name: index for index, spec in enumerate(specs)}
    for name in ("front_zigzag_side_board_left", "rear_support_leg_right"):
        index = spec_indexes[name]
        source_quat = np.asarray(specs[index].quaternion)
        expected = np.asarray(
            (
                delta_quat[0] * source_quat[0] - np.dot(delta_quat[1:], source_quat[1:]),
                delta_quat[0] * source_quat[1] + delta_quat[1] * source_quat[0]
                + delta_quat[2] * source_quat[3] - delta_quat[3] * source_quat[2],
                delta_quat[0] * source_quat[2] - delta_quat[1] * source_quat[3]
                + delta_quat[2] * source_quat[0] + delta_quat[3] * source_quat[1],
                delta_quat[0] * source_quat[3] + delta_quat[1] * source_quat[2]
                - delta_quat[2] * source_quat[1] + delta_quat[3] * source_quat[0],
            )
        )
        np.testing.assert_allclose(sampled_quaternions[index], expected, atol=1.0e-12)
