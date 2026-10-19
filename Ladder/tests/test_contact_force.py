import pytest


torch = pytest.importorskip("torch")

from ladder.utils.contact_force import (
    filtered_surface_force_max,
    non_target_rung_forces,
    physical_support_rung_id,
    target_rung_forces,
)


def test_target_rung_forces_gathers_per_environment_and_masks_invalid_ids():
    forces = torch.zeros((3, 1, 4, 3))
    forces[0, 0, 2, 2] = 12.0
    forces[0, 0, 3, 2] = 99.0
    forces[1, 0, 1, 2] = 7.0
    forces[2, 0, 0, 2] = 20.0

    selected = target_rung_forces(forces, torch.tensor([[2], [1], [-1]]))

    assert torch.allclose(selected[:, 0, 2], torch.tensor([12.0, 7.0, 0.0]))


def test_physical_support_id_uses_strongest_force_and_explicit_filter_mapping():
    rung = torch.zeros((3, 2, 3, 3))
    ground = torch.zeros((3, 1, 2, 3))
    # Filter order represents rung IDs [3, 1, 2], not ascending columns.
    rung[0, 1, 0, 2] = 7.0
    rung[0, 0, 1, 2] = 9.0
    ground[0, 0, 0, 2] = 8.0
    ground[1, 0, 1, 2] = 6.0
    rung[2, 0, 2, 2] = 4.9

    support = physical_support_rung_id(
        rung, ground, torch.tensor([3, 1, 2]), force_threshold=5.0
    )

    assert support.tolist() == [1, 0, -1]


def test_non_target_rung_forces_excludes_valid_target_and_keeps_all_without_target():
    forces = torch.ones((2, 1, 3, 3))

    selected = non_target_rung_forces(forces, torch.tensor([[1], [-1]]))

    assert torch.all(selected[0, 0, 1] == 0.0)
    assert torch.all(selected[0, 0, (0, 2)] == 1.0)
    assert torch.all(selected[1] == 1.0)


def test_filtered_surface_force_max_splits_top_front_and_underside():
    forces = torch.tensor(
        [[[[0.0, 0.0, 30.0], [40.0, 0.0, 0.0], [0.0, 0.0, -50.0]]]]
    )

    assert torch.allclose(
        filtered_surface_force_max(forces, surface="top"), torch.tensor([30.0])
    )
    assert torch.allclose(
        filtered_surface_force_max(forces, surface="front"), torch.tensor([40.0])
    )
    assert torch.allclose(
        filtered_surface_force_max(forces, surface="underside"), torch.tensor([50.0])
    )


def test_filtered_surface_force_max_uses_requested_front_normal_fraction():
    forces = torch.tensor([[[[0.9165151, 0.0, 0.4]]]])

    assert torch.allclose(
        filtered_surface_force_max(
            forces, surface="front", min_top_normal_fraction=0.5
        ),
        torch.tensor([1.0]),
    )
    assert torch.allclose(
        filtered_surface_force_max(
            forces, surface="front", min_top_normal_fraction=0.35
        ),
        torch.tensor([0.0]),
    )
