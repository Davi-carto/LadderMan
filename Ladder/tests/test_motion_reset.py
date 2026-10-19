import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ladder.utils.motion_reset import (
    load_reset_valid_mask,
    mix_fixed_frame_samples,
    mix_nominal_ladder_geometry_samples,
    mix_reset_frame_samples,
    mix_two_anchor_reset_frame_samples,
    pose_range_bounds,
    sample_reset_frames_by_segments,
    select_geometry_cell_reset_masks,
    valid_frame_sampling_weights,
)

class _NpzLike:
    def __init__(self, **arrays):
        self.files = list(arrays)
        self._arrays = arrays

    def __getitem__(self, name):
        return self._arrays[name]


def test_fixed_start_pose_range_only_overrides_selected_xy_bounds():
    pose_range = {
        "x": (-0.01, 0.01),
        "y": (-0.01, 0.01),
        "z": (0.0, 0.01),
        "roll": (-0.02, 0.02),
        "pitch": (-0.02, 0.02),
        "yaw": (-0.02, 0.02),
    }
    fixed_mask = torch.tensor([True, False, True, False])

    lower, upper, randomized = pose_range_bounds(
        pose_range,
        4,
        "cpu",
        fixed_start_mask=fixed_mask,
        fixed_start_pose_range={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
    )

    assert torch.all(lower[fixed_mask, :2] == -0.10)
    assert torch.all(upper[fixed_mask, :2] == 0.10)
    assert torch.equal(randomized, fixed_mask)
    assert torch.all(lower[~fixed_mask, :2] == -0.01)
    assert torch.all(upper[~fixed_mask, :2] == 0.01)
    expected_other_lower = torch.tensor([0.0, -0.02, -0.02, -0.02])
    expected_other_upper = torch.tensor([0.01, 0.02, 0.02, 0.02])
    assert torch.all(lower[:, 2:] == expected_other_lower)
    assert torch.all(upper[:, 2:] == expected_other_upper)


def test_fixed_start_pose_range_samples_cover_requested_xy_envelope():
    fixed_mask = torch.ones(100_000, dtype=torch.bool)
    lower, upper, randomized = pose_range_bounds(
        {"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
        len(fixed_mask),
        "cpu",
        fixed_start_mask=fixed_mask,
        fixed_start_pose_range={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
    )
    torch.manual_seed(7)
    samples = lower + (upper - lower) * torch.rand_like(lower)

    assert torch.all(randomized)
    assert torch.all(samples[:, :2].abs() <= 0.10)
    assert torch.all(samples[:, :2].amin(dim=0) < -0.099)
    assert torch.all(samples[:, :2].amax(dim=0) > 0.099)


def test_fixed_start_pose_randomization_uses_one_half_probability_mask_for_xy():
    fixed_mask = torch.ones(100_000, dtype=torch.bool)
    torch.manual_seed(7)

    lower, upper, randomized = pose_range_bounds(
        {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.01)},
        len(fixed_mask),
        "cpu",
        fixed_start_mask=fixed_mask,
        fixed_start_pose_range={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
        fixed_start_pose_randomization_probability=0.50,
    )

    assert abs(float(randomized.float().mean()) - 0.50) < 0.005
    assert torch.all(lower[randomized, :2] == -0.10)
    assert torch.all(upper[randomized, :2] == 0.10)
    assert torch.all(lower[~randomized, :2] == 0.0)
    assert torch.all(upper[~randomized, :2] == 0.0)
    assert torch.all(lower[:, 2] == 0.0)
    assert torch.all(upper[:, 2] == 0.01)


def test_total_reset_distribution_uses_ten_nominal_and_twenty_perturbed_frame_zero():
    torch.manual_seed(7)
    adaptive = torch.full((100_000,), 2, dtype=torch.long)
    uniform = torch.full((100_000,), 1, dtype=torch.long)
    _, fixed_mask, uniform_mask = mix_reset_frame_samples(
        adaptive,
        uniform,
        fixed_frame=0,
        fixed_probability=0.30,
        uniform_probability=0.35,
    )
    lower, upper, randomized = pose_range_bounds(
        {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.01),
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
            "yaw": (-0.02, 0.02),
        },
        len(fixed_mask),
        "cpu",
        fixed_start_mask=fixed_mask,
        fixed_start_pose_range={
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "z": (0.0, 0.01),
            "roll": (-0.02, 0.02),
            "pitch": (-0.20, 0.20),
            "yaw": (-0.20, 0.20),
        },
        fixed_start_pose_randomization_probability=2.0 / 3.0,
    )
    nominal = fixed_mask & ~randomized
    adaptive_mask = ~fixed_mask & ~uniform_mask

    assert abs(float(nominal.float().mean()) - 0.10) < 0.005
    assert abs(float(randomized.float().mean()) - 0.20) < 0.005
    assert abs(float(uniform_mask.float().mean()) - 0.35) < 0.005
    assert abs(float(adaptive_mask.float().mean()) - 0.35) < 0.005
    assert torch.all(lower[nominal] == 0.0)
    assert torch.all(upper[nominal] == 0.0)
    expected_lower = torch.tensor([-0.10, -0.10, 0.0, -0.02, -0.20, -0.20])
    expected_upper = torch.tensor([0.10, 0.10, 0.01, 0.02, 0.20, 0.20])
    assert torch.all(lower[randomized] == expected_lower)
    assert torch.all(upper[randomized] == expected_upper)


def test_missing_reset_mask_defaults_to_all_frames():
    mask, was_provided = load_reset_valid_mask(_NpzLike(), 4)

    assert not was_provided
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, True, True, True]


def test_reset_mask_rejects_wrong_shape_dtype_and_empty_mask():
    with pytest.raises(ValueError, match="shape"):
        load_reset_valid_mask(_NpzLike(reset_valid_mask=np.ones((3, 1), dtype=bool)), 3)
    with pytest.raises(TypeError, match="boolean dtype"):
        load_reset_valid_mask(_NpzLike(reset_valid_mask=np.ones(3, dtype=np.int8)), 3)
    with pytest.raises(ValueError, match="at least one valid frame"):
        load_reset_valid_mask(_NpzLike(reset_valid_mask=np.zeros(3, dtype=bool)), 3)


def test_valid_frame_weights_preserve_bin_probabilities():
    valid_frames = torch.tensor([0, 1, 6], dtype=torch.long)
    bin_probabilities = torch.tensor([0.8, 0.2])

    weights = valid_frame_sampling_weights(valid_frames, 10, bin_probabilities)

    assert torch.allclose(weights, torch.tensor([0.4, 0.4, 0.2]))


def test_sampling_never_returns_an_invalid_frame():
    valid_frames = torch.tensor([0, 3, 7], dtype=torch.long)
    weights = valid_frame_sampling_weights(valid_frames, 10, torch.tensor([0.5, 0.5]))
    torch.manual_seed(7)

    sampled = valid_frames[torch.multinomial(weights, 1000, replacement=True)]

    assert set(sampled.tolist()) <= {0, 3, 7}


def test_geometry_conditioned_masks_follow_each_sampled_grid_cell():
    cell_masks = torch.zeros((2, 2, 4), dtype=torch.bool)
    cell_masks[0, 0, 0] = True
    cell_masks[0, 1, 1] = True
    cell_masks[1, 0, 2] = True
    cell_masks[1, 1, 3] = True
    grid = torch.tensor([-1.0, 0.0, 1.0])

    selected = select_geometry_cell_reset_masks(
        cell_masks,
        grid,
        grid,
        torch.tensor([-0.5, -0.5, 0.5, 0.5]),
        torch.tensor([-0.5, 0.5, -0.5, 0.5]),
    )

    assert torch.equal(selected, torch.eye(4, dtype=torch.bool))


def test_mix_fixed_frame_samples_respects_probability_endpoints():
    sampled = torch.tensor([2, 3, 4, 5])

    unchanged, no_fixed = mix_fixed_frame_samples(sampled, fixed_frame=0, probability=0.0)
    fixed, all_fixed = mix_fixed_frame_samples(sampled, fixed_frame=0, probability=1.0)

    assert torch.equal(unchanged, sampled)
    assert not torch.any(no_fixed)
    assert torch.equal(fixed, torch.zeros_like(sampled))
    assert torch.all(all_fixed)


def test_mix_fixed_frame_samples_uses_requested_fraction():
    torch.manual_seed(7)
    sampled = torch.arange(100_000) + 1

    mixed, fixed_mask = mix_fixed_frame_samples(sampled, fixed_frame=0, probability=0.20)

    assert abs(float(fixed_mask.float().mean()) - 0.20) < 0.005
    assert torch.all(mixed[fixed_mask] == 0)
    assert torch.equal(mixed[~fixed_mask], sampled[~fixed_mask])


def test_mix_fixed_frame_samples_rejects_invalid_probability():
    with pytest.raises(ValueError, match="probability must be in"):
        mix_fixed_frame_samples(torch.tensor([1]), fixed_frame=0, probability=1.1)


def test_reset_frame_mixture_uses_requested_total_probabilities():
    torch.manual_seed(7)
    adaptive = torch.full((100_000,), 2, dtype=torch.long)
    uniform = torch.full((100_000,), 1, dtype=torch.long)

    mixed, fixed_mask, uniform_mask = mix_reset_frame_samples(
        adaptive,
        uniform,
        fixed_frame=0,
        fixed_probability=0.20,
        uniform_probability=0.40,
    )
    adaptive_mask = ~fixed_mask & ~uniform_mask

    assert abs(float(fixed_mask.float().mean()) - 0.20) < 0.005
    assert abs(float(uniform_mask.float().mean()) - 0.40) < 0.005
    assert abs(float(adaptive_mask.float().mean()) - 0.40) < 0.005
    assert torch.all(mixed[fixed_mask] == 0)
    assert torch.all(mixed[uniform_mask] == 1)
    assert torch.all(mixed[adaptive_mask] == 2)


def test_reset_frame_mixture_rejects_invalid_combined_probability():
    with pytest.raises(ValueError, match="must be at most 1"):
        mix_reset_frame_samples(
            torch.tensor([2]),
            torch.tensor([1]),
            fixed_frame=0,
            fixed_probability=0.7,
            uniform_probability=0.4,
        )


def test_two_anchor_reset_mixture_uses_requested_total_probabilities():
    torch.manual_seed(7)
    sample_count = 100_000
    adaptive = torch.full((sample_count,), 2, dtype=torch.long)
    uniform = torch.full((sample_count,), 1, dtype=torch.long)
    nominal_ladder = torch.zeros(sample_count, dtype=torch.bool)
    nominal_ladder[:30_000] = True

    mixed, primary, secondary, uniform_mask = mix_two_anchor_reset_frame_samples(
        adaptive,
        uniform,
        primary_frame=0,
        primary_probability=0.20,
        secondary_frame=1500,
        secondary_probability=0.10,
        uniform_probability=0.35,
        secondary_eligible_mask=nominal_ladder,
        secondary_eligible_probability=0.30,
    )
    adaptive_mask = ~primary & ~secondary & ~uniform_mask

    assert abs(float(primary.float().mean()) - 0.20) < 0.005
    assert abs(float(secondary.float().mean()) - 0.10) < 0.005
    assert abs(float(uniform_mask.float().mean()) - 0.35) < 0.005
    assert abs(float(adaptive_mask.float().mean()) - 0.35) < 0.005
    assert torch.all(secondary <= nominal_ladder)
    assert torch.all(mixed[primary] == 0)
    assert torch.all(mixed[secondary] == 1500)
    assert torch.all(mixed[uniform_mask] == 1)
    assert torch.all(mixed[adaptive_mask] == 2)


def test_two_anchor_reset_mixture_rejects_ineligible_probability():
    with pytest.raises(ValueError, match="cannot exceed"):
        mix_two_anchor_reset_frame_samples(
            torch.tensor([2]),
            torch.tensor([1]),
            primary_frame=0,
            primary_probability=0.20,
            secondary_frame=1500,
            secondary_probability=0.40,
            uniform_probability=0.20,
            secondary_eligible_mask=torch.tensor([True]),
            secondary_eligible_probability=0.30,
        )


def test_nominal_ladder_geometry_mixture_uses_one_joint_mask():
    torch.manual_seed(7)
    inclination = torch.full((100_000,), 1.0)
    spacing = torch.full((100_000,), 2.0)

    mixed_inclination, mixed_spacing, nominal_mask = (
        mix_nominal_ladder_geometry_samples(inclination, spacing, probability=0.30)
    )

    assert abs(float(nominal_mask.float().mean()) - 0.30) < 0.005
    assert torch.all(mixed_inclination[nominal_mask] == 0.0)
    assert torch.all(mixed_spacing[nominal_mask] == 0.0)
    assert torch.all(mixed_inclination[~nominal_mask] == 1.0)
    assert torch.all(mixed_spacing[~nominal_mask] == 2.0)


def test_nominal_ladder_geometry_mixture_respects_probability_endpoints():
    inclination = torch.tensor([1.0, -1.0])
    spacing = torch.tensor([2.0, -2.0])

    unchanged_phi, unchanged_spacing, no_nominal = mix_nominal_ladder_geometry_samples(
        inclination, spacing, probability=0.0
    )
    nominal_phi, nominal_spacing, all_nominal = mix_nominal_ladder_geometry_samples(
        inclination, spacing, probability=1.0
    )

    assert torch.equal(unchanged_phi, inclination)
    assert torch.equal(unchanged_spacing, spacing)
    assert not torch.any(no_nominal)
    assert torch.equal(nominal_phi, torch.zeros_like(inclination))
    assert torch.equal(nominal_spacing, torch.zeros_like(spacing))
    assert torch.all(all_nominal)


def test_nominal_ladder_geometry_mixture_reports_already_nominal_samples():
    zeros = torch.zeros(4)

    mixed_phi, mixed_spacing, nominal_mask = mix_nominal_ladder_geometry_samples(
        zeros, zeros, probability=0.0
    )

    assert torch.equal(mixed_phi, zeros)
    assert torch.equal(mixed_spacing, zeros)
    assert torch.all(nominal_mask)


def test_reset_samples_stable_segment_before_uniform_valid_frame():
    segment_ids = torch.tensor([0, 0, 1, 1, 1, 1])
    segment_bounds = ((0, 1), (2, 5))
    valid = torch.tensor(
        [
            [True, True, True, True, True, True],
            [False, False, True, False, True, True],
        ]
    )
    source_weights = torch.tensor(
        [
            [1.0, 1.0, 4.0, 4.0, 4.0, 4.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        ]
    )
    sampled, probabilities = sample_reset_frames_by_segments(
        valid,
        source_weights,
        segment_ids,
        segment_bounds,
        equal_segments=False,
    )
    assert torch.all(valid[torch.arange(2), sampled])
    torch.testing.assert_close(
        probabilities[0],
        torch.tensor([0.1, 0.1, 0.2, 0.2, 0.2, 0.2]),
    )
    torch.testing.assert_close(
        probabilities[1],
        torch.tensor([0.0, 0.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0]),
    )

    _, uniform_probabilities = sample_reset_frames_by_segments(
        valid,
        source_weights,
        segment_ids,
        segment_bounds,
        equal_segments=True,
    )
    torch.testing.assert_close(
        uniform_probabilities[0],
        torch.tensor([0.25, 0.25, 0.125, 0.125, 0.125, 0.125]),
    )
