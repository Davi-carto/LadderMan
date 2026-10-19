from __future__ import annotations

import numpy as np
import torch


POSE_RANGE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def pose_range_bounds(
    pose_range: dict[str, tuple[float, float]],
    num_samples: int,
    device: str | torch.device,
    *,
    fixed_start_mask: torch.Tensor | None = None,
    fixed_start_pose_range: dict[str, tuple[float, float]] | None = None,
    fixed_start_pose_randomization_probability: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-reset pose-noise bounds with optional fixed-start overrides."""

    unknown_keys = set(pose_range) - set(POSE_RANGE_KEYS)
    if fixed_start_pose_range is not None:
        unknown_keys |= set(fixed_start_pose_range) - set(POSE_RANGE_KEYS)
    if unknown_keys:
        raise ValueError(f"unknown pose range axes: {sorted(unknown_keys)}")

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in POSE_RANGE_KEYS]
    ranges = torch.as_tensor(range_list, dtype=torch.float32, device=device)
    if ranges.shape != (len(POSE_RANGE_KEYS), 2):
        raise ValueError("each pose range must contain exactly two bounds")
    if torch.any(ranges[:, 0] > ranges[:, 1]):
        raise ValueError("pose range lower bounds must not exceed upper bounds")

    lower = ranges[:, 0].expand(num_samples, -1).clone()
    upper = ranges[:, 1].expand(num_samples, -1).clone()
    fixed_start_randomized_mask = torch.zeros(
        num_samples, dtype=torch.bool, device=device
    )
    if not fixed_start_pose_range:
        return lower, upper, fixed_start_randomized_mask
    if fixed_start_mask is None or fixed_start_mask.shape != (num_samples,):
        raise ValueError(f"fixed_start_mask must have shape ({num_samples},)")
    if fixed_start_mask.device != lower.device:
        raise ValueError("fixed_start_mask must be on the requested device")
    if not 0.0 <= fixed_start_pose_randomization_probability <= 1.0:
        raise ValueError(
            "fixed_start_pose_randomization_probability must be in [0, 1], got "
            f"{fixed_start_pose_randomization_probability}"
        )
    if fixed_start_pose_randomization_probability == 1.0:
        fixed_start_randomized_mask = fixed_start_mask.clone()
    elif fixed_start_pose_randomization_probability > 0.0:
        fixed_start_randomized_mask = fixed_start_mask & (
            torch.rand(num_samples, device=device)
            < fixed_start_pose_randomization_probability
        )

    for key, bounds in fixed_start_pose_range.items():
        if len(bounds) != 2:
            raise ValueError(f"fixed-start pose range for {key} must contain two bounds")
        low, high = bounds
        if low > high:
            raise ValueError(
                f"fixed-start pose range lower bound for {key} exceeds its upper bound"
            )
        axis = POSE_RANGE_KEYS.index(key)
        lower[fixed_start_mask, axis] = 0.0
        upper[fixed_start_mask, axis] = 0.0
        lower[fixed_start_randomized_mask, axis] = low
        upper[fixed_start_randomized_mask, axis] = high
    return lower, upper, fixed_start_randomized_mask


def load_reset_valid_mask(data, num_frames: int) -> tuple[np.ndarray, bool]:
    """Load an optional per-frame reset mask, defaulting to all frames."""

    if "reset_valid_mask" not in data.files:
        return np.ones(num_frames, dtype=np.bool_), False

    mask = np.asarray(data["reset_valid_mask"])
    if mask.shape != (num_frames,):
        raise ValueError(
            f"reset_valid_mask must have shape ({num_frames},), got {mask.shape}"
        )
    if mask.dtype != np.bool_:
        raise TypeError(f"reset_valid_mask must have boolean dtype, got {mask.dtype}")
    if not np.any(mask):
        raise ValueError("reset_valid_mask must contain at least one valid frame")
    return mask.copy(), True


def valid_frame_sampling_weights(
    valid_frame_indexes: torch.Tensor,
    num_frames: int,
    bin_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Map adaptive bin probabilities onto valid frames without biasing dense bins."""

    num_bins = bin_probabilities.shape[0]
    frame_bins = torch.div(
        valid_frame_indexes * num_bins,
        num_frames,
        rounding_mode="floor",
    ).clamp(max=num_bins - 1)
    valid_count_per_bin = torch.bincount(frame_bins, minlength=num_bins)
    weights = bin_probabilities[frame_bins] / valid_count_per_bin[frame_bins]
    return weights / weights.sum()


def sample_reset_frames_by_segments(
    valid_mask: torch.Tensor,
    frame_weights: torch.Tensor,
    segment_ids: torch.Tensor,
    segment_bounds: tuple[tuple[int, int], ...],
    *,
    equal_segments: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a reset segment first, then a valid frame inside that segment.

    Adaptive segment scores use the mean frame density in each segment, so a
    long segment is not favored merely because it contains more frames.
    Selected segments sample their geometry-valid frames uniformly.  The
    second return value is the exact marginal frame distribution.
    """

    if valid_mask.ndim != 2 or frame_weights.shape != valid_mask.shape:
        raise ValueError("valid masks and frame weights must share shape (env, frame)")
    if segment_ids.shape != (valid_mask.shape[1],):
        raise ValueError("segment_ids must contain one ID per frame")
    if not segment_bounds:
        raise ValueError("segment_bounds must not be empty")

    valid_frame_counts = torch.stack(
        [
            valid_mask[:, start : end + 1].sum(dim=-1)
            for start, end in segment_bounds
        ],
        dim=-1,
    )
    segment_valid = valid_frame_counts > 0
    if torch.any(~segment_valid.any(dim=-1)):
        raise ValueError("sampled geometry has no reset-valid stable-support segment")

    segment_weights = torch.stack(
        [
            frame_weights[:, start : end + 1].sum(dim=-1)
            for start, end in segment_bounds
        ],
        dim=-1,
    )
    if equal_segments:
        segment_weights = segment_valid.float()
    else:
        segment_weights = torch.where(
            segment_valid,
            segment_weights / valid_frame_counts.clamp_min(1),
            torch.zeros_like(segment_weights),
        )
    if torch.any(segment_weights.sum(dim=-1) <= 0.0):
        raise ValueError("stable-support segment sampling has no positive weight")
    segment_weights = segment_weights / segment_weights.sum(dim=-1, keepdim=True)
    selected_segments = torch.multinomial(
        segment_weights, 1, replacement=True
    ).squeeze(-1)

    selected_frame_weights = (
        (segment_ids.unsqueeze(0) == selected_segments.unsqueeze(-1)) & valid_mask
    ).float()
    if torch.any(selected_frame_weights.sum(dim=-1) <= 0.0):
        raise RuntimeError("selected stable-support segment has no valid reset frame")
    selected_frame_weights = selected_frame_weights / selected_frame_weights.sum(
        dim=-1, keepdim=True
    )
    sampled_frames = torch.multinomial(
        selected_frame_weights, 1, replacement=True
    ).squeeze(-1)

    frame_segment_indexes = segment_ids.unsqueeze(0).expand(valid_mask.shape[0], -1)
    marginal_frame_probabilities = torch.gather(
        segment_weights, 1, frame_segment_indexes
    )
    marginal_frame_probabilities = torch.where(
        valid_mask,
        marginal_frame_probabilities
        / torch.gather(
            valid_frame_counts, 1, frame_segment_indexes
        ).clamp_min(1),
        0.0,
    )
    return sampled_frames, marginal_frame_probabilities


def select_geometry_cell_reset_masks(
    cell_masks: torch.Tensor,
    inclination_grid: torch.Tensor,
    spacing_grid: torch.Tensor,
    inclination_deltas: torch.Tensor,
    spacing_deltas: torch.Tensor,
) -> torch.Tensor:
    """Select the conservative reset mask for each sampled ladder geometry."""

    inclination_cells = torch.bucketize(
        inclination_deltas.contiguous(), inclination_grid[1:-1]
    )
    spacing_cells = torch.bucketize(spacing_deltas.contiguous(), spacing_grid[1:-1])
    return cell_masks[inclination_cells, spacing_cells]


def mix_fixed_frame_samples(
    sampled_frames: torch.Tensor,
    fixed_frame: int,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a fraction of sampled frames with one designated start frame."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if probability == 0.0:
        fixed_mask = torch.zeros_like(sampled_frames, dtype=torch.bool)
    elif probability == 1.0:
        fixed_mask = torch.ones_like(sampled_frames, dtype=torch.bool)
    else:
        fixed_mask = torch.rand(sampled_frames.shape, device=sampled_frames.device) < probability
    mixed_frames = torch.where(fixed_mask, fixed_frame, sampled_frames)
    return mixed_frames, fixed_mask


def mix_reset_frame_samples(
    adaptive_frames: torch.Tensor,
    uniform_frames: torch.Tensor,
    fixed_frame: int,
    fixed_probability: float,
    uniform_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route resets between fixed, uniform-valid, and adaptive frame samples."""

    if adaptive_frames.shape != uniform_frames.shape:
        raise ValueError(
            f"adaptive and uniform samples must have the same shape, got "
            f"{adaptive_frames.shape} and {uniform_frames.shape}"
        )
    if adaptive_frames.device != uniform_frames.device:
        raise ValueError("adaptive and uniform samples must be on the same device")
    if fixed_probability < 0.0 or uniform_probability < 0.0:
        raise ValueError("reset mixture probabilities must be non-negative")
    if fixed_probability + uniform_probability > 1.0:
        raise ValueError(
            "fixed_probability + uniform_probability must be at most 1, got "
            f"{fixed_probability + uniform_probability}"
        )

    route = torch.rand(adaptive_frames.shape, device=adaptive_frames.device)
    fixed_mask = route < fixed_probability
    uniform_mask = (route >= fixed_probability) & (
        route < fixed_probability + uniform_probability
    )
    mixed_frames = torch.where(
        fixed_mask,
        fixed_frame,
        torch.where(uniform_mask, uniform_frames, adaptive_frames),
    )
    return mixed_frames, fixed_mask, uniform_mask


def mix_two_anchor_reset_frame_samples(
    adaptive_frames: torch.Tensor,
    uniform_frames: torch.Tensor,
    primary_frame: int,
    primary_probability: float,
    secondary_frame: int,
    secondary_probability: float,
    uniform_probability: float,
    secondary_eligible_mask: torch.Tensor,
    secondary_eligible_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route resets between two anchors, uniform-valid, and adaptive samples.

    The secondary anchor can be restricted to an eligible subset, such as
    nominal-ladder environments.  Probabilities are unconditional totals; the
    secondary conditional probability is adjusted for the eligible fraction.
    """

    if adaptive_frames.shape != uniform_frames.shape:
        raise ValueError(
            f"adaptive and uniform samples must have the same shape, got "
            f"{adaptive_frames.shape} and {uniform_frames.shape}"
        )
    if secondary_eligible_mask.shape != adaptive_frames.shape:
        raise ValueError(
            "secondary_eligible_mask must match sampled frame shape, got "
            f"{secondary_eligible_mask.shape} and {adaptive_frames.shape}"
        )
    if (
        adaptive_frames.device != uniform_frames.device
        or adaptive_frames.device != secondary_eligible_mask.device
    ):
        raise ValueError("all reset sampling tensors must be on the same device")
    if min(primary_probability, secondary_probability, uniform_probability) < 0.0:
        raise ValueError("reset mixture probabilities must be non-negative")
    total_probability = (
        primary_probability + secondary_probability + uniform_probability
    )
    if total_probability > 1.0:
        raise ValueError(
            "primary_probability + secondary_probability + uniform_probability "
            f"must be at most 1, got {total_probability}"
        )
    if not 0.0 < secondary_eligible_probability <= 1.0:
        raise ValueError(
            "secondary_eligible_probability must be in (0, 1], got "
            f"{secondary_eligible_probability}"
        )
    if secondary_probability > secondary_eligible_probability:
        raise ValueError(
            "secondary_probability cannot exceed secondary_eligible_probability, got "
            f"{secondary_probability} and {secondary_eligible_probability}"
        )

    secondary_conditional_probability = (
        secondary_probability / secondary_eligible_probability
    )
    secondary_mask = secondary_eligible_mask & (
        torch.rand(adaptive_frames.shape, device=adaptive_frames.device)
        < secondary_conditional_probability
    )

    remaining_probability = 1.0 - secondary_probability
    if remaining_probability == 0.0:
        return (
            torch.full_like(adaptive_frames, secondary_frame),
            torch.zeros_like(secondary_mask),
            secondary_mask,
            torch.zeros_like(secondary_mask),
        )
    route = torch.rand(adaptive_frames.shape, device=adaptive_frames.device)
    primary_threshold = primary_probability / remaining_probability
    uniform_threshold = (
        primary_probability + uniform_probability
    ) / remaining_probability
    primary_mask = ~secondary_mask & (route < primary_threshold)
    uniform_mask = (
        ~secondary_mask
        & (route >= primary_threshold)
        & (route < uniform_threshold)
    )
    mixed_frames = torch.where(
        secondary_mask,
        secondary_frame,
        torch.where(
            primary_mask,
            primary_frame,
            torch.where(uniform_mask, uniform_frames, adaptive_frames),
        ),
    )
    return mixed_frames, primary_mask, secondary_mask, uniform_mask


def mix_nominal_ladder_geometry_samples(
    inclination_deltas: torch.Tensor,
    spacing_deltas: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace a fraction of paired ladder perturbations with nominal geometry."""

    if inclination_deltas.shape != spacing_deltas.shape:
        raise ValueError(
            "inclination and spacing deltas must have the same shape, got "
            f"{inclination_deltas.shape} and {spacing_deltas.shape}"
        )
    if inclination_deltas.device != spacing_deltas.device:
        raise ValueError("inclination and spacing deltas must be on the same device")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if probability == 0.0:
        nominal_mask = torch.zeros_like(inclination_deltas, dtype=torch.bool)
    elif probability == 1.0:
        nominal_mask = torch.ones_like(inclination_deltas, dtype=torch.bool)
    else:
        nominal_mask = (
            torch.rand(inclination_deltas.shape, device=inclination_deltas.device)
            < probability
        )
    mixed_inclination = torch.where(nominal_mask, 0.0, inclination_deltas)
    mixed_spacing = torch.where(nominal_mask, 0.0, spacing_deltas)
    actual_nominal_mask = nominal_mask | (
        (mixed_inclination == 0.0) & (mixed_spacing == 0.0)
    )
    return mixed_inclination, mixed_spacing, actual_nominal_mask
