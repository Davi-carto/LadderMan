"""Small tensor utilities shared by LadderMan stop-command planners."""

from __future__ import annotations

import torch


STOP_STATE_RUN = 0
STOP_STATE_SEEK = 1
STOP_STATE_HOLD = 2


def next_zero_command_frame(
    zero_command_valid_mask: torch.Tensor, time_steps: torch.Tensor
) -> torch.Tensor:
    """Return the first legal stop frame at or after each current frame."""

    valid_frames = torch.nonzero(zero_command_valid_mask, as_tuple=False).squeeze(-1)
    if valid_frames.numel() == 0:
        return torch.full_like(time_steps, -1)
    indexes = torch.searchsorted(valid_frames, time_steps)
    has_future = indexes < valid_frames.numel()
    safe_indexes = indexes.clamp(max=valid_frames.numel() - 1)
    return torch.where(
        has_future,
        valid_frames[safe_indexes],
        torch.full_like(time_steps, -1),
    )


def next_valid_frame_per_env(
    valid_masks: torch.Tensor, time_steps: torch.Tensor
) -> torch.Tensor:
    """Return each environment's first valid frame at or after its query frame."""

    if valid_masks.ndim != 2 or time_steps.ndim != 1:
        raise ValueError("valid_masks must be [N, T] and time_steps must be [N]")
    if valid_masks.shape[0] != time_steps.shape[0]:
        raise ValueError("valid_masks and time_steps must have the same batch size")
    frame_indexes = torch.arange(
        valid_masks.shape[1], dtype=time_steps.dtype, device=time_steps.device
    ).unsqueeze(0)
    eligible = valid_masks & (frame_indexes >= time_steps.unsqueeze(-1))
    has_future, first_indexes = torch.max(eligible, dim=-1)
    return torch.where(has_future, first_indexes, torch.full_like(first_indexes, -1))
