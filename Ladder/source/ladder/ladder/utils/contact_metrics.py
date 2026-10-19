"""Torch helpers for separating reference contact labels from physical contact."""

from __future__ import annotations

import torch


def encode_support_relation(
    support_ids: torch.Tensor, next_support_ids: torch.Tensor, num_rungs: int
) -> torch.Tensor:
    """Encode each limb as [contact, ground, relative_rung / num_rungs].

    All three support groups share the lowest positive rung of the next stable
    goal as origin (0 if that goal has only ground/no support). Ground and swing
    limbs have zero relative-rung channel, distinguished by the two masks.
    Raw -1 means no contact, but a *relative* -1 remains a valid lower rung.
    """

    if num_rungs < 1 or support_ids.shape != next_support_ids.shape:
        raise ValueError(
            "support ID shapes must match and num_rungs must be positive"
        )
    if support_ids.ndim != 2:
        raise ValueError("support IDs must have shape (env, limb)")
    if torch.any((support_ids < -1) | (support_ids > num_rungs)):
        raise ValueError("support IDs must use -1, 0, or 1..num_rungs")
    if torch.any((next_support_ids < -1) | (next_support_ids > num_rungs)):
        raise ValueError("next support IDs must use -1, 0, or 1..num_rungs")
    positive_next = next_support_ids > 0
    origin = torch.where(positive_next, next_support_ids, num_rungs + 1).amin(-1, keepdim=True)
    origin = torch.where(positive_next.any(-1, keepdim=True), origin, 0)
    relative = torch.where(support_ids > 0, support_ids - origin, 0).float() / num_rungs
    return torch.stack(
        ((support_ids >= 0).float(), (support_ids == 0).float(), relative),
        dim=-1,
    ).flatten(-2)


def classify_limb_contacts(
    contact_indicator: torch.Tensor,
    limb_positions_w: torch.Tensor,
    target_positions_w: torch.Tensor,
    net_forces_w: torch.Tensor,
    target_distance_thresholds: torch.Tensor | float,
    force_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reference-required, physical, and target-contact masks.

    ``physical_contact`` is a force-threshold proxy for contact on the selected
    limb body.  ``target_contact`` additionally requires the limb to be in its
    reference contact phase and within its configured distance of the target.
    ContactSensor net force is not pair-specific, so the distance gate avoids
    counting an unrelated contact as successful target contact.
    """

    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be nonnegative, got {force_threshold}")
    if contact_indicator.ndim != 2:
        raise ValueError(f"contact_indicator must have shape (env, limb), got {contact_indicator.shape}")
    if limb_positions_w.shape != target_positions_w.shape or limb_positions_w.ndim != 3:
        raise ValueError("limb and target positions must have matching shape (env, limb, 3)")
    if net_forces_w.shape != limb_positions_w.shape:
        raise ValueError("net_forces_w must have the same shape as limb positions")

    thresholds = torch.as_tensor(
        target_distance_thresholds,
        dtype=limb_positions_w.dtype,
        device=limb_positions_w.device,
    )
    if thresholds.ndim == 0:
        thresholds = thresholds.expand(limb_positions_w.shape[1])
    if thresholds.shape != (limb_positions_w.shape[1],):
        raise ValueError(
            "target_distance_thresholds must be a scalar or one value per limb, "
            f"got {thresholds.shape} for {limb_positions_w.shape[1]} limbs"
        )
    if torch.any(thresholds <= 0.0):
        raise ValueError("target distance thresholds must be positive")

    required = contact_indicator > 0.5
    physical_contact = torch.linalg.vector_norm(net_forces_w, dim=-1) > force_threshold
    target_near = torch.linalg.vector_norm(limb_positions_w - target_positions_w, dim=-1) <= thresholds
    target_contact = required & physical_contact & target_near
    return required, physical_contact, target_contact


def stable_anchor_contact_completion(
    desired: torch.Tensor,
    actual: torch.Tensor,
    target_contact: torch.Tensor,
    newly_required: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return strict/relaxed support completion and diagnostic fractions.

    Relaxed completion still requires every newly placed limb, forbids extra
    rung contacts, and permits at most one previously established support to
    flicker or momentarily unload.
    """

    tensors = (desired, actual, target_contact, newly_required)
    if desired.ndim != 2 or any(value.shape != desired.shape for value in tensors):
        raise ValueError("stable-anchor contact tensors must share shape (env, limb)")
    desired, actual, target_contact, newly_required = (
        value.bool() for value in tensors
    )
    strict = torch.all(actual == desired, dim=-1) & torch.all(
        (~desired) | target_contact, dim=-1
    )
    new_contacts_complete = torch.all((~newly_required) | target_contact, dim=-1)
    required_count = desired.sum(dim=-1)
    target_contact_count = target_contact.sum(dim=-1)
    relaxed = (
        new_contacts_complete
        & (target_contact_count >= (required_count - 1).clamp_min(0))
        & torch.all(~actual | desired, dim=-1)
    )
    contact_match_fraction = (actual == desired).float().mean(dim=-1)
    target_contact_fraction = torch.where(
        required_count > 0,
        target_contact_count.float() / required_count.clamp_min(1).float(),
        torch.ones_like(required_count, dtype=torch.float32),
    )
    return strict, relaxed, contact_match_fraction, target_contact_fraction


def stable_anchor_gate_transition(
    holding: torch.Tensor,
    hold_steps: torch.Tensor,
    confirmation_count: torch.Tensor,
    strict_complete: torch.Tensor,
    relaxed_complete: torch.Tensor,
    *,
    relaxed_after_steps: int,
    failure_after_steps: int,
    confirmation_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one control tick of the three-stage stable-anchor gate."""

    if min(relaxed_after_steps, failure_after_steps, confirmation_steps) <= 0:
        raise ValueError("stable-anchor gate step counts must be positive")
    if failure_after_steps <= relaxed_after_steps:
        raise ValueError("failure_after_steps must exceed relaxed_after_steps")
    expected = holding.shape
    if any(
        value.shape != expected
        for value in (
            hold_steps,
            confirmation_count,
            strict_complete,
            relaxed_complete,
        )
    ):
        raise ValueError("stable-anchor gate tensors must have matching shapes")

    holding = holding.bool()
    next_hold_steps = hold_steps + holding.long()
    completion = strict_complete.bool() | (
        (next_hold_steps >= relaxed_after_steps) & relaxed_complete.bool()
    )
    next_confirmation_count = torch.where(
        holding & completion,
        confirmation_count + 1,
        torch.zeros_like(confirmation_count),
    )
    release = holding & (next_confirmation_count >= confirmation_steps)
    timeout = holding & ~release & (next_hold_steps >= failure_after_steps)
    return next_hold_steps, next_confirmation_count, release, timeout
