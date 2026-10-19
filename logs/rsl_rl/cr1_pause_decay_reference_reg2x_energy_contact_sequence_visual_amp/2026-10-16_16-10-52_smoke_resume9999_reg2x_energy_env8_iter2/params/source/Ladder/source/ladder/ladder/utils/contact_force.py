"""Batched helpers for filtered ladder contact forces."""

from __future__ import annotations

import torch


STEP_SURFACES = ("all", "top", "front", "underside")


def physical_support_rung_id(
    rung_forces_w: torch.Tensor,
    ground_forces_w: torch.Tensor | None,
    filter_rung_ids: torch.Tensor,
    force_threshold: float,
) -> torch.Tensor:
    """Identify the strongest physical support, without consulting the reference.

    Filter columns are mapped explicitly to support IDs 1..N. Ground/platform
    is 0, no force above threshold is -1. Multiple contacting bodies are pooled
    by maximum force (as in the gate); simultaneous surfaces use the strongest
    pair, with ground then filter order breaking exact ties.
    """

    if rung_forces_w.ndim != 4 or rung_forces_w.shape[-1] != 3:
        raise ValueError("rung forces must have shape (env, body, filter, 3)")
    if filter_rung_ids.shape != (rung_forces_w.shape[2],):
        raise ValueError("filter_rung_ids must match the contact filter columns")
    if force_threshold < 0.0:
        raise ValueError("force_threshold must be nonnegative")
    rung_force = torch.linalg.vector_norm(rung_forces_w, dim=-1).amax(dim=1)
    ground_force = rung_force.new_zeros(rung_force.shape[0], 1)
    if ground_forces_w is not None:
        if (
            ground_forces_w.ndim != 4
            or ground_forces_w.shape[0] != rung_forces_w.shape[0]
            or ground_forces_w.shape[-1] != 3
        ):
            raise ValueError("ground forces must have shape (env, body, filter, 3)")
        ground_force = (
            torch.linalg.vector_norm(ground_forces_w, dim=-1)
            .flatten(1)
            .amax(-1, keepdim=True)
        )
    force, index = torch.cat((ground_force, rung_force), dim=-1).max(dim=-1)
    ids = torch.cat((filter_rung_ids.new_zeros(1), filter_rung_ids))[index]
    return torch.where(force > force_threshold, ids, -torch.ones_like(ids))


def target_rung_forces(
    force_matrix_w: torch.Tensor,
    rung_ids: torch.Tensor,
) -> torch.Tensor:
    """Select each body's filtered force for its per-environment target rung."""

    if force_matrix_w.ndim != 4 or force_matrix_w.shape[-1] != 3:
        raise ValueError(
            "force_matrix_w must have shape (env, body, rung, 3), got "
            f"{tuple(force_matrix_w.shape)}"
        )
    if rung_ids.shape != force_matrix_w.shape[:2]:
        raise ValueError(
            "rung_ids must match the force matrix env/body dimensions, got "
            f"{tuple(rung_ids.shape)} for {tuple(force_matrix_w.shape)}"
        )

    rung_count = force_matrix_w.shape[2]
    valid = (rung_ids >= 0) & (rung_ids < rung_count)
    gather_ids = rung_ids.clamp(0, rung_count - 1)
    selected = torch.gather(
        force_matrix_w,
        2,
        gather_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
    ).squeeze(2)
    return torch.where(valid.unsqueeze(-1), selected, torch.zeros_like(selected))


def non_target_rung_forces(
    force_matrix_w: torch.Tensor,
    rung_ids: torch.Tensor,
) -> torch.Tensor:
    """Zero each body's target-rung entry and retain every non-target rung."""

    if force_matrix_w.ndim != 4 or force_matrix_w.shape[-1] != 3:
        raise ValueError(
            "force_matrix_w must have shape (env, body, rung, 3), got "
            f"{tuple(force_matrix_w.shape)}"
        )
    if rung_ids.shape != force_matrix_w.shape[:2]:
        raise ValueError(
            "rung_ids must match the force matrix env/body dimensions, got "
            f"{tuple(rung_ids.shape)} for {tuple(force_matrix_w.shape)}"
        )

    rung_count = force_matrix_w.shape[2]
    filter_ids = torch.arange(rung_count, device=force_matrix_w.device).view(1, 1, -1)
    valid = (rung_ids >= 0) & (rung_ids < rung_count)
    keep = (~valid).unsqueeze(-1) | (filter_ids != rung_ids.unsqueeze(-1))
    return torch.where(keep.unsqueeze(-1), force_matrix_w, torch.zeros_like(force_matrix_w))


def filtered_surface_force_max(
    force_matrix_w: torch.Tensor,
    *,
    surface: str,
    min_top_normal_fraction: float = 0.5,
) -> torch.Tensor:
    """Return each environment's maximum filtered force on a step surface."""

    if force_matrix_w.ndim != 4 or force_matrix_w.shape[-1] != 3:
        raise ValueError(
            "force_matrix_w must have shape (env, body, filter, 3), got "
            f"{tuple(force_matrix_w.shape)}"
        )
    if surface not in STEP_SURFACES:
        raise ValueError(f"surface must be one of {STEP_SURFACES}, got {surface!r}")
    if not 0.0 < min_top_normal_fraction <= 1.0:
        raise ValueError(
            "min_top_normal_fraction must be in (0, 1], got "
            f"{min_top_normal_fraction}"
        )

    force_norm = torch.linalg.vector_norm(force_matrix_w, dim=-1)
    if surface == "all":
        selected = force_norm
    else:
        normal_z_fraction = force_matrix_w[..., 2] / force_norm.clamp_min(1.0e-6)
        if surface == "top":
            mask = normal_z_fraction >= min_top_normal_fraction
        elif surface == "front":
            mask = torch.abs(normal_z_fraction) < min_top_normal_fraction
        else:
            mask = normal_z_fraction <= -min_top_normal_fraction
        selected = torch.where(mask, force_norm, torch.zeros_like(force_norm))
    return selected.amax(dim=(-1, -2))
