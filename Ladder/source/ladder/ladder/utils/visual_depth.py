"""Depth preprocessing and augmentation for LadderMan visual policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DepthAugmentationCfg:
    """Paper-aligned depth augmentation parameters."""

    min_depth: float = 0.1
    max_depth: float = 2.0
    gaussian_std: float = 0.005
    pixel_dropout_probability: float = 0.05
    rung_focused_mask_probability: float = 0.1
    invalid_depth: float = 0.0


LADDERMAN_DEPTH_AUGMENTATION = DepthAugmentationCfg()


def camera_image_to_nchw(image: torch.Tensor) -> torch.Tensor:
    """Convert an Isaac Lab camera image from NHWC to NCHW."""

    if image.ndim != 4:
        raise ValueError(f"Expected a 4-D camera image, got shape {tuple(image.shape)}")
    return image.permute(0, 3, 1, 2).contiguous()


def prepare_metric_depth(
    depth_nhwc: torch.Tensor,
    *,
    min_depth: float = 0.1,
    max_depth: float = 2.0,
) -> torch.Tensor:
    """Return finite, clipped metric depth in NCHW layout."""

    if min_depth < 0.0 or max_depth <= min_depth:
        raise ValueError(f"Invalid depth interval [{min_depth}, {max_depth}]")
    depth = camera_image_to_nchw(depth_nhwc).to(dtype=torch.float32)
    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=min_depth)
    return depth.clamp(min=min_depth, max=max_depth)


def rfm_semantic_ids_from_info(semantic_info: Mapping[str, Any] | None) -> tuple[int, ...]:
    """Return runtime semantic IDs explicitly labeled as a rung or robot."""

    if not isinstance(semantic_info, Mapping):
        return ()
    id_to_labels = semantic_info.get("idToLabels", {})
    if not isinstance(id_to_labels, Mapping):
        return ()

    keep_ids = []
    for semantic_id, labels in id_to_labels.items():
        if not isinstance(labels, Mapping):
            continue
        class_label = str(labels.get("class", "")).strip().lower()
        if class_label in {"rung", "robot"}:
            keep_ids.append(int(semantic_id))
    return tuple(sorted(set(keep_ids)))


def rfm_keep_mask_from_semantic(
    semantic_nhwc: torch.Tensor, semantic_info: Mapping[str, Any] | None
) -> torch.Tensor:
    """Return the NCHW RFM keep mask for explicit rung and robot IDs."""

    semantic = camera_image_to_nchw(semantic_nhwc)
    if semantic.shape[1] != 1:
        raise ValueError(
            "RFM semantic segmentation must be non-colorized with one channel, "
            f"got shape {tuple(semantic.shape)}"
        )
    keep_ids = rfm_semantic_ids_from_info(semantic_info)
    if not keep_ids:
        return torch.zeros_like(semantic, dtype=torch.bool)
    ids = torch.as_tensor(keep_ids, dtype=semantic.dtype, device=semantic.device)
    return torch.isin(semantic, ids)


def add_depth_noise_and_dropout(
    depth: torch.Tensor,
    cfg: DepthAugmentationCfg = LADDERMAN_DEPTH_AUGMENTATION,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply Gaussian depth noise and independent invalid-pixel dropout."""

    _validate_probability(cfg.pixel_dropout_probability, "pixel_dropout_probability")
    if cfg.gaussian_std < 0.0:
        raise ValueError("gaussian_std must be non-negative")

    augmented = depth.clone()
    if cfg.gaussian_std > 0.0:
        noise = torch.randn(
            augmented.shape,
            dtype=augmented.dtype,
            device=augmented.device,
            generator=generator,
        )
        augmented.add_(noise, alpha=cfg.gaussian_std)
        augmented.clamp_(min=cfg.min_depth, max=cfg.max_depth)
    if cfg.pixel_dropout_probability > 0.0:
        dropout = torch.rand(
            augmented.shape,
            dtype=augmented.dtype,
            device=augmented.device,
            generator=generator,
        ) < cfg.pixel_dropout_probability
        augmented.masked_fill_(dropout, cfg.invalid_depth)
    return augmented


def apply_rung_focused_masking(
    depth: torch.Tensor,
    rfm_keep_mask: torch.Tensor,
    cfg: DepthAugmentationCfg = LADDERMAN_DEPTH_AUGMENTATION,
    *,
    generator: torch.Generator | None = None,
    probability: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask pixels outside the rung-and-robot RFM region for sampled images.

    Returns the masked depth and a per-environment boolean indicating whether
    RFM was applied.
    """

    if depth.shape != rfm_keep_mask.shape:
        raise ValueError(
            "Depth and RFM keep-mask shapes must match, "
            f"got {tuple(depth.shape)} and {tuple(rfm_keep_mask.shape)}"
        )
    rfm_probability = cfg.rung_focused_mask_probability if probability is None else probability
    _validate_probability(rfm_probability, "rung_focused_mask_probability")

    apply_mask = torch.rand(
        (depth.shape[0],),
        dtype=depth.dtype,
        device=depth.device,
        generator=generator,
    ) < rfm_probability
    masked = torch.where(
        apply_mask[:, None, None, None] & ~rfm_keep_mask,
        torch.as_tensor(cfg.invalid_depth, dtype=depth.dtype, device=depth.device),
        depth,
    )
    return masked, apply_mask


def process_visual_depth(
    depth_nhwc: torch.Tensor,
    semantic_nhwc: torch.Tensor,
    semantic_info: Mapping[str, Any] | None,
    cfg: DepthAugmentationCfg = LADDERMAN_DEPTH_AUGMENTATION,
    *,
    generator: torch.Generator | None = None,
    rfm_probability: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare depth and select RFM or noise augmentation per image.

    LadderMan Figure 3 uses rung-focused masking for 10% of images and the
    Gaussian-noise/dropout branch for the remaining 90%.
    """

    depth = prepare_metric_depth(
        depth_nhwc,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
    )
    rfm_keep_mask = rfm_keep_mask_from_semantic(semantic_nhwc, semantic_info)
    rung_focused, apply_rfm = apply_rung_focused_masking(
        depth,
        rfm_keep_mask,
        cfg,
        generator=generator,
        probability=rfm_probability,
    )
    noisy = add_depth_noise_and_dropout(depth, cfg, generator=generator)
    processed = torch.where(apply_rfm[:, None, None, None], rung_focused, noisy)
    return processed, apply_rfm


def process_warp_visual_depth(
    depth_nhw: torch.Tensor,
    hit_class_nhw: torch.Tensor,
    *,
    rung_class: int,
    robot_class: int,
    cfg: DepthAugmentationCfg = LADDERMAN_DEPTH_AUGMENTATION,
    generator: torch.Generator | None = None,
    rfm_probability: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the visual-policy augmentation to native Warp depth/classes.

    Warp emits metric image-plane depth and an integer hit-class image without
    Replicator's runtime semantic-ID table.  The RFM definition stays exactly
    the same as the RTX path: retain only rung and robot pixels.
    """

    if depth_nhw.ndim != 3 or hit_class_nhw.ndim != 3:
        raise ValueError(
            "Warp depth and hit classes must be [num_envs, height, width], got "
            f"{tuple(depth_nhw.shape)} and {tuple(hit_class_nhw.shape)}"
        )
    if depth_nhw.shape != hit_class_nhw.shape:
        raise ValueError(
            "Warp depth and hit classes must have matching shapes, got "
            f"{tuple(depth_nhw.shape)} and {tuple(hit_class_nhw.shape)}"
        )

    depth = torch.nan_to_num(
        depth_nhw.to(dtype=torch.float32),
        nan=cfg.max_depth,
        posinf=cfg.max_depth,
        neginf=cfg.min_depth,
    ).clamp(min=cfg.min_depth, max=cfg.max_depth).unsqueeze(1)
    rfm_keep_mask = ((hit_class_nhw == rung_class) | (hit_class_nhw == robot_class)).unsqueeze(1)
    rung_focused, apply_rfm = apply_rung_focused_masking(
        depth,
        rfm_keep_mask,
        cfg,
        generator=generator,
        probability=rfm_probability,
    )
    noisy = add_depth_noise_and_dropout(depth, cfg, generator=generator)
    processed = torch.where(apply_rfm[:, None, None, None], rung_focused, noisy)
    return processed, apply_rfm


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


__all__ = [
    "DepthAugmentationCfg",
    "LADDERMAN_DEPTH_AUGMENTATION",
    "add_depth_noise_and_dropout",
    "apply_rung_focused_masking",
    "camera_image_to_nchw",
    "prepare_metric_depth",
    "process_visual_depth",
    "process_warp_visual_depth",
    "rfm_keep_mask_from_semantic",
    "rfm_semantic_ids_from_info",
]
