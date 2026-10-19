from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ladder.tasks.ladder_climb.ladder_geometry import LADDER_A, LadderGeometry, generate_ladder, get_rung_pose
from ladder.utils.contact_schema import CR1_VIRTUAL_BODY_SPECS, DEFAULT_G1_BODY_NAMES, DEFAULT_LIMB_BODY_NAMES, LIMB_NAMES


@dataclass(frozen=True)
class ContactAnnotationCfg:
    """Rule-based labels for the first single-motion LadderMan expert."""

    ladder_geometry: LadderGeometry = field(default_factory=lambda: LADDER_A)
    limb_names: tuple[str, ...] = LIMB_NAMES
    limb_body_names: tuple[str, ...] = tuple(DEFAULT_LIMB_BODY_NAMES[name] for name in LIMB_NAMES)
    limb_target_offsets: tuple[tuple[float, float, float], ...] | None = None
    distance_threshold: float = 0.125
    limb_distance_thresholds: tuple[float, ...] | None = None
    limb_release_distance_thresholds: tuple[float, ...] | None = None
    speed_threshold: float = 0.35
    min_contact_frames: int = 4
    max_gap_frames: int = 2
    ladder_asset_translation: tuple[float, float, float] | None = None
    ladder_asset_yaw_deg: float | None = None
    ground_initial_feet: bool = False
    initial_foot_target_z: float = 0.0


@dataclass(frozen=True)
class ContactPhaseAnnotationCfg:
    """Geometry-based limb approach phases over finite ladder tread top surfaces."""

    ladder_geometry: LadderGeometry = field(default_factory=lambda: LADDER_A)
    limb_names: tuple[str, ...] = LIMB_NAMES
    limb_body_names: tuple[str, ...] = tuple(DEFAULT_LIMB_BODY_NAMES[name] for name in LIMB_NAMES)
    entry_distance_thresholds: tuple[float, ...] = (0.06, 0.06, 0.06, 0.06)
    release_distance_thresholds: tuple[float, ...] = (0.08, 0.08, 0.10, 0.10)
    entry_min_clearances: tuple[float, ...] = (-0.01, -0.01, -0.01, -0.01)
    release_min_clearances: tuple[float, ...] = (-0.015, -0.015, -0.015, -0.015)
    contact_point_z_offsets: tuple[float, ...] = (-0.045, -0.045, 0.0, 0.0)
    min_contact_frames: int = 4
    max_gap_frames: int = 5


@dataclass(frozen=True)
class DirectionalContactPhaseCfg:
    """Travel-order and departure rules applied to selected limb phases."""

    limb_names: tuple[str, ...] = ("left_hand", "right_hand")
    top_neutral_start_frame: int = 1235
    descent_start_frame: int = 1400
    core_distance_thresholds: tuple[float, ...] = (0.01, 0.01)
    departure_distance_thresholds: tuple[float, ...] = (0.01, 0.01)
    departure_rate_thresholds: tuple[float, ...] = (0.10, 0.10)
    departure_normal_velocity_thresholds: tuple[float, ...] = (0.20, 0.20)
    confirmation_frames: int = 2


def annotate_motion_file(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    cfg: ContactAnnotationCfg,
    *,
    limb_body_indexes: Sequence[int] | None = None,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Annotate a WBT-style motion npz with ladder contact labels and save it."""

    input_path = Path(input_file)
    output_path = Path(output_file)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass overwrite=True to replace it.")

    with np.load(input_path, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}

    motion = _prepare_motion(motion, cfg, limb_body_indexes)
    labels = annotate_ladder_contacts(motion, cfg, limb_body_indexes=limb_body_indexes)
    motion.update(labels)
    np.savez(output_path, **motion)
    return labels


def annotate_ladder_contacts(
    motion: Mapping[str, np.ndarray],
    cfg: ContactAnnotationCfg,
    *,
    limb_body_indexes: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Infer `contact_indicator` and `contact_rung_id` from end-effector trajectories.

    The first version intentionally assumes one upward reference motion with fixed timing.
    It labels frames as contact when the limb is close to its geometry-derived target
    rung point and moving slowly enough to be considered in a support phase.
    """

    if "fps" not in motion or "body_pos_w" not in motion:
        raise KeyError("motion must contain at least 'fps' and 'body_pos_w'")

    motion = _prepare_motion(motion, cfg, limb_body_indexes)

    body_pos_w = np.asarray(motion["body_pos_w"], dtype=np.float32)
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape (T, B, 3), got {body_pos_w.shape}")

    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    body_indexes = _resolve_limb_body_indexes(motion, cfg.limb_body_names, limb_body_indexes, body_pos_w.shape[1])
    limb_pos_w = body_pos_w[:, body_indexes, :]

    target_offsets = _limb_target_offsets(cfg.ladder_geometry, cfg.limb_names, cfg.limb_target_offsets)
    target_points = _target_points(cfg.ladder_geometry, target_offsets)
    distances = np.linalg.norm(limb_pos_w[:, :, None, :] - target_points[None, :, :, :], axis=-1)
    nearest_distances = distances.min(axis=-1)
    nearest_rung_ids = distances.argmin(axis=-1).astype(np.int64)
    speeds = _finite_difference_speed(limb_pos_w, fps)
    distance_thresholds = _limb_distance_thresholds(
        cfg.distance_threshold, cfg.limb_distance_thresholds, len(cfg.limb_names)
    )
    release_distance_thresholds = (
        distance_thresholds.copy()
        if cfg.limb_release_distance_thresholds is None
        else _limb_distance_thresholds(
            cfg.distance_threshold,
            cfg.limb_release_distance_thresholds,
            len(cfg.limb_names),
        )
    )
    if np.any(release_distance_thresholds < distance_thresholds):
        raise ValueError("limb release distance thresholds must not be smaller than entry thresholds")

    contact_indicator = np.zeros_like(nearest_distances, dtype=bool)
    hysteresis_filled_indicator = np.zeros_like(nearest_distances, dtype=bool)
    contact_rung_id = nearest_rung_ids.copy()
    raw_indicator = (nearest_distances <= distance_thresholds[None, :]) & (speeds <= cfg.speed_threshold)

    for limb_index in range(len(cfg.limb_names)):
        mask = raw_indicator[:, limb_index]
        mask = _fill_short_gaps(mask, cfg.max_gap_frames)
        mask = _remove_short_runs(mask, cfg.min_contact_frames)
        before_hysteresis = mask.copy()
        mask = _fill_near_target_contact_gaps(
            mask,
            nearest_distances[:, limb_index],
            nearest_rung_ids[:, limb_index],
            speeds[:, limb_index],
            release_distance_thresholds[limb_index],
            cfg.speed_threshold,
        )
        contact_indicator[:, limb_index] = mask
        hysteresis_filled_indicator[:, limb_index] = mask & ~before_hysteresis
        contact_rung_id[:, limb_index] = _stabilize_rung_ids(
            nearest_rung_ids[:, limb_index], mask, cfg.ladder_geometry.num_rungs
        )

    labels = {
        "contact_indicator": contact_indicator.astype(np.float32),
        "contact_rung_id": contact_rung_id.astype(np.int64),
        "limb_names": np.asarray(cfg.limb_names),
        "limb_body_names": np.asarray(cfg.limb_body_names),
        "limb_body_indexes": body_indexes.astype(np.int64),
        "contact_annotation_limb_target_offsets": target_offsets,
        "contact_annotation_limb_distance_thresholds": distance_thresholds,
        "contact_annotation_limb_release_distance_thresholds": release_distance_thresholds,
        "contact_nearest_distance": nearest_distances.astype(np.float32),
        "contact_speed": speeds.astype(np.float32),
        "contact_raw_indicator": raw_indicator.astype(np.float32),
        "contact_hysteresis_filled_indicator": hysteresis_filled_indicator.astype(np.float32),
        "contact_annotation_ladder_params": np.asarray(
            [
                cfg.ladder_geometry.phi,
                cfg.ladder_geometry.rung_spacing,
                cfg.ladder_geometry.rung_length,
                cfg.ladder_geometry.rung_width,
            ],
            dtype=np.float32,
        ),
        "contact_annotation_ladder_origin": np.asarray(cfg.ladder_geometry.origin, dtype=np.float32),
        "contact_annotation_ladder_num_rungs": np.asarray([cfg.ladder_geometry.num_rungs], dtype=np.int64),
        "contact_annotation_thresholds": np.asarray(
            [cfg.distance_threshold, cfg.speed_threshold, cfg.min_contact_frames, cfg.max_gap_frames],
            dtype=np.float32,
        ),
    }
    if cfg.ladder_asset_translation is not None:
        labels["contact_annotation_ladder_asset_translation"] = np.asarray(
            cfg.ladder_asset_translation, dtype=np.float32
        )
    if cfg.ladder_asset_yaw_deg is not None:
        labels["contact_annotation_ladder_asset_yaw_deg"] = np.asarray(
            [cfg.ladder_asset_yaw_deg], dtype=np.float32
        )
    if "contact_annotation_robot_grounding_z_offset" in motion:
        labels["contact_annotation_robot_grounding_z_offset"] = np.asarray(
            motion["contact_annotation_robot_grounding_z_offset"], dtype=np.float32
        )
    if "contact_annotation_robot_initial_foot_target_z" in motion:
        labels["contact_annotation_robot_initial_foot_target_z"] = np.asarray(
            motion["contact_annotation_robot_initial_foot_target_z"], dtype=np.float32
        )
    return labels


def annotate_ladder_contact_phases(
    motion: Mapping[str, np.ndarray],
    cfg: ContactPhaseAnnotationCfg,
    *,
    limb_body_indexes: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Annotate approach/contact phases using distance to finite tread top surfaces.

    Unlike :func:`annotate_ladder_contacts`, this annotation does not use a
    fixed lateral target point or a speed gate. Sphere-hand centers are shifted
    to their bottom surface before measuring clearance. The result is intended
    to coexist with, rather than replace, the stable support-contact labels.
    """

    if "body_pos_w" not in motion:
        raise KeyError("motion must contain 'body_pos_w'")

    body_pos_w = np.asarray(motion["body_pos_w"], dtype=np.float32)
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape (T, B, 3), got {body_pos_w.shape}")

    num_limbs = len(cfg.limb_names)
    if len(cfg.limb_body_names) != num_limbs:
        raise ValueError("limb_body_names and limb_names must have the same length")
    entry_distances = _phase_limb_values(
        "entry_distance_thresholds", cfg.entry_distance_thresholds, num_limbs, positive=True
    )
    release_distances = _phase_limb_values(
        "release_distance_thresholds", cfg.release_distance_thresholds, num_limbs, positive=True
    )
    entry_min_clearances = _phase_limb_values(
        "entry_min_clearances", cfg.entry_min_clearances, num_limbs
    )
    release_min_clearances = _phase_limb_values(
        "release_min_clearances", cfg.release_min_clearances, num_limbs
    )
    contact_point_z_offsets = _phase_limb_values(
        "contact_point_z_offsets", cfg.contact_point_z_offsets, num_limbs
    )
    if np.any(release_distances < entry_distances):
        raise ValueError("release distance thresholds must not be smaller than entry thresholds")
    if np.any(release_min_clearances > entry_min_clearances):
        raise ValueError("release minimum clearances must not exceed entry minimum clearances")
    if cfg.min_contact_frames < 1:
        raise ValueError("min_contact_frames must be at least 1")
    if cfg.max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")

    body_indexes, contact_points, surface_distances, clearances = _finite_tread_surface_measurements(
        motion,
        cfg,
        contact_point_z_offsets,
        limb_body_indexes,
    )
    nearest_rung_ids = surface_distances.argmin(axis=-1).astype(np.int64)
    nearest_distances = np.take_along_axis(
        surface_distances, nearest_rung_ids[..., None], axis=-1
    )[..., 0]
    nearest_clearances = np.take_along_axis(
        clearances, nearest_rung_ids[..., None], axis=-1
    )[..., 0]

    entry_candidates = (
        (surface_distances <= entry_distances[None, :, None])
        & (clearances >= entry_min_clearances[None, :, None])
    )
    raw_indicator = entry_candidates.any(axis=-1)
    phase_indicator = np.zeros_like(raw_indicator, dtype=bool)
    phase_rung_ids = np.full(raw_indicator.shape, -1, dtype=np.int64)

    for limb_index in range(num_limbs):
        active_rung = -1
        for frame in range(body_pos_w.shape[0]):
            if active_rung >= 0:
                within_release = (
                    surface_distances[frame, limb_index, active_rung]
                    <= release_distances[limb_index]
                    and clearances[frame, limb_index, active_rung]
                    >= release_min_clearances[limb_index]
                )
                if not within_release:
                    active_rung = -1

            if active_rung < 0:
                candidates = np.flatnonzero(entry_candidates[frame, limb_index])
                if candidates.size:
                    active_rung = int(
                        candidates[np.argmin(surface_distances[frame, limb_index, candidates])]
                    )

            if active_rung >= 0:
                phase_indicator[frame, limb_index] = True
                phase_rung_ids[frame, limb_index] = active_rung

        keep_mask = _remove_short_runs(phase_indicator[:, limb_index], cfg.min_contact_frames)
        phase_indicator[:, limb_index] = keep_mask
        phase_rung_ids[~keep_mask, limb_index] = -1
        _fill_same_rung_phase_gaps(
            phase_indicator[:, limb_index],
            phase_rung_ids[:, limb_index],
            cfg.max_gap_frames,
        )

    if "contact_indicator" in motion or "contact_rung_id" in motion:
        if "contact_indicator" not in motion or "contact_rung_id" not in motion:
            raise KeyError("contact_indicator and contact_rung_id must be provided together")
        support_indicator = np.asarray(motion["contact_indicator"]) > 0.5
        support_rung_ids = np.asarray(motion["contact_rung_id"], dtype=np.int64)
        if support_indicator.shape != phase_indicator.shape or support_rung_ids.shape != phase_rung_ids.shape:
            raise ValueError("stable support and contact-phase arrays must have matching shapes")
        phase_indicator[support_indicator] = True
        phase_rung_ids[support_indicator] = support_rung_ids[support_indicator]

    return {
        "contact_phase_indicator": phase_indicator.astype(np.float32),
        "contact_phase_rung_id": phase_rung_ids.astype(np.int64),
        "contact_phase_raw_indicator": raw_indicator.astype(np.float32),
        "contact_phase_hysteresis_filled_indicator": (phase_indicator & ~raw_indicator).astype(np.float32),
        "contact_surface_nearest_distance": nearest_distances.astype(np.float32),
        "contact_surface_nearest_clearance": nearest_clearances.astype(np.float32),
        "contact_surface_nearest_rung_id": nearest_rung_ids.astype(np.int64),
        "contact_phase_annotation_entry_distance_thresholds": entry_distances,
        "contact_phase_annotation_release_distance_thresholds": release_distances,
        "contact_phase_annotation_entry_min_clearances": entry_min_clearances,
        "contact_phase_annotation_release_min_clearances": release_min_clearances,
        "contact_phase_annotation_contact_point_z_offsets": contact_point_z_offsets,
        "contact_phase_annotation_temporal_params": np.asarray(
            [cfg.min_contact_frames, cfg.max_gap_frames], dtype=np.int64
        ),
        "contact_phase_annotation_surface_half_extents": np.asarray(
            [0.5 * cfg.ladder_geometry.rung_width, 0.5 * cfg.ladder_geometry.rung_length],
            dtype=np.float32,
        ),
    }


def annotate_directional_ladder_contact_phases(
    motion: Mapping[str, np.ndarray],
    phase_cfg: ContactPhaseAnnotationCfg,
    directional_cfg: DirectionalContactPhaseCfg,
    *,
    limb_body_indexes: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Refine selected surface phases with travel order and departure intent."""

    required = ("fps", "contact_indicator", "contact_rung_id")
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"directional phase annotation requires fields: {missing}")
    if not 0 <= directional_cfg.top_neutral_start_frame <= directional_cfg.descent_start_frame:
        raise ValueError("expected 0 <= top_neutral_start_frame <= descent_start_frame")

    labels = annotate_ladder_contact_phases(
        motion,
        phase_cfg,
        limb_body_indexes=limb_body_indexes,
    )
    base_phase_indicator = np.asarray(labels["contact_phase_indicator"]) > 0.5
    base_phase_rung_ids = np.asarray(labels["contact_phase_rung_id"], dtype=np.int64)
    num_frames, num_limbs = labels["contact_phase_indicator"].shape
    if directional_cfg.descent_start_frame >= num_frames:
        raise ValueError("descent_start_frame must be smaller than the motion frame count")

    selected_names = tuple(directional_cfg.limb_names)
    if not selected_names:
        raise ValueError("directional limb_names must not be empty")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("directional limb_names must be unique")
    missing_names = [name for name in selected_names if name not in phase_cfg.limb_names]
    if missing_names:
        raise ValueError(f"directional limbs are missing from phase_cfg.limb_names: {missing_names}")
    selected_indexes = np.asarray(
        [phase_cfg.limb_names.index(name) for name in selected_names], dtype=np.int64
    )
    num_selected = len(selected_names)
    core_distances = _phase_limb_values(
        "core_distance_thresholds",
        directional_cfg.core_distance_thresholds,
        num_selected,
        positive=True,
    )
    departure_distances = _phase_limb_values(
        "departure_distance_thresholds",
        directional_cfg.departure_distance_thresholds,
        num_selected,
        positive=True,
    )
    departure_rates = _phase_limb_values(
        "departure_rate_thresholds",
        directional_cfg.departure_rate_thresholds,
        num_selected,
        positive=True,
    )
    departure_normal_velocities = _phase_limb_values(
        "departure_normal_velocity_thresholds",
        directional_cfg.departure_normal_velocity_thresholds,
        num_selected,
        positive=True,
    )
    if directional_cfg.confirmation_frames < 1:
        raise ValueError("confirmation_frames must be at least 1")

    contact_point_z_offsets = _phase_limb_values(
        "contact_point_z_offsets", phase_cfg.contact_point_z_offsets, num_limbs
    )
    body_indexes, contact_points, surface_distances, clearances = _finite_tread_surface_measurements(
        motion,
        phase_cfg,
        contact_point_z_offsets,
        limb_body_indexes,
    )
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    if "body_lin_vel_w" in motion:
        body_lin_vel_w = np.asarray(motion["body_lin_vel_w"], dtype=np.float32)
        if body_lin_vel_w.shape != np.asarray(motion["body_pos_w"]).shape:
            raise ValueError("body_lin_vel_w must have the same shape as body_pos_w")
        limb_velocities = body_lin_vel_w[:, body_indexes, :]
    else:
        limb_velocities = _finite_difference_vectors(contact_points, fps)
    distance_rates = np.zeros_like(surface_distances, dtype=np.float32)
    if num_frames > 1:
        distance_rates[1:] = (surface_distances[1:] - surface_distances[:-1]) * fps
        distance_rates[0] = distance_rates[1]
    plane_normal = np.asarray(phase_cfg.ladder_geometry.plane_normal, dtype=np.float32)
    normal_velocities = np.einsum("tli,i->tl", limb_velocities, plane_normal)

    entry_distances = np.asarray(labels["contact_phase_annotation_entry_distance_thresholds"])
    release_distances = np.asarray(labels["contact_phase_annotation_release_distance_thresholds"])
    entry_min_clearances = np.asarray(labels["contact_phase_annotation_entry_min_clearances"])
    release_min_clearances = np.asarray(labels["contact_phase_annotation_release_min_clearances"])
    entry_candidates = (
        (surface_distances <= entry_distances[None, :, None])
        & (clearances >= entry_min_clearances[None, :, None])
    )
    support_indicator = np.asarray(motion["contact_indicator"]) > 0.5
    support_rung_ids = np.asarray(motion["contact_rung_id"], dtype=np.int64)
    phase_indicator = np.asarray(labels["contact_phase_indicator"]) > 0.5
    phase_rung_ids = np.asarray(labels["contact_phase_rung_id"], dtype=np.int64).copy()
    departure_indicator = np.zeros((num_frames, num_limbs), dtype=bool)
    departure_reason = np.zeros((num_frames, num_limbs), dtype=np.int64)

    travel_direction = np.ones(num_frames, dtype=np.int64)
    travel_direction[
        directional_cfg.top_neutral_start_frame : directional_cfg.descent_start_frame
    ] = 0
    travel_direction[directional_cfg.descent_start_frame :] = -1

    for selected_index, limb_index in enumerate(selected_indexes):
        phase_indicator[:, limb_index] = False
        phase_rung_ids[:, limb_index] = -1
        active_rung = -1
        completed_rung = -1
        previous_direction = 1
        reached_core = False
        saw_support = False
        departure_count = 0
        departure_confirmation_frames = max(
            directional_cfg.confirmation_frames,
            phase_cfg.max_gap_frames + 1,
        )

        for frame in range(num_frames):
            direction = int(travel_direction[frame])
            if direction != previous_direction:
                if active_rung >= 0:
                    departure_indicator[frame, limb_index] = True
                    departure_reason[frame, limb_index] = 4
                active_rung = -1
                reached_core = False
                saw_support = False
                departure_count = 0
                completed_rung = phase_cfg.ladder_geometry.num_rungs if direction < 0 else -1
                previous_direction = direction

            if direction == 0:
                continue

            if active_rung >= 0:
                distance = surface_distances[frame, limb_index, active_rung]
                clearance = clearances[frame, limb_index, active_rung]
                if distance <= core_distances[selected_index] and clearance >= entry_min_clearances[limb_index]:
                    reached_core = True
                if support_indicator[frame, limb_index] and support_rung_ids[frame, limb_index] == active_rung:
                    saw_support = True

                moving_away = (
                    reached_core
                    and distance > departure_distances[selected_index]
                    and distance_rates[frame, limb_index, active_rung] > departure_rates[selected_index]
                )
                support_departure = (
                    reached_core
                    and saw_support
                    and not support_indicator[frame, limb_index]
                    and normal_velocities[frame, limb_index]
                    > departure_normal_velocities[selected_index]
                )
                release_failed = (
                    distance > release_distances[limb_index]
                    or clearance < release_min_clearances[limb_index]
                )

                candidates = np.flatnonzero(entry_candidates[frame, limb_index])
                if direction > 0:
                    next_rung_candidates = candidates[candidates > active_rung]
                else:
                    next_rung_candidates = candidates[candidates < active_rung]
                reached_next_rung = next_rung_candidates.size > 0

                departure_evidence = moving_away or support_departure or release_failed
                departure_count = departure_count + 1 if departure_evidence else 0
                if reached_next_rung or departure_count >= departure_confirmation_frames:
                    departure_indicator[frame, limb_index] = True
                    departure_reason[frame, limb_index] = (
                        5
                        if reached_next_rung
                        else 3 if release_failed else 2 if support_departure else 1
                    )
                    if reached_core:
                        completed_rung = active_rung
                    active_rung = -1
                    reached_core = False
                    saw_support = False
                    departure_count = 0

            if active_rung < 0:
                candidates = np.flatnonzero(entry_candidates[frame, limb_index])
                if direction > 0 and completed_rung >= 0:
                    candidates = candidates[candidates > completed_rung]
                elif direction < 0 and completed_rung < phase_cfg.ladder_geometry.num_rungs:
                    candidates = candidates[candidates < completed_rung]
                if candidates.size:
                    active_rung = int(
                        candidates[np.argmin(surface_distances[frame, limb_index, candidates])]
                    )
                    reached_core = (
                        surface_distances[frame, limb_index, active_rung]
                        <= core_distances[selected_index]
                    )
                    saw_support = bool(
                        support_indicator[frame, limb_index]
                        and support_rung_ids[frame, limb_index] == active_rung
                    )

            if active_rung >= 0:
                phase_indicator[frame, limb_index] = True
                phase_rung_ids[frame, limb_index] = active_rung

        keep_mask = _remove_short_runs(phase_indicator[:, limb_index], phase_cfg.min_contact_frames)
        phase_indicator[:, limb_index] = keep_mask
        phase_rung_ids[~keep_mask, limb_index] = -1

    phase_indicator[support_indicator] = True
    phase_rung_ids[support_indicator] = support_rung_ids[support_indicator]
    recontact_bridge_indicator = _bridge_directional_same_rung_recontacts(
        phase_indicator,
        phase_rung_ids,
        base_phase_indicator,
        base_phase_rung_ids,
        selected_indexes,
        directional_cfg.descent_start_frame,
        phase_cfg.ladder_geometry.num_rungs,
    )

    labels["contact_phase_indicator"] = phase_indicator.astype(np.float32)
    labels["contact_phase_rung_id"] = phase_rung_ids.astype(np.int64)
    labels["contact_phase_hysteresis_filled_indicator"] = (
        phase_indicator & ~(np.asarray(labels["contact_phase_raw_indicator"]) > 0.5)
    ).astype(np.float32)
    directional_limb_mask = np.zeros(num_limbs, dtype=np.float32)
    directional_limb_mask[selected_indexes] = 1.0
    labels.update(
        {
            "contact_phase_travel_direction": travel_direction,
            "contact_phase_directional_limb_mask": directional_limb_mask,
            "contact_phase_departure_indicator": departure_indicator.astype(np.float32),
            "contact_phase_departure_reason": departure_reason,
            "contact_phase_directional_recontact_bridge_indicator": recontact_bridge_indicator.astype(
                np.float32
            ),
            "contact_phase_directional_limb_names": np.asarray(selected_names),
            "contact_phase_directional_frames": np.asarray(
                [directional_cfg.top_neutral_start_frame, directional_cfg.descent_start_frame],
                dtype=np.int64,
            ),
            "contact_phase_directional_core_distances": core_distances,
            "contact_phase_directional_departure_distances": departure_distances,
            "contact_phase_directional_departure_rates": departure_rates,
            "contact_phase_directional_departure_normal_velocities": departure_normal_velocities,
            "contact_phase_directional_confirmation_frames": np.asarray(
                [directional_cfg.confirmation_frames], dtype=np.int64
            ),
            "contact_phase_directional_effective_departure_confirmation_frames": np.asarray(
                [
                    max(
                        directional_cfg.confirmation_frames,
                        phase_cfg.max_gap_frames + 1,
                    )
                ],
                dtype=np.int64,
            ),
        }
    )
    return labels


def make_contact_annotation_cfg(
    *,
    ladder_phi: float = 65.5,
    rung_spacing: float = 0.248,
    rung_length: float = 0.60,
    rung_width: float = 0.08,
    num_rungs: int = 12,
    ladder_origin: tuple[float, float, float] = (0.55, 0.0, 0.08),
    limb_body_names: Sequence[str] | None = None,
    limb_target_offsets: Sequence[Sequence[float]] | None = None,
    distance_threshold: float = 0.125,
    limb_distance_thresholds: Sequence[float] | None = None,
    limb_release_distance_thresholds: Sequence[float] | None = None,
    speed_threshold: float = 0.35,
    min_contact_frames: int = 4,
    max_gap_frames: int = 2,
    ladder_asset_translation: tuple[float, float, float] | None = None,
    ladder_asset_yaw_deg: float | None = None,
    ground_initial_feet: bool = False,
    initial_foot_target_z: float = 0.0,
) -> ContactAnnotationCfg:
    """Create the default single-motion contact annotation config."""

    return ContactAnnotationCfg(
        ladder_geometry=generate_ladder(
            phi=ladder_phi,
            rung_spacing=rung_spacing,
            rung_length=rung_length,
            rung_radius_or_width=rung_width,
            num_rungs=num_rungs,
            origin=ladder_origin,
        ),
        limb_body_names=tuple(limb_body_names)
        if limb_body_names is not None
        else tuple(DEFAULT_LIMB_BODY_NAMES[name] for name in LIMB_NAMES),
        limb_target_offsets=(
            tuple(tuple(float(value) for value in offset) for offset in limb_target_offsets)
            if limb_target_offsets is not None
            else None
        ),
        distance_threshold=distance_threshold,
        limb_distance_thresholds=(
            tuple(float(value) for value in limb_distance_thresholds)
            if limb_distance_thresholds is not None
            else None
        ),
        limb_release_distance_thresholds=(
            tuple(float(value) for value in limb_release_distance_thresholds)
            if limb_release_distance_thresholds is not None
            else None
        ),
        speed_threshold=speed_threshold,
        min_contact_frames=min_contact_frames,
        max_gap_frames=max_gap_frames,
        ladder_asset_translation=ladder_asset_translation,
        ladder_asset_yaw_deg=ladder_asset_yaw_deg,
        ground_initial_feet=ground_initial_feet,
        initial_foot_target_z=initial_foot_target_z,
    )


def _prepare_motion(
    motion: Mapping[str, np.ndarray],
    cfg: ContactAnnotationCfg,
    limb_body_indexes: Sequence[int] | None,
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value) for key, value in motion.items()}
    if limb_body_indexes is None:
        result = _materialize_virtual_limb_bodies(result, cfg.limb_body_names)
    if cfg.ground_initial_feet:
        result = _ground_motion_at_initial_feet(
            result,
            cfg.limb_names,
            cfg.limb_body_names,
            limb_body_indexes,
            cfg.initial_foot_target_z,
        )
    return result


def _resolve_limb_body_indexes(
    motion: Mapping[str, np.ndarray],
    limb_body_names: Sequence[str],
    limb_body_indexes: Sequence[int] | None,
    num_bodies: int,
) -> np.ndarray:
    if limb_body_indexes is not None:
        indexes = np.asarray(limb_body_indexes, dtype=np.int64)
        if indexes.shape != (len(limb_body_names),):
            raise ValueError(f"expected {len(limb_body_names)} limb body indexes, got {indexes.shape}")
    else:
        if "body_names" in motion:
            body_names = _as_string_list(motion["body_names"])
        else:
            body_names = list(DEFAULT_G1_BODY_NAMES)
        missing = [name for name in limb_body_names if name not in body_names]
        if missing:
            raise ValueError(f"motion body names are missing limb bodies: {missing}")
        indexes = np.asarray([body_names.index(name) for name in limb_body_names], dtype=np.int64)

    if np.any(indexes < 0) or np.any(indexes >= num_bodies):
        raise IndexError(f"limb body indexes {indexes.tolist()} outside body_pos_w body count {num_bodies}")
    return indexes


def _as_string_list(values: np.ndarray) -> list[str]:
    items = np.asarray(values).reshape(-1).tolist()
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items]


def _materialize_virtual_limb_bodies(
    motion: Mapping[str, np.ndarray], limb_body_names: Sequence[str]
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value) for key, value in motion.items()}
    if "body_names" not in result:
        return result

    body_names = _as_string_list(result["body_names"])
    virtual_names = [
        name for name in limb_body_names if name not in body_names and name in CR1_VIRTUAL_BODY_SPECS
    ]
    if not virtual_names:
        return result
    if "body_quat_w" not in result:
        raise KeyError("motion must contain 'body_quat_w' to construct virtual sole contact points")

    body_pos_w = np.asarray(result["body_pos_w"], dtype=np.float32)
    body_quat_w = np.asarray(result["body_quat_w"], dtype=np.float32)
    if body_quat_w.shape != (*body_pos_w.shape[:2], 4):
        raise ValueError(
            f"body_quat_w must have shape {(*body_pos_w.shape[:2], 4)}, got {body_quat_w.shape}"
        )

    virtual_positions = []
    virtual_quaternions = []
    parent_indexes = []
    rotated_offsets = []
    for virtual_name in virtual_names:
        parent_name, local_offset = CR1_VIRTUAL_BODY_SPECS[virtual_name]
        if parent_name not in body_names:
            raise ValueError(f"motion body names are missing parent body '{parent_name}' for '{virtual_name}'")
        parent_index = body_names.index(parent_name)
        parent_quat_w = body_quat_w[:, parent_index, :]
        offset_w = _quat_apply_wxyz(parent_quat_w, np.asarray(local_offset, dtype=np.float32))
        virtual_positions.append(body_pos_w[:, parent_index, :] + offset_w)
        virtual_quaternions.append(parent_quat_w)
        parent_indexes.append(parent_index)
        rotated_offsets.append(offset_w)

    result["body_names"] = np.asarray(body_names + virtual_names)
    result["body_pos_w"] = np.concatenate(
        [body_pos_w, np.stack(virtual_positions, axis=1)], axis=1
    ).astype(np.float32)
    result["body_quat_w"] = np.concatenate(
        [body_quat_w, np.stack(virtual_quaternions, axis=1)], axis=1
    ).astype(np.float32)

    if "body_lin_vel_w" in result:
        body_lin_vel_w = np.asarray(result["body_lin_vel_w"], dtype=np.float32)
        if body_lin_vel_w.shape != body_pos_w.shape:
            raise ValueError(f"body_lin_vel_w must have shape {body_pos_w.shape}, got {body_lin_vel_w.shape}")
        if "body_ang_vel_w" in result:
            body_ang_vel_w = np.asarray(result["body_ang_vel_w"], dtype=np.float32)
            if body_ang_vel_w.shape != body_pos_w.shape:
                raise ValueError(f"body_ang_vel_w must have shape {body_pos_w.shape}, got {body_ang_vel_w.shape}")
            virtual_linear_velocities = [
                body_lin_vel_w[:, parent_index, :] + np.cross(body_ang_vel_w[:, parent_index, :], offset_w)
                for parent_index, offset_w in zip(parent_indexes, rotated_offsets, strict=True)
            ]
        else:
            fps = float(np.asarray(result["fps"]).reshape(-1)[0])
            virtual_linear_velocities = [_finite_difference_vectors(points, fps) for points in virtual_positions]
        result["body_lin_vel_w"] = np.concatenate(
            [body_lin_vel_w, np.stack(virtual_linear_velocities, axis=1)], axis=1
        ).astype(np.float32)

    if "body_ang_vel_w" in result:
        body_ang_vel_w = np.asarray(result["body_ang_vel_w"], dtype=np.float32)
        if body_ang_vel_w.shape != body_pos_w.shape:
            raise ValueError(f"body_ang_vel_w must have shape {body_pos_w.shape}, got {body_ang_vel_w.shape}")
        virtual_angular_velocities = body_ang_vel_w[:, parent_indexes, :]
        result["body_ang_vel_w"] = np.concatenate(
            [body_ang_vel_w, virtual_angular_velocities], axis=1
        ).astype(np.float32)

    return result


def _ground_motion_at_initial_feet(
    motion: Mapping[str, np.ndarray],
    limb_names: Sequence[str],
    limb_body_names: Sequence[str],
    limb_body_indexes: Sequence[int] | None,
    initial_foot_target_z: float,
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value) for key, value in motion.items()}
    if "contact_annotation_robot_grounding_z_offset" in result:
        return result

    foot_limb_indexes = [index for index, name in enumerate(limb_names) if name.endswith("_foot")]
    if not foot_limb_indexes:
        raise ValueError("ground_initial_feet requires at least one limb name ending in '_foot'")

    if limb_body_indexes is not None:
        indexes = np.asarray(limb_body_indexes, dtype=np.int64)
        foot_body_indexes = indexes[foot_limb_indexes]
    else:
        if "body_names" not in result:
            raise KeyError("motion must contain 'body_names' to ground initial feet")
        body_names = _as_string_list(result["body_names"])
        foot_body_names = [limb_body_names[index] for index in foot_limb_indexes]
        missing = [name for name in foot_body_names if name not in body_names]
        if missing:
            raise ValueError(f"motion body names are missing foot bodies for grounding: {missing}")
        foot_body_indexes = np.asarray([body_names.index(name) for name in foot_body_names], dtype=np.int64)

    body_pos_w = np.asarray(result["body_pos_w"], dtype=np.float32)
    initial_min_z = float(body_pos_w[0, foot_body_indexes, 2].min())
    grounding_z_offset = initial_foot_target_z - initial_min_z
    if grounding_z_offset != 0.0:
        body_pos_w = body_pos_w.copy()
        body_pos_w[..., 2] += grounding_z_offset
        result["body_pos_w"] = body_pos_w
    result["contact_annotation_robot_grounding_z_offset"] = np.asarray(
        [grounding_z_offset], dtype=np.float32
    )
    result["contact_annotation_robot_initial_foot_target_z"] = np.asarray(
        [initial_foot_target_z], dtype=np.float32
    )
    return result


def _quat_apply_wxyz(quaternions: np.ndarray, vector: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("body_quat_w contains a zero-length quaternion")
    quaternions = quaternions / norms
    vector = np.broadcast_to(vector, quaternions.shape[:-1] + (3,))
    xyz = quaternions[..., 1:]
    cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternions[..., :1] * cross + np.cross(xyz, cross)


def _finite_difference_vectors(points: np.ndarray, fps: float) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.zeros_like(points)
    return np.gradient(points, 1.0 / fps, axis=0).astype(np.float32)


def _limb_target_offsets(
    geometry: LadderGeometry,
    limb_names: Sequence[str],
    configured_offsets: Sequence[Sequence[float]] | None,
) -> np.ndarray:
    if configured_offsets is not None:
        offsets = np.asarray(configured_offsets, dtype=np.float32)
        expected_shape = (len(limb_names), 3)
        if offsets.shape != expected_shape:
            raise ValueError(f"limb_target_offsets must have shape {expected_shape}, got {offsets.shape}")
        return offsets

    offsets = []
    for limb_name in limb_names:
        if limb_name.startswith("left_"):
            offsets.append((0.0, geometry.lateral_contact_offset, 0.0))
        elif limb_name.startswith("right_"):
            offsets.append((0.0, -geometry.lateral_contact_offset, 0.0))
        else:
            offsets.append((0.0, 0.0, 0.0))
    return np.asarray(offsets, dtype=np.float32)


def _target_points(geometry: LadderGeometry, offsets: np.ndarray) -> np.ndarray:
    rung_positions = np.asarray([get_rung_pose(i, geometry)[0] for i in range(geometry.num_rungs)], dtype=np.float32)
    return rung_positions[None, :, :] + offsets[:, None, :]


def _limb_distance_thresholds(
    default_threshold: float,
    configured_thresholds: Sequence[float] | None,
    num_limbs: int,
) -> np.ndarray:
    if configured_thresholds is None:
        thresholds = np.full(num_limbs, default_threshold, dtype=np.float32)
    else:
        thresholds = np.asarray(configured_thresholds, dtype=np.float32)
        if thresholds.shape != (num_limbs,):
            raise ValueError(f"limb_distance_thresholds must have shape ({num_limbs},), got {thresholds.shape}")
    if np.any(thresholds <= 0.0):
        raise ValueError("limb distance thresholds must be positive")
    return thresholds


def _finite_difference_speed(points: np.ndarray, fps: float) -> np.ndarray:
    speeds = np.zeros(points.shape[:2], dtype=np.float32)
    if points.shape[0] <= 1:
        return speeds
    speeds[1:] = np.linalg.norm(points[1:] - points[:-1], axis=-1) * fps
    speeds[0] = speeds[1]
    return speeds


def _fill_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0:
        return mask.copy()
    result = mask.copy()
    runs = _runs(mask)
    for value, start, end in runs:
        if value or start == 0 or end == len(mask):
            continue
        if end - start <= max_gap_frames and mask[start - 1] and mask[end]:
            result[start:end] = True
    return result


def _remove_short_runs(mask: np.ndarray, min_contact_frames: int) -> np.ndarray:
    if min_contact_frames <= 1:
        return mask.copy()
    result = mask.copy()
    for value, start, end in _runs(mask):
        if value and end - start < min_contact_frames:
            result[start:end] = False
    return result


def _fill_near_target_contact_gaps(
    mask: np.ndarray,
    distances: np.ndarray,
    rung_ids: np.ndarray,
    speeds: np.ndarray,
    release_distance_threshold: float,
    speed_threshold: float,
) -> np.ndarray:
    """Keep same-rung contact through small target-distance excursions."""

    result = mask.copy()
    for value, start, end in _runs(mask):
        if value or start == 0 or end == len(mask):
            continue
        rung_id = rung_ids[start - 1]
        if rung_id != rung_ids[end] or np.any(rung_ids[start:end] != rung_id):
            continue
        if np.any(distances[start:end] > release_distance_threshold):
            continue
        if np.any(speeds[start:end] > speed_threshold):
            continue
        result[start:end] = True
    return result


def _finite_tread_surface_measurements(
    motion: Mapping[str, np.ndarray],
    cfg: ContactPhaseAnnotationCfg,
    contact_point_z_offsets: np.ndarray,
    limb_body_indexes: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    body_pos_w = np.asarray(motion["body_pos_w"], dtype=np.float32)
    body_indexes = _resolve_limb_body_indexes(
        motion, cfg.limb_body_names, limb_body_indexes, body_pos_w.shape[1]
    )
    contact_points = body_pos_w[:, body_indexes, :].copy()
    contact_points[..., 2] += contact_point_z_offsets[None, :]
    rung_centers = np.asarray(
        [get_rung_pose(index, cfg.ladder_geometry)[0] for index in range(cfg.ladder_geometry.num_rungs)],
        dtype=np.float32,
    )
    # Contact-annotation rung positions are top-surface target origins. This is
    # explicit for the wood-step preset and differs from procedural cuboid
    # center poses that happen to share the LadderGeometry container.
    delta_x = np.maximum(
        np.abs(contact_points[:, :, None, 0] - rung_centers[None, None, :, 0])
        - 0.5 * cfg.ladder_geometry.rung_width,
        0.0,
    )
    delta_y = np.maximum(
        np.abs(contact_points[:, :, None, 1] - rung_centers[None, None, :, 1])
        - 0.5 * cfg.ladder_geometry.rung_length,
        0.0,
    )
    clearances = contact_points[:, :, None, 2] - rung_centers[None, None, :, 2]
    surface_distances = np.sqrt(delta_x**2 + delta_y**2 + clearances**2)
    return body_indexes, contact_points, surface_distances, clearances


def _phase_limb_values(
    name: str,
    values: Sequence[float],
    num_limbs: int,
    *,
    positive: bool = False,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (num_limbs,):
        raise ValueError(f"{name} must have shape ({num_limbs},), got {result.shape}")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    return result


def _fill_same_rung_phase_gaps(
    phase_mask: np.ndarray,
    phase_rung_ids: np.ndarray,
    max_gap_frames: int,
) -> None:
    """Fill a short phase gap when both sides use the same tread."""

    if max_gap_frames <= 0:
        return
    for value, start, end in _runs(phase_mask):
        if value or start == 0 or end == len(phase_mask) or end - start > max_gap_frames:
            continue
        rung_id = int(phase_rung_ids[start - 1])
        if rung_id < 0 or rung_id != int(phase_rung_ids[end]):
            continue
        phase_mask[start:end] = True
        phase_rung_ids[start:end] = rung_id


def _bridge_directional_same_rung_recontacts(
    phase_indicator: np.ndarray,
    phase_rung_ids: np.ndarray,
    base_phase_indicator: np.ndarray,
    base_phase_rung_ids: np.ndarray,
    selected_limb_indexes: np.ndarray,
    descent_start_frame: int,
    num_rungs: int,
) -> np.ndarray:
    """Restore a base surface phase when directional output splits one support episode."""

    bridge_indicator = np.zeros_like(phase_indicator, dtype=bool)
    segment_bounds = ((0, descent_start_frame), (descent_start_frame, len(phase_indicator)))
    for limb_index in selected_limb_indexes:
        for segment_start, segment_end in segment_bounds:
            for rung_id in range(num_rungs):
                base_mask = (
                    base_phase_indicator[segment_start:segment_end, limb_index]
                    & (base_phase_rung_ids[segment_start:segment_end, limb_index] == rung_id)
                )
                for value, local_start, local_end in _runs(base_mask):
                    if not value:
                        continue
                    start = segment_start + local_start
                    end = segment_start + local_end
                    current_mask = (
                        phase_indicator[start:end, limb_index]
                        & (phase_rung_ids[start:end, limb_index] == rung_id)
                    )
                    active_runs = sum(value for value, _, _ in _runs(current_mask))
                    if active_runs < 2:
                        continue
                    bridge = ~current_mask
                    phase_indicator[start:end, limb_index] = True
                    phase_rung_ids[start:end, limb_index] = rung_id
                    bridge_indicator[start:end, limb_index] |= bridge
    return bridge_indicator


def _stabilize_rung_ids(rung_ids: np.ndarray, contact_mask: np.ndarray, num_rungs: int) -> np.ndarray:
    result = rung_ids.copy()
    for value, start, end in _runs(contact_mask):
        if not value:
            continue
        counts = np.bincount(rung_ids[start:end], minlength=num_rungs)
        result[start:end] = int(np.argmax(counts))
    return result


def _runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    if mask.size == 0:
        return []
    runs = []
    start = 0
    current = bool(mask[0])
    for index in range(1, len(mask)):
        value = bool(mask[index])
        if value != current:
            runs.append((current, start, index))
            start = index
            current = value
    runs.append((current, start, len(mask)))
    return runs
