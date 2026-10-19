"""Append up/down/stop command annotations without changing a CR1 motion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=np.bool_), [False])).astype(
        np.int8
    )
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [
        (int(start), int(end - 1))
        for start, end in zip(starts, ends, strict=True)
    ]


def _scalar_int(motion: dict[str, np.ndarray], name: str) -> int:
    if name not in motion:
        raise KeyError(f"motion is missing required metadata: {name}")
    return int(np.asarray(motion[name]).reshape(-1)[0])


def _stable_support_annotations(
    motion: dict[str, np.ndarray],
    *,
    middle_standing_frame: int,
    ground_surface_distance_threshold: float,
    min_segment_frames: int,
    max_limb_speed: float,
) -> dict[str, np.ndarray]:
    """Build ground-aware four-limb support states and one anchor per state run.

    Stable-support IDs deliberately do not replace the legacy ladder-only
    contact contract: ground is 0 and ladder rung ``k`` is ``k + 1``.
    """

    if ground_surface_distance_threshold <= 0.0:
        raise ValueError("ground_surface_distance_threshold must be positive")
    if min_segment_frames <= 0:
        raise ValueError("min_segment_frames must be positive")
    if max_limb_speed <= 0.0:
        raise ValueError("max_limb_speed must be positive")

    contact = np.asarray(motion["contact_indicator"]) > 0.5
    ladder_rung_id = np.asarray(motion["contact_rung_id"], dtype=np.int64)
    if contact.shape != ladder_rung_id.shape:
        raise ValueError("contact_indicator and contact_rung_id must have matching shapes")
    frame_count = contact.shape[0]
    if np.any(ladder_rung_id[contact] < 0):
        raise ValueError("active ladder contacts must have nonnegative rung IDs")

    body_names = np.asarray(motion["body_names"]).reshape(-1).tolist()
    limb_body_names = np.asarray(motion["limb_body_names"]).reshape(-1).tolist()
    if len(limb_body_names) != contact.shape[1]:
        raise ValueError("limb_body_names must align with the contact columns")
    missing_bodies = [name for name in limb_body_names if name not in body_names]
    if missing_bodies:
        raise ValueError(f"stable-support limb bodies are missing: {missing_bodies}")
    limb_body_indexes = np.asarray(
        [body_names.index(name) for name in limb_body_names], dtype=np.int64
    )
    limb_pos_z = np.asarray(motion["body_pos_w"], dtype=np.float32)[
        :, limb_body_indexes, 2
    ]
    limb_linear_speed = np.linalg.norm(
        np.asarray(motion["body_lin_vel_w"], dtype=np.float32)[
            :, limb_body_indexes
        ],
        axis=-1,
    )
    ground_z_value = motion.get(
        "reset_detection_ground_z",
        motion.get("motion_processing_default_standing_ground_z"),
    )
    if ground_z_value is None:
        raise KeyError(
            "ground-aware stable support requires reset_detection_ground_z or "
            "motion_processing_default_standing_ground_z"
    )
    ground_z = float(np.asarray(ground_z_value).reshape(-1)[0])
    target_offsets = np.asarray(
        motion["contact_annotation_limb_target_offsets"], dtype=np.float32
    )
    if target_offsets.shape != (contact.shape[1], 3):
        raise ValueError(
            "contact_annotation_limb_target_offsets must have shape (limb, 3)"
        )
    # For spherical hands, the target's vertical offset is the collision
    # radius.  For feet, use the collision-mesh minimum saved by the motion
    # ground-clearance pass; a virtual sole point is not the lowest point when
    # the foot pitches.
    ground_surface_z = limb_pos_z - target_offsets[None, :, 2]
    foot_indexes = [
        index
        for index, name in enumerate(np.asarray(motion["limb_names"]).tolist())
        if (
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
        ).endswith("_foot")
    ]
    if foot_indexes:
        if "motion_processing_ground_clearance_minimum_z_after" not in motion:
            raise KeyError(
                "foot ground support requires "
                "motion_processing_ground_clearance_minimum_z_after"
            )
        foot_minimum_z = np.asarray(
            motion["motion_processing_ground_clearance_minimum_z_after"],
            dtype=np.float32,
        )
        if foot_minimum_z.shape != (contact.shape[0], len(foot_indexes)):
            raise ValueError(
                "ground-clearance collision minima must align with the foot limbs"
            )
        ground_surface_z[:, foot_indexes] = foot_minimum_z
    ground_surface_distance = np.abs(ground_surface_z - ground_z)
    ground_contact = (~contact) & (
        ground_surface_distance <= ground_surface_distance_threshold
    )

    support = contact | ground_contact
    support_rung_id = np.full(contact.shape, -1, dtype=np.int64)
    support_rung_id[ground_contact] = 0
    support_rung_id[contact] = ladder_rung_id[contact] + 1

    all_limb_support = np.all(support, axis=1)
    segment_starts: list[int] = []
    segment_ends: list[int] = []
    start: int | None = None
    previous_state: tuple[int, ...] | None = None
    for frame, supported in enumerate(all_limb_support):
        state = tuple(support_rung_id[frame].tolist()) if supported else None
        if state == previous_state:
            continue
        if previous_state is not None and start is not None:
            end = frame - 1
            if end - start + 1 >= min_segment_frames:
                segment_starts.append(start)
                segment_ends.append(end)
        start = frame if state is not None else None
        previous_state = state
    if previous_state is not None and start is not None:
        end = len(all_limb_support) - 1
        if end - start + 1 >= min_segment_frames:
            segment_starts.append(start)
            segment_ends.append(end)

    surface_distance = np.asarray(
        motion["contact_surface_nearest_distance"], dtype=np.float32
    )
    release_distance = np.asarray(
        motion["contact_phase_annotation_release_distance_thresholds"],
        dtype=np.float32,
    ).reshape(-1)
    if surface_distance.shape != contact.shape:
        raise ValueError(
            "contact_surface_nearest_distance must match the contact annotations"
        )
    if release_distance.shape != (contact.shape[1],):
        raise ValueError(
            "contact phase release distance must provide one value per limb"
        )
    support_surface_distance = np.where(
        support_rung_id == 0,
        ground_surface_distance,
        surface_distance,
    )
    support_surface_limit = np.where(
        support_rung_id == 0,
        ground_surface_distance_threshold,
        release_distance[None, :],
    )
    stable_candidate = (
        all_limb_support
        & np.all(support_surface_distance <= support_surface_limit, axis=1)
        & (np.max(limb_linear_speed, axis=1) <= max_limb_speed)
    )
    anchor_frames_list: list[int] = []
    for start, end in zip(segment_starts, segment_ends, strict=True):
        candidates = np.flatnonzero(stable_candidate[start : end + 1]) + start
        if candidates.size == 0:
            raise ValueError(
                "four-limb support segment has no geometrically seated, low-speed "
                f"representative: [{start}, {end}]"
            )
        candidate_speed = np.max(limb_linear_speed[candidates], axis=1)
        anchor_frames_list.append(int(candidates[np.argmin(candidate_speed)]))
    support_anchor_frames = np.asarray(anchor_frames_list, dtype=np.int64)
    if support_anchor_frames.size == 0:
        raise ValueError("motion contains no ground-aware four-limb support segment")

    # Stable frames also delimit the beginning, the inserted middle standing
    # phase, and the true terminal frame.  These are intentionally separate
    # from the four-limb support anchors: frame 0 and the terminal standing
    # pose have only ground-supported feet, while the middle standing pose has
    # only the feet on the rung.  The kind field keeps that distinction
    # explicit for the loader/gate instead of weakening the support contract.
    if not 0 <= middle_standing_frame < frame_count:
        raise ValueError("middle_standing_frame must lie inside the motion")
    special_anchor_frames = np.asarray(
        [0, middle_standing_frame, frame_count - 1], dtype=np.int64
    )
    anchor_frames = np.unique(
        np.concatenate((support_anchor_frames, special_anchor_frames))
    )
    anchor_kind = np.ones(anchor_frames.shape, dtype=np.int64)
    anchor_kind[anchor_frames == 0] = 0  # initial standing frame
    anchor_kind[anchor_frames == middle_standing_frame] = 2  # middle standing frame
    anchor_kind[anchor_frames == frame_count - 1] = 3  # terminal frame
    anchor_is_full_support = anchor_kind == 1

    # Consecutive stable frames are the boundaries of one action segment.  Use
    # non-overlapping inclusive bounds for reset: every segment starts at an
    # anchor and ends before the next, except the last segment also owns the
    # terminal anchor.  Thus N stable boundaries form N-1 real transitions;
    # the final frame never becomes a spurious one-frame reset segment.
    reset_segment_starts = anchor_frames[:-1].copy()
    reset_segment_ends = anchor_frames[1:] - 1
    reset_segment_ends[-1] = frame_count - 1
    reset_segment_ids = np.searchsorted(
        reset_segment_starts, np.arange(frame_count, dtype=np.int64), side="right"
    ) - 1

    return {
        "stable_support_contact_indicator": support.astype(np.float32),
        "stable_support_rung_id": support_rung_id,
        "stable_support_ground_contact_indicator": ground_contact.astype(np.float32),
        "stable_support_all_limb_contact_mask": all_limb_support,
        "stable_support_segment_starts": np.asarray(segment_starts, dtype=np.int64),
        "stable_support_segment_ends": np.asarray(segment_ends, dtype=np.int64),
        "stable_support_anchor_frames": anchor_frames,
        "stable_support_anchor_kind": anchor_kind,
        "stable_support_anchor_is_full_support": anchor_is_full_support,
        "stable_support_reset_segment_starts": reset_segment_starts,
        "stable_support_reset_segment_ends": reset_segment_ends,
        "stable_support_reset_segment_anchor_frames": reset_segment_starts.copy(),
        "stable_support_reset_segment_ids": reset_segment_ids,
        "stable_support_anchor_max_limb_speeds": np.max(
            limb_linear_speed[anchor_frames], axis=1
        ).astype(np.float32),
        "stable_support_anchor_surface_distances": support_surface_distance[
            anchor_frames
        ].astype(np.float32),
        "stable_support_surface_distance": support_surface_distance.astype(
            np.float32
        ),
        "stable_support_ground_z": np.asarray([ground_z], dtype=np.float32),
        "stable_support_ground_surface_distance_threshold": np.asarray(
            [ground_surface_distance_threshold], dtype=np.float32
        ),
        "stable_support_ground_surface_source": np.asarray(
            [
                "sphere_bottom_for_hands_and_saved_ankle_collision_mesh_minimum_"
                "for_feet"
            ]
        ),
        "stable_support_min_segment_frames": np.asarray(
            [min_segment_frames], dtype=np.int64
        ),
        "stable_support_max_limb_speed_threshold": np.asarray(
            [max_limb_speed], dtype=np.float32
        ),
        "stable_support_rung_id_semantics": np.asarray(
            ["-1=no support, 0=ground, 1..N=ladder rung 0..N-1"]
        ),
        "stable_support_selection_method": np.asarray(
            [
                "minimum_max_limb_speed_per_constant_four_limb_support_run_"
                "after_surface_distance_and_speed_validation_plus_typed_"
                "initial_middle_standing_and_terminal_boundaries"
            ]
        ),
    }


def build_command_annotations(
    motion: dict[str, np.ndarray],
    *,
    source_file: Path,
    direction_switch_frame: int | None = None,
    stable_base_linear_speed: float = 0.10,
    stable_base_angular_speed: float = 0.30,
    stable_max_joint_speed: float = 1.00,
    terminal_min_frames: int = 10,
    stable_support_ground_surface_distance_threshold: float = 0.015,
    stable_support_min_segment_frames: int = 3,
    stable_support_max_limb_speed: float = 0.10,
) -> dict[str, np.ndarray]:
    """Return an exact motion copy with command-only arrays appended."""

    required = (
        "joint_pos",
        "joint_vel",
        "body_names",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "contact_indicator",
        "contact_rung_id",
        "contact_surface_nearest_distance",
        "contact_phase_annotation_release_distance_thresholds",
        "contact_annotation_limb_target_offsets",
        "limb_names",
        "limb_body_names",
        "reset_valid_mask",
    )
    missing = [name for name in required if name not in motion]
    if missing:
        raise KeyError(f"motion is missing required fields: {missing}")

    frame_count = len(motion["joint_pos"])
    root_index = np.asarray(motion["body_names"]).reshape(-1).tolist().index(
        "base_link"
    )
    initial_hold_frames = _scalar_int(
        motion, "motion_processing_default_standing_hold_frames"
    )
    insert_frame = _scalar_int(motion, "middle_default_standing_insert_source_frame")
    transition_in_frames = _scalar_int(
        motion, "middle_default_standing_transition_in_frames"
    )
    top_hold_frames = _scalar_int(motion, "middle_default_standing_hold_frames")
    transition_out_frames = _scalar_int(
        motion, "middle_default_standing_transition_out_frames"
    )
    top_hold_start = insert_frame + transition_in_frames
    top_hold_end = top_hold_start + top_hold_frames
    if direction_switch_frame is None:
        direction_switch_frame = top_hold_start + top_hold_frames // 2
    if not top_hold_start <= direction_switch_frame < top_hold_end:
        raise ValueError(
            "direction_switch_frame must lie inside the middle default-standing hold"
        )

    contact_indicator = np.asarray(motion["contact_indicator"]) > 0.5
    base_linear_speed = np.linalg.norm(
        np.asarray(motion["body_lin_vel_w"])[:, root_index], axis=-1
    )
    base_angular_speed = np.linalg.norm(
        np.asarray(motion["body_ang_vel_w"])[:, root_index], axis=-1
    )
    max_joint_speed = np.max(np.abs(np.asarray(motion["joint_vel"])), axis=-1)

    all_limb_contact = np.all(contact_indicator, axis=1)
    contact_runs = _runs(all_limb_contact)
    selected_contact_frames = np.asarray(
        [
            start + int(np.argmin(base_linear_speed[start : end + 1]))
            for start, end in contact_runs
        ],
        dtype=np.int64,
    )
    selected_contact_mask = np.zeros(frame_count, dtype=np.bool_)
    selected_contact_mask[selected_contact_frames] = True

    ground_default = np.zeros(frame_count, dtype=np.bool_)
    ground_default[:initial_hold_frames] = True
    top_default = np.zeros(frame_count, dtype=np.bool_)
    top_default[top_hold_start:top_hold_end] = True

    reset_envelope = np.asarray(
        motion.get("reset_valid_mask_ladder_geometry_envelope", motion["reset_valid_mask"]),
        dtype=np.bool_,
    )
    stable = (
        (base_linear_speed <= stable_base_linear_speed)
        & (base_angular_speed <= stable_base_angular_speed)
        & (max_joint_speed <= stable_max_joint_speed)
        & reset_envelope
    )
    terminal_runs = [
        (start, end)
        for start, end in _runs(stable)
        if end == frame_count - 1 and end - start + 1 >= terminal_min_frames
    ]
    if len(terminal_runs) != 1:
        raise ValueError(
            "motion must end in exactly one geometry-valid stable run of at least "
            f"{terminal_min_frames} frames"
        )
    terminal_start, terminal_end = terminal_runs[0]
    terminal_stable = np.zeros(frame_count, dtype=np.bool_)
    terminal_stable[terminal_start : terminal_end + 1] = True
    terminal_representative = terminal_start + int(
        np.argmin(base_linear_speed[terminal_start : terminal_end + 1])
    )

    zero_command_valid = (
        selected_contact_mask | ground_default | top_default | terminal_stable
    )
    direct_hold_frames = np.unique(
        np.concatenate(
            (
                selected_contact_frames,
                np.asarray(
                    [0, top_hold_start, terminal_representative], dtype=np.int64
                ),
            )
        )
    )
    if not np.all(zero_command_valid[direct_hold_frames]):
        raise RuntimeError("every direct-hold frame must be zero-command valid")

    climb_direction = np.ones((frame_count, 1), dtype=np.float32)
    climb_direction[direction_switch_frame:] = -1.0
    contact_run_starts = np.asarray(
        [start for start, _ in contact_runs], dtype=np.int64
    )
    contact_run_ends = np.asarray([end for _, end in contact_runs], dtype=np.int64)

    output = {name: np.asarray(value).copy() for name, value in motion.items()}
    stable_support = _stable_support_annotations(
        motion,
        middle_standing_frame=top_hold_start,
        ground_surface_distance_threshold=(
            stable_support_ground_surface_distance_threshold
        ),
        min_segment_frames=stable_support_min_segment_frames,
        max_limb_speed=stable_support_max_limb_speed,
    )
    output.update(
        {
            "climb_direction": climb_direction,
            "zero_command_valid_mask": zero_command_valid,
            "zero_command_valid_frame_indexes": np.flatnonzero(
                zero_command_valid
            ).astype(np.int64),
            "zero_command_direct_hold_frames": direct_hold_frames,
            "zero_command_all_limb_contact_mask": all_limb_contact,
            "zero_command_segment_min_base_speed_mask": selected_contact_mask,
            "zero_command_all_limb_contact_segment_starts": contact_run_starts,
            "zero_command_all_limb_contact_segment_ends": contact_run_ends,
            "zero_command_all_limb_contact_segment_selected_frames": (
                selected_contact_frames
            ),
            "zero_command_all_limb_contact_segment_selected_base_speeds": (
                base_linear_speed[selected_contact_frames].astype(np.float32)
            ),
            "zero_command_ground_default_standing_mask": ground_default,
            "zero_command_top_default_standing_mask": top_default,
            "zero_command_terminal_stable_ground_mask": terminal_stable,
            "zero_command_base_linear_speed": base_linear_speed.astype(np.float32),
            "zero_command_base_angular_speed": base_angular_speed.astype(np.float32),
            "zero_command_max_abs_joint_speed": max_joint_speed.astype(np.float32),
            "zero_command_stability_thresholds": np.asarray(
                [
                    stable_base_linear_speed,
                    stable_base_angular_speed,
                    stable_max_joint_speed,
                    float(terminal_min_frames),
                ],
                dtype=np.float32,
            ),
            "command_annotation_semantics": np.asarray(
                [
                    "climb_direction: +1=up, -1=down; legal zero-command frames "
                    "are one minimum-base-speed frame per all-limb-contact segment, "
                    "the initial/top default stands, and the terminal stable ground run"
                ]
            ),
            "command_annotation_source_file": np.asarray([str(source_file)]),
            "command_annotation_insert_source_frame": np.asarray(
                [insert_frame], dtype=np.int64
            ),
            "command_annotation_transition_in_frames": np.asarray(
                [transition_in_frames], dtype=np.int64
            ),
            "command_annotation_hold_frames": np.asarray(
                [top_hold_frames], dtype=np.int64
            ),
            "command_annotation_transition_out_frames": np.asarray(
                [transition_out_frames], dtype=np.int64
            ),
            "command_annotation_direction_switch_frame": np.asarray(
                [direction_switch_frame], dtype=np.int64
            ),
            "command_annotation_source_frame_offset_after_insert": np.asarray(
                [transition_in_frames + top_hold_frames + transition_out_frames],
                dtype=np.int64,
            ),
            "command_annotation_terminal_stable_start_frame": np.asarray(
                [terminal_start], dtype=np.int64
            ),
            "command_annotation_terminal_stable_end_frame": np.asarray(
                [terminal_end], dtype=np.int64
            ),
            "command_annotation_terminal_representative_frame": np.asarray(
                [terminal_representative], dtype=np.int64
            ),
            "zero_command_detection_requires_all_limb_contacts": np.asarray(
                [True], dtype=np.bool_
            ),
            "zero_command_detection_selection_method": np.asarray(
                [
                    "minimum_base_linear_speed_per_contiguous_all_limb_contact_"
                    "segment_plus_default_and_terminal_stable_ground_runs"
                ]
            ),
        }
    )
    output.update(stable_support)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--direction_switch_frame", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input motion not found: {input_file}")
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file exists: {output_file}. Pass --overwrite to replace it."
        )
    if input_file == output_file:
        raise ValueError("output_file must not overwrite input_file")

    with np.load(input_file, allow_pickle=False) as data:
        motion = {name: data[name] for name in data.files}
    output = build_command_annotations(
        motion,
        source_file=input_file,
        direction_switch_frame=args.direction_switch_frame,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **output)

    print(f"[INFO] Wrote command-annotated motion: {output_file}")
    print(
        "[INFO] Source fields preserved="
        f"{len(motion)}, output fields={len(output)}, frames={len(output['joint_pos'])}"
    )
    print(
        "[INFO] Direction switch frame="
        f"{int(output['command_annotation_direction_switch_frame'][0])}"
    )
    print(
        "[INFO] Natural stop frames="
        f"{output['zero_command_all_limb_contact_segment_selected_frames'].tolist()}"
    )
    print(
        "[INFO] Direct-hold frames="
        f"{output['zero_command_direct_hold_frames'].tolist()}"
    )
    print(
        "[INFO] Ground-aware stable-support anchors="
        f"{output['stable_support_anchor_frames'].tolist()}"
    )
    print(
        "[INFO] Zero-command valid runs="
        f"{_runs(output['zero_command_valid_mask'])}"
    )


if __name__ == "__main__":
    main()
