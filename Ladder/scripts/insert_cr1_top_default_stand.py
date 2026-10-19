"""Insert a top default-stand interval and command annotations into a CR1 motion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ladder.utils.contact_annotation import annotate_ladder_contacts, make_contact_annotation_cfg
from ladder.utils.urdf_kinematics import (
    finite_difference,
    forward_kinematics,
    quaternion_angular_velocity,
)


DEFAULT_CR1_URDF = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)
SOLE_BODY_NAMES = ("left_sole_contact_point", "right_sole_contact_point")
CONTACT_FRAME_FIELDS = {
    "contact_indicator",
    "contact_rung_id",
    "contact_nearest_distance",
    "contact_speed",
    "contact_raw_indicator",
    "contact_hysteresis_filled_indicator",
}
RESET_FIELDS = {
    "reset_valid_mask",
    "reset_valid_frame_indexes",
    "reset_invalid_frame_indexes",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_CR1_URDF)
    parser.add_argument("--insert_frame", type=int, default=1100)
    parser.add_argument("--transition_in_frames", type=int, default=100)
    parser.add_argument("--hold_frames", type=int, default=100)
    parser.add_argument("--transition_out_frames", type=int, default=100)
    parser.add_argument("--direction_switch_frame", type=int, default=1250)
    parser.add_argument("--terminal_transition_frames", type=int, default=50)
    parser.add_argument("--terminal_hold_frames", type=int, default=25)
    parser.add_argument(
        "--hand_contact_release_distance",
        type=float,
        default=0.08,
        help="Same-rung hand-contact release distance; entry distances remain unchanged.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _as_names(values: np.ndarray) -> list[str]:
    items = np.asarray(values).reshape(-1).tolist()
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items]


def _yaw_only_quaternion(quat_wxyz: np.ndarray) -> np.ndarray:
    source = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
    yaw = source.as_euler("xyz")[2]
    quat_xyzw = Rotation.from_euler("z", yaw).as_quat()
    return quat_xyzw[[3, 0, 1, 2]]


def _quintic(value: np.ndarray) -> np.ndarray:
    return 6.0 * value**5 - 15.0 * value**4 + 10.0 * value**3


def _transition_weights(frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("transition frame counts must be positive")
    # Excluding both endpoints makes the user-visible hold exactly hold_frames long.
    phase = np.arange(1, frame_count + 1, dtype=np.float64) / (frame_count + 1)
    return _quintic(phase)


def _slerp(start_wxyz: np.ndarray, end_wxyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    end = np.asarray(end_wxyz, dtype=np.float64)
    if np.dot(start_wxyz, end) < 0.0:
        end = -end
    rotations = Rotation.from_quat(np.stack((start_wxyz, end))[:, [1, 2, 3, 0]])
    quat_xyzw = Slerp([0.0, 1.0], rotations)(weights).as_quat()
    return quat_xyzw[:, [3, 0, 1, 2]]


def _blend(start: np.ndarray, end: np.ndarray, weights: np.ndarray) -> np.ndarray:
    shape = (len(weights),) + (1,) * np.asarray(start).ndim
    weight = weights.reshape(shape)
    return (1.0 - weight) * np.asarray(start)[None] + weight * np.asarray(end)[None]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=np.bool_), [False])).astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [(int(start), int(end - 1)) for start, end in zip(starts, ends, strict=True)]


def _contact_cfg(motion: dict[str, np.ndarray], hand_contact_release_distance: float):
    params = np.asarray(motion["contact_annotation_ladder_params"], dtype=np.float64).reshape(4)
    thresholds = np.asarray(motion["contact_annotation_thresholds"], dtype=np.float64).reshape(4)
    entry_distances = np.asarray(
        motion["contact_annotation_limb_distance_thresholds"], dtype=np.float64
    )
    release_distances = entry_distances.copy()
    release_distances[:2] = hand_contact_release_distance
    return make_contact_annotation_cfg(
        ladder_phi=float(params[0]),
        rung_spacing=float(params[1]),
        rung_length=float(params[2]),
        rung_width=float(params[3]),
        num_rungs=int(np.asarray(motion["contact_annotation_ladder_num_rungs"]).reshape(-1)[0]),
        ladder_origin=tuple(
            np.asarray(motion["contact_annotation_ladder_origin"], dtype=np.float64).reshape(3)
        ),
        limb_body_names=_as_names(motion["limb_body_names"]),
        limb_target_offsets=np.asarray(
            motion["contact_annotation_limb_target_offsets"], dtype=np.float64
        ),
        distance_threshold=float(thresholds[0]),
        limb_distance_thresholds=entry_distances,
        limb_release_distance_thresholds=release_distances,
        speed_threshold=float(thresholds[1]),
        min_contact_frames=int(round(float(thresholds[2]))),
        max_gap_frames=int(round(float(thresholds[3]))),
        ladder_asset_translation=tuple(
            np.asarray(motion["contact_annotation_ladder_asset_translation"], dtype=np.float64).reshape(3)
        ),
        ladder_asset_yaw_deg=float(
            np.asarray(motion["contact_annotation_ladder_asset_yaw_deg"]).reshape(-1)[0]
        ),
    )


def _default_standing_pose(
    motion: dict[str, np.ndarray], robot_asset: Path, insert_frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_names = _as_names(motion["joint_names"])
    body_names = _as_names(motion["body_names"])
    root_index = body_names.index("base_link")
    sole_indexes = [body_names.index(name) for name in SOLE_BODY_NAMES]
    standing_joint_pos = np.zeros(len(joint_names), dtype=np.float64)
    standing_root_quat = _yaw_only_quaternion(motion["body_quat_w"][insert_frame, root_index])
    body_pos, _ = forward_kinematics(
        robot_asset,
        np.zeros((1, 3), dtype=np.float64),
        standing_root_quat[None],
        standing_joint_pos[None],
        joint_names,
        body_names,
    )
    source_soles = np.asarray(motion["body_pos_w"][insert_frame, sole_indexes], dtype=np.float64)
    standing_soles = np.asarray(body_pos[0, sole_indexes], dtype=np.float64)
    standing_root_pos = np.empty(3, dtype=np.float64)
    standing_root_pos[:2] = source_soles[:, :2].mean(axis=0) - standing_soles[:, :2].mean(axis=0)
    standing_root_pos[2] = source_soles[:, 2].min() - standing_soles[:, 2].min()
    return standing_joint_pos, standing_root_pos, standing_root_quat


def _insert_auxiliary_frame_field(
    values: np.ndarray, insert_frame: int, inserted_frames: int
) -> np.ndarray:
    values = np.asarray(values)
    standing_value = 0 if np.issubdtype(values.dtype, np.number) else values[insert_frame]
    inserted = np.full((inserted_frames,) + values.shape[1:], standing_value, dtype=values.dtype)
    return np.concatenate((values[:insert_frame], inserted, values[insert_frame:]), axis=0)


def _append_auxiliary_frame_field(values: np.ndarray, appended_frames: int) -> np.ndarray:
    values = np.asarray(values)
    standing_value = 0 if np.issubdtype(values.dtype, np.number) else values[-1]
    appended = np.full((appended_frames,) + values.shape[1:], standing_value, dtype=values.dtype)
    return np.concatenate((values, appended), axis=0)


def build_command_motion(
    motion: dict[str, np.ndarray],
    *,
    robot_asset: Path,
    source_file: Path,
    insert_frame: int,
    transition_in_frames: int,
    hold_frames: int,
    transition_out_frames: int,
    direction_switch_frame: int,
    hand_contact_release_distance: float,
    terminal_transition_frames: int,
    terminal_hold_frames: int,
) -> dict[str, np.ndarray]:
    required = (
        "fps",
        "joint_pos",
        "joint_names",
        "body_pos_w",
        "body_quat_w",
        "body_names",
        "contact_indicator",
        "contact_annotation_ladder_params",
        "contact_annotation_thresholds",
    )
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"motion is missing required keys: {missing}")

    joint_pos_source = np.asarray(motion["joint_pos"], dtype=np.float64)
    frame_count = joint_pos_source.shape[0]
    if insert_frame <= 0 or insert_frame >= frame_count:
        raise ValueError(f"insert_frame must be in [1, {frame_count}), got {insert_frame}")
    inserted_frames = transition_in_frames + hold_frames + transition_out_frames
    terminal_frames = terminal_transition_frames + terminal_hold_frames
    if terminal_transition_frames <= 0 or terminal_hold_frames <= 0:
        raise ValueError("terminal transition and hold frame counts must be positive")
    pre_terminal_frame_count = frame_count + inserted_frames
    output_frame_count = pre_terminal_frame_count + terminal_frames
    if not 0 < direction_switch_frame < output_frame_count:
        raise ValueError("direction_switch_frame must lie inside the output motion")
    entry_distances = np.asarray(
        motion["contact_annotation_limb_distance_thresholds"], dtype=np.float64
    )
    if hand_contact_release_distance < float(np.max(entry_distances[:2])):
        raise ValueError(
            "hand_contact_release_distance must not be smaller than either hand entry distance"
        )

    joint_names = _as_names(motion["joint_names"])
    body_names = _as_names(motion["body_names"])
    root_index = body_names.index("base_link")
    standing_joint_pos, standing_root_pos, standing_root_quat = _default_standing_pose(
        motion, robot_asset, insert_frame
    )

    weights_in = _transition_weights(transition_in_frames)
    weights_out = _transition_weights(transition_out_frames)
    transition_in_joint_pos = _blend(
        joint_pos_source[insert_frame - 1], standing_joint_pos, weights_in
    )
    transition_out_joint_pos = _blend(
        standing_joint_pos, joint_pos_source[insert_frame], weights_out
    )
    inserted_joint_pos = np.concatenate(
        (
            transition_in_joint_pos,
            np.repeat(standing_joint_pos[None], hold_frames, axis=0),
            transition_out_joint_pos,
        ),
        axis=0,
    )
    joint_pos = np.concatenate(
        (joint_pos_source[:insert_frame], inserted_joint_pos, joint_pos_source[insert_frame:]),
        axis=0,
    )

    root_pos_source = np.asarray(motion["body_pos_w"][:, root_index], dtype=np.float64)
    root_quat_source = np.asarray(motion["body_quat_w"][:, root_index], dtype=np.float64)
    inserted_root_pos = np.concatenate(
        (
            _blend(root_pos_source[insert_frame - 1], standing_root_pos, weights_in),
            np.repeat(standing_root_pos[None], hold_frames, axis=0),
            _blend(standing_root_pos, root_pos_source[insert_frame], weights_out),
        ),
        axis=0,
    )
    inserted_root_quat = np.concatenate(
        (
            _slerp(root_quat_source[insert_frame - 1], standing_root_quat, weights_in),
            np.repeat(standing_root_quat[None], hold_frames, axis=0),
            _slerp(standing_root_quat, root_quat_source[insert_frame], weights_out),
        ),
        axis=0,
    )
    inserted_body_pos, inserted_body_quat = forward_kinematics(
        robot_asset,
        inserted_root_pos,
        inserted_root_quat,
        inserted_joint_pos,
        joint_names,
        body_names,
    )
    body_pos_w = np.concatenate(
        (
            np.asarray(motion["body_pos_w"][:insert_frame]),
            inserted_body_pos,
            np.asarray(motion["body_pos_w"][insert_frame:]),
        ),
        axis=0,
    )
    body_quat_w = np.concatenate(
        (
            np.asarray(motion["body_quat_w"][:insert_frame]),
            inserted_body_quat,
            np.asarray(motion["body_quat_w"][insert_frame:]),
        ),
        axis=0,
    )

    terminal_joint_pos, terminal_root_pos, terminal_root_quat = _default_standing_pose(
        motion, robot_asset, frame_count - 1
    )
    if "motion_processing_default_standing_root_pos" in motion:
        terminal_root_pos[2] = float(
            np.asarray(motion["motion_processing_default_standing_root_pos"]).reshape(-1, 3)[0, 2]
        )
    terminal_weights = _transition_weights(terminal_transition_frames)
    terminal_joint_frames = np.concatenate(
        (
            _blend(joint_pos[-1], terminal_joint_pos, terminal_weights),
            np.repeat(terminal_joint_pos[None], terminal_hold_frames, axis=0),
        ),
        axis=0,
    )
    terminal_root_positions = np.concatenate(
        (
            _blend(body_pos_w[-1, root_index], terminal_root_pos, terminal_weights),
            np.repeat(terminal_root_pos[None], terminal_hold_frames, axis=0),
        ),
        axis=0,
    )
    terminal_root_quaternions = np.concatenate(
        (
            _slerp(body_quat_w[-1, root_index], terminal_root_quat, terminal_weights),
            np.repeat(terminal_root_quat[None], terminal_hold_frames, axis=0),
        ),
        axis=0,
    )
    terminal_body_pos, terminal_body_quat = forward_kinematics(
        robot_asset,
        terminal_root_positions,
        terminal_root_quaternions,
        terminal_joint_frames,
        joint_names,
        body_names,
    )
    joint_pos = np.concatenate((joint_pos, terminal_joint_frames), axis=0)
    body_pos_w = np.concatenate((body_pos_w, terminal_body_pos), axis=0)
    body_quat_w = np.concatenate((body_quat_w, terminal_body_quat), axis=0)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])

    output: dict[str, np.ndarray] = {}
    for key, value in motion.items():
        array = np.asarray(value)
        if key in CONTACT_FRAME_FIELDS or key in RESET_FIELDS or key.startswith("reset_"):
            continue
        if key in {
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }:
            continue
        if array.ndim > 0 and array.shape[0] == frame_count:
            output[key] = _append_auxiliary_frame_field(
                _insert_auxiliary_frame_field(array, insert_frame, inserted_frames),
                terminal_frames,
            )
        else:
            output[key] = array.copy()
    output.update(
        {
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": finite_difference(joint_pos, fps),
            "body_pos_w": body_pos_w.astype(np.float32),
            "body_quat_w": body_quat_w.astype(np.float32),
            "body_lin_vel_w": finite_difference(body_pos_w, fps),
            "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
        }
    )
    output.update(
        annotate_ladder_contacts(output, _contact_cfg(motion, hand_contact_release_distance))
    )

    climb_direction = np.ones((output_frame_count, 1), dtype=np.float32)
    climb_direction[direction_switch_frame:] = -1.0
    output["climb_direction"] = climb_direction

    contact_indicator = np.asarray(output["contact_indicator"]) > 0.5
    base_linear_speed = np.linalg.norm(output["body_lin_vel_w"][:, root_index], axis=-1)
    base_angular_speed = np.linalg.norm(output["body_ang_vel_w"][:, root_index], axis=-1)
    max_joint_speed = np.max(np.abs(output["joint_vel"]), axis=-1)
    all_limb_contact = np.all(contact_indicator, axis=1)
    contact_runs = _runs(all_limb_contact)
    selected_contact_frames = np.asarray(
        [
            start + int(np.argmin(base_linear_speed[start : end + 1]))
            for start, end in contact_runs
        ],
        dtype=np.int64,
    )
    selected_contact_mask = np.zeros(output_frame_count, dtype=np.bool_)
    selected_contact_mask[selected_contact_frames] = True

    ground_default = np.zeros(output_frame_count, dtype=np.bool_)
    initial_hold_frames = int(
        np.asarray(motion.get("motion_processing_default_standing_hold_frames", [0])).reshape(-1)[0]
    )
    ground_default[:initial_hold_frames] = True
    top_hold_start = insert_frame + transition_in_frames
    top_hold_end = top_hold_start + hold_frames
    top_default = np.zeros(output_frame_count, dtype=np.bool_)
    top_default[top_hold_start:top_hold_end] = True
    terminal_hold_start = pre_terminal_frame_count + terminal_transition_frames
    terminal_default = np.zeros(output_frame_count, dtype=np.bool_)
    terminal_default[terminal_hold_start:] = True
    zero_command_valid = selected_contact_mask | ground_default | top_default | terminal_default

    contact_run_starts = np.asarray([start for start, _ in contact_runs], dtype=np.int64)
    contact_run_ends = np.asarray([end for _, end in contact_runs], dtype=np.int64)

    output.update(
        {
            "zero_command_valid_mask": zero_command_valid,
            "zero_command_valid_frame_indexes": np.flatnonzero(zero_command_valid).astype(np.int64),
            "zero_command_all_limb_contact_mask": all_limb_contact,
            "zero_command_segment_min_base_speed_mask": selected_contact_mask,
            "zero_command_all_limb_contact_segment_starts": contact_run_starts,
            "zero_command_all_limb_contact_segment_ends": contact_run_ends,
            "zero_command_all_limb_contact_segment_selected_frames": selected_contact_frames,
            "zero_command_all_limb_contact_segment_selected_base_speeds": base_linear_speed[
                selected_contact_frames
            ].astype(np.float32),
            "zero_command_ground_default_standing_mask": ground_default,
            "zero_command_top_default_standing_mask": top_default,
            "zero_command_terminal_default_standing_mask": terminal_default,
            "zero_command_base_linear_speed": base_linear_speed.astype(np.float32),
            "zero_command_base_angular_speed": base_angular_speed.astype(np.float32),
            "zero_command_max_abs_joint_speed": max_joint_speed.astype(np.float32),
            "command_annotation_semantics": np.asarray(
                [
                    "climb_direction: +1=up, -1=down; zero_command_valid_mask contains "
                    "one minimum-base-linear-speed frame per all-limb-contact segment plus default stands; "
                    "runtime effective command becomes 0 only after reaching a valid stop frame"
                ]
            ),
            "command_annotation_source_file": np.asarray([str(source_file)]),
            "command_annotation_insert_source_frame": np.asarray([insert_frame], dtype=np.int64),
            "command_annotation_transition_in_frames": np.asarray(
                [transition_in_frames], dtype=np.int64
            ),
            "command_annotation_hold_frames": np.asarray([hold_frames], dtype=np.int64),
            "command_annotation_transition_out_frames": np.asarray(
                [transition_out_frames], dtype=np.int64
            ),
            "command_annotation_direction_switch_frame": np.asarray(
                [direction_switch_frame], dtype=np.int64
            ),
            "command_annotation_source_frame_offset_after_insert": np.asarray(
                [inserted_frames], dtype=np.int64
            ),
            "command_annotation_terminal_transition_frames": np.asarray(
                [terminal_transition_frames], dtype=np.int64
            ),
            "command_annotation_terminal_hold_frames": np.asarray(
                [terminal_hold_frames], dtype=np.int64
            ),
            "command_annotation_terminal_default_standing_root_pos": terminal_root_pos[
                None
            ].astype(np.float32),
            "command_annotation_terminal_default_standing_root_quat_wxyz": terminal_root_quat[
                None
            ].astype(np.float32),
            "command_annotation_default_standing_root_pos": standing_root_pos[None].astype(
                np.float32
            ),
            "command_annotation_default_standing_root_quat_wxyz": standing_root_quat[
                None
            ].astype(np.float32),
            "zero_command_detection_requires_all_limb_contacts": np.asarray([True]),
            "zero_command_detection_selection_method": np.asarray(
                ["minimum_base_linear_speed_per_contiguous_all_limb_contact_segment"]
            ),
        }
    )
    return output


def main() -> None:
    args = _parse_args()
    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    robot_asset = args.robot_asset.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input motion not found: {input_file}")
    if not robot_asset.is_file():
        raise FileNotFoundError(f"Robot asset not found: {robot_asset}")
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_file}. Pass --overwrite to replace it.")
    if output_file == input_file:
        raise ValueError("output_file must not overwrite input_file")

    with np.load(input_file, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}
    output = build_command_motion(
        motion,
        robot_asset=robot_asset,
        source_file=input_file,
        insert_frame=args.insert_frame,
        transition_in_frames=args.transition_in_frames,
        hold_frames=args.hold_frames,
        transition_out_frames=args.transition_out_frames,
        direction_switch_frame=args.direction_switch_frame,
        hand_contact_release_distance=args.hand_contact_release_distance,
        terminal_transition_frames=args.terminal_transition_frames,
        terminal_hold_frames=args.terminal_hold_frames,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **output)

    selected = np.asarray(output["zero_command_segment_min_base_speed_mask"])
    valid = np.asarray(output["zero_command_valid_mask"])
    print(f"[INFO] Wrote command-annotated motion: {output_file}")
    print(
        f"[INFO] Frames={len(output['joint_pos'])}, fps={float(output['fps'][0]):g}, "
        f"top transition/hold/transition={args.transition_in_frames}/{args.hold_frames}/"
        f"{args.transition_out_frames}, terminal transition/hold="
        f"{args.terminal_transition_frames}/{args.terminal_hold_frames}"
    )
    print(
        f"[INFO] Direction: up=[0,{args.direction_switch_frame - 1}], "
        f"down=[{args.direction_switch_frame},{len(output['joint_pos']) - 1}]"
    )
    print(
        f"[INFO] Four-limb contact segments={int(selected.sum())}; selected frames="
        f"{output['zero_command_all_limb_contact_segment_selected_frames'].tolist()}"
    )
    print(
        f"[INFO] All legal zero-command frames={int(valid.sum())}; runs={_runs(valid)}"
    )


if __name__ == "__main__":
    main()
