"""Insert a CR1 default-standing interval without changing source motion frames."""

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

from ladder.tasks.ladder_climb.ladder_geometry import generate_ladder
from ladder.utils.contact_annotation import (
    ContactPhaseAnnotationCfg,
    DirectionalContactPhaseCfg,
    annotate_directional_ladder_contact_phases,
)
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
KINEMATIC_FRAME_FIELDS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}
STALE_CONTACT_FRAME_FIELDS = {
    "contact_nearest_distance",
    "contact_speed",
    "contact_raw_indicator",
    "contact_hysteresis_filled_indicator",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_CR1_URDF)
    parser.add_argument("--insert_frame", type=int, required=True)
    parser.add_argument("--transition_in_frames", type=int, default=100)
    parser.add_argument("--hold_frames", type=int, default=100)
    parser.add_argument("--transition_out_frames", type=int, default=100)
    parser.add_argument(
        "--standing_root_z_offset",
        type=float,
        default=0.0,
        help="Additional world-Z height at the default stand; transitions use the same quintic blend.",
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


def _transition_weights(frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("transition frame counts must be positive")
    phase = np.arange(1, frame_count + 1, dtype=np.float64) / (frame_count + 1)
    return 6.0 * phase**5 - 15.0 * phase**4 + 10.0 * phase**3


def _blend(start: np.ndarray, end: np.ndarray, weights: np.ndarray) -> np.ndarray:
    shape = (len(weights),) + (1,) * np.asarray(start).ndim
    weight = weights.reshape(shape)
    return (1.0 - weight) * np.asarray(start)[None] + weight * np.asarray(end)[None]


def _slerp(start_wxyz: np.ndarray, end_wxyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    end = np.asarray(end_wxyz, dtype=np.float64)
    if np.dot(start_wxyz, end) < 0.0:
        end = -end
    rotations = Rotation.from_quat(np.stack((start_wxyz, end))[:, [1, 2, 3, 0]])
    quat_xyzw = Slerp([0.0, 1.0], rotations)(weights).as_quat()
    return quat_xyzw[:, [3, 0, 1, 2]]


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


def _insert_auxiliary_field(
    values: np.ndarray, insert_frame: int, inserted_frames: int
) -> np.ndarray:
    values = np.asarray(values)
    inserted = np.repeat(values[insert_frame - 1 : insert_frame], inserted_frames, axis=0)
    return np.concatenate((values[:insert_frame], inserted, values[insert_frame:]), axis=0)


def _phase_cfg(motion: dict[str, np.ndarray]) -> ContactPhaseAnnotationCfg:
    params = np.asarray(motion["contact_annotation_ladder_params"], dtype=np.float64).reshape(4)
    origin = np.asarray(motion["contact_annotation_ladder_origin"], dtype=np.float64).reshape(3)
    temporal = np.asarray(motion["contact_phase_annotation_temporal_params"], dtype=np.int64)
    return ContactPhaseAnnotationCfg(
        ladder_geometry=generate_ladder(
            phi=float(params[0]),
            rung_spacing=float(params[1]),
            rung_length=float(params[2]),
            rung_radius_or_width=float(params[3]),
            num_rungs=int(np.asarray(motion["contact_annotation_ladder_num_rungs"]).reshape(-1)[0]),
            origin=tuple(float(value) for value in origin),
        ),
        limb_names=tuple(_as_names(motion["limb_names"])),
        limb_body_names=tuple(_as_names(motion["limb_body_names"])),
        entry_distance_thresholds=tuple(
            np.asarray(motion["contact_phase_annotation_entry_distance_thresholds"], dtype=float)
        ),
        release_distance_thresholds=tuple(
            np.asarray(motion["contact_phase_annotation_release_distance_thresholds"], dtype=float)
        ),
        entry_min_clearances=tuple(
            np.asarray(motion["contact_phase_annotation_entry_min_clearances"], dtype=float)
        ),
        release_min_clearances=tuple(
            np.asarray(motion["contact_phase_annotation_release_min_clearances"], dtype=float)
        ),
        contact_point_z_offsets=tuple(
            np.asarray(motion["contact_phase_annotation_contact_point_z_offsets"], dtype=float)
        ),
        min_contact_frames=int(temporal[0]),
        max_gap_frames=int(temporal[1]),
    )


def _directional_cfg(
    motion: dict[str, np.ndarray], insert_frame: int, inserted_frames: int
) -> DirectionalContactPhaseCfg:
    source_frames = np.asarray(motion["contact_phase_directional_frames"], dtype=np.int64)
    shifted_frames = source_frames + (source_frames >= insert_frame) * inserted_frames
    return DirectionalContactPhaseCfg(
        limb_names=tuple(_as_names(motion["contact_phase_directional_limb_names"])),
        top_neutral_start_frame=int(shifted_frames[0]),
        descent_start_frame=int(shifted_frames[1]),
        core_distance_thresholds=tuple(
            np.asarray(motion["contact_phase_directional_core_distances"], dtype=float)
        ),
        departure_distance_thresholds=tuple(
            np.asarray(motion["contact_phase_directional_departure_distances"], dtype=float)
        ),
        departure_rate_thresholds=tuple(
            np.asarray(motion["contact_phase_directional_departure_rates"], dtype=float)
        ),
        departure_normal_velocity_thresholds=tuple(
            np.asarray(
                motion["contact_phase_directional_departure_normal_velocities"], dtype=float
            )
        ),
        confirmation_frames=int(
            np.asarray(motion["contact_phase_directional_confirmation_frames"]).reshape(-1)[0]
        ),
    )


def insert_middle_default_stand(
    motion: dict[str, np.ndarray],
    *,
    robot_asset: Path,
    source_file: Path,
    insert_frame: int,
    transition_in_frames: int,
    hold_frames: int,
    transition_out_frames: int,
    standing_root_z_offset: float,
) -> dict[str, np.ndarray]:
    required = (
        "fps",
        "joint_pos",
        "joint_names",
        "body_pos_w",
        "body_quat_w",
        "body_names",
        "contact_indicator",
        "contact_rung_id",
        "contact_phase_directional_frames",
    )
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"motion is missing required keys: {missing}")

    joint_pos_source = np.asarray(motion["joint_pos"], dtype=np.float64)
    frame_count = joint_pos_source.shape[0]
    if not 0 < insert_frame < frame_count:
        raise ValueError(f"insert_frame must lie in [1, {frame_count - 1}]")
    if hold_frames <= 0:
        raise ValueError("hold_frames must be positive")
    inserted_frames = transition_in_frames + hold_frames + transition_out_frames
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    joint_names = _as_names(motion["joint_names"])
    body_names = _as_names(motion["body_names"])
    root_index = body_names.index("base_link")
    standing_joint_pos, standing_root_pos, standing_root_quat = _default_standing_pose(
        motion, robot_asset, insert_frame
    )
    standing_root_pos[2] += standing_root_z_offset

    weights_in = _transition_weights(transition_in_frames)
    weights_out = _transition_weights(transition_out_frames)
    inserted_joint_pos = np.concatenate(
        (
            _blend(joint_pos_source[insert_frame - 1], standing_joint_pos, weights_in),
            np.repeat(standing_joint_pos[None], hold_frames, axis=0),
            _blend(standing_joint_pos, joint_pos_source[insert_frame], weights_out),
        ),
        axis=0,
    )
    source_root_pos = np.asarray(motion["body_pos_w"][:, root_index], dtype=np.float64)
    source_root_quat = np.asarray(motion["body_quat_w"][:, root_index], dtype=np.float64)
    inserted_root_pos = np.concatenate(
        (
            _blend(source_root_pos[insert_frame - 1], standing_root_pos, weights_in),
            np.repeat(standing_root_pos[None], hold_frames, axis=0),
            _blend(standing_root_pos, source_root_pos[insert_frame], weights_out),
        ),
        axis=0,
    )
    inserted_root_quat = np.concatenate(
        (
            _slerp(source_root_quat[insert_frame - 1], standing_root_quat, weights_in),
            np.repeat(standing_root_quat[None], hold_frames, axis=0),
            _slerp(standing_root_quat, source_root_quat[insert_frame], weights_out),
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
    joint_pos = np.concatenate(
        (joint_pos_source[:insert_frame], inserted_joint_pos, joint_pos_source[insert_frame:]),
        axis=0,
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

    output: dict[str, np.ndarray] = {}
    for key, value in motion.items():
        array = np.asarray(value)
        if key in KINEMATIC_FRAME_FIELDS or key in STALE_CONTACT_FRAME_FIELDS:
            continue
        if key.startswith("reset_") or key.startswith("contact_phase_") or key.startswith(
            "contact_surface_"
        ):
            continue
        if array.ndim > 0 and array.shape[0] == frame_count:
            output[key] = _insert_auxiliary_field(array, insert_frame, inserted_frames)
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

    output["contact_indicator"][insert_frame : insert_frame + inserted_frames] = 0.0
    output["contact_rung_id"][insert_frame : insert_frame + inserted_frames] = -1
    phase_labels = annotate_directional_ladder_contact_phases(
        output,
        _phase_cfg(motion),
        _directional_cfg(motion, insert_frame, inserted_frames),
    )
    phase_indicator = np.asarray(phase_labels.pop("contact_phase_indicator"))
    phase_rung_id = np.asarray(phase_labels.pop("contact_phase_rung_id"))
    phase_indicator[:insert_frame] = motion["contact_indicator"][:insert_frame]
    phase_rung_id[:insert_frame] = motion["contact_rung_id"][:insert_frame]
    phase_indicator[insert_frame + inserted_frames :] = motion["contact_indicator"][insert_frame:]
    phase_rung_id[insert_frame + inserted_frames :] = motion["contact_rung_id"][insert_frame:]
    raw_indicator = np.asarray(phase_labels["contact_phase_raw_indicator"]) > 0.5
    phase_labels["contact_phase_hysteresis_filled_indicator"] = (
        (phase_indicator > 0.5) & ~raw_indicator
    ).astype(np.float32)
    output.update(phase_labels)
    output["contact_indicator"] = phase_indicator.astype(np.float32)
    output["contact_rung_id"] = phase_rung_id.astype(np.int64)
    output.update(
        {
            "middle_default_standing_source_file": np.asarray([str(source_file)]),
            "middle_default_standing_insert_source_frame": np.asarray([insert_frame], dtype=np.int64),
            "middle_default_standing_transition_in_frames": np.asarray(
                [transition_in_frames], dtype=np.int64
            ),
            "middle_default_standing_hold_frames": np.asarray([hold_frames], dtype=np.int64),
            "middle_default_standing_transition_out_frames": np.asarray(
                [transition_out_frames], dtype=np.int64
            ),
            "middle_default_standing_root_pos": standing_root_pos[None].astype(np.float32),
            "middle_default_standing_root_quat_wxyz": standing_root_quat[None].astype(np.float32),
            "middle_default_standing_root_z_offset": np.asarray(
                [standing_root_z_offset], dtype=np.float32
            ),
            "middle_default_standing_no_ik": np.asarray([True]),
            "contact_phase_annotation_source_file": np.asarray(str(source_file)),
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
    if input_file == output_file:
        raise ValueError("output_file must not overwrite input_file")

    with np.load(input_file, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}
    output = insert_middle_default_stand(
        motion,
        robot_asset=robot_asset,
        source_file=input_file,
        insert_frame=args.insert_frame,
        transition_in_frames=args.transition_in_frames,
        hold_frames=args.hold_frames,
        transition_out_frames=args.transition_out_frames,
        standing_root_z_offset=args.standing_root_z_offset,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **output)
    directional_frames = output["contact_phase_directional_frames"].tolist()
    print(f"[INFO] Wrote middle default-standing motion: {output_file}")
    print(
        f"[INFO] Frames={len(output['joint_pos'])}, fps={float(output['fps'][0]):g}, "
        f"inserted={args.transition_in_frames}/{args.hold_frames}/{args.transition_out_frames}, "
        f"standing_root_z_offset={args.standing_root_z_offset:.4f} m"
    )
    print(f"[INFO] Directional frames (top-neutral/descent): {directional_frames}")


if __name__ == "__main__":
    main()
