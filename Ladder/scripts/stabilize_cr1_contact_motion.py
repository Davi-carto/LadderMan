"""Repair CR1 stance-foot drift or ground clearance with leg-only IK."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from ladder.utils.urdf_kinematics import finite_difference, forward_kinematics, quaternion_angular_velocity


DEFAULT_ROBOT = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)
FOOT_BODY_NAMES = ("left_sole_contact_point", "right_sole_contact_point")
FOOT_COLLISION_BODY_NAMES = ("left_ankle_x_link", "right_ankle_x_link")
ARM_BODY_NAMES = ("left_sphere_hand_link", "right_sphere_hand_link")
LEG_JOINT_NAMES = (
    (
        "left_hip_y_joint",
        "left_hip_x_joint",
        "left_hip_z_joint",
        "left_knee_joint",
        "left_ankle_y_joint",
        "left_ankle_x_joint",
    ),
    (
        "right_hip_y_joint",
        "right_hip_x_joint",
        "right_hip_z_joint",
        "right_knee_joint",
        "right_ankle_y_joint",
        "right_ankle_x_joint",
    ),
)
ARM_JOINT_NAMES = (
    (
        "left_shoulder_y_joint",
        "left_shoulder_x_joint",
        "left_shoulder_z_joint",
        "left_elbow_joint",
    ),
    (
        "right_shoulder_y_joint",
        "right_shoulder_x_joint",
        "right_shoulder_z_joint",
        "right_elbow_joint",
    ),
)

BLEND_FRAMES = 15
ROOT_MAGNITUDE_WEIGHT = 0.5
ROOT_VELOCITY_WEIGHT = 1.0
ROOT_ACCELERATION_WEIGHT = 10.0
IK_ITERATIONS = 5
IK_DAMPING = 1.0e-4
IK_MAX_STEP = 0.04
IK_CORRECTION_CUTOFF_HZ = 2.5
JACOBIAN_EPSILON = 1.0e-4
HAND_IK_ITERATIONS = 10
HAND_IK_DAMPING = 5.0e-3
HAND_IK_WAIST_SCALE = 0.5
HAND_IK_WAIST_MAX_STEP = 0.008
HAND_IK_ARM_MAX_STEP = 0.02
INITIAL_RIGHT_STANCE_END_FRAME = 85
INITIAL_ROOT_FILTER_END_FRAME = 51
INITIAL_ROOT_FILTER_BLEND_FRAMES = 30
INITIAL_ROOT_XY_CUTOFF_HZ = 0.5
INITIAL_ROOT_XY_CORRECTION_SCALE = 0.75
GROUND_CLEARANCE_MAX_STEP = 0.001
GROUND_CLEARANCE_ANTICIPATION_FRAMES = 15
GROUND_CLEARANCE_SEED_DENOMINATOR = 0.2
GROUND_CLEARANCE_SEED_MAX = 0.7
GROUND_CLEARANCE_CORRECTION_CUTOFF_HZ = 1.5
GROUND_CLEARANCE_ACTIVITY_BLEND_FRAMES = 15


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True, help="Smoothed CR1 motion.")
    parser.add_argument("--output_file", type=Path, required=True, help="Repaired output motion.")
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT, help="CR1 URDF used for offline FK.")
    parser.add_argument(
        "--refine_initial_right_stance",
        action="store_true",
        help="Reduce initial root XY sway and stabilize the right sole on the ground through frame 84.",
    )
    parser.add_argument(
        "--leg_only_xy",
        action="store_true",
        help="Keep the root and sole Z unchanged; reduce stance-foot XY drift using only the corresponding leg.",
    )
    parser.add_argument(
        "--leg_only_xyz",
        action="store_true",
        help="Keep the root unchanged; reduce stance-foot XYZ drift using only the corresponding leg.",
    )
    parser.add_argument(
        "--whole_body_xyz",
        action="store_true",
        help=(
            "Share stance-foot XYZ correction with root translation and restore the original sphere-hand "
            "positions with arm IK. Root orientation and hand targets remain unchanged."
        ),
    )
    parser.add_argument(
        "--include_ground_support",
        action="store_true",
        help="Add near-ground sole frames to the annotated ladder-foot support mask.",
    )
    parser.add_argument(
        "--ground_support_max_height",
        type=float,
        default=0.07,
        help="Maximum sole height above --ground_z included by --include_ground_support.",
    )
    parser.add_argument(
        "--ground_clearance_hybrid",
        action="store_true",
        help=(
            "Repair foot/ground penetration using a disclosed root-Z schedule plus leg-only IK; "
            "arms and original sole XY targets remain unchanged."
        ),
    )
    parser.add_argument(
        "--ground_z",
        type=float,
        default=-0.03,
        help="Ground-plane height used by --ground_clearance_hybrid.",
    )
    parser.add_argument(
        "--ground_clearance_margin",
        type=float,
        default=0.001,
        help="Requested foot-collision clearance above ground in meters.",
    )
    parser.add_argument(
        "--ground_start_root_full_until",
        type=int,
        default=None,
        help="Last initial frame using the full shared root-Z clearance.",
    )
    parser.add_argument(
        "--ground_start_root_zero_at",
        type=int,
        default=None,
        help="Initial frame at which root-Z clearance reaches zero.",
    )
    parser.add_argument(
        "--ground_end_root_zero_until",
        type=int,
        default=None,
        help="Last terminal frame at which root-Z clearance remains zero.",
    )
    parser.add_argument(
        "--ground_end_root_full_at",
        type=int,
        default=None,
        help="Terminal frame at which the full shared root-Z clearance is restored.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_file if it exists.")
    return parser.parse_args()


def _as_names(values: np.ndarray) -> list[str]:
    items = np.asarray(values).reshape(-1).tolist()
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _contact_anchors_and_weights(
    foot_positions: np.ndarray,
    contact_indicator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = np.zeros_like(foot_positions, dtype=np.float64)
    weights = np.zeros(contact_indicator.shape, dtype=np.float64)
    for foot_index in range(len(FOOT_BODY_NAMES)):
        for start, end in _runs(contact_indicator[:, foot_index]):
            anchors[start:end, foot_index] = np.median(foot_positions[start:end, foot_index], axis=0)
            run_length = end - start
            blend_frames = min(BLEND_FRAMES, max(2, run_length // 3))
            blend_up = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, blend_frames))
            run_weights = np.ones(run_length, dtype=np.float64)
            if start > 0:
                run_weights[:blend_frames] = blend_up
            if end < contact_indicator.shape[0]:
                run_weights[-blend_frames:] = np.minimum(run_weights[-blend_frames:], blend_up[::-1])
            weights[start:end, foot_index] = run_weights
    return anchors, weights


def _add_ground_support_contacts(
    contact_indicator: np.ndarray,
    foot_positions: np.ndarray,
    ground_z: float,
    maximum_height: float,
) -> np.ndarray:
    contact_indicator = np.asarray(contact_indicator, dtype=bool)
    foot_positions = np.asarray(foot_positions, dtype=np.float64)
    if contact_indicator.shape != foot_positions.shape[:2]:
        raise ValueError(
            "contact indicator and foot positions must have matching frame/foot dimensions"
        )
    if maximum_height < 0.0:
        raise ValueError("ground support maximum height must be non-negative")
    return contact_indicator | (foot_positions[..., 2] <= ground_z + maximum_height)


def _slope_limited_clearance(
    required: np.ndarray,
    max_step: float,
    anticipation_frames: int | None = None,
) -> np.ndarray:
    """Return a clearance envelope with bounded decay and finite look-ahead."""

    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    required = np.maximum(np.asarray(required, dtype=np.float64), 0.0)
    envelope = required.copy()
    if anticipation_frames is None:
        anticipation_frames = len(envelope) - 1
    if anticipation_frames < 0:
        raise ValueError("anticipation_frames must be non-negative")
    for offset in range(1, min(anticipation_frames, len(envelope) - 1) + 1):
        envelope[:-offset] = np.maximum(
            envelope[:-offset], required[offset:] - max_step * offset
        )
    for frame in range(1, len(envelope)):
        envelope[frame] = max(envelope[frame], envelope[frame - 1] - max_step)
    return envelope


def _foot_collision_vertices(robot_asset: Path) -> tuple[np.ndarray, ...]:
    try:
        import trimesh
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ground-clearance repair requires trimesh in the active environment"
        ) from exc

    root = ElementTree.parse(robot_asset).getroot()
    vertices = []
    for body_name in FOOT_COLLISION_BODY_NAMES:
        mesh = root.find(f"./link[@name='{body_name}']/collision/geometry/mesh")
        if mesh is None or "filename" not in mesh.attrib:
            raise ValueError(f"link '{body_name}' does not have a collision mesh")
        mesh_path = (robot_asset.parent / mesh.attrib["filename"]).resolve()
        vertices.append(
            np.asarray(trimesh.load(mesh_path, force="mesh", process=False).vertices, dtype=np.float64)
        )
    return tuple(vertices)


def _foot_collision_minimum_z(
    robot_asset: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    source_joint_pos: np.ndarray,
    source_joint_names: list[str],
    collision_vertices: tuple[np.ndarray, ...],
) -> np.ndarray:
    positions, quaternions = forward_kinematics(
        robot_asset,
        root_pos,
        root_quat,
        source_joint_pos,
        source_joint_names,
        ["base_link", *FOOT_COLLISION_BODY_NAMES],
    )
    minimum_z = np.empty((source_joint_pos.shape[0], len(FOOT_COLLISION_BODY_NAMES)))
    for foot_index, vertices in enumerate(collision_vertices):
        rotations = Rotation.from_quat(quaternions[:, foot_index + 1, [1, 2, 3, 0]]).as_matrix()
        for frame, world_z_axis_in_body in enumerate(rotations[:, 2, :]):
            minimum_z[frame, foot_index] = (
                positions[frame, foot_index + 1, 2]
                + np.min(vertices @ world_z_axis_in_body)
            )
    return minimum_z


def _ground_clearance_anchors_and_weights(
    foot_positions: np.ndarray,
    minimum_collision_z: np.ndarray,
    ground_z: float,
    clearance_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = np.maximum(ground_z + clearance_margin - minimum_collision_z, 0.0)
    clearance = np.stack(
        [
            _slope_limited_clearance(
                required[:, foot_index],
                GROUND_CLEARANCE_MAX_STEP,
                GROUND_CLEARANCE_ANTICIPATION_FRAMES,
            )
            for foot_index in range(required.shape[1])
        ],
        axis=1,
    )
    anchors = np.asarray(foot_positions, dtype=np.float64).copy()
    anchors[..., 2] += clearance
    weights = (clearance > 1.0e-9).astype(np.float64)
    return anchors, weights, clearance


def _ground_root_blend(
    frame_count: int,
    start_full_until: int,
    start_zero_at: int,
    end_zero_until: int,
    end_full_at: int,
) -> np.ndarray:
    if not 0 <= start_full_until < start_zero_at <= end_zero_until < end_full_at < frame_count:
        raise ValueError(
            "ground root schedule must satisfy 0 <= start_full_until < start_zero_at "
            "<= end_zero_until < end_full_at < frame_count"
        )
    blend = np.zeros(frame_count, dtype=np.float64)
    blend[: start_full_until + 1] = 1.0
    start_phase = np.linspace(0.0, np.pi, start_zero_at - start_full_until + 1)
    blend[start_full_until : start_zero_at + 1] = 0.5 + 0.5 * np.cos(start_phase)
    end_phase = np.linspace(0.0, np.pi, end_full_at - end_zero_until + 1)
    blend[end_zero_until : end_full_at + 1] = 0.5 - 0.5 * np.cos(end_phase)
    blend[end_full_at:] = 1.0
    return blend


def _quintic_blend(frame_count: int) -> np.ndarray:
    if frame_count < 2:
        raise ValueError("quintic blend requires at least two frames")
    phase = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    return 6.0 * phase**5 - 15.0 * phase**4 + 10.0 * phase**3


def _ground_clearance_activity_weights(
    minimum_collision_z: np.ndarray,
    ground_z: float,
    blend_frames: int,
) -> np.ndarray:
    if blend_frames < 1:
        raise ValueError("blend_frames must be positive")
    penetration = np.asarray(minimum_collision_z, dtype=np.float64) < ground_z
    weights = np.zeros_like(minimum_collision_z, dtype=np.float64)
    for foot_index in range(penetration.shape[1]):
        for start, end in _runs(penetration[:, foot_index]):
            if start == 0:
                weights[start:end, foot_index] = 1.0
            else:
                blend_start = max(0, start - blend_frames)
                blend = _quintic_blend(start - blend_start + 1)
                weights[blend_start : start + 1, foot_index] = np.maximum(
                    weights[blend_start : start + 1, foot_index], blend
                )
                weights[start:end, foot_index] = 1.0
            if end < penetration.shape[0]:
                blend_end = min(penetration.shape[0] - 1, end + blend_frames)
                blend = _quintic_blend(blend_end - end + 1)[::-1]
                weights[end : blend_end + 1, foot_index] = np.maximum(
                    weights[end : blend_end + 1, foot_index], blend
                )
    return weights


def _ground_clearance_seed(
    robot_asset: Path,
    source_joint_pos: np.ndarray,
    source_joint_names: list[str],
    remaining_clearance: np.ndarray,
) -> np.ndarray:
    joints = {
        joint.attrib["name"]: joint
        for joint in ElementTree.parse(robot_asset).getroot().findall("joint")
    }
    seeded = source_joint_pos.copy()
    for foot_index, names in enumerate(LEG_JOINT_NAMES):
        hip_index = source_joint_names.index(names[0])
        knee_index = source_joint_names.index(names[3])
        ankle_index = source_joint_names.index(names[4])
        amplitude = np.minimum(
            np.sqrt(
                np.maximum(remaining_clearance[:, foot_index], 0.0)
                / GROUND_CLEARANCE_SEED_DENOMINATOR
            ),
            GROUND_CLEARANCE_SEED_MAX,
        )
        seeded[:, hip_index] -= 0.5 * amplitude
        seeded[:, knee_index] += amplitude
        seeded[:, ankle_index] -= 0.5 * amplitude
        for joint_index in (hip_index, knee_index, ankle_index):
            limit = joints[source_joint_names[joint_index]].find("limit")
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
            seeded[:, joint_index] = np.clip(
                seeded[:, joint_index], lower + 1.0e-6, upper - 1.0e-6
            )
    return seeded


def _filter_ground_clearance_correction(
    source_joint_pos: np.ndarray,
    corrected_joint_pos: np.ndarray,
    source_joint_names: list[str],
    activity_weights: np.ndarray,
    fps: float,
) -> np.ndarray:
    correction_filter = butter(
        4, GROUND_CLEARANCE_CORRECTION_CUTOFF_HZ, fs=fps, output="sos"
    )
    correction = sosfiltfilt(
        correction_filter, corrected_joint_pos - source_joint_pos, axis=0
    )
    for foot_index, names in enumerate(LEG_JOINT_NAMES):
        joint_indexes = [source_joint_names.index(name) for name in names]
        correction[:, joint_indexes] *= activity_weights[:, foot_index, None]
    return source_joint_pos + correction


def _refine_initial_root_xy(root_pos: np.ndarray, fps: float) -> np.ndarray:
    required_frames = INITIAL_ROOT_FILTER_END_FRAME + INITIAL_ROOT_FILTER_BLEND_FRAMES
    if root_pos.shape[0] < required_frames:
        raise ValueError(
            f"initial stance refinement requires at least {required_frames} frames, got {root_pos.shape[0]}"
        )

    root_filter = butter(4, INITIAL_ROOT_XY_CUTOFF_HZ, fs=fps, output="sos")
    filtered_xy = sosfiltfilt(root_filter, root_pos[:, :2], axis=0)
    correction = filtered_xy - root_pos[:, :2]
    correction = (correction - correction[0]) * INITIAL_ROOT_XY_CORRECTION_SCALE

    blend = np.zeros(root_pos.shape[0], dtype=np.float64)
    blend[:INITIAL_ROOT_FILTER_END_FRAME] = 1.0
    blend_end = INITIAL_ROOT_FILTER_END_FRAME + INITIAL_ROOT_FILTER_BLEND_FRAMES
    blend[INITIAL_ROOT_FILTER_END_FRAME:blend_end] = 0.5 + 0.5 * np.cos(
        np.linspace(0.0, np.pi, INITIAL_ROOT_FILTER_BLEND_FRAMES)
    )

    result = root_pos.copy()
    result[:, :2] += correction * blend[:, None]
    return result


def _refine_initial_right_stance(
    robot_asset: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    source_joint_pos: np.ndarray,
    source_joint_names: list[str],
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if root_pos.shape[0] < INITIAL_RIGHT_STANCE_END_FRAME:
        raise ValueError(
            "initial right-stance refinement requires at least "
            f"{INITIAL_RIGHT_STANCE_END_FRAME} frames, got {root_pos.shape[0]}"
        )

    hand_targets = _arm_positions(
        robot_asset, root_pos, root_quat, source_joint_pos, source_joint_names
    )
    refined_root_pos = _refine_initial_root_xy(root_pos, fps)
    foot_positions = _foot_positions(
        robot_asset, refined_root_pos, root_quat, source_joint_pos, source_joint_names
    )
    initial_contact = np.zeros((root_pos.shape[0], len(FOOT_BODY_NAMES)), dtype=bool)
    initial_contact[:INITIAL_RIGHT_STANCE_END_FRAME, 1] = True
    anchors, weights = _contact_anchors_and_weights(foot_positions, initial_contact)

    # The requested cleanup is planar; retain the reference sole height instead of locking it.
    anchors[:INITIAL_RIGHT_STANCE_END_FRAME, 1, 2] = foot_positions[
        :INITIAL_RIGHT_STANCE_END_FRAME, 1, 2
    ]
    refined_source_joint_pos = _apply_leg_ik(
        robot_asset,
        refined_root_pos,
        root_quat,
        source_joint_pos,
        source_joint_names,
        anchors,
        weights,
        fps,
    )
    refined_source_joint_pos = _apply_arm_ik(
        robot_asset,
        refined_root_pos,
        root_quat,
        refined_source_joint_pos,
        source_joint_names,
        hand_targets,
        fps,
    )
    return refined_root_pos, refined_source_joint_pos, weights


def _solve_root_translation(
    foot_positions: np.ndarray,
    anchors: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    frame_count = foot_positions.shape[0]
    first_difference = diags(
        (-np.ones(frame_count - 1), np.ones(frame_count - 1)),
        (0, 1),
        shape=(frame_count - 1, frame_count),
        format="csc",
    )
    second_difference = diags(
        (np.ones(frame_count - 2), -2.0 * np.ones(frame_count - 2), np.ones(frame_count - 2)),
        (0, 1, 2),
        shape=(frame_count - 2, frame_count),
        format="csc",
    )
    system = (
        diags(weights.sum(axis=1) + ROOT_MAGNITUDE_WEIGHT, 0, format="csc")
        + ROOT_VELOCITY_WEIGHT * (first_difference.T @ first_difference)
        + ROOT_ACCELERATION_WEIGHT * (second_difference.T @ second_difference)
    )
    target = np.sum(weights[..., None] * (anchors - foot_positions), axis=1)
    return np.stack([spsolve(system, target[:, axis]) for axis in range(3)], axis=1)


def _foot_positions(
    robot_asset: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    source_joint_pos: np.ndarray,
    source_joint_names: list[str],
) -> np.ndarray:
    positions, _ = forward_kinematics(
        robot_asset,
        root_pos,
        root_quat,
        source_joint_pos,
        source_joint_names,
        ["base_link", *FOOT_BODY_NAMES],
    )
    return positions[:, 1:]


def _arm_positions(
    robot_asset: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    source_joint_pos: np.ndarray,
    source_joint_names: list[str],
) -> np.ndarray:
    positions, _ = forward_kinematics(
        robot_asset,
        root_pos,
        root_quat,
        source_joint_pos,
        source_joint_names,
        ["base_link", *ARM_BODY_NAMES],
    )
    return positions[:, 1:]


def _apply_leg_ik(
    robot_asset: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    source_joint_pos: np.ndarray,
    source_joint_names: list[str],
    anchors: np.ndarray,
    weights: np.ndarray,
    fps: float,
) -> np.ndarray:
    result = source_joint_pos.copy()
    leg_indexes = [[source_joint_names.index(name) for name in names] for names in LEG_JOINT_NAMES]
    correction_filter = butter(4, IK_CORRECTION_CUTOFF_HZ, fs=fps, output="sos")

    for _ in range(IK_ITERATIONS):
        current_feet = _foot_positions(robot_asset, root_pos, root_quat, result, source_joint_names)
        for foot_index, joint_indexes in enumerate(leg_indexes):
            active_frames = np.flatnonzero(weights[:, foot_index] > 0.0)
            if active_frames.size == 0:
                continue
            error = anchors[:, foot_index] - current_feet[:, foot_index]
            jacobian = np.zeros((result.shape[0], 3, len(joint_indexes)), dtype=np.float64)
            for column, joint_index in enumerate(joint_indexes):
                perturbed = result.copy()
                perturbed[:, joint_index] += JACOBIAN_EPSILON
                perturbed_feet = _foot_positions(
                    robot_asset, root_pos, root_quat, perturbed, source_joint_names
                )
                jacobian[:, :, column] = (
                    perturbed_feet[:, foot_index] - current_feet[:, foot_index]
                ) / JACOBIAN_EPSILON

            raw_correction = np.zeros((result.shape[0], len(joint_indexes)), dtype=np.float64)
            for frame in active_frames:
                frame_jacobian = jacobian[frame]
                damped_system = frame_jacobian @ frame_jacobian.T + IK_DAMPING * np.eye(3)
                correction = frame_jacobian.T @ np.linalg.solve(damped_system, error[frame])
                raw_correction[frame] = (
                    np.clip(correction, -IK_MAX_STEP, IK_MAX_STEP) * weights[frame, foot_index]
                )

            for column, joint_index in enumerate(joint_indexes):
                smooth_correction = sosfiltfilt(correction_filter, raw_correction[:, column])
                result[:, joint_index] += smooth_correction * weights[:, foot_index]
    return result


def _apply_arm_ik(
    robot_asset: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    source_joint_pos: np.ndarray,
    source_joint_names: list[str],
    hand_targets: np.ndarray,
    fps: float,
) -> np.ndarray:
    result = source_joint_pos.copy()
    arm_indexes = [[source_joint_names.index(name) for name in names] for names in ARM_JOINT_NAMES]
    correction_filter = butter(4, IK_CORRECTION_CUTOFF_HZ, fs=fps, output="sos")

    for _ in range(IK_ITERATIONS):
        current_hands = _arm_positions(robot_asset, root_pos, root_quat, result, source_joint_names)
        for arm_index, joint_indexes in enumerate(arm_indexes):
            error = hand_targets[:, arm_index] - current_hands[:, arm_index]
            jacobian = np.zeros((result.shape[0], 3, len(joint_indexes)), dtype=np.float64)
            for column, joint_index in enumerate(joint_indexes):
                perturbed = result.copy()
                perturbed[:, joint_index] += JACOBIAN_EPSILON
                perturbed_hands = _arm_positions(
                    robot_asset, root_pos, root_quat, perturbed, source_joint_names
                )
                jacobian[:, :, column] = (
                    perturbed_hands[:, arm_index] - current_hands[:, arm_index]
                ) / JACOBIAN_EPSILON

            raw_correction = np.zeros((result.shape[0], len(joint_indexes)), dtype=np.float64)
            for frame in range(result.shape[0]):
                frame_jacobian = jacobian[frame]
                damped_system = frame_jacobian @ frame_jacobian.T + IK_DAMPING * np.eye(3)
                correction = frame_jacobian.T @ np.linalg.solve(damped_system, error[frame])
                raw_correction[frame] = np.clip(correction, -IK_MAX_STEP, IK_MAX_STEP)

            for column, joint_index in enumerate(joint_indexes):
                result[:, joint_index] += sosfiltfilt(correction_filter, raw_correction[:, column])
    return result


def _apply_upper_body_hand_ik(
    robot_asset: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    source_joint_pos: np.ndarray,
    source_joint_names: list[str],
    hand_targets: np.ndarray,
) -> np.ndarray:
    """Restore both sphere-hand positions using coupled waist and arm corrections."""

    result = source_joint_pos.copy()
    joint_names = (
        "waist_z_joint",
        "waist_x_joint",
        "waist_y_joint",
        *(name for names in ARM_JOINT_NAMES for name in names),
    )
    joint_indexes = [source_joint_names.index(name) for name in joint_names]
    joint_scales = np.asarray([HAND_IK_WAIST_SCALE] * 3 + [1.0] * 8)
    maximum_steps = np.asarray(
        [HAND_IK_WAIST_MAX_STEP] * 3 + [HAND_IK_ARM_MAX_STEP] * 8
    )
    identity = np.eye(6)

    for _ in range(HAND_IK_ITERATIONS):
        current_hands = _arm_positions(
            robot_asset, root_pos, root_quat, result, source_joint_names
        )
        error = (hand_targets - current_hands).reshape(result.shape[0], 6)
        jacobian = np.empty(
            (result.shape[0], 6, len(joint_indexes)), dtype=np.float64
        )
        for column, joint_index in enumerate(joint_indexes):
            perturbed = result.copy()
            perturbed[:, joint_index] += JACOBIAN_EPSILON
            perturbed_hands = _arm_positions(
                robot_asset, root_pos, root_quat, perturbed, source_joint_names
            )
            jacobian[:, :, column] = (
                (perturbed_hands - current_hands) / JACOBIAN_EPSILON
            ).reshape(result.shape[0], 6)

        scaled_jacobian = jacobian * joint_scales[None, None, :]
        damped_system = (
            scaled_jacobian @ np.swapaxes(scaled_jacobian, 1, 2)
            + HAND_IK_DAMPING * identity[None]
        )
        scaled_correction = np.einsum(
            "nij,nj->ni",
            np.swapaxes(scaled_jacobian, 1, 2),
            np.linalg.solve(damped_system, error),
        )
        correction = scaled_correction * joint_scales[None]
        result[:, joint_indexes] += np.clip(
            correction, -maximum_steps, maximum_steps
        )
    return result


def _validate_joint_limits(
    robot_asset: Path,
    source_joint_pos: np.ndarray,
    source_joint_names: list[str],
) -> None:
    joints = {joint.attrib["name"]: joint for joint in ElementTree.parse(robot_asset).getroot().findall("joint")}
    violations = []
    for index, name in enumerate(source_joint_names):
        limit = joints[name].find("limit")
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        if np.any((source_joint_pos[:, index] < lower) | (source_joint_pos[:, index] > upper)):
            violations.append(name)
    if violations:
        raise ValueError(f"contact-aware IK exceeded URDF limits: {violations}")


def _validate_no_new_joint_limit_violations(
    robot_asset: Path,
    source_joint_pos: np.ndarray,
    corrected_joint_pos: np.ndarray,
    source_joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    joints = {
        joint.attrib["name"]: joint
        for joint in ElementTree.parse(robot_asset).getroot().findall("joint")
    }
    source_excess = np.zeros(len(source_joint_names), dtype=np.float64)
    corrected_excess = np.zeros(len(source_joint_names), dtype=np.float64)
    for index, name in enumerate(source_joint_names):
        limit = joints[name].find("limit")
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        source_excess[index] = max(
            float(np.max(lower - source_joint_pos[:, index])),
            float(np.max(source_joint_pos[:, index] - upper)),
            0.0,
        )
        corrected_excess[index] = max(
            float(np.max(lower - corrected_joint_pos[:, index])),
            float(np.max(corrected_joint_pos[:, index] - upper)),
            0.0,
        )
    worsened = corrected_excess > source_excess + 1.0e-7
    if np.any(worsened):
        details = {
            source_joint_names[index]: float(corrected_excess[index])
            for index in np.flatnonzero(worsened)
        }
        raise ValueError(f"ground-clearance repair worsened joint-limit violations: {details}")
    return source_excess, corrected_excess


def _maximum_contact_range(
    foot_positions: np.ndarray,
    contact_indicator: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    ranges = []
    for foot_index in range(len(FOOT_BODY_NAMES)):
        for start, end in _runs(contact_indicator[:, foot_index]):
            indexes = np.arange(start, end)
            if weights is not None:
                indexes = indexes[weights[start:end, foot_index] > 0.9]
            if indexes.size > 1:
                ranges.append(np.ptp(foot_positions[indexes, foot_index], axis=0))
    return np.max(ranges, axis=0) if ranges else np.zeros(3)


def main() -> None:
    args = _parse_args()
    contact_modes = (args.leg_only_xy, args.leg_only_xyz, args.whole_body_xyz)
    if sum(contact_modes) > 1:
        raise ValueError("select at most one contact-stabilization mode")
    if any(contact_modes) and args.refine_initial_right_stance:
        raise ValueError(
            "contact stabilization cannot be combined with --refine_initial_right_stance"
        )
    if args.ground_clearance_hybrid and (
        any(contact_modes) or args.refine_initial_right_stance
    ):
        raise ValueError(
            "--ground_clearance_hybrid cannot be combined with contact-stabilization modes"
        )
    if args.ground_clearance_hybrid and args.include_ground_support:
        raise ValueError("--include_ground_support is only for contact stabilization")
    if args.ground_clearance_margin < 0.0:
        raise ValueError("--ground_clearance_margin must be non-negative")
    if args.ground_support_max_height < 0.0:
        raise ValueError("--ground_support_max_height must be non-negative")
    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    robot_asset = args.robot_asset.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    if not robot_asset.is_file():
        raise FileNotFoundError(robot_asset)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_file}. Pass --overwrite to replace it.")

    with np.load(input_file, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}
    required = [
        "fps",
        "joint_pos",
        "joint_names",
        "source_joint_names",
        "body_pos_w",
        "body_quat_w",
        "body_names",
    ]
    if not args.ground_clearance_hybrid:
        required.extend(("contact_indicator", "limb_names"))
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"input motion is missing keys: {missing}")
    if not args.ground_clearance_hybrid and _as_names(motion["limb_names"]) != [
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
    ]:
        raise ValueError("expected canonical CR1 limb order")

    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    joint_names = _as_names(motion["joint_names"])
    source_joint_names = _as_names(motion["source_joint_names"])
    body_names = _as_names(motion["body_names"])
    root_index = body_names.index("base_link")
    root_pos = np.asarray(motion["body_pos_w"][:, root_index], dtype=np.float64)
    root_quat = np.asarray(motion["body_quat_w"][:, root_index], dtype=np.float64)
    source_joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)[
        :, [joint_names.index(name) for name in source_joint_names]
    ]
    root_blend = None
    root_clearance_z = None
    original_feet = _foot_positions(robot_asset, root_pos, root_quat, source_joint_pos, source_joint_names)
    original_hands = (
        _arm_positions(robot_asset, root_pos, root_quat, source_joint_pos, source_joint_names)
        if args.whole_body_xyz
        else None
    )
    original_minimum_collision_z = None
    ground_clearance = None
    ground_activity_weights = None
    collision_vertices = None
    if args.ground_clearance_hybrid:
        schedule = (
            args.ground_start_root_full_until,
            args.ground_start_root_zero_at,
            args.ground_end_root_zero_until,
            args.ground_end_root_full_at,
        )
        if any(value is None for value in schedule):
            raise ValueError(
                "--ground_clearance_hybrid requires all four --ground_*root* frame arguments"
            )
        collision_vertices = _foot_collision_vertices(robot_asset)
        original_minimum_collision_z = _foot_collision_minimum_z(
            robot_asset,
            root_pos,
            root_quat,
            source_joint_pos,
            source_joint_names,
            collision_vertices,
        )
        anchors, weights, ground_clearance = _ground_clearance_anchors_and_weights(
            original_feet,
            original_minimum_collision_z,
            args.ground_z,
            args.ground_clearance_margin,
        )
        ground_activity_weights = _ground_clearance_activity_weights(
            original_minimum_collision_z,
            args.ground_z,
            GROUND_CLEARANCE_ACTIVITY_BLEND_FRAMES,
        )
        root_blend = _ground_root_blend(source_joint_pos.shape[0], *schedule)
        root_clearance_z = root_blend * np.min(ground_clearance, axis=1)
        corrected_root_pos = root_pos.copy()
        corrected_root_pos[:, 2] += root_clearance_z
        remaining_clearance = np.maximum(
            ground_clearance - root_clearance_z[:, None], 0.0
        )
        ik_source_joint_pos = _ground_clearance_seed(
            robot_asset,
            source_joint_pos,
            source_joint_names,
            remaining_clearance,
        )
        foot_contact = None
    else:
        ik_source_joint_pos = source_joint_pos
        foot_contact = np.asarray(motion["contact_indicator"][:, 2:4] > 0.5)
        if args.include_ground_support:
            foot_contact = _add_ground_support_contacts(
                foot_contact,
                original_feet,
                args.ground_z,
                args.ground_support_max_height,
            )
        anchors, weights = _contact_anchors_and_weights(original_feet, foot_contact)
    if args.leg_only_xy:
        anchors[..., 2] = original_feet[..., 2]
        corrected_root_pos = root_pos.copy()
    elif args.leg_only_xyz:
        corrected_root_pos = root_pos.copy()
    elif not args.ground_clearance_hybrid:
        root_correction = _solve_root_translation(original_feet, anchors, weights)
        corrected_root_pos = root_pos + root_correction
    corrected_source_joint_pos = _apply_leg_ik(
        robot_asset,
        corrected_root_pos,
        root_quat,
        ik_source_joint_pos,
        source_joint_names,
        anchors,
        weights,
        fps,
    )
    if args.whole_body_xyz:
        corrected_source_joint_pos = _apply_upper_body_hand_ik(
            robot_asset,
            corrected_root_pos,
            root_quat,
            corrected_source_joint_pos,
            source_joint_names,
            original_hands,
        )
    if args.ground_clearance_hybrid:
        corrected_source_joint_pos = _filter_ground_clearance_correction(
            source_joint_pos,
            corrected_source_joint_pos,
            source_joint_names,
            ground_activity_weights,
            fps,
        )
    initial_stance_weights = None
    if args.refine_initial_right_stance:
        corrected_root_pos, corrected_source_joint_pos, initial_stance_weights = _refine_initial_right_stance(
            robot_asset,
            corrected_root_pos,
            root_quat,
            corrected_source_joint_pos,
            source_joint_names,
            fps,
        )
    source_limit_excess = None
    corrected_limit_excess = None
    if args.ground_clearance_hybrid or args.leg_only_xyz or args.whole_body_xyz:
        source_limit_excess, corrected_limit_excess = _validate_no_new_joint_limit_violations(
            robot_asset,
            source_joint_pos,
            corrected_source_joint_pos,
            source_joint_names,
        )
    else:
        _validate_joint_limits(robot_asset, corrected_source_joint_pos, source_joint_names)

    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset,
        corrected_root_pos,
        root_quat,
        corrected_source_joint_pos,
        source_joint_names,
        body_names,
    )
    corrected_joint_pos = corrected_source_joint_pos[
        :, [source_joint_names.index(name) for name in joint_names]
    ].astype(np.float32)
    corrected_hands = (
        body_pos_w[:, [body_names.index(name) for name in ARM_BODY_NAMES]]
        if args.whole_body_xyz
        else None
    )
    total_root_correction = corrected_root_pos - root_pos
    total_joint_correction = corrected_source_joint_pos - source_joint_pos
    kinematic_output = {
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_pos": corrected_joint_pos,
        "joint_vel": finite_difference(corrected_joint_pos, fps),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": finite_difference(body_pos_w, fps),
        "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "source_joint_names": np.asarray(source_joint_names),
    }
    if args.ground_clearance_hybrid:
        corrected_minimum_collision_z = _foot_collision_minimum_z(
            robot_asset,
            corrected_root_pos,
            root_quat,
            corrected_source_joint_pos,
            source_joint_names,
            collision_vertices,
        )
        output = {key: np.asarray(value).copy() for key, value in motion.items()}
        output.update(kinematic_output)
        output.update(
            {
                "motion_processing_ground_clearance_source_file": np.asarray([str(input_file)]),
                "motion_processing_ground_clearance_ground_z": np.asarray(
                    [args.ground_z], dtype=np.float32
                ),
                "motion_processing_ground_clearance_margin": np.asarray(
                    [args.ground_clearance_margin], dtype=np.float32
                ),
                "motion_processing_ground_clearance_max_step": np.asarray(
                    [GROUND_CLEARANCE_MAX_STEP], dtype=np.float32
                ),
                "motion_processing_ground_clearance_anticipation_frames": np.asarray(
                    [GROUND_CLEARANCE_ANTICIPATION_FRAMES], dtype=np.int64
                ),
                "motion_processing_ground_clearance_ik_params": np.asarray(
                    [
                        IK_ITERATIONS,
                        IK_DAMPING,
                        IK_MAX_STEP,
                        IK_CORRECTION_CUTOFF_HZ,
                        GROUND_CLEARANCE_SEED_DENOMINATOR,
                        GROUND_CLEARANCE_SEED_MAX,
                        GROUND_CLEARANCE_CORRECTION_CUTOFF_HZ,
                        GROUND_CLEARANCE_ACTIVITY_BLEND_FRAMES,
                    ],
                    dtype=np.float32,
                ),
                "motion_processing_ground_clearance_root_schedule": np.asarray(
                    schedule, dtype=np.int64
                ),
                "motion_processing_ground_clearance_root_blend": root_blend.astype(
                    np.float32
                ),
                "motion_processing_ground_clearance_root_z": root_clearance_z.astype(
                    np.float32
                ),
                "motion_processing_ground_clearance_activity_weights": (
                    ground_activity_weights.astype(np.float32)
                ),
                "motion_processing_ground_clearance_requested_z": ground_clearance.astype(
                    np.float32
                ),
                "motion_processing_ground_clearance_minimum_z_before": (
                    original_minimum_collision_z.astype(np.float32)
                ),
                "motion_processing_ground_clearance_minimum_z_after": (
                    corrected_minimum_collision_z.astype(np.float32)
                ),
                "motion_processing_ground_clearance_joint_delta_rms": np.asarray(
                    [np.sqrt(np.mean(np.square(total_joint_correction)))], dtype=np.float32
                ),
                "motion_processing_ground_clearance_joint_delta_max": np.asarray(
                    [np.max(np.abs(total_joint_correction))], dtype=np.float32
                ),
                "motion_processing_ground_clearance_source_joint_limit_excess": (
                    source_limit_excess.astype(np.float32)
                ),
                "motion_processing_ground_clearance_corrected_joint_limit_excess": (
                    corrected_limit_excess.astype(np.float32)
                ),
            }
        )
    else:
        output = (
            {key: np.asarray(value).copy() for key, value in motion.items()}
            if args.leg_only_xyz or args.whole_body_xyz
            else {}
        )
        output.update(kinematic_output)
        output.update(
            {
                "contact_stabilization_source_file": np.asarray([str(input_file)]),
                "contact_stabilization_mode": np.asarray(
                    [
                        "whole_body_xyz"
                        if args.whole_body_xyz
                        else "leg_only_xyz" if args.leg_only_xyz else "legacy"
                    ]
                ),
                "contact_stabilization_blend_frames": np.asarray(
                    [BLEND_FRAMES], dtype=np.int64
                ),
                "contact_stabilization_root_weights": np.asarray(
                    [ROOT_MAGNITUDE_WEIGHT, ROOT_VELOCITY_WEIGHT, ROOT_ACCELERATION_WEIGHT],
                    dtype=np.float32,
                ),
                "contact_stabilization_ik_params": np.asarray(
                    [IK_ITERATIONS, IK_DAMPING, IK_MAX_STEP, IK_CORRECTION_CUTOFF_HZ],
                    dtype=np.float32,
                ),
                "contact_stabilization_root_delta_rms": np.asarray(
                    [np.sqrt(np.mean(np.square(total_root_correction)))], dtype=np.float32
                ),
                "contact_stabilization_root_delta_max": np.asarray(
                    [np.max(np.linalg.norm(total_root_correction, axis=1))], dtype=np.float32
                ),
                "contact_stabilization_joint_delta_rms": np.asarray(
                    [np.sqrt(np.mean(np.square(total_joint_correction)))], dtype=np.float32
                ),
                "contact_stabilization_joint_delta_max": np.asarray(
                    [np.max(np.abs(total_joint_correction))], dtype=np.float32
                ),
                "contact_stabilization_support_indicator": foot_contact.astype(np.float32),
                "contact_stabilization_include_ground_support": np.asarray(
                    [args.include_ground_support]
                ),
                "contact_stabilization_ground_support_params": np.asarray(
                    [args.ground_z, args.ground_support_max_height], dtype=np.float32
                ),
            }
        )
        if source_limit_excess is not None:
            output["contact_stabilization_source_joint_limit_excess"] = (
                source_limit_excess.astype(np.float32)
            )
            output["contact_stabilization_corrected_joint_limit_excess"] = (
                corrected_limit_excess.astype(np.float32)
            )
        if corrected_hands is not None:
            output["contact_stabilization_hand_ik_params"] = np.asarray(
                [
                    HAND_IK_ITERATIONS,
                    HAND_IK_DAMPING,
                    HAND_IK_WAIST_SCALE,
                    HAND_IK_WAIST_MAX_STEP,
                    HAND_IK_ARM_MAX_STEP,
                ],
                dtype=np.float32,
            )
            hand_error = corrected_hands - original_hands
            output["contact_stabilization_hand_position_error_rms"] = np.asarray(
                [np.sqrt(np.mean(np.square(hand_error)))], dtype=np.float32
            )
            output["contact_stabilization_hand_position_error_max"] = np.asarray(
                [np.max(np.linalg.norm(hand_error, axis=-1))], dtype=np.float32
            )
        if args.refine_initial_right_stance:
            output["contact_stabilization_initial_right_stance_end_frame"] = np.asarray(
                [INITIAL_RIGHT_STANCE_END_FRAME], dtype=np.int64
            )
            output["contact_stabilization_initial_root_filter_params"] = np.asarray(
                [
                    INITIAL_ROOT_FILTER_END_FRAME,
                    INITIAL_ROOT_FILTER_BLEND_FRAMES,
                    INITIAL_ROOT_XY_CUTOFF_HZ,
                    INITIAL_ROOT_XY_CORRECTION_SCALE,
                ],
                dtype=np.float32,
            )
        for key, value in motion.items():
            if key.startswith(("motion_conversion_", "motion_processing_")):
                output[key] = value
    for key, value in output.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite output: {key}")

    corrected_feet = body_pos_w[:, [body_names.index(name) for name in FOOT_BODY_NAMES]]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **output)
    print(f"[INFO] Wrote repaired motion: {output_file}")
    if args.ground_clearance_hybrid:
        target_z = args.ground_z + args.ground_clearance_margin
        print(
            "[INFO] Foot collision minimum Z before/after: "
            f"{np.min(original_minimum_collision_z, axis=0).round(6).tolist()} -> "
            f"{np.min(corrected_minimum_collision_z, axis=0).round(6).tolist()} m"
        )
        print(
            "[INFO] Residual below requested clearance: "
            f"{np.maximum(target_z - np.min(corrected_minimum_collision_z, axis=0), 0.0).round(6).tolist()} m"
        )
        print(
            "[INFO] Residual ground penetration: "
            f"{np.maximum(args.ground_z - np.min(corrected_minimum_collision_z, axis=0), 0.0).round(6).tolist()} m"
        )
    else:
        print(
            "[INFO] Maximum stance-foot XYZ range: "
            f"{(_maximum_contact_range(original_feet, foot_contact) * 100).round(3).tolist()} -> "
            f"{(_maximum_contact_range(corrected_feet, foot_contact) * 100).round(3).tolist()} cm"
        )
        print(
            "[INFO] Maximum core-stance XYZ range: "
            f"{(_maximum_contact_range(corrected_feet, foot_contact, weights) * 100).round(3).tolist()} cm"
        )
    print(
        f"[INFO] Root correction: rms={np.sqrt(np.mean(np.square(total_root_correction))):.6f} m, "
        f"max={np.max(np.linalg.norm(total_root_correction, axis=1)):.6f} m"
    )
    print(
        "[INFO] Joint correction: "
        f"rms={np.sqrt(np.mean(np.square(total_joint_correction))):.6f} rad, "
        f"max={np.max(np.abs(total_joint_correction)):.6f} rad"
    )
    if initial_stance_weights is not None:
        initial_end = INITIAL_ROOT_FILTER_END_FRAME
        original_initial_right = original_feet[:initial_end, 1]
        corrected_initial_right = corrected_feet[:initial_end, 1]
        print(
            "[INFO] Initial right-sole XY range/path: "
            f"{(np.ptp(original_initial_right[:, :2], axis=0) * 100).round(3).tolist()} / "
            f"{np.linalg.norm(np.diff(original_initial_right[:, :2], axis=0), axis=1).sum() * 100:.3f} -> "
            f"{(np.ptp(corrected_initial_right[:, :2], axis=0) * 100).round(3).tolist()} / "
            f"{np.linalg.norm(np.diff(corrected_initial_right[:, :2], axis=0), axis=1).sum() * 100:.3f} cm"
        )


if __name__ == "__main__":
    main()
