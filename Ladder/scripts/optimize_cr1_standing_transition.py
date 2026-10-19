"""Stabilize a local CR1 standing transition with coordinated root and leg motion.

This tool is intentionally limited to an operator-selected frame window. It
keeps root orientation and every non-leg joint unchanged, while jointly
optimizing root translation and both six-DoF legs so that the two ankle
collision bodies remain near fixed ground-support poses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix, diags, eye, kron, vstack
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
SCRIPT_DIR = REPO_ROOT / "Ladder" / "scripts"
for path in (SOURCE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ladder.utils.urdf_kinematics import (
    finite_difference,
    forward_kinematics,
    quaternion_angular_velocity,
)
from stabilize_cr1_contact_motion import (
    DEFAULT_ROBOT,
    FOOT_COLLISION_BODY_NAMES,
    LEG_JOINT_NAMES,
    _as_names,
    _foot_collision_minimum_z,
    _foot_collision_vertices,
    _validate_no_new_joint_limit_violations,
)


JACOBIAN_EPSILON = 1.0e-4
ROOT_COMPONENT_COUNT = 3
LEG_JOINT_COUNT = sum(len(names) for names in LEG_JOINT_NAMES)
VARIABLE_COUNT = ROOT_COMPONENT_COUNT + LEG_JOINT_COUNT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--start_frame", type=int, default=80)
    parser.add_argument("--core_start_frame", type=int, default=100)
    parser.add_argument("--core_end_frame", type=int, default=190)
    parser.add_argument("--end_frame", type=int, default=218)
    parser.add_argument("--ground_z", type=float, default=-0.03)
    parser.add_argument("--target_ground_clearance", type=float, default=0.002)
    parser.add_argument("--maximum_ground_penetration", type=float, default=0.001)
    parser.add_argument("--root_scale", type=float, default=0.02)
    parser.add_argument("--joint_scale", type=float, default=0.15)
    parser.add_argument("--root_max_correction", type=float, default=0.025)
    parser.add_argument("--joint_max_correction", type=float, default=0.20)
    parser.add_argument("--foot_position_weight", type=float, default=300.0)
    parser.add_argument("--foot_z_weight_scale", type=float, default=1.5)
    parser.add_argument("--foot_orientation_weight", type=float, default=30.0)
    parser.add_argument("--foot_yaw_weight_scale", type=float, default=0.25)
    parser.add_argument("--root_regularization_scale", type=float, default=2.0)
    parser.add_argument("--correction_magnitude_weight", type=float, default=1.0)
    parser.add_argument("--correction_velocity_weight", type=float, default=5.0)
    parser.add_argument("--correction_acceleration_weight", type=float, default=60.0)
    parser.add_argument("--correction_jerk_weight", type=float, default=120.0)
    parser.add_argument("--gauss_newton_iterations", type=int, default=5)
    parser.add_argument("--convergence_tolerance", type=float, default=5.0e-3)
    parser.add_argument("--solver_tolerance", type=float, default=2.0e-6)
    parser.add_argument("--solver_max_iterations", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _quintic(phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float64)
    return 6.0 * phase**5 - 15.0 * phase**4 + 10.0 * phase**3


def _transition_weights(
    start_frame: int,
    core_start_frame: int,
    core_end_frame: int,
    end_frame: int,
) -> np.ndarray:
    if not start_frame < core_start_frame <= core_end_frame < end_frame:
        raise ValueError(
            "frame order must satisfy start < core_start <= core_end < end"
        )
    frames = np.arange(start_frame, end_frame + 1)
    weights = np.ones(frames.shape[0], dtype=np.float64)
    blend_in = frames <= core_start_frame
    weights[blend_in] = _quintic(
        (frames[blend_in] - start_frame) / (core_start_frame - start_frame)
    )
    blend_out = frames >= core_end_frame
    weights[blend_out] = _quintic(
        (end_frame - frames[blend_out]) / (end_frame - core_end_frame)
    )
    return weights


def _block_jacobian(jacobian: np.ndarray) -> csr_matrix:
    frame_count, row_count, column_count = jacobian.shape
    rows = np.broadcast_to(
        np.arange(frame_count)[:, None, None] * row_count
        + np.arange(row_count)[None, :, None],
        jacobian.shape,
    ).reshape(-1)
    columns = np.broadcast_to(
        np.arange(frame_count)[:, None, None] * column_count
        + np.arange(column_count)[None, None, :],
        jacobian.shape,
    ).reshape(-1)
    return csr_matrix(
        (jacobian.reshape(-1), (rows, columns)),
        shape=(frame_count * row_count, frame_count * column_count),
    )


def _difference_operators(
    frame_count: int,
    component_weights: np.ndarray,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    first = diags(
        (-np.ones(frame_count - 1), np.ones(frame_count - 1)),
        (0, 1),
        shape=(frame_count - 1, frame_count),
        format="csr",
    )
    second = diags(
        (
            np.ones(frame_count - 2),
            -2.0 * np.ones(frame_count - 2),
            np.ones(frame_count - 2),
        ),
        (0, 1, 2),
        shape=(frame_count - 2, frame_count),
        format="csr",
    )
    third = diags(
        (
            -np.ones(frame_count - 3),
            3.0 * np.ones(frame_count - 3),
            -3.0 * np.ones(frame_count - 3),
            np.ones(frame_count - 3),
        ),
        (0, 1, 2, 3),
        shape=(frame_count - 3, frame_count),
        format="csr",
    )
    component_scale = diags(component_weights, format="csr")
    return (
        kron(first, component_scale, format="csr"),
        kron(second, component_scale, format="csr"),
        kron(third, component_scale, format="csr"),
    )


def _foot_poses(
    robot_asset: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    positions, quaternions = forward_kinematics(
        robot_asset,
        root_pos,
        root_quat,
        joint_pos,
        joint_names,
        ["base_link", *FOOT_COLLISION_BODY_NAMES],
    )
    return positions[:, 1:].astype(np.float64), quaternions[:, 1:].astype(np.float64)


def _rotation_matrices(quaternions_wxyz: np.ndarray) -> np.ndarray:
    quaternions_wxyz = np.asarray(quaternions_wxyz, dtype=np.float64)
    return Rotation.from_quat(
        quaternions_wxyz[..., [1, 2, 3, 0]].reshape(-1, 4)
    ).as_matrix().reshape(quaternions_wxyz.shape[:-1] + (3, 3))


def _rotation_error(
    target_rotations: np.ndarray,
    current_quaternions: np.ndarray,
) -> np.ndarray:
    current_rotations = _rotation_matrices(current_quaternions)
    relative = target_rotations[None, ...] @ np.swapaxes(current_rotations, -1, -2)
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(
        current_quaternions.shape[0], current_quaternions.shape[1], 3
    )


def _target_foot_poses(
    positions: np.ndarray,
    quaternions: np.ndarray,
    core_slice: slice,
    collision_vertices: tuple[np.ndarray, ...],
    ground_z: float,
    target_clearance: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_positions = np.empty((len(FOOT_COLLISION_BODY_NAMES), 3), dtype=np.float64)
    target_rotations = np.empty((len(FOOT_COLLISION_BODY_NAMES), 3, 3), dtype=np.float64)
    core_rotations = _rotation_matrices(quaternions[core_slice])
    for foot_index, vertices in enumerate(collision_vertices):
        target_rotation = Rotation.from_matrix(
            core_rotations[:, foot_index]
        ).mean().as_matrix()
        target_rotations[foot_index] = target_rotation
        target_positions[foot_index, :2] = np.median(
            positions[core_slice, foot_index, :2], axis=0
        )
        minimum_local_z = np.min(vertices @ target_rotation[2])
        target_positions[foot_index, 2] = (
            ground_z + target_clearance - minimum_local_z
        )
    return target_positions, target_rotations


def _foot_pose_jacobian(
    robot_asset: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    joint_names: list[str],
    leg_indexes: list[int],
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> np.ndarray:
    frame_count = joint_pos.shape[0]
    jacobian = np.zeros(
        (frame_count, len(FOOT_COLLISION_BODY_NAMES), 6, VARIABLE_COUNT),
        dtype=np.float64,
    )
    jacobian[:, :, :3, :3] = np.eye(3)[None, None]
    current_rotations = _rotation_matrices(quaternions)
    for column, joint_index in enumerate(leg_indexes):
        perturbed = joint_pos.copy()
        perturbed[:, joint_index] += JACOBIAN_EPSILON
        perturbed_positions, perturbed_quaternions = _foot_poses(
            robot_asset, root_pos, root_quat, perturbed, joint_names
        )
        jacobian[:, :, :3, ROOT_COMPONENT_COUNT + column] = (
            perturbed_positions - positions
        ) / JACOBIAN_EPSILON
        perturbed_rotations = _rotation_matrices(perturbed_quaternions)
        relative = perturbed_rotations @ np.swapaxes(current_rotations, -1, -2)
        jacobian[:, :, 3:, ROOT_COMPONENT_COUNT + column] = Rotation.from_matrix(
            relative.reshape(-1, 3, 3)
        ).as_rotvec().reshape(frame_count, len(FOOT_COLLISION_BODY_NAMES), 3) / JACOBIAN_EPSILON
    return jacobian.reshape(frame_count, -1, VARIABLE_COUNT)


def _joint_limits(robot_asset: Path, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    joints = {
        joint.attrib["name"]: joint
        for joint in ElementTree.parse(robot_asset).getroot().findall("joint")
    }
    lower = np.empty(len(joint_names), dtype=np.float64)
    upper = np.empty(len(joint_names), dtype=np.float64)
    for index, name in enumerate(joint_names):
        limit = joints[name].find("limit")
        lower[index] = float(limit.attrib["lower"])
        upper[index] = float(limit.attrib["upper"])
    return lower, upper


def _correction_bounds(
    baseline_joint_pos: np.ndarray,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    window_weights: np.ndarray,
    root_scale: float,
    joint_scale: float,
    root_max_correction: float,
    joint_max_correction: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = baseline_joint_pos.shape[0]
    lower = np.empty((frame_count, VARIABLE_COUNT), dtype=np.float64)
    upper = np.empty_like(lower)
    root_bound = root_max_correction * window_weights / root_scale
    lower[:, :ROOT_COMPONENT_COUNT] = -root_bound[:, None]
    upper[:, :ROOT_COMPONENT_COUNT] = root_bound[:, None]
    joint_bound = joint_max_correction * window_weights[:, None]
    lower[:, ROOT_COMPONENT_COUNT:] = np.maximum(
        -joint_bound, joint_lower[None] - baseline_joint_pos
    ) / joint_scale
    upper[:, ROOT_COMPONENT_COUNT:] = np.minimum(
        joint_bound, joint_upper[None] - baseline_joint_pos
    ) / joint_scale
    zero_width = upper - lower < 2.0e-12
    lower[zero_width] -= 1.0e-12
    upper[zero_width] += 1.0e-12
    return lower.reshape(-1), upper.reshape(-1)


def _validate_positive(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    for name in names:
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")


def main() -> None:
    args = _parse_args()
    _validate_positive(
        args,
        (
            "target_ground_clearance",
            "maximum_ground_penetration",
            "root_scale",
            "joint_scale",
            "root_max_correction",
            "joint_max_correction",
            "foot_position_weight",
            "foot_z_weight_scale",
            "foot_orientation_weight",
            "foot_yaw_weight_scale",
            "root_regularization_scale",
            "correction_magnitude_weight",
            "correction_velocity_weight",
            "correction_acceleration_weight",
            "correction_jerk_weight",
            "convergence_tolerance",
            "solver_tolerance",
        ),
    )
    if args.gauss_newton_iterations < 1 or args.solver_max_iterations < 1:
        raise ValueError("solver iteration counts must be positive")

    input_file = args.input_file.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    robot_asset = args.robot_asset.expanduser().resolve()
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_file}")
    if not robot_asset.is_file():
        raise FileNotFoundError(robot_asset)
    motion = _load(input_file)

    joint_names = _as_names(motion["joint_names"])
    source_joint_names = _as_names(motion["source_joint_names"])
    body_names = _as_names(motion["body_names"])
    frame_count = motion["joint_pos"].shape[0]
    if not 0 <= args.start_frame < args.end_frame < frame_count:
        raise ValueError("optimization window is outside the motion")
    window_weights = _transition_weights(
        args.start_frame,
        args.core_start_frame,
        args.core_end_frame,
        args.end_frame,
    )
    local_slice = slice(args.start_frame, args.end_frame + 1)
    core_slice = slice(
        args.core_start_frame - args.start_frame,
        args.core_end_frame - args.start_frame + 1,
    )

    source_indexes = [joint_names.index(name) for name in source_joint_names]
    root_index = body_names.index("base_link")
    full_root_pos = np.asarray(motion["body_pos_w"][:, root_index], dtype=np.float64).copy()
    full_root_quat = np.asarray(motion["body_quat_w"][:, root_index], dtype=np.float64).copy()
    full_joint_pos = np.asarray(motion["joint_pos"][:, source_indexes], dtype=np.float64).copy()
    baseline_root_pos = full_root_pos[local_slice].copy()
    root_quat = full_root_quat[local_slice]
    baseline_joint_pos = full_joint_pos[local_slice].copy()
    leg_indexes = [
        source_joint_names.index(name)
        for names in LEG_JOINT_NAMES
        for name in names
    ]
    if len(leg_indexes) != LEG_JOINT_COUNT:
        raise ValueError("unexpected CR1 leg joint count")

    collision_vertices = _foot_collision_vertices(robot_asset)
    baseline_positions, baseline_quaternions = _foot_poses(
        robot_asset,
        baseline_root_pos,
        root_quat,
        baseline_joint_pos,
        source_joint_names,
    )
    target_positions, target_rotations = _target_foot_poses(
        baseline_positions,
        baseline_quaternions,
        core_slice,
        collision_vertices,
        args.ground_z,
        args.target_ground_clearance,
    )

    variable_scales = np.asarray(
        [args.root_scale] * ROOT_COMPONENT_COUNT
        + [args.joint_scale] * LEG_JOINT_COUNT,
        dtype=np.float64,
    )
    component_weights = np.asarray(
        [args.root_regularization_scale] * ROOT_COMPONENT_COUNT
        + [1.0] * LEG_JOINT_COUNT,
        dtype=np.float64,
    )
    local_frame_count = baseline_joint_pos.shape[0]
    first, second, third = _difference_operators(local_frame_count, component_weights)
    magnitude = diags(
        np.tile(component_weights, local_frame_count), format="csr"
    )
    joint_lower_all, joint_upper_all = _joint_limits(robot_asset, source_joint_names)
    lower_bounds, upper_bounds = _correction_bounds(
        baseline_joint_pos[:, leg_indexes],
        joint_lower_all[leg_indexes],
        joint_upper_all[leg_indexes],
        window_weights,
        args.root_scale,
        args.joint_scale,
        args.root_max_correction,
        args.joint_max_correction,
    )

    current_root_pos = baseline_root_pos.copy()
    current_joint_pos = baseline_joint_pos.copy()
    normalized_correction = np.zeros((local_frame_count, VARIABLE_COUNT), dtype=np.float64)
    solver_diagnostics = []
    pose_weights = np.asarray(
        [
            args.foot_position_weight,
            args.foot_position_weight,
            args.foot_position_weight * args.foot_z_weight_scale,
            args.foot_orientation_weight,
            args.foot_orientation_weight,
            args.foot_orientation_weight * args.foot_yaw_weight_scale,
        ],
        dtype=np.float64,
    )
    row_weights = np.broadcast_to(
        window_weights[:, None, None] * pose_weights[None, None],
        (local_frame_count, len(FOOT_COLLISION_BODY_NAMES), 6),
    ).reshape(-1)

    for iteration in range(args.gauss_newton_iterations):
        current_positions, current_quaternions = _foot_poses(
            robot_asset,
            current_root_pos,
            root_quat,
            current_joint_pos,
            source_joint_names,
        )
        position_error = target_positions[None] - current_positions
        orientation_error = _rotation_error(target_rotations, current_quaternions)
        pose_error = np.concatenate((position_error, orientation_error), axis=-1).reshape(
            local_frame_count, -1
        )
        jacobian = _foot_pose_jacobian(
            robot_asset,
            current_root_pos,
            root_quat,
            current_joint_pos,
            source_joint_names,
            leg_indexes,
            current_positions,
            current_quaternions,
        )
        jacobian *= variable_scales[None, None]
        block_jacobian = _block_jacobian(jacobian)
        linearized_target = pose_error.reshape(-1) + block_jacobian @ normalized_correction.reshape(-1)
        tracking = diags(row_weights, format="csr") @ block_jacobian
        system = vstack(
            (
                tracking,
                args.correction_magnitude_weight * magnitude,
                args.correction_velocity_weight * first,
                args.correction_acceleration_weight * second,
                args.correction_jerk_weight * third,
            ),
            format="csr",
        )
        target = np.concatenate(
            (
                row_weights * linearized_target,
                np.zeros(local_frame_count * VARIABLE_COUNT),
                np.zeros((local_frame_count - 1) * VARIABLE_COUNT),
                np.zeros((local_frame_count - 2) * VARIABLE_COUNT),
                np.zeros((local_frame_count - 3) * VARIABLE_COUNT),
            )
        )
        solution = lsq_linear(
            system,
            target,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            tol=args.solver_tolerance,
            lsmr_tol=args.solver_tolerance,
            lsmr_maxiter=args.solver_max_iterations,
            max_iter=args.solver_max_iterations,
        )
        if not solution.success:
            raise RuntimeError(
                f"standing-transition optimization failed at iteration {iteration}: "
                f"{solution.message}"
            )
        next_correction = solution.x.reshape(local_frame_count, VARIABLE_COUNT)
        maximum_update = float(np.max(np.abs(next_correction - normalized_correction)))
        normalized_correction = next_correction
        actual_correction = normalized_correction * variable_scales[None]
        current_root_pos = baseline_root_pos + actual_correction[:, :ROOT_COMPONENT_COUNT]
        current_joint_pos = baseline_joint_pos.copy()
        current_joint_pos[:, leg_indexes] += actual_correction[:, ROOT_COMPONENT_COUNT:]
        solver_diagnostics.append(
            [
                float(solution.status),
                float(solution.nit),
                float(solution.cost),
                float(solution.optimality),
                maximum_update,
            ]
        )
        print(
            f"[INFO] Iteration {iteration + 1}: cost={solution.cost:.6g}, "
            f"optimality={solution.optimality:.3g}, max normalized update={maximum_update:.6g}"
        )
        if maximum_update < args.convergence_tolerance:
            break

    normalized_correction[window_weights == 0.0] = 0.0
    actual_correction = normalized_correction * variable_scales[None]
    current_root_pos = baseline_root_pos + actual_correction[:, :ROOT_COMPONENT_COUNT]
    current_joint_pos = baseline_joint_pos.copy()
    current_joint_pos[:, leg_indexes] += actual_correction[:, ROOT_COMPONENT_COUNT:]
    _validate_no_new_joint_limit_violations(
        robot_asset, baseline_joint_pos, current_joint_pos, source_joint_names
    )

    full_root_pos[local_slice] = current_root_pos
    full_joint_pos[local_slice] = current_joint_pos
    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset,
        full_root_pos,
        full_root_quat,
        full_joint_pos,
        source_joint_names,
        body_names,
    )
    output_joint_pos = full_joint_pos[
        :, [source_joint_names.index(name) for name in joint_names]
    ].astype(np.float32)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    minimum_collision_z = _foot_collision_minimum_z(
        robot_asset,
        full_root_pos,
        full_root_quat,
        full_joint_pos,
        source_joint_names,
        collision_vertices,
    )
    maximum_penetration = max(args.ground_z - float(np.min(minimum_collision_z)), 0.0)
    if maximum_penetration > args.maximum_ground_penetration + 1.0e-7:
        raise ValueError(
            f"ground penetration {maximum_penetration:.6f} m exceeds "
            f"{args.maximum_ground_penetration:.6f} m"
        )

    output = {key: np.asarray(value).copy() for key, value in motion.items()}
    output.update(
        {
            "joint_pos": output_joint_pos,
            "joint_vel": finite_difference(output_joint_pos, fps),
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": finite_difference(body_pos_w, fps),
            "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
            "standing_transition_source_file": np.asarray([str(input_file)]),
            "standing_transition_frames": np.asarray(
                [
                    args.start_frame,
                    args.core_start_frame,
                    args.core_end_frame,
                    args.end_frame,
                ],
                dtype=np.int64,
            ),
            "standing_transition_window_weights": window_weights.astype(np.float32),
            "standing_transition_target_foot_positions": target_positions.astype(np.float32),
            "standing_transition_target_foot_rotations": target_rotations.astype(np.float32),
            "standing_transition_scales": variable_scales.astype(np.float32),
            "standing_transition_normalized_correction": normalized_correction.astype(np.float32),
            "standing_transition_solver_diagnostics": np.asarray(
                solver_diagnostics, dtype=np.float64
            ),
            "standing_transition_convergence_tolerance": np.asarray(
                [args.convergence_tolerance], dtype=np.float32
            ),
            "standing_transition_ground_params": np.asarray(
                [
                    args.ground_z,
                    args.target_ground_clearance,
                    args.maximum_ground_penetration,
                ],
                dtype=np.float32,
            ),
            "standing_transition_tracking_weights": pose_weights.astype(np.float32),
            "standing_transition_regularization_weights": np.asarray(
                [
                    args.root_regularization_scale,
                    args.correction_magnitude_weight,
                    args.correction_velocity_weight,
                    args.correction_acceleration_weight,
                    args.correction_jerk_weight,
                ],
                dtype=np.float32,
            ),
        }
    )
    for key, value in output.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite output: {key}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **output)

    final_positions, final_quaternions = _foot_poses(
        robot_asset,
        current_root_pos,
        root_quat,
        current_joint_pos,
        source_joint_names,
    )
    final_orientation_error = np.linalg.norm(
        _rotation_error(target_rotations, final_quaternions), axis=-1
    )
    core = core_slice
    print(f"[INFO] Wrote standing-transition preview: {output_file}")
    print(
        "[INFO] Core foot XYZ ranges: "
        f"{(np.ptp(final_positions[core], axis=0) * 1000.0).round(3).tolist()} mm"
    )
    print(
        "[INFO] Core foot orientation error p95: "
        f"{np.degrees(np.percentile(final_orientation_error[core], 95, axis=0)).round(3).tolist()} deg"
    )
    print(
        "[INFO] Root correction p95/max: "
        f"{np.percentile(np.linalg.norm(actual_correction[:, :3], axis=1), 95) * 1000.0:.3f}/"
        f"{np.max(np.linalg.norm(actual_correction[:, :3], axis=1)) * 1000.0:.3f} mm"
    )
    print(
        "[INFO] Leg correction p95/max: "
        f"{np.percentile(np.abs(actual_correction[:, 3:]), 95):.5f}/"
        f"{np.max(np.abs(actual_correction[:, 3:])):.5f} rad"
    )
    print(
        "[INFO] Exact ground clearance min: "
        f"{(np.min(minimum_collision_z[local_slice], axis=0) - args.ground_z).round(6).tolist()} m"
    )


if __name__ == "__main__":
    main()
