"""Smooth CR1 leg motion while preserving annotated support-foot positions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix, diags, eye, kron, vstack


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
    FOOT_BODY_NAMES,
    JACOBIAN_EPSILON,
    LEG_JOINT_NAMES,
    _add_ground_support_contacts,
    _arm_positions,
    _as_names,
    _contact_anchors_and_weights,
    _foot_positions,
    _maximum_contact_range,
    _validate_no_new_joint_limit_violations,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_file", type=Path, required=True)
    parser.add_argument("--candidate_file", type=Path, required=True)
    parser.add_argument("--support_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--include_ground_support", action="store_true")
    parser.add_argument("--ground_z", type=float, default=-0.03)
    parser.add_argument("--ground_support_max_height", type=float, default=0.07)
    parser.add_argument("--foot_tracking_weight", type=float, default=300.0)
    parser.add_argument("--correction_magnitude_weight", type=float, default=1.0)
    parser.add_argument("--correction_velocity_weight", type=float, default=5.0)
    parser.add_argument("--correction_acceleration_weight", type=float, default=60.0)
    parser.add_argument("--correction_jerk_weight", type=float, default=120.0)
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


def _difference_operators(frame_count: int, joint_count: int):
    first = diags(
        (-np.ones(frame_count - 1), np.ones(frame_count - 1)),
        (0, 1),
        shape=(frame_count - 1, frame_count),
        format="csr",
    )
    second = diags(
        (np.ones(frame_count - 2), -2.0 * np.ones(frame_count - 2), np.ones(frame_count - 2)),
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
    identity = eye(joint_count, format="csr")
    return kron(first, identity), kron(second, identity), kron(third, identity)


def _validate_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _joint_correction_bounds(
    robot_asset: Path,
    joint_pos: np.ndarray,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    joints = {
        joint.attrib["name"]: joint
        for joint in ElementTree.parse(robot_asset).getroot().findall("joint")
    }
    lower_bounds = np.empty_like(joint_pos)
    upper_bounds = np.empty_like(joint_pos)
    for column, name in enumerate(joint_names):
        limit = joints[name].find("limit")
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        existing_excess = max(
            float(np.max(lower - joint_pos[:, column])),
            float(np.max(joint_pos[:, column] - upper)),
            0.0,
        )
        lower_bounds[:, column] = lower - existing_excess - joint_pos[:, column]
        upper_bounds[:, column] = upper + existing_excess - joint_pos[:, column]
    return lower_bounds.reshape(-1), upper_bounds.reshape(-1)


def main() -> None:
    args = _parse_args()
    for name in (
        "foot_tracking_weight",
        "correction_magnitude_weight",
        "correction_velocity_weight",
        "correction_acceleration_weight",
        "correction_jerk_weight",
        "solver_tolerance",
    ):
        _validate_positive(name, float(getattr(args, name)))
    if args.solver_max_iterations < 1:
        raise ValueError("solver_max_iterations must be positive")
    if args.ground_support_max_height < 0.0:
        raise ValueError("ground_support_max_height must be non-negative")

    input_paths = [
        args.reference_file.expanduser().resolve(),
        args.candidate_file.expanduser().resolve(),
        args.support_file.expanduser().resolve(),
    ]
    output_file = args.output_file.expanduser().resolve()
    robot_asset = args.robot_asset.expanduser().resolve()
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_file}")
    if not robot_asset.is_file():
        raise FileNotFoundError(robot_asset)
    reference, candidate, support_motion = (_load(path) for path in input_paths)

    joint_names = _as_names(candidate["joint_names"])
    source_joint_names = _as_names(candidate["source_joint_names"])
    body_names = _as_names(candidate["body_names"])
    frame_count = candidate["joint_pos"].shape[0]
    fps = float(np.asarray(candidate["fps"]).reshape(-1)[0])
    for label, motion in (("reference", reference), ("support", support_motion)):
        if _as_names(motion["joint_names"]) != joint_names:
            raise ValueError(f"{label} joint names do not match the candidate")
        if motion["joint_pos"].shape != candidate["joint_pos"].shape:
            raise ValueError(f"{label} joint positions do not match the candidate shape")
        if not np.isclose(float(np.asarray(motion["fps"]).reshape(-1)[0]), fps):
            raise ValueError(f"{label} fps does not match the candidate")
    if "contact_indicator" not in support_motion:
        raise KeyError("support_file does not contain contact_indicator")

    source_indexes = [joint_names.index(name) for name in source_joint_names]
    root_index = body_names.index("base_link")
    root_pos = np.asarray(candidate["body_pos_w"][:, root_index], dtype=np.float64)
    root_quat = np.asarray(candidate["body_quat_w"][:, root_index], dtype=np.float64)
    candidate_joint_pos = np.asarray(candidate["joint_pos"][:, source_indexes], dtype=np.float64)
    reference_joint_pos = np.asarray(reference["joint_pos"][:, source_indexes], dtype=np.float64)
    reference_root_pos = np.asarray(reference["body_pos_w"][:, root_index], dtype=np.float64)
    reference_root_quat = np.asarray(reference["body_quat_w"][:, root_index], dtype=np.float64)
    reference_feet = _foot_positions(
        robot_asset,
        reference_root_pos,
        reference_root_quat,
        reference_joint_pos,
        source_joint_names,
    )
    original_candidate_feet = _foot_positions(
        robot_asset, root_pos, root_quat, candidate_joint_pos, source_joint_names
    )
    support = np.asarray(support_motion["contact_indicator"][:, 2:4] > 0.5)
    if args.include_ground_support:
        support = _add_ground_support_contacts(
            support,
            reference_feet,
            args.ground_z,
            args.ground_support_max_height,
        )
    _, support_weights = _contact_anchors_and_weights(reference_feet, support)

    result = candidate_joint_pos.copy()
    leg_indexes = [
        [source_joint_names.index(name) for name in names] for names in LEG_JOINT_NAMES
    ]
    first, second, third = _difference_operators(frame_count, len(leg_indexes[0]))
    identity = eye(frame_count * len(leg_indexes[0]), format="csr")
    solver_diagnostics = []
    for foot_index, joint_indexes in enumerate(leg_indexes):
        jacobian = np.empty((frame_count, 3, len(joint_indexes)), dtype=np.float64)
        for column, joint_index in enumerate(joint_indexes):
            perturbed = candidate_joint_pos.copy()
            perturbed[:, joint_index] += JACOBIAN_EPSILON
            perturbed_feet = _foot_positions(
                robot_asset, root_pos, root_quat, perturbed, source_joint_names
            )
            jacobian[:, :, column] = (
                perturbed_feet[:, foot_index] - original_candidate_feet[:, foot_index]
            ) / JACOBIAN_EPSILON

        row_weights = np.repeat(support_weights[:, foot_index], 3)
        tracking = diags(row_weights, format="csr") @ _block_jacobian(jacobian)
        tracking_error = row_weights * (
            reference_feet[:, foot_index] - original_candidate_feet[:, foot_index]
        ).reshape(-1)
        system = vstack(
            (
                args.foot_tracking_weight * tracking,
                args.correction_magnitude_weight * identity,
                args.correction_velocity_weight * first,
                args.correction_acceleration_weight * second,
                args.correction_jerk_weight * third,
            ),
            format="csr",
        )
        target = np.concatenate(
            (
                args.foot_tracking_weight * tracking_error,
                np.zeros(frame_count * len(joint_indexes)),
                np.zeros((frame_count - 1) * len(joint_indexes)),
                np.zeros((frame_count - 2) * len(joint_indexes)),
                np.zeros((frame_count - 3) * len(joint_indexes)),
            )
        )
        lower_bounds, upper_bounds = _joint_correction_bounds(
            robot_asset,
            candidate_joint_pos[:, joint_indexes],
            [source_joint_names[index] for index in joint_indexes],
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
                f"bounded support optimization failed for foot {foot_index}: {solution.message}"
            )
        correction = solution.x.reshape(frame_count, len(joint_indexes))
        result[:, joint_indexes] += correction
        solver_diagnostics.append(
            (solution.status, solution.nit, solution.cost, solution.optimality)
        )

    _validate_no_new_joint_limit_violations(
        robot_asset, candidate_joint_pos, result, source_joint_names
    )
    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset, root_pos, root_quat, result, source_joint_names, body_names
    )
    output_joint_pos = result[
        :, [source_joint_names.index(name) for name in joint_names]
    ].astype(np.float32)
    output = {key: np.asarray(value).copy() for key, value in candidate.items()}
    output.update(
        {
            "joint_pos": output_joint_pos,
            "joint_vel": finite_difference(output_joint_pos, fps),
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": finite_difference(body_pos_w, fps),
            "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
            "support_optimization_reference_file": np.asarray([str(input_paths[0])]),
            "support_optimization_candidate_file": np.asarray([str(input_paths[1])]),
            "support_optimization_support_file": np.asarray([str(input_paths[2])]),
            "support_optimization_weights": np.asarray(
                [
                    args.foot_tracking_weight,
                    args.correction_magnitude_weight,
                    args.correction_velocity_weight,
                    args.correction_acceleration_weight,
                    args.correction_jerk_weight,
                ],
                dtype=np.float32,
            ),
            "support_optimization_solver": np.asarray(
                [args.solver_tolerance, args.solver_max_iterations], dtype=np.float32
            ),
            "support_optimization_solver_diagnostics": np.asarray(
                solver_diagnostics, dtype=np.float64
            ),
            "support_optimization_indicator": support.astype(np.float32),
            "support_optimization_weights_per_frame": support_weights.astype(np.float32),
            "support_optimization_ground_params": np.asarray(
                [args.ground_z, args.ground_support_max_height], dtype=np.float32
            ),
        }
    )
    for key, value in output.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite output: {key}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **output)

    corrected_feet = body_pos_w[:, [body_names.index(name) for name in FOOT_BODY_NAMES]]
    reference_hands = _arm_positions(
        robot_asset,
        reference_root_pos,
        reference_root_quat,
        reference_joint_pos,
        source_joint_names,
    )
    corrected_hands = _arm_positions(
        robot_asset, root_pos, root_quat, result, source_joint_names
    )
    print(f"[INFO] Wrote support-constrained motion: {output_file}")
    print(
        "[INFO] Maximum core support-foot XYZ range: "
        f"{(_maximum_contact_range(corrected_feet, support, support_weights) * 100).round(3).tolist()} cm"
    )
    print(
        "[INFO] Sphere-hand change relative to reference: "
        f"p95={np.percentile(np.linalg.norm(corrected_hands - reference_hands, axis=-1), 95, axis=0).round(6).tolist()} m"
    )


if __name__ == "__main__":
    main()
