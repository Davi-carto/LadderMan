"""Repair CR1 sphere-hand penetrations with smooth arm-only trajectory IK."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from detect_cr1_reset_valid_frames import (
    DEFAULT_GROUND_Z,
    DEFAULT_LADDER_ASSET,
    DEFAULT_LADDER_POS,
    DEFAULT_LADDER_YAW_DEG,
    DEFAULT_ROBOT_ASSET,
    _build_collision_model,
)
from ladder.utils.contact_annotation import annotate_ladder_contacts, make_contact_annotation_cfg
from ladder.utils.urdf_kinematics import finite_difference, forward_kinematics, quaternion_angular_velocity


DEFAULT_INPUT = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "climing_ladder_cr1_root3hz_leg2hz_other5hz_50fps_annotated_wrist_d0145_v035_trim_start200_end851.npz"
)
ARM_SPECS = {
    "left": (
        "left_sphere_hand_link",
        (
            "left_shoulder_y_joint",
            "left_shoulder_x_joint",
            "left_shoulder_z_joint",
            "left_elbow_joint",
        ),
    ),
    "right": (
        "right_sphere_hand_link",
        (
            "right_shoulder_y_joint",
            "right_shoulder_x_joint",
            "right_shoulder_z_joint",
            "right_elbow_joint",
        ),
    ),
}
SOFT_JOINT_LIMIT_FACTOR = 0.9
COLLISION_CLEARANCE = 0.001
CONSTRAINT_ACTIVATION_DISTANCE = -COLLISION_CLEARANCE
MAX_ITERATIONS = 110
MAX_ITERATION_JOINT_STEP = 0.08
MAX_ARM_SPEED = 3.0
MAX_ARM_ACCELERATION = 35.0
MAX_ARM_JERK = 1200.0
NOMINAL_WEIGHT = 1.0
VELOCITY_CORRECTION_WEIGHT = 20.0
ACCELERATION_CORRECTION_WEIGHT = 200.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove sphere-hand/environment penetrations using smooth arm-only IK."
    )
    parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT, help="Input annotated CR1 motion npz.")
    parser.add_argument("--output_file", type=Path, required=True, help="Output repaired and re-annotated npz.")
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT_ASSET, help="CR1 collision/FK URDF.")
    parser.add_argument("--ladder_asset", type=Path, default=DEFAULT_LADDER_ASSET, help="Wood-step collision URDF.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_file if it exists.")
    return parser.parse_args()


def _as_names(values: np.ndarray) -> list[str]:
    items = np.asarray(values).reshape(-1).tolist()
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items]


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _sphere_box_distance_normal(
    sphere_center: np.ndarray,
    sphere_radius: float,
    box_center: np.ndarray,
    box_rotation: np.ndarray,
    box_half_extents: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return signed sphere-box distance and the box-to-sphere outward normal."""

    local_center = box_rotation.T @ (sphere_center - box_center)
    local_outside = np.abs(local_center) - box_half_extents
    outside = np.maximum(local_outside, 0.0)
    outside_norm = float(np.linalg.norm(outside))
    if outside_norm > 1.0e-12:
        local_normal = np.sign(local_center) * outside / outside_norm
        return outside_norm - sphere_radius, box_rotation @ local_normal

    face_axis = int(np.argmax(local_outside))
    face_sign = 1.0 if local_center[face_axis] >= 0.0 else -1.0
    normal = face_sign * box_rotation[:, face_axis]
    return float(local_outside[face_axis]) - sphere_radius, normal


class ArmTrajectoryOptimizer:
    def __init__(
        self,
        mujoco,
        model,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        joint_names: list[str],
        body_names: list[str],
        fps: float,
        contact_indicator: np.ndarray,
        contact_rung_id: np.ndarray,
        limb_names: list[str],
    ) -> None:
        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.root_pos = root_pos
        self.root_quat_wxyz = root_quat_wxyz
        self.joint_names = joint_names
        self.num_frames = root_pos.shape[0]
        self.fps = fps
        self.contact_indicator = contact_indicator
        self.contact_rung_id = contact_rung_id
        self.limb_indexes = {name: limb_names.index(name) for name in limb_names}
        self.model_joint_ids = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index): index
            for index in range(model.njnt)
        }
        self.model_body_ids = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index): index
            for index in range(model.nbody)
        }
        self.root_qpos_address = int(model.jnt_qposadr[self.model_joint_ids["root_joint"]])
        missing_joints = [name for name in joint_names if name not in self.model_joint_ids]
        if missing_joints:
            raise ValueError(f"motion joints are missing from collision model: {missing_joints}")
        if "base_link" not in body_names:
            raise ValueError("motion body_names is missing 'base_link'")

        mujoco.mj_forward(model, self.data)
        self.environment_boxes = []
        for geom_id in range(model.ngeom):
            body_name = (
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[geom_id]),
                )
                or ""
            )
            if not body_name.startswith("env_"):
                continue
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError(f"environment geom {geom_id} is not a box")
            self.environment_boxes.append(
                (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    or f"geom_{geom_id}",
                    self.data.geom_xpos[geom_id].copy(),
                    self.data.geom_xmat[geom_id].reshape(3, 3).copy(),
                    model.geom_size[geom_id, :3].copy(),
                )
            )

    def _write_frame(self, frame: int, joint_pos: np.ndarray) -> None:
        self.data.qpos[:] = self.model.qpos0
        address = self.root_qpos_address
        self.data.qpos[address : address + 3] = self.root_pos[frame]
        self.data.qpos[address + 3 : address + 7] = self.root_quat_wxyz[frame]
        for joint_column, joint_name in enumerate(self.joint_names):
            joint_id = self.model_joint_ids[joint_name]
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = joint_pos[frame, joint_column]
        self.mujoco.mj_forward(self.model, self.data)

    def _linearize(
        self,
        joint_pos: np.ndarray,
        sphere_body_name: str,
        arm_joint_names: tuple[str, ...],
        active_route_constraints: set[tuple[int, int]],
    ) -> tuple[np.ndarray, list[tuple[int, float, np.ndarray]]]:
        sphere_body_id = self.model_body_ids[sphere_body_name]
        sphere_geom_ids = np.flatnonzero(self.model.geom_bodyid == sphere_body_id)
        if sphere_geom_ids.shape != (1,):
            raise ValueError(f"expected one geom for {sphere_body_name}, got {sphere_geom_ids.tolist()}")
        sphere_geom_id = int(sphere_geom_ids[0])
        sphere_radius = float(self.model.geom_size[sphere_geom_id, 0])
        arm_dof_ids = [
            int(self.model.jnt_dofadr[self.model_joint_ids[name]]) for name in arm_joint_names
        ]
        position_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        rotation_jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        min_distances = np.full(self.num_frames, np.inf, dtype=np.float64)
        linearized_constraints: list[tuple[int, float, np.ndarray]] = []

        for frame in range(self.num_frames):
            self._write_frame(frame, joint_pos)
            sphere_center = self.data.geom_xpos[sphere_geom_id].copy()
            self.mujoco.mj_jacBody(
                self.model,
                self.data,
                position_jacobian,
                rotation_jacobian,
                sphere_body_id,
            )
            arm_position_jacobian = position_jacobian[:, arm_dof_ids]
            for box_index, (
                box_name,
                box_center,
                box_rotation,
                box_half_extents,
            ) in enumerate(self.environment_boxes):
                actual_distance, actual_normal = _sphere_box_distance_normal(
                    sphere_center,
                    sphere_radius,
                    box_center,
                    box_rotation,
                    box_half_extents,
                )
                min_distances[frame] = min(min_distances[frame], actual_distance)
                route_key = (frame, box_index)
                if actual_distance < CONSTRAINT_ACTIVATION_DISTANCE:
                    active_route_constraints.add(route_key)
                if route_key not in active_route_constraints:
                    continue

                preferred_normal = None
                if box_name.startswith("step_"):
                    limb_name = "left_hand" if sphere_body_name.startswith("left_") else "right_hand"
                    limb_index = self.limb_indexes[limb_name]
                    required_step_name = (
                        f"step_{int(self.contact_rung_id[frame, limb_index]) + 1:02d}_collision"
                    )
                    if self.contact_indicator[frame, limb_index] > 0.5 and box_name == required_step_name:
                        preferred_normal = np.asarray([0.0, 0.0, 1.0])
                    else:
                        preferred_normal = np.asarray([-1.0, 0.0, 0.0])
                elif (
                    sphere_body_name.startswith("left_")
                    and box_name == "front_zigzag_side_board_right_collision"
                ):
                    preferred_normal = np.asarray([0.0, -1.0, 0.0])
                elif (
                    sphere_body_name.startswith("right_")
                    and box_name == "front_zigzag_side_board_left_collision"
                ):
                    preferred_normal = np.asarray([0.0, 1.0, 0.0])

                if preferred_normal is None:
                    constraint_distance = actual_distance
                    constraint_normal = actual_normal
                else:
                    box_support = float(
                        preferred_normal @ box_center
                        + np.sum(box_half_extents * np.abs(box_rotation.T @ preferred_normal))
                    )
                    constraint_distance = (
                        float(preferred_normal @ sphere_center) - box_support - sphere_radius
                    )
                    constraint_normal = preferred_normal
                gradient = constraint_normal @ arm_position_jacobian
                if np.linalg.norm(gradient) < 1.0e-10 and constraint_distance < COLLISION_CLEARANCE:
                    raise RuntimeError(
                        f"{sphere_body_name} collision at frame {frame} cannot be resolved by its arm joints"
                    )
                linearized_constraints.append((frame, constraint_distance, gradient))
        return min_distances, linearized_constraints

    def optimize_side(
        self,
        original_joint_pos: np.ndarray,
        current_joint_pos: np.ndarray,
        side: str,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        try:
            import cvxpy as cp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "This repair script requires cvxpy. Run it with the local miniforge 'retarget' environment."
            ) from exc

        sphere_body_name, arm_joint_names = ARM_SPECS[side]
        arm_columns = [self.joint_names.index(name) for name in arm_joint_names]
        original_arm = original_joint_pos[:, arm_columns]
        optimized = current_joint_pos.copy()

        lower_limits = []
        upper_limits = []
        for joint_name in arm_joint_names:
            joint_id = self.model_joint_ids[joint_name]
            hard_lower, hard_upper = self.model.jnt_range[joint_id]
            center = 0.5 * (hard_lower + hard_upper)
            half_range = 0.5 * SOFT_JOINT_LIMIT_FACTOR * (hard_upper - hard_lower)
            lower_limits.append(center - half_range)
            upper_limits.append(center + half_range)
        lower_limits = np.asarray(lower_limits)
        upper_limits = np.asarray(upper_limits)
        if np.any(original_arm < lower_limits - 1.0e-6) or np.any(original_arm > upper_limits + 1.0e-6):
            raise ValueError(f"original {side} arm trajectory is outside the 0.9 soft joint limits")

        initial_distances = None
        iterations = 0
        active_route_constraints: set[tuple[int, int]] = set()
        for iteration in range(MAX_ITERATIONS):
            iterations = iteration + 1
            min_distances, linearized = self._linearize(
                optimized, sphere_body_name, arm_joint_names, active_route_constraints
            )
            if initial_distances is None:
                initial_distances = min_distances.copy()
            penetrating = int(np.count_nonzero(min_distances < -COLLISION_CLEARANCE))
            print(
                f"[IK] {side} iteration={iteration}: penetrating={penetrating}, "
                f"min_distance={min_distances.min():.6f} m, constraints={len(linearized)}"
            )
            if min_distances.min() >= -COLLISION_CLEARANCE:
                break

            arm_trajectory = cp.Variable((self.num_frames, len(arm_columns)))
            correction = arm_trajectory - original_arm
            objective = (
                NOMINAL_WEIGHT * cp.sum_squares(correction)
                + VELOCITY_CORRECTION_WEIGHT
                * cp.sum_squares(correction[1:] - correction[:-1])
                + ACCELERATION_CORRECTION_WEIGHT
                * cp.sum_squares(correction[2:] - 2.0 * correction[1:-1] + correction[:-2])
            )

            row_indexes = []
            column_indexes = []
            values = []
            bounds = np.empty(len(linearized), dtype=np.float64)
            current_arm = optimized[:, arm_columns]
            for row, (frame, distance, gradient) in enumerate(linearized):
                for joint_offset, gradient_value in enumerate(gradient):
                    row_indexes.append(row)
                    column_indexes.append(frame * len(arm_columns) + joint_offset)
                    values.append(gradient_value)
                bounds[row] = COLLISION_CLEARANCE - distance + gradient @ current_arm[frame]
            constraint_matrix = sparse.csr_matrix(
                (values, (row_indexes, column_indexes)),
                shape=(len(linearized), self.num_frames * len(arm_columns)),
            )
            constraints = [
                constraint_matrix @ cp.vec(arm_trajectory, order="C") >= bounds,
                arm_trajectory >= lower_limits,
                arm_trajectory <= upper_limits,
            ]
            velocity = arm_trajectory[1:] - arm_trajectory[:-1]
            acceleration = velocity[1:] - velocity[:-1]
            jerk = acceleration[1:] - acceleration[:-1]
            constraints.extend(
                [
                    velocity <= MAX_ARM_SPEED / self.fps,
                    velocity >= -MAX_ARM_SPEED / self.fps,
                    acceleration <= MAX_ARM_ACCELERATION / self.fps**2,
                    acceleration >= -MAX_ARM_ACCELERATION / self.fps**2,
                    jerk <= MAX_ARM_JERK / self.fps**3,
                    jerk >= -MAX_ARM_JERK / self.fps**3,
                ]
            )
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(
                solver=cp.CLARABEL,
                verbose=False,
                tol_gap_abs=1.0e-8,
                tol_feas=1.0e-8,
                max_iter=300,
            )
            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or arm_trajectory.value is None:
                raise RuntimeError(f"{side} arm IK failed with solver status: {problem.status}")

            iteration_step = arm_trajectory.value - current_arm
            max_step = float(np.max(np.abs(iteration_step)))
            if penetrating < 10:
                iteration_step_limit = 0.01
            elif penetrating < 50:
                iteration_step_limit = 0.03
            else:
                iteration_step_limit = MAX_ITERATION_JOINT_STEP
            step_scale = min(1.0, iteration_step_limit / max(max_step, 1.0e-12))
            optimized[:, arm_columns] = current_arm + step_scale * iteration_step

        final_distances, _ = self._linearize(
            optimized, sphere_body_name, arm_joint_names, active_route_constraints
        )
        if final_distances.min() < -COLLISION_CLEARANCE:
            raise RuntimeError(
                f"{side} arm-only IK left a sphere-hand penetration of {-final_distances.min():.6f} m"
            )
        correction = optimized[:, arm_columns] - original_arm
        diagnostics: dict[str, float | int] = {
            "iterations": iterations,
            "penetrating_frames_before": int(np.count_nonzero(initial_distances < -COLLISION_CLEARANCE)),
            "penetrating_frames_after": int(
                np.count_nonzero(final_distances < -COLLISION_CLEARANCE)
            ),
            "min_distance_before": float(initial_distances.min()),
            "min_distance_after": float(final_distances.min()),
            "joint_delta_rms": float(np.sqrt(np.mean(np.square(correction)))),
            "joint_delta_max": float(np.max(np.abs(correction))),
        }
        return optimized, diagnostics


def _regenerate_contact_labels(motion: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = (
        "contact_annotation_ladder_params",
        "contact_annotation_ladder_origin",
        "contact_annotation_ladder_num_rungs",
        "contact_annotation_thresholds",
        "limb_names",
        "limb_body_names",
        "contact_annotation_limb_target_offsets",
    )
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"motion is missing contact annotation metadata: {missing}")
    phi, spacing, length, width = np.asarray(motion["contact_annotation_ladder_params"]).tolist()
    distance, speed, min_frames, max_gap = np.asarray(motion["contact_annotation_thresholds"]).tolist()
    cfg = make_contact_annotation_cfg(
        ladder_phi=float(phi),
        rung_spacing=float(spacing),
        rung_length=float(length),
        rung_width=float(width),
        num_rungs=int(np.asarray(motion["contact_annotation_ladder_num_rungs"]).reshape(-1)[0]),
        ladder_origin=tuple(np.asarray(motion["contact_annotation_ladder_origin"], dtype=float).tolist()),
        limb_body_names=tuple(_as_names(motion["limb_body_names"])),
        limb_target_offsets=tuple(
            tuple(float(value) for value in offset)
            for offset in np.asarray(motion["contact_annotation_limb_target_offsets"])
        ),
        distance_threshold=float(distance),
        limb_distance_thresholds=(
            tuple(
                float(value)
                for value in np.asarray(motion["contact_annotation_limb_distance_thresholds"])
            )
            if "contact_annotation_limb_distance_thresholds" in motion
            else None
        ),
        speed_threshold=float(speed),
        min_contact_frames=int(round(min_frames)),
        max_gap_frames=int(round(max_gap)),
        ladder_asset_translation=(
            tuple(np.asarray(motion["contact_annotation_ladder_asset_translation"], dtype=float).tolist())
            if "contact_annotation_ladder_asset_translation" in motion
            else None
        ),
        ladder_asset_yaw_deg=(
            float(np.asarray(motion["contact_annotation_ladder_asset_yaw_deg"]).reshape(-1)[0])
            if "contact_annotation_ladder_asset_yaw_deg" in motion
            else None
        ),
        ground_initial_feet="contact_annotation_robot_grounding_z_offset" in motion,
        initial_foot_target_z=float(
            np.asarray(motion.get("contact_annotation_robot_initial_foot_target_z", [0.0])).reshape(-1)[0]
        ),
    )
    return annotate_ladder_contacts(motion, cfg)


def main() -> None:
    args = _parse_args()
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This repair script requires MuJoCo. Run it with the local miniforge 'retarget' environment."
        ) from exc

    input_file = _require_file(args.input_file, "input motion")
    robot_asset = _require_file(args.robot_asset, "robot asset")
    ladder_asset = _require_file(args.ladder_asset, "ladder asset")
    output_file = args.output_file.expanduser().resolve()
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"output file exists: {output_file}. Pass --overwrite to replace it.")
    with np.load(input_file, allow_pickle=False) as data:
        motion = {key: data[key] for key in data.files}

    required = (
        "fps",
        "joint_pos",
        "joint_names",
        "body_pos_w",
        "body_quat_w",
        "body_names",
        "contact_indicator",
        "contact_rung_id",
        "limb_names",
    )
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"motion is missing required keys: {missing}")
    joint_names = _as_names(motion["joint_names"])
    body_names = _as_names(motion["body_names"])
    root_index = body_names.index("base_link")
    root_pos = np.asarray(motion["body_pos_w"][:, root_index], dtype=np.float64)
    root_quat_wxyz = np.asarray(motion["body_quat_w"][:, root_index], dtype=np.float64)
    original_joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    ladder_pos = tuple(
        np.asarray(
            motion.get("contact_annotation_ladder_asset_translation", DEFAULT_LADDER_POS), dtype=float
        ).tolist()
    )
    ladder_yaw_deg = float(
        np.asarray(
            motion.get("contact_annotation_ladder_asset_yaw_deg", [DEFAULT_LADDER_YAW_DEG])
        ).reshape(-1)[0]
    )
    ground_z = float(
        np.asarray(
            motion.get("contact_annotation_robot_initial_foot_target_z", [DEFAULT_GROUND_Z])
        ).reshape(-1)[0]
    )

    model = _build_collision_model(
        mujoco, robot_asset, ladder_asset, ground_z, ladder_pos, ladder_yaw_deg
    )
    optimizer = ArmTrajectoryOptimizer(
        mujoco,
        model,
        root_pos,
        root_quat_wxyz,
        joint_names,
        body_names,
        fps,
        np.asarray(motion["contact_indicator"], dtype=np.float64),
        np.asarray(motion["contact_rung_id"], dtype=np.int64),
        _as_names(motion["limb_names"]),
    )
    adjusted_joint_pos = original_joint_pos.copy()
    all_diagnostics = {}
    for side in ("left", "right"):
        adjusted_joint_pos, diagnostics = optimizer.optimize_side(
            original_joint_pos, adjusted_joint_pos, side
        )
        all_diagnostics[side] = diagnostics
        print(f"[INFO] {side} IK diagnostics: {diagnostics}")

    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset,
        root_pos,
        root_quat_wxyz,
        adjusted_joint_pos,
        joint_names,
        body_names,
    )
    output = dict(motion)
    output.update(
        {
            "joint_pos": adjusted_joint_pos.astype(np.float32),
            "joint_vel": finite_difference(adjusted_joint_pos, fps),
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": finite_difference(body_pos_w, fps),
            "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
            "motion_hand_ik_source_file": np.asarray([str(input_file)]),
            "motion_hand_ik_method": np.asarray(
                ["sequential_convex_sphere_box_arm_only_contact_aware_route"]
            ),
            "motion_hand_ik_clearance": np.asarray([COLLISION_CLEARANCE], dtype=np.float32),
            "motion_hand_ik_soft_joint_limit_factor": np.asarray(
                [SOFT_JOINT_LIMIT_FACTOR], dtype=np.float32
            ),
            "motion_hand_ik_objective_weights": np.asarray(
                [NOMINAL_WEIGHT, VELOCITY_CORRECTION_WEIGHT, ACCELERATION_CORRECTION_WEIGHT],
                dtype=np.float32,
            ),
            "motion_hand_ik_derivative_limits": np.asarray(
                [MAX_ARM_SPEED, MAX_ARM_ACCELERATION, MAX_ARM_JERK], dtype=np.float32
            ),
            "motion_hand_ik_adjusted_joint_names": np.asarray(
                [name for _, joint_names_for_side in ARM_SPECS.values() for name in joint_names_for_side]
            ),
            "motion_hand_ik_iterations": np.asarray(
                [all_diagnostics[side]["iterations"] for side in ("left", "right")], dtype=np.int64
            ),
            "motion_hand_ik_joint_delta_rms": np.asarray(
                [all_diagnostics[side]["joint_delta_rms"] for side in ("left", "right")],
                dtype=np.float32,
            ),
            "motion_hand_ik_joint_delta_max": np.asarray(
                [all_diagnostics[side]["joint_delta_max"] for side in ("left", "right")],
                dtype=np.float32,
            ),
        }
    )
    output.update(_regenerate_contact_labels(output))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **output)
    print(f"[INFO] Wrote arm-only sphere-hand IK motion: {output_file}")


if __name__ == "__main__":
    main()
