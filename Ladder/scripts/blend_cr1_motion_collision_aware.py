"""Blend a processed CR1 trajectory toward its source without worsening penetration."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
SCRIPT_DIR = REPO_ROOT / "Ladder" / "scripts"
for path in (SOURCE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from detect_cr1_reset_valid_frames import (
    DEFAULT_GROUND_Z,
    DEFAULT_LADDER_ASSET,
    DEFAULT_LADDER_POS,
    DEFAULT_LADDER_YAW_DEG,
    DEFAULT_ROBOT_ASSET,
    _build_collision_model,
)
from ladder.utils.urdf_kinematics import finite_difference, forward_kinematics, quaternion_angular_velocity
from ladder.utils.wood_step_ladder import load_wood_step_box_specs


BLEND_LEVELS = np.asarray([1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.1, 0.05, 0.0])
CR1_JOINT_GROUPS = (
    (
        "left_arm",
        (
            "left_shoulder_y_joint",
            "left_shoulder_x_joint",
            "left_shoulder_z_joint",
            "left_elbow_joint",
        ),
    ),
    (
        "right_arm",
        (
            "right_shoulder_y_joint",
            "right_shoulder_x_joint",
            "right_shoulder_z_joint",
            "right_elbow_joint",
        ),
    ),
    (
        "torso",
        (
            "waist_z_joint",
            "waist_x_joint",
            "waist_y_joint",
            "neck_z_joint",
            "neck_y_joint",
        ),
    ),
    (
        "right_leg",
        (
            "right_hip_y_joint",
            "right_hip_x_joint",
            "right_hip_z_joint",
            "right_knee_joint",
            "right_ankle_y_joint",
            "right_ankle_x_joint",
        ),
    ),
    (
        "left_leg",
        (
            "left_hip_y_joint",
            "left_hip_x_joint",
            "left_hip_z_joint",
            "left_knee_joint",
            "left_ankle_y_joint",
            "left_ankle_x_joint",
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True, help="Collision-audited source motion.")
    parser.add_argument("--candidate_file", type=Path, required=True, help="Processed motion to blend toward.")
    parser.add_argument(
        "--baseline_reset_file",
        type=Path,
        default=None,
        help="Optional legacy source-penetration sidecar used only for frame-count validation.",
    )
    parser.add_argument(
        "--collision_limit_file",
        type=Path,
        default=None,
        help=(
            "Optional motion whose pair depths define collision limits while blending from input_file. "
            "Defaults to input_file."
        ),
    )
    parser.add_argument("--output_file", type=Path, required=True, help="Output blended motion.")
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT_ASSET)
    parser.add_argument("--ladder_asset", type=Path, default=DEFAULT_LADDER_ASSET)
    parser.add_argument("--ladder_x", type=float, default=DEFAULT_LADDER_POS[0])
    parser.add_argument("--ladder_y", type=float, default=DEFAULT_LADDER_POS[1])
    parser.add_argument("--ladder_z", type=float, default=DEFAULT_LADDER_POS[2])
    parser.add_argument("--ladder_yaw_deg", type=float, default=DEFAULT_LADDER_YAW_DEG)
    parser.add_argument("--ground_z", type=float, default=DEFAULT_GROUND_Z)
    parser.add_argument("--penetration_tolerance", type=float, default=0.001)
    parser.add_argument(
        "--contact_file",
        type=Path,
        default=None,
        help="Optional contact-annotated motion with contact_indicator and contact_rung_id.",
    )
    parser.add_argument(
        "--intended_support_depth_slack",
        type=float,
        default=0.0,
        help=(
            "Additional penetration allowed relative to the source only for a required limb on its "
            "target rung top surface. Zero preserves the original strict behavior."
        ),
    )
    parser.add_argument(
        "--intended_support_top_face_tolerance",
        type=float,
        default=0.001,
        help="Maximum local distance from the target box top face for support relaxation.",
    )
    parser.add_argument(
        "--ladder_depth_slack",
        type=float,
        default=0.0,
        help=(
            "Additional depth relative to the collision-limit motion for any ladder pair, excluding "
            "the ground. Intended target-top support keeps the larger of this and its support slack."
        ),
    )
    parser.add_argument("--alpha_velocity_weight", type=float, default=20.0)
    parser.add_argument("--alpha_acceleration_weight", type=float, default=200.0)
    parser.add_argument(
        "--blend_mode",
        choices=("global", "groupwise"),
        default="global",
        help=(
            "Blend the complete pose with one coefficient or independently blend the CR1 left/right "
            "arms, torso, and left/right legs. Groupwise mode requires an unchanged root trajectory."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _names(values: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(values).reshape(-1).tolist()]


@dataclass(frozen=True)
class CollisionPairDetail:
    robot_body_name: str
    environment_body_name: str
    environment_point_w: np.ndarray
    closest_axis_w: np.ndarray


def _quat_rotate_inverse_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    w = quaternion[0]
    xyz = quaternion[1:]
    return (
        (2.0 * w * w - 1.0) * vector
        - 2.0 * w * np.cross(xyz, vector)
        + 2.0 * xyz * np.dot(xyz, vector)
    )


def _is_box_top_face(
    point_w: np.ndarray,
    closest_axis_w: np.ndarray,
    spec,
    tolerance: float,
) -> bool:
    local_point = _quat_rotate_inverse_wxyz(
        np.asarray(spec.quaternion),
        np.asarray(point_w) - np.asarray(spec.position),
    )
    local_axis = _quat_rotate_inverse_wxyz(
        np.asarray(spec.quaternion),
        np.asarray(closest_axis_w),
    )
    return bool(
        abs(local_point[2] - 0.5 * spec.size[2]) <= tolerance
        and abs(local_axis[2]) >= 0.9
    )


class IntendedSupportAllowance:
    _COLLISION_BODY_TO_LIMB_INDEX = {
        "left_sphere_hand_link": 0,
        "right_sphere_hand_link": 1,
        "left_ankle_x_link": 2,
        "right_ankle_x_link": 3,
    }

    def __init__(
        self,
        contact_indicator: np.ndarray,
        contact_rung_id: np.ndarray,
        ladder_specs,
        top_face_tolerance: float,
    ):
        self.contact_indicator = np.asarray(contact_indicator) > 0.5
        self.contact_rung_id = np.asarray(contact_rung_id, dtype=np.int64)
        if self.contact_indicator.shape != self.contact_rung_id.shape:
            raise ValueError("contact_indicator and contact_rung_id shapes must match")
        if self.contact_indicator.ndim != 2 or self.contact_indicator.shape[1] != 4:
            raise ValueError("contact annotations must have shape (frames, 4)")
        self.ladder_specs = {
            f"env_ladder_{spec.name}": spec
            for spec in ladder_specs
        }
        self.top_face_tolerance = float(top_face_tolerance)

    def pair_slack(
        self,
        frame: int,
        details: dict[tuple[int, int], CollisionPairDetail],
        depth_slack: float,
    ) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for pair, detail in details.items():
            limb_index = self._COLLISION_BODY_TO_LIMB_INDEX.get(detail.robot_body_name)
            if limb_index is None or not self.contact_indicator[frame, limb_index]:
                continue
            rung_id = int(self.contact_rung_id[frame, limb_index])
            target_body_name = f"env_ladder_step_{rung_id + 1:02d}"
            if detail.environment_body_name != target_body_name:
                continue
            spec = self.ladder_specs.get(detail.environment_body_name)
            if spec is None or not _is_box_top_face(
                detail.environment_point_w,
                detail.closest_axis_w,
                spec,
                self.top_face_tolerance,
            ):
                continue
            result[pair] = depth_slack
        return result


def _ladder_pair_slack(
    details: dict[tuple[int, int], CollisionPairDetail],
    depth_slack: float,
) -> dict[tuple[int, int], float]:
    return {
        pair: depth_slack
        for pair, detail in details.items()
        if detail.environment_body_name.startswith("env_ladder_")
    }


class CollisionDepth:
    def __init__(
        self,
        mujoco,
        robot_asset: Path,
        ladder_asset: Path,
        joint_names: list[str],
        ladder_position: tuple[float, float, float],
        ladder_yaw_deg: float,
        ground_z: float,
    ):
        self.mujoco = mujoco
        self.ladder_specs = load_wood_step_box_specs(
            ladder_asset,
            asset_translation=ladder_position,
            asset_yaw_deg=ladder_yaw_deg,
        )
        ladder_positions = np.asarray(
            [spec.position for spec in self.ladder_specs], dtype=np.float64
        )
        ladder_quaternions = np.asarray(
            [spec.quaternion for spec in self.ladder_specs], dtype=np.float64
        )
        self.model = _build_collision_model(
            mujoco,
            robot_asset,
            self.ladder_specs,
            ladder_positions,
            ladder_quaternions,
            ground_z,
        )
        self.data = mujoco.MjData(self.model)
        self.model.geom_margin[:] = 0.05
        self.model.geom_gap[:] = 0.0
        self.joint_ids = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index): index
            for index in range(self.model.njnt)
        }
        missing = [name for name in joint_names if name not in self.joint_ids]
        if missing:
            raise ValueError(f"collision model is missing motion joints: {missing}")
        self.joint_names = joint_names
        self.root_address = int(self.model.jnt_qposadr[self.joint_ids["root_joint"]])
        self.environment_geoms = {
            geom
            for geom in range(self.model.ngeom)
            if (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.geom_bodyid[geom]),
                )
                or ""
            ).startswith("env_")
        }
        self.from_to = np.empty(6, dtype=np.float64)
        self.last_pair_details: dict[tuple[int, int], CollisionPairDetail] = {}

    def __call__(
        self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray
    ) -> dict[tuple[int, int], float]:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.root_address : self.root_address + 3] = root_pos
        self.data.qpos[self.root_address + 3 : self.root_address + 7] = root_quat
        for column, name in enumerate(self.joint_names):
            joint_id = self.joint_ids[name]
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = joint_pos[column]
        self.mujoco.mj_forward(self.model, self.data)

        depths: dict[tuple[int, int], float] = {}
        details: dict[tuple[int, int], CollisionPairDetail] = {}
        checked: set[tuple[int, int]] = set()
        for contact in self.data.contact:
            geom_1, geom_2 = int(contact.geom1), int(contact.geom2)
            if (geom_1 in self.environment_geoms) == (geom_2 in self.environment_geoms):
                continue
            robot_geom, environment_geom = (
                (geom_2, geom_1) if geom_1 in self.environment_geoms else (geom_1, geom_2)
            )
            pair = (robot_geom, environment_geom)
            if pair in checked:
                continue
            checked.add(pair)
            distance = self.mujoco.mj_geomDistance(
                self.model,
                self.data,
                robot_geom,
                environment_geom,
                0.05,
                self.from_to,
            )
            depths[pair] = max(0.0, -float(distance))
            robot_body_name = self.mujoco.mj_id2name(
                self.model,
                self.mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[robot_geom]),
            ) or ""
            environment_body_name = self.mujoco.mj_id2name(
                self.model,
                self.mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[environment_geom]),
            ) or ""
            closest_axis = self.from_to[3:6] - self.from_to[:3]
            axis_norm = np.linalg.norm(closest_axis)
            if axis_norm > 1.0e-12:
                closest_axis = closest_axis / axis_norm
            details[pair] = CollisionPairDetail(
                robot_body_name=robot_body_name,
                environment_body_name=environment_body_name,
                environment_point_w=self.from_to[3:6].copy(),
                closest_axis_w=closest_axis.copy(),
            )
        self.last_pair_details = details
        return depths


def _max_depth(depths: dict[tuple[int, int], float]) -> float:
    return max(depths.values(), default=0.0)


def _within_pair_limits(
    candidate: dict[tuple[int, int], float],
    source: dict[tuple[int, int], float],
    tolerance: float,
    pair_slack: dict[tuple[int, int], float] | None = None,
) -> bool:
    pair_slack = pair_slack or {}
    return all(
        depth
        <= max(
            tolerance,
            source.get(pair, 0.0) + pair_slack.get(pair, 0.0),
        )
        + 1.0e-7
        for pair, depth in candidate.items()
    )


def _relaxed_pair_count(
    candidate: dict[tuple[int, int], float],
    source: dict[tuple[int, int], float],
    tolerance: float,
    pair_slack: dict[tuple[int, int], float],
) -> int:
    return sum(
        depth > max(tolerance, source.get(pair, 0.0)) + 1.0e-7
        for pair, depth in candidate.items()
        if pair in pair_slack
    )


def _source_joint_envelope(
    robot_asset: Path,
    joint_names: list[str],
    source_joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    joints = {
        joint.attrib["name"]: joint
        for joint in ElementTree.parse(robot_asset).getroot().findall("joint")
    }
    lower = np.empty(len(joint_names), dtype=np.float64)
    upper = np.empty(len(joint_names), dtype=np.float64)
    for index, name in enumerate(joint_names):
        limit = joints[name].find("limit")
        hard_lower = float(limit.attrib["lower"])
        hard_upper = float(limit.attrib["upper"])
        lower[index] = min(hard_lower, float(np.min(source_joint_pos[:, index])))
        upper[index] = max(hard_upper, float(np.max(source_joint_pos[:, index])))
    return lower, upper


def _within_joint_envelope(
    joint_pos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    return bool(np.all(joint_pos >= lower - 1.0e-7) and np.all(joint_pos <= upper + 1.0e-7))


def _interpolate(
    alpha: float,
    source_pos: np.ndarray,
    source_quat: np.ndarray,
    source_joint: np.ndarray,
    candidate_pos: np.ndarray,
    candidate_quat: np.ndarray,
    candidate_joint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_pos = (1.0 - alpha) * source_pos + alpha * candidate_pos
    root_quat = (1.0 - alpha) * source_quat + alpha * candidate_quat
    root_quat /= np.linalg.norm(root_quat)
    joint_pos = (1.0 - alpha) * source_joint + alpha * candidate_joint
    # Collision checks must use the exact precision written to the motion NPZ.
    # Near box edges, MuJoCo can select a different contact pair after float32
    # quantization even when the unquantized state was considered safe.
    return tuple(
        value.astype(np.float32).astype(np.float64)
        for value in (root_pos, root_quat, joint_pos)
    )


def _resolve_joint_groups(joint_names: list[str]) -> list[tuple[str, np.ndarray]]:
    resolved = []
    used: set[int] = set()
    for group_name, names in CR1_JOINT_GROUPS:
        missing = [name for name in names if name not in joint_names]
        if missing:
            raise ValueError(f"joint group '{group_name}' is missing joints: {missing}")
        indexes = np.asarray([joint_names.index(name) for name in names], dtype=np.int64)
        overlap = used.intersection(indexes.tolist())
        if overlap:
            raise ValueError(f"joint group '{group_name}' overlaps joint indexes: {sorted(overlap)}")
        used.update(indexes.tolist())
        resolved.append((group_name, indexes))
    if len(used) != len(joint_names):
        ungrouped = [name for index, name in enumerate(joint_names) if index not in used]
        raise ValueError(f"groupwise blending does not cover joints: {ungrouped}")
    return resolved


def _interpolate_joint_group(
    alpha: float,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    current_joint: np.ndarray,
    source_joint: np.ndarray,
    candidate_joint: np.ndarray,
    group_indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_pos = current_joint.copy()
    joint_pos[group_indexes] = (
        (1.0 - alpha) * source_joint[group_indexes]
        + alpha * candidate_joint[group_indexes]
    )
    return tuple(
        value.astype(np.float32).astype(np.float64)
        for value in (root_pos, root_quat, joint_pos)
    )


def _regularize_alpha(upper: np.ndarray, velocity_weight: float, acceleration_weight: float) -> np.ndarray:
    upper = np.asarray(upper, dtype=np.float64)

    def objective(alpha: np.ndarray) -> tuple[float, np.ndarray]:
        velocity = np.diff(alpha)
        acceleration = np.diff(alpha, n=2)
        value = (
            np.sum(np.square(alpha - 1.0))
            + velocity_weight * np.sum(np.square(velocity))
            + acceleration_weight * np.sum(np.square(acceleration))
        )
        gradient = 2.0 * (alpha - 1.0)
        gradient[:-1] -= 2.0 * velocity_weight * velocity
        gradient[1:] += 2.0 * velocity_weight * velocity
        gradient[:-2] += 2.0 * acceleration_weight * acceleration
        gradient[1:-1] -= 4.0 * acceleration_weight * acceleration
        gradient[2:] += 2.0 * acceleration_weight * acceleration
        return float(value), gradient

    result = minimize(
        objective,
        np.minimum(upper, 0.5),
        jac=True,
        bounds=list(zip(np.zeros_like(upper), upper)),
        method="L-BFGS-B",
        options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 1000, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"alpha regularization did not converge: {result.message}")
    alpha = result.x.copy()
    alpha[upper == 0.0] = 0.0
    return np.minimum(alpha, upper)


def main() -> None:
    args = _parse_args()
    input_file = args.input_file.expanduser().resolve()
    candidate_file = args.candidate_file.expanduser().resolve()
    baseline_file = (
        args.baseline_reset_file.expanduser().resolve()
        if args.baseline_reset_file is not None
        else None
    )
    collision_limit_file = (
        args.collision_limit_file.expanduser().resolve()
        if args.collision_limit_file is not None
        else input_file
    )
    contact_file = (
        args.contact_file.expanduser().resolve()
        if args.contact_file is not None
        else None
    )
    output_file = args.output_file.expanduser().resolve()
    robot_asset = args.robot_asset.expanduser().resolve()
    ladder_asset = args.ladder_asset.expanduser().resolve()
    for path in (
        input_file,
        candidate_file,
        robot_asset,
        ladder_asset,
        baseline_file,
        collision_limit_file,
        contact_file,
    ):
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_file}. Pass --overwrite to replace it.")
    if args.intended_support_depth_slack < 0.0:
        raise ValueError("intended_support_depth_slack must be non-negative")
    if args.ladder_depth_slack < 0.0:
        raise ValueError("ladder_depth_slack must be non-negative")
    if args.intended_support_top_face_tolerance <= 0.0:
        raise ValueError("intended_support_top_face_tolerance must be positive")
    if args.intended_support_depth_slack > 0.0 and contact_file is None:
        raise ValueError("--contact_file is required when intended support depth slack is enabled")

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Run collision-aware blending in the retarget environment.") from exc

    with np.load(input_file, allow_pickle=False) as source_data:
        source = {key: source_data[key] for key in source_data.files}
    with np.load(candidate_file, allow_pickle=False) as candidate_data:
        candidate = {key: candidate_data[key] for key in candidate_data.files}
    if collision_limit_file == input_file:
        collision_limit_motion = source
    else:
        with np.load(collision_limit_file, allow_pickle=False) as collision_limit_data:
            collision_limit_motion = {
                key: collision_limit_data[key] for key in collision_limit_data.files
            }
    contact_data = None
    if contact_file is not None:
        with np.load(contact_file, allow_pickle=False) as loaded_contact_data:
            contact_data = {key: loaded_contact_data[key] for key in loaded_contact_data.files}
    baseline_depth = None
    if baseline_file is not None:
        with np.load(baseline_file, allow_pickle=False) as baseline_data:
            baseline_depth = np.asarray(
                baseline_data["reset_max_penetration_depth"], dtype=np.float64
            )

    joint_names = _names(source["joint_names"])
    body_names = _names(source["body_names"])
    source_joint_names = _names(source["source_joint_names"])
    if joint_names != _names(candidate["joint_names"]) or body_names != _names(candidate["body_names"]):
        raise ValueError("source and candidate joint/body names must match exactly")
    frame_count = source["joint_pos"].shape[0]
    if candidate["joint_pos"].shape[0] != frame_count:
        raise ValueError("source and candidate frame counts must match")
    if (
        joint_names != _names(collision_limit_motion["joint_names"])
        or body_names != _names(collision_limit_motion["body_names"])
    ):
        raise ValueError("collision-limit and source joint/body names must match exactly")
    if collision_limit_motion["joint_pos"].shape[0] != frame_count:
        raise ValueError("collision-limit and source frame counts must match")
    if contact_data is not None:
        for key in ("contact_indicator", "contact_rung_id"):
            if key not in contact_data:
                raise KeyError(f"contact file is missing required key: {key}")
        if np.asarray(contact_data["contact_indicator"]).shape != (frame_count, 4):
            raise ValueError("contact file frame count must match source and candidate motions")
    if baseline_depth is not None and baseline_depth.shape != (frame_count,):
        raise ValueError("source motion and penetration sidecar frame counts must match")

    root_index = body_names.index("base_link")
    source_root_pos = np.asarray(source["body_pos_w"][:, root_index], dtype=np.float64)
    source_root_quat = np.asarray(source["body_quat_w"][:, root_index], dtype=np.float64)
    source_joint_pos = np.asarray(source["joint_pos"], dtype=np.float64)
    candidate_root_pos = np.asarray(candidate["body_pos_w"][:, root_index], dtype=np.float64)
    candidate_root_quat = np.asarray(candidate["body_quat_w"][:, root_index], dtype=np.float64)
    candidate_joint_pos = np.asarray(candidate["joint_pos"], dtype=np.float64)
    collision_limit_root_pos = np.asarray(
        collision_limit_motion["body_pos_w"][:, root_index], dtype=np.float64
    )
    collision_limit_root_quat = np.asarray(
        collision_limit_motion["body_quat_w"][:, root_index], dtype=np.float64
    )
    collision_limit_joint_pos = np.asarray(
        collision_limit_motion["joint_pos"], dtype=np.float64
    )
    joint_lower, joint_upper = _source_joint_envelope(
        robot_asset,
        joint_names,
        source_joint_pos,
    )
    candidate_root_quat = np.where(
        (np.sum(source_root_quat * candidate_root_quat, axis=1) < 0.0)[:, None],
        -candidate_root_quat,
        candidate_root_quat,
    )

    ladder_position = (args.ladder_x, args.ladder_y, args.ladder_z)
    collision_depth = CollisionDepth(
        mujoco,
        robot_asset,
        ladder_asset,
        joint_names,
        ladder_position,
        args.ladder_yaw_deg,
        args.ground_z,
    )
    support_allowance = None
    if args.intended_support_depth_slack > 0.0:
        if "contact_annotation_ladder_asset_translation" in contact_data:
            annotation_ladder_position = np.asarray(
                contact_data["contact_annotation_ladder_asset_translation"], dtype=np.float64
            ).reshape(3)
            if not np.allclose(annotation_ladder_position, ladder_position, atol=1.0e-6, rtol=0.0):
                raise ValueError(
                    "contact annotation ladder position does not match collision ladder position: "
                    f"{annotation_ladder_position.tolist()} vs {list(ladder_position)}"
                )
        if "contact_annotation_ladder_asset_yaw_deg" in contact_data:
            annotation_ladder_yaw = float(
                np.asarray(contact_data["contact_annotation_ladder_asset_yaw_deg"]).reshape(-1)[0]
            )
            if not np.isclose(annotation_ladder_yaw, args.ladder_yaw_deg, atol=1.0e-6):
                raise ValueError(
                    "contact annotation ladder yaw does not match collision ladder yaw: "
                    f"{annotation_ladder_yaw} vs {args.ladder_yaw_deg}"
                )
        support_allowance = IntendedSupportAllowance(
            contact_data["contact_indicator"],
            contact_data["contact_rung_id"],
            collision_depth.ladder_specs,
            args.intended_support_top_face_tolerance,
        )

    def query_candidate(
        frame: int,
        state: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        depths = collision_depth(*state)
        pair_slack = _ladder_pair_slack(
            collision_depth.last_pair_details,
            args.ladder_depth_slack,
        )
        if support_allowance is not None:
            support_slack = support_allowance.pair_slack(
                frame,
                collision_depth.last_pair_details,
                args.intended_support_depth_slack,
            )
            for pair, value in support_slack.items():
                pair_slack[pair] = max(pair_slack.get(pair, 0.0), value)
        return depths, pair_slack

    collision_limit_pair_depths = [
        collision_depth(
            collision_limit_root_pos[frame],
            collision_limit_root_quat[frame],
            collision_limit_joint_pos[frame],
        )
        for frame in range(frame_count)
    ]
    source_pair_depths: list[dict[tuple[int, int], float]] = []
    source_pair_slack: list[dict[tuple[int, int], float]] = []
    for frame in range(frame_count):
        depths, pair_slack = query_candidate(
            frame,
            (source_root_pos[frame], source_root_quat[frame], source_joint_pos[frame]),
        )
        if not _within_pair_limits(
            depths,
            collision_limit_pair_depths[frame],
            args.penetration_tolerance,
            pair_slack,
        ):
            raise ValueError(
                f"input source frame {frame} violates the selected collision-limit motion"
            )
        source_pair_depths.append(depths)
        source_pair_slack.append(pair_slack)
    group_names: list[str] = []
    group_alpha = np.empty((frame_count, 0), dtype=np.float64)
    group_upper = np.empty((frame_count, 0), dtype=np.float64)
    final_relaxed_pair_count = np.zeros(frame_count, dtype=np.int64)
    if args.blend_mode == "global":
        upper = np.zeros(frame_count, dtype=np.float64)
        for frame in range(frame_count):
            for alpha_value in BLEND_LEVELS:
                state = _interpolate(
                    float(alpha_value),
                    source_root_pos[frame],
                    source_root_quat[frame],
                    source_joint_pos[frame],
                    candidate_root_pos[frame],
                    candidate_root_quat[frame],
                    candidate_joint_pos[frame],
                )
                if not _within_joint_envelope(state[2], joint_lower, joint_upper):
                    continue
                depths, pair_slack = query_candidate(frame, state)
                if _within_pair_limits(
                    depths,
                    collision_limit_pair_depths[frame],
                    args.penetration_tolerance,
                    pair_slack,
                ):
                    upper[frame] = alpha_value
                    break

        alpha = _regularize_alpha(upper, args.alpha_velocity_weight, args.alpha_acceleration_weight)
        root_pos = np.empty_like(source_root_pos)
        root_quat = np.empty_like(source_root_quat)
        joint_pos = np.empty_like(source_joint_pos)
        final_depth = np.zeros(frame_count, dtype=np.float64)
        for frame in range(frame_count):
            state = _interpolate(
                float(alpha[frame]),
                source_root_pos[frame],
                source_root_quat[frame],
                source_joint_pos[frame],
                candidate_root_pos[frame],
                candidate_root_quat[frame],
                candidate_joint_pos[frame],
            )
            depths, pair_slack = query_candidate(frame, state)
            if not _within_joint_envelope(
                state[2], joint_lower, joint_upper
            ) or not _within_pair_limits(
                depths,
                collision_limit_pair_depths[frame],
                args.penetration_tolerance,
                pair_slack,
            ):
                alpha[frame] = 0.0
                state = (source_root_pos[frame], source_root_quat[frame], source_joint_pos[frame])
                depths = source_pair_depths[frame]
                pair_slack = source_pair_slack[frame]
            root_pos[frame], root_quat[frame], joint_pos[frame] = state
            final_depth[frame] = _max_depth(depths)
            final_relaxed_pair_count[frame] = _relaxed_pair_count(
                depths,
                collision_limit_pair_depths[frame],
                args.penetration_tolerance,
                pair_slack,
            )
    else:
        if not np.allclose(candidate_root_pos, source_root_pos, atol=1.0e-7, rtol=0.0):
            raise ValueError("groupwise blending requires candidate and source root positions to match")
        if not np.allclose(candidate_root_quat, source_root_quat, atol=1.0e-7, rtol=0.0):
            raise ValueError("groupwise blending requires candidate and source root orientations to match")
        resolved_groups = _resolve_joint_groups(joint_names)
        group_names = [name for name, _ in resolved_groups]
        group_alpha = np.zeros((frame_count, len(resolved_groups)), dtype=np.float64)
        group_upper = np.zeros_like(group_alpha)
        root_pos = source_root_pos.copy()
        root_quat = source_root_quat.copy()
        joint_pos = source_joint_pos.copy()

        for group_index, (group_name, joint_indexes) in enumerate(resolved_groups):
            upper_for_group = group_upper[:, group_index]
            for frame in range(frame_count):
                for alpha_value in BLEND_LEVELS:
                    state = _interpolate_joint_group(
                        float(alpha_value),
                        source_root_pos[frame],
                        source_root_quat[frame],
                        joint_pos[frame],
                        source_joint_pos[frame],
                        candidate_joint_pos[frame],
                        joint_indexes,
                    )
                    if not _within_joint_envelope(state[2], joint_lower, joint_upper):
                        continue
                    depths, pair_slack = query_candidate(frame, state)
                    if _within_pair_limits(
                        depths,
                        collision_limit_pair_depths[frame],
                        args.penetration_tolerance,
                        pair_slack,
                    ):
                        upper_for_group[frame] = alpha_value
                        break

            alpha_for_group = _regularize_alpha(
                upper_for_group,
                args.alpha_velocity_weight,
                args.alpha_acceleration_weight,
            )
            for frame in range(frame_count):
                state = _interpolate_joint_group(
                    float(alpha_for_group[frame]),
                    source_root_pos[frame],
                    source_root_quat[frame],
                    joint_pos[frame],
                    source_joint_pos[frame],
                    candidate_joint_pos[frame],
                    joint_indexes,
                )
                depths, pair_slack = query_candidate(frame, state)
                if not _within_joint_envelope(
                    state[2], joint_lower, joint_upper
                ) or not _within_pair_limits(
                    depths,
                    collision_limit_pair_depths[frame],
                    args.penetration_tolerance,
                    pair_slack,
                ):
                    alpha_for_group[frame] = 0.0
                    state = (
                        source_root_pos[frame],
                        source_root_quat[frame],
                        joint_pos[frame],
                    )
                joint_pos[frame] = state[2]
            group_alpha[:, group_index] = alpha_for_group
            print(
                f"[INFO] Group {group_name}: alpha min/mean/max="
                f"{alpha_for_group.min():.3f}/{alpha_for_group.mean():.3f}/"
                f"{alpha_for_group.max():.3f}; frames below 0.1="
                f"{int(np.sum(alpha_for_group < 0.1))}"
            )

        final_depth = np.zeros(frame_count, dtype=np.float64)
        for frame in range(frame_count):
            state = (root_pos[frame], root_quat[frame], joint_pos[frame])
            depths, pair_slack = query_candidate(frame, state)
            if not _within_joint_envelope(
                joint_pos[frame], joint_lower, joint_upper
            ) or not _within_pair_limits(
                depths,
                collision_limit_pair_depths[frame],
                args.penetration_tolerance,
                pair_slack,
            ):
                group_alpha[frame] = 0.0
                joint_pos[frame] = source_joint_pos[frame]
                depths = source_pair_depths[frame]
                pair_slack = source_pair_slack[frame]
            final_depth[frame] = _max_depth(depths)
            final_relaxed_pair_count[frame] = _relaxed_pair_count(
                depths,
                collision_limit_pair_depths[frame],
                args.penetration_tolerance,
                pair_slack,
            )
        alpha = np.mean(group_alpha, axis=1)
        upper = np.mean(group_upper, axis=1)

    source_indexes = [joint_names.index(name) for name in source_joint_names]
    body_pos_w, body_quat_w = forward_kinematics(
        robot_asset,
        root_pos,
        root_quat,
        joint_pos[:, source_indexes],
        source_joint_names,
        body_names,
    )
    fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    output = {
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": finite_difference(joint_pos, fps),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": finite_difference(body_pos_w, fps),
        "body_ang_vel_w": quaternion_angular_velocity(body_quat_w, fps),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "source_joint_names": np.asarray(source_joint_names),
        "collision_aware_blend_source_file": np.asarray([str(input_file)]),
        "collision_aware_blend_candidate_file": np.asarray([str(candidate_file)]),
        "collision_aware_blend_collision_limit_file": np.asarray(
            [str(collision_limit_file)]
        ),
        "collision_aware_blend_alpha": alpha.astype(np.float32),
        "collision_aware_blend_alpha_upper": upper.astype(np.float32),
        "collision_aware_blend_mode": np.asarray([args.blend_mode]),
        "collision_aware_blend_group_names": np.asarray(group_names),
        "collision_aware_blend_group_alpha": group_alpha.astype(np.float32),
        "collision_aware_blend_group_alpha_upper": group_upper.astype(np.float32),
        "collision_aware_blend_depth": final_depth.astype(np.float32),
        "collision_aware_blend_joint_lower_envelope": joint_lower.astype(np.float32),
        "collision_aware_blend_joint_upper_envelope": joint_upper.astype(np.float32),
        "collision_aware_blend_relaxed_pair_count": final_relaxed_pair_count,
        "collision_aware_blend_intended_support_contact_file": np.asarray(
            [str(contact_file) if contact_file is not None else ""]
        ),
        "collision_aware_blend_intended_support_depth_slack": np.asarray(
            [args.intended_support_depth_slack], dtype=np.float32
        ),
        "collision_aware_blend_intended_support_top_face_tolerance": np.asarray(
            [args.intended_support_top_face_tolerance], dtype=np.float32
        ),
        "collision_aware_blend_ladder_depth_slack": np.asarray(
            [args.ladder_depth_slack], dtype=np.float32
        ),
        "collision_aware_blend_alpha_regularization": np.asarray(
            [args.alpha_velocity_weight, args.alpha_acceleration_weight], dtype=np.float32
        ),
        "collision_aware_blend_ladder_position": np.asarray(ladder_position, dtype=np.float32),
        "collision_aware_blend_ladder_yaw_deg": np.asarray(
            [args.ladder_yaw_deg], dtype=np.float32
        ),
        "collision_aware_blend_ground_z": np.asarray([args.ground_z], dtype=np.float32),
    }
    for key, value in candidate.items():
        if key.startswith(
            ("world_alignment_", "motion_processing_", "standing_transition_")
        ):
            output.setdefault(key, value)
    for key, value in source.items():
        if key.startswith(("motion_export_", "collision_aware_blend_")):
            output.setdefault(key, value)
    if contact_data is not None:
        for key, value in contact_data.items():
            if key.startswith("contact_") or key in (
                "limb_names",
                "limb_body_names",
                "limb_body_indexes",
            ):
                output.setdefault(key, value)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **output)
    print(f"[INFO] Wrote collision-aware blend: {output_file}")
    if args.blend_mode == "global":
        print(
            f"[INFO] Alpha min/mean/max={alpha.min():.3f}/{alpha.mean():.3f}/{alpha.max():.3f}; "
            f"frames below 0.1={int(np.sum(alpha < 0.1))}"
        )
    else:
        print(
            f"[INFO] Mean group alpha min/mean/max="
            f"{alpha.min():.3f}/{alpha.mean():.3f}/{alpha.max():.3f}"
        )
    print(
        f"[INFO] Strict-valid frames={int(np.sum(final_depth <= args.penetration_tolerance))}/{frame_count}; "
        f"worst penetration={1000.0 * final_depth.max():.3f} mm"
    )
    if support_allowance is not None:
        print(
            f"[INFO] Intended-support relaxation used in "
            f"{int(np.sum(final_relaxed_pair_count > 0))}/{frame_count} frames; "
            f"relaxed final pairs={int(np.sum(final_relaxed_pair_count))}"
        )


if __name__ == "__main__":
    main()
