"""Detect CR1 reference frames that penetrate the training ground or ladder."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from ladder.utils.ladder_presets import HOLOSOMA_CR1_WOOD_STEP
from ladder.utils.wood_step_ladder import (
    WoodStepBoxSpec,
    load_wood_step_box_specs,
    randomized_wood_step_box_poses,
    thin_front_side_board_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "restored_prefix_v300_stride2_slow50"
    / "smooth_start"
    / "climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_train_ready.npz"
)
DEFAULT_ROBOT_ASSET = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)
DEFAULT_LADDER_ASSET = (
    REPO_ROOT / "assets" / "ladders" / "ladder_A_aframe" / "wood_step_ladder_sim.urdf"
)
DEFAULT_LADDER_POS = HOLOSOMA_CR1_WOOD_STEP.asset_translation
DEFAULT_LADDER_YAW_DEG = HOLOSOMA_CR1_WOOD_STEP.asset_yaw_deg
DEFAULT_GROUND_Z = -0.03
SPHERE_HAND_BODY_NAMES = ("left_sphere_hand_link", "right_sphere_hand_link")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build geometry-conditioned ladder reset-valid masks from signed "
            "robot/environment penetration depths."
        )
    )
    parser.add_argument("--motion_file", type=Path, default=DEFAULT_MOTION, help="CR1 reference motion npz.")
    parser.add_argument("--output_file", type=Path, required=True, help="Output reset-validity sidecar npz.")
    parser.add_argument(
        "--motion_output_file",
        type=Path,
        default=None,
        help="Optional compressed motion npz with the detector fields embedded.",
    )
    parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT_ASSET, help="CR1 collision URDF.")
    parser.add_argument(
        "--ladder_asset", type=Path, default=DEFAULT_LADDER_ASSET, help="Primitive collision ladder URDF."
    )
    parser.add_argument(
        "--inclination_delta_deg",
        type=float,
        default=1.0,
        help="Symmetric ladder-inclination envelope in degrees.",
    )
    parser.add_argument(
        "--spacing_delta",
        type=float,
        default=0.01,
        help="Symmetric rung-spacing envelope in meters.",
    )
    parser.add_argument(
        "--geometry_grid_steps",
        type=int,
        default=5,
        help="Odd number of samples per inclination/spacing axis; must be at least 3.",
    )
    parser.add_argument(
        "--penetration_tolerance",
        type=float,
        default=0.001,
        help="Maximum allowed penetration depth in meters.",
    )
    parser.add_argument(
        "--candidate_distance",
        type=float,
        default=0.05,
        help="Broad-phase contact margin in meters; must exceed penetration_tolerance.",
    )
    parser.add_argument(
        "--side_board_depth_fraction",
        type=float,
        default=1.0,
        help="Remaining depth fraction for both front side boards; current ThinRails uses 1/3.",
    )
    parser.add_argument(
        "--ignore_sphere_hands",
        action="store_true",
        help="Ignore environment penetrations from left/right sphere-hand collision bodies.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    return parser.parse_args()


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _as_names(values: np.ndarray) -> list[str]:
    items = np.asarray(values).reshape(-1).tolist()
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in items]


def _append_fixed_joint(
    root: ElementTree.Element,
    name: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> None:
    joint = ElementTree.SubElement(root, "joint", {"name": name, "type": "fixed"})
    ElementTree.SubElement(joint, "parent", {"link": parent})
    ElementTree.SubElement(joint, "child", {"link": child})
    ElementTree.SubElement(
        joint,
        "origin",
        {
            "xyz": " ".join(f"{value:.12g}" for value in xyz),
            "rpy": " ".join(f"{value:.12g}" for value in rpy),
        },
    )


def _quat_wxyz_to_rpy(quaternion: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(float(pitch_sine))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


def _build_collision_model(
    mujoco,
    robot_asset: Path,
    ladder_specs: tuple[WoodStepBoxSpec, ...],
    ladder_positions: np.ndarray,
    ladder_quaternions: np.ndarray,
    ground_z: float,
):
    """Load the robot collision URDF and add the sampled 18-box ladder and ground."""

    root = ElementTree.parse(robot_asset).getroot()
    compiler = root.find("./mujoco/compiler")
    if compiler is None:
        mujoco_element = root.find("./mujoco")
        if mujoco_element is None:
            mujoco_element = ElementTree.SubElement(root, "mujoco")
        compiler = ElementTree.SubElement(mujoco_element, "compiler")
    compiler.set("meshdir", ".")
    compiler.set("discardvisual", "true")
    for mesh in root.findall(".//mesh"):
        mesh_path = (robot_asset.parent / mesh.attrib["filename"]).resolve()
        mesh.set("filename", str(mesh_path))

    if root.find("./link[@name='world']") is not None:
        raise ValueError("robot URDF already contains a link named 'world'")
    ElementTree.SubElement(root, "link", {"name": "world"})
    root_joint = ElementTree.SubElement(root, "joint", {"name": "root_joint", "type": "floating"})
    ElementTree.SubElement(root_joint, "parent", {"link": "world"})
    ElementTree.SubElement(root_joint, "child", {"link": "base_link"})
    ElementTree.SubElement(root_joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    ground_link = ElementTree.SubElement(root, "link", {"name": "env_ground"})
    ground_collision = ElementTree.SubElement(ground_link, "collision", {"name": "env_ground_collision"})
    ElementTree.SubElement(
        ground_collision,
        "origin",
        {"xyz": f"0 0 {ground_z - 0.01:.12g}", "rpy": "0 0 0"},
    )
    ground_geometry = ElementTree.SubElement(ground_collision, "geometry")
    ElementTree.SubElement(ground_geometry, "box", {"size": "20 20 0.02"})
    _append_fixed_joint(root, "env_ground_joint", "world", "env_ground", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    for spec, position, quaternion in zip(
        ladder_specs, ladder_positions, ladder_quaternions, strict=True
    ):
        link_name = f"env_ladder_{spec.name}"
        link = ElementTree.SubElement(root, "link", {"name": link_name})
        collision = ElementTree.SubElement(
            link, "collision", {"name": f"{spec.name}_collision"}
        )
        geometry = ElementTree.SubElement(collision, "geometry")
        ElementTree.SubElement(
            geometry,
            "box",
            {"size": " ".join(f"{value:.12g}" for value in spec.size)},
        )
        _append_fixed_joint(
            root,
            f"{link_name}_joint",
            "world",
            link_name,
            tuple(position.tolist()),
            _quat_wxyz_to_rpy(quaternion),
        )

    return mujoco.MjModel.from_xml_string(ElementTree.tostring(root, encoding="unicode"))


def _invalid_runs(valid_mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], ~valid_mask, [False])).astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [(int(start), int(end - 1)) for start, end in zip(starts, ends, strict=True)]


def _evaluate_geometry(
    mujoco,
    *,
    label: str,
    model,
    joint_pos: np.ndarray,
    joint_names: list[str],
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    candidate_distance: float,
    penetration_tolerance: float,
    ignored_robot_bodies: set[str],
) -> dict[str, object]:
    sim_data = mujoco.MjData(model)
    model.geom_margin[:] = candidate_distance
    model.geom_gap[:] = 0.0

    model_joint_ids = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index): index
        for index in range(model.njnt)
    }
    missing_joints = [name for name in joint_names if name not in model_joint_ids]
    if missing_joints:
        raise ValueError(f"motion joints are missing from collision model: {missing_joints}")
    root_joint_id = model_joint_ids.get("root_joint")
    if root_joint_id is None:
        raise ValueError("collision model is missing floating joint 'root_joint'")
    root_qpos_address = int(model.jnt_qposadr[root_joint_id])

    environment_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
            or ""
        ).startswith("env_")
    }
    if not environment_geom_ids:
        raise RuntimeError("collision model has no environment geoms")

    num_frames = joint_pos.shape[0]
    max_depth = np.zeros(num_frames, dtype=np.float32)
    pair_count = np.zeros(num_frames, dtype=np.int16)
    worst_robot_body = np.full(num_frames, "", dtype="<U64")
    worst_environment_geom = np.full(num_frames, "", dtype="<U64")
    penetrating_pairs: Counter[tuple[str, str]] = Counter()
    from_to = np.zeros(6, dtype=np.float64)

    for frame in range(num_frames):
        sim_data.qpos[:] = model.qpos0
        sim_data.qpos[root_qpos_address : root_qpos_address + 3] = root_pos[frame]
        sim_data.qpos[root_qpos_address + 3 : root_qpos_address + 7] = root_quat_wxyz[frame]
        for joint_column, joint_name in enumerate(joint_names):
            joint_id = model_joint_ids[joint_name]
            sim_data.qpos[int(model.jnt_qposadr[joint_id])] = joint_pos[frame, joint_column]
        mujoco.mj_forward(model, sim_data)

        checked_pairs: set[tuple[int, int]] = set()
        for contact_index in range(sim_data.ncon):
            geom_1 = int(sim_data.contact[contact_index].geom1)
            geom_2 = int(sim_data.contact[contact_index].geom2)
            if (geom_1 in environment_geom_ids) == (geom_2 in environment_geom_ids):
                continue
            robot_geom, environment_geom = (
                (geom_2, geom_1) if geom_1 in environment_geom_ids else (geom_1, geom_2)
            )
            robot_body = (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[robot_geom])
                )
                or f"body_{int(model.geom_bodyid[robot_geom])}"
            )
            if robot_body in ignored_robot_bodies:
                continue
            pair = (robot_geom, environment_geom)
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            distance = mujoco.mj_geomDistance(
                model,
                sim_data,
                robot_geom,
                environment_geom,
                candidate_distance,
                from_to,
            )
            if distance >= -penetration_tolerance:
                continue

            depth = -float(distance)
            pair_count[frame] += 1
            environment_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, environment_geom)
                or f"geom_{environment_geom}"
            )
            penetrating_pairs[(robot_body, environment_name)] += 1
            if depth > max_depth[frame]:
                max_depth[frame] = depth
                worst_robot_body[frame] = robot_body
                worst_environment_geom[frame] = environment_name

        if frame % 100 == 0 or frame == num_frames - 1:
            print(
                f"[PROGRESS][{label}] frame {frame:4d}/{num_frames - 1}: "
                f"pairs={pair_count[frame]}, max_depth={max_depth[frame]:.5f} m"
            )

    valid_mask = pair_count == 0
    return {
        "valid_mask": valid_mask,
        "max_depth": max_depth,
        "pair_count": pair_count,
        "worst_robot_body": worst_robot_body,
        "worst_environment_geom": worst_environment_geom,
        "penetrating_pairs": penetrating_pairs,
    }


def _summarize_result(label: str, result: dict[str, object]) -> None:
    valid_mask = result["valid_mask"]
    max_depth = result["max_depth"]
    worst_robot_body = result["worst_robot_body"]
    worst_environment_geom = result["worst_environment_geom"]
    valid_count = int(valid_mask.sum())
    print(
        f"[INFO][{label}] valid={valid_count}/{len(valid_mask)} "
        f"({valid_count / len(valid_mask):.3%}), invalid={len(valid_mask) - valid_count}"
    )
    if (~valid_mask).any():
        worst_frame = int(np.argmax(max_depth))
        print(
            f"[INFO][{label}] worst frame={worst_frame}, depth={max_depth[worst_frame]:.5f} m, "
            f"pair={worst_robot_body[worst_frame]} / {worst_environment_geom[worst_frame]}"
        )


def _write_merged_motion(
    motion_file: Path,
    motion_output_file: Path,
    detector_output: dict[str, np.ndarray],
) -> None:
    with np.load(motion_file, allow_pickle=False) as source:
        merged = {key: source[key] for key in source.files}
    merged.update(detector_output)
    motion_output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(motion_output_file, **merged)


def main() -> None:
    args = _parse_args()
    if args.penetration_tolerance < 0.0:
        raise ValueError("penetration_tolerance must be non-negative")
    if args.candidate_distance <= args.penetration_tolerance:
        raise ValueError("candidate_distance must be greater than penetration_tolerance")
    if not 0.0 < args.side_board_depth_fraction <= 1.0:
        raise ValueError("side_board_depth_fraction must lie in (0, 1]")
    if args.inclination_delta_deg < 0.0:
        raise ValueError("inclination_delta_deg must be non-negative")
    if args.spacing_delta < 0.0:
        raise ValueError("spacing_delta must be non-negative")
    if args.geometry_grid_steps < 3 or args.geometry_grid_steps % 2 == 0:
        raise ValueError("geometry_grid_steps must be an odd integer of at least 3")

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This detector requires MuJoCo. Run it with the local miniforge 'retarget' environment."
        ) from exc

    motion_file = _require_file(args.motion_file, "motion file")
    robot_asset = _require_file(args.robot_asset, "robot asset")
    ladder_asset = _require_file(args.ladder_asset, "ladder asset")
    output_file = args.output_file.expanduser().resolve()
    motion_output_file = (
        args.motion_output_file.expanduser().resolve() if args.motion_output_file is not None else None
    )
    for path in (output_file, motion_output_file):
        if path is not None and path.exists() and not args.overwrite:
            raise FileExistsError(f"output file exists: {path}. Pass --overwrite to replace it.")
    if motion_output_file == motion_file:
        raise ValueError("motion_output_file must not overwrite motion_file")

    with np.load(motion_file, allow_pickle=False) as data:
        required = ("joint_pos", "joint_names", "body_pos_w", "body_quat_w", "body_names")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"motion is missing required keys: {missing}")
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        joint_names = _as_names(data["joint_names"])
        body_names = _as_names(data["body_names"])
        root_index = body_names.index("base_link")
        root_pos = np.asarray(data["body_pos_w"][:, root_index], dtype=np.float64)
        root_quat_wxyz = np.asarray(data["body_quat_w"][:, root_index], dtype=np.float64)
        ladder_origin = tuple(
            np.asarray(
                data.get("contact_annotation_ladder_origin", HOLOSOMA_CR1_WOOD_STEP.target_origin),
                dtype=np.float64,
            ).reshape(3)
        )
        ladder_asset_position = tuple(
            np.asarray(
                data.get(
                    "contact_annotation_ladder_asset_translation",
                    HOLOSOMA_CR1_WOOD_STEP.asset_translation,
                ),
                dtype=np.float64,
            ).reshape(3)
        )
        ladder_asset_yaw_deg = float(
            np.asarray(
                data.get(
                    "contact_annotation_ladder_asset_yaw_deg",
                    [HOLOSOMA_CR1_WOOD_STEP.asset_yaw_deg],
                )
            ).reshape(-1)[0]
        )

    source_phi = math.radians(HOLOSOMA_CR1_WOOD_STEP.phi_deg)
    source_spacing = HOLOSOMA_CR1_WOOD_STEP.rung_spacing
    ladder_specs = load_wood_step_box_specs(
        ladder_asset,
        asset_translation=ladder_asset_position,
        asset_yaw_deg=ladder_asset_yaw_deg,
    )
    ladder_specs = thin_front_side_board_specs(
        ladder_specs,
        remaining_depth_fraction=args.side_board_depth_fraction,
    )
    ignored_robot_bodies = set(SPHERE_HAND_BODY_NAMES) if args.ignore_sphere_hands else set()

    inclination_delta = math.radians(args.inclination_delta_deg)
    inclination_grid = np.linspace(
        -inclination_delta, inclination_delta, args.geometry_grid_steps, dtype=np.float64
    )
    spacing_grid = np.linspace(
        -args.spacing_delta, args.spacing_delta, args.geometry_grid_steps, dtype=np.float64
    )
    corner_deltas = np.asarray(
        [
            (-inclination_delta, -args.spacing_delta),
            (-inclination_delta, args.spacing_delta),
            (inclination_delta, -args.spacing_delta),
            (inclination_delta, args.spacing_delta),
        ],
        dtype=np.float64,
    )
    results: dict[tuple[int, int], dict[str, object]] = {}
    for inclination_index, delta_phi in enumerate(inclination_grid):
        for spacing_index, delta_spacing in enumerate(spacing_grid):
            label = f"grid_{inclination_index}_{spacing_index}"
            target_phi = source_phi + float(delta_phi)
            target_spacing = source_spacing + float(delta_spacing)
            ladder_positions, ladder_quaternions = randomized_wood_step_box_poses(
                ladder_specs,
                source_phi=source_phi,
                source_spacing=source_spacing,
                target_phi=target_phi,
                target_spacing=target_spacing,
                origin=ladder_origin,
            )
            model = _build_collision_model(
                mujoco,
                robot_asset,
                ladder_specs,
                ladder_positions,
                ladder_quaternions,
                DEFAULT_GROUND_Z,
            )
            result = _evaluate_geometry(
                mujoco,
                label=label,
                model=model,
                joint_pos=joint_pos,
                joint_names=joint_names,
                root_pos=root_pos,
                root_quat_wxyz=root_quat_wxyz,
                candidate_distance=args.candidate_distance,
                penetration_tolerance=args.penetration_tolerance,
                ignored_robot_bodies=ignored_robot_bodies,
            )
            results[(inclination_index, spacing_index)] = result
            _summarize_result(label, result)

    center_index = args.geometry_grid_steps // 2
    nominal = results[(center_index, center_index)]
    corner_indexes = (
        (0, 0),
        (0, args.geometry_grid_steps - 1),
        (args.geometry_grid_steps - 1, 0),
        (args.geometry_grid_steps - 1, args.geometry_grid_steps - 1),
    )
    corner_results = [results[index] for index in corner_indexes]
    grid_valid_masks = np.stack(
        [
            np.stack(
                [results[(inclination_index, spacing_index)]["valid_mask"] for spacing_index in range(args.geometry_grid_steps)]
            )
            for inclination_index in range(args.geometry_grid_steps)
        ]
    )
    grid_max_depth = np.stack(
        [
            np.stack(
                [results[(inclination_index, spacing_index)]["max_depth"] for spacing_index in range(args.geometry_grid_steps)]
            )
            for inclination_index in range(args.geometry_grid_steps)
        ]
    )
    grid_pair_count = np.stack(
        [
            np.stack(
                [results[(inclination_index, spacing_index)]["pair_count"] for spacing_index in range(args.geometry_grid_steps)]
            )
            for inclination_index in range(args.geometry_grid_steps)
        ]
    )
    cell_valid_masks = (
        grid_valid_masks[:-1, :-1]
        & grid_valid_masks[1:, :-1]
        & grid_valid_masks[:-1, 1:]
        & grid_valid_masks[1:, 1:]
    )
    corner_valid_masks = np.stack([result["valid_mask"] for result in corner_results])
    envelope_valid_mask = np.all(corner_valid_masks, axis=0)
    corner_max_depth = np.stack([result["max_depth"] for result in corner_results])
    corner_pair_count = np.stack([result["pair_count"] for result in corner_results])
    corner_worst_robot_body = np.stack(
        [result["worst_robot_body"] for result in corner_results]
    )
    corner_worst_environment_geom = np.stack(
        [result["worst_environment_geom"] for result in corner_results]
    )
    worst_corner_by_frame = np.argmax(corner_max_depth, axis=0)
    frame_indexes = np.arange(joint_pos.shape[0])

    output = {
        "reset_valid_mask": nominal["valid_mask"],
        "reset_valid_frame_indexes": np.flatnonzero(nominal["valid_mask"]).astype(np.int64),
        "reset_invalid_frame_indexes": np.flatnonzero(~nominal["valid_mask"]).astype(np.int64),
        "reset_max_penetration_depth": nominal["max_depth"],
        "reset_penetration_pair_count": nominal["pair_count"],
        "reset_worst_robot_body": nominal["worst_robot_body"],
        "reset_worst_environment_geom": nominal["worst_environment_geom"],
        "reset_valid_mask_ladder_geometry_envelope": envelope_valid_mask,
        "reset_valid_frame_indexes_ladder_geometry_envelope": np.flatnonzero(
            envelope_valid_mask
        ).astype(np.int64),
        "reset_invalid_frame_indexes_ladder_geometry_envelope": np.flatnonzero(
            ~envelope_valid_mask
        ).astype(np.int64),
        "reset_ladder_inclination_delta_range": np.asarray(
            [-inclination_delta, inclination_delta], dtype=np.float32
        ),
        "reset_ladder_spacing_delta_range": np.asarray(
            [-args.spacing_delta, args.spacing_delta], dtype=np.float32
        ),
        "reset_ladder_geometry_corner_deltas": corner_deltas.astype(np.float32),
        "reset_ladder_geometry_corner_valid_mask": corner_valid_masks,
        "reset_ladder_geometry_corner_max_penetration_depth": corner_max_depth,
        "reset_ladder_geometry_corner_penetration_pair_count": corner_pair_count,
        "reset_ladder_geometry_grid_inclination_deltas": inclination_grid.astype(np.float32),
        "reset_ladder_geometry_grid_spacing_deltas": spacing_grid.astype(np.float32),
        "reset_ladder_geometry_grid_valid_mask": grid_valid_masks,
        "reset_ladder_geometry_grid_max_penetration_depth": grid_max_depth,
        "reset_ladder_geometry_grid_penetration_pair_count": grid_pair_count,
        "reset_ladder_geometry_cell_valid_mask": cell_valid_masks,
        "reset_ladder_geometry_envelope_max_penetration_depth": np.max(
            corner_max_depth, axis=0
        ),
        "reset_ladder_geometry_envelope_penetration_pair_count": np.max(
            corner_pair_count, axis=0
        ),
        "reset_ladder_geometry_envelope_worst_robot_body": corner_worst_robot_body[
            worst_corner_by_frame, frame_indexes
        ],
        "reset_ladder_geometry_envelope_worst_environment_geom": corner_worst_environment_geom[
            worst_corner_by_frame, frame_indexes
        ],
        "reset_detection_source_file": np.asarray([str(motion_file)]),
        "reset_detection_robot_asset": np.asarray([str(robot_asset)]),
        "reset_detection_ladder_asset": np.asarray([str(ladder_asset)]),
        "reset_detection_backend": np.asarray(["mujoco_3_signed_geom_distance"]),
        "reset_detection_penetration_tolerance": np.asarray(
            [args.penetration_tolerance], dtype=np.float32
        ),
        "reset_detection_candidate_distance": np.asarray(
            [args.candidate_distance], dtype=np.float32
        ),
        "reset_detection_side_board_depth_fraction": np.asarray(
            [args.side_board_depth_fraction], dtype=np.float32
        ),
        "reset_detection_ground_z": np.asarray([DEFAULT_GROUND_Z], dtype=np.float32),
        "reset_detection_ladder_pos": np.asarray(ladder_asset_position, dtype=np.float32),
        "reset_detection_ladder_yaw_deg": np.asarray(
            [ladder_asset_yaw_deg], dtype=np.float32
        ),
        "reset_detection_ignored_robot_bodies": np.asarray(sorted(ignored_robot_bodies)),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **output)
    print(f"[INFO] Wrote reset-validity sidecar: {output_file}")

    envelope_count = int(envelope_valid_mask.sum())
    print(
        f"[INFO][envelope] valid={envelope_count}/{len(envelope_valid_mask)} "
        f"({envelope_count / len(envelope_valid_mask):.3%}); "
        f"frame_0_valid={bool(envelope_valid_mask[0])}"
    )
    if (~envelope_valid_mask).any():
        print("[INFO][envelope] invalid runs:", _invalid_runs(envelope_valid_mask))
    cell_valid_counts = cell_valid_masks.sum(axis=-1)
    print(
        f"[INFO][conditioned cells] valid-frame count min/median/max="
        f"{int(cell_valid_counts.min())}/{float(np.median(cell_valid_counts)):.1f}/"
        f"{int(cell_valid_counts.max())}; frame_0_valid_in_all_cells="
        f"{bool(np.all(cell_valid_masks[..., 0]))}"
    )

    if motion_output_file is not None:
        _write_merged_motion(motion_file, motion_output_file, output)
        print(f"[INFO] Wrote motion with embedded geometry-conditioned reset masks: {motion_output_file}")


if __name__ == "__main__":
    main()
