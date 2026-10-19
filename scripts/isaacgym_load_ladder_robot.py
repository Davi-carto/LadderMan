#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACGYM_PYTHON = Path("/root/isaacgym/python")
if ISAACGYM_PYTHON.is_dir() and str(ISAACGYM_PYTHON) not in sys.path:
    sys.path.append(str(ISAACGYM_PYTHON))

from isaacgym import gymapi

DEFAULT_ROBOT_DIR = REPO_ROOT / "assets" / "robots" / "CR1-B2-PRO-260608"
DEFAULT_ROBOT_FILE = "CR1-B2-pro_simplify_collision.urdf"
DEFAULT_LADDER_DIR = REPO_ROOT / "assets" / "ladders" / "ladder_A_aframe"
DEFAULT_LADDER_FILE = "wood_step_ladder_sim.urdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the CR1-B2 robot and wood step ladder in Isaac Gym.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer.")
    parser.add_argument("--sim-steps", type=int, default=0, help="Number of simulation steps to run in headless mode.")
    parser.add_argument("--robot-dir", type=Path, default=DEFAULT_ROBOT_DIR)
    parser.add_argument("--robot-file", default=DEFAULT_ROBOT_FILE)
    parser.add_argument("--ladder-dir", type=Path, default=DEFAULT_LADDER_DIR)
    parser.add_argument("--ladder-file", default=DEFAULT_LADDER_FILE)
    parser.add_argument("--robot-x", type=float, default=0.0)
    parser.add_argument("--robot-y", type=float, default=0.85)
    parser.add_argument("--robot-z", type=float, default=0.95)
    parser.add_argument("--robot-yaw-deg", type=float, default=180.0, help="Robot yaw angle in degrees.")
    parser.add_argument("--robot-on-step", type=int, default=None, help="Place the robot's feet on this ladder step index.")
    parser.add_argument("--fix-robot", action="store_true", help="Fix the robot base for visual placement checks.")
    parser.add_argument("--ladder-x", type=float, default=0.0)
    parser.add_argument("--ladder-y", type=float, default=0.0)
    parser.add_argument("--ladder-z", type=float, default=0.065)
    parser.add_argument("--gpu-pipeline", action="store_true", help="Use Isaac Gym GPU pipeline.")
    parser.add_argument("--viewer-no-sync", action="store_true", help="Disable synchronized viewer draw calls.")
    parser.add_argument("--force-viewer-seconds", type=float, default=0.0, help="Keep drawing for this long even if the viewer reports closed.")
    parser.add_argument("--screenshot", type=Path, default=None, help="Save a camera image and exit.")
    parser.add_argument(
        "--camera-view",
        choices=("default", "front", "front_left", "front_right", "side_left", "side_right", "high", "feet"),
        default="default",
        help="Camera preset for viewer and screenshots.",
    )
    parser.add_argument("--front-camera", action="store_true", help="Deprecated alias for --camera-view front.")
    return parser.parse_args()


def ladder_step_center(step_index: int) -> tuple[float, float]:
    step_spacing = 0.25
    step_depth = 0.18
    front_incline_deg = 65.0
    z = step_index * step_spacing
    y = -z / math.tan(math.radians(front_incline_deg)) + step_depth * 0.10
    return y, z


def ensure_robot_mesh_aliases(robot_dir: Path) -> None:
    """Create local mesh aliases for assets that are present under near-identical names.

    The CR1-B2 asset currently contains two URDF mesh references whose files are missing:
    one download-residue filename and one simplified wrist mesh. Creating these aliases
    keeps Isaac Gym's URDF loader happy without modifying the original URDF.
    """

    meshes = robot_dir / "meshes"
    aliases = {
        "left_shoulder_z_link.STL": "left_shoulder_z_link.STL.folderdownload.crdownload",
        "left_wrist_x_link.STL": "left_wrist_x_link_simplified.STL",
    }
    for target_name, source_name in aliases.items():
        target = meshes / target_name
        source = meshes / source_name
        if target.exists():
            continue
        if not source.exists():
            raise FileNotFoundError(f"Missing robot mesh {target} and fallback {source}")
        shutil.copyfile(source, target)
        print(f"[asset-fix] Created missing mesh alias: {target.name} <- {source.name}")


def make_transform(x: float, y: float, z: float, yaw: float = 0.0) -> gymapi.Transform:
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(x, y, z)
    pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, yaw)
    return pose


def get_camera_pose(view: str) -> tuple[gymapi.Vec3, gymapi.Vec3]:
    target = gymapi.Vec3(0.0, -0.34, 0.95)
    if view == "feet":
        return gymapi.Vec3(0.75, 1.05, 1.12), gymapi.Vec3(0.0, -0.34, 0.82)
    if view == "front":
        return gymapi.Vec3(0.0, 2.7, 1.45), target
    if view == "front_left":
        return gymapi.Vec3(-1.65, 2.25, 1.55), target
    if view == "front_right":
        return gymapi.Vec3(1.65, 2.25, 1.55), target
    if view == "side_left":
        return gymapi.Vec3(-2.35, 0.05, 1.35), target
    if view == "side_right":
        return gymapi.Vec3(2.35, 0.05, 1.35), target
    if view == "high":
        return gymapi.Vec3(1.45, 1.75, 2.55), gymapi.Vec3(0.0, -0.35, 0.85)
    return gymapi.Vec3(2.6, 2.7, 1.8), gymapi.Vec3(0.0, -0.35, 0.95)


def set_camera(gym, viewer, env, view: str) -> None:
    camera_pos, camera_target = get_camera_pose(view)
    gym.viewer_camera_look_at(
        viewer,
        env,
        camera_pos,
        camera_target,
    )


def main() -> None:
    args = parse_args()
    if args.front_camera:
        args.camera_view = "front"

    robot_dir = args.robot_dir.resolve()
    ladder_dir = args.ladder_dir.resolve()
    robot_file = args.robot_file
    ladder_file = args.ladder_file

    if args.robot_on_step is not None:
        if not 1 <= args.robot_on_step <= 7:
            raise ValueError("--robot-on-step must be between 1 and 7")
        step_y, step_z = ladder_step_center(args.robot_on_step)
        foot_bottom_to_base = 0.9044
        args.robot_x = args.ladder_x
        args.robot_y = args.ladder_y + step_y
        args.robot_z = args.ladder_z + step_z + 0.03 + foot_bottom_to_base
        print(
            f"[placement] robot on step {args.robot_on_step}: "
            f"step_center_y={step_y:.3f}, step_top_z={args.ladder_z + step_z + 0.03:.3f}, "
            f"robot_root=({args.robot_x:.3f}, {args.robot_y:.3f}, {args.robot_z:.3f})"
        )

    if not (robot_dir / robot_file).is_file():
        raise FileNotFoundError(robot_dir / robot_file)
    if not (ladder_dir / ladder_file).is_file():
        raise FileNotFoundError(ladder_dir / ladder_file)

    ensure_robot_mesh_aliases(robot_dir)

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 200.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = args.gpu_pipeline
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 2
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.001
    sim_params.physx.use_gpu = True

    compute_device_id = 0
    offscreen = args.headless
    graphics_device_id = -1 if offscreen else 0
    sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")

    viewer = None
    if not args.headless and args.screenshot is None:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create Isaac Gym viewer. Try --headless on a machine without display.")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0.0
    gym.add_ground(sim, plane_params)

    ladder_options = gymapi.AssetOptions()
    ladder_options.fix_base_link = True
    ladder_options.disable_gravity = True
    ladder_options.collapse_fixed_joints = True
    ladder_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
    ladder_options.thickness = 0.001

    robot_options = gymapi.AssetOptions()
    robot_options.fix_base_link = args.fix_robot
    robot_options.disable_gravity = args.fix_robot
    robot_options.collapse_fixed_joints = False
    robot_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    robot_options.angular_damping = 0.01
    robot_options.linear_damping = 0.01
    robot_options.max_angular_velocity = 100.0
    robot_options.max_linear_velocity = 100.0
    robot_options.armature = 0.01
    robot_options.thickness = 0.001

    ladder_asset = gym.load_asset(sim, str(ladder_dir), ladder_file, ladder_options)
    robot_asset = gym.load_asset(sim, str(robot_dir), robot_file, robot_options)
    if ladder_asset is None:
        raise RuntimeError(f"Failed to load ladder asset: {ladder_dir / ladder_file}")
    if robot_asset is None:
        raise RuntimeError(f"Failed to load robot asset: {robot_dir / robot_file}")

    print(f"[asset] ladder: {ladder_dir / ladder_file}")
    print(f"[asset] robot:  {robot_dir / robot_file}")
    print(f"[ladder] bodies={gym.get_asset_rigid_body_count(ladder_asset)} dofs={gym.get_asset_dof_count(ladder_asset)}")
    print(f"[robot]  bodies={gym.get_asset_rigid_body_count(robot_asset)} dofs={gym.get_asset_dof_count(robot_asset)}")

    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.5), 1)

    ladder_actor = gym.create_actor(
        env,
        ladder_asset,
        make_transform(args.ladder_x, args.ladder_y, args.ladder_z),
        "ladder_A",
        0,
        0,
    )
    robot_actor = gym.create_actor(
        env,
        robot_asset,
        make_transform(args.robot_x, args.robot_y, args.robot_z, math.radians(args.robot_yaw_deg)),
        "robot",
        0,
        1,
    )

    props = gym.get_actor_dof_properties(env, robot_actor)
    if len(props) > 0:
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(80.0)
        props["damping"].fill(4.0)
        gym.set_actor_dof_properties(env, robot_actor, props)
        default_pos = gym.get_actor_dof_states(env, robot_actor, gymapi.STATE_POS)["pos"]
        gym.set_actor_dof_position_targets(env, robot_actor, default_pos)

    gym.set_rigid_body_color(env, ladder_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.78, 0.48, 0.22))

    if viewer is not None:
        set_camera(gym, viewer, env, args.camera_view)

    gym.prepare_sim(sim)

    if args.screenshot is not None:
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        camera_handle = gym.create_camera_sensor(env, camera_props)
        camera_pos, camera_target = get_camera_pose(args.camera_view)
        gym.set_camera_location(
            camera_handle,
            env,
            camera_pos,
            camera_target,
        )
        for _ in range(max(args.sim_steps, 5)):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        gym.write_camera_image_to_file(sim, env, camera_handle, gymapi.IMAGE_COLOR, str(args.screenshot))
        print(f"[done] Saved screenshot: {args.screenshot}")
    elif args.headless:
        steps = args.sim_steps if args.sim_steps > 0 else 240
        for _ in range(steps):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        print(f"[done] Headless smoke test completed for {steps} steps.")
    else:
        print("[viewer] Close the Isaac Gym viewer window to exit.")
        frame = 0
        start_time = time.time()
        while (not gym.query_viewer_has_closed(viewer)) or (
            args.force_viewer_seconds > 0.0 and time.time() - start_time < args.force_viewer_seconds
        ):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, not args.viewer_no_sync)
            gym.sync_frame_time(sim)
            if frame == 0:
                print(f"[viewer] initial closed state: {gym.query_viewer_has_closed(viewer)}")
            frame += 1
            time.sleep(0.001)
        print(f"[viewer] exited after {frame} frames; closed={gym.query_viewer_has_closed(viewer)}")

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
