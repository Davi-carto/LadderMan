"""View the ladder and CR1-B2 robot assets together in IsaacLab."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LADDER = REPO_ROOT / "assets" / "ladders" / "ladder_A_aframe" / "wood_step_ladder_physics.usda"
DEFAULT_ROBOT = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)


parser = argparse.ArgumentParser(description="View the Ladder A asset and CR1-B2 robot in one IsaacLab scene.")
parser.add_argument("--ladder_asset", type=Path, default=DEFAULT_LADDER, help="USD/USDA ladder asset to load.")
parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT, help="URDF robot asset to load.")
parser.add_argument("--ladder_x", type=float, default=0.0, help="Ladder X translation.")
parser.add_argument("--ladder_y", type=float, default=0.0, help="Ladder Y translation.")
parser.add_argument("--ladder_z", type=float, default=0.0, help="Ladder Z translation.")
parser.add_argument("--ladder_yaw_deg", type=float, default=0.0, help="Ladder yaw angle in degrees.")
parser.add_argument("--robot_x", type=float, default=0.0, help="Robot X translation.")
parser.add_argument("--robot_y", type=float, default=0.85, help="Robot Y translation.")
parser.add_argument("--robot_z", type=float, default=0.95, help="Robot Z translation.")
parser.add_argument("--robot_yaw_deg", type=float, default=180.0, help="Robot yaw angle in degrees.")
parser.add_argument("--free_robot", action="store_true", help="Let the robot fall/respond to gravity instead of fixing it.")
parser.add_argument("--smoke_steps", type=int, default=0, help="Run this many steps and exit; 0 keeps the viewer open.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext


def yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    """Return a world-Z yaw quaternion in IsaacLab's wxyz convention."""
    half_yaw = 0.5 * math.radians(yaw_deg)
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def check_asset(path: Path, expected_suffixes: tuple[str, ...]) -> Path:
    asset_path = path.expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Asset not found: {asset_path}")
    if asset_path.suffix.lower() not in expected_suffixes:
        raise ValueError(f"Expected one of {expected_suffixes}, got: {asset_path}")
    return asset_path


def make_robot_cfg(robot_asset: Path) -> ArticulationCfg:
    fixed_base = not args_cli.free_robot
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(robot_asset),
            fix_base=fixed_base,
            merge_fixed_joints=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=fixed_base,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                fix_root_link=fixed_base,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(args_cli.robot_x, args_cli.robot_y, args_cli.robot_z),
            rot=yaw_quat_wxyz(args_cli.robot_yaw_deg),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=400.0,
                velocity_limit_sim=50.0,
                stiffness=80.0,
                damping=4.0,
            )
        },
    )


def design_scene() -> Articulation:
    ladder_asset = check_asset(args_cli.ladder_asset, (".usd", ".usda"))
    robot_asset = check_asset(args_cli.robot_asset, (".urdf",))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    ladder_cfg = sim_utils.UsdFileCfg(usd_path=str(ladder_asset))
    ladder_cfg.func(
        "/World/Ladder",
        ladder_cfg,
        translation=(args_cli.ladder_x, args_cli.ladder_y, args_cli.ladder_z),
        orientation=yaw_quat_wxyz(args_cli.ladder_yaw_deg),
    )

    robot = Articulation(make_robot_cfg(robot_asset))

    print(f"[INFO] Ladder asset: {ladder_asset}")
    print(f"[INFO] Robot asset:  {robot_asset}")
    print(f"[INFO] Robot fixed base: {not args_cli.free_robot}")
    return robot


def reset_robot(robot: Articulation) -> None:
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()


def run_simulator(sim: SimulationContext, robot: Articulation) -> None:
    sim_dt = sim.get_physics_dt()
    reset_robot(robot)
    print(f"[INFO] Robot bodies={robot.num_bodies}, joints={robot.num_joints}")
    print("[INFO] Close the IsaacSim window or press Ctrl+C in the terminal to stop.")

    count = 0
    while simulation_app.is_running():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        count += 1
        if args_cli.smoke_steps > 0 and count >= args_cli.smoke_steps:
            print(f"[INFO] Smoke test completed for {count} steps.")
            break


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.6, 2.7, 1.8], target=[0.0, -0.35, 0.95])

    robot = design_scene()

    sim.reset()
    print("[INFO] Scene setup complete.")
    run_simulator(sim, robot)


if __name__ == "__main__":
    main()
    simulation_app.close()
