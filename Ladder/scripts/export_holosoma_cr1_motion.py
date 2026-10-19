"""Export a Holosoma CR1 qpos clip to a Ladder/WBT-style motion npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "holosoma"
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "demo_data"
    / "climb_ladder"
    / "climing_ladder_mirror_x"
    / "climing_ladder_mirror_x_downsample4_half_footstick.npz"
)
DEFAULT_ROBOT = (
    REPO_ROOT
    / "assets"
    / "robots"
    / "CR1-B2-PRO-260608"
    / "CR1-B2-pro_free_wrist_fixed_spherehand.urdf"
)

CR1_25DOF_JOINT_NAMES = [
    "waist_z_joint",
    "waist_x_joint",
    "waist_y_joint",
    "left_shoulder_y_joint",
    "left_shoulder_x_joint",
    "left_shoulder_z_joint",
    "left_elbow_joint",
    "right_shoulder_y_joint",
    "right_shoulder_x_joint",
    "right_shoulder_z_joint",
    "right_elbow_joint",
    "neck_z_joint",
    "neck_y_joint",
    "left_hip_y_joint",
    "left_hip_x_joint",
    "left_hip_z_joint",
    "left_knee_joint",
    "left_ankle_y_joint",
    "left_ankle_x_joint",
    "right_hip_y_joint",
    "right_hip_x_joint",
    "right_hip_z_joint",
    "right_knee_joint",
    "right_ankle_y_joint",
    "right_ankle_x_joint",
]


parser = argparse.ArgumentParser(description="Export Holosoma CR1 qpos data to a Ladder motion npz.")
parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT, help="Holosoma qpos npz file.")
parser.add_argument("--output_file", type=Path, required=True, help="Output Ladder/WBT-style motion npz.")
parser.add_argument("--robot_asset", type=Path, default=DEFAULT_ROBOT, help="CR1 25DOF URDF used for IsaacLab replay.")
parser.add_argument(
    "--fps_override",
    type=float,
    default=None,
    help="Use this fps instead of the npz fps. Use 60 for downsample2 to preserve downsample4 duration.",
)
parser.add_argument("--max_frames", type=int, default=None, help="Optional frame cap for quick export smoke tests.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite output_file if it exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul


def _check_file(path: Path, suffix: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    if resolved.suffix.lower() != suffix:
        raise ValueError(f"Expected a {suffix} file, got: {resolved}")
    return resolved


def _load_qpos(path: Path, fps_override: float | None) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as data:
        if "qpos" not in data.files:
            raise KeyError(f"{path} is missing required key 'qpos'")
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps = float(fps_override if fps_override is not None else np.asarray(data["fps"]).reshape(-1)[0])
    if qpos.ndim != 2 or qpos.shape[1] != 7 + len(CR1_25DOF_JOINT_NAMES):
        raise ValueError(f"expected qpos shape (T, 32), got {qpos.shape}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    return qpos, fps


def _finite_difference(values: torch.Tensor, fps: float) -> torch.Tensor:
    if values.shape[0] <= 1:
        return torch.zeros_like(values)
    return torch.gradient(values, spacing=(1.0 / fps,), dim=0)[0]


def _angular_velocity(quats_wxyz: torch.Tensor, fps: float) -> torch.Tensor:
    if quats_wxyz.shape[0] <= 1:
        return torch.zeros(quats_wxyz.shape[0], 3, dtype=quats_wxyz.dtype, device=quats_wxyz.device)
    dt = 1.0 / fps
    if quats_wxyz.shape[0] == 2:
        q_rel = quat_mul(quats_wxyz[1:2], quat_conjugate(quats_wxyz[0:1]))
        omega = axis_angle_from_quat(q_rel) / dt
        return omega.repeat(2, 1)
    q_rel = quat_mul(quats_wxyz[2:], quat_conjugate(quats_wxyz[:-2]))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def make_robot_cfg(robot_asset: Path) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(robot_asset),
            fix_base=False,
            merge_fixed_joints=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
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


@configclass
class ExportSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot: ArticulationCfg = make_robot_cfg(_check_file(args_cli.robot_asset, ".urdf"))


def _export_motion(sim: SimulationContext, scene: InteractiveScene, qpos_np: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    robot = scene["robot"]
    joint_ids = robot.find_joints(CR1_25DOF_JOINT_NAMES, preserve_order=True)[0]
    if len(joint_ids) != len(CR1_25DOF_JOINT_NAMES):
        raise RuntimeError(f"Expected 25 CR1 joints, found {len(joint_ids)} in IsaacLab robot")

    qpos = torch.tensor(qpos_np, dtype=torch.float32, device=sim.device)
    root_pos = qpos[:, :3]
    root_quat = torch.nn.functional.normalize(qpos[:, 3:7], dim=-1)
    joint_pos_src = qpos[:, 7:]
    root_lin_vel = _finite_difference(root_pos, fps)
    root_ang_vel = _angular_velocity(root_quat, fps)
    joint_vel_src = _finite_difference(joint_pos_src, fps)

    log: dict[str, list[np.ndarray] | np.ndarray] = {
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for frame in range(qpos.shape[0]):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = root_pos[frame : frame + 1]
        root_state[:, 3:7] = root_quat[frame : frame + 1]
        root_state[:, 7:10] = root_lin_vel[frame : frame + 1]
        root_state[:, 10:13] = root_ang_vel[frame : frame + 1]
        robot.write_root_state_to_sim(root_state)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_ids] = joint_pos_src[frame : frame + 1]
        joint_vel[:, joint_ids] = joint_vel_src[frame : frame + 1]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].detach().cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].detach().cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].detach().cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].detach().cpu().numpy().copy())
        if (frame + 1) % 100 == 0 or frame + 1 == qpos.shape[0]:
            print(f"[INFO] Exported {frame + 1}/{qpos.shape[0]} frames", flush=True)

    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[key] = np.stack(log[key], axis=0).astype(np.float32)
    log["joint_names"] = np.asarray(robot.joint_names)
    log["body_names"] = np.asarray(robot.body_names)
    log["source_joint_names"] = np.asarray(CR1_25DOF_JOINT_NAMES)
    return log  # type: ignore[return-value]


def main() -> None:
    input_file = _check_file(args_cli.input_file, ".npz")
    output_file = args_cli.output_file.expanduser().resolve()
    if output_file.exists() and not args_cli.overwrite:
        raise FileExistsError(f"Output file already exists: {output_file}. Pass --overwrite to replace it.")

    qpos, fps = _load_qpos(input_file, args_cli.fps_override)
    if args_cli.max_frames is not None:
        if args_cli.max_frames <= 0:
            raise ValueError(f"--max_frames must be positive, got {args_cli.max_frames}")
        qpos = qpos[: args_cli.max_frames]
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / fps, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    print("[INFO] Creating IsaacLab export scene...", flush=True)
    scene = InteractiveScene(ExportSceneCfg(num_envs=1, env_spacing=2.0))
    print("[INFO] Resetting simulation...", flush=True)
    sim.reset()
    print("[INFO] Scene setup complete.", flush=True)
    print(f"[INFO] Input: {input_file}", flush=True)
    print(f"[INFO] Frames: {qpos.shape[0]}, fps: {fps:g}, duration: {(qpos.shape[0] - 1) / fps:.3f}s", flush=True)

    motion = _export_motion(sim, scene, qpos, fps)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **motion)
    print(f"[INFO] Wrote motion: {output_file}", flush=True)
    print(f"[INFO] joint_pos: {motion['joint_pos'].shape}, body_pos_w: {motion['body_pos_w'].shape}", flush=True)
    print("[INFO] First body names:", ", ".join(motion["body_names"][:8].tolist()), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
