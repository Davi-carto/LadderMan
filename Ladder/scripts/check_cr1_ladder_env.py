"""Run a deterministic single-environment smoke check for the CR1 ladder task."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Ladder" / "source" / "ladder"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
DEFAULT_MOTION = (
    REPO_ROOT
    / "Ladder"
    / "motions"
    / "restored_prefix_v300_stride2_slow50"
    / "smooth_start"
    / "climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_train_ready.npz"
)


parser = argparse.ArgumentParser(description="Check one CR1 ladder reference frame in IsaacLab.")
parser.add_argument("--task", type=str, default="Ladder-Climb-CR1-v0", help="Registered Ladder task.")
parser.add_argument("--motion_file", type=str, default=str(DEFAULT_MOTION), help="Annotated 50 Hz CR1 motion npz.")
parser.add_argument("--reference_frame", type=int, default=0, help="Reference frame used at reset.")
parser.add_argument(
    "--all_key_frames",
    action="store_true",
    help="Check frame 0, each contact transition and its adjacent frames, and the final frame.",
)
parser.add_argument("--steps", type=int, default=3, help="Number of zero-action environment steps.")
parser.add_argument(
    "--action_value",
    type=float,
    default=0.0,
    help="Constant raw joint action used during stepping; useful for action-limit smoke tests.",
)
parser.add_argument("--hold_frame", action="store_true", help="Keep the command at reference_frame while stepping.")
parser.add_argument("--debug_vis", action="store_true", help="Enable command/contact debug visualization.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument(
    "--enable_ladder_randomization",
    action="store_true",
    help="Retain the task's configured per-environment ladder geometry randomization.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.steps < 0:
    parser.error("--steps must be non-negative")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_error_magnitude

import ladder.tasks  # noqa: F401


CONTACT_FORCE_BODY_NAMES = (
    "left_sphere_hand_link",
    "right_sphere_hand_link",
    "left_ankle_x_link",
    "right_ankle_x_link",
)


def _is_finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_is_finite(item) for item in value.values())
    return True


def _print_tensor_tree(name: str, value, indent: str = "") -> None:
    if isinstance(value, dict):
        print(f"{indent}{name}:")
        for child_name, child_value in value.items():
            _print_tensor_tree(child_name, child_value, indent + "  ")
    elif isinstance(value, torch.Tensor):
        print(f"{indent}{name}: shape={tuple(value.shape)}, dtype={value.dtype}, finite={_is_finite(value)}")
    else:
        print(f"{indent}{name}: {type(value).__name__}")


def _print_command_state(env, label: str) -> None:
    command = env.command_manager.get_term("motion")
    joint_error = torch.linalg.norm(command.robot_joint_pos - command.joint_pos, dim=-1)
    body_position_error = torch.linalg.norm(command.robot_body_pos_w - command.body_pos_w, dim=-1)
    body_orientation_error = quat_error_magnitude(command.robot_body_quat_w, command.body_quat_w)
    anchor_position_error = torch.linalg.norm(command.robot_anchor_pos_w - command.anchor_pos_w, dim=-1)
    anchor_orientation_error = quat_error_magnitude(command.robot_anchor_quat_w, command.anchor_quat_w)
    contact_position_error = torch.linalg.norm(command.robot_limb_pos_w - command.target_contact_pos_w, dim=-1)

    print(f"[{label}] reference_frame={int(command.time_steps[0])}")
    print(f"  joint_error_l2={float(joint_error[0]):.6f}")
    print(f"  body_position_error_mean={float(body_position_error[0].mean()):.6f}")
    print(f"  body_position_error_max={float(body_position_error[0].max()):.6f}")
    print(f"  body_orientation_error_mean={float(body_orientation_error[0].mean()):.6f}")
    print(f"  body_orientation_error_max={float(body_orientation_error[0].max()):.6f}")
    print(f"  anchor_position_error={float(anchor_position_error[0]):.6f}")
    print(f"  anchor_orientation_error={float(anchor_orientation_error[0]):.6f}")
    print(f"  contact_indicator={command.contact_indicator[0].int().tolist()}")
    print(f"  contact_rung_id={command.contact_rung_id[0].tolist()}")
    print(f"  contact_position_error={contact_position_error[0].tolist()}")
    print(f"  command_metrics_finite={_is_finite(command.metrics)}")

    sole_ids, sole_names = env.scene["robot"].find_bodies(
        ["left_sole_contact_point", "right_sole_contact_point"], preserve_order=True
    )
    sole_z = env.scene["robot"].data.body_pos_w[0, sole_ids, 2]
    print(f"  sole_z={dict(zip(sole_names, sole_z.tolist(), strict=True))}")


def _print_contact_forces(env, label: str) -> None:
    sensor = env.scene.sensors["contact_forces"]
    force_norms = torch.linalg.norm(sensor.data.net_forces_w[0], dim=-1)
    name_to_index = {name: index for index, name in enumerate(sensor.body_names)}
    selected = {
        name: float(force_norms[name_to_index[name]])
        for name in CONTACT_FORCE_BODY_NAMES
        if name in name_to_index
    }
    top_count = min(8, len(sensor.body_names))
    top_values, top_indexes = torch.topk(force_norms, k=top_count)
    top = [
        (sensor.body_names[int(index)], float(value))
        for value, index in zip(top_values, top_indexes, strict=True)
        if float(value) > 0.0
    ]
    print(f"[{label}] selected_contact_force_norm_N={selected}")
    print(f"[{label}] top_contact_force_norm_N={top}")

    reward_cfg = env.reward_manager.get_term_cfg("undesired_contacts")
    sensor_cfg = reward_cfg.params["sensor_cfg"]
    threshold = float(reward_cfg.params["threshold"])
    history_norms = torch.linalg.norm(sensor.data.net_forces_w_history[0, :, sensor_cfg.body_ids], dim=-1)
    history_peaks = torch.max(history_norms, dim=0).values
    undesired = [
        (sensor.body_names[body_id], float(peak))
        for body_id, peak in zip(sensor_cfg.body_ids, history_peaks, strict=True)
        if float(peak) > threshold
    ]
    print(f"[{label}] undesired_contact_history_over_{threshold:g}N={undesired}")


def _print_manager_terms(env, label: str) -> None:
    rewards = {
        name: float(values[0]) for name, values in env.reward_manager.get_active_iterable_terms(env_idx=0)
    }
    terminations = {
        name: bool(values[0])
        for name, values in env.termination_manager.get_active_iterable_terms(env_idx=0)
    }
    print(f"[{label}] weighted_reward_terms_per_second={rewards}")
    print(f"[{label}] reward_terms_finite={all(math.isfinite(value) for value in rewards.values())}")
    print(f"[{label}] termination_terms={terminations}")


def _key_reference_frames(command) -> list[int]:
    indicator = command.motion.contact_indicator > 0.5
    transition_mask = torch.any(indicator[1:] != indicator[:-1], dim=-1)
    transitions = (torch.where(transition_mask)[0] + 1).tolist()
    last_frame = command.motion.time_step_total - 1
    key_frames = {0, last_frame}
    for transition in transitions:
        key_frames.update(max(0, min(last_frame, transition + offset)) for offset in (-1, 0, 1))
    return sorted(key_frames)


def main() -> None:
    motion_file = os.path.abspath(args_cli.motion_file)
    if not os.path.isfile(motion_file):
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.commands.motion.motion_file = motion_file
    env_cfg.commands.motion.fixed_reference_frame = 0 if args_cli.all_key_frames else args_cli.reference_frame
    env_cfg.commands.motion.advance_reference = not args_cli.hold_frame
    env_cfg.commands.motion.debug_vis = args_cli.debug_vis
    env_cfg.commands.motion.pose_range = {name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.commands.motion.fixed_start_pose_range = {}
    env_cfg.commands.motion.velocity_range = {
        name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    if not args_cli.enable_ladder_randomization:
        env_cfg.commands.motion.ladder_inclination_delta_range = (0.0, 0.0)
        env_cfg.commands.motion.ladder_spacing_delta_range = (0.0, 0.0)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.contact_forces.debug_vis = args_cli.debug_vis
    for event_name in ("robot_material", "robot_mass", "joint_position_offset", "torso_com"):
        setattr(env_cfg.events, event_name, None)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    try:
        command = unwrapped.command_manager.get_term("motion")
        reference_frames = _key_reference_frames(command) if args_cli.all_key_frames else [args_cli.reference_frame]
        action_dim = unwrapped.action_manager.total_action_dim
        if action_dim != 25:
            raise RuntimeError(f"CR1 action dimension must be 25, got {action_dim}")
        actions = torch.full(
            (args_cli.num_envs, action_dim),
            args_cli.action_value,
            dtype=torch.float32,
            device=unwrapped.device,
        )
        print(f"[SETUP] task={args_cli.task}")
        print(f"[SETUP] motion={motion_file}")
        print(f"[SETUP] robot_asset={unwrapped.cfg.scene.robot.spawn.asset_path}")
        print(f"[SETUP] device={unwrapped.device}, step_dt={unwrapped.step_dt:g}")
        print(f"[SETUP] ground_z={unwrapped.cfg.scene.ground.init_state.pos[2]:g}")
        print(f"[SETUP] domain_randomization_disabled={unwrapped.cfg.events is None}")
        print(f"[SETUP] reference_frames={reference_frames}")
        print(f"[SETUP] action_shape={tuple(actions.shape)}, expected_25={action_dim == 25}")
        print(f"[SETUP] constant_action_value={args_cli.action_value:g}")
        print(
            f"[SETUP] ladder_randomization_enabled={args_cli.enable_ladder_randomization} "
            f"num_envs={args_cli.num_envs}"
        )

        for case_index, reference_frame in enumerate(reference_frames, start=1):
            command.cfg.fixed_reference_frame = reference_frame
            observations, _ = env.reset()
            print(f"\n[CASE {case_index}/{len(reference_frames)}] requested_reference_frame={reference_frame}")
            print(
                f"[LADDER] phi_deg_min_max="
                f"({float(torch.rad2deg(command.ladder_phi).min()):.4f}, "
                f"{float(torch.rad2deg(command.ladder_phi).max()):.4f}) "
                f"spacing_min_max=({float(command.ladder_spacing.min()):.5f}, "
                f"{float(command.ladder_spacing.max()):.5f})"
            )
            if command._ladder_asset is not None:
                physical_rung_centers = command._ladder_asset.data.object_state_w[
                    :, command._ladder_rung_box_indexes, :3
                ]
                physical_rung_tops = (
                    physical_rung_centers - command._ladder_rung_center_offsets_e.unsqueeze(0)
                )
                rung_sync_error = torch.linalg.norm(
                    physical_rung_tops - command.rung_positions_w, dim=-1
                ).max()
                print(f"[LADDER] physical_rung_target_max_error={float(rung_sync_error):.8f}m")
            _print_tensor_tree("observations", observations)
            print(f"[RESET] observations_finite={_is_finite(observations)}")
            _print_command_state(unwrapped, "RESET")

            immediate_done = unwrapped.termination_manager.compute()
            print(f"[RESET] immediate_done={bool(immediate_done[0])}")
            _print_manager_terms(unwrapped, "RESET")
            _print_contact_forces(unwrapped, "RESET")

            action_term = unwrapped.action_manager._terms["joint_pos"]
            previous_target = unwrapped.scene["robot"].data.joint_pos[
                :, action_term._joint_ids
            ].clone()

            for step in range(args_cli.steps):
                observations, reward, terminated, truncated, _ = env.step(actions)
                label = f"STEP {step + 1}"
                processed_target = action_term.processed_actions
                target_delta = torch.abs(processed_target - previous_target)
                if hasattr(action_term, "_max_target_delta"):
                    configured_delta = action_term._max_target_delta
                    if torch.any(target_delta > configured_delta + 1.0e-6):
                        raise RuntimeError(
                            f"processed joint target exceeded configured rate limit: "
                            f"{float(target_delta.max()):.6f} > {float(configured_delta.max()):.6f} rad/step"
                        )
                    print(
                        f"[{label}] processed_target_delta_max={float(target_delta.max()):.6f} rad, "
                        f"configured_delta_max={float(configured_delta.max()):.6f} rad"
                    )
                previous_target.copy_(processed_target)
                print(
                    f"[{label}] reward_shape={tuple(reward.shape)}, terminated_shape={tuple(terminated.shape)}, "
                    f"truncated_shape={tuple(truncated.shape)}, reward={float(reward[0]):.6f}, "
                    f"terminated={bool(terminated[0])}, "
                    f"truncated={bool(truncated[0])}, observations_finite={_is_finite(observations)}"
                )
                _print_command_state(unwrapped, label)
                _print_manager_terms(unwrapped, label)
                _print_contact_forces(unwrapped, label)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
