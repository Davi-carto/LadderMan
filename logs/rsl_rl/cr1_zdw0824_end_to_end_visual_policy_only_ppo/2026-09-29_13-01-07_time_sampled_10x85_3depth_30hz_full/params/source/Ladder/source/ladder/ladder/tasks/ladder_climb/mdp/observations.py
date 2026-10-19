from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    convert_camera_frame_orientation_convention,
    matrix_from_quat,
    quat_from_euler_xyz,
    quat_apply,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # Isaac Lab 2.1 uses the equivalent legacy name.
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from ladder.tasks.ladder_climb.mdp.commands import LadderMotionCommand
from ladder.rsl_rl.amp import build_amp_style_observation
from ladder.robots.cr1 import CR1_ACTION_SCALE, CR1_URDF
from ladder.utils.contact_schema import CR1_VIRTUAL_BODY_SPECS
from ladder.utils.contact_metrics import encode_support_relation
from ladder.utils.limb_heightmaps import (
    LADDER_OTHER,
    RAIL as HEIGHTMAP_RAIL,
    RUNG as HEIGHTMAP_RUNG,
    LimbHeightMapCfg,
    heightmap_cache_is_current,
    sample_limb_heightmap,
)
from ladder.utils.visual_depth import process_visual_depth, process_warp_visual_depth
from ladder.utils.warp_camera import CR1WarpBatchCamera, RUNG, ROBOT, WarpCameraCfg
from ladder.utils.wood_step_ladder import WOOD_STEP_RUNG_BOX_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


CR1_TRANSFORMER_VISION_ANGULAR_VELOCITY_SCALE = 0.2
CR1_TRANSFORMER_VISION_JOINT_VELOCITY_SCALE = 0.1
CR1_TRANSFORMER_VISION_PROPRIO_CLIP = 100.0


def amp_style_observation(
    env: ManagerBasedEnv, command_name: str = "motion"
) -> torch.Tensor:
    """Return the task-agnostic local robot state used by the AMP critic.

    This is a training-only privileged observation.  It intentionally excludes
    images, reference time, target poses, commands, contacts, and rung IDs.
    """

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return build_amp_style_observation(
        anchor_pos_w=command.robot_anchor_pos_w,
        anchor_quat_w=command.robot_anchor_quat_w,
        anchor_lin_vel_w=command.robot_anchor_lin_vel_w,
        anchor_ang_vel_w=command.robot_anchor_ang_vel_w,
        joint_pos=command.robot_joint_pos,
        joint_vel=command.robot_joint_vel,
        limb_pos_w=command.robot_limb_pos_w,
    )


def _cr1_transformer_vision_joint_position_scale(
    env: ManagerBasedEnv, robot
) -> torch.Tensor:
    """Return the articulation-ordered action scale used by visual pose fields."""

    joint_position_scale = getattr(
        env, "_cr1_transformer_vision_joint_position_scale", None
    )
    if joint_position_scale is None:
        joint_names = tuple(robot.joint_names)
        missing = [name for name in joint_names if name not in CR1_ACTION_SCALE]
        if missing:
            raise KeyError(
                f"Missing CR1 TransformerVision joint-position scales for: {missing}"
            )
        joint_position_scale = torch.tensor(
            [CR1_ACTION_SCALE[name] for name in joint_names],
            dtype=robot.data.joint_pos.dtype,
            device=robot.data.joint_pos.device,
        ).unsqueeze(0)
        if joint_position_scale.shape[-1] != robot.data.joint_pos.shape[-1]:
            raise RuntimeError(
                "CR1 TransformerVision joint-position scale count does not match the articulation: "
                f"{joint_position_scale.shape[-1]} != {robot.data.joint_pos.shape[-1]}"
            )
        env._cr1_transformer_vision_joint_position_scale = joint_position_scale
    return joint_position_scale


def climb_direction(env: ManagerBasedEnv, direction: float = 1.0) -> torch.Tensor:
    """Return the deployable binary climbing-direction command."""

    if direction not in (-1.0, 1.0):
        raise ValueError(f"Climbing direction must be -1 or 1, got {direction}")
    return torch.full((env.num_envs, 1), direction, device=env.device)


def motion_climb_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the motion planner's effective deployable command in {-1, 0, +1}."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.effective_climb_command


def visual_depth(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    min_depth: float = 0.1,
    max_depth: float = 2.0,
    gaussian_std: float = 0.005,
    pixel_dropout_probability: float = 0.05,
    rung_focused_mask_probability: float = 0.1,
) -> torch.Tensor:
    """Return augmented depth in the NCHW layout expected by RSL-RL CNNs."""

    from ladder.utils.visual_depth import DepthAugmentationCfg

    camera = env.scene.sensors[sensor_cfg.name]
    cfg = DepthAugmentationCfg(
        min_depth=min_depth,
        max_depth=max_depth,
        gaussian_std=gaussian_std,
        pixel_dropout_probability=pixel_dropout_probability,
        rung_focused_mask_probability=rung_focused_mask_probability,
    )
    depth, _ = process_visual_depth(
        camera.data.output["distance_to_image_plane"],
        camera.data.output["semantic_segmentation"],
        camera.data.info.get("semantic_segmentation"),
        cfg,
    )
    # The SSR history term is evaluated after this actor depth group.  Reuse
    # this exact augmented frame rather than rendering the camera a second time.
    env._cr1_visual_depth_frame = depth
    return depth


def cr1_transformer_vision_proprio_frame(
    env: ManagerBasedEnv,
    climb_direction_value: float = 1.0,
    climb_command_name: str | None = None,
    include_climb_command: bool = True,
    add_noise: bool = False,
    root_angular_velocity_noise: float = 0.5,
    orientation_noise_rad: float = 0.05,
    joint_position_noise_normalized: float = 0.01,
    joint_velocity_noise: float = 1.5,
) -> torch.Tensor:
    """Return the field-normalized 85-D CR1 proprioceptive frame.

    The frame contains root angular velocity (3), root orientation (6), all
    joint positions (25), joint velocities (25), the previous action (25),
    and climbing direction (1).  When ``include_climb_command`` is false, the
    last value is a fixed zero compatibility pad and carries no command
    information. Following the X2 visual reference, angular
    velocity is multiplied by 0.2, joint-position error is divided by the
    per-joint action scale, and joint velocity is multiplied by 0.1. Rotation,
    action, and direction are already bounded representations and stay
    unchanged. The final vector is clipped to [-100, 100].
    """

    robot = env.scene.articulations["robot"]
    joint_position_scale = _cr1_transformer_vision_joint_position_scale(env, robot)

    root_angular_velocity = robot.data.root_ang_vel_b
    root_quaternion = robot.data.root_quat_w
    joint_position = robot.data.joint_pos
    joint_velocity = robot.data.joint_vel
    if add_noise:
        root_angular_velocity = root_angular_velocity + _uniform_sensor_noise(
            root_angular_velocity, root_angular_velocity_noise
        )
        orientation_rpy_noise = _uniform_sensor_noise(
            torch.empty((env.num_envs, 3), device=env.device), orientation_noise_rad
        )
        orientation_noise_quaternion = quat_from_euler_xyz(
            orientation_rpy_noise[:, 0],
            orientation_rpy_noise[:, 1],
            orientation_rpy_noise[:, 2],
        )
        root_quaternion = quat_mul(root_quaternion, orientation_noise_quaternion)
        joint_velocity = joint_velocity + _uniform_sensor_noise(
            joint_velocity, joint_velocity_noise
        )

    root_rotation = matrix_from_quat(root_quaternion)[..., :2].reshape(env.num_envs, -1)
    joint_position_normalized = (
        joint_position - robot.data.default_joint_pos
    ) / joint_position_scale
    if add_noise:
        joint_position_normalized = joint_position_normalized + _uniform_sensor_noise(
            joint_position_normalized, joint_position_noise_normalized
        )
    if include_climb_command:
        direction = (
            motion_climb_command(env, climb_command_name)
            if climb_command_name is not None
            else climb_direction(env, direction=climb_direction_value)
        )
    else:
        if climb_command_name is not None:
            raise ValueError(
                "climb_command_name must be None when include_climb_command is false"
            )
        direction = torch.zeros((env.num_envs, 1), device=env.device)
    frame = torch.cat(
        (
            root_angular_velocity * CR1_TRANSFORMER_VISION_ANGULAR_VELOCITY_SCALE,
            root_rotation,
            joint_position_normalized,
            joint_velocity * CR1_TRANSFORMER_VISION_JOINT_VELOCITY_SCALE,
            env.action_manager.action,
            direction,
        ),
        dim=-1,
    )
    return torch.clamp(
        frame,
        min=-CR1_TRANSFORMER_VISION_PROPRIO_CLIP,
        max=CR1_TRANSFORMER_VISION_PROPRIO_CLIP,
    )


def _uniform_sensor_noise(value: torch.Tensor, magnitude: float) -> torch.Tensor:
    """Sample zero-mean uniform sensor noise with the input tensor's layout."""

    if magnitude < 0.0:
        raise ValueError(f"Sensor-noise magnitude must be non-negative, got {magnitude}")
    if magnitude == 0.0:
        return torch.zeros_like(value)
    return (2.0 * torch.rand_like(value) - 1.0) * magnitude


def cr1_transformer_vision_base_lin_vel_target(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the body-frame root linear velocity used by the velocity head."""

    return env.scene.articulations["robot"].data.root_lin_vel_b


class cr1_transformer_vision_history(ManagerTermBase):
    """Maintain policy-rate proprio and camera-rate depth histories.

    The manager executes the existing actor depth group first.  This term then
    consumes that cached frame without an additional RTX/Warp render. Warp
    frame IDs keep held images from displacing distinct depth history frames.
    Reset environments receive repeated current frames instead of old-episode
    or zero-filled temporal inputs.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._num_proprio_frames = int(params.get("num_proprio_frames", 10))
        self._num_depth_frames = int(params.get("num_depth_frames", 2))
        self._depth_height = int(params.get("depth_height", 60))
        self._depth_width = int(params.get("depth_width", 80))
        self._proprio_interval_s = float(params.get("proprio_interval_s", env.step_dt))
        configured_depth_offsets = params.get("depth_history_offsets_s")
        if configured_depth_offsets is None:
            configured_depth_offsets = tuple(0.2 * index for index in range(self._num_depth_frames))
        self._depth_history_offsets_s = tuple(float(value) for value in configured_depth_offsets)
        self._legacy_policy_history = (
            "proprio_interval_s" not in params and "depth_history_offsets_s" not in params
        )
        self._history_buffer_capacity = int(params.get("history_buffer_capacity", 64))
        self._climb_direction_value = float(params.get("climb_direction_value", 1.0))
        self._climb_command_name = params.get("climb_command_name")
        self._include_climb_command = bool(params.get("include_climb_command", True))
        if self._climb_command_name is not None:
            self._climb_command_name = str(self._climb_command_name)
        if not self._include_climb_command and self._climb_command_name is not None:
            raise ValueError(
                "climb_command_name must be omitted when include_climb_command is false"
            )
        self._depth_mean = float(params.get("depth_mean", 0.0))
        self._depth_scale = float(params.get("depth_scale", 1.0))
        self._add_proprio_noise = bool(params.get("add_proprio_noise", False))
        self._root_angular_velocity_noise = float(
            params.get("root_angular_velocity_noise", 0.5)
        )
        self._orientation_noise_rad = float(params.get("orientation_noise_rad", 0.05))
        self._joint_position_noise_normalized = float(
            params.get("joint_position_noise_normalized", 0.01)
        )
        self._joint_velocity_noise = float(params.get("joint_velocity_noise", 1.5))
        if self._num_proprio_frames <= 0 or self._num_depth_frames <= 0:
            raise ValueError("TransformerVision history lengths must be positive")
        if self._proprio_interval_s <= 0.0 or not math.isfinite(self._proprio_interval_s):
            raise ValueError("proprio_interval_s must be finite and positive")
        if len(self._depth_history_offsets_s) != self._num_depth_frames:
            raise ValueError(
                "depth_history_offsets_s length must equal num_depth_frames"
            )
        if (
            any(value < 0.0 or not math.isfinite(value) for value in self._depth_history_offsets_s)
            or tuple(sorted(self._depth_history_offsets_s)) != self._depth_history_offsets_s
            or self._depth_history_offsets_s[0] != 0.0
        ):
            raise ValueError(
                "depth_history_offsets_s must be finite, non-negative, sorted, and start at 0"
            )
        if self._history_buffer_capacity < max(self._num_proprio_frames, self._num_depth_frames, 2):
            raise ValueError("history_buffer_capacity is too small for the requested histories")
        if self._depth_scale <= 0.0:
            raise ValueError("TransformerVision depth_scale must be positive")
        for name, value in (
            ("root_angular_velocity_noise", self._root_angular_velocity_noise),
            ("orientation_noise_rad", self._orientation_noise_rad),
            (
                "joint_position_noise_normalized",
                self._joint_position_noise_normalized,
            ),
            ("joint_velocity_noise", self._joint_velocity_noise),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        proprio_dim = cr1_transformer_vision_proprio_frame(
            env,
            climb_direction_value=self._climb_direction_value,
            climb_command_name=self._climb_command_name,
            include_climb_command=self._include_climb_command,
        ).shape[1]
        self._proprio_history = torch.zeros(
            (env.num_envs, self._num_proprio_frames, proprio_dim),
            dtype=torch.float32,
            device=env.device,
        )
        self._depth_history = torch.zeros(
            (env.num_envs, self._num_depth_frames, self._depth_height, self._depth_width),
            dtype=torch.float32,
            device=env.device,
        )
        # Fixed-size device tensors act as per-environment shared ring buffers.
        # They retain timestamps so sampling is temporal rather than “last N calls”.
        capacity = self._history_buffer_capacity
        self._proprio_buffer = torch.zeros(
            env.num_envs, capacity, proprio_dim, dtype=torch.float32, device=env.device
        )
        self._command_buffer = torch.zeros(
            env.num_envs, capacity, 1, dtype=torch.float32, device=env.device
        )
        self._depth_buffer = torch.zeros(
            env.num_envs, capacity, self._depth_height, self._depth_width,
            dtype=torch.float32, device=env.device
        )
        self._proprio_timestamps = torch.full(
            (env.num_envs, capacity), -1, dtype=torch.long, device=env.device
        )
        self._depth_timestamps = torch.full_like(self._proprio_timestamps, -1)
        self._proprio_write_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._depth_write_index = torch.zeros_like(self._proprio_write_index)
        self._proprio_count = torch.zeros_like(self._proprio_write_index)
        self._depth_count = torch.zeros_like(self._proprio_write_index)
        self._proprio_stride_steps = max(1, round(self._proprio_interval_s / float(env.step_dt)))
        self._depth_offset_steps = torch.tensor(
            [round(value / float(env.step_dt)) for value in self._depth_history_offsets_s],
            dtype=torch.long, device=env.device
        )
        self._pending_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_update_step: int | None = None
        self._last_depth_frame_id = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self._depth_updated = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self._command_timestamps = torch.full_like(self._proprio_timestamps, -1)
        self._command_write_index = torch.zeros_like(self._proprio_write_index)
        self._command_count = torch.zeros_like(self._proprio_write_index)
        self._last_depth_capture_step = torch.full_like(self._last_depth_frame_id, -1)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._pending_reset.numel(), device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._pending_reset[env_ids] = True
        # Advanced-index reads return copies; use indexed assignment so a
        # partial reset really clears the backing ring tensors.
        self._proprio_buffer[env_ids] = 0
        self._command_buffer[env_ids] = 0
        self._depth_buffer[env_ids] = 0
        self._proprio_timestamps[env_ids] = -1
        self._depth_timestamps[env_ids] = -1
        self._proprio_write_index[env_ids] = 0
        self._depth_write_index[env_ids] = 0
        self._proprio_count[env_ids] = 0
        self._depth_count[env_ids] = 0
        self._last_depth_frame_id[env_ids] = -1
        self._last_depth_capture_step[env_ids] = -1
        # Re-evaluate the complete policy boundary after a partial reset; this
        # prevents a stale same-step cache from being sampled for the reset env.
        self._last_update_step = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        num_proprio_frames: int = 10,
        num_depth_frames: int = 2,
        depth_height: int = 60,
        depth_width: int = 80,
        climb_direction_value: float = 1.0,
        climb_command_name: str | None = None,
        include_climb_command: bool = True,
        depth_mean: float = 0.0,
        depth_scale: float = 1.0,
        add_proprio_noise: bool = False,
        root_angular_velocity_noise: float = 0.5,
        orientation_noise_rad: float = 0.05,
        joint_position_noise_normalized: float = 0.01,
        joint_velocity_noise: float = 1.5,
        proprio_interval_s: float | None = None,
        depth_history_offsets_s: tuple[float, ...] = (0.0, 0.2),
        history_buffer_capacity: int | None = None,
    ) -> torch.Tensor:
        """Return time-sampled proprio/command and visual histories.

        Camera frames are inserted only when the camera clock produces a new
        exposure.  The policy receives the nearest available frame at the
        requested capture times, ordered oldest-to-newest.  Proprioception and
        command are sampled from their own rings at a fixed 0.1 s interval,
        rather than from the last N policy calls.
        """

        self._validate_runtime_parameters(
            num_proprio_frames, num_depth_frames, depth_height, depth_width,
            climb_direction_value, climb_command_name, include_climb_command,
            depth_mean, depth_scale, add_proprio_noise, root_angular_velocity_noise,
            orientation_noise_rad, joint_position_noise_normalized,
            joint_velocity_noise,
            self._proprio_interval_s if proprio_interval_s is None else proprio_interval_s,
            depth_history_offsets_s,
            self._history_buffer_capacity if history_buffer_capacity is None else history_buffer_capacity,
        )
        depth = getattr(env, "_cr1_visual_depth_frame", None)
        if depth is None:
            raise RuntimeError(
                "TransformerVision history requires transformer_vision_depth to run before transformer_vision_inputs"
            )
        expected_depth_shape = (env.num_envs, 1, self._depth_height, self._depth_width)
        if tuple(depth.shape) != expected_depth_shape:
            raise RuntimeError(
                f"Expected cached visual depth shape {expected_depth_shape}, got {tuple(depth.shape)}"
            )
        proprio = cr1_transformer_vision_proprio_frame(
            env, climb_direction_value=self._climb_direction_value,
            climb_command_name=self._climb_command_name,
            include_climb_command=self._include_climb_command,
            add_noise=self._add_proprio_noise,
            root_angular_velocity_noise=self._root_angular_velocity_noise,
            orientation_noise_rad=self._orientation_noise_rad,
            joint_position_noise_normalized=self._joint_position_noise_normalized,
            joint_velocity_noise=self._joint_velocity_noise,
        )
        if proprio.shape[-1] != self._proprio_history.shape[-1]:
            raise RuntimeError(
                "CR1 TransformerVision proprioceptive dimension changed after history initialization: "
                f"expected {self._proprio_history.shape[-1]}, got {proprio.shape[-1]}"
            )

        current_step = int(env.common_step_counter)
        encoder_depth = (depth[:, 0] - self._depth_mean) * self._depth_scale
        frame_id = getattr(env, "_cr1_visual_depth_frame_id", None)
        capture_step = getattr(env, "_cr1_visual_depth_capture_step", None)
        if capture_step is None:
            capture_steps = torch.full(
                (env.num_envs,), current_step, dtype=torch.long, device=self.device
            )
        else:
            capture_steps = torch.as_tensor(capture_step, dtype=torch.long, device=self.device)
            if tuple(capture_steps.shape) != (env.num_envs,):
                raise RuntimeError("visual depth capture steps must have one value per environment")
            capture_steps = torch.where(capture_steps >= 0, capture_steps,
                                        torch.full_like(capture_steps, current_step))
        if frame_id is not None:
            frame_id = torch.as_tensor(frame_id, dtype=torch.long, device=self.device)
            if tuple(frame_id.shape) != (env.num_envs,):
                raise RuntimeError("visual depth frame IDs must have one value per environment")
            new_depth = frame_id != self._last_depth_frame_id
        elif capture_step is not None:
            new_depth = capture_steps != self._last_depth_capture_step
        else:
            # Non-Warp camera terms do not expose a frame ID; their term call
            # is the only available exposure boundary.
            new_depth = torch.ones_like(self._pending_reset)

        reset_now = self._pending_reset.clone()
        if self._last_update_step != current_step:
            self._append_ring(
                self._proprio_buffer, self._proprio_timestamps,
                self._proprio_write_index, self._proprio_count,
                proprio, torch.full_like(self._proprio_write_index, current_step),
                torch.ones_like(self._pending_reset),
            )
            # Command uses the same timestamped sample, but has its own storage
            # and write cursor so the two buffers can be inspected independently.
            self._append_ring(
                self._command_buffer, self._command_timestamps,
                self._command_write_index, self._command_count,
                proprio[..., -1:], torch.full_like(self._command_write_index, current_step),
                torch.ones_like(self._pending_reset),
            )
            depth_mask = new_depth | reset_now
            self._append_ring(
                self._depth_buffer, self._depth_timestamps,
                self._depth_write_index, self._depth_count,
                encoder_depth, capture_steps, depth_mask,
            )
            self._depth_updated.copy_(new_depth | reset_now)
            self._last_update_step = current_step
        elif torch.any(reset_now):
            self._append_ring(
                self._proprio_buffer, self._proprio_timestamps,
                self._proprio_write_index, self._proprio_count,
                proprio, torch.full_like(self._proprio_write_index, current_step),
                reset_now,
            )
            self._append_ring(
                self._command_buffer, self._command_timestamps,
                self._command_write_index, self._command_count,
                proprio[..., -1:], torch.full_like(self._command_write_index, current_step),
                reset_now,
            )
            self._append_ring(
                self._depth_buffer, self._depth_timestamps,
                self._depth_write_index, self._depth_count,
                encoder_depth, capture_steps, reset_now,
            )
            self._depth_updated.copy_(new_depth | reset_now)
        if frame_id is not None:
            self._last_depth_frame_id.copy_(frame_id)
        self._last_depth_capture_step.copy_(capture_steps)

        proprio_offsets = torch.arange(
            self._num_proprio_frames - 1, -1, -1, device=self.device, dtype=torch.long
        ) * self._proprio_stride_steps
        proprio_targets = current_step - proprio_offsets[None, :].expand(env.num_envs, -1)
        sampled_proprio = self._sample_ring(
            self._proprio_buffer, self._proprio_timestamps, proprio_targets
        )
        command_targets = current_step - proprio_offsets[None, :].expand(env.num_envs, -1)
        sampled_command = self._sample_ring(
            self._command_buffer, self._command_timestamps, command_targets
        )
        sampled_proprio[..., -1:] = sampled_command
        self._proprio_history.copy_(sampled_proprio)

        if self._legacy_policy_history:
            sampled_depth = self._sample_latest_ring(
                self._depth_buffer, self._depth_timestamps, self._num_depth_frames
            )
        else:
            depth_targets = current_step - self._depth_offset_steps[None, :].expand(env.num_envs, -1)
            sampled_depth = self._sample_ring(
                self._depth_buffer, self._depth_timestamps, depth_targets
            )
            # Offsets are configured current-first; policy tensors are chronological.
            sampled_depth = torch.flip(sampled_depth, dims=(1,))
        self._depth_history.copy_(sampled_depth)
        self._pending_reset[reset_now] = False
        env._cr1_transformer_vision_history_cache = {
            "depth_history": self._depth_history,
            "proprio_history": self._proprio_history,
            "command_history": sampled_command,
            "depth_updated": self._depth_updated[:, None].to(dtype=torch.float32),
        }
        return self._proprio_history

    def _validate_runtime_parameters(
        self, num_proprio_frames: int, num_depth_frames: int, depth_height: int,
        depth_width: int, climb_direction_value: float, climb_command_name: str | None,
        include_climb_command: bool, depth_mean: float, depth_scale: float,
        add_proprio_noise: bool, root_angular_velocity_noise: float,
        orientation_noise_rad: float, joint_position_noise_normalized: float,
        joint_velocity_noise: float, proprio_interval_s: float,
        depth_history_offsets_s: tuple[float, ...], history_buffer_capacity: int,
    ) -> None:
        configured = (
            self._num_proprio_frames, self._num_depth_frames, self._depth_height,
            self._depth_width, self._climb_direction_value, self._climb_command_name,
            self._include_climb_command, self._depth_mean, self._depth_scale,
            self._add_proprio_noise, self._root_angular_velocity_noise,
            self._orientation_noise_rad, self._joint_position_noise_normalized,
            self._joint_velocity_noise, self._proprio_interval_s,
            self._depth_history_offsets_s, self._history_buffer_capacity,
        )
        received = (
            int(num_proprio_frames), int(num_depth_frames), int(depth_height),
            int(depth_width), float(climb_direction_value),
            None if climb_command_name is None else str(climb_command_name),
            bool(include_climb_command), float(depth_mean), float(depth_scale),
            bool(add_proprio_noise), float(root_angular_velocity_noise),
            float(orientation_noise_rad), float(joint_position_noise_normalized),
            float(joint_velocity_noise), float(proprio_interval_s),
            tuple(float(value) for value in depth_history_offsets_s),
            int(history_buffer_capacity),
        )
        if received != configured:
            raise RuntimeError("TransformerVision history parameters changed after initialization")

    @staticmethod
    def _append_ring(buffer, timestamps, write_index, count, values, sample_time, mask):
        env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        slots = write_index[env_ids]
        buffer[env_ids, slots] = values[env_ids]
        timestamps[env_ids, slots] = sample_time[env_ids]
        capacity = buffer.shape[1]
        write_index[env_ids] = (slots + 1) % capacity
        count[env_ids] = torch.clamp(count[env_ids] + 1, max=capacity)

    @staticmethod
    def _sample_latest_ring(buffer, timestamps, count):
        valid = timestamps >= 0
        fill = torch.where(valid, timestamps, torch.full_like(timestamps, -1))
        _, indices = torch.topk(fill, k=min(count, buffer.shape[1]), dim=-1, largest=True)
        indices = torch.flip(indices, dims=(1,))
        gather_shape = (indices.shape[0], indices.shape[1], *buffer.shape[2:])
        expanded = indices.reshape(indices.shape[0], indices.shape[1], *([1] * (buffer.ndim - 2))).expand(gather_shape)
        sampled = torch.gather(buffer, 1, expanded)
        selected_times = torch.gather(timestamps, 1, indices)
        sampled = torch.where(
            (selected_times >= 0).reshape(*selected_times.shape, *([1] * (buffer.ndim - 2))),
            sampled, sampled[:, -1:].expand_as(sampled),
        )
        if sampled.shape[1] < count:
            sampled = torch.cat((sampled[:, :1].expand(-1, count - sampled.shape[1], *sampled.shape[2:]), sampled), dim=1)
        return sampled

    @staticmethod
    def _sample_ring(buffer, timestamps, targets):
        valid = timestamps >= 0
        distance = (timestamps[:, None, :] - targets[:, :, None]).abs()
        distance = distance.masked_fill(~valid[:, None, :], torch.iinfo(torch.long).max)
        indices = distance.argmin(dim=-1)
        any_valid = valid.any(dim=-1)
        indices = torch.where(any_valid[:, None], indices, torch.zeros_like(indices))
        gather_shape = (indices.shape[0], indices.shape[1], *buffer.shape[2:])
        expanded = indices.reshape(indices.shape[0], indices.shape[1], *([1] * (buffer.ndim - 2))).expand(gather_shape)
        return torch.gather(buffer, 1, expanded)


def cr1_transformer_vision_depth_history(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the depth history prepared by :class:`cr1_transformer_vision_history`."""

    cache = getattr(env, "_cr1_transformer_vision_history_cache", None)
    if cache is None:
        raise RuntimeError("TransformerVision depth history must follow TransformerVision proprio history")
    return cache["depth_history"]


def cr1_transformer_vision_depth_updated(env: ManagerBasedEnv) -> torch.Tensor:
    """Return whether each environment received a fresh depth exposure this step."""

    cache = getattr(env, "_cr1_transformer_vision_history_cache", None)
    if cache is None:
        raise RuntimeError(
            "TransformerVision depth update mask must follow TransformerVision proprio history"
        )
    return cache["depth_updated"]


# Compatibility aliases for the retired prototype task. New code must use the
# TransformerVision names above and must not depend on these aliases.
cr1_ssr_proprio_frame = cr1_transformer_vision_proprio_frame
cr1_ssr_base_lin_vel_target = cr1_transformer_vision_base_lin_vel_target
cr1_ssr_visual_history = cr1_transformer_vision_history
cr1_ssr_depth_history = cr1_transformer_vision_depth_history


def _cr1_wood_step_box_specs(spec_set: str):
    from ladder.tasks.ladder_climb.config.cr1.wood_step_geometry_dr_env_cfg import (
        STOP_COMMAND_GROUNDED_THIN_RAIL_BOX_SPECS,
        THIN_RAIL_WOOD_STEP_BOX_SPECS,
        WOOD_STEP_BOX_SPECS,
    )

    if spec_set == "standard":
        return WOOD_STEP_BOX_SPECS
    if spec_set == "thin_rails":
        return THIN_RAIL_WOOD_STEP_BOX_SPECS
    if spec_set == "thin_rails_grounded":
        return STOP_COMMAND_GROUNDED_THIN_RAIL_BOX_SPECS
    raise ValueError(
        "ladder_box_spec_set must be 'standard', 'thin_rails', or "
        "'thin_rails_grounded', got "
        f"{spec_set!r}"
    )


def _compute_cr1_limb_heightmap_target(
    env: ManagerBasedEnv,
    robot_name: str = "robot",
    ladder_name: str = "ladder",
    rows: int = 10,
    columns: int = 3,
    x_extent: float = 0.315,
    y_extent: float = 0.160,
    ground_height: float = -0.03,
    ray_start_height: float = 2.0,
    max_ray_distance: float = 4.0,
    ladder_box_spec_set: str = "standard",
    output: str = "relative_height_m",
) -> torch.Tensor:
    """Return CR1's four-limb relative-height targets for auxiliary training.

    The returned tensor is ``[B, 4, rows, columns]`` in the fixed order
    ``left_hand, right_hand, left_foot, right_foot``.  Hand references are
    sphere centers and foot references are the virtual sole points.  Rays and
    grid axes remain aligned with world gravity and world X/Y; limb rotations
    are not used to rotate the map.  ``hit_valid`` is returned as a float mask
    when requested and must mask the later reconstruction loss.
    """

    if output not in (
        "relative_height_m",
        "relative_height_flat",
        "hit_valid",
        "hit_class",
    ):
        raise ValueError(f"Unsupported CR1 limb height-map output: {output!r}")

    current_step = int(env.common_step_counter)
    query_key = (
        robot_name,
        ladder_name,
        rows,
        columns,
        x_extent,
        y_extent,
        ground_height,
        ray_start_height,
        max_ray_distance,
        ladder_box_spec_set,
    )
    cache = getattr(env, "_cr1_limb_heightmap_target_cache", None)
    if heightmap_cache_is_current(
        cache, common_step_counter=current_step, output=output
    ) and cache.get("query_key") == query_key:
        return cache[output]

    robot = env.scene.articulations[robot_name]
    ladder = env.scene.rigid_object_collections[ladder_name]
    device = robot.data.body_pos_w.device
    dtype = robot.data.body_pos_w.dtype

    centers = []
    for side in ("left", "right"):
        hand_index = robot.body_names.index(f"{side}_sphere_hand_link")
        centers.append(robot.data.body_pos_w[:, hand_index])

    for side in ("left", "right"):
        sole_name = f"{side}_sole_contact_point"
        ankle_name, sole_offset = CR1_VIRTUAL_BODY_SPECS[sole_name]
        ankle_index = robot.body_names.index(ankle_name)
        offset = torch.as_tensor(sole_offset, dtype=dtype, device=device).expand(
            robot.data.body_pos_w.shape[0], -1
        )
        centers.append(
            robot.data.body_pos_w[:, ankle_index]
            + quat_apply(robot.data.body_quat_w[:, ankle_index], offset)
        )
    limb_centers_w = torch.stack(centers, dim=1)

    box_specs = _cr1_wood_step_box_specs(ladder_box_spec_set)
    specs_by_name = {spec.name: spec for spec in box_specs}
    runtime_names = set(ladder.object_names)
    spec_names = set(specs_by_name)
    if runtime_names != spec_names:
        raise ValueError(
            "CR1 wood-step runtime objects and height-map box specs differ: "
            f"missing_specs={sorted(runtime_names - spec_names)}, "
            f"missing_runtime_objects={sorted(spec_names - runtime_names)}"
        )
    box_sizes = torch.as_tensor(
        [specs_by_name[name].size for name in ladder.object_names],
        dtype=dtype,
        device=device,
    )
    box_classes = torch.tensor(
        [
            HEIGHTMAP_RUNG
            if name in WOOD_STEP_RUNG_BOX_NAMES
            else HEIGHTMAP_RAIL
            if name.startswith("front_zigzag_side_board_")
            else LADDER_OTHER
            for name in ladder.object_names
        ],
        dtype=torch.long,
        device=device,
    )
    map_cfg = LimbHeightMapCfg(
        rows=rows,
        columns=columns,
        x_extent=x_extent,
        y_extent=y_extent,
        ray_start_height=ray_start_height,
        max_ray_distance=max_ray_distance,
    )
    ground_height_w = torch.full(
        (env.num_envs,), ground_height, dtype=dtype, device=device
    )
    maps = [
        sample_limb_heightmap(
            limb_centers_w=limb_centers_w[:, index],
            box_positions_w=ladder.data.object_pos_w,
            box_quaternions_wxyz=ladder.data.object_quat_w,
            box_sizes=box_sizes,
            box_classes=box_classes,
            cfg=map_cfg,
            ground_height_w=ground_height_w,
        )
        for index in range(4)
    ]
    cache = {
        "common_step_counter": current_step,
        "query_key": query_key,
        "relative_height_m": torch.stack(
            [height_map.relative_height_m for height_map in maps], dim=1
        ),
        "hit_valid": torch.stack(
            [height_map.hit_valid for height_map in maps], dim=1
        ).to(dtype=dtype),
        "hit_class": torch.stack(
            [height_map.hit_class for height_map in maps], dim=1
        ),
        # Diagnostic geometry for placing target and predicted relative maps
        # back on the same world-XY sampling grids. These keys are not exposed
        # as policy or critic observations.
        "plane_points_w": torch.stack(
            [height_map.plane_points_w for height_map in maps], dim=1
        ),
    }
    cache["relative_height_flat"] = cache["relative_height_m"].reshape(
        env.num_envs, -1
    )
    env._cr1_limb_heightmap_target_cache = cache
    return cache[output]


class cr1_limb_heightmap_target(ManagerTermBase):
    """Share one limb-map ray query within a step and invalidate it on reset."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids=None) -> None:
        # A manual reset can occur without advancing common_step_counter. Clear
        # the shared cache so the post-reset observation cannot reuse geometry
        # from the previous episode.
        self._env._cr1_limb_heightmap_target_cache = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        robot_name: str = "robot",
        ladder_name: str = "ladder",
        rows: int = 10,
        columns: int = 3,
        x_extent: float = 0.315,
        y_extent: float = 0.160,
        ground_height: float = -0.03,
        ray_start_height: float = 2.0,
        max_ray_distance: float = 4.0,
        ladder_box_spec_set: str = "standard",
        output: str = "relative_height_m",
    ) -> torch.Tensor:
        return _compute_cr1_limb_heightmap_target(
            env,
            robot_name=robot_name,
            ladder_name=ladder_name,
            rows=rows,
            columns=columns,
            x_extent=x_extent,
            y_extent=y_extent,
            ground_height=ground_height,
            ray_start_height=ray_start_height,
            max_ray_distance=max_ray_distance,
            ladder_box_spec_set=ladder_box_spec_set,
            output=output,
        )


class warp_visual_depth(ManagerTermBase):
    """GPU-resident Warp depth/RFM observation for the CR1 visual policy.

    This replaces only the RTX camera source.  The actor still receives the
    same processed NCHW depth format and the critic remains privileged.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from ladder.tasks.ladder_climb.config.cr1.wood_step_env_cfg import CR1_GROUND_Z
        params = cfg.params
        self._robot_name = str(params.get("robot_name", "robot"))
        self._ladder_name = str(params.get("ladder_name", "ladder"))
        self._camera_body_name = str(params.get("camera_body_name", "d435_head_link"))
        self._ladder_box_spec_set = str(params.get("ladder_box_spec_set", "standard"))
        self._augmentation_cfg = _visual_depth_augmentation_cfg(params)
        self._ground_height = float(params.get("ground_height", CR1_GROUND_Z))
        self._translation_range = tuple(params.get("translation_range", (-0.01, 0.01)))
        self._rotation_range = tuple(params.get("rotation_range", (-0.0261799388, 0.0261799388)))
        self._horizontal_fov_deg = float(params.get("horizontal_fov_deg", 87.0))
        self._vertical_fov_deg = float(params.get("vertical_fov_deg", 58.0))
        self._horizontal_fov_randomization_range_deg = tuple(
            params.get("horizontal_fov_randomization_range_deg", (0.0, 0.0))
        )
        self._vertical_fov_randomization_range_deg = tuple(
            params.get("vertical_fov_randomization_range_deg", (0.0, 0.0))
        )
        ranges = (
            self._translation_range,
            self._rotation_range,
            self._horizontal_fov_randomization_range_deg,
            self._vertical_fov_randomization_range_deg,
        )
        if any(len(value_range) != 2 for value_range in ranges):
            raise ValueError("Warp camera randomization ranges must each have two values")
        for name, nominal, value_range in (
            (
                "horizontal",
                self._horizontal_fov_deg,
                self._horizontal_fov_randomization_range_deg,
            ),
            (
                "vertical",
                self._vertical_fov_deg,
                self._vertical_fov_randomization_range_deg,
            ),
        ):
            if value_range[0] > value_range[1]:
                raise ValueError(f"Warp camera {name} FOV randomization range is reversed")
            if nominal + value_range[0] <= 0.0 or nominal + value_range[1] >= 180.0:
                raise ValueError(f"Randomized Warp camera {name} FOV must remain in (0, 180)")

        camera_cfg = WarpCameraCfg(
            width=int(params.get("width", 80)),
            height=int(params.get("height", 60)),
            horizontal_fov_deg=self._horizontal_fov_deg,
            vertical_fov_deg=self._vertical_fov_deg,
            horizontal_fov_margin_deg=max(
                abs(value) for value in self._horizontal_fov_randomization_range_deg
            ),
            vertical_fov_margin_deg=max(
                abs(value) for value in self._vertical_fov_randomization_range_deg
            ),
            min_depth=self._augmentation_cfg.min_depth,
            max_depth=self._augmentation_cfg.max_depth,
        )
        robot = env.scene.articulations[self._robot_name]
        ladder = env.scene.rigid_object_collections[self._ladder_name]
        if self._camera_body_name not in robot.body_names:
            raise ValueError(f"Unknown Warp camera body {self._camera_body_name!r}")
        self._camera_body_id = robot.body_names.index(self._camera_body_name)
        self._renderer = CR1WarpBatchCamera(
            camera_cfg,
            num_envs=env.num_envs,
            robot_urdf=CR1_URDF,
            robot_body_names=robot.body_names,
            ladder_box_specs=_cr1_wood_step_box_specs(self._ladder_box_spec_set),
            ladder_object_names=ladder.object_names,
            rung_box_names=WOOD_STEP_RUNG_BOX_NAMES,
            device=env.device,
        )
        self._local_camera_positions = torch.zeros((env.num_envs, 3), device=env.device)
        self._local_camera_quaternions = torch.zeros((env.num_envs, 4), device=env.device)
        self._local_camera_quaternions[:, 0] = 1.0
        self._horizontal_fov_tangents = torch.empty(env.num_envs, device=env.device)
        self._vertical_fov_tangents = torch.empty(env.num_envs, device=env.device)
        self._sample_camera_parameters(torch.arange(env.num_envs, device=env.device))
        # Warp is queried at policy boundaries, independently of its camera clock.
        # Keep fractional phase: rounding 50/30 to two policy steps gives 25 Hz.
        self._update_period = float(params.get("update_period", 0.0))
        if not math.isfinite(self._update_period) or self._update_period < 0.0:
            raise ValueError("Warp camera update_period must be finite and non-negative")
        if 0.0 < self._update_period < env.step_dt - 1.0e-9:
            raise ValueError("Warp camera rate cannot exceed the policy-boundary sampling rate")
        self._last_capture_slot: int | None = None
        self._pending_capture = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._depth_frame = torch.zeros(
            (env.num_envs, 1, camera_cfg.output_height, camera_cfg.output_width),
            device=env.device,
        )
        self._frame_id = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._capture_step = torch.full_like(self._frame_id, -1)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._sample_camera_parameters(env_ids)
        # A reset must never expose the previous episode's cached image. It does
        # not advance or rephase the camera clock for the other environments.
        self._pending_capture[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedEnv,
        robot_name: str = "robot",
        ladder_name: str = "ladder",
        camera_body_name: str = "d435_head_link",
        width: int = 80,
        height: int = 60,
        horizontal_fov_deg: float = 87.0,
        vertical_fov_deg: float = 58.0,
        min_depth: float = 0.1,
        max_depth: float = 2.0,
        gaussian_std: float = 0.005,
        pixel_dropout_probability: float = 0.05,
        rung_focused_mask_probability: float = 0.1,
        translation_range: tuple[float, float] = (-0.01, 0.01),
        rotation_range: tuple[float, float] = (-0.0261799388, 0.0261799388),
        horizontal_fov_randomization_range_deg: tuple[float, float] = (0.0, 0.0),
        vertical_fov_randomization_range_deg: tuple[float, float] = (0.0, 0.0),
        ladder_box_spec_set: str = "standard",
        update_period: float = 0.0,
    ) -> torch.Tensor:
        """Return the latest camera sample, rendering only when due or reset.

        IsaacLab passes ``ObservationTermCfg.params`` to every invocation of a
        stateful term as well as to its constructor, and its 2.1 manager
        validates the full explicit signature. Configuration is consumed once
        in ``__init__``; the repeated parameters are intentionally unused.
        """
        current_step = int(env.common_step_counter)
        capture_slot = (
            math.floor(current_step * env.step_dt / self._update_period + 1.0e-8)
            if self._update_period > 0.0
            else current_step
        )
        if capture_slot != self._last_capture_slot:
            env_ids = None
        else:
            env_ids = self._pending_capture.nonzero(as_tuple=False).squeeze(-1)
        if env_ids is None or env_ids.numel() > 0:
            self._capture(env, env_ids)
            selection = slice(None) if env_ids is None else env_ids
            self._pending_capture[selection] = False
            self._frame_id[selection] += 1
            self._capture_step[selection] = current_step
            self._last_capture_slot = capture_slot
        env._cr1_visual_depth_frame = self._depth_frame
        env._cr1_visual_depth_frame_id = self._frame_id
        env._cr1_visual_depth_capture_step = self._capture_step
        return self._depth_frame

    def _capture(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
        selection = slice(None) if env_ids is None else env_ids
        robot = env.scene.articulations[self._robot_name]
        ladder = env.scene.rigid_object_collections[self._ladder_name]
        positions, quaternions = combine_frame_transforms(
            robot.data.body_pos_w[selection, self._camera_body_id],
            robot.data.body_quat_w[selection, self._camera_body_id],
            self._local_camera_positions[selection],
            self._local_camera_quaternions[selection],
        )
        camera_quaternions_opengl = convert_camera_frame_orientation_convention(
            quaternions, origin="world", target="opengl"
        ).contiguous()
        output = self._renderer.render(
            camera_positions_w=positions.contiguous(),
            camera_quaternions_wxyz_opengl=camera_quaternions_opengl,
            robot_body_positions_w=robot.data.body_pos_w[selection],
            robot_body_quaternions_wxyz=robot.data.body_quat_w[selection],
            ladder_positions_w=ladder.data.object_pos_w[selection],
            ladder_quaternions_wxyz=ladder.data.object_quat_w[selection],
            ground_height=self._ground_height,
            env_ids=env_ids,
            horizontal_tangents=self._horizontal_fov_tangents[selection],
            vertical_tangents=self._vertical_fov_tangents[selection],
        )
        depth, _ = process_warp_visual_depth(
            output.depth,
            output.hit_class,
            rung_class=RUNG,
            robot_class=ROBOT,
            cfg=self._augmentation_cfg,
        )
        self._depth_frame[selection] = depth

    def _sample_camera_parameters(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        self._local_camera_positions[env_ids] = sample_uniform(
            self._translation_range[0],
            self._translation_range[1],
            (count, 3),
            device=self.device,
        )
        local_rpy = sample_uniform(
            self._rotation_range[0],
            self._rotation_range[1],
            (count, 3),
            device=self.device,
        )
        self._local_camera_quaternions[env_ids] = quat_from_euler_xyz(
            local_rpy[:, 0], local_rpy[:, 1], local_rpy[:, 2]
        )
        horizontal_fov = self._horizontal_fov_deg + sample_uniform(
            self._horizontal_fov_randomization_range_deg[0],
            self._horizontal_fov_randomization_range_deg[1],
            (count,),
            device=self.device,
        )
        vertical_fov = self._vertical_fov_deg + sample_uniform(
            self._vertical_fov_randomization_range_deg[0],
            self._vertical_fov_randomization_range_deg[1],
            (count,),
            device=self.device,
        )
        self._horizontal_fov_tangents[env_ids] = torch.tan(torch.deg2rad(horizontal_fov) * 0.5)
        self._vertical_fov_tangents[env_ids] = torch.tan(torch.deg2rad(vertical_fov) * 0.5)


def _visual_depth_augmentation_cfg(params: dict) -> object:
    """Build the shared paper-aligned augmentation configuration from term params."""

    from ladder.utils.visual_depth import DepthAugmentationCfg

    return DepthAugmentationCfg(
        min_depth=float(params.get("min_depth", 0.1)),
        max_depth=float(params.get("max_depth", 2.0)),
        gaussian_std=float(params.get("gaussian_std", 0.005)),
        pixel_dropout_probability=float(params.get("pixel_dropout_probability", 0.05)),
        rung_focused_mask_probability=float(params.get("rung_focused_mask_probability", 0.1)),
    )


def ladder_params(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.ladder_params


def contact_indicator(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.contact_indicator


def contact_rung_id(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.contact_rung_id.float() / max(command.ladder_geometry.num_rungs - 1, 1)


def reference_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos


def reference_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.joint_vel


def motion_phase(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    denominator = max(command.motion.time_step_total - 1, 1)
    return (command.time_steps.float() / denominator).unsqueeze(-1)


def reference_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference tracked-body positions expressed in the reference root frame."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.body_pos_w[:, :1, :].repeat(1, num_bodies, 1),
        command.body_quat_w[:, :1, :].repeat(1, num_bodies, 1),
        command.body_pos_w,
        command.body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def root_orientation_6d(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    rotation = matrix_from_quat(asset.data.root_quat_w)
    return rotation[..., :2].reshape(env.num_envs, -1)


def ladder_relative_root_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot root position in the sampled ladder axial/lateral/normal frame."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    relative = command.robot.data.root_pos_w - command.ladder_origins_w
    if command.cfg.ladder_relative_observation_frame == "world":
        return relative
    if command.cfg.ladder_relative_observation_frame != "ladder":
        raise ValueError(
            "ladder_relative_observation_frame must be 'world' or 'ladder', got "
            f"{command.cfg.ladder_relative_observation_frame!r}"
        )
    cosine = torch.cos(command.ladder_phi)
    sine = torch.sin(command.ladder_phi)
    axial = relative[:, 0] * cosine + relative[:, 2] * sine
    normal = -relative[:, 0] * sine + relative[:, 2] * cosine
    return torch.stack((axial, relative[:, 1], normal), dim=-1)


def target_contact_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    target_pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, len(command.limb_names), 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, len(command.limb_names), 1),
        command.target_contact_pos_w,
        None,
    )
    return target_pos_b.reshape(env.num_envs, -1)


def task_limb_goal_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Current/next four-limb ladder goals in the robot ``body`` frame."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    goal_pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, len(command.limb_names), 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, len(command.limb_names), 1),
        command.task_limb_goal_pos_w,
        None,
    )
    return goal_pos_b.reshape(env.num_envs, -1)


def task_limb_goal_valid(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Position-supervision mask for current/next four-limb ladder goals."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.task_goal_valid.float()


def motion_semantic_target_contact_pos_b(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Next stable four-limb reference points in the robot anchor frame."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    target_pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, len(command.limb_names), 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, len(command.limb_names), 1),
        command.next_stable_limb_target_pos_w,
        None,
    )
    target_pos_b = torch.where(
        (command.next_stable_target_frame >= 0)[:, None, None],
        target_pos_b,
        torch.zeros_like(target_pos_b),
    )
    return target_pos_b.reshape(env.num_envs, -1)


def motion_semantic_target_contact_valid(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Validity mask for the next stable four-limb support target."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.next_stable_limb_target_valid.unsqueeze(-1).float()


def next_stable_contact_indicator(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Desired four-limb contacts at the next stable support frame."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.next_stable_contact_indicator


def actual_support_relation(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Measured four-limb support identities relative to the pending anchor."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return encode_support_relation(
        command.actual_stable_support_rung_id,
        command.next_stable_support_rung_id,
        command.ladder_geometry.num_rungs,
    )


def current_expected_support_relation(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Current reference-stage support identities relative to the pending anchor."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return encode_support_relation(
        command.stable_support_rung_id,
        command.next_stable_support_rung_id,
        command.ladder_geometry.num_rungs,
    )


def next_stable_support_relation(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Pending stable support identities in their shared relative rung frame."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    next_ids = command.next_stable_support_rung_id
    return encode_support_relation(
        next_ids, next_ids, command.ladder_geometry.num_rungs
    )


def robot_limb_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Four configured end-effector positions in the robot anchor frame."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    limb_pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, len(command.limb_names), -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, len(command.limb_names), -1),
        command.robot_limb_pos_w,
        None,
    )
    return limb_pos_b.reshape(env.num_envs, -1)


def actual_ladder_contact_indicator(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Whether each limb physically contacts any ladder rung.

    The four existing sensors are filtered to rung collision prims, so this
    signal does not consult the reference contact label or target rung ID.
    """

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    sensor_names = {
        "left_hand": "left_hand_step_contacts",
        "right_hand": "right_hand_step_contacts",
        "left_foot": "left_foot_step_contacts",
        "right_foot": "right_foot_step_contacts",
    }
    values = []
    for limb_name in command.limb_names:
        sensor_name = sensor_names[limb_name]
        if sensor_name not in env.scene.sensors:
            raise KeyError(
                f"Privileged actual-ladder-contact observation requires sensor {sensor_name!r}"
            )
        matrix = env.scene.sensors[sensor_name].data.force_matrix_w
        values.append(
            (torch.linalg.vector_norm(matrix, dim=-1).amax(dim=(-1, -2))
             > command.cfg.contact_force_threshold).float()
        )
    return torch.stack(values, dim=-1)


def actual_stable_support_contact_indicator(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Whether each limb physically contacts its ground/rung support domain."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.actual_stable_support_contacts.float()


def robot_limb_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Four configured end-effector linear velocities in the robot frame."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    limb_velocity_w = command.robot.data.body_link_lin_vel_w[:, command.limb_body_indexes]
    velocity_b = quat_apply_inverse(
        command.robot_anchor_quat_w[:, None, :].expand(-1, len(command.limb_names), -1),
        limb_velocity_w,
    )
    return (velocity_b * CR1_TRANSFORMER_VISION_JOINT_VELOCITY_SCALE).reshape(
        env.num_envs, -1
    )


def motion_semantic_target_joint_pos(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Next stable joint pose in the visual proprioception normalization."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    robot = env.scene.articulations["robot"]
    scale = _cr1_transformer_vision_joint_position_scale(env, robot)
    return (command.motion_semantic_target_joint_pos - robot.data.default_joint_pos) / scale


def motion_semantic_target_root_orientation_6d(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Next stable reference-root orientation represented by two matrix columns."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    rotation = matrix_from_quat(command.motion_semantic_target_root_quat_w)
    return rotation[..., :2].reshape(env.num_envs, -1)


def motion_semantic_target_valid(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Whether the state-locked next-stable-pose supervision is valid."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.motion_semantic_target_valid.unsqueeze(-1).float()


def contact_error_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    anchor_pos_w = command.robot_anchor_pos_w[:, None, :].repeat(1, len(command.limb_names), 1)
    anchor_quat_w = command.robot_anchor_quat_w[:, None, :].repeat(1, len(command.limb_names), 1)
    limb_pos_b, _ = subtract_frame_transforms(
        anchor_pos_w,
        anchor_quat_w,
        command.robot_limb_pos_w,
        None,
    )
    target_pos_b, _ = subtract_frame_transforms(
        anchor_pos_w,
        command.robot_anchor_quat_w[:, None, :].repeat(1, len(command.limb_names), 1),
        command.target_contact_pos_w,
        None,
    )
    return (target_pos_b - limb_pos_b).reshape(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_body_pos_w[:, :1, :].repeat(1, num_bodies, 1),
        command.robot_body_quat_w[:, :1, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def robot_body_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    root_quat_w = command.robot_body_quat_w[:, :1, :].repeat(1, num_bodies, 1)
    return quat_apply_inverse(root_quat_w, command.robot_body_lin_vel_w).reshape(env.num_envs, -1)


def reference_body_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    root_quat_w = command.body_quat_w[:, :1, :].repeat(1, num_bodies, 1)
    return quat_apply_inverse(root_quat_w, command.body_lin_vel_w).reshape(env.num_envs, -1)


def body_pos_error_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return robot_body_pos_b(env, command_name) - reference_body_pos_b(env, command_name)


def body_lin_vel_error_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return robot_body_lin_vel_b(env, command_name) - reference_body_lin_vel_b(env, command_name)


def joint_pos_error(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.robot_joint_pos - command.joint_pos


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
