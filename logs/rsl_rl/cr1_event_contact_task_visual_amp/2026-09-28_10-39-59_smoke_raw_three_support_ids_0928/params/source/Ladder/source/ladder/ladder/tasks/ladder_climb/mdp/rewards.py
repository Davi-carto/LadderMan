from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ladder.tasks.ladder_climb.mdp.commands import LadderMotionCommand
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from ladder.utils.contact_force import (
    filtered_surface_force_max,
    non_target_rung_forces,
    target_rung_forces,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: LadderMotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    temperature: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track selected joint positions with the LadderMan exponential reward."""

    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(
        command.robot_joint_pos[:, asset_cfg.joint_ids] - command.joint_pos[:, asset_cfg.joint_ids]
    ).sum(dim=-1)
    return torch.exp(-error / temperature)


def motion_joint_velocity_error_deadband_huber(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float,
    delta: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected reference-velocity errors outside a robust deadband."""

    if deadband < 0.0:
        raise ValueError(f"deadband must be nonnegative, got {deadband}")
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    error = torch.abs(
        command.robot_joint_vel[:, asset_cfg.joint_ids]
        - command.joint_vel[:, asset_cfg.joint_ids]
    )
    error = torch.clamp_min(error - deadband, 0.0)
    loss = torch.where(
        error <= delta,
        0.5 * torch.square(error),
        delta * (error - 0.5 * delta),
    )
    return loss.mean(dim=-1)


class action_second_difference_l2(ManagerTermBase):
    """Penalize selected raw-action curvature after two valid episode steps."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_action = torch.zeros_like(env.action_manager.action)
        self._previous_previous_action = torch.zeros_like(env.action_manager.action)
        self._history_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_action[env_ids] = 0.0
        self._previous_previous_action[env_ids] = 0.0
        self._history_count[env_ids] = 0

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        action = env.action_manager.action
        second_difference = (
            action - 2.0 * self._previous_action + self._previous_previous_action
        )
        loss = torch.sum(
            torch.square(second_difference[:, asset_cfg.joint_ids]), dim=-1
        )
        loss = loss * (self._history_count >= 2).to(loss.dtype)

        self._previous_previous_action.copy_(self._previous_action)
        self._previous_action.copy_(action)
        self._history_count.add_(1)
        return loss


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def ladder_contact_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    limb_names: list[str],
    active_weight: float,
    temperature: float,
) -> torch.Tensor:
    """Track selected limb positions using the LadderMan contact reward."""

    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    limb_indexes = [command.limb_names.index(name) for name in limb_names]
    required = command.contact_indicator[:, limb_indexes]
    squared_error = torch.sum(
        torch.square(
            command.robot_limb_pos_w[:, limb_indexes] - command.target_contact_pos_w[:, limb_indexes]
        ),
        dim=-1,
    )
    reward = active_weight * required * torch.exp(-squared_error / temperature) + (1.0 - required)
    return reward.mean(dim=-1)


def ladder_phase_conditioned_limb_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    limb_names: list[str],
    reference_body_names: list[str],
    target_temperature: float,
    reference_temperature: float,
) -> torch.Tensor:
    """Track ladder targets in contact phases and reference limbs in swing phases."""

    if target_temperature <= 0.0:
        raise ValueError(
            f"target_temperature must be positive, got {target_temperature}"
        )
    if reference_temperature <= 0.0:
        raise ValueError(
            "reference_temperature must be positive, "
            f"got {reference_temperature}"
        )
    if len(reference_body_names) != len(limb_names):
        raise ValueError(
            "reference_body_names must contain one body for every limb: "
            f"got {len(reference_body_names)} bodies for {len(limb_names)} limbs"
        )

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    limb_indexes = [command.limb_names.index(name) for name in limb_names]
    try:
        body_indexes = [command.cfg.body_names.index(name) for name in reference_body_names]
    except ValueError as exc:
        raise ValueError(
            f"reference limb body is not tracked by the motion command: {reference_body_names}"
        ) from exc

    required = command.contact_indicator[:, limb_indexes]
    target_squared_error = torch.sum(
        torch.square(
            command.robot_limb_pos_w[:, limb_indexes]
            - command.target_contact_pos_w[:, limb_indexes]
        ),
        dim=-1,
    )
    reference_squared_error = torch.sum(
        torch.square(
            command.robot_body_pos_w[:, body_indexes]
            - command.body_pos_relative_w[:, body_indexes]
        ),
        dim=-1,
    )
    contact_reward = torch.exp(-target_squared_error / target_temperature)
    reference_reward = torch.exp(
        -reference_squared_error / reference_temperature
    )
    reward = required * contact_reward + (1.0 - required) * reference_reward
    return reward.mean(dim=-1)


def stable_support_phase_conditioned_limb_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    limb_names: list[str],
    reference_body_names: list[str],
    target_temperature: float,
    reference_temperature: float,
) -> torch.Tensor:
    """Track the ground-aware stable-support contract instead of legacy contacts."""

    if target_temperature <= 0.0:
        raise ValueError(
            f"target_temperature must be positive, got {target_temperature}"
        )
    if reference_temperature <= 0.0:
        raise ValueError(
            "reference_temperature must be positive, "
            f"got {reference_temperature}"
        )
    if len(reference_body_names) != len(limb_names):
        raise ValueError(
            "reference_body_names must contain one body for every limb: "
            f"got {len(reference_body_names)} bodies for {len(limb_names)} limbs"
        )

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    limb_indexes = [command.limb_names.index(name) for name in limb_names]
    try:
        body_indexes = [
            command.cfg.body_names.index(name) for name in reference_body_names
        ]
    except ValueError as exc:
        raise ValueError(
            f"reference limb body is not tracked by the motion command: {reference_body_names}"
        ) from exc

    required = command.stable_support_contact_indicator[:, limb_indexes]
    # This target property resolves the matching stable_support_rung_id:
    # -1 is inactive, 0 is ground, and 1..N select ladder rungs.
    target_squared_error = torch.sum(
        torch.square(
            command.robot_limb_pos_w[:, limb_indexes]
            - command.stable_support_target_pos_w[:, limb_indexes]
        ),
        dim=-1,
    )
    reference_squared_error = torch.sum(
        torch.square(
            command.robot_body_pos_w[:, body_indexes]
            - command.body_pos_relative_w[:, body_indexes]
        ),
        dim=-1,
    )
    contact_reward = torch.exp(-target_squared_error / target_temperature)
    reference_reward = torch.exp(
        -reference_squared_error / reference_temperature
    )
    reward = required * contact_reward + (1.0 - required) * reference_reward
    return reward.mean(dim=-1)


class next_stable_target_distance_progress(ManagerTermBase):
    """Reward signed distance reduction toward the pending stable-support target."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._previous_target_frame = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self._initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_distance[env_ids] = 0.0
        self._previous_target_frame[env_ids] = -1
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
    ) -> torch.Tensor:
        if env.step_dt <= 0.0:
            raise ValueError(f"environment step_dt must be positive, got {env.step_dt}")

        command: LadderMotionCommand = env.command_manager.get_term(command_name)
        valid_limb = command.next_stable_limb_target_valid
        distance = torch.linalg.vector_norm(
            command.robot_limb_pos_w - command.next_stable_limb_target_pos_w,
            dim=-1,
        )
        valid_count = valid_limb.sum(dim=-1)
        mean_distance = torch.where(
            valid_count > 0,
            (distance * valid_limb).sum(dim=-1)
            / valid_count.clamp_min(1).to(distance.dtype),
            torch.zeros_like(distance[:, 0]),
        )
        target_frame = command.next_stable_target_frame
        target_unchanged = target_frame == self._previous_target_frame
        can_measure = self._initialized & target_unchanged & (valid_count > 0)
        progress_rate = (self._previous_distance - mean_distance) / env.step_dt
        reward = torch.where(can_measure, progress_rate, torch.zeros_like(progress_rate))

        self._previous_distance.copy_(mean_distance)
        self._previous_target_frame.copy_(target_frame)
        self._initialized.copy_((target_frame >= 0) & (valid_count > 0))
        return reward


def stable_anchor_completion_bonus(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Return a control-rate-independent bonus when the stable gate releases."""

    if env.step_dt <= 0.0:
        raise ValueError(f"environment step_dt must be positive, got {env.step_dt}")
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.stable_anchor_completion_event.float() / env.step_dt


def _limb_contact_data(
    env: ManagerBasedRLEnv,
    command_name: str,
    limb_names: list[str],
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    limb_indexes = [command.limb_names.index(name) for name in limb_names]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    if forces_w.shape[1] != len(limb_indexes):
        raise ValueError(
            f"expected one contact body per limb ({len(limb_indexes)}), got {forces_w.shape[1]}"
        )
    required = command.contact_indicator[:, limb_indexes] > 0.5
    return required, forces_w


def limb_swing_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    limb_names: list[str],
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """Penalize selected limb contacts while their reference contact phase is inactive."""

    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be nonnegative, got {force_threshold}")

    required, forces_w = _limb_contact_data(env, command_name, limb_names, sensor_cfg)
    unwanted_contact = (~required) & (torch.linalg.vector_norm(forces_w, dim=-1) > force_threshold)
    return unwanted_contact.float().mean(dim=-1)


def limb_downward_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """Penalize downward world-Z contact force, a proxy for hitting a horizontal step underside."""

    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be nonnegative, got {force_threshold}")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    underside_hit = forces_z < -force_threshold
    return underside_hit.float().mean(dim=-1)


def limb_upward_contact_force_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    limb_names: list[str],
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    force_saturation: float,
    normalization_limb_names: list[str] | None = None,
    filtered_contacts_only: bool = False,
    target_rung_only: bool = False,
) -> torch.Tensor:
    """Reward bounded upward world-Z force for limbs in their required contact phase."""

    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be nonnegative, got {force_threshold}")
    if force_saturation <= force_threshold:
        raise ValueError(
            f"force_saturation ({force_saturation}) must exceed force_threshold ({force_threshold})"
        )
    if target_rung_only and not filtered_contacts_only:
        raise ValueError("target_rung_only requires filtered_contacts_only=True")

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    limb_indexes = [command.limb_names.index(name) for name in limb_names]
    required, forces_w = _limb_contact_data(env, command_name, limb_names, sensor_cfg)
    if filtered_contacts_only:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        force_matrix_w = contact_sensor.data.force_matrix_w
        if force_matrix_w is None:
            raise RuntimeError(
                f"contact sensor '{sensor_cfg.name}' must define filter_prim_paths_expr"
            )
        filtered_forces_w = force_matrix_w[:, sensor_cfg.body_ids, :, :]
        if filtered_forces_w.shape[1] != len(limb_names):
            raise ValueError(
                f"expected one filtered contact body per limb ({len(limb_names)}), "
                f"got {filtered_forces_w.shape[1]}"
            )
        if target_rung_only:
            rung_ids = command.contact_rung_id[:, limb_indexes]
            forces_w = target_rung_forces(filtered_forces_w, rung_ids)
        else:
            forces_w = filtered_forces_w.sum(dim=-2)
    upward_force = forces_w[..., 2]
    normalized_force = torch.clamp(
        (upward_force - force_threshold) / (force_saturation - force_threshold),
        min=0.0,
        max=1.0,
    )
    if normalization_limb_names is None:
        normalization_required = required
    else:
        normalization_indexes = [command.limb_names.index(name) for name in normalization_limb_names]
        normalization_required = command.contact_indicator[:, normalization_indexes] > 0.5
    required_count = normalization_required.sum(dim=-1).clamp(min=1)
    return (normalized_force * required).sum(dim=-1) / required_count


def filtered_contact_force_excess_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_soft: float,
    force_scale: float,
    surface: str = "all",
    min_top_normal_fraction: float = 0.5,
    max_excess: float | None = None,
    command_name: str | None = None,
    limb_name: str | None = None,
    rung_scope: str = "all",
) -> torch.Tensor:
    """Penalize only filtered contact force above a surface-specific soft limit."""

    if force_soft < 0.0:
        raise ValueError(f"force_soft must be nonnegative, got {force_soft}")
    if force_scale <= 0.0:
        raise ValueError(f"force_scale must be positive, got {force_scale}")
    if max_excess is not None and max_excess <= 0.0:
        raise ValueError(f"max_excess must be positive when set, got {max_excess}")
    if rung_scope not in ("all", "target", "non_target"):
        raise ValueError(
            f"rung_scope must be 'all', 'target', or 'non_target', got {rung_scope!r}"
        )
    if rung_scope != "all" and (command_name is None or limb_name is None):
        raise ValueError(
            "command_name and limb_name are required for target/non-target rung filtering"
        )

    forces_w = _filtered_contact_forces_w(env, sensor_cfg)
    if rung_scope != "all":
        if forces_w.shape[1] != 1:
            raise ValueError(
                f"rung-scoped force penalty expects one sensor body, got {forces_w.shape[1]}"
            )
        command: LadderMotionCommand = env.command_manager.get_term(command_name)
        limb_index = command.limb_names.index(limb_name)
        rung_ids = command.contact_rung_id[:, limb_index : limb_index + 1]
        if rung_scope == "target":
            forces_w = target_rung_forces(forces_w, rung_ids).unsqueeze(2)
            required = command.contact_indicator[:, limb_index] > 0.5
            forces_w = torch.where(
                required.view(-1, 1, 1, 1), forces_w, torch.zeros_like(forces_w)
            )
        else:
            forces_w = non_target_rung_forces(forces_w, rung_ids)
    force_max = filtered_surface_force_max(
        forces_w,
        surface=surface,
        min_top_normal_fraction=min_top_normal_fraction,
    )
    excess = torch.clamp_min((force_max - force_soft) / force_scale, 0.0)
    if max_excess is not None:
        excess = torch.clamp_max(excess, max_excess)
    return torch.square(excess)


def ladder_surface_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    limb_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    allow_target_contact: bool,
    target_distance_threshold: float = 0.08,
) -> torch.Tensor:
    """Penalize filtered ladder contacts outside the intended hand target.

    The sensor filters specific ladder boxes, so ``force_matrix_w`` separates
    these surfaces from ground and other robot contacts. Step-box contact is
    allowed only when the reference requires that hand and its center is near
    the target. Side-board contact passes ``allow_target_contact=False`` and
    is always penalized.
    """

    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be nonnegative, got {force_threshold}")
    if target_distance_threshold <= 0.0:
        raise ValueError(
            f"target_distance_threshold must be positive, got {target_distance_threshold}"
        )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_w = contact_sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise ValueError(
            f"contact sensor {sensor_cfg.name!r} must configure filter_prim_paths_expr "
            "to use ladder_surface_contact_penalty"
        )
    filtered_force_norm = torch.linalg.vector_norm(
        force_matrix_w[:, sensor_cfg.body_ids, :, :], dim=-1
    )
    ladder_contact = filtered_force_norm.amax(dim=-1).amax(dim=-1) > force_threshold

    if allow_target_contact:
        command: LadderMotionCommand = env.command_manager.get_term(command_name)
        limb_index = command.limb_names.index(limb_name)
        required = command.contact_indicator[:, limb_index] > 0.5
        target_error = torch.linalg.vector_norm(
            command.robot_limb_pos_w[:, limb_index] - command.target_contact_pos_w[:, limb_index], dim=-1
        )
        ladder_contact &= (~required) | (target_error > target_distance_threshold)

    return ladder_contact.float()


def _filtered_contact_forces_w(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_w = contact_sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise ValueError(
            f"contact sensor {sensor_cfg.name!r} must configure filter_prim_paths_expr"
        )
    return force_matrix_w[:, sensor_cfg.body_ids, :, :]


def filtered_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """Penalize any contact reported by a filtered one-to-many sensor."""

    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be nonnegative, got {force_threshold}")

    forces_w = _filtered_contact_forces_w(env, sensor_cfg)
    contact = torch.linalg.vector_norm(forces_w, dim=-1) > force_threshold
    return contact.any(dim=-1).any(dim=-1).float()


def whole_robot_rail_force_max(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    expected_rail_count: int,
) -> torch.Tensor:
    """Return per-rail maximum force across one-to-many sensors on robot bodies."""

    if not sensor_names:
        raise ValueError("sensor_names must not be empty")
    if expected_rail_count <= 0:
        raise ValueError(
            f"expected_rail_count must be positive, got {expected_rail_count}"
        )

    step = getattr(env, "common_step_counter", None)
    cache_key = (step, sensor_names, expected_rail_count)
    cache = getattr(env, "_whole_robot_rail_force_cache", None)
    if step is not None and cache is not None and cache[0] == cache_key:
        return cache[1]

    force_max = None
    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        forces_w = sensor.data.force_matrix_w
        if forces_w is None:
            raise RuntimeError(
                f"contact sensor {sensor_name!r} must define filter_prim_paths_expr"
            )
        if forces_w.shape[1] != 1 or forces_w.shape[2] != expected_rail_count:
            raise ValueError(
                f"contact sensor {sensor_name!r} must have shape "
                f"(env, 1, {expected_rail_count}, 3), got {tuple(forces_w.shape)}"
            )
        sensor_force_max = torch.linalg.vector_norm(forces_w[:, 0], dim=-1)
        force_max = (
            sensor_force_max
            if force_max is None
            else torch.maximum(force_max, sensor_force_max)
        )

    if step is not None:
        env._whole_robot_rail_force_cache = (cache_key, force_max)
    return force_max


def whole_robot_rail_binary_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    rail_index: int,
    force_threshold: float,
    expected_rail_count: int = 2,
) -> torch.Tensor:
    """Penalize any robot collision body contacting one selected ladder rail."""

    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be nonnegative, got {force_threshold}")
    if not 0 <= rail_index < expected_rail_count:
        raise ValueError(
            f"rail_index must be in [0, {expected_rail_count}), got {rail_index}"
        )
    force_max = whole_robot_rail_force_max(env, sensor_names, expected_rail_count)
    return (force_max[:, rail_index] > force_threshold).float()


def whole_robot_rail_force_excess_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    rail_index: int,
    force_soft: float,
    force_scale: float,
    max_excess: float,
    expected_rail_count: int = 2,
) -> torch.Tensor:
    """Penalize squared rail force above a soft limit for all robot collision bodies."""

    if force_soft < 0.0:
        raise ValueError(f"force_soft must be nonnegative, got {force_soft}")
    if force_scale <= 0.0:
        raise ValueError(f"force_scale must be positive, got {force_scale}")
    if max_excess <= 0.0:
        raise ValueError(f"max_excess must be positive, got {max_excess}")
    if not 0 <= rail_index < expected_rail_count:
        raise ValueError(
            f"rail_index must be in [0, {expected_rail_count}), got {rail_index}"
        )
    force_max = whole_robot_rail_force_max(env, sensor_names, expected_rail_count)
    excess = torch.clamp(
        (force_max[:, rail_index] - force_soft) / force_scale,
        min=0.0,
        max=max_excess,
    )
    return torch.square(excess)


def step_surface_orientation_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    min_top_normal_fraction: float,
    surface: str,
) -> torch.Tensor:
    """Penalize step front/edge or underside contact while allowing the horizontal top face.

    IsaacLab filtered contact forces are normal forces in world coordinates. The
    wood-step tops remain horizontal under ladder geometry randomization, so an
    upward normal is a top-face contact and may be allowed.
    """

    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be nonnegative, got {force_threshold}")
    if not 0.0 < min_top_normal_fraction <= 1.0:
        raise ValueError(
            "min_top_normal_fraction must be in (0, 1], "
            f"got {min_top_normal_fraction}"
        )
    if surface not in ("front", "underside"):
        raise ValueError(f"surface must be 'front' or 'underside', got {surface!r}")

    forces_w = _filtered_contact_forces_w(env, sensor_cfg)
    force_norm = torch.linalg.vector_norm(forces_w, dim=-1)
    normal_z_fraction = forces_w[..., 2] / force_norm.clamp_min(1.0e-6)
    contact = force_norm > force_threshold
    if surface == "front":
        forbidden = contact & (torch.abs(normal_z_fraction) < min_top_normal_fraction)
    else:
        forbidden = contact & (normal_z_fraction <= -min_top_normal_fraction)
    return forbidden.any(dim=-1).any(dim=-1).float()


def missing_required_step_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    limb_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    target_distance_threshold: float,
    grace_steps: int,
    target_rung_only: bool = False,
    require_near_target: bool = True,
) -> torch.Tensor:
    """Penalize a stable required hand target that has no upward step support force.

    ``target_rung_only`` prevents support on a non-indicated rung from satisfying
    the required-contact condition.
    """

    if force_threshold < 0.0:
        raise ValueError(f"force_threshold must be nonnegative, got {force_threshold}")
    if target_distance_threshold <= 0.0:
        raise ValueError(
            f"target_distance_threshold must be positive, got {target_distance_threshold}"
        )
    if grace_steps < 0:
        raise ValueError(f"grace_steps must be nonnegative, got {grace_steps}")

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    limb_index = command.limb_names.index(limb_name)
    stable_required = command.contact_indicator[:, limb_index] > 0.5
    if grace_steps > 0:
        stable_required &= command.time_steps >= grace_steps
        stable_required &= env.episode_length_buf > grace_steps
        for lag in range(1, grace_steps + 1):
            previous_steps = (command.time_steps - lag).clamp_min(0)
            stable_required &= command.motion.contact_indicator[previous_steps, limb_index] > 0.5

    target_error = torch.linalg.vector_norm(
        command.robot_limb_pos_w[:, limb_index]
        - command.target_contact_pos_w[:, limb_index],
        dim=-1,
    )
    near_target = target_error < target_distance_threshold
    forces_w = _filtered_contact_forces_w(env, sensor_cfg)
    if target_rung_only:
        if forces_w.shape[1] != 1:
            raise ValueError(
                f"target-rung missing-contact penalty expects one sensor body, got {forces_w.shape[1]}"
            )
        rung_ids = command.contact_rung_id[:, limb_index : limb_index + 1]
        forces_w = target_rung_forces(forces_w, rung_ids).unsqueeze(2)
    upward_force = torch.clamp(forces_w[..., 2], min=0.0).sum(dim=(-1, -2))
    missing_support = upward_force <= force_threshold
    missing_contact = stable_required & missing_support
    if require_near_target:
        missing_contact &= near_target
    return missing_contact.float()


def asymmetric_motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    lower_body_std: float,
    upper_body_std: float,
    lower_body_names: list[str],
    upper_body_names: list[str],
) -> torch.Tensor:
    """Lower-body strict / upper-body relaxed body-position tracking."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    lower_indexes = _get_body_indexes(command, lower_body_names)
    upper_indexes = _get_body_indexes(command, upper_body_names)
    lower_error = torch.sum(
        torch.square(command.body_pos_relative_w[:, lower_indexes] - command.robot_body_pos_w[:, lower_indexes]),
        dim=-1,
    ).mean(dim=-1)
    upper_error = torch.sum(
        torch.square(command.body_pos_relative_w[:, upper_indexes] - command.robot_body_pos_w[:, upper_indexes]),
        dim=-1,
    ).mean(dim=-1)
    return 0.5 * torch.exp(-lower_error / lower_body_std**2) + 0.5 * torch.exp(-upper_error / upper_body_std**2)
