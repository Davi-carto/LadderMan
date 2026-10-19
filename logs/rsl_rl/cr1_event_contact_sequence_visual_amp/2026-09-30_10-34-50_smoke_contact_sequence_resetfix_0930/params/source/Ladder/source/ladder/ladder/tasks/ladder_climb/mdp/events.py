from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.actuators import ImplicitActuator
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_from_euler_xyz,
    sample_uniform,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_camera_extrinsics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    camera_cfg: SceneEntityCfg,
    parent_cfg: SceneEntityCfg,
    translation_range: tuple[float, float],
    rotation_range: tuple[float, float],
) -> None:
    """Randomize a camera pose around its parent body frame."""

    camera = env.scene.sensors[camera_cfg.name]
    parent: Articulation = env.scene[parent_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=parent.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=parent.device)
    if not isinstance(parent_cfg.body_ids, list) or len(parent_cfg.body_ids) != 1:
        raise ValueError("Camera parent_cfg must resolve exactly one articulation body")

    count = len(env_ids)
    translation_bounds = torch.tensor(translation_range, device=parent.device)
    rotation_bounds = torch.tensor(rotation_range, device=parent.device)
    local_pos = sample_uniform(
        translation_bounds[0], translation_bounds[1], (count, 3), device=parent.device
    )
    local_rpy = sample_uniform(
        rotation_bounds[0], rotation_bounds[1], (count, 3), device=parent.device
    )
    local_quat = quat_from_euler_xyz(local_rpy[:, 0], local_rpy[:, 1], local_rpy[:, 2])

    body_id = parent_cfg.body_ids[0]
    camera_pos_w, camera_quat_w = combine_frame_transforms(
        parent.data.body_pos_w[env_ids, body_id],
        parent.data.body_quat_w[env_ids, body_id],
        local_pos,
        local_quat,
    )
    camera.set_world_poses(camera_pos_w, camera_quat_w, env_ids=env_ids, convention="world")


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Randomize the joint-position calibration offset and matching action offset."""

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if asset_cfg.joint_ids == slice(None):
        joint_ids = torch.arange(asset.num_joints, device=asset.device)
    else:
        joint_ids = torch.as_tensor(asset_cfg.joint_ids, dtype=torch.long, device=asset.device)

    default_pos = _randomize_prop_by_op(
        asset.data.default_joint_pos.clone(),
        pos_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution=distribution,
    )
    randomized_pos = default_pos[env_ids[:, None], joint_ids]
    asset.data.default_joint_pos[env_ids[:, None], joint_ids] = randomized_pos
    env.action_manager.get_term("joint_pos")._offset[env_ids[:, None], joint_ids] = randomized_pos


def randomize_actuator_strength_and_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    motor_strength_range: tuple[float, float],
    stiffness_factor_range: tuple[float, float],
    damping_factor_range: tuple[float, float],
) -> None:
    """Apply the X2 motor-strength, Kp-factor, and Kd-factor randomization.

    X2 multiplies the complete PD torque by one motor-strength factor and also
    multiplies the P and D terms by independent gain factors.  For an implicit
    PD actuator, the equivalent effective gains are::

        stiffness = default_stiffness * motor_strength * stiffness_factor
        damping = default_damping * motor_strength * damping_factor

    The same per-environment, per-joint ``motor_strength`` sample is therefore
    deliberately shared between stiffness and damping.
    """

    for name, value_range in (
        ("motor_strength_range", motor_strength_range),
        ("stiffness_factor_range", stiffness_factor_range),
        ("damping_factor_range", damping_factor_range),
    ):
        if len(value_range) != 2 or value_range[0] <= 0.0 or value_range[1] < value_range[0]:
            raise ValueError(f"{name} must be a positive ordered pair, got {value_range}")

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device)
    if env_ids.numel() == 0:
        return

    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            else:
                global_indices = torch.as_tensor(
                    actuator.joint_indices, dtype=torch.long, device=asset.device
                )
        elif isinstance(actuator.joint_indices, slice):
            actuator_indices = global_indices = torch.as_tensor(
                asset_cfg.joint_ids, dtype=torch.long, device=asset.device
            )
        else:
            actuator_joint_indices = torch.as_tensor(
                actuator.joint_indices, dtype=torch.long, device=asset.device
            )
            requested_joint_indices = torch.as_tensor(
                asset_cfg.joint_ids, dtype=torch.long, device=asset.device
            )
            actuator_indices = torch.nonzero(
                torch.isin(actuator_joint_indices, requested_joint_indices)
            ).view(-1)
            if actuator_indices.numel() == 0:
                continue
            global_indices = actuator_joint_indices[actuator_indices]

        default_stiffness = asset.data.default_joint_stiffness[env_ids][:, global_indices]
        default_damping = asset.data.default_joint_damping[env_ids][:, global_indices]
        sample_shape = default_stiffness.shape
        motor_strength = sample_uniform(
            motor_strength_range[0],
            motor_strength_range[1],
            sample_shape,
            device=asset.device,
        )
        stiffness_factor = sample_uniform(
            stiffness_factor_range[0],
            stiffness_factor_range[1],
            sample_shape,
            device=asset.device,
        )
        damping_factor = sample_uniform(
            damping_factor_range[0],
            damping_factor_range[1],
            sample_shape,
            device=asset.device,
        )

        stiffness = actuator.stiffness[env_ids].clone()
        damping = actuator.damping[env_ids].clone()
        stiffness[:, actuator_indices] = default_stiffness * motor_strength * stiffness_factor
        damping[:, actuator_indices] = default_damping * motor_strength * damping_factor
        actuator.stiffness[env_ids] = stiffness
        actuator.damping[env_ids] = damping
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim(
                stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids
            )
            asset.write_joint_damping_to_sim(
                damping, joint_ids=actuator.joint_indices, env_ids=env_ids
            )


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
) -> None:
    """Add a uniformly sampled center-of-mass offset to selected bodies."""

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    else:
        body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")

    ranges = torch.tensor([com_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")], device="cpu")
    offsets = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids[:, None], body_ids, :3] += offsets[:, None, :]
    asset.root_physx_view.set_coms(coms, env_ids)
