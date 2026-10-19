from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms

from ladder.tasks.ladder_climb.mdp.commands import LadderMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


def motion_complete(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Time out environments after they execute the final reference frame."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    if not command.cfg.advance_reference:
        return torch.zeros_like(command.time_steps, dtype=torch.bool)
    holding = getattr(
        command,
        "stop_holding",
        torch.zeros_like(command.time_steps, dtype=torch.bool),
    )
    return (
        (env.episode_length_buf > 0)
        & (command.time_steps >= command.motion.time_step_total - 1)
        & ~holding
    )


def stable_anchor_timeout(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate after a gated stable anchor remains incomplete past its deadline."""

    command: LadderMotionCommand = env.command_manager.get_term(command_name)
    return command.stable_anchor_timeout_failed


class _ConsecutiveFailure(ManagerTermBase):
    """Accumulate a failure condition once per control step for each environment."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._failure_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._last_episode_step = torch.full_like(self._failure_steps, -1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._failure_steps[env_ids] = 0
        self._last_episode_step[env_ids] = -1

    def _update(self, failed: torch.Tensor, consecutive_steps: int) -> torch.Tensor:
        if consecutive_steps <= 0:
            raise ValueError(f"consecutive_steps must be positive, got {consecutive_steps}")

        episode_step = self._env.episode_length_buf
        should_update = (episode_step > 0) & (episode_step != self._last_episode_step)
        self._failure_steps = torch.where(
            should_update,
            torch.where(failed, self._failure_steps + 1, 0),
            self._failure_steps,
        )
        self._last_episode_step = torch.where(should_update, episode_step, self._last_episode_step)
        return self._failure_steps >= consecutive_steps


class bad_root_pose_consecutive(_ConsecutiveFailure):
    """Terminate after sustained global root position or orientation tracking failure."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        position_threshold: float,
        orientation_threshold: float,
        consecutive_steps: int,
    ) -> torch.Tensor:
        command: LadderMotionCommand = env.command_manager.get_term(command_name)
        position_error = torch.linalg.norm(command.robot_body_pos_w[:, 0] - command.body_pos_w[:, 0], dim=-1)
        orientation_error = quat_error_magnitude(command.robot_body_quat_w[:, 0], command.body_quat_w[:, 0])
        failed = (position_error > position_threshold) | (orientation_error > orientation_threshold)
        return self._update(failed, consecutive_steps)


class bad_body_local_pose_consecutive(_ConsecutiveFailure):
    """Terminate after sustained tracked-body local pose failure."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        position_threshold: float,
        orientation_threshold: float,
        consecutive_steps: int,
    ) -> torch.Tensor:
        command: LadderMotionCommand = env.command_manager.get_term(command_name)
        num_bodies = len(command.cfg.body_names)
        reference_pos_b, reference_quat_b = subtract_frame_transforms(
            command.body_pos_w[:, :1, :].repeat(1, num_bodies, 1),
            command.body_quat_w[:, :1, :].repeat(1, num_bodies, 1),
            command.body_pos_w,
            command.body_quat_w,
        )
        robot_pos_b, robot_quat_b = subtract_frame_transforms(
            command.robot_body_pos_w[:, :1, :].repeat(1, num_bodies, 1),
            command.robot_body_quat_w[:, :1, :].repeat(1, num_bodies, 1),
            command.robot_body_pos_w,
            command.robot_body_quat_w,
        )
        position_error = torch.linalg.norm(robot_pos_b - reference_pos_b, dim=-1)
        orientation_error = quat_error_magnitude(robot_quat_b, reference_quat_b)
        failed = torch.any(
            (position_error > position_threshold) | (orientation_error > orientation_threshold), dim=-1
        )
        return self._update(failed, consecutive_steps)


class bad_required_contact_consecutive(_ConsecutiveFailure):
    """Terminate after a required limb remains too far from its rung target."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        position_threshold: float,
        consecutive_steps: int,
    ) -> torch.Tensor:
        command: LadderMotionCommand = env.command_manager.get_term(command_name)
        required = command.contact_indicator > 0.5
        position_error = torch.linalg.norm(command.robot_limb_pos_w - command.target_contact_pos_w, dim=-1)
        failed = torch.any(required & (position_error > position_threshold), dim=-1)
        return self._update(failed, consecutive_steps)
