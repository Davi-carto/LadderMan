"""Ladder-specific action terms.

The reference policy outputs joint-position targets.  This term keeps the target
trajectory physically executable by applying a per-joint target rate limit before
the target is sent to Isaac Lab's articulation drive.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _resolve_joint_values(
    value: float | dict[str, float], joint_names: list[str], device: torch.device
) -> torch.Tensor:
    if isinstance(value, (float, int)):
        return torch.full((len(joint_names),), float(value), dtype=torch.float32, device=device)
    if not isinstance(value, dict):
        raise ValueError(f"joint value must be a float or dict, got {type(value)}")
    indexes, _, values = string_utils.resolve_matching_names_values(value, joint_names)
    result = torch.zeros(len(joint_names), dtype=torch.float32, device=device)
    result[indexes] = torch.tensor(values, dtype=torch.float32, device=device)
    return result


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Joint position targets with a hardware-oriented target rate limit."""

    class_type: type = JointPositionAction
    max_target_velocity: float | dict[str, float] = 6.0
    """Maximum target change in rad/s, per joint."""

    min_control_delay: int = 0
    """Minimum delay in physics steps, sampled per environment at reset."""

    max_control_delay: int = 0
    """Maximum delay in physics steps; the sampled lag stays fixed until reset."""

    def __post_init__(self):
        super().__post_init__()
        self.class_type = RateLimitedJointPositionAction


class RateLimitedJointPositionAction(JointPositionAction):
    """Rate-limit joint position targets before applying them."""

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._step_dt = float(env.step_dt)
        if self._step_dt <= 0.0:
            raise ValueError(f"environment step_dt must be positive, got {self._step_dt}")
        max_velocity = _resolve_joint_values(cfg.max_target_velocity, self._joint_names, self.device)
        if torch.any(max_velocity <= 0.0):
            raise ValueError("max_target_velocity must be positive for every joint")
        if cfg.min_control_delay < 0 or cfg.max_control_delay < cfg.min_control_delay:
            raise ValueError("Control delay must satisfy 0 <= min_control_delay <= max_control_delay")

        self._max_target_delta = max_velocity.unsqueeze(0) * self._step_dt
        self._last_target = torch.zeros_like(self.raw_actions)
        self._reset_target = torch.zeros_like(self.raw_actions)
        self._physics_steps_since_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._min_control_delay = int(cfg.min_control_delay)
        self._max_control_delay = int(cfg.max_control_delay)
        self._action_delay_buffer = (
            DelayBuffer(
                history_length=self._max_control_delay,
                batch_size=self.num_envs,
                device=self.device,
            )
            if self._max_control_delay > 0
            else None
        )

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        current_position = self._asset.data.joint_pos[:, self._joint_ids]
        uninitialized = ~self._initialized
        if torch.any(uninitialized):
            # The command term writes the sampled reset pose before the first policy
            # action.  Starting from measured position prevents a reset kick.
            self._last_target[uninitialized] = current_position[uninitialized]
            self._reset_target[uninitialized] = current_position[uninitialized]
            self._initialized[uninitialized] = True

        desired = self._processed_actions
        delta = torch.clamp(
            desired - self._last_target,
            min=-self._max_target_delta,
            max=self._max_target_delta,
        )
        self._processed_actions = self._last_target + delta
        self._last_target.copy_(self._processed_actions)

    def apply_actions(self):
        # IsaacLab calls this once per physics substep. Keep target rate limiting
        # in process_actions() at policy rate; only command delivery is delayed.
        target = self._processed_actions
        if self._action_delay_buffer is not None:
            target = self._action_delay_buffer.compute(target)
            # DelayBuffer fills an empty history with its first input. Hold the
            # sampled reset pose until the first command has actually arrived.
            waiting = self._physics_steps_since_reset < self._action_delay_buffer.time_lags
            target = torch.where(waiting.unsqueeze(-1), self._reset_target, target)
            self._physics_steps_since_reset += 1
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        selected_env_ids: Sequence[int] | slice
        if env_ids is None:
            selected_env_ids = slice(None)
            selected_count = self.num_envs
        else:
            selected_env_ids = env_ids
            selected_count = len(env_ids)
        self._initialized[selected_env_ids] = False
        self._physics_steps_since_reset[selected_env_ids] = 0
        if self._action_delay_buffer is not None:
            sampled_delay = torch.randint(
                self._min_control_delay,
                self._max_control_delay + 1,
                (selected_count,),
                dtype=torch.int32,
                device=self.device,
            )
            self._action_delay_buffer.set_time_lag(sampled_delay, selected_env_ids)
            self._action_delay_buffer.reset(selected_env_ids)


RateLimitedJointPositionActionCfg.class_type = RateLimitedJointPositionAction


__all__ = ["RateLimitedJointPositionAction", "RateLimitedJointPositionActionCfg"]
