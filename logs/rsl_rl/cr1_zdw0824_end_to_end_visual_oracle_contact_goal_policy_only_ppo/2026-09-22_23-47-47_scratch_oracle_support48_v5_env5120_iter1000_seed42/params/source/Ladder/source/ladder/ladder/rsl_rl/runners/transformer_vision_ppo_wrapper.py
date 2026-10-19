"""Vector-environment adapter for TransformerVision asymmetric PPO."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from ladder.rsl_rl.modules.transformer_vision_actor import (
    CURRENT_PROPRIO_KEY,
    DEPTH_HISTORY_KEY,
    PROPRIO_HISTORY_KEY,
)
from ladder.rsl_rl.storage.transformer_vision_ppo_storage import (
    CONTACT_INDICATOR_KEY,
    CRITIC_KEY,
    HEIGHTMAP_HIT_CLASS_KEY,
    HEIGHTMAP_KEY,
    HEIGHTMAP_VALID_KEY,
    LIMB_GOAL_KEY,
    LIMB_GOAL_VALID_KEY,
    MOTION_JOINT_POS_KEY,
    MOTION_ROOT_ORIENTATION_KEY,
    MOTION_TARGET_VALID_KEY,
    NEXT_PROPRIO_KEY,
    VELOCITY_KEY,
)


def _as_transformer_vision_ppo_tensordict(
    observations: dict | TensorDict,
    *,
    num_envs: int,
    device: str | torch.device,
    clip_observations: float | None = None,
    actor_only: bool = False,
) -> TensorDict:
    if clip_observations is not None and clip_observations <= 0.0:
        raise ValueError("clip_observations must be positive or None")
    if isinstance(observations, TensorDict):
        observations = observations.to_dict()
    inputs = observations["transformer_vision_inputs"]
    proprio_history = inputs["proprio_history"]
    depth_history = inputs["depth_history"]
    if clip_observations is not None:
        proprio_history = torch.clamp(proprio_history, -clip_observations, clip_observations)
        depth_history = torch.clamp(depth_history, -clip_observations, clip_observations)
    actor_observations = {
        CURRENT_PROPRIO_KEY: proprio_history[:, -1],
        PROPRIO_HISTORY_KEY: proprio_history,
        DEPTH_HISTORY_KEY: depth_history,
    }
    if actor_only:
        return TensorDict(
            actor_observations,
            batch_size=[num_envs],
            device=device,
        )

    targets = observations["transformer_vision_targets"]
    auxiliary = observations["transformer_vision_auxiliary_targets"]
    critic = observations["critic"]
    if clip_observations is not None:
        critic = torch.clamp(critic, -clip_observations, clip_observations)
    privileged_observations = {
        HEIGHTMAP_KEY: targets["relative_height"],
        HEIGHTMAP_VALID_KEY: targets["hit_valid"],
        HEIGHTMAP_HIT_CLASS_KEY: targets["hit_class"],
        VELOCITY_KEY: auxiliary["base_lin_vel"],
        NEXT_PROPRIO_KEY: auxiliary["next_proprio"],
        CRITIC_KEY: critic,
    }
    task_names = (
        "task_limb_goal_b",
        "task_limb_goal_valid",
        "task_contact_indicator",
        "task_motion_joint_pos",
        "task_motion_root_orientation_6d",
        "task_motion_valid",
    )
    available_task_names = tuple(name for name in task_names if name in auxiliary)
    if available_task_names and available_task_names != task_names:
        raise KeyError(
            "TransformerVision task-goal labels must be configured together; found "
            f"{available_task_names}"
        )
    if available_task_names:
        privileged_observations.update(
            {
                LIMB_GOAL_KEY: auxiliary["task_limb_goal_b"].reshape(num_envs, 4, 3),
                LIMB_GOAL_VALID_KEY: auxiliary["task_limb_goal_valid"],
                CONTACT_INDICATOR_KEY: auxiliary["task_contact_indicator"],
                MOTION_JOINT_POS_KEY: auxiliary["task_motion_joint_pos"],
                MOTION_ROOT_ORIENTATION_KEY: auxiliary[
                    "task_motion_root_orientation_6d"
                ],
                MOTION_TARGET_VALID_KEY: auxiliary["task_motion_valid"].reshape(
                    num_envs
                ),
            }
        )
    return TensorDict(
        {**actor_observations, **privileged_observations},
        batch_size=[num_envs],
        device=device,
    )


class TransformerVisionPPOVecEnvWrapper(VecEnv):
    """Expose deployable actor inputs and privileged labels as named fields."""

    def __init__(
        self,
        env,
        clip_actions: float | None = None,
        clip_observations: float | None = None,
    ) -> None:
        if clip_actions is not None and clip_actions <= 0.0:
            raise ValueError("clip_actions must be positive or None")
        if clip_observations is not None and clip_observations <= 0.0:
            raise ValueError("clip_observations must be positive or None")
        self.env = env
        self.clip_actions = clip_actions
        self.clip_observations = clip_observations
        self.actor_only = False
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.max_episode_length = env.unwrapped.max_episode_length
        self.num_actions = env.unwrapped.action_manager.total_action_dim
        self.env.reset()

    @property
    def cfg(self):
        return self.env.unwrapped.cfg

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.unwrapped.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        if hasattr(self.env.unwrapped, "observation_manager"):
            observations = self.env.unwrapped.observation_manager.compute()
        else:
            observations = self.env.unwrapped._get_observations()
        return _as_transformer_vision_ppo_tensordict(
            observations,
            num_envs=self.num_envs,
            device=self.device,
            clip_observations=self.clip_observations,
            actor_only=self.actor_only,
        )

    def reset(self) -> TensorDict:
        observations, _ = self.env.reset()
        return _as_transformer_vision_ppo_tensordict(
            observations,
            num_envs=self.num_envs,
            device=self.device,
            clip_observations=self.clip_observations,
            actor_only=self.actor_only,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.bool)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return (
            _as_transformer_vision_ppo_tensordict(
                observations,
                num_envs=self.num_envs,
                device=self.device,
                clip_observations=self.clip_observations,
                actor_only=self.actor_only,
            ),
            rewards,
            dones,
            extras,
        )

    def close(self) -> None:
        self.env.close()


__all__ = ["TransformerVisionPPOVecEnvWrapper"]
