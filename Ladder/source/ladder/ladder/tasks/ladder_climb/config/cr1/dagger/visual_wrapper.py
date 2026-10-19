"""TensorDict adapter for visual CR1 DAgger training."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv

from ladder.rsl_rl.storage.transformer_vision_ppo_storage import (
    CRITIC_KEY,
    HEIGHTMAP_HIT_CLASS_KEY,
    HEIGHTMAP_KEY,
    HEIGHTMAP_VALID_KEY,
    VELOCITY_KEY,
)

from .student_teacher import TEACHER_KEY
from .visual_model import DEPTH_KEY, PROPRIO_KEY


def as_visual_dagger_tensordict(
    observations: dict | TensorDict,
    *,
    num_envs: int,
    device: str | torch.device,
    actor_only: bool = False,
) -> TensorDict:
    if isinstance(observations, TensorDict):
        observations = observations.to_dict()
    inputs = observations["transformer_vision_inputs"]
    actor = {
        PROPRIO_KEY: inputs["proprio_history"],
        DEPTH_KEY: inputs["depth_history"],
    }
    if actor_only:
        return TensorDict(actor, batch_size=[num_envs], device=device)
    targets = observations["transformer_vision_targets"]
    auxiliary = observations["transformer_vision_auxiliary_targets"]
    return TensorDict(
        {
            **actor,
            TEACHER_KEY: observations[TEACHER_KEY],
            CRITIC_KEY: observations[CRITIC_KEY],
            HEIGHTMAP_KEY: targets["relative_height"],
            HEIGHTMAP_VALID_KEY: targets["hit_valid"],
            HEIGHTMAP_HIT_CLASS_KEY: targets["hit_class"],
            VELOCITY_KEY: auxiliary["base_lin_vel"],
        },
        batch_size=[num_envs],
        device=device,
    )


class CR1DaggerVisualVecEnvWrapper(VecEnv):
    def __init__(self, env, clip_actions: float | None = None) -> None:
        self.env = env
        self.clip_actions = clip_actions
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
    def episode_length_buf(self):
        return self.env.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.unwrapped.episode_length_buf = value

    def _convert(self, observations):
        return as_visual_dagger_tensordict(
            observations,
            num_envs=self.num_envs,
            device=self.device,
            actor_only=self.actor_only,
        )

    def get_observations(self):
        return self._convert(self.env.unwrapped.observation_manager.compute())

    def reset(self):
        observations, _ = self.env.reset()
        return self._convert(observations)

    def step(self, actions):
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.bool)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self._convert(observations), rewards, dones, extras

    def close(self):
        self.env.close()


__all__ = ["CR1DaggerVisualVecEnvWrapper", "as_visual_dagger_tensordict"]
