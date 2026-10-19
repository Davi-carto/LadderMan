"""Vector-environment adapter for the named CR1 SSR observation groups."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv


class LadderSSRVecEnvWrapper(VecEnv):
    """Keep all named observations instead of requiring RSL-RL's ``policy`` group."""

    def __init__(self, env, clip_actions: float | None = None) -> None:
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.max_episode_length = env.unwrapped.max_episode_length
        self.num_actions = env.unwrapped.action_manager.total_action_dim
        self.env.reset()

    @property
    def cfg(self):
        return self.env.unwrapped.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.unwrapped.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        return self._as_tensordict(self._compute_observations())

    def reset(self) -> TensorDict:
        observations, _ = self.env.reset()
        return self._as_tensordict(observations)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self._as_tensordict(observations), rewards, dones, extras

    def close(self) -> None:
        self.env.close()

    def _compute_observations(self):
        if hasattr(self.env.unwrapped, "observation_manager"):
            return self.env.unwrapped.observation_manager.compute()
        return self.env.unwrapped._get_observations()

    def _as_tensordict(self, observations) -> TensorDict:
        if isinstance(observations, TensorDict):
            observations = observations.to_dict()
        inputs = observations["visual_encoder_inputs"]
        targets = observations["visual_targets"]
        auxiliary = observations["visual_aux_targets"]
        return TensorDict(
            {
                "ssr_proprio_history": inputs["proprio_history"],
                "ssr_depth_history": inputs["depth_history"],
                "ssr_limb_heightmaps": targets["relative_height"],
                "ssr_heightmap_valid": targets["hit_valid"],
                "ssr_base_lin_vel": auxiliary["base_lin_vel"],
                "ssr_next_proprio": auxiliary["next_proprio"],
                "critic": observations["critic"],
            },
            batch_size=[self.num_envs],
            device=self.device,
        )
