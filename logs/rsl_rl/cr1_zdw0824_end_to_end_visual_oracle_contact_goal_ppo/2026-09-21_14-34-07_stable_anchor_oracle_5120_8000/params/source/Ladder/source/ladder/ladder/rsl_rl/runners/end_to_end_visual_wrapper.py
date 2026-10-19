"""End-to-end visual PPO observations, with reconstruction labels kept separate."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from ladder.rsl_rl.modules.end_to_end_visual_actor import ORACLE_CONTACT_GOAL_KEY

from ladder.rsl_rl.runners.transformer_vision_ppo_wrapper import (
    TransformerVisionPPOVecEnvWrapper,
    _as_transformer_vision_ppo_tensordict,
)
from ladder.rsl_rl.storage.transformer_vision_ppo_storage import (
    CRITIC_KEY,
    HEIGHTMAP_HIT_CLASS_KEY,
    HEIGHTMAP_KEY,
    HEIGHTMAP_VALID_KEY,
    VELOCITY_KEY,
)


def _as_end_to_end_visual_tensordict(
    observations: dict | TensorDict,
    *,
    num_envs: int,
    device: str | torch.device,
    clip_observations: float | None = None,
    actor_only: bool = False,
) -> TensorDict:
    # Reuse only the deployable-input conversion. No future/reference labels
    # are required by this task, including during training.
    result = _as_transformer_vision_ppo_tensordict(
        observations,
        num_envs=num_envs,
        device=device,
        clip_observations=clip_observations,
        actor_only=True,
    )
    if ORACLE_CONTACT_GOAL_KEY in observations:
        oracle = observations[ORACLE_CONTACT_GOAL_KEY]
        fields = (
            ("next_stable_goal_b", 12),
            ("actual_limb_pos_b", 12),
            ("actual_support_contact", 4),
            ("desired_contact", 4),
        )
        for name, width in fields:
            if tuple(oracle[name].shape) != (num_envs, width):
                raise ValueError(f"oracle {name} must have shape {(num_envs, width)}")
        result[ORACLE_CONTACT_GOAL_KEY] = torch.cat(
            tuple(oracle[name].to(oracle["next_stable_goal_b"].dtype) for name, _ in fields),
            dim=-1,
        )
    if actor_only:
        return result
    targets = observations["transformer_vision_targets"]
    auxiliary = observations["transformer_vision_auxiliary_targets"]
    critic = observations["critic"]
    if clip_observations is not None:
        critic = critic.clamp(-clip_observations, clip_observations)
    result.update({
        CRITIC_KEY: critic,
        HEIGHTMAP_KEY: targets["relative_height"],
        HEIGHTMAP_VALID_KEY: targets["hit_valid"],
        HEIGHTMAP_HIT_CLASS_KEY: targets["hit_class"],
        VELOCITY_KEY: auxiliary["base_lin_vel"],
    })
    return result


class EndToEndVisualPPOVecEnvWrapper(TransformerVisionPPOVecEnvWrapper):
    """Use the standard wrapper lifecycle without next-frame supervision."""

    def _convert(self, observations) -> TensorDict:
        return _as_end_to_end_visual_tensordict(
            observations,
            num_envs=self.num_envs,
            device=self.device,
            clip_observations=self.clip_observations,
            actor_only=self.actor_only,
        )

    def get_observations(self) -> TensorDict:
        if hasattr(self.unwrapped, "observation_manager"):
            observations = self.unwrapped.observation_manager.compute()
        else:
            observations = self.unwrapped._get_observations()
        return self._convert(observations)

    def reset(self) -> TensorDict:
        observations, _ = self.env.reset()
        return self._convert(observations)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).bool()
        if not self.cfg.is_finite_horizon:
            # A real failure on the timeout step must not receive a bootstrap.
            extras["time_outs"] = truncated & ~terminated
        return self._convert(observations), rewards, dones, extras
