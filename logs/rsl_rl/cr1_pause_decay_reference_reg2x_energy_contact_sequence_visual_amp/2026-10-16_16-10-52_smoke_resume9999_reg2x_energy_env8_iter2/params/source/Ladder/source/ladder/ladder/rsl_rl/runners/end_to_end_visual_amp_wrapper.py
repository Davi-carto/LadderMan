"""AMP adapter for the recurrent end-to-end visual PPO environment."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from ladder.rsl_rl.amp import (
    AMP_STYLE_OBSERVATION_DIM,
    LadderAMPReferenceDataset,
)
from ladder.rsl_rl.runners.end_to_end_visual_wrapper import (
    EndToEndVisualPPOVecEnvWrapper,
    _as_end_to_end_visual_tensordict,
)


AMP_STYLE_KEY = "amp_style_observation"


def _as_end_to_end_visual_amp_tensordict(
    observations: dict | TensorDict,
    *,
    num_envs: int,
    device: str | torch.device,
    clip_observations: float | None = None,
    actor_only: bool = False,
) -> TensorDict:
    result = _as_end_to_end_visual_tensordict(
        observations,
        num_envs=num_envs,
        device=device,
        clip_observations=clip_observations,
        actor_only=actor_only,
    )
    if actor_only:
        return result
    amp_style = observations["amp_style"]
    if isinstance(amp_style, dict):
        amp_style = amp_style["state"]
    expected = (num_envs, AMP_STYLE_OBSERVATION_DIM)
    if tuple(amp_style.shape) != expected:
        raise ValueError(
            f"AMP style observation must have shape {expected}, got {tuple(amp_style.shape)}"
        )
    result[AMP_STYLE_KEY] = amp_style
    return result


class EndToEndVisualAMPPPOVecEnvWrapper(EndToEndVisualPPOVecEnvWrapper):
    """Expose policy AMP states and a phase-free expert transition sampler."""

    def __init__(
        self,
        env,
        clip_actions: float | None = None,
        clip_observations: float | None = None,
        *,
        anchor_hold_probability: float = 0.20,
        balance_directions: bool = True,
        num_frames: int = 5,
        history_stride: int = 2,
    ) -> None:
        super().__init__(env, clip_actions, clip_observations)
        command = self.unwrapped.command_manager.get_term("motion")
        self.amp_reference_dataset = LadderAMPReferenceDataset(
            command,
            anchor_hold_probability=anchor_hold_probability,
            balance_directions=balance_directions,
            num_frames=num_frames,
            history_stride=history_stride,
        )

    def _convert(self, observations) -> TensorDict:
        return _as_end_to_end_visual_amp_tensordict(
            observations,
            num_envs=self.num_envs,
            device=self.device,
            clip_observations=self.clip_observations,
            actor_only=self.actor_only,
        )

    def sample_amp_expert_transitions(self, batch_size: int) -> torch.Tensor:
        return self.amp_reference_dataset.sample(batch_size)


__all__ = [
    "AMP_STYLE_KEY",
    "EndToEndVisualAMPPPOVecEnvWrapper",
    "_as_end_to_end_visual_amp_tensordict",
]
