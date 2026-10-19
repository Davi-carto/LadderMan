"""Recurrent DAgger sequences with on-policy targets for student envs only."""

from __future__ import annotations

from typing import Iterator

import torch

from .recurrent_visual_mse_storage import RecurrentVisualMseDaggerStorage


class RecurrentVisualPpoDaggerStorage(RecurrentVisualMseDaggerStorage):
    """Keep teacher samples for imitation and exclude them from PPO targets."""

    _RL_FIELDS = (
        "critic_observations",
        "actions",
        "old_action_mean",
        "old_action_std",
        "old_action_log_prob",
        "values",
        "rewards",
        "time_outs",
        "returns",
        "advantages",
    )

    def __init__(
        self, num_steps, num_envs, action_dim, critic_observation_dim, device
    ) -> None:
        super().__init__(num_steps, num_envs, action_dim, device)
        shape = (self.num_steps, self.num_envs)
        self.critic_observations = torch.zeros(
            *shape, critic_observation_dim, device=self.device
        )
        for name in ("actions", "old_action_mean", "old_action_std"):
            setattr(self, name, torch.zeros(*shape, action_dim, device=self.device))
        for name in (
            "old_action_log_prob", "values", "rewards", "returns", "advantages"
        ):
            setattr(self, name, torch.zeros(*shape, device=self.device))
        self.time_outs = torch.zeros(*shape, dtype=torch.bool, device=self.device)

    def add_rl(
        self,
        critic_observation,
        actions,
        old_action_mean,
        old_action_std,
        old_action_log_prob,
        values,
        rewards,
        time_outs,
    ) -> None:
        """Record RL data before the inherited ``add`` advances the cursor.

        Rewards stay unmodified. The caller supplies timeout-only truncations,
        excluding simultaneous true terminations, for return bootstrapping.
        """
        if self.step >= self.num_steps:
            raise OverflowError("Recurrent visual PPO-DAgger rollout storage is full")
        for name, value in (
            ("critic_observations", critic_observation),
            ("actions", actions),
            ("old_action_mean", old_action_mean),
            ("old_action_std", old_action_std),
            ("old_action_log_prob", old_action_log_prob),
            ("values", values),
            ("rewards", rewards),
            ("time_outs", time_outs),
        ):
            target = getattr(self, name)[self.step]
            target.copy_(value.detach().reshape_as(target))

    def compute_returns(self, last_values, gamma: float, lam: float) -> None:
        """Use RSL-RL's timeout bootstrap and GAE on student trajectories."""
        if self.step != self.num_steps:
            raise RuntimeError(
                f"Expected {self.num_steps} transitions, stored {self.step}"
            )
        owners = self.teacher_controlled[0]
        if not torch.equal(self.teacher_controlled, owners.expand_as(self.teacher_controlled)):
            raise ValueError("Controller ownership must remain fixed within a PPO rollout")
        student = ~owners
        self.returns.zero_()
        self.advantages.zero_()
        if not student.any():
            return
        values = self.values[:, student]
        # Match the installed RSL-RL convention: bootstrap timeout rewards with
        # the current state's value. A done still cuts the GAE recursion.
        rewards = self.rewards[:, student] + (
            gamma * values * self.time_outs[:, student]
        )
        final_values = last_values.detach().reshape(self.num_envs)[student]
        advantage = torch.zeros_like(final_values)
        for step in reversed(range(self.num_steps)):
            next_values = final_values if step == self.num_steps - 1 else values[step + 1]
            not_done = (~self.dones[step, student]).float()
            delta = rewards[step] + gamma * not_done * next_values - values[step]
            advantage = delta + gamma * lam * not_done * advantage
            self.returns[step, student] = advantage + values[step]
        advantages = self.returns[:, student] - values
        self.advantages[:, student] = (
            (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
        )

    def batches(
        self, num_mini_batches: int, num_epochs: int
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Shuffle envs, retaining complete aligned visual and PPO sequences."""
        if self.step != self.num_steps:
            raise RuntimeError(
                f"Expected {self.num_steps} transitions, stored {self.step}"
            )
        if not 1 <= num_mini_batches <= self.num_envs:
            raise ValueError("num_mini_batches must not exceed num_envs")
        for _ in range(num_epochs):
            permutation = torch.randperm(self.num_envs, device=self.device)
            for env_ids in torch.tensor_split(permutation, num_mini_batches):
                batch = {
                    "env_ids": env_ids,
                    "proprio_history": self._reconstruct_proprio_history(env_ids),
                    "depth": self.depth[:, env_ids],
                    "depth_updated": self.depth_updated[:, env_ids],
                    "teacher_action_mean": self.teacher_action_mean[:, env_ids],
                    "dones": self.dones[:, env_ids],
                    "teacher_controlled": self.teacher_controlled[:, env_ids],
                    "initial_hidden": self.initial_hidden[:, env_ids],
                }
                batch.update({name: getattr(self, name)[:, env_ids] for name in self._RL_FIELDS})
                yield batch


__all__ = ["RecurrentVisualPpoDaggerStorage"]
