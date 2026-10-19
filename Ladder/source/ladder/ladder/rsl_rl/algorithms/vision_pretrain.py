"""Auxiliary-only optimization for TransformerVision pretraining."""

from __future__ import annotations

import torch
from torch import nn

from ladder.rsl_rl.modules import (
    TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT,
    TRANSFORMER_VISION_DATA_CONTRACT,
    TaskGoalTransformerVisionLossWeights,
    TransformerVisionEncoder,
    TransformerVisionLossWeights,
    TransformerVisionOutput,
    TransformerVisionTargets,
)
from ladder.rsl_rl.storage import TransformerVisionRolloutStorage


class TransformerVisionOptimizationResult:
    """Outputs and metrics from one visual-only optimizer step."""

    def __init__(
        self,
        output: TransformerVisionOutput,
        losses: dict[str, torch.Tensor],
        gradient_norm: float,
        parameter_change_max_abs: float,
    ) -> None:
        self.output = output
        self.losses = losses
        self.gradient_norm = gradient_norm
        self.parameter_change_max_abs = parameter_change_max_abs


class TransformerVisionPretrainer:
    """Train visual estimators without an action policy or PPO update."""

    def __init__(
        self,
        encoder: TransformerVisionEncoder,
        *,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
        mini_batch_envs: int = 256,
    ) -> None:
        if mini_batch_envs <= 0:
            raise ValueError("mini_batch_envs must be positive")
        self.encoder = encoder
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        self.mini_batch_envs = mini_batch_envs

    def _loss_weights(self) -> TransformerVisionLossWeights | TaskGoalTransformerVisionLossWeights:
        if self.encoder.data_contract == TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT:
            return TaskGoalTransformerVisionLossWeights()
        return TransformerVisionLossWeights()

    def replay(
        self,
        storage: TransformerVisionRolloutStorage,
        *,
        deterministic: bool = False,
        env_slice: slice = slice(None),
        depth_history_override: torch.Tensor | None = None,
    ) -> tuple[TransformerVisionOutput, TransformerVisionTargets]:
        if storage.step != storage.num_steps:
            raise RuntimeError(f"Expected {storage.num_steps} transitions, got {storage.step}")
        depth_history = storage.depth_history[:, env_slice]
        if depth_history_override is not None:
            if depth_history_override.shape != depth_history.shape:
                raise ValueError(
                    "depth_history_override must match the selected stored depth shape, "
                    f"got {tuple(depth_history_override.shape)} and {tuple(depth_history.shape)}"
                )
            depth_history = depth_history_override
        output = self.encoder(
            storage.proprio_history[:, env_slice],
            depth_history,
            hidden_state=storage.initial_hidden_state[:, env_slice],
            dones=storage.reset_before_mask()[:, env_slice],
            masks=torch.ones_like(storage.dones[:, env_slice]),
            deterministic=deterministic,
        )
        targets = TransformerVisionTargets(
            velocity=storage.velocity_targets[:, env_slice],
            limb_heightmaps=storage.limb_heightmaps[:, env_slice],
            heightmap_valid=storage.heightmap_valid[:, env_slice],
            heightmap_hit_class=storage.heightmap_hit_class[:, env_slice],
            next_proprio=storage.next_proprio_targets[:, env_slice],
            future_valid=storage.future_valid[:, env_slice],
            limb_goal_b=(
                storage.limb_goal_targets[:, env_slice]
                if storage.task_goal_available
                else None
            ),
            limb_goal_valid=(
                storage.limb_goal_valid[:, env_slice]
                if storage.task_goal_available
                else None
            ),
            contact_indicator=(
                storage.contact_indicator_targets[:, env_slice]
                if storage.task_goal_available
                else None
            ),
            motion_joint_pos=(
                storage.motion_joint_pos_targets[:, env_slice]
                if storage.task_goal_available
                else None
            ),
            motion_root_orientation_6d=(
                storage.motion_root_orientation_targets[:, env_slice]
                if storage.task_goal_available
                else None
            ),
            motion_valid=(
                storage.motion_target_valid[:, env_slice]
                if storage.task_goal_available
                else None
            ),
        )
        return output, targets

    def evaluate_depth_ablation(
        self, storage: TransformerVisionRolloutStorage
    ) -> dict[str, tuple[TransformerVisionOutput, dict[str, torch.Tensor]]]:
        """Compare deterministic estimates under normal, zero, and mismatched depth.

        The proprioceptive history, GRU initial state, reset sequence, and all
        supervision stay fixed. Only depth changes. A model that genuinely uses
        vision should become less accurate when depth is removed or assigned
        from a different environment.
        """

        was_training = self.encoder.training
        self.encoder.eval()
        depth_variants = {
            "normal": storage.depth_history,
            "zero": torch.zeros_like(storage.depth_history),
        }
        if storage.num_envs > 1:
            depth_variants["shuffled"] = torch.roll(storage.depth_history, shifts=1, dims=1)
        results = {}
        try:
            with torch.inference_mode():
                for name, depth_history in depth_variants.items():
                    output, targets = self.replay(
                        storage,
                        deterministic=True,
                        depth_history_override=depth_history,
                    )
                    results[name] = (
                        output,
                        self.encoder.losses(output, targets, self._loss_weights()),
                    )
        finally:
            self.encoder.train(was_training)
        return results

    def loss(self, storage: TransformerVisionRolloutStorage) -> tuple[TransformerVisionOutput, dict[str, torch.Tensor]]:
        # This path is followed by a real backward pass in the no-update
        # gradient diagnostic. cuDNN recurrent backward requires training mode.
        self.encoder.train()
        output, targets = self.replay(storage)
        return output, self.encoder.losses(output, targets, self._loss_weights())

    def gradient_norm_without_step(self, loss: torch.Tensor) -> float:
        """Measure the exact pretraining gradient while keeping weights fixed."""

        snapshot = [parameter.detach().clone() for parameter in self.encoder.parameters()]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        squared_norm = sum(
            (parameter.grad.detach().square().sum() for parameter in self.encoder.parameters() if parameter.grad is not None),
            torch.zeros((), device=loss.device),
        )
        self.optimizer.zero_grad(set_to_none=True)
        changed = max(
            float((parameter.detach() - before).abs().max())
            for parameter, before in zip(self.encoder.parameters(), snapshot, strict=True)
        )
        if changed != 0.0:
            raise RuntimeError(f"No-update diagnostic changed visual parameters by {changed}")
        return float(torch.sqrt(squared_norm))

    def optimize(self, storage: TransformerVisionRolloutStorage) -> TransformerVisionOptimizationResult:
        """Apply one visual training epoch over environment mini-batches.

        The stored trajectory has already been generated by the frozen actor.
        This method never has an action, critic, PPO ratio, or policy parameter
        in its computation graph. Each mini-batch retains all rollout time steps
        for its environments, so the GRU reset sequence remains intact.
        """

        self.encoder.train()
        snapshot = [parameter.detach().clone() for parameter in self.encoder.parameters()]
        weighted_loss_sums: dict[str, torch.Tensor] | None = None
        max_gradient_norm = 0.0
        total_envs = 0
        output: TransformerVisionOutput | None = None

        for start in range(0, storage.num_envs, self.mini_batch_envs):
            stop = min(start + self.mini_batch_envs, storage.num_envs)
            batch_envs = stop - start
            output, targets = self.replay(storage, env_slice=slice(start, stop))
            losses = self.encoder.losses(output, targets, self._loss_weights())
            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"TransformerVision gradient norm is non-finite: {float(gradient_norm)}")
            self.optimizer.step()

            if weighted_loss_sums is None:
                weighted_loss_sums = {name: value.detach() * batch_envs for name, value in losses.items()}
            else:
                for name, value in losses.items():
                    weighted_loss_sums[name].add_(value.detach() * batch_envs)
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            total_envs += batch_envs

        if output is None or weighted_loss_sums is None or total_envs == 0:
            raise RuntimeError("TransformerVision storage did not provide a mini-batch")
        mean_losses = {name: value / total_envs for name, value in weighted_loss_sums.items()}
        parameter_change = max(
            float((parameter.detach() - before).abs().max())
            for parameter, before in zip(self.encoder.parameters(), snapshot, strict=True)
        )
        if parameter_change == 0.0:
            raise RuntimeError("TransformerVision optimizer step did not change any visual parameter")
        return TransformerVisionOptimizationResult(
            output=output,
            losses=mean_losses,
            gradient_norm=max_gradient_norm,
            parameter_change_max_abs=parameter_change,
        )

    def state_dict(self) -> dict:
        """Return the encoder and its visual-only optimizer state."""

        return {
            "data_contract": self.encoder.data_contract,
            "model_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict, *, load_optimizer: bool = True) -> None:
        """Restore a checkpoint produced by :meth:`state_dict`."""

        if "model_state_dict" not in state:
            raise KeyError("TransformerVision checkpoint is missing model_state_dict")
        checkpoint_contract = state.get("data_contract")
        if checkpoint_contract != self.encoder.data_contract:
            raise RuntimeError(
                "This TransformerVision checkpoint uses data contract "
                f"{checkpoint_contract!r}, but the selected encoder requires "
                f"{self.encoder.data_contract!r}. Select the matching legacy encoder for inspection, "
                "or start a new run without --resume."
            )
        self.encoder.load_state_dict(state["model_state_dict"], strict=True)
        if load_optimizer:
            if "optimizer_state_dict" not in state:
                raise KeyError("TransformerVision checkpoint is missing optimizer_state_dict")
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
