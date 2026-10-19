"""Action-MSE DAgger with privileged student inputs for information ablation."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .privileged_recurrent_visual_model import (
    CR1PrivilegedRecurrentVisualDaggerStudent,
    PRIVILEGED_ACTOR_DIM,
    PRIVILEGED_KEY,
    PRIVILEGED_LIMB_NAMES,
    PRIVILEGED_RECURRENT_VISUAL_DAGGER_DATA_CONTRACT,
)
from .recurrent_visual_mse_algorithm import CR1RecurrentVisualMseDagger


class CR1PrivilegedRecurrentVisualMseDagger(CR1RecurrentVisualMseDagger):
    """Recurrent visual DAgger with an oracle-only task-information branch."""

    student_class = CR1PrivilegedRecurrentVisualDaggerStudent
    data_contract = PRIVILEGED_RECURRENT_VISUAL_DAGGER_DATA_CONTRACT

    @staticmethod
    def _actor_observations(obs: TensorDict) -> TensorDict:
        if PRIVILEGED_KEY not in obs:
            raise KeyError(f"Missing privileged recurrent visual observation: {PRIVILEGED_KEY}")
        result = CR1RecurrentVisualMseDagger._actor_observations(obs).to_dict()
        result[PRIVILEGED_KEY] = obs[PRIVILEGED_KEY]
        return TensorDict(result, batch_size=obs.batch_size, device=obs.device)

    @staticmethod
    def _validate_observations(obs: TensorDict, teacher_observation_dim: int) -> None:
        CR1RecurrentVisualMseDagger._validate_observations(obs, teacher_observation_dim)
        if PRIVILEGED_KEY not in obs:
            raise KeyError(f"Missing privileged recurrent visual observation: {PRIVILEGED_KEY}")
        batch = obs["proprio_history"].shape[0]
        expected = (batch, PRIVILEGED_ACTOR_DIM)
        if tuple(obs[PRIVILEGED_KEY].shape) != expected:
            raise ValueError(
                f"{PRIVILEGED_KEY} must have shape {expected}, "
                f"got {tuple(obs[PRIVILEGED_KEY].shape)}"
            )
        if not torch.isfinite(obs[PRIVILEGED_KEY]).all():
            raise FloatingPointError(f"{PRIVILEGED_KEY} contains non-finite values")
        if float(obs[PRIVILEGED_KEY].abs().amax()) > 100.0 + 1.0e-5:
            raise ValueError(f"{PRIVILEGED_KEY} exceeds the source clip range [-100, 100]")

    def configure_diagnostics(self, log_dir: str, manifest: dict) -> None:
        super().configure_diagnostics(
            log_dir,
            {
                **manifest,
                "data_contract": self.data_contract,
                "ablation": "visual/proprio recurrent student plus privileged oracle task signals",
                "feature_normalization": (
                    "LayerNorm(128) per proprio/depth branch before GRU"
                ),
                "privileged_actor_input": {
                    "total_dim": PRIVILEGED_ACTOR_DIM,
                    "target_position": "next stable four-limb support target in robot anchor frame, 12-D, meters",
                    "actual_ladder_contact": "rung-filtered force threshold, 4-D binary",
                    "limb_linear_velocity": "four end-effectors in robot anchor frame, 12-D, scaled by 0.1",
                    "deployable": False,
                },
            },
        )

    @staticmethod
    def _privileged_rollout_metrics(privileged: torch.Tensor) -> dict[str, float]:
        """Summarize oracle coverage without changing the optimization path."""

        if privileged.ndim != 3 or privileged.shape[-1] != PRIVILEGED_ACTOR_DIM:
            raise ValueError(
                "Stored privileged rollout must have shape [time, env, "
                f"{PRIVILEGED_ACTOR_DIM}], got {tuple(privileged.shape)}"
            )
        target = privileged[..., :12].reshape(*privileged.shape[:2], 4, 3)
        contact = privileged[..., 12:16]
        velocity_scaled = privileged[..., 16:28].reshape(
            *privileged.shape[:2], 4, 3
        )
        target_distance = torch.linalg.vector_norm(target, dim=-1)
        limb_speed_scaled = torch.linalg.vector_norm(velocity_scaled, dim=-1)

        names = [
            "oracle_target_position_abs_mean_m",
            "oracle_target_position_abs_max_m",
            "oracle_target_distance_mean_m",
            "oracle_target_distance_max_m",
            "oracle_contact_bit_fraction",
            "oracle_any_limb_contact_fraction",
            "oracle_limb_speed_mean_mps",
            "oracle_limb_speed_max_mps",
        ]
        values = [
            target.abs().mean(),
            target.abs().amax(),
            target_distance.mean(),
            target_distance.amax(),
            contact.mean(),
            contact.to(dtype=torch.bool).any(dim=-1).float().mean(),
            limb_speed_scaled.mean() / 0.1,
            limb_speed_scaled.amax() / 0.1,
        ]
        for limb_index, limb_name in enumerate(PRIVILEGED_LIMB_NAMES):
            names.extend(
                (
                    f"oracle_{limb_name}_target_distance_mean_m",
                    f"oracle_{limb_name}_contact_fraction",
                    f"oracle_{limb_name}_speed_mean_mps",
                )
            )
            values.extend(
                (
                    target_distance[..., limb_index].mean(),
                    contact[..., limb_index].mean(),
                    limb_speed_scaled[..., limb_index].mean() / 0.1,
                )
            )
        host_values = torch.stack(values).detach().cpu().tolist()
        return dict(zip(names, host_values, strict=True))

    def update(self) -> dict[str, float]:
        if self.storage.privileged is None:
            raise RuntimeError("Privileged DAgger storage has no oracle tensor")
        privileged_metrics = self._privileged_rollout_metrics(
            self.storage.privileged
        )
        losses = super().update()
        losses.update(privileged_metrics)
        return losses


__all__ = ["CR1PrivilegedRecurrentVisualMseDagger"]
