"""Sequence-trained recurrent visual action-MSE DAgger."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from .recurrent_visual_model import (
    CR1RecurrentVisualDaggerStudent,
    DEPTH_UPDATE_KEY,
    DEPTH_INPUT_MAX,
    DEPTH_INPUT_MIN,
    PROPRIO_DIM,
    PROPRIO_HISTORY_FRAMES,
    RECURRENT_VISUAL_DAGGER_DATA_CONTRACT,
    PRIVILEGED_KEY,
)
from .recurrent_visual_mse_storage import (
    RecurrentVisualMseDaggerStorage,
    RecurrentVisualMseDaggerTransition,
)
from .student_teacher import TEACHER_KEY, load_frozen_teacher
from .visual_model import DEPTH_KEY, PROPRIO_KEY
from .visual_mse_algorithm import CR1VisualMseDagger


class CR1RecurrentVisualMseDagger(CR1VisualMseDagger):
    """Preserve recurrent time order for student-only or mixed DAgger rollouts."""

    student_class = CR1RecurrentVisualDaggerStudent
    data_contract = RECURRENT_VISUAL_DAGGER_DATA_CONTRACT

    def __init__(
        self,
        *args,
        bc_warmup_iterations: int,
        teacher_rollout_fraction: float,
        teacher_rollout_schedule_start: int | None = None,
        teacher_rollout_schedule_end: int | None = None,
        teacher_rollout_schedule_initial_fraction: float | None = None,
        teacher_rollout_schedule_final_fraction: float | None = None,
        termination_manager=None,
        **kwargs,
    ) -> None:
        if bc_warmup_iterations < 0:
            raise ValueError("bc_warmup_iterations must be non-negative")
        if not 0.0 <= teacher_rollout_fraction <= 1.0:
            raise ValueError("teacher_rollout_fraction must be in [0, 1]")
        self.bc_warmup_iterations = int(bc_warmup_iterations)
        self.teacher_rollout_fraction = float(teacher_rollout_fraction)
        schedule_values = (
            teacher_rollout_schedule_start,
            teacher_rollout_schedule_end,
            teacher_rollout_schedule_initial_fraction,
            teacher_rollout_schedule_final_fraction,
        )
        if any(value is not None for value in schedule_values):
            if any(value is None for value in schedule_values):
                raise ValueError(
                    "teacher rollout schedule requires start, end, initial "
                    "fraction, and final fraction"
                )
            if teacher_rollout_schedule_start < 0:
                raise ValueError("teacher_rollout_schedule_start must be non-negative")
            if teacher_rollout_schedule_end <= teacher_rollout_schedule_start:
                raise ValueError(
                    "teacher_rollout_schedule_end must be greater than start"
                )
            for name, value in (
                ("teacher_rollout_schedule_initial_fraction", teacher_rollout_schedule_initial_fraction),
                ("teacher_rollout_schedule_final_fraction", teacher_rollout_schedule_final_fraction),
            ):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1]")
            self.teacher_rollout_schedule_start = int(teacher_rollout_schedule_start)
            self.teacher_rollout_schedule_end = int(teacher_rollout_schedule_end)
            self.teacher_rollout_schedule_initial_fraction = float(
                teacher_rollout_schedule_initial_fraction
            )
            self.teacher_rollout_schedule_final_fraction = float(
                teacher_rollout_schedule_final_fraction
            )
        else:
            self.teacher_rollout_schedule_start = None
            self.teacher_rollout_schedule_end = None
            self.teacher_rollout_schedule_initial_fraction = None
            self.teacher_rollout_schedule_final_fraction = None
        self._termination_manager = termination_manager
        self._termination_names = (
            tuple(termination_manager.active_terms)
            if termination_manager is not None
            else ()
        )
        super().__init__(*args, **kwargs)
        self._pending: RecurrentVisualMseDaggerTransition | None = None
        self._rollout_hidden: torch.Tensor | None = None

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env, cfg: dict, device: str):
        policy_cfg = cfg["policy"]
        algorithm_cfg = dict(cfg["algorithm"])
        for key in (
            "class_name",
            "rnd_cfg",
            "symmetry_cfg",
            "optimizer",
            "normalize_advantage_per_mini_batch",
            "schedule",
            "desired_kl",
            "gamma",
            "lam",
            "clip_param",
            "value_loss_coef",
            "entropy_coef",
            "use_clipped_value_loss",
        ):
            algorithm_cfg.pop(key, None)

        teacher_checkpoint = algorithm_cfg.pop("teacher_checkpoint")
        teacher_hidden_dims = list(algorithm_cfg.pop("teacher_hidden_dims"))
        expected_teacher_observation_dim = int(
            algorithm_cfg.pop("expected_teacher_observation_dim")
        )
        cls._validate_observations(obs, expected_teacher_observation_dim)
        if not teacher_checkpoint:
            raise ValueError("Recurrent visual MSE-DAgger requires teacher_checkpoint")
        teacher = load_frozen_teacher(
            teacher_checkpoint,
            observation_dim=expected_teacher_observation_dim,
            action_dim=env.num_actions,
            hidden_dims=teacher_hidden_dims,
            device=device,
        )
        init_noise_std = float(policy_cfg["init_noise_std"])
        if init_noise_std <= 0.0:
            raise ValueError("Recurrent visual DAgger rollout noise must be positive")
        policy = cls.student_class(
            env.num_actions,
            list(policy_cfg["actor_hidden_dims"]),
            init_noise_std,
        )
        storage = RecurrentVisualMseDaggerStorage(
            int(cfg["num_steps_per_env"]),
            env.num_envs,
            env.num_actions,
            device,
            privileged_dim=getattr(policy, "privileged_dim", 0),
        )
        unwrapped_env = getattr(env, "unwrapped", None)
        instance = cls(
            policy,
            teacher,
            storage,
            teacher_checkpoint=teacher_checkpoint,
            device=device,
            num_learning_epochs=int(algorithm_cfg.pop("num_learning_epochs")),
            num_mini_batches=int(algorithm_cfg.pop("num_mini_batches")),
            learning_rate=float(algorithm_cfg.pop("learning_rate")),
            max_grad_norm=float(algorithm_cfg.pop("max_grad_norm")),
            diagnostic_snapshot_interval=int(
                algorithm_cfg.pop("diagnostic_snapshot_interval")
            ),
            diagnostic_sample_envs=int(
                algorithm_cfg.pop("diagnostic_sample_envs")
            ),
            bc_warmup_iterations=int(algorithm_cfg.pop("bc_warmup_iterations", 0)),
            teacher_rollout_fraction=float(
                algorithm_cfg.pop("teacher_rollout_fraction", 0.0)
            ),
            teacher_rollout_schedule_start=algorithm_cfg.pop(
                "teacher_rollout_schedule_start", None
            ),
            teacher_rollout_schedule_end=algorithm_cfg.pop(
                "teacher_rollout_schedule_end", None
            ),
            teacher_rollout_schedule_initial_fraction=algorithm_cfg.pop(
                "teacher_rollout_schedule_initial_fraction", None
            ),
            teacher_rollout_schedule_final_fraction=algorithm_cfg.pop(
                "teacher_rollout_schedule_final_fraction", None
            ),
            termination_manager=getattr(
                unwrapped_env, "termination_manager", None
            ),
        )
        if algorithm_cfg:
            raise ValueError(
                "Unused recurrent visual MSE-DAgger settings: "
                f"{sorted(algorithm_cfg)}"
            )
        return instance

    def configure_diagnostics(self, log_dir: str, manifest: dict) -> None:
        self._diagnostic_dir = Path(log_dir) / "recurrent_visual_mse_diagnostics"
        self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
        post_warmup_teacher_envs = int(
            self.storage.num_envs * self.teacher_rollout_fraction
        )
        mixed_rollout = (
            self.bc_warmup_iterations > 0
            or post_warmup_teacher_envs > 0
            or self.teacher_rollout_schedule_start is not None
        )
        if self.teacher_rollout_schedule_start is not None:
            rollout_policy = (
                "teacher_bc_warmup_then_linear_teacher_student_mix"
            )
        elif self.bc_warmup_iterations > 0:
            rollout_policy = (
                "teacher_bc_warmup_then_fixed_per_environment_teacher_student_mix"
            )
        elif post_warmup_teacher_envs > 0:
            rollout_policy = (
                "fixed_per_environment_teacher_student_mix_from_iteration_zero"
            )
        else:
            rollout_policy = "student_stochastic_only"
        payload = {
            "data_contract": self.data_contract,
            "rollout_policy": rollout_policy,
            "teacher_role": (
                "online_action_mean_label_for_all_states_and_controller_for_fixed_subset"
                if mixed_rollout
                else "online_action_mean_label_at_student_visited_state"
            ),
            "controller_schedule": {
                "bc_warmup_iterations": self.bc_warmup_iterations,
                "teacher_rollout_fraction_after_warmup": (
                    self.teacher_rollout_fraction
                ),
                "teacher_rollout_schedule_start": self.teacher_rollout_schedule_start,
                "teacher_rollout_schedule_end": self.teacher_rollout_schedule_end,
                "teacher_rollout_schedule_initial_fraction": (
                    self.teacher_rollout_schedule_initial_fraction
                ),
                "teacher_rollout_schedule_final_fraction": (
                    self.teacher_rollout_schedule_final_fraction
                ),
                "teacher_envs_after_warmup": post_warmup_teacher_envs,
                "student_envs_after_warmup": (
                    self.storage.num_envs - post_warmup_teacher_envs
                ),
                "mixed_rollout_starts_at_iteration": self.bc_warmup_iterations,
                "teacher_env_selection": "fixed_lowest_environment_indices",
                "teacher_action": "deterministic_mean",
                "student_action": "mean_plus_fixed_gaussian_noise",
                "loss_samples": "all_teacher_and_student_controlled_environments",
                "split_diagnostics": (
                    "action_error_reward_done_and_termination_cause_by_controller"
                ),
            },
            "optimization_loss": "sequence_mean_squared_error(student_mean, teacher_mean)",
            "recurrent_training": {
                "sequence_length_policy_steps": self.storage.num_steps,
                "mini_batch_axis": "environment",
                "time_axis_shuffled": False,
                "hidden_state_source": "rollout_start_hidden",
                "done_mask_timing": "reset_before_t_plus_1",
                "gru_update_timing": "fresh_depth_frames_only",
            },
            "excluded_losses": [
                "ppo",
                "policy_kl",
                "teacher_feature_mse",
                "velocity_prediction",
                "heightmap_reconstruction",
                "decoder_reconstruction",
            ],
            "student_inputs": {
                "proprio_history_shape": [PROPRIO_HISTORY_FRAMES, PROPRIO_DIM],
                "command_in_every_proprio_frame": True,
                "climb_command_index_per_proprio_frame": PROPRIO_DIM - 1,
                "depth_shape": [1, 60, 80],
                "depth_update_mask_shape": [1],
            },
            "normalization_contract": {
                "student_observation_normalization": "disabled_running_statistics",
                "student_proprio": {
                    "root_angular_velocity_scale": 0.2,
                    "root_orientation": "rotation_matrix_first_two_columns_in[-1,1]",
                    "joint_position": "(q-default_q)/CR1_ACTION_SCALE",
                    "joint_velocity_scale": 0.1,
                    "previous_action": "normalized_joint_position_action_space",
                    "command_values": [-1.0, 0.0, 1.0],
                    "final_clip": [-100.0, 100.0],
                },
                "student_depth": {
                    "metric_clip_m": [0.1, 2.0],
                    "center_m": 1.0,
                    "scale": 1.0,
                    "invalid_pixel_after_centering": -1.0,
                    "encoder_input_range": [DEPTH_INPUT_MIN, DEPTH_INPUT_MAX],
                },
                "teacher_observation": {
                    "running_statistics": False,
                    "checkpoint_actor_observation_normalization": False,
                    "reason": "must match the 780-D teacher checkpoint contract",
                },
                "action_loss": {
                    "space": "raw normalized joint-position action coordinates",
                    "normalization": "none beyond environment action contract",
                    "reduction": "mean over batch,time,action",
                },
                "gru_hidden": "zero initialized and reset on done; no running normalization",
            },
            "teacher_observation_dim": int(self.teacher[0].in_features),
            "student_network": repr(self.policy),
            "fixed_student_rollout_std": self.policy.output_std.detach().cpu().tolist(),
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            **manifest,
        }
        (self._diagnostic_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self._pending is not None:
            raise RuntimeError("process_env_step must follow every act call")
        self._validate_observations(obs, int(self.teacher[0].in_features))
        actor_obs = self._actor_observations(obs)
        hidden_before = self._rollout_hidden
        if hidden_before is None:
            hidden_before = self.policy.initial_hidden(
                obs.batch_size[0],
                device=obs.device,
                dtype=obs[PROPRIO_KEY].dtype,
            )
        student_mean, _, next_hidden, activation_tensors = (
            self.policy.action_mean_and_feature_step(actor_obs, hidden_before)
        )
        self._rollout_hidden = next_hidden.detach()
        student_mean = student_mean.detach()
        teacher_mean = self.teacher(obs[TEACHER_KEY]).detach()
        if not torch.isfinite(student_mean).all():
            raise FloatingPointError("Student action mean contains non-finite values")
        if not torch.isfinite(teacher_mean).all():
            raise FloatingPointError("Teacher action mean contains non-finite values")
        exploration = torch.randn_like(student_mean) * self.policy.output_std
        student_actions = student_mean + exploration
        teacher_controlled = self._teacher_controlled_mask(student_mean.shape[0])
        actions = torch.where(
            teacher_controlled[:, None], teacher_mean, student_actions
        )
        executed_exploration = torch.where(
            teacher_controlled[:, None], torch.zeros_like(exploration), exploration
        )

        error = student_mean - teacher_mean
        self._accumulate_rollout_metrics(
            obs,
            student_mean,
            teacher_mean,
            student_actions,
            error,
            executed_exploration,
            hidden_before,
            next_hidden,
            activation_tensors,
        )
        self._accumulate_controller_action_metrics(error, teacher_controlled)
        if (
            self._diagnostic_dir is not None
            and self._snapshot is None
            and self.update_iteration % self.diagnostic_snapshot_interval == 0
        ):
            count = min(self.diagnostic_sample_envs, student_mean.shape[0])
            env_ids = self._diagnostic_env_ids(teacher_controlled, count)
            self._snapshot = {
                "iteration": self.update_iteration,
                "environment_ids": env_ids.detach().cpu().clone(),
                "teacher_controlled": (
                    teacher_controlled[env_ids].detach().cpu().clone()
                ),
                "proprio_history": obs[PROPRIO_KEY][env_ids].detach().cpu().clone(),
                "depth": obs[DEPTH_KEY][env_ids].detach().cpu().clone(),
                "depth_updated": obs[DEPTH_UPDATE_KEY][env_ids].detach().cpu().clone(),
                "teacher_observation": (
                    obs[TEACHER_KEY][env_ids].detach().cpu().clone()
                ),
                "hidden_before": hidden_before[:, env_ids].detach().cpu().clone(),
                "hidden_after": next_hidden[:, env_ids].detach().cpu().clone(),
                "proprio_feature": (
                    activation_tensors["proprio_feature"][env_ids]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "depth_feature": (
                    activation_tensors["depth_feature"][env_ids]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "recurrent_input": (
                    activation_tensors["recurrent_input"][env_ids]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "gru_candidate": (
                    activation_tensors["gru_candidate"][:, env_ids]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "recurrent_feature": (
                    activation_tensors["recurrent_feature"][env_ids]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "actor_feature": (
                    activation_tensors["actor_feature"][env_ids]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "student_mean": student_mean[env_ids].cpu().clone(),
                "teacher_mean": teacher_mean[env_ids].cpu().clone(),
                "candidate_student_action": student_actions[env_ids].cpu().clone(),
                "executed_action": actions[env_ids].cpu().clone(),
            }
            if PRIVILEGED_KEY in actor_obs:
                self._snapshot[PRIVILEGED_KEY] = (
                    actor_obs[PRIVILEGED_KEY][env_ids].detach().cpu().clone()
                )

        self._pending = RecurrentVisualMseDaggerTransition(
            proprio_history=obs[PROPRIO_KEY].detach(),
            depth=obs[DEPTH_KEY].detach(),
            depth_updated=obs[DEPTH_UPDATE_KEY].detach(),
            teacher_action_mean=teacher_mean,
            hidden_before=hidden_before.detach(),
            teacher_controlled=teacher_controlled,
            privileged=(
                actor_obs[PRIVILEGED_KEY].detach()
                if PRIVILEGED_KEY in actor_obs
                else None
            ),
        )
        return actions

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        del obs, extras
        if self._pending is None:
            raise RuntimeError("act must be called before process_env_step")
        transition = self._pending
        self.storage.add(transition, dones)
        self._pending = None
        self._rollout_reward_sum += rewards.sum()
        done = dones.to(dtype=torch.bool)
        self._rollout_done_count += done.to(torch.float32).sum()
        teacher_controlled = transition.teacher_controlled
        student_controlled = ~teacher_controlled
        self._rollout_teacher_controlled_reward_sum += rewards[
            teacher_controlled
        ].sum()
        self._rollout_student_controlled_reward_sum += rewards[
            student_controlled
        ].sum()
        self._rollout_teacher_controlled_done_count += done[
            teacher_controlled
        ].to(torch.float32).sum()
        self._rollout_student_controlled_done_count += done[
            student_controlled
        ].to(torch.float32).sum()
        if self._termination_manager is not None:
            for name in self._termination_names:
                terminated = self._termination_manager.get_term(name).to(
                    dtype=torch.bool
                )
                self._rollout_teacher_controlled_termination_counts[name] += (
                    terminated[teacher_controlled].to(torch.float32).sum()
                )
                self._rollout_student_controlled_termination_counts[name] += (
                    terminated[student_controlled].to(torch.float32).sum()
                )
        self._rollout_hidden_reset_count += done.to(torch.float32).sum()
        if self._rollout_hidden is not None:
            self._rollout_hidden = self.policy.reset_hidden_state(
                self._rollout_hidden, done
            ).detach()
        if self._snapshot is not None and "reward" not in self._snapshot:
            env_ids = self._snapshot["environment_ids"].to(rewards.device)
            self._snapshot["reward"] = rewards[env_ids].detach().cpu().clone()
            self._snapshot["done"] = done[env_ids].detach().cpu().clone()

    def update(self) -> dict[str, float]:
        parameters_before = {
            name: value.detach().clone()
            for name, value in self.policy.named_parameters()
            if value.requires_grad
        }
        training_squared_error = torch.zeros(
            self.policy.action_dim, device=self.device
        )
        training_sample_count = 0
        training_teacher_squared_error = torch.zeros((), device=self.device)
        training_student_squared_error = torch.zeros((), device=self.device)
        training_teacher_sample_count = 0
        training_student_sample_count = 0
        gradient_norm_sum = {name: 0.0 for name in self._module_prefixes()}
        gradient_norm_max = {name: 0.0 for name in self._module_prefixes()}
        optimizer_steps = 0

        for batch in self.storage.batches(
            self.num_mini_batches, self.num_learning_epochs
        ):
            student_mean, _, _ = self.policy.action_mean_and_feature_sequence(
                batch[PROPRIO_KEY],
                batch["depth"],
                batch[DEPTH_UPDATE_KEY],
                batch["dones"],
                batch["initial_hidden"],
                batch.get(PRIVILEGED_KEY),
            )
            per_element_error = (
                student_mean - batch["teacher_action_mean"]
            ).square()
            loss = per_element_error.mean()
            if not torch.isfinite(student_mean).all():
                raise FloatingPointError(
                    "Student sequence action mean contains non-finite values"
                )
            if not torch.isfinite(batch["teacher_action_mean"]).all():
                raise FloatingPointError(
                    "Stored teacher action mean contains non-finite values"
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Recurrent visual MSE loss is non-finite")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in self.policy.parameters()
                    if parameter.requires_grad
                ),
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            batch_gradient_norms = self._gradient_norms()
            for name, value in batch_gradient_norms.items():
                gradient_norm_sum[name] += value
                gradient_norm_max[name] = max(gradient_norm_max[name], value)
            self.optimizer.step()
            training_squared_error += per_element_error.detach().sum(dim=(0, 1))
            training_sample_count += per_element_error.shape[0] * per_element_error.shape[1]
            teacher_controlled = batch["teacher_controlled"]
            student_controlled = ~teacher_controlled
            training_teacher_squared_error += per_element_error.detach()[
                teacher_controlled
            ].sum()
            training_student_squared_error += per_element_error.detach()[
                student_controlled
            ].sum()
            training_teacher_sample_count += int(teacher_controlled.sum().item())
            training_student_sample_count += int(student_controlled.sum().item())
            optimizer_steps += 1

        self.storage.clear()
        self.update_iteration += 1
        if optimizer_steps == 0 or training_sample_count == 0:
            raise RuntimeError("Recurrent visual DAgger update produced no optimizer steps")

        losses = self._finalize_rollout_metrics()
        losses["behavior"] = float(
            training_squared_error.sum()
            / (training_sample_count * self.policy.action_dim)
        )
        losses["training_action_rmse"] = losses["behavior"] ** 0.5
        for index, squared_error in enumerate(training_squared_error):
            losses[f"training_joint_{index:02d}_rmse"] = float(
                (squared_error / training_sample_count).sqrt()
            )
        losses["training_teacher_controlled_action_rmse"] = self._split_rmse(
            training_teacher_squared_error, training_teacher_sample_count
        )
        losses["training_student_controlled_action_rmse"] = self._split_rmse(
            training_student_squared_error, training_student_sample_count
        )
        losses.update(self._parameter_delta_metrics(parameters_before))
        for name in self._module_prefixes():
            losses[f"student_{name}_grad_l2_mean_after_clip"] = (
                gradient_norm_sum[name] / optimizer_steps
            )
            losses[f"student_{name}_grad_l2_max_after_clip"] = gradient_norm_max[name]
        losses["optimizer_steps"] = float(optimizer_steps)
        losses["recurrent_sequence_length"] = float(self.storage.num_steps)

        self._assert_teacher_frozen()
        if self._snapshot is not None and self._diagnostic_dir is not None:
            snapshot_iteration = int(self._snapshot["iteration"])
            torch.save(
                self._snapshot,
                self._diagnostic_dir
                / f"iteration_{snapshot_iteration:06d}_sample.pt",
            )
        self._snapshot = None
        self._reset_rollout_metrics()
        return losses

    def save(self) -> dict:
        return {
            "data_contract": self.data_contract,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_iteration": self.update_iteration,
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_checkpoint_sha256": self.teacher_checkpoint_sha256,
            "bc_warmup_iterations": self.bc_warmup_iterations,
            "teacher_rollout_fraction": self.teacher_rollout_fraction,
            "teacher_rollout_schedule_start": self.teacher_rollout_schedule_start,
            "teacher_rollout_schedule_end": self.teacher_rollout_schedule_end,
            "teacher_rollout_schedule_initial_fraction": (
                self.teacher_rollout_schedule_initial_fraction
            ),
            "teacher_rollout_schedule_final_fraction": (
                self.teacher_rollout_schedule_final_fraction
            ),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if loaded_dict.get("data_contract") != self.data_contract:
            raise RuntimeError(
                "Incompatible recurrent visual MSE-DAgger checkpoint data contract"
            )
        source_bc_warmup = int(loaded_dict.get("bc_warmup_iterations", 0))
        source_teacher_fraction = float(
            loaded_dict.get("teacher_rollout_fraction", 0.0)
        )
        source_schedule = (
            loaded_dict.get("teacher_rollout_schedule_start"),
            loaded_dict.get("teacher_rollout_schedule_end"),
            loaded_dict.get("teacher_rollout_schedule_initial_fraction"),
            loaded_dict.get("teacher_rollout_schedule_final_fraction"),
        )
        current_schedule = (
            self.teacher_rollout_schedule_start,
            self.teacher_rollout_schedule_end,
            self.teacher_rollout_schedule_initial_fraction,
            self.teacher_rollout_schedule_final_fraction,
        )
        if (
            source_bc_warmup != self.bc_warmup_iterations
            or source_teacher_fraction != self.teacher_rollout_fraction
            or source_schedule != current_schedule
        ):
            raise RuntimeError(
                "The recurrent visual DAgger checkpoint used a different "
                "controller schedule: "
                f"warmup={source_bc_warmup}, teacher_fraction="
                f"{source_teacher_fraction} != warmup="
                f"{self.bc_warmup_iterations}, teacher_fraction="
                f"{self.teacher_rollout_fraction}, schedule="
                f"{source_schedule} != {current_schedule}."
            )
        source_teacher_digest = loaded_dict.get("teacher_checkpoint_sha256")
        if source_teacher_digest != self.teacher_checkpoint_sha256:
            raise RuntimeError(
                "The recurrent visual DAgger checkpoint used a different teacher: "
                f"{source_teacher_digest} != {self.teacher_checkpoint_sha256}."
            )
        load_cfg = load_cfg or {"actor": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor", True):
            self.policy.load_state_dict(
                loaded_dict["policy_state_dict"], strict=strict
            )
        if load_cfg.get("optimizer", True):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.update_iteration = int(loaded_dict.get("update_iteration", 0))
        self._rollout_hidden = None
        self.policy.reset()
        self.teacher.eval().requires_grad_(False)
        self._teacher_reference = {
            name: value.detach().clone()
            for name, value in self.teacher.state_dict().items()
        }
        return bool(load_cfg.get("iteration", True))

    def _teacher_controlled_mask(self, num_envs: int) -> torch.Tensor:
        teacher_count = int(
            num_envs * self._teacher_control_fraction(self.update_iteration)
        )
        return torch.arange(num_envs, device=self.device) < teacher_count

    def _teacher_control_fraction(self, iteration: int) -> float:
        """Return the deterministic teacher-controlled environment fraction."""

        if iteration < self.bc_warmup_iterations:
            return 1.0
        if self.teacher_rollout_schedule_start is None:
            return self.teacher_rollout_fraction
        if iteration <= self.teacher_rollout_schedule_start:
            return self.teacher_rollout_schedule_initial_fraction
        if iteration >= self.teacher_rollout_schedule_end:
            return self.teacher_rollout_schedule_final_fraction
        progress = (
            (iteration - self.teacher_rollout_schedule_start)
            / (self.teacher_rollout_schedule_end - self.teacher_rollout_schedule_start)
        )
        return (
            self.teacher_rollout_schedule_initial_fraction
            + progress
            * (
                self.teacher_rollout_schedule_final_fraction
                - self.teacher_rollout_schedule_initial_fraction
            )
        )

    @staticmethod
    def _diagnostic_env_ids(
        teacher_controlled: torch.Tensor, count: int
    ) -> torch.Tensor:
        teacher_ids = torch.nonzero(teacher_controlled, as_tuple=False).flatten()
        student_ids = torch.nonzero(~teacher_controlled, as_tuple=False).flatten()
        if teacher_ids.numel() == 0:
            return student_ids[:count]
        if student_ids.numel() == 0:
            return teacher_ids[:count]
        teacher_count = min((count + 1) // 2, teacher_ids.numel())
        student_count = min(count - teacher_count, student_ids.numel())
        remaining = count - teacher_count - student_count
        if remaining > 0:
            teacher_extra = min(remaining, teacher_ids.numel() - teacher_count)
            teacher_count += teacher_extra
            student_count += remaining - teacher_extra
        return torch.cat(
            (teacher_ids[:teacher_count], student_ids[:student_count])
        )

    def _accumulate_controller_action_metrics(
        self, error: torch.Tensor, teacher_controlled: torch.Tensor
    ) -> None:
        student_controlled = ~teacher_controlled
        self._rollout_teacher_controlled_squared_error += error[
            teacher_controlled
        ].square().sum()
        self._rollout_student_controlled_squared_error += error[
            student_controlled
        ].square().sum()
        self._rollout_teacher_controlled_samples += teacher_controlled.to(
            torch.float32
        ).sum()
        self._rollout_student_controlled_samples += student_controlled.to(
            torch.float32
        ).sum()

    def _split_rmse(
        self, squared_error: torch.Tensor, sample_count: int | torch.Tensor
    ) -> float:
        denominator = torch.as_tensor(
            sample_count, device=self.device, dtype=torch.float32
        ) * self.policy.action_dim
        if float(denominator.item()) == 0.0:
            return 0.0
        return float((squared_error / denominator).sqrt())

    @staticmethod
    def _actor_observations(obs: TensorDict) -> TensorDict:
        result = {
                PROPRIO_KEY: obs[PROPRIO_KEY],
                DEPTH_KEY: obs[DEPTH_KEY],
                DEPTH_UPDATE_KEY: obs[DEPTH_UPDATE_KEY],
            }
        if PRIVILEGED_KEY in obs:
            result[PRIVILEGED_KEY] = obs[PRIVILEGED_KEY]
        return TensorDict(
            result,
            batch_size=obs.batch_size,
            device=obs.device,
        )

    @staticmethod
    def _validate_observations(obs: TensorDict, teacher_observation_dim: int) -> None:
        required = (PROPRIO_KEY, DEPTH_KEY, DEPTH_UPDATE_KEY, TEACHER_KEY)
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f"Missing recurrent visual DAgger observations: {missing}")
        batch = obs[PROPRIO_KEY].shape[0]
        expected_shapes = {
            PROPRIO_KEY: (batch, PROPRIO_HISTORY_FRAMES, PROPRIO_DIM),
            DEPTH_KEY: (batch, 1, 60, 80),
            TEACHER_KEY: (batch, teacher_observation_dim),
        }
        for key, expected in expected_shapes.items():
            if tuple(obs[key].shape) != expected:
                raise ValueError(
                    f"Recurrent visual DAgger {key} must have shape {expected}, "
                    f"got {tuple(obs[key].shape)}"
                )
            if not torch.isfinite(obs[key]).all():
                raise FloatingPointError(
                    f"Recurrent visual DAgger {key} contains non-finite values"
                )
        proprio_max = float(obs[PROPRIO_KEY].abs().amax())
        if proprio_max > 100.0 + 1.0e-5:
            raise ValueError(
                "Recurrent visual DAgger proprio history exceeds [-100, 100]: "
                f"max_abs={proprio_max:.6g}"
            )
        depth_min = float(obs[DEPTH_KEY].amin())
        depth_max = float(obs[DEPTH_KEY].amax())
        if depth_min < DEPTH_INPUT_MIN - 1.0e-5 or depth_max > DEPTH_INPUT_MAX + 1.0e-5:
            raise ValueError(
                "Recurrent visual DAgger depth is outside the centered metric-depth "
                f"contract [{DEPTH_INPUT_MIN}, {DEPTH_INPUT_MAX}]: "
                f"range=[{depth_min:.6g}, {depth_max:.6g}]"
            )
        if tuple(obs[DEPTH_UPDATE_KEY].shape) not in ((batch,), (batch, 1)):
            raise ValueError("depth_updated must have shape [B] or [B,1]")
        update_values = obs[DEPTH_UPDATE_KEY]
        if not ((update_values == 0) | (update_values == 1)).all():
            raise ValueError("depth_updated must contain only zero or one")
        command = obs[PROPRIO_KEY][..., PROPRIO_DIM - 1]
        valid_command = (
            torch.isclose(command, torch.full_like(command, -1.0))
            | torch.isclose(command, torch.zeros_like(command))
            | torch.isclose(command, torch.ones_like(command))
        )
        if not valid_command.all():
            raise ValueError(
                "Every proprio history frame command must be -1, 0, or +1"
            )

    def _accumulate_rollout_metrics(
        self,
        obs: TensorDict,
        student_mean: torch.Tensor,
        teacher_mean: torch.Tensor,
        student_actions: torch.Tensor,
        error: torch.Tensor,
        exploration: torch.Tensor,
        hidden_before: torch.Tensor,
        hidden_after: torch.Tensor,
        activation_tensors: dict[str, torch.Tensor],
    ) -> None:
        del student_actions
        sample_count = error.shape[0]
        self._rollout_squared_error += error.square().sum(dim=0)
        self._rollout_absolute_error += error.abs().sum(dim=0)
        self._rollout_max_error = torch.maximum(
            self._rollout_max_error, error.abs().amax(dim=0)
        )
        self._rollout_exploration_squared += exploration.square().sum()
        self._rollout_student_squared += student_mean.square().sum()
        self._rollout_teacher_squared += teacher_mean.square().sum()
        self._rollout_student_sum += student_mean.sum(dim=0)
        self._rollout_teacher_sum += teacher_mean.sum(dim=0)
        self._rollout_cross_sum += (student_mean * teacher_mean).sum(dim=0)
        self._rollout_student_square_sum += student_mean.square().sum(dim=0)
        self._rollout_teacher_square_sum += teacher_mean.square().sum(dim=0)
        self._rollout_cosine_sum += F.cosine_similarity(
            student_mean, teacher_mean, dim=-1, eps=1.0e-8
        ).sum()
        depth = obs[DEPTH_KEY]
        self._rollout_depth_sum += depth.sum()
        self._rollout_depth_square_sum += depth.square().sum()
        self._rollout_depth_value_count += depth.numel()
        self._rollout_depth_min = torch.minimum(self._rollout_depth_min, depth.min())
        self._rollout_depth_max = torch.maximum(self._rollout_depth_max, depth.max())
        self._rollout_distinct_depth_frames += obs[DEPTH_UPDATE_KEY].to(
            dtype=torch.float32
        ).sum()
        proprio = obs[PROPRIO_KEY]
        self._rollout_repeated_proprio_history += (
            (proprio - proprio[:, -1:]).abs().flatten(1).amax(dim=1) <= 1.0e-8
        ).to(torch.float32).sum()
        command = proprio[:, -1, -1]
        self._rollout_command_down += (command < -0.5).to(torch.float32).sum()
        self._rollout_command_stop += (command.abs() <= 0.5).to(torch.float32).sum()
        self._rollout_command_up += (command > 0.5).to(torch.float32).sum()
        self._rollout_hidden_before_squared += hidden_before.square().sum()
        self._rollout_hidden_after_squared += hidden_after.square().sum()
        self._rollout_hidden_after_max_abs = torch.maximum(
            self._rollout_hidden_after_max_abs, hidden_after.abs().max()
        )
        self._rollout_hidden_after_saturated += (
            hidden_after.abs() >= 0.9999
        ).to(torch.float32).sum()
        self._rollout_hidden_after_exact_zero += (hidden_after == 0.0).to(
            torch.float32
        ).sum()
        self._rollout_hidden_value_count += hidden_after.numel()
        for name, values in activation_tensors.items():
            values = values.detach()
            self._rollout_activation_squared[name] += values.square().sum()
            self._rollout_activation_max_abs[name] = torch.maximum(
                self._rollout_activation_max_abs[name], values.abs().max()
            )
            self._rollout_activation_value_count[name] += values.numel()
        gru_candidate = activation_tensors["gru_candidate"].detach()
        self._rollout_gru_candidate_saturated += (
            gru_candidate.abs() >= 0.9999
        ).to(torch.float32).sum()
        self._rollout_samples += sample_count

    def _finalize_rollout_metrics(self) -> dict[str, float]:
        result = super()._finalize_rollout_metrics()
        result["rollout_fresh_depth_fraction"] = result.pop(
            "rollout_distinct_depth_frame_fraction"
        )
        hidden_count = max(self._rollout_hidden_value_count, 1)
        result["rollout_hidden_before_rms"] = float(
            (self._rollout_hidden_before_squared / hidden_count).sqrt()
        )
        result["rollout_hidden_after_rms"] = float(
            (self._rollout_hidden_after_squared / hidden_count).sqrt()
        )
        result["rollout_hidden_after_max_abs"] = float(
            self._rollout_hidden_after_max_abs
        )
        result["rollout_hidden_after_saturation_fraction"] = float(
            self._rollout_hidden_after_saturated / hidden_count
        )
        result["rollout_hidden_after_exact_zero_fraction"] = float(
            self._rollout_hidden_after_exact_zero / hidden_count
        )
        for name in self._rollout_activation_squared:
            value_count = max(self._rollout_activation_value_count[name], 1)
            result[f"rollout_{name}_rms"] = float(
                (self._rollout_activation_squared[name] / value_count).sqrt()
            )
            result[f"rollout_{name}_max_abs"] = float(
                self._rollout_activation_max_abs[name]
            )
        gru_candidate_count = max(
            self._rollout_activation_value_count["gru_candidate"], 1
        )
        result["rollout_gru_candidate_saturation_fraction"] = float(
            self._rollout_gru_candidate_saturated / gru_candidate_count
        )
        result["rollout_hidden_reset_fraction"] = float(
            self._rollout_hidden_reset_count / max(self._rollout_samples, 1)
        )
        teacher_samples = self._rollout_teacher_controlled_samples
        student_samples = self._rollout_student_controlled_samples
        result["rollout_teacher_fraction"] = float(
            teacher_samples / max(self._rollout_samples, 1)
        )
        result["rollout_student_fraction"] = float(
            student_samples / max(self._rollout_samples, 1)
        )
        result["rollout_teacher_controlled_action_rmse"] = self._split_rmse(
            self._rollout_teacher_controlled_squared_error, teacher_samples
        )
        result["rollout_student_controlled_action_rmse"] = self._split_rmse(
            self._rollout_student_controlled_squared_error, student_samples
        )
        teacher_denominator = teacher_samples.clamp_min(1.0)
        student_denominator = student_samples.clamp_min(1.0)
        result["rollout_teacher_controlled_mean_reward"] = float(
            self._rollout_teacher_controlled_reward_sum / teacher_denominator
        )
        result["rollout_student_controlled_mean_reward"] = float(
            self._rollout_student_controlled_reward_sum / student_denominator
        )
        result["rollout_teacher_controlled_done_fraction"] = float(
            self._rollout_teacher_controlled_done_count / teacher_denominator
        )
        result["rollout_student_controlled_done_fraction"] = float(
            self._rollout_student_controlled_done_count / student_denominator
        )
        teacher_done_denominator = (
            self._rollout_teacher_controlled_done_count.clamp_min(1.0)
        )
        student_done_denominator = (
            self._rollout_student_controlled_done_count.clamp_min(1.0)
        )
        for name in self._termination_names:
            result[
                f"rollout_teacher_controlled_termination_{name}_fraction_of_dones"
            ] = float(
                self._rollout_teacher_controlled_termination_counts[name]
                / teacher_done_denominator
            )
            result[
                f"rollout_student_controlled_termination_{name}_fraction_of_dones"
            ] = float(
                self._rollout_student_controlled_termination_counts[name]
                / student_done_denominator
            )
        return result

    @staticmethod
    def _module_prefixes() -> tuple[str, ...]:
        return (
            "all",
            "proprio_encoder",
            "depth_encoder",
            "gru",
            "recurrent_projection",
            "feature_mlp",
            "action_head",
        )

    def _reset_rollout_metrics(self) -> None:
        super()._reset_rollout_metrics()
        scalar = lambda: torch.zeros((), device=self.device)
        self._rollout_hidden_before_squared = scalar()
        self._rollout_hidden_after_squared = scalar()
        self._rollout_hidden_after_max_abs = scalar()
        self._rollout_hidden_after_saturated = scalar()
        self._rollout_hidden_after_exact_zero = scalar()
        self._rollout_hidden_reset_count = scalar()
        self._rollout_hidden_value_count = 0
        activation_names = (
            "proprio_feature",
            "depth_feature",
            "recurrent_input",
            "gru_candidate",
            "recurrent_feature",
            "actor_feature",
            PRIVILEGED_KEY,
        )
        self._rollout_activation_squared = {
            name: scalar() for name in activation_names
        }
        self._rollout_activation_max_abs = {
            name: scalar() for name in activation_names
        }
        self._rollout_activation_value_count = {
            name: 0 for name in activation_names
        }
        self._rollout_gru_candidate_saturated = scalar()
        self._rollout_teacher_controlled_squared_error = scalar()
        self._rollout_student_controlled_squared_error = scalar()
        self._rollout_teacher_controlled_reward_sum = scalar()
        self._rollout_student_controlled_reward_sum = scalar()
        self._rollout_teacher_controlled_done_count = scalar()
        self._rollout_student_controlled_done_count = scalar()
        self._rollout_teacher_controlled_samples = scalar()
        self._rollout_student_controlled_samples = scalar()
        self._rollout_teacher_controlled_termination_counts = {
            name: scalar() for name in self._termination_names
        }
        self._rollout_student_controlled_termination_counts = {
            name: scalar() for name in self._termination_names
        }


__all__ = ["CR1RecurrentVisualMseDagger"]
