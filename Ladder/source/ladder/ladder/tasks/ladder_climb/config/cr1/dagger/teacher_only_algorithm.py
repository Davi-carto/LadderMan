"""Teacher-distribution BC control experiment; no student-controlled training steps."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .pure_algorithm import CR1PureVisualDagger
from .pure_storage import PureVisualDaggerTransition
from .student_teacher import TEACHER_KEY
from .teacher_bc_metrics import TeacherBCMetrics
from .visual_model import DEPTH_KEY, PROPRIO_KEY


ROLLOUT_MODE = "teacher_only_bc_v1"


class CR1TeacherOnlyVisualBC(CR1PureVisualDagger):
    """Keep the original student/loss; isolate its fit on held-out teacher trajectories.

    A fixed suffix of environment IDs never enters the optimizer. Reference and
    termination metadata are used only for diagnostics, never as student inputs.
    """

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env, cfg: dict, device: str):
        algorithm_cfg = cfg["algorithm"]
        if algorithm_cfg["teacher_rollout_fraction"] != 1.0:
            raise ValueError("Teacher-only BC requires teacher_rollout_fraction=1.0")
        validation_envs = int(algorithm_cfg["validation_num_envs"])
        if not 1 <= validation_envs < env.num_envs:
            raise ValueError("validation_num_envs must be positive and smaller than num_envs")
        if env.num_actions != 25:
            raise ValueError("Teacher-only CR1 diagnostics expect 25 actions and the 9x85 input contract")
        algorithm = super().construct_algorithm(obs, env, cfg, device)
        algorithm.storage.num_training_envs = env.num_envs - validation_envs
        algorithm._env = env
        algorithm._command = env.unwrapped.command_manager.get_term("motion")
        algorithm._metrics = TeacherBCMetrics(device)
        algorithm._diagnostic_log = None
        algorithm._snapshot_interval = int(algorithm_cfg["diagnostic_snapshot_interval"])
        if algorithm._snapshot_interval < 1:
            raise ValueError("diagnostic_snapshot_interval must be positive")
        algorithm._episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        algorithm._episode_ids = torch.zeros_like(algorithm._episode_steps)
        algorithm._env_ids = torch.arange(env.num_envs, device=device)
        train_end = algorithm.storage.num_training_envs
        algorithm._splits = {"train": slice(0, train_end), "validation": slice(train_end, env.num_envs)}
        algorithm._snapshot_ids = torch.tensor(
            list(range(min(2, train_end))) + list(range(train_end, min(train_end + 2, env.num_envs))),
            device=device,
        )
        algorithm._termination_names = list(env.unwrapped.termination_manager.active_terms)
        # These are existing command-manager metrics, read before env.step/reset.
        metric_names = (
            "error_joint_pos", "error_anchor_pos", "error_anchor_rot",
            "required_physical_contact_fraction", "required_target_contact_fraction",
            "mean_required_contact_error", "reference_anchor_speed", "robot_anchor_speed",
            "stationary_during_reference_motion_fraction", "reference_completion_fraction",
            "anchor_progress_ratio", "reference_anchor_displacement", "robot_anchor_displacement",
            "fixed_start_episode", "secondary_fixed_start_episode", "uniform_start_episode",
            "adaptive_start_episode", "reset_valid_fraction", "ladder_inclination_deg", "ladder_rung_spacing",
            "nominal_ladder_geometry_episode",
        )
        algorithm._metric_names = [name for name in metric_names if name in algorithm._command.metrics]
        algorithm._clear_diagnostics()
        return algorithm

    def configure_diagnostics(self, log_dir, runner_cfg, motion_file, teacher_checkpoint, logger, resume_path=None):
        from .teacher_bc_logging import TeacherBCLog

        self._diagnostic_log = TeacherBCLog(
            log_dir, env=self._env, runner_cfg=runner_cfg, motion_file=motion_file,
            teacher_checkpoint=teacher_checkpoint, logger=logger,
            metadata={
                "rollout_mode": ROLLOUT_MODE,
                "student_network": str(self.policy), "teacher_network": str(self.teacher),
                "teacher_std": self.teacher_std.detach().cpu().tolist(),
                "train_env_ids": [0, self.storage.num_training_envs - 1],
                "validation_env_ids": [self.storage.num_training_envs, self.storage.num_envs - 1],
                "snapshot_env_ids": self._snapshot_ids.cpu().tolist(),
                "environment_metric_names": self._metric_names,
                "termination_names": self._termination_names,
                "resume_checkpoint": str(resume_path) if resume_path else None,
                "initial_policy_updates": self.update_iteration,
                "observation_timing": "Pre-action, including teacher labels and frame; env.step auto-resets.",
                "metric_timing": "Existing command metrics sampled pre-action; command metrics can lag reference cursor.",
                "validation_scope": "Held-out env trajectories of the same motion, sharing the adaptive sampler; not unseen-motion generalization.",
                "iteration_convention": "Zero-based optimizer-update index; rollout metrics/predictions are PRE-update, model_i is POST-update.",
                "episode_length_convention": "Observed steps tracked separately from randomized initial environment timeout counter.",
            },
        )

    def _clear_diagnostics(self):
        self._metrics.clear()
        self._episodes = []
        self._snapshots = []
        self._environment_sums = {
            split: torch.zeros(len(self._metric_names), device=self.device) for split in self._splits
        }
        self._rollout_steps = 0

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        if self._pending is not None:
            raise RuntimeError("process_env_step must follow every act call")
        self._validate_observations(obs)
        actor_obs = TensorDict(
            {PROPRIO_KEY: obs[PROPRIO_KEY], DEPTH_KEY: obs[DEPTH_KEY]},
            batch_size=obs.batch_size, device=obs.device,
        )
        student_mean, student_feature = self.policy.action_mean_and_feature(actor_obs)
        teacher_feature = self.teacher[:-1](obs[TEACHER_KEY])
        teacher_mean = self.teacher[-1](teacher_feature)
        latest_proprio = obs[PROPRIO_KEY][:, -1]
        frame = self._command.time_steps.clone()
        environment_metrics = (
            torch.stack([self._command.metrics[name] for name in self._metric_names], dim=-1)
            if self._metric_names else torch.empty(self.storage.num_envs, 0, device=self.device)
        )
        for split, selected in self._splits.items():
            self._metrics.add(
                split, student_mean[selected], teacher_mean[selected], self.policy.output_std,
                self.teacher_std, student_feature[selected], teacher_feature[selected],
                latest_proprio[selected, 59:84], latest_proprio[selected, 84],
                self._episode_steps[selected], frame[selected], self._command.motion.time_step_total,
            )
            self._environment_sums[split] += environment_metrics[selected].sum(dim=0)
        # Clone before env.step: both the reference cursor and episode state may reset.
        self._pre = {
            "env_id": self._env_ids,
            "validation": self._env_ids >= self.storage.num_training_envs,
            "episode_id": self._episode_ids.clone(),
            "start_frame": self._command._episode_start_time_steps.clone(),
            "last_pre_action_frame": frame,
            "observed_length": self._episode_steps + 1,
            "environment_length": self._env.episode_length_buf.clone() + 1,
            "command": latest_proprio[:, 84].clone(),
        }
        self._pre.update({f"pre_action_metrics/{name}": environment_metrics[:, i]
                          for i, name in enumerate(self._metric_names)})
        if self._diagnostic_log is not None and self.update_iteration % self._snapshot_interval == 0:
            selected = self._snapshot_ids
            sample = {
                **{name: value[selected].clone() for name, value in self._pre.items()},
                PROPRIO_KEY: obs[PROPRIO_KEY][selected].clone(),
                DEPTH_KEY: obs[DEPTH_KEY][selected].clone(),
                TEACHER_KEY: obs[TEACHER_KEY][selected].clone(),
                "student_action_mean": student_mean[selected].clone(),
                "teacher_action_mean": teacher_mean[selected].clone(),
                "student_feature": student_feature[selected].clone(),
                "teacher_feature": teacher_feature[selected].clone(),
                "reference_contact": self._command.contact_indicator[selected].clone(),
                "reference_contact_rung": self._command.contact_rung_id[selected].clone(),
            }
            self._snapshots.append(sample)
        self._pending = PureVisualDaggerTransition(
            obs[PROPRIO_KEY].detach(), obs[DEPTH_KEY].detach(), teacher_mean.detach(), teacher_feature.detach(),
        )
        self._teacher_action_count += teacher_mean.shape[0]
        self._rollout_action_count += teacher_mean.shape[0]
        self._rollout_steps += 1
        # Never use student outputs for behavior selection, including after resume.
        return teacher_mean

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        super().process_env_step(obs, rewards, dones, extras)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        reasons = {name: self._env.unwrapped.termination_manager.get_term(name)
                   for name in self._termination_names}
        if done_ids.numel():
            row = {name: value[done_ids].clone() for name, value in self._pre.items()}
            row.update({f"termination/{name}": value[done_ids].clone() for name, value in reasons.items()})
            self._episodes.append(row)
        if self._snapshots and len(self._snapshots) == self._rollout_steps:
            self._snapshots[-1]["done"] = dones[self._snapshot_ids].clone()
            self._snapshots[-1].update({f"termination/{name}": value[self._snapshot_ids].clone()
                                       for name, value in reasons.items()})
        self._episode_steps += 1
        self._episode_steps[dones] = 0
        self._episode_ids[dones] += 1

    def update(self) -> dict[str, float]:
        iteration = self.update_iteration
        metrics = self._metrics.summary()
        teacher_std = self.teacher_std.detach().clone()
        student_std = self.policy.output_std.detach().clone()
        # Parent uses storage.num_training_envs; the held-out suffix is never sampled.
        losses = super().update()
        metrics.update({f"optimizer/{name}": value for name, value in losses.items()})
        metrics["optimizer/learning_rate"] = self.learning_rate
        metrics["optimizer/action_kl_weight"] = self.action_kl_weight
        metrics["optimizer/feature_weight"] = self.feature_weight
        for name, module in (("depth_encoder", self.policy.depth_encoder),
                             ("feature_mlp", self.policy.feature_mlp), ("action_head", self.policy.action_head)):
            gradients = [parameter.grad.detach().square().sum() for parameter in module.parameters()
                         if parameter.grad is not None]
            metrics[f"optimizer/last_clipped_grad_norm/{name}"] = float(torch.stack(gradients).sum().sqrt())
        metrics["optimizer/last_clipped_grad_norm/log_std"] = float(self.policy.log_std.grad.detach().norm())
        episode_columns = ({name: torch.cat([row[name] for row in self._episodes])
                            for name in self._episodes[0]} if self._episodes else {})
        for split, selected in self._splits.items():
            means = self._environment_sums[split] / ((selected.stop - selected.start) * self._rollout_steps)
            metrics.update({f"teacher_rollout/{split}/{name}": value
                            for name, value in zip(self._metric_names, means.cpu().tolist())})
            mask = (episode_columns["validation"] == (split == "validation")) if episode_columns else None
            count = int(mask.sum()) if mask is not None else 0
            metrics[f"teacher_episodes/{split}/count"] = count
            if count:
                metrics[f"teacher_episodes/{split}/observed_length_mean"] = float(
                    episode_columns["observed_length"][mask].float().mean())
                for name in self._termination_names:
                    metrics[f"teacher_episodes/{split}/{name}_fraction"] = float(
                        episode_columns[f"termination/{name}"][mask].float().mean())
        if self._diagnostic_log is not None:
            self._diagnostic_log.write_iteration(iteration, metrics)
            if episode_columns:
                self._diagnostic_log.write_episodes(iteration, episode_columns)
            if self._snapshots:
                payload = {name: torch.stack([row[name] for row in self._snapshots]).cpu()
                           for name in self._snapshots[0]}
                payload.update({"teacher_std": teacher_std.cpu(), "student_std": student_std.cpu(),
                                "policy_updates_at_collection": iteration, "rollout_mode": ROLLOUT_MODE})
                self._diagnostic_log.save_snapshot(iteration, payload)
        self._clear_diagnostics()
        losses["teacher_validation_action_rmse"] = metrics["validation/action_rmse"]
        losses["teacher_validation_feature_mse"] = metrics["validation/feature_mse"]
        return losses

    def save(self) -> dict:
        payload = super().save()
        payload.update({"rollout_mode": ROLLOUT_MODE, "validation_num_envs": self.storage.num_envs - self.storage.num_training_envs})
        return payload

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if loaded_dict.get("rollout_mode") != ROLLOUT_MODE:
            raise RuntimeError("Teacher-only BC resume requires a teacher_only_bc_v1 checkpoint; start a fresh control experiment")
        return super().load(loaded_dict, load_cfg, strict)


__all__ = ["CR1TeacherOnlyVisualBC"]
