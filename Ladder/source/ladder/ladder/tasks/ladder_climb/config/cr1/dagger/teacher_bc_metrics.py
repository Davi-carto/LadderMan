"""Streaming, sample-weighted diagnostics for teacher-distribution behavior cloning."""

import torch


class TeacherBCMetrics:
    """Accumulate detached sufficient statistics, not observations or autograd graphs.

    Actions and previous actions must use the same, unnormalized policy-action units.
    KL values are ``KL(student || teacher)`` averaged over action dimensions.
    Reference phases use zero-based frames and the total number of motion frames.
    """

    _FIELDS = (
        "action_mse", "action_mae", "action_kl_mean_part", "action_kl_std_part",
        "feature_mse", "teacher_feature_squared", "previous_action_baseline_mse",
        "teacher_action_squared", "student_action_squared", "teacher_action_change_mae",
        "teacher_std_mean", "student_std_mean",
    )
    _GROUPS = ("up", "down", "stop", "reset_history", "moving_label") + tuple(
        f"phase_{index:02d}" for index in range(10)
    )

    def __init__(self, device):
        self.device = torch.device(device)
        self.clear()

    def clear(self):
        self._totals = {}
        self._joint_squared_error = {}
        self._std_extrema = {}
        self._groups = {}
        self._counts = {}

    @torch.no_grad()
    def add(
        self, split, student_mean, teacher_mean, student_std, teacher_std,
        student_feature, teacher_feature, previous_action, command, episode_step,
        reference_frame, reference_length,
    ):
        student_mean = student_mean.detach().to(device=self.device, dtype=torch.float32)
        teacher_mean = teacher_mean.detach().to(device=self.device, dtype=torch.float32)
        student_feature = student_feature.detach().to(device=self.device, dtype=torch.float32)
        teacher_feature = teacher_feature.detach().to(device=self.device, dtype=torch.float32)
        previous_action = previous_action.detach().to(device=self.device, dtype=torch.float32)
        student_std = student_std.detach().to(self.device).float().expand_as(student_mean).clamp_min(1.0e-6)
        teacher_std = teacher_std.detach().to(self.device).float().expand_as(teacher_mean).clamp_min(1.0e-6)
        batch_size = student_mean.shape[0]
        if batch_size == 0:
            return

        action_squared_error = (student_mean - teacher_mean).square()
        action_mse = action_squared_error.mean(dim=-1)
        feature_mse = (student_feature - teacher_feature).square().mean(dim=-1)
        teacher_action_change = teacher_mean - previous_action
        baseline_mse = teacher_action_change.square().mean(dim=-1)
        teacher_action_change_mae = teacher_action_change.abs().mean(dim=-1)
        kl_mean_part = (action_squared_error / (2.0 * teacher_std.square())).mean(dim=-1)
        kl_std_part = (
            torch.log(teacher_std / student_std)
            + student_std.square() / (2.0 * teacher_std.square()) - 0.5
        ).mean(dim=-1)
        values = torch.stack(
            (
                action_mse, (student_mean - teacher_mean).abs().mean(dim=-1),
                kl_mean_part, kl_std_part, feature_mse,
                teacher_feature.square().mean(dim=-1), baseline_mse,
                teacher_mean.square().mean(dim=-1), student_mean.square().mean(dim=-1),
                teacher_action_change_mae, teacher_std.mean(dim=-1), student_std.mean(dim=-1),
            ),
            dim=-1,
        )
        totals = values.sum(dim=0, dtype=torch.float64)
        joint_squared_error = action_squared_error.sum(dim=0, dtype=torch.float64)
        std_extrema = torch.stack(
            (teacher_std.amin(), teacher_std.amax(), student_std.amin(), student_std.amax())
        )

        command = command.detach().to(self.device).reshape(-1)
        episode_step = episode_step.detach().to(self.device).reshape(-1)
        reference_frame = reference_frame.detach().to(self.device).reshape(-1).float()
        reference_length = torch.as_tensor(reference_length, device=self.device).detach().reshape(-1).float()
        phase = (10.0 * reference_frame / reference_length.clamp_min(1.0)).long().clamp(0, 9)
        direction_masks = torch.stack(
            (command > 0, command < 0, command == 0, episode_step < 9, teacher_action_change_mae > 0.05),
            dim=-1,
        )
        phase_masks = phase[:, None] == torch.arange(10, device=self.device)[None, :]
        masks = torch.cat((direction_masks, phase_masks), dim=-1)
        # One batched reduction for every subgroup; empty groups keep only their count.
        group_values = torch.stack((torch.ones_like(action_mse), action_mse, feature_mse, baseline_mse), dim=-1)
        # Avoid TF32 matmul rounding: these are numerical diagnostics, not network layers.
        groups = (masks[:, :, None] * group_values[:, None, :]).sum(dim=0, dtype=torch.float64)
        if split not in self._totals:
            self._totals[split] = totals
            self._joint_squared_error[split] = joint_squared_error
            self._std_extrema[split] = std_extrema
            self._groups[split] = groups.to(torch.float64)
            self._counts[split] = batch_size
        else:
            self._totals[split] += totals
            self._joint_squared_error[split] += joint_squared_error
            previous_extrema = self._std_extrema[split]
            self._std_extrema[split] = torch.stack(
                (
                    torch.minimum(previous_extrema[0], std_extrema[0]),
                    torch.maximum(previous_extrema[1], std_extrema[1]),
                    torch.minimum(previous_extrema[2], std_extrema[2]),
                    torch.maximum(previous_extrema[3], std_extrema[3]),
                )
            )
            self._groups[split] += groups
            self._counts[split] += batch_size

    def summary(self):
        """Return JSON-compatible scalars without clearing the running statistics."""
        result = {}
        for split, totals in self._totals.items():
            count = self._counts[split]
            means = dict(zip(self._FIELDS, (totals / count).cpu().tolist()))
            means["count"] = float(count)
            means["action_rmse"] = means["action_mse"] ** 0.5
            means["action_kl"] = means["action_kl_mean_part"] + means["action_kl_std_part"]
            means["teacher_feature_rms"] = means.pop("teacher_feature_squared") ** 0.5
            means["teacher_action_rms"] = means.pop("teacher_action_squared") ** 0.5
            means["student_action_rms"] = means.pop("student_action_squared") ** 0.5
            baseline = means["previous_action_baseline_mse"]
            # A zero-error persistence baseline has no meaningful relative improvement.
            if baseline > 0.0:
                means["previous_action_baseline_skill"] = 1.0 - means["action_mse"] / baseline
            extrema = self._std_extrema[split].cpu().tolist()
            means.update(zip(("teacher_std_min", "teacher_std_max", "student_std_min", "student_std_max"), extrema))
            result.update({f"{split}/{key}": value for key, value in means.items()})
            for index, value in enumerate((self._joint_squared_error[split] / count).sqrt().cpu().tolist()):
                result[f"{split}/joint_{index:02d}/rmse"] = value
            for group, row in zip(self._GROUPS, self._groups[split].cpu().tolist()):
                result[f"{split}/{group}/count"] = row[0]
                if row[0] > 0:
                    for key, total in zip(("action_mse", "feature_mse", "previous_action_baseline_mse"), row[1:]):
                        result[f"{split}/{group}/{key}"] = total / row[0]
        return result
