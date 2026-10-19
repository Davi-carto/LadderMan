import importlib.util
import json
from pathlib import Path

import pytest
import torch


_PATH = Path(__file__).resolve().parents[1] / "source/ladder/ladder/tasks/ladder_climb/config/cr1/dagger/teacher_bc_metrics.py"
_SPEC = importlib.util.spec_from_file_location("teacher_bc_metrics_unit", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
TeacherBCMetrics = _MODULE.TeacherBCMetrics


def _batch(student, teacher=None, **overrides):
    student = torch.as_tensor(student, dtype=torch.float32)
    count, actions = student.shape
    teacher = torch.zeros_like(student) if teacher is None else torch.as_tensor(teacher, dtype=torch.float32)
    args = {
        "student_mean": student,
        "teacher_mean": teacher,
        "student_std": torch.ones(actions),
        "teacher_std": torch.ones(actions),
        "student_feature": torch.zeros(count, 3),
        "teacher_feature": torch.ones(count, 3),
        "previous_action": torch.ones_like(student),
        "command": torch.ones(count),
        "episode_step": torch.full((count,), 20),
        "reference_frame": torch.zeros(count),
        "reference_length": torch.tensor(100),
    }
    args.update(overrides)
    return args


def test_kl_decomposition_matches_diagonal_gaussian_and_feature_units():
    metrics = TeacherBCMetrics("cpu")
    batch = _batch([[1, 3], [2, 4]], teacher_std=torch.tensor([2.0, 3.0]), student_std=torch.tensor([0.7, 1.2]))
    metrics.add("train", **batch)
    summary = metrics.summary()
    teacher = torch.distributions.Normal(batch["teacher_mean"], batch["teacher_std"])
    student = torch.distributions.Normal(batch["student_mean"], batch["student_std"])
    expected_kl = torch.distributions.kl_divergence(student, teacher).mean().item()
    expected_mean_part = (batch["student_mean"].square() / (2 * batch["teacher_std"].square())).mean().item()
    assert summary["train/action_kl"] == pytest.approx(expected_kl)
    assert summary["train/action_kl_mean_part"] == pytest.approx(expected_mean_part)
    assert summary["train/action_kl_std_part"] == pytest.approx(expected_kl - expected_mean_part)
    assert summary["train/feature_mse"] == 1
    assert summary["train/teacher_feature_rms"] == 1
    assert summary["train/teacher_std_mean"] == 2.5
    assert summary["train/teacher_std_min"] == 2
    assert summary["train/teacher_std_max"] == 3


def test_nonuniform_batches_are_sample_weighted_and_splits_independent():
    metrics = TeacherBCMetrics("cpu")
    metrics.add("train", **_batch([[1, 2]]))
    metrics.add("train", **_batch([[3, 4], [5, 6], [7, 8]]))
    metrics.add("validation", **_batch([[10, 10]]))
    result = metrics.summary()
    data = torch.arange(1, 9, dtype=torch.float32).reshape(4, 2)
    assert result["train/count"] == 4
    assert result["train/action_mse"] == pytest.approx(data.square().mean().item())
    assert result["train/action_rmse"] == pytest.approx(data.square().mean().sqrt().item())
    assert result["train/action_mae"] == 4.5
    assert result["train/joint_00/rmse"] == pytest.approx(data[:, 0].square().mean().sqrt().item())
    assert result["train/joint_01/rmse"] == pytest.approx(data[:, 1].square().mean().sqrt().item())
    assert result["validation/count"] == 1
    assert result["validation/action_mse"] == 100
    assert metrics.summary() == result
    json.dumps(result, allow_nan=False)


def test_subgroups_have_counts_and_empty_buckets_do_not_report_zero_loss():
    metrics = TeacherBCMetrics("cpu")
    metrics.add(
        "validation", **_batch(
            [[1], [2], [3]], command=torch.tensor([1, -1, 1]),
            episode_step=torch.tensor([0, 9, 8]), reference_frame=torch.tensor([0, 50, 99]),
            previous_action=torch.tensor([[0.0], [0.03], [1.0]]),
        ),
    )
    result = metrics.summary()
    assert result["validation/up/count"] == 2
    assert result["validation/up/action_mse"] == 5
    assert result["validation/down/count"] == 1
    assert result["validation/stop/count"] == 0
    assert "validation/stop/action_mse" not in result
    assert result["validation/reset_history/count"] == 2
    assert result["validation/moving_label/count"] == 1
    for index in (0, 5, 9):
        assert result[f"validation/phase_{index:02d}/count"] == 1
    assert result["validation/phase_01/count"] == 0
    assert "validation/phase_01/action_mse" not in result


def test_previous_action_baseline_reports_skill_but_omits_undefined_zero_baseline():
    metrics = TeacherBCMetrics("cpu")
    metrics.add("train", **_batch([[0.5, 0.5]]))
    result = metrics.summary()
    assert result["train/previous_action_baseline_mse"] == 1
    assert result["train/previous_action_baseline_skill"] == 0.75
    metrics.clear()
    assert metrics.summary() == {}
    metrics.add("train", **_batch([[1.0]], previous_action=torch.zeros(1, 1)))
    result = metrics.summary()
    assert result["train/previous_action_baseline_mse"] == 0
    assert "train/previous_action_baseline_skill" not in result
    json.dumps(result, allow_nan=False)


def test_statistics_never_retain_gradients_or_mutate_inputs():
    metrics = TeacherBCMetrics("cpu")
    batch = _batch([[1.0, 2.0]], student_feature=torch.ones(1, 3, requires_grad=True))
    batch["student_mean"].requires_grad_(True)
    batch["student_std"].requires_grad_(True)
    original = batch["student_mean"].detach().clone()
    metrics.add("train", **batch)
    metrics.add("train", **batch)
    for attribute in ("_totals", "_joint_squared_error", "_std_extrema", "_groups"):
        for value in getattr(metrics, attribute).values():
            assert not value.requires_grad
            assert value.grad_fn is None
    torch.testing.assert_close(batch["student_mean"], original)
    assert batch["student_mean"].grad is None
    assert metrics.summary()["train/count"] == 2
