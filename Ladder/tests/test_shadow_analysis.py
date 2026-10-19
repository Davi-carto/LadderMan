"""Simulator-free checks of first-episode shadow diagnostic summaries."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


_PATH = Path(__file__).parents[1] / "source/ladder/ladder/tasks/ladder_climb/config/cr1/dagger/shadow_analysis.py"
_SPEC = importlib.util.spec_from_file_location("ladder_shadow_analysis_unit", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
summarize_shadow_trace = _MODULE.summarize_shadow_trace


def _trace(steps=5, envs=2):
    trace = {
        "valid": torch.ones(steps, envs, dtype=torch.bool),
        "done": torch.zeros(steps, envs, dtype=torch.bool),
        "reference_frame": torch.arange(steps)[:, None].expand(steps, envs) + 100,
        "student_action": torch.zeros(steps, envs, 25),
        "teacher_action": torch.zeros(steps, envs, 25),
        "student_feature": torch.zeros(steps, envs, 256),
        "teacher_feature": torch.ones(steps, envs, 256),
        "delay": torch.zeros(steps, envs, dtype=torch.long),
        "termination/motion_complete": torch.zeros(steps, envs, dtype=torch.bool),
        "termination/root_pose": torch.zeros(steps, envs, dtype=torch.bool),
    }
    for timing in ("pre", "post"):
        for quantity in ("anchor_pos_error", "anchor_rot_error", "root_tilt"):
            trace[f"{timing}_{quantity}"] = torch.zeros(steps, envs)
        trace[f"{timing}_root_ang_vel"] = torch.zeros(steps, envs, 3)
        trace[f"{timing}_required_contact"] = torch.zeros(steps, envs, 4, dtype=torch.bool)
        trace[f"{timing}_physical_contact"] = torch.zeros(steps, envs, 4, dtype=torch.bool)
    return trace


def test_done_excludes_following_steps_and_aggregate_uses_sample_weighting():
    trace = _trace()
    trace["done"][1, 0] = True
    trace["termination/root_pose"][1, 0] = True
    trace["valid"][2:, 0] = False
    trace["student_action"][:, 0] = 1.0
    trace["student_action"][2:, 0] = 1000.0
    trace["student_action"][:, 1] = 3.0
    result = summarize_shadow_trace(trace, 0.02, "student")
    aggregate = result["aggregate"]
    assert aggregate["action_metrics"]["count"] == 7
    assert aggregate["action_metrics"]["action_rmse"] == pytest.approx(((2 + 5 * 9) / 7) ** 0.5)
    assert aggregate["action_metrics"]["action_mae"] == pytest.approx(17 / 7)
    assert aggregate["action_metrics"]["per_joint"][0]["max_abs"] == 3
    assert aggregate["action_metrics"]["feature_mse"] == 1
    assert aggregate["failed_episodes"] == aggregate["censored_episodes"] == 1
    assert aggregate["success_fraction_of_terminated"] == 0
    assert result["episodes"][0]["length_steps"] == 2
    assert result["episodes"][0]["terminal_action_step"] == 1
    assert result["episodes"][0]["end_pre_action_reference_frame"] == 101
    assert result["episodes"][1]["outcome"] == "censored"
    assert result["episodes"][1]["success"] is None
    json.dumps(result, allow_nan=False)
    trace["valid"][:] = True
    assert summarize_shadow_trace(trace, 0.02, "student") == result


def test_success_marker_with_concurrent_failure_is_not_success():
    trace = _trace(steps=1, envs=3)
    trace["done"][:] = True
    trace["termination/motion_complete"][:] = True
    trace["termination/root_pose"][0, 1] = True
    trace["termination/time_out"] = torch.tensor([[False, False, True]])
    result = summarize_shadow_trace(trace, 0.02, "teacher")
    assert result["aggregate"]["motion_complete_markers"] == 3
    assert result["aggregate"]["successful_episodes"] == 2
    assert result["aggregate"]["failed_episodes"] == 1
    assert result["episodes"][1]["failure_causes"] == ["root_pose"]
    assert result["episodes"][2]["success"] is True


def test_threshold_markers_capture_order_and_pre_vs_post_action_time():
    trace = _trace(envs=1)
    trace["student_action"][1:, 0] = 0.06
    trace["student_action"][3:, 0] = 0.11
    trace["post_anchor_pos_error"][2:, 0] = 0.06
    trace["post_anchor_rot_error"][3:, 0] = 0.25
    trace["post_root_tilt"][0, 0] = 0.3
    trace["post_required_contact"][2:, 0, 1] = True
    trace["post_physical_contact"][2, 0, 1] = True
    result = summarize_shadow_trace(trace, 0.02, "student")
    events = result["episodes"][0]["first_events"]
    assert events["action_rmse_over_0p05"]["action_step"] == 1
    assert events["action_rmse_over_0p05"]["time_s"] == pytest.approx(0.02)
    assert events["action_rmse_over_0p10"]["action_step"] == 3
    assert events["post_anchor_pos_error_over_0p05"]["action_step"] == 2
    assert events["post_anchor_pos_error_over_0p05"]["time_s"] == pytest.approx(0.06)
    assert events["post_anchor_pos_error_over_0p05"]["pre_action_reference_frame"] == 102
    assert events["post_anchor_pos_error_over_0p10"] is None
    assert events["post_required_contact_miss"]["action_step"] == 3
    assert events["post_root_tilt_over_0p20"]["action_step"] == 0
    assert result["episodes"][0]["post_required_contact_miss_steps"] == 2


def test_windows_are_exact_and_last_window_ends_at_last_valid_sample():
    trace = _trace(steps=240, envs=1)
    trace["student_action"][:] = torch.arange(240)[:, None, None]
    trace["valid"][223:] = False
    result = summarize_shadow_trace(trace, 0.02, "student")
    windows = result["episodes"][0]["windows"]
    for name, start, stop in (("steps_1_9", 0, 9), ("steps_10_50", 9, 50),
                              ("steps_51_100", 50, 100), ("steps_101_200", 100, 200),
                              ("steps_201_plus", 200, 223), ("last_25_steps", 198, 223)):
        expected = torch.arange(start, stop, dtype=torch.float64)
        assert windows[name]["count"] == stop - start
        assert windows[name]["action_rmse"] == pytest.approx(expected.square().mean().sqrt().item())
        assert windows[name]["action_mae"] == expected.mean().item()


def test_empty_groups_and_no_data_do_not_emit_nan_or_false_successes():
    trace = _trace(steps=1)
    trace["valid"][:, 1] = False
    result = summarize_shadow_trace(trace, 0.02, "student")
    assert result["aggregate"]["empty_environments"] == 1
    assert result["aggregate"]["censored_episodes"] == 1
    assert result["aggregate"]["success_fraction_of_terminated"] is None
    assert result["episodes"][1]["outcome"] == "no_data"
    assert result["episodes"][1]["action_metrics"]["action_rmse"] is None
    assert result["episodes"][0]["windows"]["steps_10_50"]["count"] == 0
    json.dumps(result, allow_nan=False)
    trace["valid"][:] = False
    assert summarize_shadow_trace(trace, 0.02, "student")["aggregate"]["action_metrics"]["count"] == 0


@pytest.mark.parametrize("step_dt", [0, -0.01, float("nan"), float("inf")])
def test_invalid_step_dt_rejected(step_dt):
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_shadow_trace(_trace(), step_dt, "student")
