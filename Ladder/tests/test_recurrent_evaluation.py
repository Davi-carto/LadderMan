"""Evaluation statistics must never count autoreset episodes or false successes."""

import importlib.util
from pathlib import Path

import pytest
import torch


_PATH = Path(__file__).parents[1] / "source/ladder/ladder/tasks/ladder_climb/config/cr1/dagger/recurrent_evaluation.py"
_SPEC = importlib.util.spec_from_file_location("recurrent_evaluation_test_module", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
FirstEpisodeAccumulator = _MODULE.FirstEpisodeAccumulator


def state(frames, heights):
    return {
        "reference_frame": torch.tensor(frames),
        "root_pos": torch.tensor([[0.0, 0.0, height] for height in heights]),
        "root_tilt_rad": torch.zeros(len(frames)),
    }


def test_first_episode_freezes_rewards_and_true_terminal_state_after_autoreset():
    accounting = FirstEpisodeAccumulator(state([0, 0], [1.0, 1.0]), 0.02)
    accounting.add(state([1, 1], [0.4, 1.2]), torch.tensor([-1.0, 2.0]),
                   torch.tensor([True, False]), {"root_pose": torch.tensor([True, False])}, torch.tensor([4.0, 1.0]))
    accounting.add(state([0, 2], [99.0, 1.3]), torch.tensor([1000.0, 3.0]),
                   torch.tensor([True, True]), {"motion_complete": torch.tensor([True, True])}, torch.tensor([100.0, 9.0]))
    episodes, summary = accounting.report()
    assert episodes[0]["length_steps"] == 1
    assert episodes[0]["reward_sum"] == -1.0
    assert episodes[0]["termination_terms"] == ["root_pose"]
    assert episodes[0]["last_state_before_any_reset"]["root_pos"][2] == pytest.approx(0.4)
    assert episodes[0]["root_z_max_m"] == 1.0
    assert episodes[1]["length_steps"] == 2
    assert episodes[1]["reward_sum"] == 5.0
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["mean_step_reward"] == pytest.approx(4.0 / 3.0)
    assert summary["action_mean_rmse_to_teacher"] == pytest.approx((14.0 / 3.0) ** 0.5)


def test_simultaneous_motion_complete_and_failure_is_not_success():
    accounting = FirstEpisodeAccumulator(state([0, 0, 0], [1.0] * 3), 0.02)
    accounting.add(state([3060] * 3, [1.0] * 3), torch.zeros(3), torch.ones(3, dtype=torch.bool),
                   {"motion_complete": torch.tensor([True, True, False]),
                    "required_contact": torch.tensor([True, False, False]),
                    "time_out": torch.tensor([False, True, True])}, torch.zeros(3))
    episodes, summary = accounting.report()
    assert [episode["success"] for episode in episodes] == [False, True, False]
    assert summary["success_count"] == 1
    assert summary["termination_counts"]["motion_complete"] == 2


def test_unfinished_episodes_are_censored_in_all_episode_denominator():
    accounting = FirstEpisodeAccumulator(state([0, 0], [1.0, 1.0]), 0.02)
    accounting.add(state([3060, 300], [1.0, 1.4]), torch.ones(2), torch.tensor([True, False]),
                   {"motion_complete": torch.tensor([True, False])}, torch.zeros(2))
    episodes, summary = accounting.report()
    assert episodes[1]["outcome"] == "censored"
    assert not episodes[1]["success"]
    assert not episodes[1]["last_state_is_terminal"]
    assert summary["censored_count"] == 1
    assert summary["success_fraction_all_episodes"] == 0.5
    assert summary["success_fraction_completed_episodes"] == 1.0


def test_ascent_marker_requires_live_boundary_crossing_without_failure():
    accounting = FirstEpisodeAccumulator(state([0, 0, 1550], [1.0, 1.0, 3.0]), 0.02, 1550)
    accounting.add(state([1550, 1550, 1600], [3.0, 2.5, 2.0]), torch.ones(3), torch.tensor([False, True, False]),
                   {"body_local_pose": torch.tensor([False, True, False])}, torch.zeros(3))
    episodes, summary = accounting.report()
    assert [row["reached_downward_start_without_failure"] for row in episodes] == [True, False, None]
    assert episodes[0]["root_z_max_increase_m"] == 2.0
    assert episodes[2]["root_z_max_decrease_m"] == 1.0
    assert summary["ascent_eligible_episodes"] == 2
    assert summary["ascent_reached_downward_start_count"] == 1
