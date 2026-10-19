"""CPU checks for student-only GAE and aligned recurrent PPO/DAgger batches."""

import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


_PACKAGE_NAME = "cr1_ppo_dagger_storage_test_package"
_PACKAGE = ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(
    Path(__file__).resolve().parents[1]
    / "source/ladder/ladder/tasks/ladder_climb/config/cr1/dagger"
)]
sys.modules.setdefault(_PACKAGE_NAME, _PACKAGE)
_storage = importlib.import_module(f"{_PACKAGE_NAME}.recurrent_visual_ppo_storage")
_base = importlib.import_module(f"{_PACKAGE_NAME}.recurrent_visual_mse_storage")
Storage = _storage.RecurrentVisualPpoDaggerStorage
Transition = _base.RecurrentVisualMseDaggerTransition


def _add(storage, teacher, values, rewards, dones, time_outs=None):
    step = storage.step
    num_envs = storage.num_envs
    # Codes distinguish every env, time, and action for sequence checks.
    codes = torch.arange(num_envs, dtype=torch.float32) + step * 10
    actions = codes[:, None] + torch.arange(2, dtype=torch.float32)[None] / 10
    storage.add_rl(
        critic_observation=codes[:, None].expand(-1, 3),
        actions=actions,
        old_action_mean=actions + 1,
        old_action_std=torch.ones_like(actions) * 0.1,
        old_action_log_prob=codes + 2,
        values=torch.as_tensor(values, dtype=torch.float32),
        rewards=torch.as_tensor(rewards, dtype=torch.float32),
        time_outs=torch.as_tensor(time_outs if time_outs is not None else [False] * num_envs),
    )
    assert storage.step == step
    storage.add(
        Transition(
            proprio_history=codes[:, None, None].expand(-1, 25, 85),
            depth=codes[:, None, None, None].expand(-1, 1, 60, 80),
            depth_updated=torch.ones(num_envs, dtype=torch.bool),
            teacher_action_mean=actions + 3,
            hidden_before=codes[None, :, None].expand(1, -1, 256),
            teacher_controlled=torch.tensor(teacher),
        ),
        torch.tensor(dones),
    )


def _example_rollout(teacher_scale=1.0):
    storage = Storage(4, 3, 2, 3, "cpu")
    for step, student_value in enumerate([1.0, 2.0, 30.0, 4.0]):
        _add(
            storage,
            teacher=[True, False, False],
            values=[teacher_scale * 100, student_value, student_value],
            rewards=[teacher_scale * -100, step + 1, step + 1],
            dones=[False, step == 1, step == 1],
            time_outs=[False, step == 1, False],
        )
    storage.compute_returns(torch.tensor([teacher_scale * 1000, 5, 5]), 0.9, 0.8)
    return storage


def test_gae_timeout_bootstrap_and_true_done_cut_next_episode():
    storage = _example_rollout()
    # Env 1 times out at t=1: r'=2+.9*2=3.8; env 2 truly terminates.
    # Their large next-episode V(s_2)=30 must not affect either t=1 target.
    expected_returns = torch.tensor([
        [0.0, 4.096, 2.8],
        [0.0, 3.8, 2.0],
        [0.0, 9.84, 9.84],
        [0.0, 8.5, 8.5],
    ])
    torch.testing.assert_close(storage.returns, expected_returns)
    unnormalized = expected_returns[:, 1:] - storage.values[:, 1:]
    expected_advantages = (unnormalized - unnormalized.mean()) / (
        unnormalized.std(unbiased=False) + 1.0e-8
    )
    torch.testing.assert_close(storage.advantages[:, 1:], expected_advantages)
    assert torch.count_nonzero(storage.advantages[:, 0]) == 0
    # Timeout adjustment must not overwrite the raw environment reward.
    torch.testing.assert_close(storage.rewards[:, 1], torch.arange(1.0, 5.0))


def test_teacher_rewards_and_values_cannot_contaminate_student_targets():
    baseline = _example_rollout()
    changed = _example_rollout(teacher_scale=1.0e8)
    torch.testing.assert_close(baseline.returns, changed.returns)
    torch.testing.assert_close(baseline.advantages, changed.advantages)


@pytest.mark.parametrize("teacher", [[False], [True]])
def test_single_student_sample_and_all_teacher_rollout_are_finite(teacher):
    storage = Storage(1, 1, 2, 3, "cpu")
    _add(storage, teacher, values=[2], rewards=[1], dones=[False])
    storage.compute_returns(torch.tensor([[3.0]]), 0.9, 0.8)
    assert torch.isfinite(storage.advantages).all()
    assert torch.count_nonzero(storage.advantages) == 0
    torch.testing.assert_close(
        storage.returns, torch.tensor([[0.0 if teacher[0] else 3.7]])
    )


def test_controller_switch_inside_rollout_is_rejected():
    storage = Storage(2, 2, 2, 3, "cpu")
    _add(storage, [True, False], [0, 0], [1, 1], [True, True])
    _add(storage, [False, True], [0, 0], [1, 1], [False, False])
    with pytest.raises(ValueError, match="ownership must remain fixed"):
        storage.compute_returns(torch.zeros(2), 0.9, 0.8)


def test_batches_cover_all_envs_and_preserve_visual_rl_sequence_alignment():
    storage = Storage(3, 5, 2, 3, "cpu")
    for step in range(3):
        _add(storage, [True, False, False, False, False], [step] * 5,
             [step + 1] * 5, [step == 0, False, False, False, False])
    storage.compute_returns(torch.ones(5), 0.9, 0.8)
    batches = list(storage.batches(num_mini_batches=2, num_epochs=2))
    assert len(batches) == 4
    for epoch in range(2):
        visited = torch.cat([batch["env_ids"] for batch in batches[epoch * 2:epoch * 2 + 2]])
        assert sorted(visited.tolist()) == list(range(5))
        for batch in batches[epoch * 2:epoch * 2 + 2]:
            ids = batch["env_ids"]
            assert batch["proprio_history"].shape == (3, len(ids), 25, 85)
            for name in storage._RL_FIELDS:
                torch.testing.assert_close(batch[name], getattr(storage, name)[:, ids])
            for name in ("depth", "depth_updated", "teacher_action_mean", "dones", "teacher_controlled"):
                torch.testing.assert_close(batch[name], getattr(storage, name)[:, ids])
            torch.testing.assert_close(batch["initial_hidden"], storage.initial_hidden[:, ids])
            for step in range(3):
                codes = ids.float() + step * 10
                torch.testing.assert_close(batch["proprio_history"][step, :, -1, 0], codes)
                torch.testing.assert_close(batch["actions"][step, :, 0], codes)
                torch.testing.assert_close(batch["critic_observations"][step, :, 0], codes)
            if (ids == 0).any():
                # Env 0 reset after t=0; t=1 history repeats its new first frame.
                assert (batch["proprio_history"][1, ids == 0] == 10).all()


def test_rl_snapshots_detach_and_copy_tensor_inputs():
    storage = Storage(1, 2, 2, 3, "cpu")
    critic = torch.ones(2, 3, requires_grad=True)
    actions = torch.ones(2, 2, requires_grad=True)
    scalar = torch.ones(2, 1, requires_grad=True)
    storage.add_rl(critic, actions, actions, actions, scalar, scalar, scalar,
                   torch.zeros(2, 1, dtype=torch.bool))
    with torch.no_grad():
        critic.fill_(99)
        actions.fill_(99)
        scalar.fill_(99)
    for name in storage._RL_FIELDS:
        assert not getattr(storage, name).requires_grad
    assert (storage.critic_observations == 1).all()
    assert (storage.actions == 1).all()
    assert (storage.rewards == 1).all()


def test_incomplete_rollout_and_invalid_minibatches_fail_before_training():
    storage = Storage(1, 1, 2, 3, "cpu")
    with pytest.raises(RuntimeError, match="stored 0"):
        storage.compute_returns(torch.zeros(1), 0.9, 0.8)
    with pytest.raises(RuntimeError, match="stored 0"):
        next(storage.batches(1, 1))
    _add(storage, [False], [0], [1], [False])
    with pytest.raises(ValueError, match="num_mini_batches"):
        next(storage.batches(2, 1))
    storage.clear()
    assert storage.step == 0
