"""CPU checks for the isolated end-to-end oracle contact-goal diagnostic."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from ladder.rsl_rl.algorithms.end_to_end_visual_ppo import EndToEndVisualPPO
from ladder.rsl_rl.modules.end_to_end_visual_actor import (
    END_TO_END_VISUAL_DATA_CONTRACT,
    ORACLE_CONTACT_GOAL_DATA_CONTRACT,
    ORACLE_CONTACT_GOAL_DIM,
    ORACLE_CONTACT_GOAL_KEY,
    EndToEndVisualActor,
)
from ladder.rsl_rl.modules.transformer_vision_actor import (
    CURRENT_PROPRIO_KEY,
    DEPTH_HISTORY_KEY,
    PROPRIO_HISTORY_KEY,
)
from ladder.rsl_rl.runners.end_to_end_visual_wrapper import (
    _as_end_to_end_visual_tensordict,
)
from ladder.rsl_rl.storage.transformer_vision_ppo_storage import CRITIC_KEY


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _observations(count=4, *, oracle=False, oracle_value=None):
    history = torch.randn(count, 25, 85) * 0.2
    values = {
        CURRENT_PROPRIO_KEY: history[:, -1].clone(),
        PROPRIO_HISTORY_KEY: history,
        DEPTH_HISTORY_KEY: torch.rand(count, 2, 60, 80),
        CRITIC_KEY: torch.randn(count, 7),
    }
    if oracle:
        values[ORACLE_CONTACT_GOAL_KEY] = (
            torch.randn(count, ORACLE_CONTACT_GOAL_DIM)
            if oracle_value is None
            else oracle_value.clone()
        )
    return TensorDict(values, batch_size=[count])


def _config(*, oracle=False):
    policy = {
        "actor_hidden_dims": [32, 16],
        "critic_hidden_dims": [16],
        "init_noise_std": 1.0,
    }
    if oracle:
        policy["use_oracle_contact_goal"] = True
    return {
        "num_steps_per_env": 3,
        "multi_gpu": None,
        "policy": policy,
        "algorithm": {
            "class_name": "unused",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "sequence_micro_batch_envs": 2,
            "learning_rate": 3.0e-4,
            "schedule": "fixed",
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def _algorithm(*, oracle=False, count=4):
    torch.manual_seed(123)
    return EndToEndVisualPPO.construct_algorithm(
        _observations(count, oracle=oracle),
        SimpleNamespace(num_envs=count, num_actions=5),
        _config(oracle=oracle),
        "cpu",
    )


def _actor_observations(count=2, oracle_value=None):
    history = torch.randn(count, 25, 85) * 0.2
    values = {
        CURRENT_PROPRIO_KEY: history[:, -1].clone(),
        PROPRIO_HISTORY_KEY: history,
        DEPTH_HISTORY_KEY: torch.rand(count, 2, 60, 80),
    }
    if oracle_value is not None:
        values[ORACLE_CONTACT_GOAL_KEY] = oracle_value
    return TensorDict(values, batch_size=[count])


def _raw_oracle_observations(count=4):
    history = torch.randn(count, 25, 85)
    goals = torch.arange(count * 12, dtype=torch.float32).reshape(count, 12) / 10.0
    actual = torch.randn(count, 12)
    current_expected = torch.randn(count, 12)
    next_stable = torch.randn(count, 12)
    return {
        "transformer_vision_inputs": {
            "proprio_history": history,
            "depth_history": torch.rand(count, 2, 60, 80),
            "depth_updated": torch.ones(count, 1),
        },
        ORACLE_CONTACT_GOAL_KEY: {
            "next_stable_goal_b": goals,
            "actual_support_relation": actual,
            "current_expected_support_relation": current_expected,
            "next_stable_support_relation": next_stable,
        },
    }


def test_actor_uses_oracle_input_and_baseline_ignores_it():
    torch.manual_seed(41)
    oracle_actor = EndToEndVisualActor(
        action_dim=5, actor_hidden_dims=[32, 16], use_oracle_contact_goal=True,
    ).eval()
    base = _actor_observations(oracle_value=torch.zeros(2, ORACLE_CONTACT_GOAL_DIM))
    changed = base.clone()
    changed[ORACLE_CONTACT_GOAL_KEY] = torch.full(
        (2, ORACLE_CONTACT_GOAL_DIM), 3.0
    )
    with torch.no_grad():
        oracle_actor.reset()
        first = oracle_actor(base)
        oracle_actor.reset()
        second = oracle_actor(changed)
    assert oracle_actor.actor_input_dim == 389
    assert not torch.allclose(first, second)

    torch.manual_seed(41)
    baseline = EndToEndVisualActor(
        action_dim=5, actor_hidden_dims=[32, 16], use_oracle_contact_goal=False,
    ).eval()
    without_oracle = base.exclude(ORACLE_CONTACT_GOAL_KEY)
    with torch.no_grad():
        baseline.reset()
        expected = baseline(without_oracle)
        baseline.reset()
        actual = baseline(changed)
    assert baseline.actor_input_dim == 341
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_oracle_actor_and_algorithm_reject_missing_or_wrong_shapes():
    actor = EndToEndVisualActor(
        action_dim=5, actor_hidden_dims=[32, 16], use_oracle_contact_goal=True,
    ).eval()
    missing = _actor_observations()
    with pytest.raises(KeyError, match="oracle_contact_goal"):
        actor(missing)
    wrong = _actor_observations(oracle_value=torch.zeros(2, 47))
    with pytest.raises(ValueError, match="48"):
        actor(wrong)

    env = SimpleNamespace(num_envs=4, num_actions=5)
    with pytest.raises(KeyError, match="oracle_contact_goal"):
        EndToEndVisualPPO.construct_algorithm(
            _observations(oracle=False), env, _config(oracle=True), "cpu",
        )
    wrong_training = _observations(oracle=True)
    wrong_training[ORACLE_CONTACT_GOAL_KEY] = torch.zeros(4, 47)
    with pytest.raises(ValueError, match="48"):
        EndToEndVisualPPO.construct_algorithm(
            wrong_training, env, _config(oracle=True), "cpu",
        )


def test_actor_only_wrapper_packs_goal_state_and_contact_fields():
    raw = _raw_oracle_observations()
    result = _as_end_to_end_visual_tensordict(
        raw, num_envs=4, device="cpu", actor_only=True,
    )
    assert set(result.keys()) == {
        CURRENT_PROPRIO_KEY,
        PROPRIO_HISTORY_KEY,
        DEPTH_HISTORY_KEY,
        ORACLE_CONTACT_GOAL_KEY,
    }
    oracle = raw[ORACLE_CONTACT_GOAL_KEY]
    expected = torch.cat(
        (
            oracle["next_stable_goal_b"],
            oracle["actual_support_relation"],
            oracle["current_expected_support_relation"],
            oracle["next_stable_support_relation"],
        ),
        dim=-1,
    )
    torch.testing.assert_close(result[ORACLE_CONTACT_GOAL_KEY], expected)
    assert result[ORACLE_CONTACT_GOAL_KEY].shape == (4, ORACLE_CONTACT_GOAL_DIM)

    missing = deepcopy(raw)
    del missing[ORACLE_CONTACT_GOAL_KEY]["actual_support_relation"]
    with pytest.raises(KeyError):
        _as_end_to_end_visual_tensordict(
            missing, num_envs=4, device="cpu", actor_only=True,
        )
    wrong = deepcopy(raw)
    wrong[ORACLE_CONTACT_GOAL_KEY]["next_stable_goal_b"] = torch.zeros(4, 11)
    with pytest.raises(ValueError, match="next_stable_goal_b"):
        _as_end_to_end_visual_tensordict(
            wrong, num_envs=4, device="cpu", actor_only=True,
        )


def test_oracle_raw_sequence_replays_after_done_and_cached_camera():
    algorithm = _algorithm(oracle=True)
    obs = _observations(oracle=True)
    oracle_sequence = []
    done_sequence = []
    for step in range(algorithm.storage.num_steps):
        oracle_sequence.append(obs[ORACLE_CONTACT_GOAL_KEY].clone())
        algorithm.act(obs)
        following = _observations(oracle=True)
        if step == 1:
            following[DEPTH_HISTORY_KEY] = obs[DEPTH_HISTORY_KEY].clone()
        done = torch.tensor([step == 0, False, step == 1, False])
        done_sequence.append(done)
        algorithm.process_env_step(
            following, torch.tensor([0.2, 0.6, -0.5, 0.9]), done, {},
        )
        obs = following
    algorithm.compute_returns(obs)

    stored_oracle = algorithm.storage.actor_inputs[..., -ORACLE_CONTACT_GOAL_DIM:]
    torch.testing.assert_close(stored_oracle, torch.stack(oracle_sequence))
    torch.testing.assert_close(
        algorithm.storage.depth_history[2], algorithm.storage.depth_history[1],
    )
    assert torch.equal(
        algorithm.storage.reset_before_mask()[1], done_sequence[0],
    )
    ids = torch.arange(4)
    terms = algorithm._batch_terms(ids, algorithm._denominators(ids))
    assert terms["logprob_error"] < 5.0e-6
    assert terms["ratio_error"] < 5.0e-6
    assert terms["policy_kl"].abs() < 1.0e-6


def test_baseline_and_oracle_checkpoints_are_strictly_isolated():
    baseline = _algorithm(oracle=False)
    oracle = _algorithm(oracle=True)
    assert baseline.data_contract == END_TO_END_VISUAL_DATA_CONTRACT
    assert oracle.data_contract == ORACLE_CONTACT_GOAL_DATA_CONTRACT
    assert baseline.data_contract != oracle.data_contract
    assert (baseline.storage.actor_input_dim, oracle.storage.actor_input_dim) == (341, 389)

    oracle.num_updates = 7
    payload = deepcopy(oracle.save())
    restored = _algorithm(oracle=True)
    assert restored.load(payload)
    assert restored.num_updates == 7
    for name, value in oracle.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[name], value)

    with pytest.raises(RuntimeError, match="data contract"):
        baseline.load(deepcopy(oracle.save()))
    with pytest.raises(RuntimeError, match="data contract"):
        oracle.load(deepcopy(baseline.save()))
    with pytest.raises(RuntimeError, match="data contract"):
        baseline.load(
            deepcopy(oracle.save()),
            load_cfg={"actor": True, "critic": False, "optimizer": False},
        )
