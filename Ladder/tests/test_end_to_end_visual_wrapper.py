"""Policy-only visual inputs and privileged critic remain separate."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from ladder.rsl_rl.modules.transformer_vision_actor import (
    CURRENT_PROPRIO_KEY, DEPTH_HISTORY_KEY, PROPRIO_HISTORY_KEY,
)
from ladder.rsl_rl.modules.end_to_end_visual_actor import ORACLE_CONTACT_GOAL_KEY
from ladder.rsl_rl.runners.end_to_end_visual_wrapper import (
    EndToEndVisualPPOVecEnvWrapper, _as_end_to_end_visual_tensordict,
)
from ladder.rsl_rl.storage.transformer_vision_ppo_storage import CRITIC_KEY


def _raw_observations(actor_only=False):
    # A distinct already-noisy current frame makes independent current-state
    # recomputation or accidental use of a clean label easy to detect.
    history = torch.randn(4, 25, 85)
    history[:, -1] += 5.0
    history[..., -1] = torch.tensor([1.0, -1.0, 0.0, 1.0])[:, None]
    raw = {"transformer_vision_inputs": {
        "proprio_history": history,
        "depth_history": torch.rand(4, 2, 60, 80),
        "depth_updated": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
    }}
    if not actor_only:
        raw["critic"] = torch.randn(4, 780)
    return raw


@pytest.mark.parametrize("width", [57, 61, 64])
def test_contact_task_conversion_preserves_legacy_phased_and_body_goal(width):
    raw = _raw_observations()
    raw["contact_task"] = torch.randn(4, width)
    converted = _as_end_to_end_visual_tensordict(raw, num_envs=4, device="cpu")
    torch.testing.assert_close(converted[ORACLE_CONTACT_GOAL_KEY], raw["contact_task"])


@pytest.mark.parametrize("width", [60, 63, 65])
def test_contact_task_conversion_rejects_unsupported_width(width):
    raw = _raw_observations()
    raw["contact_task"] = torch.randn(4, width)
    with pytest.raises(ValueError, match="Contact-task observation"):
        _as_end_to_end_visual_tensordict(raw, num_envs=4, device="cpu")


class _FakeEnvironment:
    def __init__(self, *, actor_only=False, finite_horizon=False):
        self.unwrapped = self
        self.num_envs = 4
        self.device = "cpu"
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.cfg = SimpleNamespace(is_finite_horizon=finite_horizon)
        self.action_manager = SimpleNamespace(total_action_dim=25)
        self.observations = _raw_observations(actor_only)
        self.observation_manager = SimpleNamespace(compute=lambda: self.observations)
        self.reset_count = 0
        self.closed = False
        self.last_actions = None

    def reset(self):
        self.reset_count += 1
        self.episode_length_buf.zero_()
        return self.observations, {}

    def _get_observations(self):
        return self.observations

    def step(self, actions):
        self.last_actions = actions.clone()
        self.episode_length_buf += 1
        terminated = torch.tensor([False, True, True, False])
        truncated = torch.tensor([True, False, True, False])
        return self.observations, torch.arange(4.0), terminated, truncated, {"test": True}

    def close(self):
        self.closed = True


@pytest.mark.parametrize("tensor_dict", [False, True])
def test_actor_only_conversion_does_not_require_any_privileged_group(tensor_dict):
    raw = _raw_observations(actor_only=True)
    observations = TensorDict(raw, batch_size=[4]) if tensor_dict else raw
    result = _as_end_to_end_visual_tensordict(observations, num_envs=4, device="cpu", actor_only=True)
    assert set(result.keys()) == {CURRENT_PROPRIO_KEY, PROPRIO_HISTORY_KEY, DEPTH_HISTORY_KEY}
    assert result.batch_size == torch.Size([4])
    torch.testing.assert_close(result[CURRENT_PROPRIO_KEY], raw["transformer_vision_inputs"]["proprio_history"][:, -1])
    assert result[PROPRIO_HISTORY_KEY].shape == (4, 25, 85)
    assert result[DEPTH_HISTORY_KEY].shape == (4, 2, 60, 80)


@pytest.mark.parametrize("tensor_dict", [False, True])
def test_training_conversion_needs_only_policy_inputs_and_privileged_critic(tensor_dict):
    raw = _raw_observations()
    observations = TensorDict(raw, batch_size=[4]) if tensor_dict else raw
    result = _as_end_to_end_visual_tensordict(observations, num_envs=4, device="cpu")
    assert set(result.keys()) == {
        CURRENT_PROPRIO_KEY, PROPRIO_HISTORY_KEY, DEPTH_HISTORY_KEY,
        CRITIC_KEY,
    }
    assert not any("reference" in key or "teacher" in key for key in result.keys())
    torch.testing.assert_close(result[CURRENT_PROPRIO_KEY], result[PROPRIO_HISTORY_KEY][:, -1])
    torch.testing.assert_close(result[CURRENT_PROPRIO_KEY], raw["transformer_vision_inputs"]["proprio_history"][:, -1])
    torch.testing.assert_close(result[CRITIC_KEY], raw["critic"])


def test_optional_observation_clip_preserves_current_history_alignment_and_critic():
    raw = _raw_observations()
    result = _as_end_to_end_visual_tensordict(raw, num_envs=4, device="cpu", clip_observations=0.5)
    expected = raw["transformer_vision_inputs"]["proprio_history"].clamp(-0.5, 0.5)
    torch.testing.assert_close(result[PROPRIO_HISTORY_KEY], expected)
    torch.testing.assert_close(result[CURRENT_PROPRIO_KEY], expected[:, -1])
    assert result[CRITIC_KEY].abs().max() <= 0.5


def test_wrapper_step_excludes_failure_timeouts_and_does_not_clip_actions_by_default():
    environment = _FakeEnvironment()
    wrapper = EndToEndVisualPPOVecEnvWrapper(environment)
    assert wrapper.clip_actions is None
    assert wrapper.clip_observations is None
    actions = torch.linspace(-100.0, 100.0, 100).reshape(4, 25)
    observations, rewards, dones, extras = wrapper.step(actions)
    torch.testing.assert_close(environment.last_actions, actions, rtol=0, atol=0)
    torch.testing.assert_close(rewards, torch.arange(4.0))
    assert dones.tolist() == [True, True, True, False]
    assert extras["time_outs"].tolist() == [True, False, False, False]
    assert extras["test"]
    assert observations[PROPRIO_HISTORY_KEY].shape == (4, 25, 85)


def test_wrapper_applies_configured_action_safety_bound_before_environment_step():
    environment = _FakeEnvironment()
    wrapper = EndToEndVisualPPOVecEnvWrapper(environment, clip_actions=10.0)
    actions = torch.linspace(-20.0, 20.0, 100).reshape(4, 25)
    wrapper.step(actions)
    torch.testing.assert_close(
        environment.last_actions, actions.clamp(-10.0, 10.0), rtol=0, atol=0
    )


def test_finite_horizon_does_not_add_timeout_bootstrap_mask():
    wrapper = EndToEndVisualPPOVecEnvWrapper(_FakeEnvironment(finite_horizon=True))
    _, _, dones, extras = wrapper.step(torch.zeros(4, 25))
    assert "time_outs" not in extras
    assert dones.tolist() == [True, True, True, False]


@pytest.mark.parametrize("use_observation_manager", [False, True])
def test_actor_only_reset_and_get_observations_keep_standard_wrapper_lifecycle(use_observation_manager):
    environment = _FakeEnvironment(actor_only=True)
    if not use_observation_manager:
        del environment.observation_manager
    wrapper = EndToEndVisualPPOVecEnvWrapper(environment)
    wrapper.actor_only = True
    assert environment.reset_count == 1
    reset_observations = wrapper.reset()
    assert isinstance(reset_observations, TensorDict)
    assert environment.reset_count == 2
    current = wrapper.get_observations()
    assert set(current.keys()) == {CURRENT_PROPRIO_KEY, PROPRIO_HISTORY_KEY, DEPTH_HISTORY_KEY}
    for name in current.keys():
        torch.testing.assert_close(current[name], reset_observations[name])
    wrapper.episode_length_buf = torch.arange(4)
    torch.testing.assert_close(environment.episode_length_buf, torch.arange(4))
    wrapper.close()
    assert environment.closed
