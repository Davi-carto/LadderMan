"""LayerNorm ablation keeps the existing mixed recurrent MSE training path."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


_PACKAGE_NAME = "cr1_recurrent_visual_ln_mse_test_package"
_PACKAGE = ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(
    Path(__file__).resolve().parents[1]
    / "source/ladder/ladder/tasks/ladder_climb/config/cr1/dagger"
)]
sys.modules.setdefault(_PACKAGE_NAME, _PACKAGE)
_ln_module = importlib.import_module(f"{_PACKAGE_NAME}.recurrent_visual_ln_mse_algorithm")
_base_module = importlib.import_module(f"{_PACKAGE_NAME}.recurrent_visual_mse_algorithm")
_teacher_module = importlib.import_module(f"{_PACKAGE_NAME}.student_teacher")
LnAlgorithm = _ln_module.CR1RecurrentVisualLnMseDagger
BaseAlgorithm = _base_module.CR1RecurrentVisualMseDagger


@pytest.fixture(autouse=True)
def _small_cpu_test_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(43)
    yield
    torch.set_num_threads(previous)


def _setup(tmp_path, cls=LnAlgorithm):
    teacher = _teacher_module.elu_mlp(17, [32, 256], 4)
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"actor_state_dict": {
        f"mlp.{name}": value for name, value in teacher.state_dict().items()
    }}, checkpoint)
    history = torch.randn(4, 25, 85)
    history[..., -1] = torch.tensor([1.0, -1.0, 1.0, -1.0])[:, None]
    obs = TensorDict({
        "proprio_history": history,
        "depth_history": torch.rand(4, 1, 60, 80),
        "depth_updated": torch.ones(4, dtype=torch.bool),
        "teacher": torch.randn(4, 17),
    }, batch_size=[4])
    cfg = {
        "num_steps_per_env": 3,
        "policy": {"actor_hidden_dims": [32, 256], "init_noise_std": 0.1},
        "algorithm": {
            "class_name": "unused", "teacher_checkpoint": str(checkpoint),
            "teacher_hidden_dims": [32, 256], "expected_teacher_observation_dim": 17,
            "num_learning_epochs": 1, "num_mini_batches": 1,
            "learning_rate": 1.0e-3, "max_grad_norm": 1.0,
            "diagnostic_snapshot_interval": 1, "diagnostic_sample_envs": 4,
            "bc_warmup_iterations": 0, "teacher_rollout_fraction": 0.25,
        },
    }
    env = SimpleNamespace(num_envs=4, num_actions=4)
    return cls.construct_algorithm(obs, env, cfg, "cpu"), obs, cfg, env


def _collect(algorithm, obs):
    with torch.inference_mode():
        for step in range(3):
            actions = algorithm.act(obs)
            torch.testing.assert_close(actions[0], algorithm.teacher(obs["teacher"])[0])
            assert algorithm._pending.teacher_controlled.tolist() == [True, False, False, False]
            dones = torch.tensor([False, step == 0, False, False])
            frame = torch.randn(4, 85)
            frame[:, -1] = obs["proprio_history"][:, -1, -1]
            shifted = torch.cat((obs["proprio_history"][:, 1:], frame[:, None]), dim=1)
            obs = TensorDict({
                "proprio_history": torch.where(dones[:, None, None], frame[:, None], shifted),
                "depth_history": torch.rand(4, 1, 60, 80),
                "depth_updated": torch.tensor([step % 2 == 0, True, True, False]),
                "teacher": torch.randn(4, 17),
            }, batch_size=[4])
            algorithm.process_env_step(obs, torch.ones(4), dones, {})


def test_network_only_adds_two_128_dimensional_layernorms():
    torch.manual_seed(7)
    base = BaseAlgorithm.student_class(4, [32, 256], 0.1)
    torch.manual_seed(7)
    normalized = LnAlgorithm.student_class(4, [32, 256], 0.1)
    base_state = base.state_dict()
    normalized_state = normalized.state_dict()
    extra = set(normalized_state) - set(base_state)
    assert extra == {
        "proprio_encoder.layer_norm.weight", "proprio_encoder.layer_norm.bias",
        "depth_encoder.projection.layer_norm.weight", "depth_encoder.projection.layer_norm.bias",
    }
    assert sum(normalized_state[name].numel() for name in extra) == 512
    for name, value in base_state.items():
        torch.testing.assert_close(value, normalized_state[name], rtol=0, atol=0)
    assert isinstance(normalized.proprio_encoder.layer_norm, torch.nn.LayerNorm)
    assert isinstance(normalized.depth_encoder.projection.layer_norm, torch.nn.LayerNorm)


def test_ln_update_is_mse_only_with_fixed_std_and_trainable_encoders(tmp_path):
    algorithm, obs, _, _ = _setup(tmp_path)
    before = deepcopy(algorithm.policy.state_dict())
    teacher_before = deepcopy(algorithm.teacher.state_dict())
    assert algorithm.get_policy() is algorithm.policy
    assert not hasattr(algorithm, "critic")
    assert not hasattr(algorithm.storage, "returns")
    assert not algorithm.policy.log_std.requires_grad
    _collect(algorithm, obs)
    batch = next(algorithm.storage.batches(1, 1))
    with torch.no_grad():
        mean, _, _ = algorithm.policy.action_mean_and_feature_sequence(
            batch["proprio_history"], batch["depth"], batch["depth_updated"],
            batch["dones"], batch["initial_hidden"],
        )
        expected_mse = (mean - batch["teacher_action_mean"]).square().mean().item()
    losses = algorithm.update()
    assert losses["behavior"] == pytest.approx(expected_mse, rel=2.0e-6)
    assert losses["optimizer_steps"] == 1
    assert losses["rollout_teacher_fraction"] == 0.25
    assert losses["rollout_student_fraction"] == 0.75
    assert not any(term in name for name in losses for term in ("ppo", "kl", "surrogate", "entropy", "reconstruction"))
    torch.testing.assert_close(algorithm.policy.log_std, before["log_std"], rtol=0, atol=0)
    torch.testing.assert_close(algorithm.policy.output_std, torch.full((4,), 0.1))
    for prefix in ("proprio_encoder", "depth_encoder", "gru", "action_head"):
        assert losses[f"student_{prefix}_grad_l2_max_after_clip"] > 0
    for name, value in algorithm.policy.state_dict().items():
        if "layer_norm" in name:
            assert not torch.equal(value, before[name]), name
    for name, value in algorithm.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
    assert algorithm.learning_rate == 1.0e-3
    assert algorithm.update_iteration == 1


def test_manifest_and_checkpoint_have_independent_normalized_mse_contract(tmp_path):
    algorithm, obs, cfg, env = _setup(tmp_path)
    algorithm.configure_diagnostics(str(tmp_path), {"test": True})
    directory = tmp_path / "recurrent_visual_mse_diagnostics"
    manifest = json.loads((directory / "manifest.json").read_text())
    contract = _ln_module.RECURRENT_VISUAL_LN_MSE_DAGGER_DATA_CONTRACT
    assert manifest["data_contract"] == contract
    assert "LayerNorm(128)" in manifest["feature_normalization"]
    assert "MSE DAgger" in manifest["ablation"]
    assert "ppo" in manifest["excluded_losses"]
    assert manifest["controller_schedule"]["bc_warmup_iterations"] == 0
    assert manifest["controller_schedule"]["teacher_rollout_fraction_after_warmup"] == 0.25
    _collect(algorithm, obs)
    algorithm.update()
    snapshot = torch.load(directory / "iteration_000000_sample.pt", weights_only=False)
    for name in ("proprio_feature", "depth_feature"):
        feature = snapshot[name]
        torch.testing.assert_close(feature.mean(-1), torch.zeros(feature.shape[0]), atol=1.0e-6, rtol=0)
        assert (feature.square().mean(-1).sqrt() <= 1.001).all()
    payload = deepcopy(algorithm.save())
    assert payload["data_contract"] == contract
    restored = LnAlgorithm.construct_algorithm(obs, env, cfg, "cpu")
    assert restored.load(payload, None, True)
    assert restored.update_iteration == 1
    for name, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[name], value, rtol=0, atol=0)
    baseline = BaseAlgorithm.construct_algorithm(obs, env, cfg, "cpu")
    with pytest.raises(RuntimeError, match="data contract"):
        baseline.load(payload, None, True)
    with pytest.raises(RuntimeError, match="data contract"):
        restored.load(baseline.save(), None, True)


@pytest.mark.parametrize("teacher_fraction", [0.0, 0.25])
def test_original_mse_checkpoint_remains_compatible(tmp_path, teacher_fraction):
    _, obs, cfg, env = _setup(tmp_path)
    cfg["algorithm"]["teacher_rollout_fraction"] = teacher_fraction
    original = BaseAlgorithm.construct_algorithm(obs, env, cfg, "cpu")
    payload = deepcopy(original.save())
    assert payload["data_contract"] == _base_module.RECURRENT_VISUAL_DAGGER_DATA_CONTRACT
    if teacher_fraction == 0.0:
        # The earliest student-only checkpoints did not store a controller schedule.
        payload.pop("bc_warmup_iterations")
        payload.pop("teacher_rollout_fraction")
    restored = BaseAlgorithm.construct_algorithm(obs, env, cfg, "cpu")
    assert restored.load(payload, None, True)
    assert not any("layer_norm" in name for name in restored.policy.state_dict())
    for name, value in original.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[name], value, rtol=0, atol=0)


def test_runner_preserves_baseline_training_settings():
    pytest.importorskip("isaaclab.sim", reason="Isaac Sim is required for task config imports")
    from ladder.tasks.ladder_climb.config.cr1.dagger.recurrent_visual_ln_mse_agent_cfg import (
        CR1RecurrentVisualLnMseDaggerRunnerCfg,
    )
    from ladder.tasks.ladder_climb.config.cr1.dagger.recurrent_visual_mse_agent_cfg import (
        CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg,
    )
    baseline = CR1RecurrentVisualNoWarmupMixedMseDaggerRunnerCfg()
    normalized = CR1RecurrentVisualLnMseDaggerRunnerCfg()
    assert normalized.use_recurrent_visual_ln_mse_dagger_runner
    assert not getattr(normalized, "use_recurrent_visual_ppo_dagger_runner", False)
    assert normalized.max_iterations == 2000
    assert normalized.save_interval == 500
    assert normalized.policy.to_dict() == baseline.policy.to_dict()
    expected = baseline.algorithm.to_dict()
    actual = normalized.algorithm.to_dict()
    assert actual.pop("class_name").endswith(":CR1RecurrentVisualLnMseDagger")
    expected.pop("class_name")
    assert actual == expected
