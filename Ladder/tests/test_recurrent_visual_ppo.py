"""CPU integration checks for normalized recurrent PPO plus teacher imitation."""

from copy import deepcopy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.distributions import Normal
from tensordict import TensorDict


_PACKAGE_NAME = "cr1_recurrent_visual_ppo_test_package"
_PACKAGE = ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(
    Path(__file__).resolve().parents[1]
    / "source/ladder/ladder/tasks/ladder_climb/config/cr1/dagger"
)]
sys.modules.setdefault(_PACKAGE_NAME, _PACKAGE)
_algorithm = importlib.import_module(f"{_PACKAGE_NAME}.recurrent_visual_ppo_algorithm")
_model = importlib.import_module(f"{_PACKAGE_NAME}.recurrent_visual_ppo_model")
_teacher = importlib.import_module(f"{_PACKAGE_NAME}.student_teacher")
Algorithm = _algorithm.CR1RecurrentVisualPpoDagger
Student = _model.CR1RecurrentVisualPpoStudent


@pytest.fixture(autouse=True)
def _small_cpu_test_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(41)
    yield
    torch.set_num_threads(previous)


def _observations():
    history = torch.randn(4, 25, 85)
    history[..., -1] = torch.tensor([1.0, -1.0, 1.0, -1.0])[:, None]
    return TensorDict({
        "proprio_history": history,
        "depth_history": torch.rand(4, 1, 60, 80),
        "depth_updated": torch.ones(4, dtype=torch.bool),
        "teacher": torch.randn(4, 17),
    }, batch_size=[4])


def _configuration(checkpoint):
    return {
        "num_steps_per_env": 3,
        "multi_gpu": None,
        "policy": {
            "actor_hidden_dims": [32, 256], "critic_hidden_dims": [32, 16],
            "init_noise_std": 0.1,
        },
        "algorithm": {
            "class_name": "unused", "teacher_checkpoint": str(checkpoint),
            "teacher_hidden_dims": [32, 256], "expected_teacher_observation_dim": 17,
            "num_learning_epochs": 1, "num_mini_batches": 1,
            "learning_rate": 3.0e-4, "max_grad_norm": 1.0,
            "diagnostic_snapshot_interval": 1, "diagnostic_sample_envs": 4,
            "bc_warmup_iterations": 0, "teacher_rollout_fraction": 0.25,
            "gamma": 0.99, "lam": 0.95, "clip_param": 0.2,
            "value_loss_coef": 1.0, "entropy_coef": 0.01,
            "use_clipped_value_loss": True, "schedule": "fixed", "desired_kl": 0.01,
            "min_learning_rate": 1.0e-5, "max_learning_rate": 1.0e-3,
            "ppo_initial_coeff": 0.1, "ppo_warmup_iterations": 500,
            "dagger_initial_coeff": 1.0, "dagger_final_coeff": 0.1,
            "dagger_anneal_start": 2000, "dagger_anneal_iterations": 5000,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None, "symmetry_cfg": None,
        },
    }


def _make_algorithm(tmp_path, termination_manager=None):
    checkpoint = tmp_path / "teacher.pt"
    teacher = _teacher.elu_mlp(17, [32, 256], 4)
    torch.save({"actor_state_dict": {
        **{f"mlp.{key}": value for key, value in teacher.state_dict().items()},
        "distribution.std_param": torch.ones(4),
    }}, checkpoint)
    obs = _observations()
    cfg = _configuration(checkpoint)
    env = SimpleNamespace(num_envs=4, num_actions=4,
                          unwrapped=SimpleNamespace(termination_manager=termination_manager))
    algorithm = Algorithm.construct_algorithm(obs, env, cfg, "cpu")
    return algorithm, obs, cfg, env


def _next_observations(obs, dones, next_step):
    current = torch.randn(4, 85)
    current[:, -1] = obs["proprio_history"][:, -1, -1]
    shifted = torch.cat((obs["proprio_history"][:, 1:], current[:, None]), dim=1)
    history = torch.where(dones[:, None, None], current[:, None].expand(-1, 25, -1), shifted)
    updated = torch.tensor([
        [False, True, False, True],
        [True, False, True, False],
        [True, True, True, True],
    ][next_step - 1])
    depth = torch.where(updated[:, None, None, None],
                        torch.rand_like(obs["depth_history"]), obs["depth_history"])
    return TensorDict({"proprio_history": history, "depth_history": depth,
                       "depth_updated": updated, "teacher": torch.randn(4, 17)},
                      batch_size=[4])


def _collect(algorithm, obs):
    histories, actions, means = [], [], []
    done_sequence = torch.tensor([
        [False, True, False, False],
        [True, False, True, False],
        [False, False, False, False],
    ])
    # The official runner uses inference_mode for the complete rollout.
    with torch.inference_mode():
        for step in range(3):
            histories.append(obs["proprio_history"].clone())
            actions.append(algorithm.act(obs).clone())
            means.append(algorithm._ppo_pending["old_action_mean"].clone())
            dones = done_sequence[step]
            next_obs = _next_observations(obs, dones, step + 1)
            rewards = torch.tensor([100.0, 1.0 + step, -0.5 * step, 0.25 + step])
            algorithm.process_env_step(next_obs, rewards, dones, {})
            obs = next_obs
        algorithm.compute_returns(obs)
    return obs, torch.stack(histories), torch.stack(actions), torch.stack(means)


def test_on_policy_ratio_and_action_provenance_survive_done_and_depth_holds(tmp_path):
    algorithm, obs, _, _ = _make_algorithm(tmp_path)
    _, histories, executed, rollout_means = _collect(algorithm, obs)
    batch = next(algorithm.storage.batches(1, 1))
    ids = batch["env_ids"]
    torch.testing.assert_close(batch["proprio_history"], histories[:, ids])
    torch.testing.assert_close(batch["actions"], executed[:, ids])
    torch.testing.assert_close(executed[:, 0], algorithm.storage.teacher_action_mean[:, 0])
    assert not torch.equal(executed[:, 1:], rollout_means[:, 1:])
    expected_log_prob = Normal(rollout_means, algorithm.storage.old_action_std).log_prob(executed).sum(-1)
    torch.testing.assert_close(algorithm.storage.old_action_log_prob, expected_log_prob)
    assert algorithm.storage.teacher_controlled.tolist() == [[True, False, False, False]] * 3
    terms = algorithm._batch_losses(batch)
    torch.testing.assert_close(terms["mean"], rollout_means[:, ids], atol=2.0e-6, rtol=2.0e-5)
    assert terms["ratio_error"] < 2.0e-5
    assert abs(float(terms["policy_kl"])) < 1.0e-6
    assert not batch["initial_hidden"].is_inference()


def test_teacher_rl_nans_are_excluded_before_losses_but_teacher_labels_are_used(tmp_path):
    algorithm, obs, _, _ = _make_algorithm(tmp_path)
    _collect(algorithm, obs)
    batch = next(algorithm.storage.batches(1, 1))
    original = algorithm._batch_losses(batch)
    contaminated = {name: value.clone() for name, value in batch.items()}
    teacher = contaminated["teacher_controlled"]
    for name in ("critic_observations", "actions", "old_action_mean", "old_action_std",
                 "old_action_log_prob", "values", "rewards", "returns", "advantages"):
        contaminated[name][teacher] = float("nan")
    actual = algorithm._batch_losses(contaminated)
    for name in original:
        torch.testing.assert_close(actual[name], original[name])
        assert torch.isfinite(actual[name]).all()
    contaminated["teacher_action_mean"][teacher] += 10.0
    with_changed_labels = algorithm._batch_losses(contaminated)
    assert with_changed_labels["behavior"] > original["behavior"]
    torch.testing.assert_close(with_changed_labels["surrogate"], original["surrogate"])


def test_ppo_surrogate_alone_reaches_both_encoders_and_gru(tmp_path):
    algorithm, obs, _, _ = _make_algorithm(tmp_path)
    _collect(algorithm, obs)
    terms = algorithm._batch_losses(next(algorithm.storage.batches(1, 1)))
    named = list(algorithm.policy.named_parameters())
    grads = torch.autograd.grad(terms["surrogate"], [value for _, value in named], allow_unused=True)
    for prefix in ("proprio_encoder", "depth_encoder", "gru", "action_head"):
        norm_squared = sum(float(grad.square().sum())
                           for (name, _), grad in zip(named, grads)
                           if name.startswith(prefix + ".") and grad is not None)
        assert norm_squared > 0.0, prefix
    assert all(parameter.grad is None for parameter in algorithm.teacher.parameters())


def test_ppo_ramp_and_delayed_dagger_schedule_boundaries(tmp_path):
    algorithm, _, _, _ = _make_algorithm(tmp_path)
    for iteration, ppo, dagger in (
        (0, 0.1, 1.0),
        (1, 0.1018, 1.0),
        (250, 0.55, 1.0),
        (499, 0.9982, 1.0),
        (500, 1.0, 1.0),
        (501, 1.0, 1.0),
        (1999, 1.0, 1.0),
        (2000, 1.0, 1.0),
        (2001, 1.0, 0.99982),
        (4500, 1.0, 0.55),
        (7000, 1.0, 0.1),
        (8000, 1.0, 0.1),
    ):
        algorithm.update_iteration = iteration
        assert algorithm.ppo_coeff == pytest.approx(ppo)
        assert algorithm.dagger_coeff == pytest.approx(dagger)


@pytest.mark.parametrize("initial", [0.0, 0.1, 1.0])
def test_zero_ppo_warmup_means_full_ppo_from_iteration_zero(tmp_path, initial):
    _, obs, cfg, env = _make_algorithm(tmp_path)
    cfg["algorithm"].update(ppo_initial_coeff=initial, ppo_warmup_iterations=0)
    algorithm = Algorithm.construct_algorithm(obs, env, cfg, "cpu")
    assert algorithm.ppo_coeff == 1.0
    algorithm.update_iteration = 1000
    assert algorithm.ppo_coeff == 1.0


@pytest.mark.parametrize("ppo_coeff", [0.0, 0.1, 1.0])
def test_update_weights_actual_actor_loss_and_entropy_but_not_critic(
    tmp_path, monkeypatch, ppo_coeff
):
    algorithm, obs, _, _ = _make_algorithm(tmp_path)
    algorithm.ppo_initial_coeff = ppo_coeff
    # Avoid clipping hiding an accidental scaling of the critic gradient.
    algorithm.max_grad_norm = 1.0e6
    _collect(algorithm, obs)
    batch = next(algorithm.storage.batches(1, 1))
    terms = algorithm._batch_losses(batch)
    expected_actor_grads = torch.autograd.grad(
        ppo_coeff * (terms["surrogate"] - algorithm.entropy_coef * terms["entropy"])
        + algorithm.dagger_coeff * terms["behavior"], tuple(algorithm.policy.parameters())
    )
    expected_critic_grads = torch.autograd.grad(
        algorithm.value_loss_coef * terms["value"], tuple(algorithm.critic.parameters())
    )
    actual_actor_grads = []
    actual_critic_grads = []
    original_step = algorithm.optimizer.step

    def capture_optimizer_step(*args, **kwargs):
        actual_actor_grads.extend(parameter.grad.detach().clone()
                                 for parameter in algorithm.policy.parameters())
        actual_critic_grads.extend(parameter.grad.detach().clone()
                                  for parameter in algorithm.critic.parameters())
        return original_step(*args, **kwargs)

    monkeypatch.setattr(algorithm.optimizer, "step", capture_optimizer_step)
    losses = algorithm.update()
    assert losses["ppo_coeff"] == pytest.approx(ppo_coeff)
    assert losses["dagger_coeff"] == 1.0
    assert losses["effective_entropy_coef"] == pytest.approx(ppo_coeff * algorithm.entropy_coef)
    assert losses["weighted_ppo_surrogate_loss"] == pytest.approx(ppo_coeff * losses["surrogate"])
    assert losses["weighted_entropy_loss"] == pytest.approx(
        -ppo_coeff * algorithm.entropy_coef * losses["entropy"]
    )
    assert losses["weighted_ppo_actor_loss"] == pytest.approx(
        losses["weighted_ppo_surrogate_loss"] + losses["weighted_entropy_loss"]
    )
    assert losses["weighted_value_loss"] == pytest.approx(
        algorithm.value_loss_coef * losses["value"]
    )
    assert losses["total"] == pytest.approx(
        losses["weighted_ppo_actor_loss"] + losses["weighted_value_loss"]
        + losses["weighted_dagger_mse"], rel=2.0e-6,
    )
    assert len(actual_critic_grads) == len(expected_critic_grads)
    for actual, expected in zip(actual_critic_grads, expected_critic_grads):
        torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-5)
    assert len(actual_actor_grads) == len(expected_actor_grads)
    for actual, expected in zip(actual_actor_grads, expected_actor_grads):
        torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-5)
    for prefix in ("all", "proprio_encoder", "depth_encoder", "gru", "action_head"):
        raw = losses[f"ppo_{prefix}_grad_l2_probe"]
        assert raw > 0.0
        assert losses[f"weighted_ppo_{prefix}_grad_l2_probe"] == pytest.approx(ppo_coeff * raw)


def test_update_trains_actor_critic_and_std_and_preserves_teacher(tmp_path):
    algorithm, obs, _, _ = _make_algorithm(tmp_path)
    algorithm.configure_diagnostics(tmp_path, {"test": True})
    actor_before = deepcopy(algorithm.policy.state_dict())
    critic_before = deepcopy(algorithm.critic.state_dict())
    teacher_before = deepcopy(algorithm.teacher.state_dict())
    final_obs, _, _, _ = _collect(algorithm, obs)
    losses = algorithm.update()
    assert losses["optimizer_steps"] == 1.0
    assert losses["ppo_student_samples"] == 9.0
    assert losses["ppo_teacher_samples"] == 0.0
    assert losses["rollout_teacher_fraction"] == 0.25
    assert losses["ppo_preupdate_ratio_max_abs_error_probe"] < 2.0e-5
    for prefix in ("proprio_encoder", "depth_encoder", "gru", "action_head"):
        assert losses[f"ppo_{prefix}_grad_l2_probe"] > 0
        assert any(not torch.equal(actor_before[name], value)
                   for name, value in algorithm.policy.state_dict().items()
                   if name.startswith(prefix + "."))
    assert not torch.equal(actor_before["log_std"], algorithm.policy.log_std)
    assert any(not torch.equal(critic_before[name], value)
               for name, value in algorithm.critic.state_dict().items())
    for name, value in algorithm.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
    assert not any(value.requires_grad for value in algorithm.teacher.parameters())
    assert algorithm.storage.step == 0
    assert algorithm.update_iteration == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    snapshot = torch.load(
        tmp_path / "recurrent_visual_ppo_diagnostics/iteration_000000_sample.pt",
        map_location="cpu", weights_only=False,
    )
    assert snapshot["parameter_timing"] == "before_optimizer_update"
    replay = Student(4, [32, 256], 0.1)
    replay.load_state_dict(snapshot["policy_state_dict"], strict=True)
    replay_obs = TensorDict({
        "proprio_history": snapshot["proprio_history"],
        "depth_history": snapshot["depth"],
        "depth_updated": snapshot["depth_updated"],
    }, batch_size=[snapshot["student_mean"].shape[0]])
    with torch.no_grad():
        replay_mean, _, replay_hidden, _ = replay.action_mean_and_feature_step(
            replay_obs, snapshot["hidden_before"],
        )
    torch.testing.assert_close(replay_mean, snapshot["student_mean"])
    torch.testing.assert_close(replay_hidden, snapshot["hidden_after"])
    # A second rollout carries the first rollout's nonzero recurrent state.
    _collect(algorithm, final_obs)
    assert algorithm.storage.initial_hidden.abs().sum() > 0
    assert algorithm._batch_losses(next(algorithm.storage.batches(1, 1)))["ratio_error"] < 2.0e-5
    algorithm.update()


def test_normalized_step_sequence_outputs_and_feature_scales_match():
    student = Student(4, [32, 256], 0.1).eval()
    obs = _observations()
    obs["proprio_history"][..., :-1] *= 100.0
    hidden = torch.randn(1, 4, 256)
    initial_hidden = hidden.clone()
    means, history, depth, fresh, dones = [], [], [], [], []
    with torch.no_grad():
        for step in range(3):
            if step > 0:
                hidden = student.reset_hidden_state(hidden, dones[-1])
            mean, _, hidden, activations = student.action_mean_and_feature_step(obs, hidden)
            for name in ("proprio_feature", "depth_feature"):
                value = activations[name]
                torch.testing.assert_close(value.mean(-1), torch.zeros(4), atol=1.0e-5, rtol=0)
                assert (value.square().mean(-1).sqrt() <= 1.001).all()
                assert (value.square().mean(-1).sqrt() >= 0.99).all()
            means.append(mean)
            history.append(obs["proprio_history"])
            depth.append(obs["depth_history"])
            fresh.append(obs["depth_updated"])
            dones.append(torch.tensor([step == 0, step == 1, False, False]))
            obs = _next_observations(obs, dones[-1], step + 1)
        sequence, _, final_hidden = student.action_mean_and_feature_sequence(
            torch.stack(history), torch.stack(depth), torch.stack(fresh),
            torch.stack(dones), initial_hidden,
        )
    torch.testing.assert_close(sequence, torch.stack(means), atol=2.0e-6, rtol=2.0e-5)
    torch.testing.assert_close(final_hidden, hidden, atol=2.0e-6, rtol=2.0e-5)


def test_checkpoint_contract_resume_and_actor_only_loading(tmp_path):
    algorithm, obs, cfg, env = _make_algorithm(tmp_path)
    _collect(algorithm, obs)
    algorithm.update()
    payload = deepcopy(algorithm.save())
    restored = Algorithm.construct_algorithm(obs, env, cfg, "cpu")
    assert restored.load(payload, strict=True)
    assert restored.update_iteration == 1
    assert restored.ppo_coeff == pytest.approx(0.1018)
    assert restored.dagger_coeff == 1.0
    assert payload["training_contract"]["ppo_initial_coeff"] == 0.1
    assert payload["training_contract"]["ppo_warmup_iterations"] == 500
    for name, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[name], value)
    for name, value in algorithm.critic.state_dict().items():
        torch.testing.assert_close(restored.critic.state_dict()[name], value)
    with pytest.raises(RuntimeError, match="data contract"):
        restored.load({**payload, "data_contract": "old_mse_without_normalization"})
    with pytest.raises(RuntimeError, match="different teacher"):
        restored.load({**payload, "teacher_checkpoint_sha256": "wrong"})
    changed_contract = {**payload, "training_contract": {}}
    with pytest.raises(RuntimeError, match="training contract"):
        restored.load(changed_contract)
    actor_only = Algorithm.construct_algorithm(obs, env, cfg, "cpu")
    untouched_critic = deepcopy(actor_only.critic.state_dict())
    legacy_contract = {name: value for name, value in payload["training_contract"].items()
                       if name not in ("ppo_initial_coeff", "ppo_warmup_iterations")}
    actor_payload = {name: value for name, value in payload.items()
                     if name not in ("critic_state_dict", "optimizer_state_dict")}
    actor_payload["training_contract"] = legacy_contract
    assert not actor_only.load(actor_payload, {
        "actor": True, "critic": False, "optimizer": False, "iteration": False,
    })
    assert actor_only.update_iteration == 0
    for name, value in actor_only.critic.state_dict().items():
        torch.testing.assert_close(value, untouched_critic[name], rtol=0, atol=0)
    actor_obs = algorithm._actor_observations(obs)
    with torch.inference_mode():
        expected, _, _, _ = algorithm.policy.action_mean_and_feature_step(actor_obs, None)
        actual, _, _, _ = actor_only.policy.action_mean_and_feature_step(actor_obs, None)
    torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("checkpoint_iteration", [250, 2500])
def test_checkpoint_restores_current_ppo_and_dagger_schedule_progress(tmp_path, checkpoint_iteration):
    algorithm, obs, cfg, env = _make_algorithm(tmp_path)
    algorithm.update_iteration = checkpoint_iteration
    payload = deepcopy(algorithm.save())
    restored = Algorithm.construct_algorithm(obs, env, cfg, "cpu")
    restored.load(payload)
    assert restored.ppo_coeff == pytest.approx(algorithm.ppo_coeff)
    assert restored.dagger_coeff == pytest.approx(algorithm.dagger_coeff)
    _collect(restored, obs)
    losses = restored.update()
    assert losses["ppo_coeff"] == pytest.approx(algorithm.ppo_coeff)
    assert losses["dagger_coeff"] == pytest.approx(algorithm.dagger_coeff)
    assert restored.update_iteration == checkpoint_iteration + 1


@pytest.mark.parametrize("field,new_value", [
    ("ppo_initial_coeff", 0.2),
    ("ppo_warmup_iterations", 1000),
    ("dagger_anneal_start", 1000),
    ("dagger_anneal_iterations", 4000),
])
def test_resume_rejects_changed_objective_schedule(tmp_path, field, new_value):
    algorithm, obs, cfg, env = _make_algorithm(tmp_path)
    payload = deepcopy(algorithm.save())
    cfg["algorithm"][field] = new_value
    changed = Algorithm.construct_algorithm(obs, env, cfg, "cpu")
    with pytest.raises(RuntimeError, match="training contract"):
        changed.load(payload)


def test_timeout_bootstrap_excludes_simultaneous_true_failure(tmp_path):
    termination = SimpleNamespace(active_terms=(),
                                  terminated=torch.tensor([False, True, False, True]))
    algorithm, obs, _, _ = _make_algorithm(tmp_path, termination)
    with torch.inference_mode():
        algorithm.act(obs)
        algorithm.process_env_step(obs, torch.ones(4),
                                   torch.tensor([True, True, False, True]),
                                   {"time_outs": torch.tensor([True, True, False, False])})
    assert algorithm.storage.time_outs[0].tolist() == [True, False, False, False]
    torch.testing.assert_close(algorithm.storage.rewards[0], torch.ones(4))
