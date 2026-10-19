"""CPU checks for policy-only recurrent visual PPO and microbatch equivalence."""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from ladder.rsl_rl.algorithms.end_to_end_visual_ppo import EndToEndVisualPPO
from ladder.rsl_rl.modules.end_to_end_visual_actor import (
    ORACLE_CONTACT_GOAL_KEY, BODY_GOAL_CONTACT_TASK_DATA_CONTRACT,
    PHASED_CONTACT_TASK_DATA_CONTRACT,
)
from ladder.rsl_rl.modules.transformer_vision_actor import (
    CURRENT_PROPRIO_KEY, DEPTH_HISTORY_KEY, PROPRIO_HISTORY_KEY,
)
from ladder.rsl_rl.storage.transformer_vision_ppo_storage import CRITIC_KEY


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _observations(count=4):
    history = torch.randn(count, 25, 85) * 0.2
    return TensorDict({
        CURRENT_PROPRIO_KEY: history[:, -1].clone(), PROPRIO_HISTORY_KEY: history,
        DEPTH_HISTORY_KEY: torch.rand(count, 2, 60, 80),
        CRITIC_KEY: torch.randn(count, 7),
    }, batch_size=[count])


def _config():
    return {
        "num_steps_per_env": 3, "multi_gpu": None,
        "policy": {"actor_hidden_dims": [32, 16], "critic_hidden_dims": [16], "init_noise_std": 1.0},
        "algorithm": {"class_name": "unused", "num_learning_epochs": 1, "num_mini_batches": 1,
                      "sequence_micro_batch_envs": 2, "learning_rate": 3e-4, "schedule": "fixed",
                      "normalize_advantage_per_mini_batch": False, "rnd_cfg": None, "symmetry_cfg": None},
    }


def _algorithm(**settings):
    count = settings.pop("count", 4)
    torch.manual_seed(123)
    cfg = _config()
    cfg["algorithm"].update(settings)
    return EndToEndVisualPPO.construct_algorithm(
        _observations(count), SimpleNamespace(num_envs=count, num_actions=5), cfg, "cpu",
    )


def test_body_goal_contract_constructs_and_rejects_old_checkpoint(tmp_path):
    cfg = _config()
    cfg["policy"].update(use_contact_task=True, use_contact_task_phases=True,
                         use_contact_task_body_goal=True)
    obs = _observations()
    obs[ORACLE_CONTACT_GOAL_KEY] = torch.randn(4, 64)
    obs[CRITIC_KEY] = torch.randn(4, 135)
    alg = EndToEndVisualPPO.construct_algorithm(
        obs, SimpleNamespace(num_envs=4, num_actions=5), cfg, "cpu")
    assert alg.policy.actor_input_dim == 405
    assert alg.critic[0].in_features == 135
    assert alg.storage.actor_inputs.shape[-1] == 405
    assert alg.data_contract == BODY_GOAL_CONTACT_TASK_DATA_CONTRACT
    actions = alg.act(obs)
    assert actions.shape == (4, 5) and torch.isfinite(actions).all()
    alg.configure_diagnostics(tmp_path)
    import json
    manifest = json.loads((tmp_path / "end_to_end_visual_ppo_diagnostics/manifest.json").read_text())
    assert "body_goal_error_b3" in manifest["actor_input_layout"]
    with pytest.raises(RuntimeError, match="data contract"):
        alg.load({**alg.save(), "data_contract": PHASED_CONTACT_TASK_DATA_CONTRACT})


def test_body_goal_amp_retains_two_value_channels_and_updates():
    from ladder.rsl_rl.algorithms.end_to_end_visual_amp_ppo import EndToEndVisualAMPPPO
    from ladder.rsl_rl.runners.end_to_end_visual_amp_wrapper import AMP_STYLE_KEY
    cfg = _config()
    cfg["num_steps_per_env"] = 10
    cfg["policy"].update(use_contact_task=True, use_contact_task_phases=True,
                         use_contact_task_body_goal=True)
    cfg["algorithm"].update(amp_replay_capacity=64, amp_discriminator_batch_size=8,
                            amp_discriminator_updates=1)
    history = torch.randn(4, 10, 85) * 0.1
    obs = TensorDict({CURRENT_PROPRIO_KEY: history[:, -1], PROPRIO_HISTORY_KEY: history,
        DEPTH_HISTORY_KEY: torch.rand(4, 3, 60, 80), CRITIC_KEY: torch.randn(4, 135),
        ORACLE_CONTACT_GOAL_KEY: torch.randn(4, 64), AMP_STYLE_KEY: torch.randn(4, 71)}, [4])
    env = SimpleNamespace(num_envs=4, num_actions=5, unwrapped=SimpleNamespace(step_dt=0.02),
        sample_amp_expert_transitions=lambda n: torch.randn(n, 355),
        amp_reference_dataset=SimpleNamespace(observation_dim=71, num_frames=5, history_stride=2,
                                              anchor_hold_probability=0.2, balance_directions=True))
    alg = EndToEndVisualAMPPPO.construct_algorithm(obs, env, cfg, "cpu")
    assert alg.critic[0].in_features == 135
    assert alg.critic[-1].out_features == 2
    assert alg.policy.actor_input_dim == 405
    with torch.inference_mode():
        for _ in range(10):
            alg.act(obs)
            obs[AMP_STYLE_KEY] = torch.randn(4, 71)
            alg.process_env_step(obs, torch.randn(4), torch.zeros(4, dtype=torch.bool), {})
        alg.compute_returns(obs)
    assert alg.storage.rewards.shape == (10, 4, 2)
    torch.testing.assert_close(alg._reward_group_weights(), torch.tensor([1.0, 0.2]))
    metrics = alg.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def _playback_config_helper():
    path = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/play_transformer_vision_ppo.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_configure_end_to_end_playback"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_configure_end_to_end_playback"]


def _rollout(algorithm):
    obs = _observations()
    for step in range(algorithm.storage.num_steps):
        algorithm.act(obs)
        following = _observations()
        # A cached camera image still accompanies fresh proprioception.
        if step == 1:
            following[DEPTH_HISTORY_KEY] = obs[DEPTH_HISTORY_KEY].clone()
        done = torch.tensor([step == 0, False, step == 1, False])
        algorithm.process_env_step(following, torch.tensor([0.2, 0.6, -0.5, 0.9]), done, {})
        obs = following
    algorithm.compute_returns(obs)


def _objective(algorithm, terms):
    return (terms["surrogate"] + algorithm.value_loss_coef * terms["value"]
            - algorithm.entropy_coef * terms["entropy"])


def test_raw_sequence_replays_rollout_after_done_and_cached_depth():
    algorithm = _algorithm()
    _rollout(algorithm)
    ids = torch.arange(4)
    terms = algorithm._batch_terms(ids, algorithm._denominators(ids))
    assert terms["logprob_error"] < 5e-6
    assert terms["ratio_error"] < 5e-6
    assert terms["policy_kl"].abs() < 1e-6
    assert torch.equal(algorithm.storage.reset_before_mask()[1], algorithm.storage.dones[0])
    assert algorithm.storage.next_proprio_targets is None
    assert algorithm.storage.future_valid is None


def test_ppo_gradient_reaches_lightweight_visual_encoder_and_gru():
    algorithm = _algorithm()
    _rollout(algorithm)
    ids = torch.arange(4)
    terms = algorithm._batch_terms(ids, algorithm._denominators(ids))
    terms["surrogate"].backward()
    for prefix in ("encoder.state_encoder", "encoder.depth_encoder",
                   "encoder.depth_projection", "encoder.gru", "action_mlp"):
        grads = [parameter.grad for name, parameter in algorithm.policy.named_parameters()
                 if name.startswith(prefix + ".") and parameter.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert sum(float(grad.square().sum()) for grad in grads) > 0


def test_pending_transition_owns_observations_and_sampled_actions_without_labels():
    algorithm = _algorithm()
    obs = _observations()
    before = obs.clone()
    actions = algorithm.act(obs)
    original_actions = actions.clone()
    for value in obs.values():
        value.fill_(99)
    actions.fill_(99)
    algorithm.process_env_step(obs, torch.zeros(4), torch.zeros(4, dtype=torch.bool), {})
    st = algorithm.storage
    for actual, original in (
        (st.proprio_history[0], before[PROPRIO_HISTORY_KEY]),
        (st.depth_history[0], before[DEPTH_HISTORY_KEY]),
        (st.actor_inputs[0, :, :85], before[CURRENT_PROPRIO_KEY]),
        (st.critic_observations[0], before[CRITIC_KEY]),
        (st.actions[0], original_actions),
    ):
        torch.testing.assert_close(actual, original)
    assert st.velocity_targets is None
    assert st.limb_heightmaps is st.heightmap_valid is st.heightmap_hit_class is None
    assert st.next_proprio_targets is st.future_valid is None


def test_timeout_bootstrap_matches_rsl_and_excludes_simultaneous_failure():
    algorithm = _algorithm()
    algorithm._termination_manager = SimpleNamespace(terminated=torch.tensor([False, True, False, False]))
    algorithm.act(_observations())
    algorithm._pending.value.fill_(2)
    algorithm.process_env_step(None, torch.ones(4), torch.tensor([True, True, False, False]),
                               {"time_outs": torch.tensor([True, True, False, False])})
    torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([2.98, 1.0, 1.0, 1.0]))
    assert torch.equal(algorithm.visual_hidden_state[:, :2], torch.zeros(1, 2, 256))
    torch.testing.assert_close(algorithm.raw_rewards[0, :, 0], torch.ones(4))


def test_microbatch_losses_and_gradients_match_logical_batch():
    algorithm = _algorithm()
    _rollout(algorithm)
    ids = torch.arange(4)
    denominators = algorithm._denominators(ids)
    full = algorithm._batch_terms(ids, denominators)
    parameters = [*algorithm.policy.parameters(), *algorithm.critic.parameters()]
    expected = torch.autograd.grad(_objective(algorithm, full), parameters, allow_unused=True)
    algorithm.optimizer.zero_grad(set_to_none=True)
    summed = {}
    for micro in (ids[:1], ids[1:3], ids[3:]):
        terms = algorithm._batch_terms(micro, denominators)
        _objective(algorithm, terms).backward()
        for name in ("surrogate", "value", "entropy"):
            summed[name] = summed.get(name, 0.0) + terms[name].detach()
    for name, actual in summed.items():
        torch.testing.assert_close(actual, full[name], atol=2e-6, rtol=2e-5)
    for parameter, expected_grad in zip(parameters, expected):
        if expected_grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected_grad, atol=1e-6, rtol=2e-4)


def test_update_caches_reset_mask_and_preserves_logging_sums(monkeypatch):
    algorithm = _algorithm()
    _rollout(algorithm)
    original_mask = algorithm.storage.reset_before_mask
    mask_calls = []
    def counted_mask():
        mask_calls.append(True)
        return original_mask()
    monkeypatch.setattr(algorithm.storage, "reset_before_mask", counted_mask)
    original_terms = algorithm._batch_terms
    expected = {}
    def tracked_terms(ids, denominators, reset_before=None):
        terms = original_terms(ids, denominators, reset_before)
        for name, value in terms.items():
            if name not in ("ratio_error", "logprob_error"):
                expected[name] = expected.get(name, 0.0) + float(value.detach())
        expected["total"] = expected.get("total", 0.0) + float(_objective(algorithm, terms).detach())
        return terms
    monkeypatch.setattr(algorithm, "_batch_terms", tracked_terms)
    metrics = algorithm.update()
    assert len(mask_calls) == 1
    for name, value in expected.items():
        assert metrics[name] == value


def test_joint_update_diagnostics_and_checkpoint_contract(tmp_path):
    algorithm = _algorithm()
    algorithm.configure_diagnostics(tmp_path)
    _rollout(algorithm)
    before = {name: parameter.detach().clone() for name, parameter in algorithm.policy.named_parameters()}
    metrics = algorithm.update()
    for prefix in ("encoder.depth_encoder", "encoder.depth_projection", "encoder.gru", "action_mlp", "log_std"):
        assert any(not torch.equal(parameter, before[name])
                   for name, parameter in algorithm.policy.named_parameters()
                   if name == prefix or name.startswith(prefix + "."))
    assert metrics["ppo_encoder.depth_encoder_grad_l2_probe"] > 0
    assert metrics["ppo_encoder.depth_projection_grad_l2_probe"] > 0
    assert metrics["preupdate_ratio_max_abs_error_probe"] < 5e-6
    assert metrics["optimizer_steps"] == 1
    assert algorithm.num_updates == 1 and algorithm.storage.step == 0
    sample = torch.load(tmp_path / "end_to_end_visual_ppo_diagnostics/iteration_000000_sample.pt", weights_only=False)
    assert sample["initial_hidden_state"].shape == (1, 4, 256)
    assert sample["actions"].shape == (3, 4, 5)
    assert sample["parameter_timing"] == "before_optimizer_update"
    assert (tmp_path / "end_to_end_visual_ppo_diagnostics/metrics.jsonl").is_file()
    payload = deepcopy(algorithm.save())
    restored = _algorithm()
    assert restored.load(payload)
    assert restored.num_updates == 1
    assert restored.learning_rate == algorithm.learning_rate
    for key, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[key], value)
    assert next(iter(restored.optimizer.state.values()))["step"] == 1
    with pytest.raises(RuntimeError, match="data contract"):
        restored.load({**payload, "data_contract": "old_visual_ppo"})
    with pytest.raises(RuntimeError, match="training contract"):
        restored.load({**payload, "training_contract": {}})
    actor_only = _algorithm(gamma=0.95)
    assert not actor_only.load(payload, {"optimizer": False, "critic": False, "iteration": False})
    assert actor_only.num_updates == 0


def test_kl_sums_action_dimensions_and_adaptive_lr_uses_logical_batch():
    algorithm = _algorithm(schedule="adaptive")
    _rollout(algorithm)
    algorithm.storage.old_action_mean += 0.2
    ids = torch.arange(4)
    terms = algorithm._batch_terms(ids, algorithm._denominators(ids))
    assert float(terms["policy_kl"]) == pytest.approx(5 * 0.2**2 / 2, abs=1e-6)
    algorithm._adapt_learning_rate(float(terms["policy_kl"]))
    assert algorithm.learning_rate == pytest.approx(2e-4)
    assert all(group["lr"] == algorithm.learning_rate for group in algorithm.optimizer.param_groups)


def test_installed_runner_handles_inference_rollout_update_and_reload():
    from rsl_rl.runners import OnPolicyRunner

    class FakeVecEnv:
        num_envs = 4
        num_actions = 5
        device = "cpu"
        cfg = {}
        episode_length_buf = torch.zeros(4, dtype=torch.long)
        max_episode_length = 100

        def get_observations(self):
            return _observations()

        def step(self, actions):
            return _observations(), torch.randn(4), torch.zeros(4, dtype=torch.bool), {}

    cfg = _config()
    cfg.update({"save_interval": 500, "logger": "tensorboard"})
    cfg["algorithm"]["class_name"] = (
        "ladder.rsl_rl.algorithms.end_to_end_visual_ppo:EndToEndVisualPPO"
    )
    runner = OnPolicyRunner(FakeVecEnv(), cfg, log_dir=None, device="cpu")
    runner.learn(num_learning_iterations=2)
    assert runner.alg.num_updates == 2
    assert runner.alg.visual_hidden_state.is_inference()
    assert runner.alg.load(deepcopy(runner.alg.save()))
    assert not runner.alg.visual_hidden_state.is_inference()


def test_single_env_playback_clamps_training_minibatches_and_loads_actor_only():
    configure = _playback_config_helper()
    runner_cfg = {"algorithm": {"num_mini_batches": 4}}
    load_cfg = configure(runner_cfg, num_envs=1)
    assert runner_cfg["algorithm"]["num_mini_batches"] == 1
    assert load_cfg == {
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
    }

    source = _algorithm(num_mini_batches=4)
    playback = _algorithm(count=1, num_mini_batches=1)
    assert not playback.load(deepcopy(source.save()), load_cfg=load_cfg, strict=True)
    for name, value in source.policy.state_dict().items():
        torch.testing.assert_close(playback.policy.state_dict()[name], value)

    unchanged = {"algorithm": {"num_mini_batches": 4}}
    configure(unchanged, num_envs=8)
    assert unchanged["algorithm"]["num_mini_batches"] == 4
