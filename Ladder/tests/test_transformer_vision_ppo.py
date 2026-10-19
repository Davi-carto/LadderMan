from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner

from ladder.rsl_rl.algorithms import (
    TRANSFORMER_VISION_DATA_CONTRACT,
    TransformerVisionPPO,
    TransformerVisionPretrainer,
)
from ladder.rsl_rl.modules import TransformerVisionActor, TransformerVisionEncoder
from ladder.rsl_rl.runners import TransformerVisionPPOVecEnvWrapper
from ladder.rsl_rl.storage import TransformerVisionPPORolloutStorage


def _observations(num_envs: int = 2, critic_dim: int = 17) -> TensorDict:
    history = torch.rand(num_envs, 10, 85)
    return TensorDict(
        {
            "transformer_vision_current_proprio": history[:, -1].clone(),
            "transformer_vision_proprio_history": history,
            "transformer_vision_depth_history": torch.rand(num_envs, 2, 60, 80),
            "transformer_vision_limb_heightmaps": torch.rand(num_envs, 4, 10, 3),
            "transformer_vision_heightmap_valid": torch.ones(num_envs, 4, 10, 3),
            "transformer_vision_heightmap_hit_class": torch.randint(0, 5, (num_envs, 4, 10, 3)),
            "transformer_vision_velocity_target": torch.rand(num_envs, 3),
            "transformer_vision_next_proprio_target": torch.rand(num_envs, 85),
            "critic": torch.rand(num_envs, critic_dim),
        },
        batch_size=[num_envs],
    )


def _task_goal_observations(num_envs: int = 2, critic_dim: int = 17) -> TensorDict:
    observations = _observations(num_envs, critic_dim)
    observations.set(
        "transformer_vision_limb_goal_target",
        torch.rand(num_envs, 4, 3),
    )
    observations.set(
        "transformer_vision_limb_goal_valid",
        torch.ones(num_envs, 4),
    )
    observations.set(
        "transformer_vision_contact_indicator_target",
        torch.randint(0, 2, (num_envs, 4)).float(),
    )
    observations.set(
        "transformer_vision_motion_joint_pos_target",
        torch.rand(num_envs, 25),
    )
    observations.set(
        "transformer_vision_motion_root_orientation_6d_target",
        torch.rand(num_envs, 6),
    )
    observations.set(
        "transformer_vision_motion_target_valid",
        torch.ones(num_envs, dtype=torch.bool),
    )
    return observations


def _raw_observations(num_envs: int = 2, critic_dim: int = 17) -> dict:
    observations = _observations(num_envs, critic_dim)
    return {
        "transformer_vision_inputs": {
            "proprio_history": observations["transformer_vision_proprio_history"],
            "depth_history": observations["transformer_vision_depth_history"],
        },
        "transformer_vision_targets": {
            "relative_height": observations["transformer_vision_limb_heightmaps"],
            "hit_valid": observations["transformer_vision_heightmap_valid"],
            "hit_class": observations["transformer_vision_heightmap_hit_class"],
        },
        "transformer_vision_auxiliary_targets": {
            "base_lin_vel": observations["transformer_vision_velocity_target"],
            "next_proprio": observations["transformer_vision_next_proprio_target"],
        },
        "critic": observations["critic"],
    }


def _visual_checkpoint(tmp_path) -> str:
    encoder = TransformerVisionEncoder()
    payload = TransformerVisionPretrainer(encoder).state_dict()
    path = tmp_path / "visual_pretrain.pt"
    torch.save(payload, path)
    return str(path)


def _config(visual_checkpoint: str, num_steps: int = 2) -> dict:
    return {
        "num_steps_per_env": num_steps,
        "multi_gpu": None,
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [32, 16],
            "critic_hidden_dims": [32, 16],
        },
        "algorithm": {
            "class_name": "ladder.rsl_rl.algorithms:TransformerVisionPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.005,
            "learning_rate": 1.0e-3,
            "min_learning_rate": 1.0e-5,
            "max_learning_rate": 1.0e-3,
            "use_grad_scaler": True,
            "auxiliary_learning_rate": 1.0e-3,
            "auxiliary_mini_batch_envs": 2,
            "auxiliary_num_epochs": 1,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "visual_checkpoint": visual_checkpoint,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def test_constructs_actor_critic_and_visual_encoder_from_scratch():
    observations = _observations(critic_dim=780)
    environment = SimpleNamespace(num_envs=2, num_actions=25)
    config = _config("")
    config["algorithm"]["visual_initialization"] = "scratch"

    algorithm = TransformerVisionPPO.construct_algorithm(
        observations, environment, config, "cpu"
    )

    assert algorithm.visual_checkpoint == ""
    assert algorithm.policy.action_mlp[0].in_features == 136
    assert algorithm.critic[0].in_features == 780


def test_constructs_independent_task_goal_encoder_and_storage():
    observations = _task_goal_observations(critic_dim=780)
    environment = SimpleNamespace(num_envs=2, num_actions=25)
    config = _config("")
    config["algorithm"]["visual_initialization"] = "scratch"
    config["algorithm"]["visual_architecture"] = "task_goal"

    algorithm = TransformerVisionPPO.construct_algorithm(
        observations, environment, config, "cpu"
    )
    algorithm.act(observations)
    algorithm.process_env_step(
        observations,
        torch.ones(2),
        torch.zeros(2, dtype=torch.bool),
        {},
    )

    assert algorithm.policy.encoder_variant == "task_goal"
    assert algorithm.policy.action_mlp[0].in_features == 152
    assert algorithm.storage.actor_inputs.shape[-1] == 152
    assert algorithm.storage.task_goal_supervision
    torch.testing.assert_close(
        algorithm.storage.limb_goal_targets[0],
        observations["transformer_vision_limb_goal_target"],
    )
    torch.testing.assert_close(
        algorithm.storage.motion_joint_pos_targets[0],
        observations["transformer_vision_motion_joint_pos_target"],
    )


def _algorithm(tmp_path) -> TransformerVisionPPO:
    observations = _observations()
    environment = SimpleNamespace(num_envs=2, num_actions=25)
    return TransformerVisionPPO.construct_algorithm(
        observations,
        environment,
        _config(_visual_checkpoint(tmp_path)),
        "cpu",
    )


def _collect_two_steps(algorithm: TransformerVisionPPO) -> TensorDict:
    observations = _observations()
    algorithm.act(observations)
    next_observations = _observations()
    algorithm.process_env_step(
        next_observations,
        torch.ones(2),
        torch.tensor([False, True]),
        {},
    )
    assert torch.count_nonzero(algorithm.visual_hidden_state[:, 1]) == 0
    algorithm.act(next_observations)
    final_observations = _observations()
    algorithm.process_env_step(
        final_observations,
        torch.ones(2),
        torch.zeros(2, dtype=torch.bool),
        {},
    )
    return final_observations


def test_actor_uses_current_frame_plus_51_latent_and_resets_inference_state():
    actor = TransformerVisionActor(actor_hidden_dims=(32, 16)).eval()
    observations = _observations()

    actions = actor(observations)
    assert actions.shape == (2, 25)
    assert actor.action_mlp[0].in_features == 136
    assert actor._inference_hidden_state.shape == (1, 2, 256)
    assert torch.count_nonzero(actor._inference_hidden_state) > 0

    actor.reset(torch.tensor([False, True]))
    assert torch.count_nonzero(actor._inference_hidden_state[:, 0]) > 0
    assert torch.count_nonzero(actor._inference_hidden_state[:, 1]) == 0


def test_actor_diagnostic_forward_returns_the_output_used_by_the_action():
    torch.manual_seed(7)
    actor = TransformerVisionActor(actor_hidden_dims=(32, 16)).eval()
    observations = _observations()

    actor.reset()
    actions, output = actor.forward_with_output(observations)
    expected = actor.action_mean(
        torch.cat((observations["transformer_vision_current_proprio"], output.latent), dim=-1)
    )

    torch.testing.assert_close(actions, expected)
    assert output.velocity.shape == (2, 3)
    assert output.hand_left_map.shape == (2, 10, 3)
    assert output.next_hidden_state.shape == (1, 2, 256)


def test_ppo_uses_fixed_rollout_latent_and_optimizer_boundaries(tmp_path, monkeypatch):
    torch.manual_seed(19)
    algorithm = _algorithm(tmp_path)
    final_observations = _collect_two_steps(algorithm)
    assert algorithm.storage.actor_inputs.shape[-1] == 136
    torch.testing.assert_close(
        algorithm.storage.actor_inputs[0, :, :85],
        algorithm.storage.proprio_history[0, :, -1],
    )
    assert algorithm.rollout_log_prob_max_error().item() < 1.0e-5
    algorithm.compute_returns(final_observations)

    visual_before = [parameter.detach().clone() for parameter in algorithm.policy.encoder.parameters()]
    with monkeypatch.context() as patch:
        patch.setattr(algorithm, "auxiliary_grad_scaler", object())
        patch.setattr(
            torch,
            "autocast",
            lambda *args, **kwargs: pytest.fail("FP32 PPO must not enter autocast"),
        )
        algorithm._ppo_update()
    assert torch.all(algorithm.policy.output_std >= 0.2)
    for parameter, previous in zip(algorithm.policy.encoder.parameters(), visual_before, strict=True):
        torch.testing.assert_close(parameter, previous)

    actor_before_aux = [parameter.detach().clone() for parameter in algorithm.policy.action_mlp.parameters()]
    auxiliary = algorithm._auxiliary_update()
    assert auxiliary["aux_total"] > 0.0
    assert auxiliary["aux_optimizer_steps"] == 1.0
    assert any(
        not torch.equal(parameter, previous)
        for parameter, previous in zip(algorithm.policy.encoder.parameters(), visual_before, strict=True)
    )
    for parameter, previous in zip(algorithm.policy.action_mlp.parameters(), actor_before_aux, strict=True):
        torch.testing.assert_close(parameter, previous)


def test_auxiliary_logical_batches_match_five_by_four_schedule():
    storage = TransformerVisionPPORolloutStorage(
        num_steps=1,
        num_envs=16,
        action_dim=25,
        critic_dim=17,
        device="cpu",
    )

    batches = list(storage.auxiliary_logical_env_batches(4, 5))

    assert len(batches) == 20
    for epoch in range(5):
        indexes = torch.cat(batches[epoch * 4 : (epoch + 1) * 4])
        assert torch.equal(torch.sort(indexes).values, torch.arange(16))


def test_checkpoint_reload_preserves_deterministic_visual_policy(tmp_path):
    torch.manual_seed(20)
    algorithm = _algorithm(tmp_path)
    final_observations = _collect_two_steps(algorithm)
    algorithm.compute_returns(final_observations)
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())

    payload = algorithm.save()
    assert payload["data_contract"] == TRANSFORMER_VISION_DATA_CONTRACT
    assert "grad_scaler_state_dict" in payload
    restored = _algorithm(tmp_path)
    restored.load(payload, load_cfg=None, strict=True)
    algorithm.reset_inference()
    restored.reset_inference()
    torch.testing.assert_close(
        algorithm.act_inference(final_observations),
        restored.act_inference(final_observations),
    )


def test_vector_wrapper_exposes_current_frame_without_old_660d_policy():
    raw_observations = _raw_observations()
    raw_observations["transformer_vision_inputs"]["proprio_history"][:, -1, 0] = 200.0
    raw_observations["transformer_vision_inputs"]["depth_history"][:, :, 0, 0] = 200.0
    raw_observations["critic"][:, 0] = -200.0
    raw_observations["transformer_vision_targets"]["relative_height"][:, 0, 0, 0] = -200.0
    raw_observations["transformer_vision_auxiliary_targets"]["next_proprio"][:, 0] = 200.0

    class _FakeEnv:
        def __init__(self):
            self.last_actions = None
            self.unwrapped = SimpleNamespace(
                num_envs=2,
                device="cpu",
                max_episode_length=100,
                action_manager=SimpleNamespace(total_action_dim=25),
                cfg=SimpleNamespace(is_finite_horizon=True),
                episode_length_buf=torch.zeros(2, dtype=torch.long),
                observation_manager=SimpleNamespace(compute=lambda: raw_observations),
            )

        def reset(self):
            return raw_observations, {}

        def step(self, actions):
            self.last_actions = actions
            return (
                raw_observations,
                torch.ones(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

        def close(self):
            pass

    fake_env = _FakeEnv()
    wrapper = TransformerVisionPPOVecEnvWrapper(
        fake_env,
        clip_actions=10.0,
        clip_observations=100.0,
    )
    packed = wrapper.get_observations()
    assert "policy" not in packed
    assert packed["transformer_vision_current_proprio"].shape == (2, 85)
    torch.testing.assert_close(
        packed["transformer_vision_current_proprio"],
        packed["transformer_vision_proprio_history"][:, -1],
    )
    assert torch.all(packed["transformer_vision_current_proprio"][:, 0] == 100.0)
    assert torch.all(packed["transformer_vision_depth_history"][:, :, 0, 0] == 100.0)
    assert torch.all(packed["critic"][:, 0] == -100.0)
    assert torch.all(packed["transformer_vision_limb_heightmaps"][:, 0, 0, 0] == -200.0)
    assert torch.all(packed["transformer_vision_next_proprio_target"][:, 0] == 200.0)
    wrapper.step(torch.full((2, 25), 20.0))
    assert torch.all(fake_env.last_actions == 10.0)


def test_vector_wrapper_actor_only_does_not_require_privileged_groups():
    raw_observations = _raw_observations()
    actor_observations = {
        "transformer_vision_inputs": raw_observations["transformer_vision_inputs"]
    }

    class _FakeEnv:
        def __init__(self):
            self.unwrapped = SimpleNamespace(
                num_envs=2,
                device="cpu",
                max_episode_length=100,
                action_manager=SimpleNamespace(total_action_dim=25),
                cfg=SimpleNamespace(is_finite_horizon=True),
                episode_length_buf=torch.zeros(2, dtype=torch.long),
                observation_manager=SimpleNamespace(compute=lambda: actor_observations),
            )

        def reset(self):
            return actor_observations, {}

        def step(self, actions):
            return (
                actor_observations,
                torch.ones(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

        def close(self):
            pass

    wrapper = TransformerVisionPPOVecEnvWrapper(_FakeEnv())
    wrapper.actor_only = True

    packed = wrapper.get_observations()

    assert set(packed.keys()) == {
        "transformer_vision_current_proprio",
        "transformer_vision_proprio_history",
        "transformer_vision_depth_history",
    }
    assert packed["transformer_vision_current_proprio"].shape == (2, 85)
    assert wrapper.step(torch.zeros(2, 25))[0].batch_size == torch.Size([2])


def test_adaptive_learning_rate_respects_configured_bounds(tmp_path):
    algorithm = _algorithm(tmp_path)
    old_mean = torch.zeros(2, 25)
    old_std = torch.ones(2, 25)

    algorithm.learning_rate = 9.0e-4
    algorithm._adapt_learning_rate(
        old_mean,
        old_std,
        torch.full_like(old_mean, 0.01),
        old_std,
    )
    assert algorithm.learning_rate == 1.0e-3
    assert algorithm.ppo_optimizer.param_groups[0]["lr"] == 1.0e-3

    algorithm.learning_rate = 1.1e-5
    algorithm._adapt_learning_rate(
        old_mean,
        old_std,
        torch.ones_like(old_mean),
        old_std,
    )
    assert algorithm.learning_rate == 1.0e-5
    assert algorithm.ppo_optimizer.param_groups[0]["lr"] == 1.0e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GradScaler")
def test_cuda_grad_scaler_is_reserved_for_visual_updates(tmp_path):
    observations = _observations().to("cuda:0")
    environment = SimpleNamespace(num_envs=2, num_actions=25)
    algorithm = TransformerVisionPPO.construct_algorithm(
        observations,
        environment,
        _config(_visual_checkpoint(tmp_path), num_steps=1),
        "cuda:0",
    )

    assert algorithm.auxiliary_grad_scaler.is_enabled()
    algorithm.act(observations)
    next_observations = _observations().to("cuda:0")
    algorithm.process_env_step(
        next_observations,
        torch.ones(2, device="cuda:0"),
        torch.zeros(2, dtype=torch.bool, device="cuda:0"),
        {},
    )
    algorithm.compute_returns(next_observations)
    scale_before_ppo = algorithm.auxiliary_grad_scaler.get_scale()
    ppo_losses = algorithm._ppo_update()
    assert algorithm.auxiliary_grad_scaler.get_scale() == scale_before_ppo
    auxiliary_losses = algorithm._auxiliary_update()
    losses = {**ppo_losses, **auxiliary_losses}

    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_constructs_through_installed_rsl_runner(tmp_path):
    raw_observations = _raw_observations()

    class _FakeEnv:
        def __init__(self):
            self.unwrapped = SimpleNamespace(
                num_envs=2,
                device="cpu",
                max_episode_length=100,
                action_manager=SimpleNamespace(total_action_dim=25),
                cfg=SimpleNamespace(is_finite_horizon=True),
                episode_length_buf=torch.zeros(2, dtype=torch.long),
                observation_manager=SimpleNamespace(compute=lambda: raw_observations),
            )

        def reset(self):
            return raw_observations, {}

        def step(self, actions):
            return (
                raw_observations,
                torch.ones(2),
                torch.zeros(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

        def close(self):
            pass

    config = _config(_visual_checkpoint(tmp_path))
    config.update({"save_interval": 100, "logger": "tensorboard"})
    runner = OnPolicyRunner(
        TransformerVisionPPOVecEnvWrapper(_FakeEnv()),
        config,
        log_dir=None,
        device="cpu",
    )
    runner.learn(num_learning_iterations=1)
    assert runner.current_learning_iteration == 0
