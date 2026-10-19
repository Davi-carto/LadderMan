from types import SimpleNamespace

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from ladder.algorithms import LadderSSRPPO
from ladder.runners import LadderSSRVecEnvWrapper


def _observations(num_envs: int = 2, critic_dim: int = 17) -> TensorDict:
    return TensorDict(
        {
            "ssr_proprio_history": torch.rand(num_envs, 10, 85),
            "ssr_depth_history": torch.rand(num_envs, 2, 60, 80),
            "ssr_limb_heightmaps": torch.rand(num_envs, 4, 10, 3),
            "ssr_heightmap_valid": torch.ones(num_envs, 4, 10, 3),
            "ssr_base_lin_vel": torch.rand(num_envs, 3),
            "ssr_next_proprio": torch.rand(num_envs, 85),
            "critic": torch.rand(num_envs, critic_dim),
        },
        batch_size=[num_envs],
    )


def _raw_observations(num_envs: int = 2, critic_dim: int = 17) -> dict:
    observations = _observations(num_envs, critic_dim)
    return {
        "visual_encoder_inputs": {
            "proprio_history": observations["ssr_proprio_history"],
            "depth_history": observations["ssr_depth_history"],
        },
        "visual_targets": {
            "relative_height": observations["ssr_limb_heightmaps"],
            "hit_valid": observations["ssr_heightmap_valid"],
        },
        "visual_aux_targets": {
            "base_lin_vel": observations["ssr_base_lin_vel"],
            "next_proprio": observations["ssr_next_proprio"],
        },
        "critic": observations["critic"],
    }


def _config(num_steps: int = 2) -> dict:
    return {
        "num_steps_per_env": num_steps,
        "multi_gpu": None,
        "policy": {
            "init_noise_std": 1.0,
            "critic_hidden_dims": [32, 16],
        },
        "algorithm": {
            "class_name": "ladder.algorithms:LadderSSRPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.005,
            "learning_rate": 1.0e-3,
            "auxiliary_learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "enable_auxiliary_update": True,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "normalize_advantage_per_mini_batch": False,
        },
    }


def _algorithm() -> LadderSSRPPO:
    observations = _observations()
    environment = SimpleNamespace(num_envs=2, num_actions=25)
    return LadderSSRPPO.construct_algorithm(observations, environment, _config(), "cpu")


def test_ssr_ppo_keeps_rollout_latent_fixed_and_records_post_step_target(tmp_path):
    torch.manual_seed(17)
    algorithm = _algorithm()
    observations = _observations()

    first_actions = algorithm.act(observations)
    first_next_observations = _observations()
    first_done = torch.tensor([False, True])
    algorithm.process_env_step(first_next_observations, torch.ones(2), first_done, {})

    assert first_actions.shape == (2, 25)
    assert algorithm.storage.actor_inputs.shape[-1] == 85 + 51
    assert algorithm.storage.future_valid[0].tolist() == [True, False]
    torch.testing.assert_close(
        algorithm.storage.next_proprio_targets[0],
        first_next_observations["ssr_next_proprio"],
    )
    assert torch.count_nonzero(algorithm.hidden_state[:, 1]) == 0

    algorithm.act(first_next_observations)
    final_observations = _observations()
    algorithm.process_env_step(final_observations, torch.ones(2), torch.zeros(2, dtype=torch.bool), {})
    assert algorithm.rollout_log_prob_max_error().item() < 1.0e-5

    algorithm.compute_returns(final_observations)
    losses = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)).item() for value in losses.values())
    assert losses["aux_total"] > 0.0

    checkpoint = tmp_path / "ssr_ppo.pt"
    torch.save(algorithm.save(), checkpoint)
    restored = _algorithm()
    restored.load(torch.load(checkpoint, weights_only=False), load_cfg=None, strict=True)
    algorithm.eval_mode()
    restored.eval_mode()
    algorithm.reset_inference()
    restored.reset_inference()
    torch.testing.assert_close(algorithm.act_inference(final_observations), restored.act_inference(final_observations))


def test_ssr_vecenv_wrapper_preserves_named_observation_groups():
    observations = _raw_observations()

    class _FakeEnv:
        def __init__(self):
            self.unwrapped = SimpleNamespace(
                num_envs=2,
                device="cpu",
                max_episode_length=100,
                action_manager=SimpleNamespace(total_action_dim=25),
                cfg=SimpleNamespace(is_finite_horizon=True),
                episode_length_buf=torch.zeros(2, dtype=torch.long),
                observation_manager=SimpleNamespace(compute=lambda: observations),
            )

        def reset(self):
            return observations, {}

        def step(self, actions):
            return observations, torch.ones(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}

        def close(self):
            pass

    wrapper = LadderSSRVecEnvWrapper(_FakeEnv())
    packed = wrapper.get_observations()
    stepped, rewards, dones, _ = wrapper.step(torch.zeros(2, 25))

    assert packed["ssr_proprio_history"].shape == (2, 10, 85)
    assert stepped["ssr_limb_heightmaps"].shape == (2, 4, 10, 3)
    assert rewards.shape == (2,)
    assert dones.dtype == torch.long


def test_ssr_ppo_constructs_and_updates_through_rsl_runner():
    observations = _raw_observations()

    class _FakeEnv:
        def __init__(self):
            self.unwrapped = SimpleNamespace(
                num_envs=2,
                device="cpu",
                max_episode_length=100,
                action_manager=SimpleNamespace(total_action_dim=25),
                cfg=SimpleNamespace(is_finite_horizon=True),
                episode_length_buf=torch.zeros(2, dtype=torch.long),
                observation_manager=SimpleNamespace(compute=lambda: observations),
            )

        def reset(self):
            return observations, {}

        def step(self, actions):
            return observations, torch.ones(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}

        def close(self):
            pass

    config = _config()
    config.update({"save_interval": 100, "algorithm": config["algorithm"], "logger": "tensorboard"})
    runner = OnPolicyRunner(LadderSSRVecEnvWrapper(_FakeEnv()), config, log_dir=None, device="cpu")

    runner.learn(num_learning_iterations=1)

    assert runner.current_learning_iteration == 0
