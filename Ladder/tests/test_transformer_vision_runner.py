from types import SimpleNamespace

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution

from ladder.rsl_rl.runners import FrozenPolicyVisionRunner, FrozenRslPolicy, TransformerVisionVecEnvWrapper
from ladder.rsl_rl.modules import TaskGoalTransformerVisionEncoder


def _raw_observations(num_envs: int = 2) -> dict:
    return {
        "policy": torch.rand(num_envs, 660),
        "critic": torch.rand(num_envs, 631),
        "transformer_vision_inputs": {
            "proprio_history": torch.rand(num_envs, 10, 85),
            "depth_history": torch.rand(num_envs, 2, 60, 80),
        },
        "transformer_vision_targets": {
            "relative_height": torch.rand(num_envs, 4, 10, 3),
            "hit_valid": torch.ones(num_envs, 4, 10, 3),
            "hit_class": torch.ones(num_envs, 4, 10, 3, dtype=torch.long),
        },
        "transformer_vision_auxiliary_targets": {
            "base_lin_vel": torch.rand(num_envs, 3),
            "next_proprio": torch.rand(num_envs, 85),
        },
    }


def _task_goal_raw_observations(num_envs: int = 2) -> dict:
    observations = _raw_observations(num_envs)
    observations["policy"] = torch.rand(num_envs, 780)
    observations["transformer_vision_auxiliary_targets"].update(
        {
            "task_limb_goal_b": torch.rand(num_envs, 12),
            "task_limb_goal_valid": torch.ones(num_envs, 4),
            "task_contact_indicator": torch.randint(0, 2, (num_envs, 4)).float(),
            "task_motion_joint_pos": torch.rand(num_envs, 25),
            "task_motion_root_orientation_6d": torch.rand(num_envs, 6),
            "task_motion_valid": torch.ones(num_envs, 1),
        }
    )
    return observations


def _actor(observations: torch.Tensor) -> MLPModel:
    return MLPModel(
        TensorDict({"policy": observations}, batch_size=[observations.shape[0]]),
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=25,
        hidden_dims=(512, 256, 256),
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": GaussianDistribution, "init_std": 1.0, "std_type": "scalar"},
    )


def test_frozen_split_checkpoint_uses_deterministic_rsl_actor(tmp_path):
    torch.manual_seed(11)
    policy_obs = torch.rand(3, 660)
    reference = _actor(policy_obs).eval()
    checkpoint = tmp_path / "split_checkpoint.pt"
    torch.save({"actor_state_dict": reference.state_dict(), "critic_state_dict": {}}, checkpoint)

    restored = FrozenRslPolicy(policy_obs, checkpoint)

    with torch.no_grad():
        expected = reference(TensorDict({"policy": policy_obs}, batch_size=[3]))
        actual = restored(policy_obs)
    torch.testing.assert_close(actual, expected)
    assert not any(parameter.requires_grad for parameter in restored.parameters())


def test_frozen_split_checkpoint_infers_full_state_observation_dimension(tmp_path):
    policy_obs = torch.rand(2, 780)
    reference = _actor(policy_obs).eval()
    checkpoint = tmp_path / "full_state_checkpoint.pt"
    torch.save({"actor_state_dict": reference.state_dict(), "critic_state_dict": {}}, checkpoint)

    restored = FrozenRslPolicy(policy_obs, checkpoint)

    assert restored.observation_dim == 780
    with torch.no_grad():
        expected = reference(TensorDict({"policy": policy_obs}, batch_size=[2]))
        actual = restored(policy_obs)
    torch.testing.assert_close(actual, expected)


def test_frozen_full_state_runner_trains_task_goal_visual_head_only(tmp_path):
    observations = _task_goal_raw_observations()

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

    reference = _actor(observations["policy"])
    actor_checkpoint = tmp_path / "full_state_actor.pt"
    torch.save({"actor_state_dict": reference.state_dict(), "critic_state_dict": {}}, actor_checkpoint)
    runner = FrozenPolicyVisionRunner(
        TransformerVisionVecEnvWrapper(_FakeEnv()),
        actor_checkpoint,
        rollout_steps=2,
        encoder=TaskGoalTransformerVisionEncoder(),
    )

    result = runner.collect_and_update()

    assert result.actor_parameter_change_max_abs == 0.0
    assert result.vision_parameter_change_max_abs > 0.0
    assert "limb_goal" in result.losses
    assert "contact_bce" in result.losses
    assert "motion_joint" in result.losses
    assert "motion_root_orientation" in result.losses
    assert "future" not in result.losses


def test_wrapper_preserves_frozen_actor_and_visual_groups():
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

    wrapper = TransformerVisionVecEnvWrapper(_FakeEnv())
    packed = wrapper.get_observations()
    stepped, rewards, dones, _ = wrapper.step(torch.zeros(2, 25))

    assert packed["policy"].shape == (2, 660)
    assert packed["transformer_vision_proprio_history"].shape == (2, 10, 85)
    assert packed["transformer_vision_depth_history"].shape == (2, 2, 60, 80)
    assert stepped["transformer_vision_limb_heightmaps"].shape == (2, 4, 10, 3)
    assert stepped["transformer_vision_heightmap_hit_class"].shape == (2, 4, 10, 3)
    assert rewards.shape == (2,)
    assert dones.dtype == torch.bool


def test_frozen_runner_updates_only_transformer_vision(tmp_path):
    torch.manual_seed(13)
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

    reference = _actor(observations["policy"])
    actor_checkpoint = tmp_path / "frozen_actor.pt"
    torch.save({"actor_state_dict": reference.state_dict(), "critic_state_dict": {}}, actor_checkpoint)
    runner = FrozenPolicyVisionRunner(
        TransformerVisionVecEnvWrapper(_FakeEnv()),
        actor_checkpoint,
        rollout_steps=2,
    )
    before_actor = [parameter.detach().clone() for parameter in runner.frozen_policy.parameters()]

    result = runner.collect_and_update()
    checkpoint = tmp_path / "vision.pt"
    runner.save_visual_checkpoint(checkpoint, iteration=1)
    restored_iteration = runner.load_visual_checkpoint(checkpoint)

    assert result.actor_parameter_change_max_abs == 0.0
    assert result.vision_parameter_change_max_abs > 0.0
    assert result.vision_gradient_norm > 0.0
    assert restored_iteration == 1
    for parameter, previous in zip(runner.frozen_policy.parameters(), before_actor, strict=True):
        torch.testing.assert_close(parameter, previous)


def test_frozen_runner_carries_gru_state_between_rollouts_and_resets_done_envs(tmp_path):
    torch.manual_seed(14)
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
            return observations, torch.ones(2), torch.tensor([False, True]), torch.zeros(2, dtype=torch.bool), {}

        def close(self):
            pass

    reference = _actor(observations["policy"])
    actor_checkpoint = tmp_path / "frozen_actor.pt"
    torch.save({"actor_state_dict": reference.state_dict(), "critic_state_dict": {}}, actor_checkpoint)
    runner = FrozenPolicyVisionRunner(
        TransformerVisionVecEnvWrapper(_FakeEnv()),
        actor_checkpoint,
        rollout_steps=1,
    )

    callback_steps = []
    callback_shapes = []

    def record_step(step, output, packed_observations):
        callback_steps.append(step)
        callback_shapes.append(
            (
                tuple(output.hand_left_map.shape),
                tuple(packed_observations["transformer_vision_limb_heightmaps"].shape),
            )
        )

    runner._collect_trajectory(step_callback=record_step)
    first_rollout_end = runner._visual_hidden_state.clone()
    assert callback_steps == [0]
    assert callback_shapes == [((2, 10, 3), (2, 4, 10, 3))]
    assert torch.count_nonzero(first_rollout_end[:, 0]) > 0
    assert torch.count_nonzero(first_rollout_end[:, 1]) == 0

    runner._collect_trajectory()
    torch.testing.assert_close(runner.storage.initial_hidden_state, first_rollout_end)


def test_manual_reset_clears_gru_before_next_action_and_marks_sequence_boundary(tmp_path):
    torch.manual_seed(15)
    observations = _raw_observations()

    class _FakeEnv:
        def __init__(self):
            self.reset_count = 0
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
            self.reset_count += 1
            return observations, {}

        def step(self, actions):
            return observations, torch.ones(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}

        def close(self):
            pass

    fake_env = _FakeEnv()
    reference = _actor(observations["policy"])
    actor_checkpoint = tmp_path / "frozen_actor.pt"
    torch.save({"actor_state_dict": reference.state_dict(), "critic_state_dict": {}}, actor_checkpoint)
    runner = FrozenPolicyVisionRunner(
        TransformerVisionVecEnvWrapper(fake_env),
        actor_checkpoint,
        rollout_steps=2,
    )
    runner._visual_hidden_state.fill_(7.0)
    reset_checks = 0
    hidden_at_callback = []

    def request_reset_on_second_step():
        nonlocal reset_checks
        reset_checks += 1
        return reset_checks == 3

    def record_hidden(step, output, packed_observations):
        hidden_at_callback.append(runner._visual_hidden_state.clone())

    _, _, dones = runner._collect_trajectory(
        step_callback=record_hidden,
        reset_requested=request_reset_on_second_step,
    )

    assert fake_env.reset_count == 2
    assert torch.count_nonzero(hidden_at_callback[0]) > 0
    assert torch.count_nonzero(hidden_at_callback[1]) == 0
    assert torch.all(dones[0])
    assert torch.all(runner.storage.dones[0])
    assert torch.count_nonzero(runner.storage.future_valid[0]) == 0
