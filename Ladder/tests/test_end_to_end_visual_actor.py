"""CPU checks for the lightweight raw-sequence visual policy."""

import pytest
import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from ladder.rsl_rl.modules.end_to_end_visual_actor import (
    END_TO_END_VISUAL_DATA_CONTRACT,
    CONTACT_TASK_DATA_CONTRACT,
    PHASED_CONTACT_TASK_DATA_CONTRACT,
    BODY_GOAL_CONTACT_TASK_DATA_CONTRACT,
    EndToEndVisualActor,
)
from ladder.rsl_rl.modules.transformer_vision_actor import (
    CURRENT_PROPRIO_KEY,
    DEPTH_HISTORY_KEY,
    PROPRIO_HISTORY_KEY,
)


@pytest.fixture(autouse=True)
def _small_cpu_test_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(37)
    yield
    torch.set_num_threads(previous)


def _sequence():
    dones = torch.tensor([
        [True, False, False],
        [False, True, False],
        [False, False, False],
        [False, False, True],
    ])
    histories = []
    history = torch.randn(3, 25, 85)
    history[..., -1] = torch.tensor([1.0, -1.0, 0.0])[:, None]
    for step in range(4):
        frame = torch.randn(3, 85)
        frame[:, -1] = history[:, -1, -1]
        shifted = torch.cat((history[:, 1:], frame[:, None]), dim=1)
        history = torch.where(dones[step, :, None, None], frame[:, None], shifted)
        histories.append(history)
    proprio = torch.stack(histories)
    depth = torch.rand(4, 3, 2, 60, 80)
    depth[1] = depth[0]
    depth[3] = depth[2]
    return proprio[:, :, -1], proprio, depth, torch.randn(1, 3, 256), dones


def test_network_is_policy_only_cnn_gru_and_keeps_actor_layers():
    actor = EndToEndVisualActor(action_dim=25)
    assert actor.data_contract == actor.encoder.data_contract == END_TO_END_VISUAL_DATA_CONTRACT
    assert actor.encoder.history_frames == 25
    assert actor.encoder.actor_latent_dim == 256
    assert actor.actor_input_dim == 341
    assert [module.out_features for module in actor.action_mlp if isinstance(module, nn.Linear)] == [
        512, 256, 128, 25,
    ]
    assert [module.out_channels for module in actor.encoder.depth_encoder if isinstance(module, nn.Conv2d)] == [
        16, 32, 64,
    ]
    assert actor.encoder.depth_pool.output_size == (2, 4)
    assert actor.encoder.depth_projection[0].out_features == 128
    assert actor.encoder.state_encoder[-1].out_features == 128
    assert actor.encoder.gru.input_size == actor.encoder.gru.hidden_size == 256
    forbidden = ("transformer", "decoder", "velocity_head", "hand_left_head", "foot_left_head")
    assert not any(any(term in name for term in forbidden) for name in actor.encoder.state_dict())

    mean, output = actor.forward_sequence(*_sequence())
    assert mean.shape == (4, 3, 25)
    assert output.recurrent_feature.shape == output.latent.shape == (4, 3, 256)
    assert output.valid_mask.shape == (4, 3)
    assert output.next_hidden_state.shape == (1, 3, 256)


@pytest.mark.parametrize("phases,body_goal,width,contract", [
    (False, False, 57, CONTACT_TASK_DATA_CONTRACT),
    (True, False, 61, PHASED_CONTACT_TASK_DATA_CONTRACT),
    (True, True, 64, BODY_GOAL_CONTACT_TASK_DATA_CONTRACT),
])
def test_contact_task_actor_contracts_remain_distinct(phases, body_goal, width, contract):
    actor = EndToEndVisualActor(
        use_contact_task=True, use_contact_task_phases=phases,
        use_contact_task_body_goal=body_goal,
    )
    assert actor.oracle_input_dim == width
    assert actor.actor_input_dim == 341 + width
    assert actor.data_contract == actor.encoder.data_contract == contract
    current, proprio, depth, hidden, dones = _sequence()
    task_state = torch.randn(4, 3, width)
    mean, _ = actor.forward_sequence(current, proprio, depth, hidden, dones, task_state)
    assert mean.shape == (4, 3, 25)
    with pytest.raises(ValueError, match="oracle_contact_goal must have shape"):
        actor.make_actor_input(current, torch.zeros(4, 3, 256), torch.zeros(4, 3, width - 1))


def test_body_goal_actor_requires_phased_contact_task():
    with pytest.raises(ValueError, match="Body goal requires use_contact_task_phases"):
        EndToEndVisualActor(use_contact_task=True, use_contact_task_body_goal=True)


def test_sequence_matches_play_step_api_across_resets_and_repeated_depth():
    actor = EndToEndVisualActor(action_dim=4, actor_hidden_dims=[32, 16]).train()
    current, proprio, depth, initial_hidden, dones = _sequence()
    sequence_mean, sequence_output = actor.forward_sequence(
        current, proprio, depth, initial_hidden, dones
    )
    actor._inference_hidden_state = initial_hidden.clone()
    step_means, step_features, hidden_states = [], [], []
    for step in range(4):
        actor.reset(dones[step])
        observations = TensorDict(
            {
                CURRENT_PROPRIO_KEY: current[step],
                PROPRIO_HISTORY_KEY: proprio[step],
                DEPTH_HISTORY_KEY: depth[step],
            },
            batch_size=[3],
        )
        mean, output = actor.forward_with_output(observations)
        step_means.append(mean)
        step_features.append(output.recurrent_feature)
        hidden_states.append(output.next_hidden_state)
    torch.testing.assert_close(sequence_mean, torch.stack(step_means), atol=2.0e-6, rtol=2.0e-5)
    torch.testing.assert_close(
        sequence_output.recurrent_feature,
        torch.stack(step_features),
        atol=2.0e-6,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        sequence_output.next_hidden_state, hidden_states[-1], atol=2.0e-6, rtol=2.0e-5
    )
    assert not torch.equal(hidden_states[1][:, 0], hidden_states[0][:, 0])


def test_ppo_surrogate_reaches_all_policy_encoder_stages_and_actor():
    actor = EndToEndVisualActor(action_dim=4, actor_hidden_dims=[32, 16]).train()
    mean, _ = actor.forward_sequence(*_sequence())
    distribution = Normal(mean, actor.output_std.expand_as(mean))
    actions = distribution.sample()
    old_log_prob = distribution.log_prob(actions).sum(-1).detach()
    advantages = torch.linspace(-1.0, 1.0, 12).reshape(4, 3)
    ratio = (distribution.log_prob(actions).sum(-1) - old_log_prob).exp()
    loss = (-advantages * ratio).mean()
    named = list(actor.named_parameters())
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in named], allow_unused=True)
    for prefix in (
        "encoder.state_encoder",
        "encoder.depth_encoder",
        "encoder.depth_projection",
        "encoder.gru",
        "action_mlp",
    ):
        squared_norm = sum(
            float(gradient.square().sum())
            for (name, _), gradient in zip(named, gradients)
            if name.startswith(prefix + ".") and gradient is not None
        )
        assert squared_norm > 0, prefix


def test_play_rejects_old_ten_frame_input_and_uses_declared_noise_std():
    actor = EndToEndVisualActor(action_dim=4, actor_hidden_dims=[32, 16], init_noise_std=0.05)
    torch.testing.assert_close(actor.output_std, torch.full((4,), 0.05))
    obs = TensorDict(
        {
            CURRENT_PROPRIO_KEY: torch.zeros(1, 85),
            PROPRIO_HISTORY_KEY: torch.zeros(1, 10, 85),
            DEPTH_HISTORY_KEY: torch.zeros(1, 2, 60, 80),
        },
        batch_size=[1],
    )
    with pytest.raises(ValueError, match="25"):
        actor(obs)
