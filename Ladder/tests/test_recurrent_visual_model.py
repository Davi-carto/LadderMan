import torch

from ladder.models import LadderSSRVisualActor, LadderVisualTargets


PROPRIO_DIM = 85


def _rollout_inputs(batch_size: int = 3):
    return (
        torch.rand(batch_size, 10, PROPRIO_DIM),
        torch.rand(batch_size, 2, 60, 80),
    )


def test_recurrent_visual_actor_rollout_shapes_and_token_geometry():
    model = LadderSSRVisualActor(proprio_dim=PROPRIO_DIM).eval()
    obs_history, depth = _rollout_inputs()

    output = model(obs_history, depth, deterministic=True)

    assert output.state_token.shape == (3, 1, 64)
    assert output.visual_tokens.shape == (3, 10, 64)
    assert output.transformer_tokens.shape == (3, 11, 64)
    assert output.fused_features.shape == (3, 256)
    assert output.recurrent_features.shape == (3, 256)
    assert output.actor_latent.shape == (3, 51)
    assert output.actor_input.shape == (3, PROPRIO_DIM + 51)
    assert output.actions.shape == (3, 25)
    assert output.hand_left_map.shape == (3, 10, 3)
    assert output.hand_right_map.shape == (3, 10, 3)
    assert output.foot_left_map.shape == (3, 10, 3)
    assert output.foot_right_map.shape == (3, 10, 3)
    assert output.next_proprio.shape == (3, PROPRIO_DIM)
    assert output.next_hidden_state.shape == (1, 3, 256)
    assert torch.equal(output.dynamic_latent, output.dynamic_mu)
    assert model.action_distribution(output).sample().shape == (3, 25)


def test_recurrent_visual_actor_sequence_and_selected_hidden_reset():
    torch.manual_seed(7)
    model = LadderSSRVisualActor(proprio_dim=PROPRIO_DIM).eval()
    obs_history = torch.rand(3, 2, 10, PROPRIO_DIM)
    depth = torch.rand(3, 2, 2, 1, 60, 80)
    hidden = torch.ones(1, 2, 256)
    dones = torch.tensor([[False, False], [False, True], [False, False]])
    masks = torch.tensor([[True, True], [True, True], [True, False]])

    output = model(obs_history, depth, hidden_state=hidden, dones=dones, masks=masks, deterministic=True)
    reset = model.reset_hidden_state(hidden, torch.tensor([False, True]))

    assert torch.equal(reset[:, 0], hidden[:, 0])
    assert torch.equal(reset[:, 1], torch.zeros_like(hidden[:, 1]))
    assert output.actions.shape == (3, 2, 25)
    assert output.actor_latent.shape == (3, 2, 51)
    assert output.next_hidden_state.shape == (1, 2, 256)
    assert torch.equal(output.valid_mask, masks)
    assert torch.isfinite(output.next_hidden_state).all()


def test_dynamic_logvar_is_not_an_actor_feature_in_deterministic_mode():
    torch.manual_seed(11)
    model = LadderSSRVisualActor(proprio_dim=PROPRIO_DIM).eval()
    obs_history, depth = _rollout_inputs(batch_size=2)

    deterministic = model(obs_history, depth, deterministic=True)
    stochastic = model(
        obs_history,
        depth,
        deterministic=False,
        dynamic_noise=torch.ones_like(deterministic.dynamic_mu),
    )
    expected_dynamic = deterministic.dynamic_mu + torch.exp(0.5 * deterministic.dynamic_logvar)
    with torch.no_grad():
        model.dynamic_logvar_head.bias.add_(10.0)
    changed_logvar = model(obs_history, depth, deterministic=True)

    assert torch.allclose(stochastic.dynamic_latent, expected_dynamic)
    assert torch.equal(deterministic.actor_input[..., -16:], deterministic.dynamic_mu)
    assert torch.allclose(deterministic.actions, changed_logvar.actions)


def test_auxiliary_losses_are_finite_and_backpropagate_to_both_encoders():
    torch.manual_seed(13)
    model = LadderSSRVisualActor(proprio_dim=PROPRIO_DIM)
    obs_history = torch.rand(2, 2, 10, PROPRIO_DIM)
    depth = torch.rand(2, 2, 2, 60, 80)
    output = model(obs_history, depth, deterministic=False)
    targets = LadderVisualTargets(
        velocity=torch.rand(2, 2, 3),
        limb_heightmaps=torch.rand(2, 2, 4, 10, 3),
        heightmap_valid=torch.tensor(
            [
                [[[[1, 1, 1]] * 10] * 4, [[[1, 1, 1]] * 10] * 4],
                [[[[1, 1, 1]] * 10] * 4, [[[0, 0, 0]] * 10] * 4],
            ],
            dtype=torch.float32,
        ),
        next_proprio=torch.rand(2, 2, PROPRIO_DIM),
        future_valid=torch.tensor([[True, True], [True, False]]),
    )

    losses = model.auxiliary_losses(output, targets)
    losses["total"].backward()

    assert set(losses) == {"velocity", "hand", "foot", "future", "kl", "total"}
    assert all(torch.isfinite(loss).item() for loss in losses.values())
    assert model.state_encoder[0].weight.grad is not None
    assert model.depth_encoder[0].weight.grad is not None


def test_masked_mse_uses_only_its_own_valid_element_count():
    prediction = torch.tensor([[[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]]])
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, False]])

    loss = LadderSSRVisualActor._masked_mse(prediction, target, valid)

    torch.testing.assert_close(loss, torch.tensor(14.0 / 3.0))
