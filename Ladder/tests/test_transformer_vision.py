import pytest
import torch
from torch import nn

from ladder.rsl_rl.algorithms import TransformerVisionPretrainer
from ladder.rsl_rl.modules import (
    FUTURE_PHYSICAL_STATE_DIM,
    LegacyTaskGoalTransformerVisionEncoder,
    TaskGoalTransformerVisionEncoder,
    TransformerVisionEncoder,
    TransformerVisionTargets,
    VariationalTransformerVisionEncoder,
)
from ladder.rsl_rl.storage import TransformerVisionRolloutStorage


def _storage(num_steps: int = 2, num_envs: int = 2) -> TransformerVisionRolloutStorage:
    storage = TransformerVisionRolloutStorage(num_steps, num_envs, "cpu")
    for step in range(num_steps):
        dones = torch.zeros(num_envs, dtype=torch.bool)
        dones[-1] = step == 0
        storage.add(
            proprio_history=torch.full((num_envs, 10, 85), float(step)),
            depth_history=torch.full((num_envs, 2, 60, 80), float(step)),
            velocity=torch.full((num_envs, 3), float(step)),
            limb_heightmaps=torch.full((num_envs, 4, 10, 3), float(step)),
            heightmap_valid=torch.ones(num_envs, 4, 10, 3),
            heightmap_hit_class=torch.ones(num_envs, 4, 10, 3, dtype=torch.long),
            next_proprio=torch.full((num_envs, 85), float(step + 10)),
            dones=dones,
        )
    return storage


def test_legacy_task_goal_v1_retains_51_dimensional_actor_latent():
    model = LegacyTaskGoalTransformerVisionEncoder()
    assert model.actor_latent_dim == 51
    assert not hasattr(model, "motion_head")
    assert not hasattr(model, "motion_decoder")


def test_legacy_task_goal_v1_losses_do_not_require_stable_motion_targets():
    model = LegacyTaskGoalTransformerVisionEncoder().eval()
    output = model(
        torch.rand(2, 10, 85),
        torch.rand(2, 2, 60, 80),
        deterministic=True,
    )
    targets = TransformerVisionTargets(
        velocity=output.velocity.detach(),
        limb_heightmaps=torch.stack(
            (
                output.hand_left_map,
                output.hand_right_map,
                output.foot_left_map,
                output.foot_right_map,
            ),
            dim=1,
        ).detach(),
        heightmap_valid=torch.ones(2, 4, 10, 3),
        heightmap_hit_class=torch.ones(2, 4, 10, 3, dtype=torch.long),
        next_proprio=torch.zeros(2, 85),
        limb_goal_b=output.limb_goal_b.detach(),
        limb_goal_valid=torch.ones(2, 4),
        contact_indicator=torch.zeros(2, 4),
    )

    losses = model.losses(output, targets)

    assert "motion_joint" not in losses
    assert "motion_root_orientation" not in losses
    assert all(torch.isfinite(value) for value in losses.values())


def test_transformer_vision_shapes_and_latent_layout():
    model = TransformerVisionEncoder().eval()
    output = model(
        torch.rand(2, 10, 85),
        torch.rand(2, 2, 60, 80),
        deterministic=True,
    )

    assert output.velocity.shape == (2, 3)
    assert output.latent.shape == (2, 51)
    assert output.next_proprio.shape == (2, FUTURE_PHYSICAL_STATE_DIM)
    assert output.dynamic_latent.shape == (2, 16)
    assert output.dynamic_mu is None
    assert output.dynamic_logvar is None
    torch.testing.assert_close(output.latent[:, -16:], output.dynamic_latent)
    assert output.next_hidden_state.shape == (1, 2, 256)
    assert output.hand_left_map.shape == (2, 10, 3)
    assert output.hand_right_map.shape == (2, 10, 3)
    assert output.foot_left_map.shape == (2, 10, 3)
    assert output.foot_right_map.shape == (2, 10, 3)
    assert model.future_decoder[0].in_features == 16
    assert model.future_decoder[-1].out_features == FUTURE_PHYSICAL_STATE_DIM
    assert isinstance(model.future_decoder_activation, nn.ELU)


def test_active_dynamic_latent_is_deterministic_and_has_no_vae_heads():
    torch.manual_seed(4)
    model = TransformerVisionEncoder().eval()
    proprio = torch.rand(2, 10, 85)
    depth = torch.rand(2, 2, 60, 80)

    first = model(proprio, depth, deterministic=False)
    second = model(proprio, depth, deterministic=True)

    torch.testing.assert_close(first.dynamic_latent, second.dynamic_latent)
    assert not hasattr(model, "dynamic_mu_head")
    assert not hasattr(model, "dynamic_logvar_head")


def test_task_and_motion_heads_are_mlps_and_produce_deployable_32d_semantics():
    model = TaskGoalTransformerVisionEncoder().eval()
    output = model(
        torch.rand(2, 10, 85),
        torch.rand(2, 2, 60, 80),
        deterministic=True,
    )

    linear_layers = [layer for layer in model.task_head if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (256, 128),
        (128, 64),
        (64, 16),
    ]
    motion_linear_layers = [
        layer for layer in model.motion_head if isinstance(layer, nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in motion_linear_layers] == [
        (256, 128),
        (128, 64),
        (64, 16),
    ]
    decoder_linear_layers = [
        layer for layer in model.motion_decoder if isinstance(layer, nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in decoder_linear_layers] == [
        (16, 64),
        (64, 128),
        (128, 31),
    ]
    assert output.next_proprio is None
    assert output.latent.shape == (2, 67)
    assert output.limb_goal_b.shape == (2, 4, 3)
    assert output.contact_logits.shape == (2, 4)
    assert output.contact_probability.shape == (2, 4)
    assert torch.all((0.0 <= output.contact_probability) & (output.contact_probability <= 1.0))
    assert output.motion_latent.shape == (2, 16)
    assert output.motion_joint_pos.shape == (2, 25)
    assert output.motion_root_orientation_6d.shape == (2, 6)
    torch.testing.assert_close(
        output.dynamic_latent,
        torch.cat(
            (output.limb_goal_b.reshape(2, 12), output.contact_probability), dim=-1
        ),
    )


def test_task_goal_losses_use_goal_mask_and_contact_logits():
    model = TaskGoalTransformerVisionEncoder().eval()
    output = model(
        torch.rand(2, 10, 85),
        torch.rand(2, 2, 60, 80),
        deterministic=True,
    )
    targets = TransformerVisionTargets(
        velocity=output.velocity.detach(),
        limb_heightmaps=torch.stack(
            (
                output.hand_left_map,
                output.hand_right_map,
                output.foot_left_map,
                output.foot_right_map,
            ),
            dim=1,
        ).detach(),
        heightmap_valid=torch.ones(2, 4, 10, 3),
        heightmap_hit_class=torch.ones(2, 4, 10, 3, dtype=torch.long),
        next_proprio=torch.zeros(2, 85),
        limb_goal_b=output.limb_goal_b.detach().clone(),
        limb_goal_valid=torch.tensor([[True, True, True, True], [False, False, False, False]]),
        contact_indicator=torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
        ),
        motion_joint_pos=output.motion_joint_pos.detach().clone(),
        motion_root_orientation_6d=(
            output.motion_root_orientation_6d.detach().clone()
        ),
        motion_valid=torch.tensor([True, False]),
    )
    targets.limb_goal_b[1].fill_(1.0e6)
    targets.motion_joint_pos[0].add_(0.1)
    targets.motion_root_orientation_6d[0].add_(0.1)
    targets.motion_joint_pos[1].fill_(1.0e6)
    targets.motion_root_orientation_6d[1].fill_(1.0e6)

    losses = model.losses(output, targets)

    torch.testing.assert_close(losses["limb_goal"], torch.tensor(0.0))
    assert losses["contact_bce"] > 0.0
    torch.testing.assert_close(losses["motion_joint"], torch.tensor(0.075))
    torch.testing.assert_close(
        losses["motion_root_orientation"], torch.tensor(0.075)
    )
    assert "future" not in losses
    assert all(torch.isfinite(value) for value in losses.values())


def test_explicit_velocity_latent_is_clipped_to_reference_bounds():
    model = TransformerVisionEncoder().eval()
    with torch.no_grad():
        model.velocity_head.weight.zero_()
        model.velocity_head.bias.fill_(20.0)
    output = model(
        torch.rand(2, 10, 85),
        torch.rand(2, 2, 60, 80),
        deterministic=True,
    )

    assert torch.all(output.velocity == 10.0)
    assert torch.all(output.latent[:, :3] == 10.0)


def test_masked_mse_only_counts_valid_elements():
    prediction = torch.tensor([[1.0, 100.0]])
    target = torch.tensor([[3.0, 0.0]])
    valid = torch.tensor([[True, False]])

    loss = TransformerVisionEncoder.masked_mse(prediction, target, valid)

    torch.testing.assert_close(loss, torch.tensor(4.0))


def test_rung_points_receive_twice_the_reconstruction_weight():
    model = TransformerVisionEncoder().eval()
    output = model(
        torch.rand(1, 10, 85),
        torch.rand(1, 2, 60, 80),
        deterministic=True,
    )
    predictions = torch.stack(
        (output.hand_left_map, output.hand_right_map, output.foot_left_map, output.foot_right_map),
        dim=1,
    )
    targets_map = predictions.detach().clone()
    valid = torch.zeros(1, 4, 10, 3)
    hit_class = torch.zeros(1, 4, 10, 3, dtype=torch.long)
    valid[0, 0, 0, :2] = 1.0
    hit_class[0, 0, 0, 0] = 1
    hit_class[0, 0, 0, 1] = 2
    targets_map[0, 0, 0, 0] += 2.0
    targets_map[0, 0, 0, 1] += 1.0
    targets = TransformerVisionTargets(
        velocity=output.velocity.detach(),
        limb_heightmaps=targets_map,
        heightmap_valid=valid,
        heightmap_hit_class=hit_class,
        next_proprio=torch.cat(
            (output.next_proprio.detach(), torch.zeros(1, 85 - FUTURE_PHYSICAL_STATE_DIM)),
            dim=-1,
        ),
        future_valid=torch.ones(1),
    )

    losses = model.losses(output, targets)

    torch.testing.assert_close(losses["hand_left"], torch.tensor(3.0))
    torch.testing.assert_close(losses["heightmap_rung"], torch.tensor(4.0))
    torch.testing.assert_close(losses["heightmap_non_rung"], torch.tensor(1.0))


def test_storage_aligns_next_proprio_and_future_reset_mask():
    storage = _storage()

    torch.testing.assert_close(storage.next_proprio_targets[0], torch.full((2, 85), 10.0))
    assert storage.future_valid.tolist() == [[True, False], [True, True]]
    assert storage.reset_before_mask().tolist() == [[False, False], [False, True]]


def test_sequence_losses_mask_terminal_future_targets():
    storage = _storage()
    model = TransformerVisionEncoder().eval()
    output = model(
        storage.proprio_history,
        storage.depth_history,
        dones=storage.reset_before_mask(),
        deterministic=True,
    )
    targets = TransformerVisionTargets(
        velocity=storage.velocity_targets,
        limb_heightmaps=storage.limb_heightmaps,
        heightmap_valid=storage.heightmap_valid,
        heightmap_hit_class=storage.heightmap_hit_class,
        next_proprio=storage.next_proprio_targets,
        future_valid=storage.future_valid,
    )

    losses = model.losses(output, targets)

    assert {
        "velocity",
        "velocity_zero_baseline",
        "velocity_skill_score",
        "hand_left",
        "hand_right",
        "foot_left",
        "foot_right",
        "hand",
        "foot",
        "heightmap_rung",
        "heightmap_non_rung",
        "future",
        "dynamic_abs_max",
        "future_root_angular_velocity",
        "future_root_orientation_6d",
        "future_joint_position",
        "future_joint_velocity",
        "total",
    }.issubset(losses)
    assert all(torch.isfinite(value) for value in losses.values())


def test_visual_only_optimizer_updates_encoder_and_restores_checkpoint():
    torch.manual_seed(9)
    storage = _storage()
    model = TransformerVisionEncoder()
    pretrainer = TransformerVisionPretrainer(model, learning_rate=1.0e-3, max_grad_norm=1.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = pretrainer.optimize(storage)
    checkpoint = pretrainer.state_dict()
    restored = TransformerVisionEncoder()
    restored_pretrainer = TransformerVisionPretrainer(restored, learning_rate=1.0e-3, max_grad_norm=1.0)
    restored_pretrainer.load_state_dict(checkpoint)

    assert result.parameter_change_max_abs > 0.0
    assert result.gradient_norm > 0.0
    assert any(not torch.equal(parameter, previous) for parameter, previous in zip(model.parameters(), before, strict=True))
    for parameter, restored_parameter in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(parameter, restored_parameter)


def test_pre_normalization_checkpoint_is_rejected():
    model = TransformerVisionEncoder()
    pretrainer = TransformerVisionPretrainer(model)
    checkpoint = pretrainer.state_dict()
    checkpoint.pop("data_contract")

    with pytest.raises(RuntimeError, match="data contract"):
        pretrainer.load_state_dict(checkpoint)


def test_pre_future_elu_checkpoint_is_rejected():
    model = TransformerVisionEncoder()
    pretrainer = TransformerVisionPretrainer(model)
    checkpoint = pretrainer.state_dict()
    checkpoint["data_contract"] = "cr1_field_normalized_proprio_rung_weighted_dynamic16_v1"

    with pytest.raises(RuntimeError, match="selected encoder requires"):
        pretrainer.load_state_dict(checkpoint)


def test_legacy_variational_encoder_retains_full_future_decoder_and_kl():
    model = VariationalTransformerVisionEncoder().eval()
    output = model(
        torch.rand(2, 10, 85),
        torch.rand(2, 2, 60, 80),
        deterministic=True,
    )
    targets = TransformerVisionTargets(
        velocity=output.velocity.detach(),
        limb_heightmaps=torch.stack(
            (
                output.hand_left_map,
                output.hand_right_map,
                output.foot_left_map,
                output.foot_right_map,
            ),
            dim=1,
        ).detach(),
        heightmap_valid=torch.ones(2, 4, 10, 3),
        heightmap_hit_class=torch.ones(2, 4, 10, 3, dtype=torch.long),
        next_proprio=output.next_proprio.detach(),
    )

    losses = model.losses(output, targets)

    assert output.next_proprio.shape == (2, 85)
    assert output.dynamic_mu is not None
    assert output.dynamic_logvar is not None
    assert "kl" in losses
    assert hasattr(model, "dynamic_mu_head")
    assert hasattr(model, "dynamic_logvar_head")


def test_visual_only_optimizer_preserves_full_gru_sequences_in_environment_mini_batches():
    torch.manual_seed(10)
    storage = _storage(num_steps=3, num_envs=5)
    model = TransformerVisionEncoder()
    pretrainer = TransformerVisionPretrainer(
        model,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        mini_batch_envs=2,
    )

    result = pretrainer.optimize(storage)

    assert result.output.velocity.shape == (3, 1, 3)
    assert result.parameter_change_max_abs > 0.0
    assert result.gradient_norm > 0.0
    assert all(torch.isfinite(value) for value in result.losses.values())


def test_depth_ablation_keeps_proprio_fixed_and_replays_depth_variants():
    torch.manual_seed(11)
    storage = _storage(num_steps=2, num_envs=3)
    model = TransformerVisionEncoder()
    pretrainer = TransformerVisionPretrainer(model)

    evaluations = pretrainer.evaluate_depth_ablation(storage)

    assert set(evaluations) == {"normal", "zero", "shuffled"}
    for output, losses in evaluations.values():
        assert output.hand_left_map.shape == (2, 3, 10, 3)
        assert all(torch.isfinite(value) for value in losses.values())
