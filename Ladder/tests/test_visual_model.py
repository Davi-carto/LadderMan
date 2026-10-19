import torch
from tensordict import TensorDict

from ladder.models import LadderDepthEncoder, LadderVisualActorModel


def test_ladder_depth_encoder_matches_expected_shape():
    encoder = LadderDepthEncoder(input_dim=(60, 80))

    encoded = encoder(torch.rand(4, 1, 60, 80))

    assert encoded.shape == (4, 224)
    assert encoder.output_dim == 224


def test_visual_actor_fuses_proprioception_and_depth():
    observations = TensorDict(
        {
            "visual_proprio": torch.rand(3, 310),
            "visual_depth": torch.rand(3, 1, 60, 80),
            "critic": torch.rand(3, 631),
        },
        batch_size=[3],
    )
    model = LadderVisualActorModel(
        observations,
        {"actor": ["visual_proprio", "visual_depth"], "critic": ["critic"]},
        "actor",
        output_dim=25,
        hidden_dims=[64, 32],
    )

    actions = model(observations)
    actions.square().mean().backward()

    assert actions.shape == (3, 25)
    assert any(parameter.grad is not None for parameter in model.parameters())
