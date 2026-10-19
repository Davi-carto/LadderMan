import importlib
import sys
from pathlib import Path
from types import ModuleType

import torch
from tensordict import TensorDict


_DAGGER_DIR = (
    Path(__file__).resolve().parents[1]
    / "source/ladder/ladder/tasks/ladder_climb/config/cr1/dagger"
)
_PACKAGE_NAME = "cr1_privileged_recurrent_visual_dagger_test_package"
_PACKAGE = ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(_DAGGER_DIR)]
sys.modules.setdefault(_PACKAGE_NAME, _PACKAGE)

model_module = importlib.import_module(
    f"{_PACKAGE_NAME}.privileged_recurrent_visual_model"
)
storage_module = importlib.import_module(
    f"{_PACKAGE_NAME}.recurrent_visual_mse_storage"
)
teacher_module = importlib.import_module(f"{_PACKAGE_NAME}.student_teacher")
algorithm_module = importlib.import_module(
    f"{_PACKAGE_NAME}.privileged_recurrent_visual_mse_algorithm"
)


def _observations(batch_size=3):
    proprio = torch.randn(batch_size, 25, 85)
    proprio[..., -1] = 1.0
    return TensorDict(
        {
            "proprio_history": proprio,
            "depth_history": torch.rand(batch_size, 1, 60, 80) * 2.0 - 1.0,
            "depth_updated": torch.ones(batch_size, dtype=torch.bool),
            "privileged_actor": torch.randn(batch_size, 28).clamp(-3.0, 3.0),
        },
        batch_size=[batch_size],
    )


def test_privileged_student_accepts_oracle_vector_and_produces_actions():
    torch.manual_seed(3)
    student = model_module.CR1PrivilegedRecurrentVisualDaggerStudent(
        25, [32, 256], 0.1
    )
    observations = _observations()
    mean, feature, hidden, diagnostics = student.action_mean_and_feature_step(
        observations, None
    )
    assert model_module.PRIVILEGED_ACTOR_DIM == 28
    assert student.feature_mlp[0].in_features == 85 + 256 + 28
    assert mean.shape == (3, 25)
    assert feature.shape == (3, 256)
    assert hidden.shape == (1, 3, 256)
    assert diagnostics["privileged_actor"].shape == (3, 28)
    assert isinstance(student.proprio_encoder.layer_norm, torch.nn.LayerNorm)
    assert isinstance(
        student.depth_encoder.projection.layer_norm, torch.nn.LayerNorm
    )
    torch.testing.assert_close(
        diagnostics["proprio_feature"].mean(dim=-1),
        torch.zeros(3),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["depth_feature"].mean(dim=-1),
        torch.zeros(3),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_privileged_layernorm_only_adds_two_branch_normalizers():
    torch.manual_seed(17)
    legacy = model_module.CR1PrivilegedRecurrentVisualLegacyDaggerStudent(
        25, [32, 256], 0.1
    )
    torch.manual_seed(17)
    normalized = model_module.CR1PrivilegedRecurrentVisualDaggerStudent(
        25, [32, 256], 0.1
    )
    legacy_state = legacy.state_dict()
    normalized_state = normalized.state_dict()
    extra = set(normalized_state) - set(legacy_state)
    assert extra == {
        "proprio_encoder.layer_norm.weight",
        "proprio_encoder.layer_norm.bias",
        "depth_encoder.projection.layer_norm.weight",
        "depth_encoder.projection.layer_norm.bias",
    }
    assert sum(normalized_state[name].numel() for name in extra) == 512
    for name, value in legacy_state.items():
        torch.testing.assert_close(value, normalized_state[name], rtol=0, atol=0)


def test_privileged_student_sequence_and_storage_keep_time_alignment():
    torch.manual_seed(5)
    student = model_module.CR1PrivilegedRecurrentVisualDaggerStudent(
        25, [32, 256], 0.1
    )
    steps, envs = 3, 2
    first = _observations(envs)
    histories = torch.stack([first["proprio_history"]] * steps)
    depths = torch.stack([first["depth_history"]] * steps)
    updates = torch.ones(steps, envs, dtype=torch.bool)
    dones = torch.zeros(steps, envs, dtype=torch.bool)
    env_code = torch.arange(envs, dtype=torch.float32).view(1, envs, 1)
    privileged = env_code + torch.arange(steps, dtype=torch.float32).view(steps, 1, 1)
    privileged = privileged.expand(steps, envs, 28).clone()
    means, _, _ = student.action_mean_and_feature_sequence(
        histories,
        depths,
        updates,
        dones,
        student.initial_hidden(envs, device=torch.device("cpu"), dtype=torch.float32),
        privileged,
    )
    assert means.shape == (steps, envs, 25)

    storage = storage_module.RecurrentVisualMseDaggerStorage(
        steps, envs, 25, "cpu", privileged_dim=28
    )
    for step in range(steps):
        storage.add(
            storage_module.RecurrentVisualMseDaggerTransition(
                proprio_history=first["proprio_history"],
                depth=first["depth_history"],
                depth_updated=first["depth_updated"],
                teacher_action_mean=torch.zeros(envs, 25),
                hidden_before=torch.zeros(1, envs, 256),
                teacher_controlled=torch.zeros(envs, dtype=torch.bool),
                privileged=privileged[step],
            ),
            torch.zeros(envs, dtype=torch.bool),
        )
    batch = next(storage.batches(1, 1))
    assert batch["privileged_actor"].shape == (steps, envs, 28)
    # Storage intentionally shuffles environment IDs; compare the per-step
    # values after sorting that harmless environment-axis permutation.
    torch.testing.assert_close(
        batch["privileged_actor"].sort(dim=1).values,
        privileged.sort(dim=1).values,
    )


def test_privileged_rollout_metrics_report_per_limb_coverage_and_physical_speed():
    privileged = torch.zeros(2, 3, 28)
    privileged[..., 0] = 3.0
    privileged[..., 1] = 4.0
    privileged[..., 12] = 1.0
    privileged[..., 16] = 0.1
    metrics = algorithm_module.CR1PrivilegedRecurrentVisualMseDagger._privileged_rollout_metrics(
        privileged
    )
    assert metrics["oracle_left_hand_target_distance_mean_m"] == 5.0
    assert metrics["oracle_left_hand_contact_fraction"] == 1.0
    assert metrics["oracle_right_hand_contact_fraction"] == 0.0
    assert metrics["oracle_any_limb_contact_fraction"] == 1.0
    assert abs(metrics["oracle_left_hand_speed_mean_mps"] - 1.0) < 1.0e-6


def test_privileged_dagger_constructs_and_updates_with_online_teacher(tmp_path):
    teacher = teacher_module.elu_mlp(17, [32, 256], 25)
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {"actor_state_dict": {f"mlp.{name}": value for name, value in teacher.state_dict().items()}},
        checkpoint,
    )
    batch_size = 4
    observations = _observations(batch_size)
    observations["teacher"] = torch.randn(batch_size, 17)
    cfg = {
        "num_steps_per_env": 2,
        "policy": {"actor_hidden_dims": [32, 256], "init_noise_std": 0.1},
        "algorithm": {
            "class_name": "unused",
            "teacher_checkpoint": str(checkpoint),
            "teacher_hidden_dims": [32, 256],
            "expected_teacher_observation_dim": 17,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "diagnostic_snapshot_interval": 1,
            "diagnostic_sample_envs": 2,
            "bc_warmup_iterations": 0,
            "teacher_rollout_fraction": 0.0,
        },
    }
    env = type("Env", (), {"num_envs": batch_size, "num_actions": 25})()
    algorithm = algorithm_module.CR1PrivilegedRecurrentVisualMseDagger.construct_algorithm(
        observations, env, cfg, "cpu"
    )
    before = algorithm.policy.action_head.weight.detach().clone()
    proprio_norm_before = algorithm.policy.proprio_encoder.layer_norm.weight.detach().clone()
    depth_norm_before = (
        algorithm.policy.depth_encoder.projection.layer_norm.weight.detach().clone()
    )
    with torch.inference_mode():
        for step in range(2):
            algorithm.act(observations)
            dones = torch.zeros(batch_size, dtype=torch.bool)
            algorithm.process_env_step(
                observations, torch.zeros(batch_size), dones, {}
            )
    losses = algorithm.update()
    assert losses["optimizer_steps"] == 1
    assert "oracle_left_hand_contact_fraction" in losses
    assert losses["student_proprio_encoder_grad_l2_max_after_clip"] > 0.0
    assert losses["student_depth_encoder_grad_l2_max_after_clip"] > 0.0
    assert not torch.equal(
        proprio_norm_before, algorithm.policy.proprio_encoder.layer_norm.weight
    )
    assert not torch.equal(
        depth_norm_before,
        algorithm.policy.depth_encoder.projection.layer_norm.weight,
    )
    assert not torch.equal(before, algorithm.policy.action_head.weight)
