import importlib
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import pytest
from tensordict import TensorDict

_DAGGER_DIR = (
    Path(__file__).resolve().parents[1]
    / "source/ladder/ladder/tasks/ladder_climb/config/cr1/dagger"
)
_PACKAGE_NAME = "cr1_dagger_test_package"
_PACKAGE = ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(_DAGGER_DIR)]
sys.modules.setdefault(_PACKAGE_NAME, _PACKAGE)

_student_teacher = importlib.import_module(f"{_PACKAGE_NAME}.student_teacher")
elu_mlp = _student_teacher.elu_mlp
load_frozen_teacher = _student_teacher.load_frozen_teacher
_visual_model = importlib.import_module(f"{_PACKAGE_NAME}.visual_model")
CR1DaggerVisualStudent = _visual_model.CR1DaggerVisualStudent
CR1VisualHybridDaggerPPO = importlib.import_module(
    f"{_PACKAGE_NAME}.visual_algorithm"
).CR1VisualHybridDaggerPPO
as_visual_dagger_tensordict = importlib.import_module(
    f"{_PACKAGE_NAME}.visual_wrapper"
).as_visual_dagger_tensordict
_pure_model = importlib.import_module(f"{_PACKAGE_NAME}.pure_model")
CR1PureVisualDaggerStudent = _pure_model.CR1PureVisualDaggerStudent
CR1PureVisualDagger = importlib.import_module(
    f"{_PACKAGE_NAME}.pure_algorithm"
).CR1PureVisualDagger
CR1SameInputDagger = importlib.import_module(
    f"{_PACKAGE_NAME}.same_input_algorithm"
).CR1SameInputDagger
_visual_mse_algorithm = importlib.import_module(
    f"{_PACKAGE_NAME}.visual_mse_algorithm"
)
CR1VisualMseDagger = _visual_mse_algorithm.CR1VisualMseDagger
VISUAL_MSE_DAGGER_DATA_CONTRACT = (
    _visual_mse_algorithm.VISUAL_MSE_DAGGER_DATA_CONTRACT
)
as_pure_visual_dagger_tensordict = importlib.import_module(
    f"{_PACKAGE_NAME}.pure_wrapper"
).as_pure_visual_dagger_tensordict
_recurrent_visual_model = importlib.import_module(
    f"{_PACKAGE_NAME}.recurrent_visual_model"
)
CR1RecurrentVisualDaggerStudent = (
    _recurrent_visual_model.CR1RecurrentVisualDaggerStudent
)
DEPTH_UPDATE_KEY = _recurrent_visual_model.DEPTH_UPDATE_KEY
_recurrent_visual_storage = importlib.import_module(
    f"{_PACKAGE_NAME}.recurrent_visual_mse_storage"
)
RecurrentVisualMseDaggerStorage = (
    _recurrent_visual_storage.RecurrentVisualMseDaggerStorage
)
RecurrentVisualMseDaggerTransition = (
    _recurrent_visual_storage.RecurrentVisualMseDaggerTransition
)
_recurrent_visual_algorithm = importlib.import_module(
    f"{_PACKAGE_NAME}.recurrent_visual_mse_algorithm"
)
CR1RecurrentVisualMseDagger = (
    _recurrent_visual_algorithm.CR1RecurrentVisualMseDagger
)


def _teacher_checkpoint(tmp_path):
    teacher = elu_mlp(17, [32, 16], 4)
    path = tmp_path / "teacher.pt"
    torch.save(
        {
            "actor_state_dict": {
                **{f"mlp.{key}": value for key, value in teacher.state_dict().items()},
                "distribution.std_param": torch.ones(4),
            }
        },
        path,
    )
    return str(path), teacher


def _config(checkpoint):
    return {
        "num_steps_per_env": 2,
        "multi_gpu": None,
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [32, 16],
            "critic_hidden_dims": [32, 16],
        },
        "algorithm": {
            "class_name": "unused",
            "teacher_checkpoint": checkpoint,
            "teacher_hidden_dims": [32, 16],
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.005,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "fixed",
            "desired_kl": 0.01,
            "dagger_initial_coeff": 1.0,
            "dagger_final_coeff": 0.05,
            "dagger_anneal_start": 1,
            "dagger_anneal_iterations": 4,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            "normalize_advantage_per_mini_batch": False,
        },
    }


def test_teacher_loader_restores_and_freezes_split_checkpoint(tmp_path):
    checkpoint, original = _teacher_checkpoint(tmp_path)
    restored = load_frozen_teacher(
        checkpoint,
        observation_dim=17,
        action_dim=4,
        hidden_dims=[32, 16],
        device="cpu",
    )
    observation = torch.randn(3, 17)
    torch.testing.assert_close(restored(observation), original(observation))
    assert not any(parameter.requires_grad for parameter in restored.parameters())


def _visual_observations(num_envs=2):
    proprio = torch.randn(num_envs, 5, 85)
    proprio[..., -1] = torch.tensor([1.0, -1.0])[:num_envs, None]
    return TensorDict(
        {
            "proprio_history": proprio,
            "depth_history": torch.rand(num_envs, 2, 60, 80),
            "teacher": torch.randn(num_envs, 17),
            "critic": torch.randn(num_envs, 17),
            "transformer_vision_limb_heightmaps": torch.rand(num_envs, 4, 10, 3),
            "transformer_vision_heightmap_valid": torch.ones(num_envs, 4, 10, 3),
            "transformer_vision_heightmap_hit_class": torch.randint(0, 5, (num_envs, 4, 10, 3)),
            "transformer_vision_velocity_target": torch.rand(num_envs, 3),
        },
        batch_size=[num_envs],
    )


def _visual_config(checkpoint):
    config = _config(checkpoint)
    config["policy"]["actor_hidden_dims"] = [32, 16]
    config["algorithm"].update(
        {
            "min_learning_rate": 1.0e-5,
            "max_learning_rate": 1.0e-3,
            "use_grad_scaler": False,
            "auxiliary_learning_rate": 1.0e-3,
            "auxiliary_mini_batch_envs": 2,
            "auxiliary_num_epochs": 1,
            "visual_architecture": "deterministic",
            "visual_initialization": "scratch",
            "visual_checkpoint": "",
            "normalize_advantage_per_mini_batch": False,
        }
    )
    return config


def test_visual_student_uses_five_command_frames_and_no_motion_head():
    student = CR1DaggerVisualStudent(25, [32, 16], 1.0).eval()
    observations = _visual_observations()
    actions = student(observations)
    assert actions.shape == (2, 25)
    assert student.action_mlp[0].in_features == 460
    assert student.encoder.history_frames == 5
    assert student.encoder.actor_latent_dim == 35
    assert student.encoder.dynamic_head is None
    assert student.encoder.future_decoder is None
    assert not hasattr(student.encoder, "motion_head")


def test_visual_dagger_constructs_and_collects_teacher_labels(tmp_path):
    checkpoint, _ = _teacher_checkpoint(tmp_path)
    observations = _visual_observations()
    algorithm = CR1VisualHybridDaggerPPO.construct_algorithm(
        observations,
        SimpleNamespace(num_envs=2, num_actions=4),
        _visual_config(checkpoint),
        "cpu",
    )
    actions = algorithm.act(observations)
    algorithm.process_env_step(
        observations, torch.ones(2), torch.zeros(2, dtype=torch.bool), {}
    )
    assert actions.shape == (2, 4)
    assert algorithm.storage.proprio_history.shape[-2:] == (5, 85)
    assert algorithm.storage.actor_inputs.shape[-1] == 460
    assert algorithm.storage.teacher_action_mean is not None
    algorithm.act(observations)
    algorithm.process_env_step(
        observations, torch.ones(2), torch.zeros(2, dtype=torch.bool), {}
    )
    algorithm.compute_returns(observations)
    losses = algorithm.update()
    assert losses["dagger_action_mse"] >= 0.0
    assert losses["aux_dagger_encoder_mse"] >= 0.0


def test_visual_wrapper_preserves_binary_command():
    proprio = torch.randn(2, 5, 85)
    proprio[0, :, -1] = 1.0
    proprio[1, :, -1] = -1.0
    raw = {
        "transformer_vision_inputs": {
            "proprio_history": proprio,
            "depth_history": torch.rand(2, 2, 60, 80),
        },
        "transformer_vision_targets": {
            "relative_height": torch.rand(2, 4, 10, 3),
            "hit_valid": torch.ones(2, 4, 10, 3),
            "hit_class": torch.zeros(2, 4, 10, 3),
        },
        "transformer_vision_auxiliary_targets": {
            "base_lin_vel": torch.rand(2, 3),
        },
        "teacher": torch.rand(2, 17),
        "critic": torch.rand(2, 17),
    }
    packed = as_visual_dagger_tensordict(raw, num_envs=2, device="cpu")
    assert packed["proprio_history"][:, :, -1].unique().sort().values.tolist() == [-1.0, 1.0]


def _pure_teacher_checkpoint(tmp_path):
    teacher = elu_mlp(17, [32, 256], 4)
    path = tmp_path / "pure_teacher.pt"
    torch.save(
        {
            "actor_state_dict": {
                **{f"mlp.{key}": value for key, value in teacher.state_dict().items()},
                "distribution.std_param": torch.full((4,), 0.25),
            }
        },
        path,
    )
    return str(path)


def _pure_observations(num_envs=2):
    observations = TensorDict(
        {
            "proprio_history": torch.randn(num_envs, 9, 85),
            "depth_history": torch.rand(num_envs, 2, 60, 80),
            "teacher": torch.randn(num_envs, 17),
        },
        batch_size=[num_envs],
    )
    observations["proprio_history"][..., -1] = 1.0
    observations["proprio_history"][1::2, ..., -1] = -1.0
    return observations


def _visual_mse_config(checkpoint):
    return {
        "num_steps_per_env": 2,
        "policy": {
            "init_noise_std": 0.1,
            "actor_hidden_dims": [32, 256],
        },
        "algorithm": {
            "class_name": "unused",
            "teacher_checkpoint": checkpoint,
            "teacher_hidden_dims": [32, 256],
            "expected_teacher_observation_dim": 17,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "diagnostic_snapshot_interval": 1,
            "diagnostic_sample_envs": 2,
        },
    }


def test_visual_mse_dagger_is_student_rollout_and_action_mse_only(tmp_path):
    checkpoint = _pure_teacher_checkpoint(tmp_path)
    observations = _pure_observations()
    algorithm = CR1VisualMseDagger.construct_algorithm(
        observations,
        SimpleNamespace(num_envs=2, num_actions=4),
        _visual_mse_config(checkpoint),
        "cpu",
    )
    algorithm.configure_diagnostics(str(tmp_path), {"test": True})
    student_before = {
        name: value.detach().clone()
        for name, value in algorithm.policy.named_parameters()
    }
    teacher_before = {
        name: value.detach().clone()
        for name, value in algorithm.teacher.state_dict().items()
    }

    for _ in range(2):
        student_mean = algorithm.policy(
            TensorDict(
                {
                    "proprio_history": observations["proprio_history"],
                    "depth_history": observations["depth_history"],
                },
                batch_size=[2],
            )
        ).detach()
        actions = algorithm.act(observations)
        assert not torch.equal(actions, student_mean)
        torch.testing.assert_close(
            algorithm._pending.teacher_action_mean,
            algorithm.teacher(observations["teacher"]),
        )
        algorithm.process_env_step(
            observations,
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            {},
        )

    assert not hasattr(algorithm.storage, "teacher_feature")
    losses = algorithm.update()
    assert losses["behavior"] >= 0.0
    assert losses["training_action_rmse"] == pytest.approx(
        losses["behavior"] ** 0.5
    )
    assert losses["rollout_student_fraction"] == 1.0
    assert losses["optimizer_steps"] == 1.0
    assert losses["student_depth_encoder_parameter_delta_l2"] > 0.0
    assert losses["student_feature_mlp_parameter_delta_l2"] > 0.0
    assert losses["student_action_head_parameter_delta_l2"] > 0.0
    assert not any(
        forbidden in key.lower()
        for key in losses
        for forbidden in ("kl", "feature_mse", "ppo", "reconstruction")
    )
    torch.testing.assert_close(
        algorithm.policy.log_std, student_before["log_std"], rtol=0.0, atol=0.0
    )
    assert all(
        torch.equal(teacher_before[name], value)
        for name, value in algorithm.teacher.state_dict().items()
    )
    assert (tmp_path / "visual_mse_diagnostics" / "manifest.json").is_file()
    assert (
        tmp_path / "visual_mse_diagnostics" / "iteration_000000_sample.pt"
    ).is_file()

    payload = algorithm.save()
    assert payload["data_contract"] == VISUAL_MSE_DAGGER_DATA_CONTRACT
    assert payload["update_iteration"] == 1
    assert payload["teacher_checkpoint_sha256"] == algorithm.teacher_checkpoint_sha256
    restored = CR1VisualMseDagger.construct_algorithm(
        observations,
        SimpleNamespace(num_envs=2, num_actions=4),
        _visual_mse_config(checkpoint),
        "cpu",
    )
    assert restored.load(payload, None, True)
    assert restored.update_iteration == 1
    wrong_teacher_payload = dict(payload)
    wrong_teacher_payload["teacher_checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="different teacher"):
        restored.load(wrong_teacher_payload, None, True)


def _recurrent_visual_observations(num_envs=2):
    observations = TensorDict(
        {
            "proprio_history": torch.randn(num_envs, 25, 85),
            "depth_history": torch.rand(num_envs, 1, 60, 80),
            "depth_updated": torch.ones(num_envs, 1),
            "teacher": torch.randn(num_envs, 17),
        },
        batch_size=[num_envs],
    )
    observations["proprio_history"][..., -1] = 1.0
    observations["proprio_history"][1::2, ..., -1] = -1.0
    return observations


def test_recurrent_visual_student_holds_and_resets_per_environment_hidden():
    torch.manual_seed(7)
    student = CR1RecurrentVisualDaggerStudent(4, [32, 256], 0.1)
    observations = _recurrent_visual_observations()
    hidden = torch.randn(1, 2, 256)
    observations[DEPTH_UPDATE_KEY] = torch.tensor([[1.0], [0.0]])
    _, _, next_hidden, diagnostics = student.action_mean_and_feature_step(
        observations, hidden
    )
    assert set(diagnostics) == {
        "proprio_feature",
        "depth_feature",
        "recurrent_input",
        "gru_candidate",
        "recurrent_feature",
        "actor_feature",
    }
    assert not torch.equal(next_hidden[:, 0], hidden[:, 0])
    torch.testing.assert_close(next_hidden[:, 1], hidden[:, 1])

    reset = student.reset_hidden_state(
        next_hidden, torch.tensor([False, True])
    )
    torch.testing.assert_close(reset[:, 0], next_hidden[:, 0])
    torch.testing.assert_close(reset[:, 1], torch.zeros_like(reset[:, 1]))


def test_recurrent_visual_student_rejects_nonfinite_and_unscaled_depth_inputs():
    student = CR1RecurrentVisualDaggerStudent(4, [32, 256], 0.1)
    observations = _recurrent_visual_observations()
    observations["depth_history"][0, 0, 0, 0] = 1.5
    with pytest.raises(ValueError, match="centered metric-depth contract"):
        student.action_mean_and_feature_step(observations, None)

    observations = _recurrent_visual_observations()
    observations["proprio_history"][0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        student.action_mean_and_feature_step(observations, None)


def test_recurrent_visual_sequence_matches_masked_step_inference():
    torch.manual_seed(11)
    student = CR1RecurrentVisualDaggerStudent(4, [32, 256], 0.1).eval()
    steps, num_envs = 4, 2
    proprio = torch.randn(steps, num_envs, 25, 85)
    proprio[..., -1] = 1.0
    depth = torch.rand(steps, num_envs, 1, 60, 80)
    updated = torch.tensor(
        [[True, True], [False, True], [True, False], [False, True]]
    )
    dones = torch.tensor(
        [[False, True], [False, False], [True, False], [False, False]]
    )
    initial_hidden = torch.randn(1, num_envs, 256)
    sequence_mean, _, final_hidden = student.action_mean_and_feature_sequence(
        proprio, depth, updated, dones, initial_hidden
    )

    hidden = initial_hidden
    step_means = []
    for step in range(steps):
        if step > 0:
            hidden = student.reset_hidden_state(hidden, dones[step - 1])
        observations = TensorDict(
            {
                "proprio_history": proprio[step],
                "depth_history": depth[step],
                "depth_updated": updated[step],
            },
            batch_size=[num_envs],
        )
        mean, _, hidden, _ = student.action_mean_and_feature_step(
            observations, hidden
        )
        step_means.append(mean)
    torch.testing.assert_close(sequence_mean, torch.stack(step_means))
    torch.testing.assert_close(final_hidden, hidden)


def test_recurrent_storage_reconstructs_25_frame_history_across_done():
    storage = RecurrentVisualMseDaggerStorage(3, 1, 4, "cpu")
    initial_history = torch.zeros(1, 25, 85)
    initial_history[0, :, 0] = torch.arange(25)
    initial_history[..., -1] = 1.0
    histories = [initial_history]
    shifted = torch.cat(
        (initial_history[:, 1:], initial_history[:, -1:].clone()), dim=1
    )
    shifted[:, -1, 0] = 100.0
    histories.append(shifted)
    reset = torch.zeros(1, 25, 85)
    reset[..., 0] = 200.0
    reset[..., -1] = -1.0
    histories.append(reset)
    dones = (False, True, False)
    for step, history in enumerate(histories):
        storage.add(
            RecurrentVisualMseDaggerTransition(
                proprio_history=history,
                depth=torch.full((1, 1, 60, 80), float(step)),
                depth_updated=torch.ones(1, 1),
                teacher_action_mean=torch.zeros(1, 4),
                hidden_before=torch.zeros(1, 1, 256),
                teacher_controlled=torch.zeros(1, dtype=torch.bool),
            ),
            torch.tensor([dones[step]]),
        )
    batch = next(storage.batches(1, 1))
    torch.testing.assert_close(
        batch["proprio_history"], torch.stack(histories, dim=0)
    )


def test_recurrent_visual_dagger_trains_all_student_paths_with_sequences(tmp_path):
    checkpoint = _pure_teacher_checkpoint(tmp_path)
    observations = _recurrent_visual_observations()
    algorithm = CR1RecurrentVisualMseDagger.construct_algorithm(
        observations,
        SimpleNamespace(num_envs=2, num_actions=4),
        _visual_mse_config(checkpoint),
        "cpu",
    )
    algorithm.configure_diagnostics(str(tmp_path), {"test": True})
    before = {
        name: value.detach().clone()
        for name, value in algorithm.policy.named_parameters()
    }
    for step in range(2):
        observations[DEPTH_UPDATE_KEY] = torch.tensor(
            [[1.0], [float(step == 0)]]
        )
        actions = algorithm.act(observations)
        assert actions.shape == (2, 4)
        algorithm.process_env_step(
            observations,
            torch.ones(2),
            torch.tensor([False, step == 0]),
            {},
        )
    losses = algorithm.update()
    assert losses["optimizer_steps"] == 1.0
    assert losses["recurrent_sequence_length"] == 2.0
    assert losses["rollout_fresh_depth_fraction"] == pytest.approx(0.75)
    for prefix in (
        "proprio_encoder",
        "depth_encoder",
        "gru",
        "recurrent_projection",
        "feature_mlp",
        "action_head",
    ):
        assert losses[f"student_{prefix}_grad_l2_max_after_clip"] > 0.0
        changed = any(
            name.startswith(prefix + ".") and not torch.equal(before[name], value)
            for name, value in algorithm.policy.named_parameters()
        )
        assert changed
    assert not any(
        forbidden in key.lower()
        for key in losses
        for forbidden in ("kl", "feature_mse", "ppo", "reconstruction")
    )
    assert (
        tmp_path / "recurrent_visual_mse_diagnostics" / "manifest.json"
    ).is_file()
    snapshot = torch.load(
        tmp_path
        / "recurrent_visual_mse_diagnostics"
        / "iteration_000000_sample.pt",
        map_location="cpu",
        weights_only=False,
    )
    for name in (
        "proprio_feature",
        "depth_feature",
        "recurrent_input",
        "gru_candidate",
        "recurrent_feature",
        "actor_feature",
    ):
        assert name in snapshot


def test_recurrent_visual_dagger_switches_from_bc_to_fixed_mixed_rollout(
    tmp_path, monkeypatch
):
    checkpoint = _pure_teacher_checkpoint(tmp_path)
    observations = _recurrent_visual_observations(num_envs=4)
    cfg = _visual_mse_config(checkpoint)
    cfg["algorithm"]["bc_warmup_iterations"] = 1
    cfg["algorithm"]["teacher_rollout_fraction"] = 0.25
    termination_values = {
        "motion_complete": torch.zeros(4, dtype=torch.bool),
        "required_contact": torch.zeros(4, dtype=torch.bool),
    }
    termination_manager = SimpleNamespace(
        active_terms=tuple(termination_values),
        get_term=lambda name: termination_values[name],
    )
    env = SimpleNamespace(
        num_envs=4,
        num_actions=4,
        unwrapped=SimpleNamespace(termination_manager=termination_manager),
    )
    algorithm = CR1RecurrentVisualMseDagger.construct_algorithm(
        observations,
        env,
        cfg,
        "cpu",
    )
    monkeypatch.setattr(
        torch, "randn_like", lambda value: torch.zeros_like(value)
    )

    teacher_actions = algorithm.teacher(observations["teacher"]).detach()
    torch.testing.assert_close(algorithm.act(observations), teacher_actions)
    assert algorithm._pending.teacher_controlled.tolist() == [True] * 4
    algorithm.process_env_step(
        observations,
        torch.ones(4),
        torch.zeros(4, dtype=torch.bool),
        {},
    )

    algorithm.update_iteration = 1
    actor_obs = algorithm._actor_observations(observations)
    student_actions, _, _, _ = algorithm.policy.action_mean_and_feature_step(
        actor_obs, algorithm._rollout_hidden
    )
    mixed_actions = algorithm.act(observations)
    torch.testing.assert_close(mixed_actions[:1], teacher_actions[:1])
    torch.testing.assert_close(mixed_actions[1:], student_actions[1:])
    assert algorithm._pending.teacher_controlled.tolist() == [
        True,
        False,
        False,
        False,
    ]
    termination_values["motion_complete"] = torch.tensor(
        [True, False, False, False]
    )
    termination_values["required_contact"] = torch.tensor(
        [False, False, True, False]
    )
    algorithm.process_env_step(
        observations,
        torch.tensor([10.0, 20.0, 30.0, 40.0]),
        torch.tensor([True, False, True, False]),
        {},
    )

    losses = algorithm.update()
    assert losses["rollout_teacher_fraction"] == pytest.approx(5.0 / 8.0)
    assert losses["rollout_student_fraction"] == pytest.approx(3.0 / 8.0)
    assert losses["rollout_teacher_controlled_mean_reward"] == pytest.approx(
        14.0 / 5.0
    )
    assert losses["rollout_student_controlled_mean_reward"] == pytest.approx(30.0)
    assert losses["rollout_teacher_controlled_done_fraction"] == pytest.approx(
        1.0 / 5.0
    )
    assert losses["rollout_student_controlled_done_fraction"] == pytest.approx(
        1.0 / 3.0
    )
    assert losses[
        "rollout_teacher_controlled_termination_motion_complete_fraction_of_dones"
    ] == pytest.approx(1.0)
    assert losses[
        "rollout_student_controlled_termination_required_contact_fraction_of_dones"
    ] == pytest.approx(1.0)
    assert losses["training_teacher_controlled_action_rmse"] >= 0.0
    assert losses["training_student_controlled_action_rmse"] >= 0.0
    for name in (
        "proprio_feature",
        "depth_feature",
        "recurrent_input",
        "gru_candidate",
        "recurrent_feature",
        "actor_feature",
    ):
        assert losses[f"rollout_{name}_rms"] >= 0.0
        assert losses[f"rollout_{name}_max_abs"] >= 0.0
    assert 0.0 <= losses["rollout_gru_candidate_saturation_fraction"] <= 1.0
    assert 0.0 <= losses["rollout_hidden_after_saturation_fraction"] <= 1.0
    assert 0.0 <= losses["rollout_hidden_after_exact_zero_fraction"] <= 1.0

    payload = algorithm.save()
    assert payload["bc_warmup_iterations"] == 1
    assert payload["teacher_rollout_fraction"] == pytest.approx(0.25)
    baseline_cfg = _visual_mse_config(checkpoint)
    baseline = CR1RecurrentVisualMseDagger.construct_algorithm(
        observations,
        SimpleNamespace(num_envs=4, num_actions=4),
        baseline_cfg,
        "cpu",
    )
    with pytest.raises(RuntimeError, match="different controller schedule"):
        baseline.load(payload, None, True)


def test_recurrent_visual_dagger_linear_teacher_schedule_boundaries():
    algorithm = object.__new__(CR1RecurrentVisualMseDagger)
    algorithm.bc_warmup_iterations = 500
    algorithm.teacher_rollout_fraction = 0.1
    algorithm.teacher_rollout_schedule_start = 500
    algorithm.teacher_rollout_schedule_end = 4000
    algorithm.teacher_rollout_schedule_initial_fraction = 1.0
    algorithm.teacher_rollout_schedule_final_fraction = 0.1

    fractions = [
        algorithm._teacher_control_fraction(iteration)
        for iteration in (0, 499, 500, 2250, 4000, 10000)
    ]
    assert fractions == pytest.approx([1.0, 1.0, 1.0, 0.55, 0.1, 0.1])


def test_pure_visual_dagger_uses_student_rollouts_and_only_distillation(tmp_path):
    checkpoint = _pure_teacher_checkpoint(tmp_path)
    cfg = _config(checkpoint)
    cfg["policy"]["actor_hidden_dims"] = [32, 256]
    cfg["algorithm"]["teacher_hidden_dims"] = [32, 256]
    cfg["algorithm"]["action_kl_weight"] = 2.0
    cfg["algorithm"]["feature_weight"] = 1.0
    cfg["algorithm"]["bc_warmup_iterations"] = 1
    cfg["algorithm"]["teacher_rollout_fraction"] = 0.5
    observations = _pure_observations()
    algorithm = CR1PureVisualDagger.construct_algorithm(
        observations,
        SimpleNamespace(num_envs=2, num_actions=4),
        cfg,
        "cpu",
    )
    parameters_before = {
        name: value.detach().clone() for name, value in algorithm.policy.named_parameters()
    }
    for _ in range(2):
        actions = algorithm.act(observations)
        assert actions.shape == (2, 4)
        algorithm.process_env_step(observations, torch.ones(2), torch.zeros(2, dtype=torch.bool), {})
    algorithm.compute_returns(observations)
    losses = algorithm.update()
    assert set(losses) == {
        "distillation_kl",
        "distillation_feature_mse",
        "weighted_action_kl",
        "weighted_feature_mse",
        "distillation_total",
        "rollout_teacher_fraction",
    }
    assert losses["distillation_total"] == pytest.approx(
        losses["weighted_action_kl"] + losses["weighted_feature_mse"]
    )
    assert losses["distillation_total"] > 0.0
    assert losses["rollout_teacher_fraction"] == 1.0
    assert all(
        not torch.equal(parameters_before[name], value)
        for name, value in algorithm.policy.named_parameters()
    )


def test_pure_visual_dagger_switches_from_bc_to_fixed_mixed_rollout(tmp_path):
    checkpoint = _pure_teacher_checkpoint(tmp_path)
    cfg = _config(checkpoint)
    cfg["policy"]["actor_hidden_dims"] = [32, 256]
    cfg["algorithm"]["teacher_hidden_dims"] = [32, 256]
    cfg["algorithm"]["action_kl_weight"] = 2.0
    cfg["algorithm"]["feature_weight"] = 1.0
    cfg["algorithm"]["bc_warmup_iterations"] = 1
    cfg["algorithm"]["teacher_rollout_fraction"] = 0.25
    observations = _pure_observations(num_envs=4)
    algorithm = CR1PureVisualDagger.construct_algorithm(
        observations,
        SimpleNamespace(num_envs=4, num_actions=4),
        cfg,
        "cpu",
    )

    with torch.no_grad():
        teacher_actions = algorithm.teacher(observations["teacher"])
        student_actions = algorithm.policy(observations)
    torch.testing.assert_close(algorithm.act(observations), teacher_actions)
    algorithm.process_env_step(observations, None, None, None)
    algorithm.update_iteration = 1
    mixed_actions = algorithm.act(observations)
    torch.testing.assert_close(mixed_actions[:1], teacher_actions[:1])
    torch.testing.assert_close(mixed_actions[1:], student_actions[1:])
    saved = algorithm.save()
    restored = CR1PureVisualDagger.construct_algorithm(
        observations,
        SimpleNamespace(num_envs=4, num_actions=4),
        cfg,
        "cpu",
    )
    restored.load(saved, load_cfg=None, strict=True)
    assert restored.update_iteration == 1


def test_pure_visual_student_is_decoder_free_paper_cnn():
    student = CR1PureVisualDaggerStudent(25, [64, 256], 1.0)
    assert [layer.out_channels for layer in student.depth_encoder.network if isinstance(layer, torch.nn.Conv2d)] == [8, 8, 8]
    assert all(layer.kernel_size == (5, 5) for layer in student.depth_encoder.network if isinstance(layer, torch.nn.Conv2d))
    assert not any(
        token in name.lower()
        for name, _ in student.named_modules()
        for token in ("decoder", "velocity", "transformer", "gru")
    )
    assert student(_pure_observations()).shape == (2, 25)


def test_pure_visual_wrapper_excludes_teacher_for_actor_only():
    observations = _pure_observations().to_dict()
    raw = {
        "transformer_vision_inputs": {
            "proprio_history": observations["proprio_history"],
            "depth_history": observations["depth_history"],
        },
        "teacher": observations["teacher"],
    }
    packed = as_pure_visual_dagger_tensordict(raw, num_envs=2, device="cpu")
    actor_only = as_pure_visual_dagger_tensordict(
        raw, num_envs=2, device="cpu", actor_only=True
    )
    assert set(packed.keys()) == {"proprio_history", "depth_history", "teacher"}
    assert set(actor_only.keys()) == {"proprio_history", "depth_history"}


def test_pure_visual_wrapper_preserves_optional_camera_update_mask():
    observations = _recurrent_visual_observations().to_dict()
    raw = {
        "transformer_vision_inputs": {
            "proprio_history": observations["proprio_history"],
            "depth_history": observations["depth_history"],
            "depth_updated": observations["depth_updated"],
        },
        "teacher": observations["teacher"],
    }
    actor_only = as_pure_visual_dagger_tensordict(
        raw, num_envs=2, device="cpu", actor_only=True
    )
    assert set(actor_only.keys()) == {
        "proprio_history",
        "depth_history",
        "depth_updated",
    }
    torch.testing.assert_close(
        actor_only["depth_updated"], observations["depth_updated"]
    )


def _same_input_config(checkpoint):
    model = {
        "class_name": "MLPModel",
        "hidden_dims": [32, 16],
        "activation": "elu",
        "obs_normalization": False,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.1,
            "std_type": "scalar",
        },
    }
    return {
        "num_steps_per_env": 2,
        "multi_gpu": None,
        "obs_groups": {
            "student": ["policy"],
            "teacher": ["policy"],
        },
        "student": deepcopy(model),
        "teacher": deepcopy(model),
        "algorithm": {
            "class_name": (
                f"{_PACKAGE_NAME}.same_input_algorithm:CR1SameInputDagger"
            ),
            "teacher_checkpoint": checkpoint,
            "expected_observation_dim": 17,
            "diagnostic_snapshot_interval": 1,
            "diagnostic_sample_envs": 2,
            "num_learning_epochs": 1,
            "gradient_length": 1,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "loss_type": "mse",
            "optimizer": "adam",
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def test_same_input_dagger_uses_student_rollout_and_shared_online_labels(
    tmp_path,
):
    checkpoint, teacher = _teacher_checkpoint(tmp_path)
    observations = TensorDict(
        {"policy": torch.randn(4, 17)}, batch_size=[4]
    )
    algorithm = CR1SameInputDagger.construct_algorithm(
        observations,
        SimpleNamespace(num_envs=4, num_actions=4),
        _same_input_config(checkpoint),
        "cpu",
    )
    algorithm.configure_diagnostics(str(tmp_path), {"test": True})
    teacher_before = {
        name: value.detach().clone()
        for name, value in algorithm.teacher.state_dict().items()
    }
    student_before = {
        name: value.detach().clone()
        for name, value in algorithm.student.mlp.state_dict().items()
    }

    for _ in range(2):
        actions = algorithm.act(observations)
        expected_teacher = teacher(observations["policy"])
        torch.testing.assert_close(
            algorithm.transition.privileged_actions, expected_teacher
        )
        assert actions.data_ptr() == algorithm.transition.actions.data_ptr()
        algorithm.process_env_step(
            observations,
            torch.ones(4),
            torch.zeros(4, dtype=torch.bool),
            {},
        )
    losses = algorithm.update()

    assert losses["behavior"] >= 0.0
    assert losses["rollout_student_fraction"] == 1.0
    assert losses["shared_input_max_abs_diff"] == 0.0
    assert losses["rollout_action_rmse"] >= 0.0
    assert losses["student_parameter_delta_l2"] > 0.0
    assert any(
        not torch.equal(student_before[name], value)
        for name, value in algorithm.student.mlp.state_dict().items()
    )
    assert all(
        torch.equal(teacher_before[name], value)
        for name, value in algorithm.teacher.state_dict().items()
    )
    assert (tmp_path / "same_input_diagnostics" / "manifest.json").is_file()
    assert (
        tmp_path
        / "same_input_diagnostics"
        / "iteration_000000_sample.pt"
    ).is_file()

    payload = algorithm.save()
    assert payload["num_updates"] == 1
    assert payload["teacher_checkpoint_sha256"] == algorithm.teacher_checkpoint_sha256
    restored = CR1SameInputDagger.construct_algorithm(
        observations,
        SimpleNamespace(num_envs=4, num_actions=4),
        _same_input_config(checkpoint),
        "cpu",
    )
    assert restored.load(payload, None, True)
    assert restored.num_updates == 1

    wrong_teacher_payload = dict(payload)
    wrong_teacher_payload["teacher_checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="different teacher"):
        restored.load(wrong_teacher_payload, None, True)
