"""Teacher-only control and held-out gradient isolation without importing IsaacSim."""

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


_SOURCE_DIR = Path(__file__).resolve().parents[1] / "source/ladder"
sys.path.insert(0, str(_SOURCE_DIR))
_DAGGER_DIR = _SOURCE_DIR / "ladder/tasks/ladder_climb/config/cr1/dagger"
_PACKAGE_NAME = "cr1_teacher_only_bc_test_package"
_PACKAGE = ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(_DAGGER_DIR)]
sys.modules.setdefault(_PACKAGE_NAME, _PACKAGE)
CR1TeacherOnlyVisualBC = importlib.import_module(
    f"{_PACKAGE_NAME}.teacher_only_algorithm"
).CR1TeacherOnlyVisualBC
elu_mlp = importlib.import_module(f"{_PACKAGE_NAME}.student_teacher").elu_mlp


@pytest.fixture(autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def checkpoint(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(3)
        teacher = elu_mlp(17, [32, 256], 25)
    path = tmp_path / "teacher.pt"
    torch.save(
        {
            "actor_state_dict": {
                **{f"mlp.{key}": value for key, value in teacher.state_dict().items()},
                "distribution.std_param": torch.full((25,), 0.4),
            }
        },
        path,
    )
    return str(path)


def _config(checkpoint):
    return {
        "num_steps_per_env": 2,
        "multi_gpu": None,
        "policy": {"actor_hidden_dims": [32, 256], "init_noise_std": 1.0},
        "algorithm": {
            "class_name": "unused",
            "teacher_checkpoint": checkpoint,
            "teacher_hidden_dims": [32, 256],
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 0.5,
            "action_kl_weight": 2.0,
            "feature_weight": 1.0,
            "bc_warmup_iterations": 0,
            "teacher_rollout_fraction": 1.0,
            "validation_num_envs": 1,
            "diagnostic_snapshot_interval": 500,
        },
    }


def _env(num_envs=4):
    motion = SimpleNamespace(
        time_steps=torch.arange(num_envs, dtype=torch.long),
        motion=SimpleNamespace(time_step_total=120),
        _episode_start_time_steps=torch.arange(num_envs, dtype=torch.long),
        contact_indicator=torch.zeros(num_envs, 4),
        contact_rung_id=torch.zeros(num_envs, 4, dtype=torch.long),
        metrics={},
    )
    terminations = {
        "time_out": torch.zeros(num_envs, dtype=torch.bool),
        "motion_complete": torch.zeros(num_envs, dtype=torch.bool),
    }
    env = SimpleNamespace(
        num_envs=num_envs,
        num_actions=25,
        step_dt=0.02,
        action_manager=SimpleNamespace(
            active_terms=["joint_pos"],
            get_term=lambda name: SimpleNamespace(_joint_names=[f"joint_{index}" for index in range(25)]),
        ),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        command_manager=SimpleNamespace(get_term=lambda name: motion),
        termination_manager=SimpleNamespace(
            active_terms=list(terminations), get_term=lambda name: terminations[name],
        ),
    )
    env.unwrapped = env
    return env


def _observations(num_envs=4):
    proprio = torch.randn(num_envs, 9, 85)
    proprio[:, :, -1] = 1.0
    return TensorDict(
        {
            "proprio_history": proprio,
            "depth_history": torch.rand(num_envs, 2, 60, 80),
            "teacher": torch.randn(num_envs, 17),
        },
        batch_size=[num_envs],
    )


def _collect(algorithm, observations, env):
    for _ in range(algorithm.storage.num_steps):
        actions = algorithm.act(observations)
        torch.testing.assert_close(actions, algorithm.teacher(observations["teacher"]))
        env.episode_length_buf += 1
        algorithm.process_env_step(
            observations, torch.zeros(env.num_envs), torch.zeros(env.num_envs, dtype=torch.bool), {},
        )


def test_teacher_only_control_never_decays_and_updates_all_student_subnetworks(checkpoint):
    observations, env = _observations(), _env()
    algorithm = CR1TeacherOnlyVisualBC.construct_algorithm(observations, env, _config(checkpoint), "cpu")
    algorithm.update_iteration = 100000
    student_before = {name: value.detach().clone() for name, value in algorithm.policy.named_parameters()}
    teacher_before = {name: value.detach().clone() for name, value in algorithm.teacher.named_parameters()}
    _collect(algorithm, observations, env)
    algorithm.compute_returns(observations)
    losses = algorithm.update()

    assert losses["rollout_teacher_fraction"] == 1.0
    assert losses["weighted_action_kl"] == pytest.approx(2.0 * losses["distillation_kl"])
    assert losses["weighted_feature_mse"] == pytest.approx(losses["distillation_feature_mse"])
    assert losses["distillation_total"] == pytest.approx(
        losses["weighted_action_kl"] + losses["weighted_feature_mse"]
    )
    assert losses["distillation_kl"] > 0
    assert losses["distillation_feature_mse"] > 0
    changed = {name for name, value in algorithm.policy.named_parameters() if not torch.equal(student_before[name], value)}
    for prefix in ("depth_encoder.", "feature_mlp.", "action_head.", "log_std"):
        assert any(name.startswith(prefix) for name in changed)
    for name, value in algorithm.teacher.named_parameters():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
        assert not value.requires_grad
        assert value.grad is None


def test_held_out_environment_never_enters_any_training_batch(checkpoint):
    observations, env = _observations(), _env()
    algorithm = CR1TeacherOnlyVisualBC.construct_algorithm(observations, env, _config(checkpoint), "cpu")
    assert algorithm.storage.num_training_envs == env.num_envs - 1
    for index in range(env.num_envs):
        observations["proprio_history"][index].fill_(float(index))
    _collect(algorithm, observations, env)
    for _ in range(3):
        labels = []
        for batch in algorithm.storage.batches(num_mini_batches=4, num_epochs=1):
            labels.extend(batch["proprio_history"][:, 0, 0].tolist())
        assert sorted(labels) == [0, 0, 1, 1, 2, 2]


def test_changing_only_validation_data_cannot_change_training_update(checkpoint):
    observations, first_env, second_env = _observations(), _env(), _env()
    altered = observations.clone()
    altered["teacher"][-1].add_(1000.0)
    altered["proprio_history"][-1].mul_(100.0)
    altered["depth_history"][-1].fill_(1.0)
    torch.manual_seed(19)
    first = CR1TeacherOnlyVisualBC.construct_algorithm(observations, first_env, _config(checkpoint), "cpu")
    torch.manual_seed(19)
    second = CR1TeacherOnlyVisualBC.construct_algorithm(altered, second_env, _config(checkpoint), "cpu")
    _collect(first, observations, first_env)
    _collect(second, altered, second_env)
    torch.manual_seed(29)
    first.update()
    torch.manual_seed(29)
    second.update()
    for name, value in first.policy.state_dict().items():
        torch.testing.assert_close(value, second.policy.state_dict()[name], rtol=0, atol=0)


def test_checkpoint_marks_teacher_only_mode_and_rejects_mixed_resume(checkpoint):
    observations, env = _observations(), _env()
    config = _config(checkpoint)
    algorithm = CR1TeacherOnlyVisualBC.construct_algorithm(observations, env, config, "cpu")
    _collect(algorithm, observations, env)
    algorithm.update()
    saved = copy.deepcopy(algorithm.save())
    assert saved["rollout_mode"] == "teacher_only_bc_v1"
    restored = CR1TeacherOnlyVisualBC.construct_algorithm(observations, _env(), config, "cpu")
    assert restored.load(saved, load_cfg=None, strict=True)
    assert restored.update_iteration == algorithm.update_iteration
    for name, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(value, restored.policy.state_dict()[name], rtol=0, atol=0)
    saved.pop("rollout_mode")
    with pytest.raises((RuntimeError, ValueError), match="(?i)teacher|rollout|mode"):
        restored.load(saved, load_cfg=None, strict=True)


def test_episode_diagnostics_keep_terminal_pre_action_frame_across_auto_reset(checkpoint):
    observations, env = _observations(), _env()
    algorithm = CR1TeacherOnlyVisualBC.construct_algorithm(observations, env, _config(checkpoint), "cpu")
    motion = env.command_manager.get_term("motion")
    motion.time_steps[1] = 79
    motion._episode_start_time_steps[1] = 60
    algorithm.act(observations)
    # Simulate the environment changing the reference cursor during automatic reset.
    motion.time_steps[1] = 5
    motion._episode_start_time_steps[1] = 5
    env.termination_manager.get_term("motion_complete")[1] = True
    done = torch.tensor([False, True, False, False])
    algorithm.process_env_step(observations, torch.zeros(4), done, {})
    assert len(algorithm._episodes) == 1
    recorded = algorithm._episodes[0]
    assert recorded["env_id"].tolist() == [1]
    assert recorded["last_pre_action_frame"].tolist() == [79]
    assert recorded["start_frame"].tolist() == [60]
    assert recorded["observed_length"].tolist() == [1]
    assert recorded["termination/motion_complete"].tolist() == [True]
    assert algorithm._episode_steps.tolist() == [1, 0, 1, 1]
    assert algorithm._episode_ids.tolist() == [0, 1, 0, 0]


def test_inference_rollout_writes_provenance_metrics_episodes_and_aligned_snapshots(checkpoint, tmp_path):
    observations, env = _observations(), _env()
    observations["proprio_history"][-1, :, -1] = -1
    config = _config(checkpoint)
    config.update(seed=42, save_interval=500)
    motion = env.command_manager.get_term("motion")
    motion.metrics["error_joint_pos"] = torch.arange(4, dtype=torch.float32) / 10
    motion.time_steps.copy_(torch.tensor([0, 10, 20, 110]))
    motion._episode_start_time_steps.copy_(torch.tensor([0, 0, 0, 100]))
    env.episode_length_buf[-1] = 77  # Randomized timeout counters are not observed history lengths.
    algorithm = CR1TeacherOnlyVisualBC.construct_algorithm(observations, env, config, "cpu")
    motion_file = tmp_path / "motion.npz"
    motion_file.write_bytes(b"synthetic motion bytes for provenance hashing")
    tensorboard_rows = []
    logger = SimpleNamespace(
        writer=SimpleNamespace(add_scalar=lambda name, value, iteration: tensorboard_rows.append((name, value, iteration)))
    )
    algorithm.configure_diagnostics(tmp_path / "run", config, motion_file, checkpoint, logger)
    original_student_std = algorithm.policy.output_std.detach().clone()
    original_student_mean = algorithm.policy(observations).detach().clone()
    original_teacher_mean = algorithm.teacher(observations["teacher"]).detach().clone()
    first_teacher_input = observations["teacher"].clone()
    first_frame = motion.time_steps.clone()
    with torch.inference_mode():
        actions = algorithm.act(observations)
        torch.testing.assert_close(actions, original_teacher_mean)
        done = torch.tensor([False, False, False, True])
        # Auto-reset mutates reference state, but diagnostics must retain the terminal pre-action frame.
        motion.time_steps[-1] = 5
        motion._episode_start_time_steps[-1] = 5
        env.termination_manager.get_term("motion_complete")[-1] = True
        algorithm.process_env_step(observations, torch.zeros(4), done, {})
        next_observations = observations.clone()
        next_observations["teacher"].add_(0.4)
        next_observations["proprio_history"][:, -1, 59:84] = actions
        env.termination_manager.get_term("motion_complete").zero_()
        algorithm.act(next_observations)
        algorithm.process_env_step(next_observations, torch.zeros(4), torch.zeros(4, dtype=torch.bool), {})
    losses = algorithm.update()

    log_dir = tmp_path / "run" / "teacher_bc"
    manifest = json.loads((log_dir / "manifest.json").read_text())
    assert manifest["rollout_controller"] == "frozen_teacher_deterministic_action_mean_only"
    assert manifest["rollout_mode"] == "teacher_only_bc_v1"
    assert manifest["validation"]["num_envs"] == 1
    assert manifest["train_env_ids"] == [0, 2]
    assert manifest["validation_env_ids"] == [3, 3]
    assert manifest["loss"]["rl_loss"] is False
    assert manifest["loss"]["visual_auxiliary_loss"] is False
    assert manifest["action_joint_names"]["joint_pos"] == [f"joint_{index}" for index in range(25)]
    assert manifest["motion"]["sha256"] == hashlib.sha256(motion_file.read_bytes()).hexdigest()
    assert manifest["teacher_checkpoint"]["sha256"] == hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    assert (log_dir / "sources.tar.gz").is_file()

    rows = [json.loads(line) for line in (log_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    metrics = rows[0]
    assert metrics["iteration"] == 0
    assert metrics["train/count"] == 6
    assert metrics["validation/count"] == 2
    assert metrics["validation/down/count"] == 2
    assert metrics["optimizer/rollout_teacher_fraction"] == 1
    assert metrics["validation/action_rmse"] == losses["teacher_validation_action_rmse"]
    assert metrics["teacher_episodes/train/count"] == 0
    assert metrics["teacher_episodes/validation/count"] == 1
    assert metrics["teacher_episodes/validation/motion_complete_fraction"] == 1
    assert metrics["teacher_rollout/validation/error_joint_pos"] == pytest.approx(0.3)
    assert ("BC/validation/action_rmse", metrics["validation/action_rmse"], 0) in tensorboard_rows

    episodes = [json.loads(line) for line in (log_dir / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == 1
    assert episodes[0]["validation"] is True
    assert episodes[0]["last_pre_action_frame"] == 110
    assert episodes[0]["start_frame"] == 100
    assert episodes[0]["observed_length"] == 1
    assert episodes[0]["environment_length"] == 78
    assert episodes[0]["termination/motion_complete"] is True
    assert episodes[0]["pre_action_metrics/error_joint_pos"] == pytest.approx(0.3)

    snapshot = torch.load(log_dir / "snapshots" / "rollout_000000.pt", weights_only=False)
    selected = torch.tensor(manifest["snapshot_env_ids"])
    assert snapshot["proprio_history"].shape == (2, 3, 9, 85)
    assert snapshot["depth_history"].shape == (2, 3, 2, 60, 80)
    assert snapshot["teacher_feature"].shape == (2, 3, 256)
    assert snapshot["done"].tolist() == [[False, False, True], [False, False, False]]
    torch.testing.assert_close(snapshot["last_pre_action_frame"][0], first_frame[selected])
    torch.testing.assert_close(snapshot["teacher"][0], first_teacher_input[selected])
    torch.testing.assert_close(snapshot["teacher_action_mean"][0], original_teacher_mean[selected])
    torch.testing.assert_close(snapshot["student_action_mean"][0], original_student_mean[selected])
    torch.testing.assert_close(snapshot["student_std"], original_student_std)
    torch.testing.assert_close(snapshot["teacher_action_mean"][1], algorithm.teacher(snapshot["teacher"][1]))
    assert snapshot["policy_updates_at_collection"] == 0
    assert snapshot["rollout_mode"] == "teacher_only_bc_v1"
    assert snapshot["episode_id"][:, -1].tolist() == [0, 1]
    assert not algorithm._episodes and not algorithm._snapshots


def test_teacher_only_checkpoint_can_be_loaded_by_existing_pure_student_inference(checkpoint):
    observations, env = _observations(), _env()
    config = _config(checkpoint)
    algorithm = CR1TeacherOnlyVisualBC.construct_algorithm(observations, env, config, "cpu")
    _collect(algorithm, observations, env)
    algorithm.update()
    pure_algorithm_class = importlib.import_module(f"{_PACKAGE_NAME}.pure_algorithm").CR1PureVisualDagger
    playback = pure_algorithm_class.construct_algorithm(observations, _env(), config, "cpu")
    playback.load(algorithm.save(), load_cfg={"actor": True, "optimizer": False, "iteration": False}, strict=True)
    actor_observations = observations.select("proprio_history", "depth_history")
    torch.testing.assert_close(playback.act_inference(actor_observations), algorithm.policy(actor_observations))
