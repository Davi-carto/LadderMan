import importlib.util
from pathlib import Path

import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "rsl_rl_compat.py"
_SPEC = importlib.util.spec_from_file_location("ladder_rsl_rl_compat", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
convert_legacy_runner_cfg = _MODULE.convert_legacy_runner_cfg
convert_legacy_distillation_runner_cfg = (
    _MODULE.convert_legacy_distillation_runner_cfg
)
convert_split_actor_checkpoint_to_actor_module_state_dict = (
    _MODULE.convert_split_actor_checkpoint_to_actor_module_state_dict
)
is_split_actor_critic_checkpoint = _MODULE.is_split_actor_critic_checkpoint


def test_legacy_policy_config_is_split_for_rsl_rl_5():
    config = {
        "empirical_normalization": True,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 256],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        "algorithm": {"class_name": "PPO"},
    }

    converted = convert_legacy_runner_cfg(config)

    assert "policy" not in converted
    assert converted["actor"]["class_name"] == "MLPModel"
    assert converted["critic"]["class_name"] == "MLPModel"
    assert converted["actor"]["hidden_dims"] == [512, 256, 256]
    assert converted["critic"]["hidden_dims"] == [512, 256, 128]
    assert converted["actor"]["distribution_cfg"]["class_name"] == "GaussianDistribution"
    assert converted["obs_groups"] == {"actor": ["policy"], "critic": ["critic"]}
    assert "policy" in config


def test_new_runner_config_is_copied_without_conversion():
    config = {"actor": {"class_name": "MLPModel"}, "critic": {"class_name": "MLPModel"}}

    converted = convert_legacy_runner_cfg(config)

    assert converted == config
    assert converted is not config


def test_legacy_distillation_config_maps_both_models_to_one_policy_tensor():
    config = {
        "obs_groups": {"policy": ["policy"], "teacher": ["policy"]},
        "policy": {
            "init_noise_std": 0.1,
            "noise_std_type": "scalar",
            "student_obs_normalization": False,
            "teacher_obs_normalization": False,
            "student_hidden_dims": [512, 256, 256],
            "teacher_hidden_dims": [512, 256, 256],
            "activation": "elu",
        },
        "algorithm": {"class_name": "Distillation"},
    }

    converted = convert_legacy_distillation_runner_cfg(config)

    assert "policy" not in converted
    assert converted["obs_groups"] == {
        "student": ["policy"],
        "teacher": ["policy"],
    }
    assert converted["student"]["hidden_dims"] == [512, 256, 256]
    assert converted["teacher"]["hidden_dims"] == [512, 256, 256]
    assert converted["student"]["obs_normalization"] is False
    assert converted["teacher"]["obs_normalization"] is False
    assert converted["student"]["distribution_cfg"] == {
        "class_name": "GaussianDistribution",
        "init_std": 0.1,
        "std_type": "scalar",
    }
    assert "policy" in config


def test_visual_actor_class_is_preserved_during_conversion():
    config = {
        "obs_groups": {"actor": ["visual_proprio", "visual_depth"], "critic": ["critic"]},
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [2048, 1024, 512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_class_name": "ladder.models:LadderVisualActorModel",
        },
    }

    converted = convert_legacy_runner_cfg(config)

    assert converted["actor"]["class_name"] == "ladder.models:LadderVisualActorModel"
    assert converted["critic"]["class_name"] == "MLPModel"
    assert converted["obs_groups"]["actor"] == ["visual_proprio", "visual_depth"]


def test_split_checkpoint_actor_weights_map_to_current_actor_module():
    payload = {
        "actor_state_dict": {
            "distribution.std_param": torch.ones(2),
            "mlp.0.weight": torch.ones(3, 4),
            "mlp.0.bias": torch.zeros(3),
            "mlp.2.weight": torch.full((2, 3), 5.0),
        },
        "critic_state_dict": {},
    }

    assert is_split_actor_critic_checkpoint(payload)
    converted = convert_split_actor_checkpoint_to_actor_module_state_dict(payload)

    assert list(converted) == ["0.weight", "0.bias", "2.weight"]
    torch.testing.assert_close(converted["0.weight"], torch.ones(3, 4))
