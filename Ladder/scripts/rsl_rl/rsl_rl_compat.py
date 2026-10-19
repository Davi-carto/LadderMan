from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch


def convert_legacy_runner_cfg(config: dict) -> dict:
    """Convert IsaacLab's legacy policy config to rsl-rl 5's split models."""

    converted = deepcopy(config)
    if "actor" in converted and "critic" in converted:
        return converted

    policy = converted.pop("policy", None)
    if policy is None:
        raise KeyError("RSL-RL config must contain either policy or actor/critic")

    actor_obs_groups = ["policy"]
    critic_obs_groups = ["policy"]
    configured_groups = converted.get("obs_groups")
    if isinstance(configured_groups, dict):
        actor_obs_groups = configured_groups.get("actor", configured_groups.get("policy", actor_obs_groups))
        critic_obs_groups = configured_groups.get("critic", configured_groups.get("policy", critic_obs_groups))
    converted["obs_groups"] = {"actor": actor_obs_groups, "critic": critic_obs_groups}

    actor_normalization = policy.get(
        "actor_obs_normalization", converted.get("empirical_normalization", False)
    )
    critic_normalization = policy.get(
        "critic_obs_normalization", converted.get("empirical_normalization", False)
    )
    distribution_class = (
        "HeteroscedasticGaussianDistribution"
        if policy.get("state_dependent_std", False)
        else "GaussianDistribution"
    )
    converted["actor"] = {
        "class_name": policy.get("actor_class_name", "MLPModel"),
        "hidden_dims": policy["actor_hidden_dims"],
        "activation": policy["activation"],
        "obs_normalization": actor_normalization,
        "distribution_cfg": {
            "class_name": distribution_class,
            "init_std": policy["init_noise_std"],
            "std_type": policy.get("noise_std_type", "scalar"),
        },
    }
    converted["critic"] = {
        "class_name": policy.get("critic_class_name", "MLPModel"),
        "hidden_dims": policy["critic_hidden_dims"],
        "activation": policy["activation"],
        "obs_normalization": critic_normalization,
    }
    return converted


def convert_legacy_distillation_runner_cfg(config: dict) -> dict:
    """Convert IsaacLab's StudentTeacher config to rsl-rl 5 models.

    IsaacLab 2.3 still exposes the legacy combined ``policy`` schema while
    rsl-rl 5 constructs independent ``student`` and ``teacher`` MLP models.
    Keeping this translation next to the PPO compatibility adapter makes the
    exact observation mapping and normalization choices explicit.
    """

    converted = deepcopy(config)
    if "student" in converted and "teacher" in converted:
        return converted

    policy = converted.pop("policy", None)
    if policy is None:
        raise KeyError(
            "RSL-RL distillation config must contain either policy or "
            "student/teacher"
        )

    configured_groups = converted.get("obs_groups", {})
    student_groups = configured_groups.get(
        "student", configured_groups.get("policy", ["policy"])
    )
    teacher_groups = configured_groups.get("teacher", ["teacher"])
    converted["obs_groups"] = {
        "student": list(student_groups),
        "teacher": list(teacher_groups),
    }

    distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": policy["init_noise_std"],
        "std_type": policy.get("noise_std_type", "scalar"),
    }
    converted["student"] = {
        "class_name": "MLPModel",
        "hidden_dims": list(policy["student_hidden_dims"]),
        "activation": policy["activation"],
        "obs_normalization": policy["student_obs_normalization"],
        "distribution_cfg": deepcopy(distribution_cfg),
    }
    converted["teacher"] = {
        "class_name": "MLPModel",
        "hidden_dims": list(policy["teacher_hidden_dims"]),
        "activation": policy["activation"],
        "obs_normalization": policy["teacher_obs_normalization"],
        "distribution_cfg": deepcopy(distribution_cfg),
    }
    return converted


def load_checkpoint_payload(path: str) -> dict:
    """Load a checkpoint payload from disk."""

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected checkpoint payload to be a dict, got {type(loaded).__name__}")
    return loaded


def is_split_actor_critic_checkpoint(payload: dict) -> bool:
    """Return True for checkpoints that store actor and critic separately."""

    return "model_state_dict" not in payload and "actor_state_dict" in payload


def convert_split_actor_checkpoint_to_actor_module_state_dict(payload: dict) -> OrderedDict:
    """Map a split actor checkpoint onto the RSL-RL 2 actor submodule layout.

    RSL-RL 5 checkpoints save the actor MLP under 'actor_state_dict' with
    'mlp.*' parameter names and a separate action-distribution tensor. The
    older monolithic actor-critic module needs only the MLP for deterministic
    play.
    """

    if "actor_state_dict" not in payload:
        raise KeyError("Split checkpoint is missing actor_state_dict")

    converted = OrderedDict()
    for key, value in payload["actor_state_dict"].items():
        if key == "distribution.std_param":
            continue
        if not key.startswith("mlp."):
            raise KeyError(f"Unsupported split actor parameter name: {key}")
        converted[key.removeprefix("mlp.")] = value
    return converted
