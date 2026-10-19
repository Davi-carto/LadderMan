"""Collect TransformerVision supervision under a frozen RSL-RL policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution

from ladder.rsl_rl.algorithms import TransformerVisionOptimizationResult, TransformerVisionPretrainer
from ladder.rsl_rl.modules import (
    TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT,
    VARIATIONAL_TRANSFORMER_VISION_DATA_CONTRACT,
    TaskGoalTransformerVisionEncoder,
    TransformerVisionEncoder,
    TransformerVisionOutput,
    VariationalTransformerVisionEncoder,
)
from ladder.rsl_rl.storage import TransformerVisionRolloutStorage


_POLICY_KEY = "policy"
_PROPRIO_HISTORY_KEY = "transformer_vision_proprio_history"
_DEPTH_HISTORY_KEY = "transformer_vision_depth_history"
_HEIGHTMAP_KEY = "transformer_vision_limb_heightmaps"
_HEIGHTMAP_VALID_KEY = "transformer_vision_heightmap_valid"
_HEIGHTMAP_HIT_CLASS_KEY = "transformer_vision_heightmap_hit_class"
_VELOCITY_KEY = "transformer_vision_velocity_target"
_NEXT_PROPRIO_KEY = "transformer_vision_next_proprio_target"
_LIMB_GOAL_KEY = "transformer_vision_limb_goal_target"
_LIMB_GOAL_VALID_KEY = "transformer_vision_limb_goal_valid"
_CONTACT_INDICATOR_KEY = "transformer_vision_contact_indicator_target"
_MOTION_JOINT_POS_KEY = "transformer_vision_motion_joint_pos_target"
_MOTION_ROOT_ORIENTATION_KEY = "transformer_vision_motion_root_orientation_6d_target"
_MOTION_TARGET_VALID_KEY = "transformer_vision_motion_target_valid"


def _as_tensor_dict(observations: dict | TensorDict, *, num_envs: int, device: str | torch.device) -> TensorDict:
    """Expose the named TransformerVision groups in RSL-RL's TensorDict format."""

    if isinstance(observations, TensorDict):
        observations = observations.to_dict()
    inputs = observations["transformer_vision_inputs"]
    targets = observations["transformer_vision_targets"]
    auxiliary_targets = observations["transformer_vision_auxiliary_targets"]
    packed = {
            _POLICY_KEY: observations[_POLICY_KEY],
            _PROPRIO_HISTORY_KEY: inputs["proprio_history"],
            _DEPTH_HISTORY_KEY: inputs["depth_history"],
            _HEIGHTMAP_KEY: targets["relative_height"],
            _HEIGHTMAP_VALID_KEY: targets["hit_valid"],
            _HEIGHTMAP_HIT_CLASS_KEY: targets["hit_class"],
            _VELOCITY_KEY: auxiliary_targets["base_lin_vel"],
            _NEXT_PROPRIO_KEY: auxiliary_targets["next_proprio"],
            "critic": observations["critic"],
    }
    task_names = (
        "task_limb_goal_b",
        "task_limb_goal_valid",
        "task_contact_indicator",
        "task_motion_joint_pos",
        "task_motion_root_orientation_6d",
        "task_motion_valid",
    )
    available_task_names = tuple(name for name in task_names if name in auxiliary_targets)
    if available_task_names and available_task_names != task_names:
        raise KeyError(
            "TransformerVision task-goal labels must be configured together; found "
            f"{available_task_names}"
        )
    if available_task_names:
        packed.update(
            {
                _LIMB_GOAL_KEY: auxiliary_targets["task_limb_goal_b"].reshape(
                    num_envs, 4, 3
                ),
                _LIMB_GOAL_VALID_KEY: auxiliary_targets["task_limb_goal_valid"],
                _CONTACT_INDICATOR_KEY: auxiliary_targets["task_contact_indicator"],
                _MOTION_JOINT_POS_KEY: auxiliary_targets["task_motion_joint_pos"],
                _MOTION_ROOT_ORIENTATION_KEY: auxiliary_targets[
                    "task_motion_root_orientation_6d"
                ],
                _MOTION_TARGET_VALID_KEY: auxiliary_targets[
                    "task_motion_valid"
                ].reshape(num_envs),
            }
        )
    return TensorDict(
        packed,
        batch_size=[num_envs],
        device=device,
    )


class TransformerVisionVecEnvWrapper(VecEnv):
    """Keep frozen-policy, visual input, and supervision groups separate."""

    def __init__(self, env, clip_actions: float | None = None) -> None:
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.max_episode_length = env.unwrapped.max_episode_length
        self.num_actions = env.unwrapped.action_manager.total_action_dim
        self.env.reset()

    @property
    def cfg(self):
        return self.env.unwrapped.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.unwrapped.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        if hasattr(self.env.unwrapped, "observation_manager"):
            observations = self.env.unwrapped.observation_manager.compute()
        else:
            observations = self.env.unwrapped._get_observations()
        return _as_tensor_dict(observations, num_envs=self.num_envs, device=self.device)

    def reset(self) -> TensorDict:
        observations, _ = self.env.reset()
        return _as_tensor_dict(observations, num_envs=self.num_envs, device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.bool)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return _as_tensor_dict(observations, num_envs=self.num_envs, device=self.device), rewards, dones, extras

    def close(self) -> None:
        self.env.close()


class FrozenRslPolicy(torch.nn.Module):
    """An exact frozen MLP actor restored from an RSL-RL 5 split checkpoint."""

    def __init__(
        self,
        policy_observation: torch.Tensor,
        checkpoint_path: str | Path,
        *,
        action_dim: int = 25,
        hidden_dims: tuple[int, ...] = (512, 256, 256),
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "actor_state_dict" not in payload:
            raise ValueError(
                f"{checkpoint_path} is not an RSL-RL 5 split checkpoint with actor_state_dict"
            )
        first_layer = payload["actor_state_dict"].get("mlp.0.weight")
        if first_layer is None or first_layer.ndim != 2:
            raise ValueError(f"{checkpoint_path} does not contain actor mlp.0.weight")
        self.observation_dim = int(first_layer.shape[1])
        if policy_observation.ndim != 2 or policy_observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"The frozen CR1 policy must receive [B,{self.observation_dim}] observations, got "
                f"{tuple(policy_observation.shape)}"
            )
        self.actor = MLPModel(
            TensorDict({_POLICY_KEY: policy_observation}, batch_size=[policy_observation.shape[0]]),
            obs_groups={"actor": [_POLICY_KEY]},
            obs_set="actor",
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": GaussianDistribution,
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
        self.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        self.actor.to(policy_observation.device)
        self.actor.eval()
        self.actor.requires_grad_(False)

    def forward(self, policy_observation: torch.Tensor) -> torch.Tensor:
        if policy_observation.ndim != 2 or policy_observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Frozen policy observation must have shape [B,{self.observation_dim}], "
                f"got {tuple(policy_observation.shape)}"
            )
        observations = TensorDict({_POLICY_KEY: policy_observation}, batch_size=[policy_observation.shape[0]])
        return self.actor(observations, stochastic_output=False)


@dataclass
class FrozenPolicyVisionRollout:
    """One no-update visual diagnostic collected on a frozen-policy trajectory."""

    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    output: TransformerVisionOutput
    losses: dict[str, torch.Tensor]
    actor_parameter_change_max_abs: float
    vision_parameter_change_max_abs: float
    vision_gradient_norm: float


@dataclass
class TransformerVisionTrainingStep:
    """Metrics from one frozen-policy rollout and visual-only update."""

    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    output: TransformerVisionOutput
    losses: dict[str, torch.Tensor]
    actor_parameter_change_max_abs: float
    vision_parameter_change_max_abs: float
    vision_gradient_norm: float


class FrozenPolicyVisionRunner:
    """RSL-RL-style rollout collector without an actor update or PPO phase."""

    def __init__(
        self,
        env: TransformerVisionVecEnvWrapper,
        checkpoint_path: str | Path,
        *,
        rollout_steps: int,
        encoder: TransformerVisionEncoder | TaskGoalTransformerVisionEncoder | None = None,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
        mini_batch_envs: int = 256,
    ) -> None:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        self.env = env
        self.device = torch.device(env.device)
        self.rollout_steps = rollout_steps
        self.frozen_policy_checkpoint = str(Path(checkpoint_path).expanduser().resolve())
        self._vision_learning_rate = learning_rate
        self._vision_max_grad_norm = max_grad_norm
        self._vision_mini_batch_envs = mini_batch_envs
        observations = env.get_observations()
        self.encoder = (encoder or TransformerVisionEncoder()).to(self.device)
        self._require_task_goal = (
            self.encoder.data_contract == TASK_GOAL_TRANSFORMER_VISION_DATA_CONTRACT
        )
        self._validate_observations(observations)
        self.frozen_policy = FrozenRslPolicy(observations[_POLICY_KEY], checkpoint_path).to(self.device)
        self._policy_observation_dim = self.frozen_policy.observation_dim
        self._validate_observations(observations)
        self.pretrainer = TransformerVisionPretrainer(
            self.encoder,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            mini_batch_envs=mini_batch_envs,
        )
        self.storage = TransformerVisionRolloutStorage(rollout_steps, env.num_envs, self.device)
        # This is detached at every rollout boundary. It carries deployment-time
        # GRU context across rollout chunks and is reset per environment on done.
        self._visual_hidden_state = torch.zeros(1, env.num_envs, 256, device=self.device)

    def collect_no_update(
        self,
        step_callback: Callable[[int, TransformerVisionOutput, TensorDict], None] | None = None,
        reset_requested: Callable[[], bool] | None = None,
    ) -> FrozenPolicyVisionRollout:
        """Run the frozen actor, replay its visual data, and measure a gradient only."""

        actor_snapshot = self._snapshot(self.frozen_policy)
        vision_snapshot = self._snapshot(self.encoder)
        actions, rewards, dones = self._collect_trajectory(
            step_callback=step_callback,
            reset_requested=reset_requested,
        )

        output, losses = self.pretrainer.loss(self.storage)
        gradient_norm = self.pretrainer.gradient_norm_without_step(losses["total"])
        actor_change = self._max_parameter_change(self.frozen_policy, actor_snapshot)
        vision_change = self._max_parameter_change(self.encoder, vision_snapshot)
        if actor_change != 0.0:
            raise RuntimeError(f"Frozen actor changed during visual diagnostic: {actor_change}")
        if vision_change != 0.0:
            raise RuntimeError(f"No-update visual diagnostic changed encoder parameters: {vision_change}")
        return FrozenPolicyVisionRollout(
            actions=actions,
            rewards=rewards,
            dones=dones,
            output=output,
            losses=losses,
            actor_parameter_change_max_abs=actor_change,
            vision_parameter_change_max_abs=vision_change,
            vision_gradient_norm=gradient_norm,
        )

    def collect_and_update(self) -> TransformerVisionTrainingStep:
        """Collect one real trajectory and update only TransformerVision once."""

        actor_snapshot = self._snapshot(self.frozen_policy)
        actions, rewards, dones = self._collect_trajectory()
        update = self.pretrainer.optimize(self.storage)
        actor_change = self._max_parameter_change(self.frozen_policy, actor_snapshot)
        if actor_change != 0.0:
            raise RuntimeError(f"Frozen actor changed during visual-only training: {actor_change}")
        return TransformerVisionTrainingStep(
            actions=actions,
            rewards=rewards,
            dones=dones,
            output=update.output,
            losses=update.losses,
            actor_parameter_change_max_abs=actor_change,
            vision_parameter_change_max_abs=update.parameter_change_max_abs,
            vision_gradient_norm=update.gradient_norm,
        )

    def evaluate_depth_ablation(
        self,
    ) -> dict[str, tuple[TransformerVisionOutput, dict[str, torch.Tensor]]]:
        """Evaluate whether height-map estimates depend on the stored depth."""

        return self.pretrainer.evaluate_depth_ablation(self.storage)

    def save_visual_checkpoint(self, path: str | Path, *, iteration: int) -> None:
        """Save only the trainable visual module and its optimizer state."""

        payload = self.pretrainer.state_dict()
        payload.update(
            {
                "iteration": iteration,
                "frozen_policy_checkpoint": self.frozen_policy_checkpoint,
            }
        )
        torch.save(payload, Path(path))

    def load_visual_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
        allow_legacy_variational: bool = False,
    ) -> int:
        """Restore a visual-only checkpoint without changing the frozen actor."""

        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected visual checkpoint dict, got {type(payload).__name__}")
        is_legacy_variational = (
            payload.get("data_contract") == VARIATIONAL_TRANSFORMER_VISION_DATA_CONTRACT
        )
        if is_legacy_variational and not allow_legacy_variational:
            raise RuntimeError(
                "This checkpoint uses the retained variational TransformerVision encoder. "
                "It may be inspected explicitly, but new training must use the deterministic "
                "physical-state-59 data contract."
            )
        if (
            is_legacy_variational
            and self.encoder.data_contract != VARIATIONAL_TRANSFORMER_VISION_DATA_CONTRACT
        ):
            self.encoder = VariationalTransformerVisionEncoder().to(self.device)
            self.pretrainer = TransformerVisionPretrainer(
                self.encoder,
                learning_rate=self._vision_learning_rate,
                max_grad_norm=self._vision_max_grad_norm,
                mini_batch_envs=self._vision_mini_batch_envs,
            )
        self.pretrainer.load_state_dict(payload, load_optimizer=load_optimizer)
        return int(payload.get("iteration", 0))

    def _collect_trajectory(
        self,
        step_callback: Callable[[int, TransformerVisionOutput, TensorDict], None] | None = None,
        reset_requested: Callable[[], bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Store one sequence produced solely by the frozen nonvisual actor."""

        self.storage.clear()
        self.storage.initial_hidden_state.copy_(self._visual_hidden_state)
        observations = self.env.get_observations().to(self.device)
        actions = []
        rewards = []
        dones = []
        self.frozen_policy.eval()
        self.encoder.eval()

        def apply_manual_reset(step: int) -> TensorDict:
            reset_observations = self.env.reset().to(self.device)
            self._visual_hidden_state.zero_()
            if step == 0:
                self.storage.initial_hidden_state.zero_()
            else:
                self.storage.mark_previous_transition_done()
                dones[-1] = torch.ones_like(dones[-1], dtype=torch.bool)
            print(
                "[TRANSFORMER VISION] Manual environment reset applied; visual GRU state cleared.",
                flush=True,
            )
            return reset_observations

        for step in range(self.rollout_steps):
            if reset_requested is not None and reset_requested():
                observations = apply_manual_reset(step)
            while True:
                self._validate_observations(observations)
                with torch.inference_mode():
                    action = self.frozen_policy(observations[_POLICY_KEY])
                    visual_output = self.encoder(
                        observations[_PROPRIO_HISTORY_KEY],
                        observations[_DEPTH_HISTORY_KEY],
                        hidden_state=self._visual_hidden_state,
                        deterministic=True,
                    )
                if step_callback is not None:
                    step_callback(step, visual_output, observations)
                # OpenCV can deliver R from inside the live callback. Apply it
                # before env.step so no action from the old state is executed.
                if reset_requested is None or not reset_requested():
                    break
                observations = apply_manual_reset(step)
            next_observations, reward, done, _ = self.env.step(action)
            next_observations = next_observations.to(self.device)
            self.storage.add(
                proprio_history=observations[_PROPRIO_HISTORY_KEY],
                depth_history=observations[_DEPTH_HISTORY_KEY],
                velocity=observations[_VELOCITY_KEY],
                limb_heightmaps=observations[_HEIGHTMAP_KEY],
                heightmap_valid=observations[_HEIGHTMAP_VALID_KEY],
                heightmap_hit_class=observations[_HEIGHTMAP_HIT_CLASS_KEY],
                next_proprio=next_observations[_NEXT_PROPRIO_KEY],
                dones=done.to(self.device),
                limb_goal_b=observations.get(_LIMB_GOAL_KEY),
                limb_goal_valid=observations.get(_LIMB_GOAL_VALID_KEY),
                contact_indicator=observations.get(_CONTACT_INDICATOR_KEY),
                motion_joint_pos=observations.get(_MOTION_JOINT_POS_KEY),
                motion_root_orientation_6d=observations.get(
                    _MOTION_ROOT_ORIENTATION_KEY
                ),
                motion_valid=observations.get(_MOTION_TARGET_VALID_KEY),
            )
            actions.append(action.detach())
            rewards.append(reward.detach().to(self.device))
            dones.append(done.detach().to(self.device))
            self._visual_hidden_state = visual_output.next_hidden_state.detach()
            self._visual_hidden_state *= (~done.to(self.device, dtype=torch.bool))[None, :, None]
            observations = next_observations
        return torch.stack(actions), torch.stack(rewards), torch.stack(dones)

    @staticmethod
    def _snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
        return [parameter.detach().clone() for parameter in module.parameters()]

    @staticmethod
    def _max_parameter_change(module: torch.nn.Module, snapshot: list[torch.Tensor]) -> float:
        return max(
            (float((parameter.detach() - previous).abs().max()) for parameter, previous in zip(module.parameters(), snapshot, strict=True)),
            default=0.0,
        )

    def _validate_observations(self, observations: TensorDict) -> None:
        expected_shapes = {
            _PROPRIO_HISTORY_KEY: (10, 85),
            _DEPTH_HISTORY_KEY: (2, 60, 80),
            _HEIGHTMAP_KEY: (4, 10, 3),
            _HEIGHTMAP_VALID_KEY: (4, 10, 3),
            _HEIGHTMAP_HIT_CLASS_KEY: (4, 10, 3),
            _VELOCITY_KEY: (3,),
            _NEXT_PROPRIO_KEY: (85,),
        }
        policy_shape = tuple(observations[_POLICY_KEY].shape) if _POLICY_KEY in observations else None
        expected_policy_dim = getattr(self, "_policy_observation_dim", None)
        if (
            policy_shape is None
            or len(policy_shape) != 2
            or (expected_policy_dim is not None and policy_shape[-1] != expected_policy_dim)
        ):
            expectation = "[B,N]" if expected_policy_dim is None else f"[B,{expected_policy_dim}]"
            raise ValueError(f"{_POLICY_KEY} must have shape {expectation}, got {policy_shape}")
        for key, trailing_shape in expected_shapes.items():
            if key not in observations or tuple(observations[key].shape[1:]) != trailing_shape:
                received = None if key not in observations else tuple(observations[key].shape)
                raise ValueError(f"{key} must have shape [B,{','.join(map(str, trailing_shape))}], got {received}")
        if self._require_task_goal:
            task_shapes = {
                _LIMB_GOAL_KEY: (4, 3),
                _LIMB_GOAL_VALID_KEY: (4,),
                _CONTACT_INDICATOR_KEY: (4,),
                _MOTION_JOINT_POS_KEY: (25,),
                _MOTION_ROOT_ORIENTATION_KEY: (6,),
                _MOTION_TARGET_VALID_KEY: (),
            }
            for key, trailing_shape in task_shapes.items():
                if key not in observations or tuple(observations[key].shape[1:]) != trailing_shape:
                    received = None if key not in observations else tuple(observations[key].shape)
                    raise ValueError(
                        f"{key} must have shape [B,{','.join(map(str, trailing_shape))}], "
                        f"got {received}"
                    )
