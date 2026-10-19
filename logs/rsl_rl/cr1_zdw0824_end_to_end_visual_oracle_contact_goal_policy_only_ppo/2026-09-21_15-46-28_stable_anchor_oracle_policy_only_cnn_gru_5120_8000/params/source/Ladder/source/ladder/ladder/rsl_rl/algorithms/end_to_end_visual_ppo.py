"""Policy-only recurrent PPO with gradients through the complete visual encoder."""

from __future__ import annotations

from itertools import chain
import json
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from ladder.rsl_rl.modules.end_to_end_visual_actor import EndToEndVisualActor, ORACLE_CONTACT_GOAL_KEY
from ladder.rsl_rl.modules.transformer_vision_actor import (
    CURRENT_PROPRIO_KEY, DEPTH_HISTORY_KEY, PROPRIO_HISTORY_KEY,
)
from ladder.rsl_rl.storage.transformer_vision_ppo_storage import (
    CRITIC_KEY, TransformerVisionPPORolloutStorage, TransformerVisionPPOTransition,
)


def _critic(input_dim, hidden_dims):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, 1))


class EndToEndVisualPPO:
    """Replay raw visual sequences so PPO gradients reach the complete encoder."""

    def __init__(
        self, policy, critic, storage, *, num_learning_epochs=5,
        num_mini_batches=4, sequence_micro_batch_envs=32, clip_param=0.2,
        gamma=0.99, lam=0.95, value_loss_coef=1.0, entropy_coef=0.005,
        learning_rate=3.0e-4, min_learning_rate=1.0e-5, max_learning_rate=1.0e-3,
        max_grad_norm=1.0, use_clipped_value_loss=True, schedule="adaptive",
        desired_kl=0.01, action_clip=None, device="cpu",
        termination_manager=None,
    ):
        if not 1 <= num_mini_batches <= storage.num_envs:
            raise ValueError("num_mini_batches must be between one and num_envs")
        if num_learning_epochs < 1 or sequence_micro_batch_envs < 1:
            raise ValueError("Epoch and microbatch counts must be positive")
        if not 0 < min_learning_rate <= learning_rate <= max_learning_rate:
            raise ValueError("Invalid learning-rate bounds")
        if schedule not in ("fixed", "adaptive"):
            raise ValueError("Unsupported learning-rate schedule")
        if schedule == "adaptive" and (desired_kl is None or desired_kl <= 0):
            raise ValueError("Adaptive PPO requires positive desired_kl")
        if action_clip is not None and action_clip <= 0.0:
            raise ValueError("action_clip must be positive or None")
        self.device = torch.device(device)
        self.policy = policy.to(self.device).float()
        self.critic = critic.to(self.device).float()
        self.storage = storage
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.sequence_micro_batch_envs = int(sequence_micro_batch_envs)
        self.clip_param, self.gamma, self.lam = clip_param, gamma, lam
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.learning_rate = learning_rate
        self.min_learning_rate, self.max_learning_rate = min_learning_rate, max_learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule, self.desired_kl = schedule, desired_kl
        self.action_clip = None if action_clip is None else float(action_clip)
        self.optimizer = torch.optim.Adam(
            chain(self.policy.parameters(), self.critic.parameters()), lr=learning_rate,
        )
        self.data_contract = self.policy.encoder.data_contract
        self.num_updates = 0
        self.visual_hidden_state = torch.zeros(1, storage.num_envs, 256, device=self.device)
        self.raw_rewards = torch.zeros_like(storage.rewards)
        self.time_outs = torch.zeros_like(storage.dones)
        self._termination_manager = termination_manager
        self._diagnostic_dir = None
        self._pending = None
        self._returns_ready = False
        self.rnd = self.intrinsic_rewards = None

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        cls._validate_observations(obs)
        settings = dict(cfg["algorithm"])
        settings.pop("class_name", None)
        if cfg.get("multi_gpu"):
            raise ValueError("End-to-end visual PPO currently supports one device")
        for name in ("rnd_cfg", "symmetry_cfg"):
            if settings.pop(name, None):
                raise ValueError(f"End-to-end visual PPO does not support {name}")
        if settings.pop("optimizer", "adam").lower() != "adam":
            raise ValueError("End-to-end visual PPO uses Adam")
        if settings.pop("normalize_advantage_per_mini_batch", False):
            raise ValueError("Advantages are normalized over the whole rollout")
        policy_cfg = cfg["policy"]
        policy = EndToEndVisualActor(
            action_dim=env.num_actions, actor_hidden_dims=policy_cfg["actor_hidden_dims"],
            init_noise_std=policy_cfg["init_noise_std"],
            use_oracle_contact_goal=policy_cfg.get("use_oracle_contact_goal", False),
        )
        if policy.use_oracle_contact_goal:
            policy.make_actor_input(obs[CURRENT_PROPRIO_KEY],
                                    obs[CURRENT_PROPRIO_KEY].new_zeros(env.num_envs, 256),
                                    obs.get(ORACLE_CONTACT_GOAL_KEY))
        critic = _critic(obs[CRITIC_KEY].shape[-1], policy_cfg["critic_hidden_dims"])
        storage = TransformerVisionPPORolloutStorage(
            cfg["num_steps_per_env"], env.num_envs, env.num_actions,
            obs[CRITIC_KEY].shape[-1], device, actor_input_dim=policy.actor_input_dim,
            proprio_history_frames=25, reconstruction_supervision=False,
        )
        return cls(policy, critic, storage, device=device,
                   action_clip=cfg.get("clip_actions"),
                   termination_manager=getattr(getattr(env, "unwrapped", None),
                                               "termination_manager", None), **settings)

    @staticmethod
    def _validate_observations(obs):
        required = (CURRENT_PROPRIO_KEY, PROPRIO_HISTORY_KEY, DEPTH_HISTORY_KEY, CRITIC_KEY)
        missing = [name for name in required if name not in obs]
        if missing:
            raise KeyError(f"Missing end-to-end visual PPO observations: {missing}")
        count = obs[CRITIC_KEY].shape[0]
        shapes = {
            CURRENT_PROPRIO_KEY: (count, 85),
            PROPRIO_HISTORY_KEY: (count, 25, 85),
            DEPTH_HISTORY_KEY: (count, 2, 60, 80),
        }
        for name, shape in shapes.items():
            if tuple(obs[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(obs[name].shape)}")
        if obs[CRITIC_KEY].ndim != 2:
            raise ValueError("Critic observations must have shape [B,C]")

    @torch.no_grad()
    def act(self, obs):
        if self._pending is not None:
            raise RuntimeError("process_env_step must follow every act call")
        self._validate_observations(obs)
        if self.storage.step == 0:
            self.storage.begin_rollout(self.visual_hidden_state)
        output = self.policy.encoder(
            obs[PROPRIO_HISTORY_KEY], obs[DEPTH_HISTORY_KEY],
            hidden_state=self.visual_hidden_state, deterministic=True,
        )
        actor_input = self.policy.make_actor_input(
            obs[CURRENT_PROPRIO_KEY], self.policy.actor_visual_feature(output),
            obs.get(ORACLE_CONTACT_GOAL_KEY),
        )
        mean = self.policy.action_mean(actor_input)
        std = self.policy.output_std.expand_as(mean)
        distribution = Normal(mean, std)
        actions = distribution.sample()
        # IsaacLab may reuse observation buffers during env.step. Every pending
        # tensor (including labels and current proprioception) owns its data.
        values = dict(
            proprio_history=obs[PROPRIO_HISTORY_KEY], depth_history=obs[DEPTH_HISTORY_KEY],
            critic_observation=obs[CRITIC_KEY], actor_input=actor_input, actions=actions,
            old_action_mean=mean, old_action_std=std,
            old_action_log_prob=distribution.log_prob(actions).sum(-1, keepdim=True),
            value=self.critic(obs[CRITIC_KEY]),
        )
        self._pending = TransformerVisionPPOTransition(**{
            name: value.detach().clone() for name, value in values.items()
        })
        self.visual_hidden_state = output.next_hidden_state.detach()
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        del obs
        if self._pending is None:
            raise RuntimeError("act must precede process_env_step")
        reward = rewards.detach().clone().to(self.device)
        done = dones.to(device=self.device, dtype=torch.bool)
        timeout = extras.get("time_outs", torch.zeros_like(done)).to(self.device).bool().clone()
        if self._termination_manager is not None:
            timeout &= ~self._termination_manager.terminated.bool()
        self.raw_rewards[self.storage.step].copy_(reward[:, None])
        self.time_outs[self.storage.step].copy_(timeout)
        # Match installed RSL-RL's current-state value bootstrap on truncation.
        reward += self.gamma * self._pending.value.squeeze(-1) * timeout
        self.storage.add(self._pending, reward=reward, done=done, next_proprio=None)
        self.visual_hidden_state = self.policy.encoder.reset_hidden_state(
            self.visual_hidden_state, done,
        )
        self._pending = None

    @torch.no_grad()
    def compute_returns(self, obs):
        self.storage.compute_returns(self.critic(obs[CRITIC_KEY]), self.gamma, self.lam)
        self._returns_ready = True

    def _denominators(self, ids):
        return {"samples": self.storage.num_steps * ids.numel()}

    def _batch_terms(self, ids, denominators, reset_before=None):
        """Return additive contributions to a single logical minibatch loss."""
        st = self.storage
        if reset_before is None:
            reset_before = st.reset_before_mask()
        mean, output = self.policy.forward_sequence(
            st.actor_inputs[:, ids, :85], st.proprio_history[:, ids], st.depth_history[:, ids],
            st.initial_hidden_state[:, ids], reset_before[:, ids],
            oracle_contact_goal=(st.actor_inputs[:, ids, -32:]
                                 if self.policy.use_oracle_contact_goal else None),
        )
        std = self.policy.output_std.expand_as(mean)
        distribution = Normal(mean, std)
        logprob = distribution.log_prob(st.actions[:, ids]).sum(-1, keepdim=True)
        difference = logprob - st.old_action_log_prob[:, ids]
        ratio = difference.exp()
        advantages = st.advantages[:, ids]
        count = denominators["samples"]
        surrogate = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(
            1.0 - self.clip_param, 1.0 + self.clip_param,
        )).sum() / count
        value = self.critic(st.critic_observations[:, ids])
        value_error = (value - st.returns[:, ids]).square()
        if self.use_clipped_value_loss:
            clipped = st.values[:, ids] + (value - st.values[:, ids]).clamp(
                -self.clip_param, self.clip_param,
            )
            value_error = torch.maximum(value_error, (clipped - st.returns[:, ids]).square())
        with torch.no_grad():
            old_mean, old_std = st.old_action_mean[:, ids], st.old_action_std[:, ids]
            # The official RSL KL sums action dimensions, then averages samples.
            kl = (torch.log(std / old_std) + (old_std.square() + (old_mean - mean).square())
                  / (2.0 * std.square()) - 0.5).sum() / count
        return {
            "surrogate": surrogate,
            "value": value_error.sum() / count,
            "entropy": distribution.entropy().sum() / count,
            "policy_kl": kl,
            "clip_fraction": ((ratio.detach() - 1).abs() > self.clip_param).float().sum() / count,
            "ratio_error": (ratio.detach() - 1).abs().max(),
            "logprob_error": difference.detach().abs().max(),
        }

    def _gradient_probe(self, terms):
        named = list(self.policy.named_parameters())
        parameters = [value for _, value in named]
        ppo = torch.autograd.grad(terms["surrogate"], parameters, retain_graph=True, allow_unused=True)
        result = {}
        for prefix in (
            "all",
            "encoder",
            "encoder.state_encoder",
            "encoder.depth_encoder",
            "encoder.depth_projection",
            "encoder.gru",
            "action_mlp",
        ):
            selected = [i for i, (name, _) in enumerate(named)
                        if prefix == "all" or name.startswith(prefix + ".")]
            norm = sum(
                (ppo[i].detach().square().sum() for i in selected if ppo[i] is not None),
                torch.zeros((), device=self.device),
            ).sqrt()
            result[f"ppo_{prefix}_grad_l2_probe"] = float(norm)
        return result

    def _adapt_learning_rate(self, kl):
        if self.schedule == "adaptive":
            if kl > 2.0 * self.desired_kl:
                self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
            elif 0.0 < kl < self.desired_kl / 2.0:
                self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate

    def update(self):
        if not self._returns_ready:
            raise RuntimeError("compute_returns must precede update")
        totals = {}
        diagnostics = {}
        snapshot = self._diagnostic_snapshot()
        updates = 0
        max_actor_norm = max_critic_norm = 0.0
        self.train_mode()
        # Reset boundaries do not change during PPO epochs. Materialize the
        # complete rollout mask once, not once per sequence microbatch.
        reset_before = self.storage.reset_before_mask()
        for _ in range(self.num_learning_epochs):
            permutation = torch.randperm(self.storage.num_envs, device=self.device)
            for ids in torch.tensor_split(permutation, self.num_mini_batches):
                denominators = self._denominators(ids)
                batch_sums = None
                self.optimizer.zero_grad(set_to_none=True)
                for start in range(0, ids.numel(), self.sequence_micro_batch_envs):
                    micro = ids[start:start + self.sequence_micro_batch_envs]
                    terms = self._batch_terms(micro, denominators, reset_before)
                    if updates == 0 and start == 0:
                        diagnostics.update(self._gradient_probe(terms))
                        diagnostics["preupdate_ratio_max_abs_error_probe"] = float(terms["ratio_error"])
                        diagnostics["preupdate_logprob_max_abs_error_probe"] = float(terms["logprob_error"])
                    loss = (
                        terms["surrogate"]
                        + self.value_loss_coef * terms["value"]
                        - self.entropy_coef * terms["entropy"]
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite end-to-end visual PPO loss")
                    loss.backward()
                    # Keep logging reductions on device: converting each loss
                    # to float forced a CUDA synchronization per metric per
                    # microbatch. Float64 matches the former Python sums;
                    # loss gradients and logical-batch denominators are unchanged.
                    metric_names = [name for name in terms if name not in ("ratio_error", "logprob_error")]
                    metric_values = torch.stack([terms[name].detach() for name in metric_names]
                                                + [loss.detach()]).double()
                    batch_sums = metric_values if batch_sums is None else batch_sums + metric_values
                batch_totals = dict(zip(metric_names + ["total"], batch_sums.cpu().tolist()))
                self._adapt_learning_rate(batch_totals["policy_kl"])
                actor_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True,
                )
                critic_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True,
                )
                max_actor_norm = max(max_actor_norm, float(actor_norm))
                max_critic_norm = max(max_critic_norm, float(critic_norm))
                self.optimizer.step()
                for name, value in batch_totals.items():
                    totals[name] = totals.get(name, 0.0) + value
                updates += 1
        st = self.storage
        with torch.no_grad():
            diagnostics.update({
                "rollout_action_max_abs": float(st.actions.abs().max()),
                "rollout_action_rms": float(st.actions.square().mean().sqrt()),
                "rollout_action_mean_max_abs": float(st.old_action_mean.abs().max()),
                "rollout_action_mean_rms": float(st.old_action_mean.square().mean().sqrt()),
                "critic_explained_variance": float(1 - (st.returns - st.values).var(unbiased=False)
                    / st.returns.var(unbiased=False).clamp_min(1e-8)),
                "return_mean": float(st.returns.mean()), "value_mean": float(st.values.mean()),
                "action_std_min": float(self.policy.output_std.min()),
                "action_std_max": float(self.policy.output_std.max()),
            })
            if self.action_clip is not None:
                diagnostics["rollout_action_safety_clip_fraction"] = float(
                    (st.actions.abs() > self.action_clip).float().mean()
                )
        result = {name: value / updates for name, value in totals.items()}
        result.update(diagnostics)
        result.update({"learning_rate": self.learning_rate, "optimizer_steps": float(updates),
                       "actor_grad_norm_before_clip_max": max_actor_norm,
                       "critic_grad_norm_before_clip_max": max_critic_norm,
                       "weighted_value": self.value_loss_coef * result["value"],
                       "weighted_entropy": -self.entropy_coef * result["entropy"]})
        if self._diagnostic_dir is not None:
            record = {"iteration": self.num_updates, **result}
            with (self._diagnostic_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if snapshot is not None:
                snapshot["update_metrics"] = result
                torch.save(snapshot, self._diagnostic_dir / f"iteration_{self.num_updates:06d}_sample.pt")
        self.storage.clear()
        self._returns_ready = False
        self.num_updates += 1
        return result

    def configure_diagnostics(self, log_dir):
        self._diagnostic_dir = Path(log_dir) / "end_to_end_visual_ppo_diagnostics"
        self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "data_contract": self.data_contract, "training_contract": self._training_contract(),
            "policy_network": repr(self.policy), "critic_network": repr(self.critic),
            "rollout": "100% current policy; no teacher or imitation labels",
            "actor_contains_reference": self.policy.use_oracle_contact_goal,
            "actor_input_layout": (
                "current_proprio85, gru_feature256, next_stable_goal_b12, "
                "actual_limb_pos_b12, actual_support_contact4, desired_contact4"
                if self.policy.use_oracle_contact_goal
                else "current_proprio85, gru_feature256"
            ),
            "oracle_semantics": (
                "next-stable reference intent plus measured limb position and ground/rung contact; "
                "all positions share the robot anchor frame"
            ),
            "gradient_probe": "first sequence microbatch, before updates; PPO surrogate only",
            "snapshot": "every 500 updates, first up to 16 envs, complete rollout with before-update weights",
            "hidden_reset": "done[t] resets before t+1; GRU every policy step; camera cached at 30 Hz",
            "optimizer": "one Adam over the policy-only visual encoder, actor, and privileged critic",
            "reconstruction_supervision": False,
            "action_safety_clip": self.action_clip,
        }
        (self._diagnostic_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    def _diagnostic_snapshot(self):
        if self._diagnostic_dir is None or self.num_updates % 500 != 0:
            return None
        st = self.storage
        ids = torch.arange(min(16, st.num_envs), device=self.device)
        fields = ("proprio_history", "depth_history", "critic_observations", "actor_inputs",
                  "actions", "old_action_mean", "old_action_std", "old_action_log_prob",
                  "values", "rewards", "returns", "advantages", "dones")
        return {
            "iteration": self.num_updates, "environment_ids": ids.cpu(),
            "parameter_timing": "before_optimizer_update",
            "policy_state_dict": {name: value.detach().cpu().clone()
                                  for name, value in self.policy.state_dict().items()},
            "critic_state_dict": {name: value.detach().cpu().clone()
                                  for name, value in self.critic.state_dict().items()},
            "initial_hidden_state": st.initial_hidden_state[:, ids].detach().cpu().clone(),
            "raw_rewards": self.raw_rewards[:, ids].cpu().clone(),
            "time_outs": self.time_outs[:, ids].cpu().clone(),
            **{name: getattr(st, name)[:, ids].detach().cpu().clone() for name in fields},
        }

    def _training_contract(self):
        names = ("num_learning_epochs", "num_mini_batches", "clip_param", "gamma", "lam",
                 "value_loss_coef", "entropy_coef", "min_learning_rate", "max_learning_rate",
                 "max_grad_norm", "use_clipped_value_loss", "schedule", "desired_kl")
        return {"algorithm": "end_to_end_visual_policy_only_ppo_v2", "precision": "float32",
                "num_steps_per_env": self.storage.num_steps,
                **{name: getattr(self, name) for name in names}}

    def save(self):
        return {"data_contract": self.data_contract, "training_contract": self._training_contract(),
                "policy_state_dict": self.policy.state_dict(), "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(), "learning_rate": self.learning_rate,
                "num_updates": self.num_updates}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if loaded_dict.get("data_contract") != self.data_contract:
            raise RuntimeError("Incompatible end-to-end visual PPO checkpoint data contract")
        options = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg is not None:
            options.update(load_cfg)
        if options["optimizer"] and loaded_dict.get("training_contract") != self._training_contract():
            raise RuntimeError("End-to-end visual PPO checkpoint training contract differs")
        if options["actor"]:
            self.policy.load_state_dict(loaded_dict["policy_state_dict"], strict=strict)
        if options["critic"]:
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if options["optimizer"]:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = float(loaded_dict["learning_rate"])
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate
        if options["iteration"]:
            self.num_updates = int(loaded_dict["num_updates"])
        # A completed rollout may leave an inference-mode tensor here. Replace
        # it instead of mutating it outside inference mode when loading.
        self.visual_hidden_state = torch.zeros(1, self.storage.num_envs, 256, device=self.device)
        self.policy.reset()
        self._pending = None
        self._returns_ready = False
        self.storage.clear()
        return options["iteration"]

    def train_mode(self):
        self.policy.train()
        self.critic.train()

    def eval_mode(self):
        self.policy.eval()
        self.critic.eval()

    def get_policy(self):
        return self.policy


__all__ = ["EndToEndVisualPPO"]
