"""Bounded visual-input ablation; no training or production configuration changes.

Run with the normal play CLI. Each condition gets the same 32 reset fixtures:
the 28 stable segment starts plus four extra frame-zero replicas. Only the first
episode of each environment is scored; unfinished cases are censored.
"""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Ladder/scripts/rsl_rl"))
import play_transformer_vision_ppo as play

import numpy as np
import torch

from ladder.rsl_rl.modules.end_to_end_visual_actor import EndToEndVisualActor, ORACLE_CONTACT_GOAL_KEY
from ladder.rsl_rl.modules.transformer_vision_actor import (
    CURRENT_PROPRIO_KEY, PROPRIO_HISTORY_KEY, DEPTH_HISTORY_KEY,
)
from ladder.rsl_rl.runners.end_to_end_visual_wrapper import EndToEndVisualPPOVecEnvWrapper
from ladder.tasks.ladder_climb.mdp.contact_task import ContactTaskCommand

MODES = ("normal", "flat_1m", "flat_2m", "frozen_reset", "wrong_stage")


def depth_variant(mode, depth, frozen, donor):
    if mode == "normal":
        return depth
    if mode == "flat_1m":
        return torch.zeros_like(depth)  # Network depth is meters minus one.
    if mode == "flat_2m":
        return torch.ones_like(depth)
    if mode == "frozen_reset":
        return frozen
    if mode == "wrong_stage":
        return donor
    raise ValueError(mode)


def cpu(value):
    return value.detach().cpu().clone()


def main():
    args = play.args_cli
    checkpoint = Path(args.checkpoint).resolve()
    out = args.output_dir.resolve() / checkpoint.stem
    out.mkdir(parents=True, exist_ok=False)
    motion_path = args.motion_file.resolve()
    with np.load(motion_path) as motion:
        fixtures = motion["stable_support_anchor_frames"][:-1].astype(int).tolist() + [0] * 4
    n = len(fixtures)
    count = n * len(MODES)
    if args.num_envs != count or args.max_steps <= 0:
        raise ValueError(f"Use --num_envs {count} and a positive --max_steps")

    def fixed_frames(self, ids):
        frames = torch.tensor(fixtures * len(MODES), device=self.device)
        self.time_steps[ids] = frames[ids]
        self.task_index[ids] = self.motion.stable_reset_segment_ids[self.time_steps[ids]]
        self._fixed_start_episode[ids] = self.time_steps[ids] == 0
        self._secondary_fixed_start_episode[ids] = False
        self._uniform_start_episode[ids] = True
        self._current_reset_valid_fraction[ids] = 1.0

    # Process-local fixture selection, not a change to the environment source.
    ContactTaskCommand._adaptive_sampling = fixed_frames
    cfg = play.parse_env_cfg(args.task, device=args.device, num_envs=count)
    cfg.seed = 42
    cfg.commands.motion.motion_file = str(motion_path)
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    play._disable_randomization(cfg)
    play.configure_depth_camera_timing(cfg, checkpoint=checkpoint)
    agent_cfg = play.cli_args.parse_rsl_rl_cfg(args.task, args)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = play.gym.make(args.task, cfg=cfg)
    env = EndToEndVisualPPOVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions,
                                       clip_observations=agent_cfg.clip_observations)
    base = raw.unwrapped
    c = base.command_manager.get_term("motion")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    obs = env.reset()
    actor = EndToEndVisualActor(
        action_dim=env.num_actions, actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
        use_contact_task=True, use_contact_task_phases=True, use_contact_task_body_goal=True,
        history_frames=obs[PROPRIO_HISTORY_KEY].shape[1], depth_frames=obs[DEPTH_HISTORY_KEY].shape[1],
    ).to(env.device).eval().requires_grad_(False)
    if payload["data_contract"] != actor.data_contract:
        raise ValueError("Checkpoint observation contract mismatch")
    actor.load_state_dict(payload["policy_state_dict"], strict=True)
    actor.reset()
    frozen = obs[DEPTH_HISTORY_KEY].clone()
    # Fixed nonidentity mapping across different reset stages, from normal envs.
    donor_ids = (torch.arange(n, device=env.device) + n // 2) % n
    first_done = torch.zeros(count, device=env.device, dtype=torch.bool)
    returns = torch.zeros(count, device=env.device)
    lengths = torch.zeros(count, device=env.device, dtype=torch.long)
    cases = {}
    shadow = {mode: [] for mode in MODES}
    snapshots, trace = [], []
    checks = {"actor_input_dim": actor.actor_input_dim, "all_actions_finite": True,
              "shadow_identity_max_error": 0.0}
    # Non-depth inputs should match at reset across condition groups.
    checks["initial_nondepth_max_difference"] = {
        key: max(float((obs[key][i*n:(i+1)*n] - obs[key][:n]).abs().max())
                 for i in range(1, len(MODES)))
        for key in (CURRENT_PROPRIO_KEY, PROPRIO_HISTORY_KEY, ORACLE_CONTACT_GOAL_KEY)
    }
    metadata = {
        "checkpoint": str(checkpoint), "iter": payload["iter"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "motion": str(motion_path), "motion_sha256": hashlib.sha256(motion_path.read_bytes()).hexdigest(),
        "task": args.task, "seed": cfg.seed, "dt": base.step_dt, "fixtures": fixtures,
        "modes": MODES, "num_envs": count, "max_steps": args.max_steps,
        "depth_shape": list(frozen.shape), "randomization": False,
        "protocol": "Deterministic actor mean; depth-only interventions. First episode per fixture. "
                    "Five frame-zero replicas are not five independent random seeds. "
                    "Same-state shadows use the normal GRU hidden state and never modify it. "
                    "Closed-loop conditions have their own evolving hidden states. "
                    "Wrong-stage input is a fixed permutation of live normal-environment images. "
                    "Fixtures are coverage checks, not IID deployment success estimates.",
        "reward_weights": {name: term.weight for name, term in
                           zip(base.reward_manager._term_names, base.reward_manager._term_cfgs)},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    try:
        with torch.inference_mode():
            for step in range(args.max_steps):
                active = ~first_done
                before_task = c.task_index.clone()
                original_depth = obs[DEPTH_HISTORY_KEY]
                intervention = original_depth.clone()
                donor = original_depth[donor_ids]
                for i, mode in enumerate(MODES):
                    sl = slice(i*n, (i+1)*n)
                    intervention[sl] = depth_variant(mode, original_depth[sl], frozen[sl], donor)
                inputs = obs.clone(False)
                inputs[DEPTH_HISTORY_KEY] = intervention
                h = actor._inference_hidden_state.clone()
                if h.shape[1] == 0:
                    h = frozen.new_zeros(1, count, 256)
                actions, _ = actor.forward_with_output(inputs)
                checks["all_actions_finite"] &= bool(torch.isfinite(actions).all())
                if not checks["all_actions_finite"]:
                    raise RuntimeError("Non-finite action during evaluation")
                if step % 20 == 0 and active[:n].any():
                    for mode in MODES:
                        replacement = depth_variant(mode, original_depth[:n], frozen[:n], donor)
                        feature = actor.encoder(obs[PROPRIO_HISTORY_KEY][:n], replacement,
                                                hidden_state=h[:, :n], deterministic=True)
                        mean = actor.action_mean(actor.make_actor_input(
                            obs[CURRENT_PROPRIO_KEY][:n], feature.recurrent_feature,
                            obs[ORACLE_CONTACT_GOAL_KEY][:n]))
                        delta = mean - actions[:n]
                        mask = active[:n]
                        shadow[mode].append(cpu(torch.stack((delta[mask].square().mean(-1),
                                                            actions[:n][mask].square().mean(-1)), -1)))
                        if mode == "normal":
                            checks["shadow_identity_max_error"] = max(
                                checks["shadow_identity_max_error"], float(delta.abs().max()))
                    if active[0]:
                        snapshots.append({"step": step, "hidden": cpu(h[:, :1]),
                            "observations": {k: cpu(obs[k][:1]) for k in
                                (CURRENT_PROPRIO_KEY, PROPRIO_HISTORY_KEY, DEPTH_HISTORY_KEY, ORACLE_CONTACT_GOAL_KEY)},
                            "action": cpu(actions[:1])})
                following, rewards, dones, _ = env.step(actions)
                returns += rewards * active
                lengths += active.long()
                trace.append({"task_before": cpu(before_task), "active": cpu(active),
                              "dones": cpu(dones), "actions": cpu(actions)})
                newly_done = dones & active
                for env_id in newly_done.nonzero().flatten().tolist():
                    cases[env_id] = {
                        "mode": MODES[env_id // n], "fixture": env_id % n,
                        "reset_frame": fixtures[env_id % n], "length": int(lengths[env_id]),
                        "reward": float(returns[env_id]), "censored": False,
                        "causes": play._termination_values(base, env_id),
                        "metrics": {key: float(value[env_id]) for key, value in c.last_episode_metrics.items()},
                    }
                first_done |= dones
                actor.reset(dones)
                frozen[dones] = following[DEPTH_HISTORY_KEY][dones]
                obs = following
                if step % 100 == 0:
                    print("DEPTH_AUDIT", step, "first_episodes_done", int(first_done.sum()), "/", count, flush=True)
                if first_done.all():
                    break
        for env_id in (~first_done).nonzero().flatten().tolist():
            cases[env_id] = {"mode": MODES[env_id // n], "fixture": env_id % n,
                "reset_frame": fixtures[env_id % n], "length": int(lengths[env_id]),
                "reward": float(returns[env_id]), "censored": True, "causes": {},
                "metrics": {key: float(value[env_id]) for key, value in c.metrics.items()}}
        summaries = {}
        for mode in MODES:
            rows = [v for v in cases.values() if v["mode"] == mode]
            completed = lambda r: r["metrics"].get("sequence_success", 0) > 0.5 and not r["censored"]
            zeros = [r for r in rows if r["reset_frame"] == 0]
            values = torch.cat(shadow[mode])
            summaries[mode] = {
                "fixtures": len(rows), "successes": sum(completed(r) for r in rows),
                "censored": sum(r["censored"] for r in rows),
                "frame0_count": len(zeros), "frame0_successes": sum(completed(r) for r in zeros),
                "mean_completion_fraction": float(np.mean([r["metrics"].get("sequence_completion_fraction", 0) for r in rows])),
                "shadow_raw_action_rmse": float(values[:, 0].mean().sqrt()),
                "shadow_normal_action_rms": float(values[:, 1].mean().sqrt()),
                "shadow_samples": len(values),
            }
        if checks["shadow_identity_max_error"] > 1e-5:
            raise RuntimeError("Counterfactual identity check failed")
        result = {"metadata": metadata, "checks": checks, "summary": summaries,
                  "cases": [cases[i] for i in sorted(cases)]}
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        torch.save({k: torch.stack([r[k] for r in trace]) for k in trace[0]}, out / "trace.pt")
        torch.save(snapshots, out / "normal_observation_snapshots.pt")
        print("DEPTH_AUDIT_RESULT", json.dumps(summaries), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        play.simulation_app.close()
