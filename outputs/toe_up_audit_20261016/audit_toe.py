"""Bounded deterministic toe-up diagnostic, without optimizer/model/env edits.

Uses the normal play CLI, nominal environment and seven reset fixtures. Samples
after reward computation, before automatic reset and goal/reference advancement.
Only the first episode of each fixture is analyzed. This is not a success-rate
estimate or an intervention on the phase machine.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Ladder/scripts/rsl_rl"))
import play_transformer_vision_ppo as play

import torch
from ladder.rsl_rl.modules.end_to_end_visual_actor import EndToEndVisualActor
from ladder.rsl_rl.modules.transformer_vision_actor import PROPRIO_HISTORY_KEY, DEPTH_HISTORY_KEY
from ladder.rsl_rl.runners.end_to_end_visual_wrapper import EndToEndVisualPPOVecEnvWrapper
from ladder.tasks.ladder_climb.mdp.contact_task import ContactTaskCommand

FRAMES = [0, 373, 514, 686, 824, 996, 1142]


def cpu_fields(fields):
    return {key: value.detach().cpu().clone() for key, value in fields.items()}


def fixed_frames(self, ids):
    frames = torch.tensor(FRAMES, device=self.device)
    self.time_steps[ids] = frames[ids]
    self.task_index[ids] = self.motion.stable_reset_segment_ids[self.time_steps[ids]]
    self._fixed_start_episode[ids] = self.time_steps[ids] == 0
    self._secondary_fixed_start_episode[ids] = False
    self._uniform_start_episode[ids] = True
    self._current_reset_valid_fraction[ids] = 1.0


def main():
    args = play.args_cli
    if args.num_envs != len(FRAMES) or not 0 < args.max_steps <= 1500:
        raise ValueError("Use seven environments and 1–1500 steps")
    checkpoint = Path(args.checkpoint).resolve()
    out = args.output_dir.resolve() / checkpoint.stem
    out.mkdir(parents=True, exist_ok=False)
    ContactTaskCommand._adaptive_sampling = fixed_frames
    cfg = play.parse_env_cfg(args.task, device=args.device, num_envs=len(FRAMES))
    cfg.seed = 42
    cfg.commands.motion.motion_file = str(args.motion_file.resolve())
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
    guide = c.reference_guidance
    obs = env.reset()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = EndToEndVisualActor(
        action_dim=env.num_actions, actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
        use_contact_task=True, use_contact_task_phases=True, use_contact_task_body_goal=True,
        history_frames=obs[PROPRIO_HISTORY_KEY].shape[1],
        depth_frames=obs[DEPTH_HISTORY_KEY].shape[1],
    ).to(env.device).eval().requires_grad_(False)
    if payload["data_contract"] != actor.data_contract:
        raise ValueError("Checkpoint observation contract mismatch")
    actor.load_state_dict(payload["policy_state_dict"], strict=True)
    actor.reset()
    feet = [c.robot.body_names.index(n) for n in ("left_ankle_x_link", "right_ankle_x_link")]
    metadata = {
        "checkpoint": str(checkpoint), "iter": payload["iter"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "motion": str(args.motion_file.resolve()), "task": args.task,
        "dt": base.step_dt, "fixtures": FRAMES, "max_steps": args.max_steps,
        "seed": cfg.seed, "randomization": False, "action": "deterministic mean",
        "sampling": "reward time before auto reset and goal/reference advancement; phase_before is issued phase",
        "reward_names": list(base.reward_manager._term_names),
        "reward_weights": [t.weight for t in base.reward_manager._term_cfgs],
        "joint_names": list(c.robot.joint_names), "body_names": list(c.cfg.body_names),
        "rung_sizes": [list(c._ladder_asset.cfg.rigid_objects[n].spawn.size)
                       for n in c.cfg.ladder_rung_box_names],
        "robot_asset": str(cfg.scene.robot.spawn.asset_path),
        "foot_limb_names": c.limb_body_names[2:],
        "limb_distance_thresholds_m": c._limb_target_distance_thresholds.cpu().tolist(),
        "pitch_range_deg": c.cfg.task_foot_swing_pitch_range_deg,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    torch.save(cpu_fields({
        "reference_joint_pos": c.motion.joint_pos,
        "reference_body_quat": c.motion.body_quat_w,
        "reference_body_pos": c.motion.body_pos_w,
        "support_ids": c.motion.stable_support_rung_id,
        "anchors": c.task_anchors, "origins": base.scene.env_origins,
        "rung_pos": c._ladder_asset.data.object_pos_w[:, c._ladder_rung_box_indexes],
        "rung_quat": c._ladder_asset.data.object_quat_w[:, c._ladder_rung_box_indexes],
    }), out / "reference.pt")
    captured = {}
    compute = base.reward_manager.compute

    def record_reward(dt):
        result = compute(dt)
        p = c.contact_phases
        fields = {
            "phase_after": p.phase, "acquired_after": p.acquired,
            "actual": c.actual_stable_support_rung_id, "target": c.target_ids,
            "source": c.source_ids, "active_limbs": c.active_limbs,
            "task": c.task_index, "task_steps": c.task_steps,
            "completed": c.completed, "failed": c.failed,
            "reward_terms": base.reward_manager._step_reward * dt, "reward": result,
            "joint_pos": c.robot_joint_pos, "joint_vel": c.robot.data.joint_vel,
            "torque": c.robot.data.applied_torque,
            "feet_pos": c.robot.data.body_pos_w[:, feet],
            "feet_quat": c.robot.data.body_quat_w[:, feet],
            "limb_pos": c.robot_limb_pos_w, "goal_pos": c.goal_pos_w,
            "limb_speed": c.robot.data.body_lin_vel_w[:, c.limb_body_indexes].norm(dim=-1),
            "rung_force": torch.stack([s.data.force_matrix_w.sum(1)
                                       for s in c._stable_anchor_limb_contact_sensors], 1),
            **{"foot_" + k: v for k, v in c.foot_reward_diagnostics.items()},
            **{"guide_" + k: guide.diagnostics[k] for k in (
                "frame", "matched_frame", "valid", "pause_age_s", "reward_scale",
                "weighted_rate", "raw_weighted_rate")},
        }
        captured.clear()
        captured.update(cpu_fields(fields))
        return result

    base.reward_manager.compute = record_reward
    first_done = torch.zeros(len(FRAMES), device=env.device, dtype=torch.bool)
    records, episodes = [], []
    try:
        with torch.inference_mode():
            for step in range(args.max_steps):
                before = cpu_fields({"phase_before": c.contact_phases.phase,
                                     "acquired_before": c.contact_phases.acquired,
                                     "first_episode": ~first_done})
                actions, _ = actor.forward_with_output(obs)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Non-finite action")
                obs, rewards, dones, _ = env.step(actions)
                records.append({**before, **captured, **cpu_fields({"actions": actions, "dones": dones})})
                for env_id in (dones & ~first_done).nonzero().flatten().tolist():
                    episodes.append({"env": env_id, "reset_frame": FRAMES[env_id], "steps": step + 1,
                        "causes": play._termination_values(base, env_id),
                        "metrics": {k: float(v[env_id]) for k, v in c.last_episode_metrics.items()}})
                first_done |= dones
                actor.reset(dones)
                if step % 100 == 0:
                    print("TOE_AUDIT", step, "first_episodes_done", int(first_done.sum()), flush=True)
                if first_done.all():
                    break
    finally:
        if records:
            torch.save({k: torch.stack([r[k] for r in records]) for k in records[0]}, out / "trace.pt")
        (out / "episodes.json").write_text(json.dumps(episodes, indent=2))
        print("TOE_AUDIT_SAVED", str(out), len(records), flush=True)
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        play.simulation_app.close()
