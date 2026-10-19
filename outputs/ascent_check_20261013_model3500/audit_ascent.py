"""Read-only policy diagnostic. No optimizer updates or production-code patches.

Uses the normal play setup; only reset-frame selection is replaced in this process.
Reward-time snapshots are taken BEFORE automatic resets and command advancement.
"""

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path('/home/zhangdawei/LadderMan')
sys.path.insert(0, str(ROOT / 'Ladder/scripts/rsl_rl'))
import play_transformer_vision_ppo as play

import torch
from ladder.tasks.ladder_climb.mdp.contact_task import ContactTaskCommand
from ladder.rsl_rl.runners.end_to_end_visual_amp_wrapper import EndToEndVisualAMPPPOVecEnvWrapper

FRAMES = [373, 514, 686, 824, 996, 1142, 408, 562, 727, 873, 1038, 1195]
OUT = ROOT / 'outputs/ascent_check_20261013_model3500'


def fixed_frames(self, ids):
    frames = torch.tensor(FRAMES, device=self.device)
    self.time_steps[ids] = frames[ids]
    self.task_index[ids] = self.motion.stable_reset_segment_ids[self.time_steps[ids]]
    self._fixed_start_episode[ids] = False
    self._secondary_fixed_start_episode[ids] = False
    self._uniform_start_episode[ids] = True
    self._current_reset_valid_fraction[ids] = 1


def copy_cpu(fields):
    return {k: v.detach().cpu().clone() for k, v in fields.items()}


def main():
    args = play.args_cli
    args.checkpoint = Path(args.checkpoint).resolve()
    assert args.num_envs == len(FRAMES)
    ContactTaskCommand._adaptive_sampling = fixed_frames
    cfg = play.parse_env_cfg(args.task, device=args.device, num_envs=len(FRAMES))
    cfg.commands.motion.motion_file = str(args.motion_file.resolve())
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    play._disable_randomization(cfg)
    agent_cfg = play.cli_args.parse_rsl_rl_cfg(args.task, args)
    runner_cfg = agent_cfg.to_dict()
    play.configure_depth_camera_timing(cfg, checkpoint=args.checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = play.gym.make(args.task, cfg=cfg)
    env = EndToEndVisualAMPPPOVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions,
                                          clip_observations=getattr(agent_cfg, 'clip_observations', None))
    load_cfg = play._configure_end_to_end_playback(runner_cfg, len(FRAMES))
    runner = play.OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    c = raw.unwrapped.command_manager.get_term('motion')
    base = raw.unwrapped
    feet = [c.robot.body_names.index(n) for n in ('left_ankle_x_link', 'right_ankle_x_link')]
    guide = c.reference_guidance
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        'frames': FRAMES, 'task': args.task, 'dt': base.step_dt,
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'scope': 'six ascent anchor fixtures and six mid-swing controls; not iid success rates',
        'seed': 42, 'randomization': False, 'steps_per_checkpoint': args.max_steps,
        'timing': 'before action + after RewardManager.compute, before auto reset; phase_after already updated',
        'reward_names': list(base.reward_manager._term_names),
        'reward_weights': [t.weight for t in base.reward_manager._term_cfgs],
        'body_names': list(c.cfg.body_names), 'robot_joint_names': list(c.robot.joint_names),
        'source_files': ['isaaclab/managers/reward_manager.py', 'isaaclab/envs/manager_based_rl_env.py',
                         'Ladder/scripts/rsl_rl/play_transformer_vision_ppo.py'],
        'contact_threshold_n': c.cfg.contact_force_threshold,
        'confirmation_steps': c.cfg.stable_anchor_confirmation_steps,
        'rung_filter_paths': [list(s.cfg.filter_prim_paths_expr) for s in c._stable_anchor_limb_contact_sensors],
        'rung_sizes': [list(c._ladder_asset.cfg.rigid_objects[n].spawn.size) for n in c.cfg.ladder_rung_box_names],
        'robot_asset': str(cfg.scene.robot.spawn.asset_path),
        'ground_world_z': float(cfg.scene.ground.init_state.pos[2]),
    }
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    torch.save(copy_cpu({
        'reference_body_pos': c.motion.body_pos_w, 'reference_body_quat': c.motion.body_quat_w,
        'reference_joint_pos': c.motion.joint_pos, 'support_ids': c.motion.stable_support_rung_id,
        'anchors': c.task_anchors, 'guide_lower': guide.clock.lower, 'guide_upper': guide.clock.upper,
        'origins': base.scene.env_origins,
        'rung_positions': c._ladder_asset.data.object_pos_w[:, c._ladder_rung_box_indexes],
        'rung_quats': c._ladder_asset.data.object_quat_w[:, c._ladder_rung_box_indexes],
    }), OUT / 'reference.pt')
    compute = base.reward_manager.compute
    captured = {}

    def record_reward(dt):
        result = compute(dt)
        p = c.contact_phases
        fields = {
            'actual': c.actual_stable_support_rung_id,
            'phase_after': p.phase, 'expected_after': p.expected,
            'release_count': p.release_count, 'safe_release_count': p.safe_release_count,
            'missing_steps': c.missing_steps, 'released': p.released, 'acquired': p.acquired,
            'task': c.task_index, 'task_steps': c.task_steps, 'episode_steps': base.episode_length_buf,
            'source': c.source_ids, 'target': c.target_ids, 'active': c.active_limbs,
            'completed': c.completed, 'failed': c.failed,
            'timeout': c.metrics['task_timeout'], 'fall': c.metrics['task_fall'],
            'collision': c.metrics['task_collision'], 'unsafe': c.metrics['task_unsafe'],
            'reward_terms': base.reward_manager._step_reward * dt, 'reward': result,
            'body_pos': c.robot_body_pos_w, 'joint_pos': c.robot_joint_pos,
            'feet_pos': c.robot.data.body_pos_w[:, feet],
            'feet_quat': c.robot.data.body_quat_w[:, feet],
            'feet_vel': c.robot.data.body_lin_vel_w[:, feet],
            'limb_pos': c.robot_limb_pos_w, 'goal_pos': c.goal_pos_w,
            'anchor_pos': c.robot_anchor_pos_w, 'anchor_quat': c.robot_anchor_quat_w,
            'anchor_lin_vel': c.robot_anchor_lin_vel_w, 'anchor_ang_vel': c.robot_anchor_ang_vel_w,
            'joint_vel': c.robot.data.joint_vel, 'torque': c.robot.data.applied_torque,
            'rung_force': torch.stack([s.data.force_matrix_w.sum(1) for s in c._stable_anchor_limb_contact_sensors], 1),
            'net_force': torch.stack([s.data.net_forces_w.sum(1) for s in c._stable_anchor_ground_contact_sensors], 1),
            'ground_filtered_force': torch.stack([s.data.force_matrix_w.sum((1, 2)) for s in c._stable_anchor_ground_contact_sensors], 1),
            'guide_frame': guide.diagnostics['frame'], 'guide_match': guide.diagnostics['matched_frame'],
            'guide_valid': guide.diagnostics['valid'], 'guide_rate': guide.diagnostics['weighted_rate'],
            'guide_body_targets': guide.body_targets_w,
            'guide_raw_rate': guide.diagnostics['raw_weighted_rate'],
            'guide_scale': guide.diagnostics['reward_scale'],
            'guide_pause_age': guide.diagnostics['pause_age_s'],
            'phase_steps': p.phase_steps,
            'unsafe_steps': c.unsafe_steps, 'danger_steps': c.danger_steps,
            'confirmation': c.confirmation, 'deadline_steps': c.task_deadline_steps,
            **{'foot_' + k: v for k, v in c.foot_reward_diagnostics.items()},
        }
        captured.clear()
        captured.update(copy_cpu(fields))
        return result

    base.reward_manager.compute = record_reward
    try:
        for label, checkpoint in [('current3500', args.checkpoint)]:
            runner.load(str(checkpoint), load_cfg=load_cfg, strict=True)
            actor = runner.alg.get_policy().to(env.device).eval()
            torch.manual_seed(42)
            obs = env.reset()
            actor.reset()
            records = []
            print('AUDIT_START', label, flush=True)
            try:
                with torch.no_grad():
                    for step in range(args.max_steps):
                        before = copy_cpu({'phase_before': c.contact_phases.phase,
                                           'expected_before': c.contact_phases.expected,
                                           'actual_before': c.actual_stable_support_rung_id,
                                           'protected_before': c.contact_phases.protected,
                                           'task_before': c.task_index,
                                           'frame_before': guide.clock.frame})
                        actions, _ = actor.forward_with_output(obs)
                        obs, rewards, dones, extras = env.step(actions)
                        records.append({**before, **captured, **copy_cpu({'actions': actions, 'dones': dones})})
                        actor.reset(dones)
                        if step % 100 == 0:
                            print('AUDIT_STEP', label, step, 'release_left', int((c.contact_phases.phase[:, 2] == 1).sum()), flush=True)
            finally:
                if records:
                    series = {k: torch.stack([r[k] for r in records]) for k in records[0]}
                    torch.save({'checkpoint': str(checkpoint), 'frames': FRAMES, **series}, OUT / f'{label}.pt')
                    print('AUDIT_SAVED', label, len(records), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    finally:
        play.simulation_app.close()
