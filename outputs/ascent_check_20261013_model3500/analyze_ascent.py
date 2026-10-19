"""First-goal, pre-reset diagnostics. Fixed fixtures are not a success-rate sample."""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
meta = json.loads((ROOT / 'metadata.json').read_text())
d = torch.load(ROOT / 'current3500.pt', map_location='cpu', weights_only=False)
ref = torch.load(ROOT / 'reference.pt', map_location='cpu', weights_only=False)
motion = np.load('/home/zhangdawei/LadderMan/Ladder/motions/zdw0824_capture1/pitchminus3_yawminus4_xminus030/37_visual_up_down_stop_command/zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_reset5mm_visual_up_down_stop_50hz_motion.npz')
dt = meta['dt']
phase_names = ['PREPARE', 'RELEASE', 'MOVE', 'ACQUIRE']
rows = []

def first_time(mask):
    indexes = mask.nonzero().flatten()
    return round((int(indexes[0]) + 1) * dt, 3) if indexes.numel() else None

for env, frame in enumerate(d['frames']):
    task = int(d['task'][0, env])
    limb = int(d['active'][0, env].nonzero()[0, 0])
    foot = limb - 2
    ends = (d['completed'][:, env] | d['dones'][:, env] | (d['task'][:, env] != task)).nonzero().flatten()
    end = int(ends[0]) + 1 if ends.numel() else len(d['task'])
    series = lambda key: d[key][:end, env]
    phase = series('phase_after')[:, limb]
    actual = series('actual')[:, limb]
    settled = series('episode_steps') > 10
    source = int(series('source')[0, limb])
    target = int(series('target')[0, limb])
    force = (series('net_force')[:, limb].norm(dim=-1) if source == 0 else
             series('rung_force')[:, limb, source - 1].norm(dim=-1))
    stalled = settled & (phase == 1) & (actual == source) & ~series('failed')
    foot_pos = series('feet_pos')[:, foot]
    foot_speed = series('feet_vel')[:, foot].norm(dim=-1)
    distance = (series('limb_pos')[:, limb] - series('goal_pos')[:, limb]).norm(dim=-1)
    q = series('anchor_quat')
    upright = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
    swing = series('foot_swing_mask')[:, foot] & settled
    transitions = [0] + (phase[1:] != phase[:-1]).nonzero().flatten().add(1).tolist()
    transitions.append(end)
    intervals = [{'phase': phase_names[int(phase[a])], 'start_s': round(a * dt, 3),
                  'end_s': round(b * dt, 3)} for a, b in zip(transitions[:-1], transitions[1:])]
    tail = slice(max(10, end - 50), end)
    distances = (series('limb_pos') - series('goal_pos')).norm(dim=-1)
    row = {
        'frame': frame, 'task': task, 'limb': ['LH','RH','LF','RF'][limb],
        'source': source, 'target': target, 'kind': 'anchor' if env < 6 else 'mid_swing',
        'reset_valid_annotation': bool(motion['reset_valid_mask'][frame]),
        'reset_max_penetration_m': float(motion['reset_max_penetration_depth'][frame]),
        'duration_s': round(end * dt, 3),
        'completed': bool(series('completed').any()), 'timeout': bool(series('timeout').any()),
        'fall': bool(series('fall').any()), 'collision': bool(series('collision').any()),
        'phase_intervals': intervals,
        'first_no_contact_after_settle_s': first_time(settled & (actual < 0)),
        'first_MOVE_s': first_time(phase == 2), 'first_ACQUIRE_s': first_time(phase == 3),
        'first_target_contact_after_settle_s': first_time(settled & (actual == target)),
        'source_contact_RELEASE_s': round(stalled.sum().item() * dt, 3),
        'source_force_during_stall_mean_N': float(force[stalled].mean()) if stalled.any() else None,
        'foot_lift_from_first_sample_max_m': float(foot_pos[:, 2].max() - foot_pos[0, 2]),
        'foot_displacement_max_m': float((foot_pos - foot_pos[0]).norm(dim=-1).max()),
        'foot_tail_speed_mean_m_s': float(foot_speed[tail].mean()),
        'foot_tail_xyz_range_m': (foot_pos[tail].max(0).values - foot_pos[tail].min(0).values).tolist(),
        'goal_distance_initial_m': float(distance[0]), 'goal_distance_final_m': float(distance[-1]),
        'goal_distance_min_m': float(distance.min()),
        'all_limb_goal_distance_tail_mean_m': distances[tail].mean(0).tolist(),
        'all_limb_goal_distance_tail_min_m': distances[tail].min(0).values.tolist(),
        'all_limb_goal_distance_tail_max_m': distances[tail].max(0).values.tolist(),
        'completion_confirmation_max': int(series('confirmation').max()),
        'front_peak_N': float(series('foot_front_force_n')[settled, foot].max()),
        'underside_peak_N': float(series('foot_underside_force_n')[settled, foot].max()),
        'first_front_over5N_s': first_time(settled & (series('foot_front_force_n')[:, foot] > 5)),
        'first_underside_over5N_s': first_time(settled & (series('foot_underside_force_n')[:, foot] > 5)),
        'toe_up_swing_max_deg': float(series('foot_toe_up_deg')[swing, foot].max()) if swing.any() else None,
        'min_upright': float(upright.min()),
        'first_fall_confirmation_active_s': first_time(series('unsafe_steps') > 0),
        'ref_clock_final': float(series('guide_frame')[-1]),
        'ref_paid_rate_tail_mean': float(series('guide_rate')[tail].mean()),
        'ref_raw_rate_tail_mean': float(series('guide_raw_rate')[tail].mean()),
        'final_actual_all_limbs': series('actual')[-1].tolist(),
        'final_phase_all_limbs': series('phase_after')[-1].tolist(),
    }
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False))

checks = {
    'steps': len(d['task']), 'envs': len(d['frames']),
    'finite_actions': bool(torch.isfinite(d['actions']).all()),
    'finite_rewards': bool(torch.isfinite(d['reward_terms']).all()),
    'reward_sum_error_max': float((d['reward_terms'].sum(-1) - d['reward']).abs().max()),
    'confirmed_airborne_still_RELEASE_samples': int(((d['phase_before'] == 1) &
        (d['phase_after'] == 1) & (d['actual'] < 0) & (d['release_count'] >= 4) &
        ~d['failed'][..., None]).sum()),
}
print('CHECKS', checks)
(ROOT / 'summary.json').write_text(json.dumps({'metadata': meta, 'checks': checks, 'fixtures': rows}, indent=2))
