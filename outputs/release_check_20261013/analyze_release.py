"""Summarize paired fixed-frame physics traces; no training or state mutation."""
import csv
import json
import re
from pathlib import Path

import torch
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, quat_error_magnitude, yaw_quat
from ladder.utils.reference_guidance import reference_reward_candidates, select_reference_frame

ROOT = Path(__file__).parent
meta = json.loads((ROOT / 'metadata.json').read_text())
ref = torch.load(ROOT / 'reference.pt', weights_only=False)
names = meta['body_names']
joints = meta['robot_joint_names']
lower_bodies = [names.index(n) for n in ('base_link', 'left_hip_x_link', 'left_knee_link',
    'left_ankle_x_link', 'right_hip_x_link', 'right_knee_link', 'right_ankle_x_link')]
upper_bodies = [names.index(n) for n in ('body', 'left_shoulder_x_link', 'left_elbow_link',
    'left_wrist_x_link', 'left_sphere_hand_link', 'right_shoulder_x_link', 'right_elbow_link',
    'right_wrist_x_link', 'right_sphere_hand_link')]
lower_joints = [i for i,n in enumerate(joints) if re.fullmatch(r'.*_(?:hip_.*|knee|ankle_.*)_joint',n)]
upper_joints = [i for i,n in enumerate(joints) if re.fullmatch(r'(?:waist_.*|neck_.*|.*_shoulder_.*|.*_elbow|.*_wrist_.*)_joint',n)]
feet = [names.index(n) for n in ('left_ankle_x_link', 'right_ankle_x_link')]
anchor = names.index('body')
weights = torch.tensor([.15,.15,.25,.15,.30,.10])
ref_indexes = [i for i,n in enumerate(meta['reward_names']) if n.startswith('reference_')]
dt = meta['dt']
rows, checks, frozen = [], {}, []

for task in (2,4,6,10,17):
    start, end = ref['anchors'][task:task+2].tolist()
    limb = (ref['support_ids'][start] != ref['support_ids'][end]).nonzero()[0,0].item()
    release = start + (ref['support_ids'][start:end+1,limb] == -1).nonzero()[0,0].item()
    cap = ref['guide_upper'][task,limb,1].item()
    frames = torch.arange(cap-3,cap+1).unsqueeze(0)
    static_frame = release-1
    target_pos = ref['reference_body_pos'][frames]
    target_q = ref['reference_body_quat'][frames,anchor]
    current_pos = ref['reference_body_pos'][static_frame].unsqueeze(0)
    current_q = ref['reference_body_quat'][static_frame,anchor].expand_as(target_q)
    delta_q = yaw_quat(quat_mul(current_q,quat_inv(target_q)))
    delta_p = current_pos[:,None,anchor].expand_as(target_pos[:,:,anchor]).clone()
    delta_p[...,2] = target_pos[:,:,anchor,2]
    aligned = delta_p[:,:,None]+quat_apply(delta_q[:,:,None].expand(*target_pos.shape[:-1],4),target_pos-target_pos[:,:,anchor,None])
    body_sq = (aligned-current_pos[:,None]).square().sum(-1)
    joint_sq = (ref['reference_joint_pos'][frames]-ref['reference_joint_pos'][static_frame]).square()
    angle_sq = quat_error_magnitude(target_q,current_q).square()
    swing = torch.zeros(1,2,dtype=torch.bool); swing[0,limb-2] = True
    rewards,errors = reference_reward_candidates(body_sq,joint_sq,angle_sq,lower_bodies,upper_bodies,
                                                  lower_joints,upper_joints,feet,swing)
    match,values,err,valid = select_reference_frame(frames,torch.ones_like(frames,dtype=torch.bool),rewards,errors,weights)
    body = feet[limb-2]
    frozen.append({'task':task,'release_frame':release,'clock_cap':cap,'frozen_frame':static_frame,
                   'best_match':match.item(),'frozen_reference_rate':(values*weights).sum().item(),
                   'ankle_distance_to_best_m':err[0,4].item(),
                   'ankle_vertical_change_at_cap_m':(ref['reference_body_pos'][cap,body,2]-ref['reference_body_pos'][static_frame,body,2]).item(),
                   'note':'kinematic reference-pose counterfactual, NOT a physically validated static hold'})

for path in sorted(ROOT.glob('*5500*.pt'))+sorted(ROOT.glob('*9999*.pt')):
    d = torch.load(path, weights_only=False)
    reward_residual = (d['reward_terms'].sum(-1)-d['reward']).abs().max().item()
    sensor_max = d['rung_force'].norm(dim=-1).amax(-1)
    # Four consecutive airborne samples in running RELEASE must not remain there.
    inconsistent = ((d['phase_before']==1)&(d['phase_after']==1)&(d['actual']<0)&(d['release_count']>=4)&~d['failed'][...,None]).sum().item()
    checks[path.stem] = {'reward_sum_max_error':reward_residual,'confirmed_release_still_RELEASE_samples':inconsistent,
                         'all_reward_finite':bool(torch.isfinite(d['reward_terms']).all())}
    for env,frame in enumerate(d['frames']):
        task = int(d['task'][0,env]); active = d['active'][0,env].nonzero().flatten();limb = int(active[0])
        endmask = (d['dones'][:,env]|d['completed'][:,env]|(d['task'][:,env]!=task)).nonzero().flatten()
        end = int(endmask[0])+1 if endmask.numel() else d['task'].shape[0]
        valid = torch.arange(d['task'].shape[0]) < end
        settled = valid & (d['episode_steps'][:,env]>10) & ~d['failed'][:,env]
        release_mask = settled & (d['phase_before'][:,env,limb]==1)
        stalled = release_mask & (d['actual'][:,env,limb]==d['source'][:,env,limb])
        terms = d['reward_terms'][:end,env]; ref_sum = terms[:,ref_indexes].sum().item()
        total = d['reward'][:end,env].sum().item()
        def mean(v,mask):
            return v[mask].float().mean().item() if mask.any() else None
        force = torch.where(d['source'][:,env,limb]==0,d['net_force'][:,env,limb].norm(dim=-1),sensor_max[:,env,limb])
        dist = (d['limb_pos'][:,env,limb]-d['goal_pos'][:,env,limb]).norm(dim=-1)
        released = (settled & (d['actual'][:,env,limb]<0)).nonzero().flatten()
        shift = (d['frame_before'][1:,env]-d['frame_before'][:-1,env])
        paused = torch.cat((torch.zeros(1,dtype=torch.bool),shift==0))&settled
        row={'policy':path.stem,'frame':frame,'task':task,'limb':limb,'duration_s':end*dt,
             'initial_phase':int(d['phase_before'][0,env,limb]),'completed':bool(d['completed'][:end,env].any()),
             'timeout':bool(d['timeout'][:end,env].any()),'fall':bool(d['fall'][:end,env].any()),
             'collision':bool(d['collision'][:end,env].any()),'release_time_s':release_mask.sum().item()*dt,
             'stalled_source_contact_time_s':stalled.sum().item()*dt,'first_no_contact_s':float(released[0])*dt if released.numel() else None,
             'stalled_contact_force_n':mean(force,stalled),'stalled_reference_rate':mean(d['guide_rate'][:,env],stalled),
             'stalled_total_reward_rate':mean(d['reward'][:,env]/dt,stalled),
             'reference_pause_time_s':paused.sum().item()*dt,'reference_integral':ref_sum,
             'total_reward':total,'without_reference_reward':total-ref_sum,
             'discounted_return':float((d['reward'][:end,env]*.99**torch.arange(end)).sum()),
             'initial_goal_distance_m':float(dist[0]),'final_goal_distance_m':float(dist[end-1]),
             'reward_breakdown':dict(zip(meta['reward_names'],terms.sum(0).tolist()))}
        rows.append(row)

summary={'measurement':'20 distinct fixed-frame deterministic nominal fixtures per policy, not independent success-rate estimates',
         'old_policy_environment':'same new reference task, same 10s deadlines, same reward accounting; not the old training environment',
         'checks':checks,'frozen_reference_pose_check':frozen,'fixtures':rows}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
with (ROOT/'fixtures.csv').open('w',newline='') as f:
    keys=[k for k in rows[0] if k!='reward_breakdown']
    writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in rows)
print('CHECKS',json.dumps(checks))
print('FROZEN_REFERENCE',json.dumps(frozen))
for r in rows:
    print(r['policy'],r['frame'],'task',r['task'],'initial_phase',r['initial_phase'],
          'complete',r['completed'],'timeout',r['timeout'],'fall',r['fall'],'duration',r['duration_s'],
          'release_s',round(r['release_time_s'],2),'force',r['stalled_contact_force_n'],
          'ref_rate',r['stalled_reference_rate'],'net_rate',r['stalled_total_reward_rate'])
