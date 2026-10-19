"""Geometric cross-check of held feet; bounds are NOT penetration estimates."""
import json
from pathlib import Path
import torch
import trimesh
import yaml
from isaaclab.utils.math import quat_apply, quat_inv

root = Path(__file__).parent
meta = json.loads((root/'metadata.json').read_text())
ref = torch.load(root/'reference.pt', weights_only=False)
d = torch.load(root/'new5500.pt', weights_only=False)
asset = Path(meta['robot_asset']).parent
mesh = trimesh.load(asset/'meshes/left_ankle_x_link.STL', force='mesh')
vertices = torch.tensor(mesh.convex_hull.vertices, dtype=torch.float32)
# The deployed ground is below the environment origin; do not assume z=0.
run_cfg = root.parents[1]/'logs/rsl_rl/cr1_reference_guided_contact_sequence_visual_amp/2026-10-12_15-05-54_reference_w3_scratch_env5120_iter10000_seed42/params/env.yaml'
ground_z = float(yaml.load(run_cfg.read_text(), Loader=yaml.BaseLoader)['scene']['ground']['init_state']['pos'][2])
results = []
for env in [0,4,8,16]:
    # Inspect the first episode after settling, before its timeout.
    mask = (d['episode_steps'][:,env]>50)&(d['episode_steps'][:,env]<450)
    mask &= (d['task'][:,env]==d['task'][0,env])&(d['phase_before'][:,env,2]==1)
    first_done = d['dones'][:,env].nonzero().flatten()
    if first_done.numel():
        mask &= torch.arange(len(mask)) < int(first_done[0])
    source = int(d['source'][0,env,2])
    pos,q = d['feet_pos'][mask,env,0],d['feet_quat'][mask,env,0]
    pts = pos[:,None]+quat_apply(q[:,None].expand(-1,len(vertices),-1),vertices[None].expand(len(pos),-1,-1))
    if source == 0:
        clearance = pts[...,2].amin(-1)-ground_z
        overlap = torch.ones_like(clearance,dtype=torch.bool)
    else:
        center=ref['rung_positions'][env,source-1];rq=ref['rung_quats'][env,source-1]
        local=quat_apply(quat_inv(rq.expand(*pts.shape[:-1],4)),pts-center)
        half=torch.tensor(meta['rung_sizes'][source-1])/2
        clearance=local[...,2].amin(-1)-half[2]
        overlap=((local[...,:2].amin(-2)<=half[:2])&(local[...,:2].amax(-2)>=-half[:2])).all(-1)
    row={'frame':meta['frames'][env],'source_rung':source,'samples':len(pos),
         'foot_lowest_point_minus_support_top_median_m':float(clearance.median()),
         'foot_lowest_point_minus_support_top_min_m':float(clearance.min()),
         'foot_lowest_point_minus_support_top_max_m':float(clearance.max()),
         'projected_aabb_overlap_fraction':float(overlap.float().mean()),
         'note':'Convex-hull foot vertices; height bounds and XY projection only, not exact signed mesh separation/penetration.'}
    results.append(row)
(root/'geometry.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
