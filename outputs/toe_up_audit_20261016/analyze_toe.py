"""Offline temporal/geometry checks of audit_toe.py traces (no simulation)."""
import csv
import json
from pathlib import Path

import torch
import trimesh

ROOT = Path(__file__).resolve().parent
PHASE = ("PREPARE", "RELEASE", "MOVE", "ACQUIRE")


def rotate(q, p):
    # Scalar first quaternion, matching IsaacLab.
    v = q[..., 1:]
    uv = torch.linalg.cross(v, p)
    return p + 2 * (q[..., :1] * uv + torch.linalg.cross(v, uv))


def inv_rotate(q, p):
    q = q.clone()
    q[..., 1:] *= -1
    return rotate(q, p)


def analyze(path):
    m = json.loads((path / "metadata.json").read_text())
    t = torch.load(path / "trace.pt", map_location="cpu", weights_only=True)
    ref = torch.load(path / "reference.pt", map_location="cpu", weights_only=True)
    n = m["reward_names"]
    dt = m["dt"]
    sizes = torch.tensor(m["rung_sizes"])
    joints = m["joint_names"]
    toe_markers = []
    for side in ("left", "right"):
        mesh = trimesh.load(Path(m["robot_asset"]).parent / "meshes" / f"{side}_ankle_x_link.STL")
        # Actual mesh vertex: lowest vertex among the frontmost 3 cm. A marker,
        # not a claim of mesh/rung signed clearance or PhysX contact position.
        vertices = mesh.vertices[mesh.vertices[:, 0] >= mesh.bounds[1, 0] - .03]
        toe_markers.append(torch.tensor(vertices[vertices[:, 2].argmin()], dtype=torch.float32))

    rows = []
    for e in range(len(m["fixtures"])):
        for step in t["first_episode"][:, e].nonzero().flatten().tolist():
            for f in (0, 1):
                limb = f + 2
                if not t["active_limbs"][step, e, limb]:
                    continue
                target = int(t["target"][step, e, limb])
                source = int(t["source"][step, e, limb])
                match = int(t["guide_matched_frame"][step, e])
                side = ("left", "right")[f]
                j = joints.index(side + "_ankle_y_joint")
                body = m["body_names"].index(side + "_ankle_x_link")
                ankle_q = t["feet_quat"][step, e, f]
                sole = t["limb_pos"][step, e, limb]
                toe = t["feet_pos"][step, e, f] + rotate(ankle_q, toe_markers[f])
                row = {
                    "env": e, "reset_frame": m["fixtures"][e], "step": step,
                    "time_s": round((step + 1) * dt, 4), "task": int(t["task"][step, e]),
                    "task_step": int(t["task_steps"][step, e]), "foot": side,
                    "source": source, "target": target, "actual": int(t["actual"][step, e, limb]),
                    "phase_before": PHASE[int(t["phase_before"][step, e, limb])],
                    "phase_after": PHASE[int(t["phase_after"][step, e, limb])],
                    "toe_deg": float(t["foot_toe_up_deg"][step, e, f]),
                    "side_deg": float(t["foot_side_tilt_deg"][step, e, f]),
                    "swing_mask": bool(t["foot_swing_mask"][step, e, f]),
                    "acquired_before": bool(t["acquired_before"][step, e, limb]),
                    "guide_frame": float(t["guide_frame"][step, e]), "guide_match": match,
                    "guide_valid": bool(t["guide_valid"][step, e]),
                    "pause_s": float(t["guide_pause_age_s"][step, e]),
                    "reference_scale": float(t["guide_reward_scale"][step, e]),
                    "reference_rate": float(t["guide_weighted_rate"][step, e]),
                    "foot_pose_rate": float(t["reward_terms"][step, e, n.index("foot_swing_pose")] / dt),
                    "joint_reference_rate": float(t["reward_terms"][step, e, n.index("reference_joint_lower")] / dt),
                    "foot_reference_rate": float(t["reward_terms"][step, e, n.index("reference_swing_foot")] / dt),
                    "danger": bool(t["reward_terms"][step, e, n.index("danger")] < 0),
                    "completed": bool(t["completed"][step, e]),
                    "ankle_y_deg": float(t["joint_pos"][step, e, j].rad2deg()),
                    "reference_ankle_y_deg": float(ref["reference_joint_pos"][match, j].rad2deg()) if match >= 0 else None,
                    "reference_toe_deg": float(torch.asin(rotate(ref["reference_body_quat"][match, body],
                        torch.tensor([1., 0., 0.]))[2].clamp(-1, 1)).rad2deg()) if match >= 0 else None,
                    "sole_goal_distance_cm": float((sole - t["goal_pos"][step, e, limb]).norm() * 100),
                    "speed_m_s": float(t["limb_speed"][step, e, limb]),
                    "front_force_n": float(t["foot_front_force_n"][step, e, f]),
                    "underside_force_n": float(t["foot_underside_force_n"][step, e, f]),
                    "sole_height_above_top_cm": None, "toe_marker_height_above_top_cm": None,
                    "sole_front_outside_cm": None, "toe_marker_front_outside_cm": None,
                    "target_force_n": None, "level_sole_height_above_top_cm": None,
                }
                if target > 0:
                    k = target - 1
                    q, p = ref["rung_quat"][e, k], ref["rung_pos"][e, k]
                    s_local, toe_local = inv_rotate(q, sole - p), inv_rotate(q, toe - p)
                    # Nominal rung local +Y points towards robot (-world X).
                    for label, value in (("sole", s_local), ("toe_marker", toe_local)):
                        row[label + "_height_above_top_cm"] = float((value[2] - sizes[k, 2] / 2) * 100)
                        row[label + "_front_outside_cm"] = float((value[1] - sizes[k, 1] / 2) * 100)
                    row["target_force_n"] = float(t["rung_force"][step, e, limb, k].norm())
                    # Nominal rungs are horizontal. Kinematic comparison only:
                    # same ankle position, level sole, offset -0.075 m from URDF.
                    row["level_sole_height_above_top_cm"] = float(100 * (
                        t["feet_pos"][step, e, f, 2] - .075 - p[2] - sizes[k, 2] / 2))
                rows.append(row)
    with (path / "foot_steps.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    upward = [r for r in rows if r["target"] > r["source"] >= 0]
    excessive = [r for r in upward if r["toe_deg"] > 15]
    acq_excess = [r for r in excessive if r["phase_before"] == "ACQUIRE" and not r["acquired_before"]]
    flights = [r for r in upward if r["actual"] < 0]
    summary = {
        "coverage": "Seven nominal deterministic first episodes, overlapping stage coverage, not IID success estimate.",
        "upward_foot_steps": len(upward), "upward_toe_excess_steps": len(excessive),
        "excess_with_reference_fade": sum(r["reference_scale"] < .999 for r in excessive),
        "excess_with_reference_scale_zero": sum(r["reference_scale"] == 0 for r in excessive),
        "excess_with_reference_invalid": sum(not r["guide_valid"] for r in excessive),
        "excess_invalid_with_danger": sum(not r["guide_valid"] and r["danger"] for r in excessive),
        "excess_without_foot_pose_mask": sum(not r["swing_mask"] for r in excessive),
        "acquire_not_acquired_excess_steps": len(acq_excess),
        "acquire_not_acquired_excess_with_pose_mask": sum(r["swing_mask"] for r in acq_excess),
        "upward_airborne_steps": len(flights),
        "upward_airborne_in_prepare": sum(r["phase_before"] == "PREPARE" for r in flights),
        "frame0_stages": [],
    }
    keys = ("step", "time_s", "task_step", "toe_deg", "side_deg", "phase_before", "phase_after",
            "actual", "swing_mask", "guide_frame", "guide_match", "guide_valid", "pause_s",
            "reference_scale", "foot_pose_rate", "joint_reference_rate", "foot_reference_rate",
            "danger", "ankle_y_deg", "reference_ankle_y_deg", "reference_toe_deg", "sole_goal_distance_cm",
            "sole_height_above_top_cm", "toe_marker_height_above_top_cm", "sole_front_outside_cm",
            "toe_marker_front_outside_cm", "level_sole_height_above_top_cm",
            "target_force_n", "front_force_n", "underside_force_n")
    for task in sorted({r["task"] for r in upward if r["env"] == 0}):
        segment = [r for r in upward if r["env"] == 0 and r["task"] == task]
        pick = lambda r: {k: r[k] for k in keys} if r else None
        summary["frame0_stages"].append({
            "task": task, "foot": segment[0]["foot"], "source": segment[0]["source"], "target": segment[0]["target"],
            "duration_s": len(segment) * dt,
            "reference_segment_duration_s": float(ref["anchors"][task + 1] - ref["anchors"][task]) * dt,
            "first_excess": pick(next((r for r in segment if r["toe_deg"] > 15), None)),
            "peak": pick(max(segment, key=lambda r: r["toe_deg"])),
            "transitions": [pick(r) for r in segment if r["phase_before"] != r["phase_after"]],
            "finish": pick(segment[-1]),
        })
    checks = {
        "all_actions_finite": bool(torch.isfinite(t["actions"]).all()),
        "reward_sum_max_error": float((t["reward_terms"].sum(-1) - t["reward"]).abs().max()),
        "acquire_excess_always_has_foot_pose_mask": all(r["swing_mask"] for r in acq_excess),
        "toe_mesh_markers_local_m": [v.tolist() for v in toe_markers],
        "geometry_note": "Sole and mesh-toe marker heights are not whole-foot clearance/contact-point measurements.",
    }
    assert checks["all_actions_finite"] and checks["reward_sum_max_error"] < 1e-5
    assert checks["acquire_excess_always_has_foot_pose_mask"]
    summary["checks"] = checks
    (path / "analysis.json").write_text(json.dumps(summary, indent=2))
    print(path.name, json.dumps({k: v for k, v in summary.items() if k != "frame0_stages"}))
    for stage in summary["frame0_stages"]:
        print("STAGE", stage["task"], json.dumps(stage["peak"]))


if __name__ == "__main__":
    for path in (ROOT / "reference1x/model_9999", ROOT / "reference2x/model_10000"):
        analyze(path)
