"""Matched-window review of 1x versus 2x reference continuation; no training changes."""
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {
    "baseline": ROOT / "logs/rsl_rl/cr1_pause_decay_reference_contact_sequence_visual_amp/2026-10-13_11-03-28_reference_pause_decay_scratch_env5120_iter10000_seed42",
    "reference2x": ROOT / "logs/rsl_rl/cr1_pause_decay_reference2x_contact_sequence_visual_amp/2026-10-15_13-26-46_resume6000_reference2x_env5120_add4000_seed42",
}
P = "Metrics/motion/"


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [item for key in sorted(set(a) | set(b))
                for item in differences(a.get(key), b.get(key), f"{path}/{key}")]
    return [] if a == b else [{"path": path, "old": a, "new": b}]


def read_run(run):
    diagnostic = []
    for line in (run / "end_to_end_visual_ppo_diagnostics/metrics.jsonl").read_text().splitlines():
        if line.strip():
            try:
                diagnostic.append(json.loads(line))
            except json.JSONDecodeError:
                break  # live append may leave a final partial record
    cutoff = max(r["iteration"] for r in diagnostic)
    values, times = defaultdict(dict), {}
    duplicates, conflicts = defaultdict(int), defaultdict(int)
    files = sorted(run.glob("events.out.tfevents.*"))
    for path in files:
        for event in LegacyEventFileLoader(str(path)).Load():
            if event.step > cutoff:
                continue
            for item in event.summary.value:
                if not item.HasField("simple_value") or item.tag.endswith("/time"):
                    continue
                if event.step in values[item.tag]:
                    duplicates[item.tag] += 1
                    conflicts[item.tag] += values[item.tag][event.step] != item.simple_value
                values[item.tag][event.step] = item.simple_value
                times[event.step] = max(times.get(event.step, 0), event.wall_time)
    last = min(cutoff, max(values[P + "sequence_success"]))
    meta = {"run": str(run), "event_files": [str(p) for p in files], "latest_iteration": last,
            "latest_time_Asia_Shanghai": datetime.fromtimestamp(times[last], ZoneInfo("Asia/Shanghai")).isoformat(),
            "nonfinite_scalars": sum(not math.isfinite(v) for s in values.values() for v in s.values()),
            "duplicates": dict(duplicates), "conflicting_duplicates": {k: v for k, v in conflicts.items() if v},
            "diagnostic_records": len(diagnostic),
            "diagnostic_unique_iterations": len({r["iteration"] for r in diagnostic})}
    return values, diagnostic, meta


data = {name: read_run(run) for name, run in RUNS.items()}
latest = min(v[2]["latest_iteration"] for v in data.values())
windows = [(6001, 6200), (6500, 6699), (7000, 7199), (7500, 7699),
           (8000, 8199), (8500, 8699), (9000, 9199), (9400, 9599),
           (9600, 9799), (latest - 199, latest)]
result = {
    "scope": "200-iteration averages of logged reset-batch means, not raw episode-weighted success rates. "
             "Random resets and handoffs are mixed. Reference errors are terminal/reset metric samples, "
             "not episode-integrated errors, and invalid reference matches contribute zero. "
             "Both runs share model6000 ancestry, but resume resets simulator/RNG/replay state; one seed is not a causal replicate.",
    "matched_latest": latest, "runs": {}, "configuration_differences": {},
}
for config in ("env.yaml", "agent.yaml", "end_to_end_amp_run_manifest.yaml"):
    # BaseLoader safely compares structure without instantiating Python YAML tags.
    a, b = [yaml.load((run / "params" / config).read_text(), Loader=yaml.BaseLoader) for run in RUNS.values()]
    result["configuration_differences"][config] = differences(a, b)

for name, (values, diagnostic, meta) in data.items():
    sections = {}
    for lo, hi in windows:
        if hi > meta["latest_iteration"]:
            continue
        means, counts = {}, {}
        for tag, series in values.items():
            selected = [v for i, v in series.items() if lo <= i <= hi]
            if selected:
                means[tag], counts[tag] = statistics.mean(selected), len(selected)
        records = [r for r in diagnostic if lo <= r["iteration"] <= hi]
        dmeans = {k: statistics.mean(r[k] for r in records) for k in records[0] if k != "iteration"}
        stages = {}
        for task in range(28):
            a, s, f = [means[P + f"task_{task:02d}_{suffix}"] for suffix in ("attempts", "successes", "failures")]
            stages[str(task)] = {"attempts": a, "successes": s, "failures": f,
                                 "success_ratio": s / a if a else None, "residual": a - s - f}
        foot = {}
        for side in ("left", "right"):
            prefix = P + "foot_" + side + "_"
            swing = means[prefix + "swing_time_s"]
            foot[side] = {
                "swing_time_s": swing,
                "toe_up_over15_fraction": means[prefix + "swing_toe_up_excess_time_s"] / swing if swing else None,
                "mean_episode_peak_toe_up_deg": means[prefix + "swing_toe_up_max_deg"],
                "mean_episode_peak_side_tilt_deg": means[prefix + "swing_side_tilt_max_deg"],
                "front_contact_time_s": means[prefix + "front_contact_time_s"],
                "underside_contact_time_s": means[prefix + "underside_contact_time_s"],
            }
        references = {tag.removeprefix("Episode_Reward/"): mean for tag, mean in means.items()
                      if tag.startswith("Episode_Reward/reference_")}
        sections[f"{lo}-{hi}"] = {"means": means, "counts": counts, "diagnostics": dmeans,
            "diagnostic_count": len(records), "stages": stages, "feet": foot,
            "reference_paid_log_sum": sum(references.values()),
            "reference_paid_log_sum_at_1x_weights": sum(references.values()) / (2 if name == "reference2x" else 1),
            "total_stage_successes": sum(s["successes"] for s in stages.values())}
        print(name, lo, hi, {k: round(means[P+k], 5) for k in (
            "sequence_success", "task_completed_count", "task_fall", "task_timeout", "task_collision",
            "reset_frame", "guidance_valid", "phase_release_time_s", "phase_acquire_time_s")}, flush=True)
    result["runs"][name] = {"metadata": meta, "windows": sections}

result["current_source_matches"] = {}
manifest = yaml.load((RUNS["reference2x"] / "params/end_to_end_amp_run_manifest.yaml").read_text(), Loader=yaml.BaseLoader)
for path, digest in manifest["source_sha256"].items():
    result["current_source_matches"][path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
(OUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
print("METADATA", {name: values["metadata"] for name, values in result["runs"].items()}, flush=True)
print("ENV_DIFF", result["configuration_differences"]["env.yaml"], flush=True)
print("LATEST", latest, flush=True)
