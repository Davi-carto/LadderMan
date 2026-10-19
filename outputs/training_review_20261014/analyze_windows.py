"""Read-only training-log review; no policy or environment changes."""
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'logs/rsl_rl/cr1_pause_decay_reference_contact_sequence_visual_amp/2026-10-13_11-03-28_reference_pause_decay_scratch_env5120_iter10000_seed42'
OUT = Path(__file__).parent
values = defaultdict(dict)
timestamps = {}
duplicates = defaultdict(int)
files = sorted(RUN.glob('events.out.tfevents.*'))
for path in files:
    for event in LegacyEventFileLoader(str(path)).Load():
        for item in event.summary.value:
            if not item.HasField('simple_value') or item.tag.endswith('/time'):
                continue
            if event.step in values[item.tag]:
                duplicates[item.tag] += 1
            values[item.tag][event.step] = item.simple_value
            timestamps[event.step] = event.wall_time
latest = min(max(v) for v in values.values())
windows = [(3200, 3399), (4000, 4199), (4500, 4699), (5000, 5199),
           (5500, 5699), (6000, 6199), (6500, 6699), (6700, 6899)]
result = {'run': str(RUN), 'event_files': [str(p) for p in files], 'latest_complete_step': latest,
          'latest_time_Asia_Shanghai': datetime.fromtimestamp(timestamps[latest], ZoneInfo('Asia/Shanghai')).isoformat(),
          'duplicate_tag_counts': dict(duplicates),
          'nonfinite_values': sum(not math.isfinite(v) for series in values.values() for v in series.values()),
          'scope': '200-iteration mean of logged metrics; episode metrics are means of reset-batch means, not raw episode-weighted rates. Stage ratios mix random resets and handoffs.',
          'windows': {}}
prefix = 'Metrics/motion/'
for lo, hi in windows:
    means = {tag: statistics.mean(series[i] for i in range(lo, hi + 1) if i in series)
             for tag, series in values.items() if any(i in series for i in range(lo, hi + 1))}
    stages = {}
    for task in range(28):
        a, s, f = [means[prefix + f'task_{task:02d}_{suffix}'] for suffix in ('attempts', 'successes', 'failures')]
        stages[str(task)] = {'attempts_mean': a, 'successes_mean': s, 'failures_mean': f,
                             'success_to_attempt_ratio': s/a if a else None,
                             'unaccounted': a-s-f}
    result['windows'][f'{lo}-{hi}'] = {'means': means, 'stages': stages}
    fields = ['task_completed_count', 'sequence_success', 'task_fall', 'task_timeout', 'task_collision',
              'sequence_start_task_index', 'reset_frame', 'phase_release_time_s', 'phase_acquire_time_s',
              'phase_release_events', 'phase_contact_events', 'guidance_pause_time_s']
    print(lo, hi, json.dumps({k: means[prefix+k] for k in fields}), flush=True)
    print('TRAIN', {k: means[k] for k in ('Train/mean_reward', 'Train/mean_episode_length',
                                         'Perf/collection_time', 'Perf/learning_time')}, flush=True)
    print('STAGES', {k: round(stages[str(k)]['success_to_attempt_ratio'], 4)
                      for k in (2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23, 25)}, flush=True)
(OUT / 'summary.json').write_text(json.dumps(result, indent=2))
print('LATEST', latest, result['latest_time_Asia_Shanghai'], 'NONFINITE', result['nonfinite_values'], flush=True)
