"""Reuse the toe audit; assert real reset/handoff counter ownership in IsaacLab."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs/toe_up_audit_20261016"))
import audit_toe as audit
import torch

checks = {"physical_resamples": 0, "partial_resamples": 0, "handoff_envs": 0,
          "reset_settling_samples": 0, "live_goal_samples_before_step10": 0}
Command = audit.ContactTaskCommand
original_resample = Command._resample_command
original_goals = Command._set_task_goals
original_refresh = Command.refresh


def checked_resample(self, ids):
    before = self._reset_steps.clone()
    original_resample(self, ids)
    assert self._reset_steps[ids].eq(0).all()
    other = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
    other[ids] = False
    torch.testing.assert_close(self._reset_steps[other], before[other])
    checks["physical_resamples"] += 1
    checks["partial_resamples"] += int(bool(other.any()))


def checked_goals(self, ids):
    before = self._reset_steps.clone()
    original_goals(self, ids)
    torch.testing.assert_close(self._reset_steps, before)
    checks["handoff_envs"] += int((before[ids] > 0).sum())


def checked_refresh(self):
    before = self._reset_steps.clone()
    fresh = self._last_refresh_step != self._env.common_step_counter
    running = self._env.episode_length_buf > 0
    original_refresh(self)
    torch.testing.assert_close(self._reset_steps, before + (running.long() if fresh else 0))
    if fresh:
        settled = self._reset_steps >= self.cfg.task_settle_steps
        checks["reset_settling_samples"] += int((running & ~settled).sum())
        checks["live_goal_samples_before_step10"] += int((running & settled & (self.task_steps < 10)).sum())


Command._resample_command = checked_resample
Command._set_task_goals = checked_goals
Command.refresh = checked_refresh
try:
    audit.main()
    assert all(value > 0 for value in checks.values())
    checks["command_sha256"] = hashlib.sha256((ROOT / "Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/contact_task.py").read_bytes()).hexdigest()
    checks["passed"] = True
    out = audit.play.args_cli.output_dir.resolve() / Path(audit.play.args_cli.checkpoint).stem
    (out / "reset_handoff_checks.json").write_text(json.dumps(checks, indent=2))
    print("HANDOFF_SMOKE", json.dumps(checks), flush=True)
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    audit.play.simulation_app.close()
