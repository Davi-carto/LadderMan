# Ladder

## GitHub snapshot: 2026-10-19

The repository includes LadderMan source, training/task configurations, motion
data, robot assets, experiment configurations and structured diagnostics.
The older first-stage instructions below are historical; see
[`docs/implementation_log.md`](../docs/implementation_log.md) for current tasks.

Checkpoint retention is **an upload filter, not a local deletion policy**:

- Blind teacher: `cr1_nonvisual_zdw0824_no_command_full_state_middle_stand_smooth_output/2026-09-09_13-56-37_train10000_scratch_delay0or5ms_phys5ms_seed42/model_9999.pt`, under `logs/rsl_rl/`.
- Visual runs with checkpoint/TensorBoard activity since **2026-10-04 00:00 Asia/Shanghai**: only the highest-numbered checkpoint per run, including smoke runs.
- Exact paths and SHA-256 checksums: [`training_artifacts/checkpoints.json`](../training_artifacts/checkpoints.json). This is a fixed snapshot; `.gitignore` does not implement an automatic rolling 15-day window.

All existing TensorBoard event files, JSONL episode/metric data and `.log` console
logs under `logs/` and `outputs/` are retained **losslessly**, even for old runs
whose checkpoints are excluded. They are compressed into split gzip streams
under `training_artifacts/logs_snapshot_2026-10-19/`; each part is at most 48 MiB.
The manifest records original paths, byte counts and SHA-256 checksums. No Git LFS
client is needed. From the repository root, restore the original log layout:

```bash
python scripts/training_log_archive.py verify
python scripts/training_log_archive.py restore
tensorboard --logdir logs/rsl_rl --port 6006
```

Restoration skips identical existing files and refuses to overwrite different
ones. For a separate destination, add `--root /path/to/restore-directory`; then
point TensorBoard to that directory's `logs/rsl_rl`. These commands require
Python 3.9+; TensorBoard must be installed separately. Restoring all logs needs
several GB of free disk space.

Historical run YAML/JSON/pickle configurations and saved source snapshots remain
in their original directories. Paths to excluded old checkpoints are retained
as provenance, not a promise that every historical resume command can run.
External reference checkouts (`isaaclab_v2.1.0`, `whole_body_tracking`, `holosoma`,
`holosoma-ws`), caches, credentials, personal deliverables and bulk diagnostic
image/depth/tensor dumps are not published. Install Isaac Lab and the runtime
dependencies independently; they are not vendored by this snapshot.

## Historical first-stage setup

LadderMan first-stage expert implementation.

This directory is an Isaac Lab extension rooted at `source/ladder`. It currently registers:

- `Ladder-Climb-CR1-v0` (current task)
- `Ladder-Climb-G1-v0`

Install from an IsaacLab Python 3.10 environment:

```bash
python -m pip install -e Ladder/source/ladder
```

Smooth the CR1 reference and resample it from 30 Hz to the 50 Hz control rate:

```bash
python Ladder/scripts/smooth_resample_cr1_motion.py \
  --input_file Ladder/motions/climing_ladder_cr1_downsample4_motion_annotated_wrist_d0145_v035.npz \
  --output_file Ladder/motions/climing_ladder_cr1_leg3hz_other5hz_50fps_motion.npz \
  --output_fps 50 \
  --joint_cutoff_hz 5 \
  --leg_joint_cutoff_hz 3 \
  --overwrite
```

Regenerate contact annotations on the smoothed motion:

```bash
python Ladder/scripts/preprocess_ladder_motion.py \
  --input_file Ladder/motions/climing_ladder_cr1_leg3hz_other5hz_50fps_motion.npz \
  --output_file Ladder/motions/climing_ladder_cr1_leg3hz_other5hz_50fps_annotated_wrist_d0145_v035.npz \
  --robot cr1 \
  --min_contact_frames 5 \
  --max_gap_frames 5 \
  --overwrite
```

Contact preprocessing preserves the source robot trajectory by default. Use
`--ground_initial_robot --initial_foot_target_z <z>` only when an explicit global
Z correction is required; this changes the robot-to-ladder alignment.

Run a deterministic single-environment reset/step check before training:

```bash
python Ladder/scripts/check_cr1_ladder_env.py \
  --reference_frame 0 \
  --steps 3 \
  --hold_frame \
  --headless
```

Train with the annotated motion:

```bash
./isaaclab_v2.1.0/isaaclab.sh -p Ladder/scripts/rsl_rl/train.py \
  --task Ladder-Climb-CR1-v0 \
  --motion_file Ladder/motions/climing_ladder_cr1_leg3hz_other5hz_50fps_annotated_wrist_d0145_v035.npz
```

The current milestone is a state-based CR1 expert on the fixed wood step ladder with one upward motion and fixed 50 Hz reference timing. The preprocessing script writes:

```text
contact_indicator[t, limb]
contact_rung_id[t, limb]
```

alongside the same motion-tracking keys used by the BeyondMimic-style reference loader.
