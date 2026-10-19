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

## Current environment snapshot and server migration

The current `rl-training` inventory is in
[`environments/rl-training/2026-10-19/`](../environments/rl-training/2026-10-19/).
It records Ubuntu 22.04.5 x86_64, glibc 2.35, Python 3.11.15, Isaac Lab 2.3.2,
Isaac Sim 5.1.0.0, PyTorch 2.7.0+cu128, RSL-RL 5.0.1 and driver 570.172.18.
The checked-out `isaaclab_v2.1.0` reference directory is **not** the active runtime.
These are observed versions, not minimum requirements for arbitrary hardware.

- `environment.yml`: 26 pinned Conda packages and 290 non-editable pip dependencies, with the NVIDIA/PyTorch indexes. No machine-specific prefix or environment-variable values.
- `conda-explicit-linux-64.txt`: exact Conda artifacts with SHA-256 hashes; this file alone does not install pip packages.
- `requirements-pip.txt`: pip-installed dependencies; the SMPL-X Git dependency is pinned to its source commit.
- `pip-freeze.txt`: all non-editable Python distribution versions, including Conda-managed pip; an inventory rather than a separate installer.
- `editable-installs.json`: eight links to source outside the environment. Four are Isaac Lab addons; four belong to other projects and are not imported by the LadderMan source/scripts scan.
- `runtime.json` and `audit.json`: host/runtime metadata, source integrity checks and known migration caveats.

Only manifests and deployment instructions are published, **not an environment
bundle**. Recreate the environment on a compatible Linux x86_64 GPU server.
The OS libraries and NVIDIA driver remain host requirements; Conda/pip do not
install the host driver. Start from the recorded Ubuntu 22.04 baseline and verify
that the target GPU/driver supports Isaac Sim 5.1 and the CUDA 12.8 PyTorch build.
Downloads require access to conda-forge, PyPI, NVIDIA, PyTorch and GitHub.

The four active Isaac Lab addons are editable installs from an external checkout
of official tag `v2.3.2`, commit `37ddf626871758333d6ed89cf64ad702aef127d0`.
Install these addons from that revision at their new location, and keep that
checkout available: editable installs are links to source, not copies. Leave
the working training environment unchanged.

From the cloned LadderMan repository root, reconstruct in a **new** environment.
Run each step only after the preceding step succeeds; use an unused environment
name and checkout destination:

```bash
conda create -n ladderman-reproduced --file environments/rl-training/2026-10-19/conda-explicit-linux-64.txt
conda activate ladderman-reproduced
python -m pip install -r environments/rl-training/2026-10-19/requirements-pip.txt
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab_v2.3.2
git -C ../IsaacLab_v2.3.2 rev-parse HEAD
# Must print: 37ddf626871758333d6ed89cf64ad702aef127d0
python -m pip install --no-deps --no-build-isolation -e ../IsaacLab_v2.3.2/source/isaaclab_assets -e ../IsaacLab_v2.3.2/source/isaaclab_mimic -e ../IsaacLab_v2.3.2/source/isaaclab_rl -e ../IsaacLab_v2.3.2/source/isaaclab_tasks
python -m pip install --no-deps --no-build-isolation -e Ladder/source/ladder
# Match the observed cv2 file owner; run LAST in the new environment only.
python -m pip install --no-deps --force-reinstall opencv-python==4.11.0.86
python -m pip check
```

The `--no-deps` addon install preserves the recorded RSL-RL 5.0.1 setup; installing
Isaac Lab's `rsl-rl` extra instead would request RSL-RL 3.1.2.
The four unrelated editable projects are intentionally not installed by these
instructions. Full fresh-server reconstruction has not been executed; validate
imports, CUDA, simulator startup and a small reset/step run before long training.

Known caveat: both `opencv-python` and `opencv-python-headless` 4.11.0.86 are
installed in the observed environment. They share `cv2` paths; the checked current
files, including `cv2.abi3.so`, match the GUI distribution. The inventory preserves
both names rather than silently changing the environment. Isaac Sim core requires
the headless distribution, but the observed shared files belong to GUI OpenCV;
the final reinstall above deliberately reproduces that ownership. This is a
compatibility workaround, not a recommendation to mix OpenCV distributions.
Do not uninstall either one casually: it can remove the other's shared files.
`pip check` does not detect this conflict; validate `import cv2` and any GUI tools
on the destination server (GUI playback also requires a working display).

Before training, check the recorded key versions and execute a small CUDA operation:

```bash
nvidia-smi
python - <<'PY'
import json
from importlib.metadata import version
from pathlib import Path
import cv2
import torch

snapshot = json.loads(Path("environments/rl-training/2026-10-19/runtime.json").read_text())
for name, expected in {**snapshot["key_versions"], "isaacsim": snapshot["isaacsim"]}.items():
    actual = version(name)
    assert actual == expected, (name, actual, expected)
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.version.cuda == snapshot["pytorch_cuda_build"]
print("CUDA:", torch.ones(4, device="cuda").sum().item())
print("OpenCV:", cv2.__version__)
print("Recorded key versions match")
PY
python Ladder/scripts/check_cr1_ladder_env.py --num_envs 1 --reference_frame 0 --steps 3 --hold_frame --headless
```

The last command checks the baseline CR1 task's simulator startup/reset/step, not
full visual-AMP training. Its default motion is included in the repository.
Isaac Sim may request license acceptance and download additional assets at first
startup. After this check, test the intended task with a few environments before
launching a long run. These destination-server checks have **not** been executed
as a fresh-server reconstruction for this snapshot.

Source audit: 47,912 installed `.py`/`.pyi`/`.pth` RECORD entries were checked.
Only two entries from the headless OpenCV distribution mismatched; both match
the GUI distribution's original files. Core Isaac Lab, RSL-RL, PyTorch, NumPy,
Gymnasium, Warp and TensorDict sources matched their installation records.
The four editable Isaac Lab addon directories were clean against the verified
official release commit. This is a current source/import-hook audit, not proof
that every binary or historical environment state is unchanged. Project-specific
RSL-RL adaptation remains in `Ladder/scripts/rsl_rl/rsl_rl_compat.py` and
`Ladder/source/ladder/ladder/rsl_rl/`.

References: [Conda environment specifications](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html),
[OpenCV package selection](https://github.com/opencv/opencv-python#installation-and-usage),
[Python installed-file records](https://packaging.python.org/en/latest/specifications/recording-installed-packages/).

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
