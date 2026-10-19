# CR1 Reference Motion Jump-Repair Workflow

This workflow addresses localized high-frequency jumps without changing the robot,
ladder, contact geometry, or kinematic path more than necessary. It is intended for
reuse when a future retarget source has bad intervals at different frame numbers.

## Tool Responsibilities

- `Ladder/scripts/replay_ladder_reference.py`: visual inspection and frame-by-frame
  playback. It never edits motion data.
- `Ladder/scripts/analyze_cr1_reference_dynamics.py`: read-only reporting of joint
  and key-body velocity, acceleration, and jerk peaks.
- `Ladder/scripts/repair_cr1_reference_jumps.py`: local time reparameterization only.
  It stretches selected source intervals at the existing output FPS, tapers the
  stretch at interval boundaries, performs PCHIP/SLERP interpolation, and rebuilds
  body kinematics with URDF FK.
- `Ladder/scripts/retime_cr1_reference_by_arc_length.py`: coordinated spatial-progress
  retiming for a dwell-plus-burst interval. It redistributes samples using the
  selected sphere hand, base, both hands, both sole points, and a small joint-path
  contribution. It also treats release interpolation as swing and delays contact
  onset until the contact endpoint.
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`: collision/penetration screening
  after a trajectory change. It owns reset-valid masks, not motion smoothing.
- `RateLimitedJointPositionAction`: applies only the configured joint-target rate
  limit and optional control delay to policy-generated targets. It does not smooth
  or repair the reference file.

## Required Sequence

1. Replay suspicious ranges at 10 FPS and use pause/single-step controls.
2. Run the dynamics audit on those ranges and record the actual joint/body peaks.
3. Repair only confirmed bad ranges. Use uniform time stretch only when the source
   displacement is already reasonably distributed. Use coordinated arc-length
   retiming when a long dwell is followed by a short burst. Keep acceptable ranges
   out of the segment list.
4. Replay the mapped output ranges before accepting the candidate.
5. Regenerate reset-validity fields because inserted FK samples can change signed
   penetration even though the source path is nominally unchanged.
6. Start a new training run when action processing/rate limiting changes. Old policy
   checkpoints were trained against different action dynamics.

Do not re-run distance-threshold contact annotation after a pure time warp. Slower
swing motion can satisfy a velocity threshold earlier and create false contact
phases. The repair tool therefore time-warps the existing categorical
`contact_indicator` and `contact_rung_id` values with their source frames.

For inserted samples at a contact boundary, arc-length retiming uses a conservative
rule: interpolating from contact to swing is swing, while interpolating from swing
to contact remains swing until the endpoint. This avoids rewarding a moving hand as
if it were still supporting the rung.

## Arc-Length Preview Recipe

Use this when added time-stretch frames remain concentrated around the original
burst. The current preview is intentionally not connected to a training task until
visual inspection and collision screening pass.

```bash
PYTHONPATH=Ladder/source/ladder \
/home/zhangdawei/miniforge3/envs/retarget/bin/python \
Ladder/scripts/retime_cr1_reference_by_arc_length.py \
  --input_file INPUT.npz \
  --output_file PREVIEW.npz \
  --segment 440:500:115:right_sphere_hand_link \
  --segment 775:835:170:right_sphere_hand_link \
  --segment 956:984:72:left_sphere_hand_link \
  --joint_weight_m_per_rad 0.05 \
  --coordination_body_weight 1.0
```

This produces 1386 frames at 50 Hz. The mapped preview ranges are `440:555`,
`830:1000`, and `1121:1193`. The source-time lineage is stored in
`arc_retime_source_time`; use it to map any other source frame.

## Current Repair Recipe

Input:

```text
Ladder/motions/restored_prefix_v300_stride2_slow50/smooth_start/ladder_geometry_dr/
climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_train_ready_ladder_dr.npz
```

Command:

```bash
PYTHONPATH=Ladder/source/ladder \
/home/zhangdawei/miniforge3/envs/retarget/bin/python \
Ladder/scripts/repair_cr1_reference_jumps.py \
  --input_file INPUT.npz \
  --output_file OUTPUT.npz \
  --segment 440:500:2 \
  --segment 775:835:3 \
  --segment 968:983:2 \
  --transition_frames 5
```

The result remains 50 Hz and changes from 1177 to 1352 frames. Source-to-output
inspection ranges are:

```text
source 0-60    -> output 0-60
source 440-500 -> output 440-555
source 630-680 -> output 685-735 (unchanged trajectory/timing inside the range)
source 775-835 -> output 830-1000
source 850-880 -> output 1015-1045 (unchanged trajectory/timing inside the range)
source 968-983 -> output 1133-1158
```

The output stores `reference_repair_source_time`, source frame/fraction lineage,
segment specifications, transition width, method, and robot asset. Use this lineage
instead of manually guessing frame offsets after a future repair.

## Action Envelope

The separate `Ladder-Climb-CR1-Reference-RateLimited-v0` task applies:

```text
max target rate: 6.0 rad/s = 0.12 rad per 50 Hz policy step
first-order filtering: disabled
post-reset ramp: disabled
episode limit: 30.0 s (the repaired reference lasts 27.02 s)
```

The 6 rad/s value is a conservative training envelope, about 25-31% of the current
URDF velocity limits (19.38-23.76 rad/s). It is not yet a measured hardware
controller limit. Replace it with joint-specific measured command limits before
claiming exact sim-to-real equivalence. This layer limits target rate only; it is
not a smoothing, acceleration, torque, or thermal safety controller.
