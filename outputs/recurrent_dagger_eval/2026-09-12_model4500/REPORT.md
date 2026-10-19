# PPO model4500 deterministic evaluation and LayerNorm-only MSE control

Date: 2026-09-12. All four first-episode evaluations are complete.
The LayerNorm-only training is running, not a completed comparison.

## Pinned models

- Student: `logs/rsl_rl/cr1_zdw0824_recurrent_visual_ppo_dagger_model9999/2026-09-12_00-37-16_ppo_mse_ln_ramp_teacher9999_env5120_iter10000_seed42/model_4500.pt`.
  Checkpoint label4500, post-update count4501. SHA256:
  `0ddc10c66efdccaed1d0eb24af10085459e598890d8cc433147e98c1bc86a284`.
- Teacher: `logs/rsl_rl/cr1_nonvisual_zdw0824_no_command_full_state_middle_stand_smooth_output/2026-09-09_13-56-37_train10000_scratch_delay0or5ms_phys5ms_seed42/model_9999.pt`.
  SHA256: `dc03d17436781a3fd24eae746fb340ce0da3b4f133be62b56cf1f9ee64f59dba`.
- Motion37 SHA256:
  `f0ab0814b30e991edfd6e097d586e67d62ca81f598ca03d649447e090ea42038`.

## Protocol

32 first episodes per controller/start, seed42, max3500 policy steps; student
executes its deterministic mean, teacher its frozen actor output. Full training
randomization, rewards, sensors and 0-or-5ms control delay are retained. Only
the reset reference frame is fixed. Camera remains30Hz and policy50Hz.
The teacher and student initial state/observation tensor hashes match exactly
for both starts. Subsequent random-number streams need not stay paired.

Frame0 tests the full up/down reference. Frame1550 is the first downward-command
frame; frame1500 would still be an upward start. The GRU uses normal online
updates and episode reset handling. Episode statistics exclude subsequent
autoreset episodes; terminal state is captured before autoreset. Success means
motion completion without a simultaneous non-timeout failure. Unfinished first
episodes are censored, not successes.

## Results

| Fixed start | Controller | Clean successes | Mean episode duration | Main termination terms |
|---|---|---:|---:|---|
| 0, full up/down | student mean | 0/32 | 12.05s | required_contact30, body_local_pose2 |
| 0, full up/down | teacher | 30/32 | 58.22s | motion_complete30, body_local_pose2 |
| 1550, descent | student mean | 0/32 | 4.85s | body_local_pose18, root_pose13, required_contact6 |
| 1550, descent | teacher | 16/32 | 16.95s | motion_complete16, root_pose15, body_local_pose4 |

Termination counts can overlap when several conditions trigger simultaneously.
All four cohorts have zero censored episodes. The student cohorts' action-mean
RMSEs to the online teacher are0.453 (frame0) and0.470 (frame1550), measured in
raw actor action units, not joint angles.

From frame0, student root maximum rise averages only0.0352m (maximum0.0538m),
and no episode reaches the downward phase. Its mean net forward displacement
is0.134m: it is not motionless, but lacks meaningful sustained climbing.
23/32 episodes terminate at frame683; six at453 and one at609. These are exactly
29 frames after new reference-required contacts at654,424 and580 respectively.
`bad_required_contact_consecutive` checks marker-to-rung-target distance
greater than0.45m for30 policy steps (0.6s), not contact force. The timing is
consistent with failing to follow a new contact target. Full per-limb distance
traces were not recorded, so the responsible limb is not directly established
for each environment.

From frame0, the teacher completes the entire reference in30/32 cases and its
root maximum rise averages0.623m, versus0.0352m for the student. This establishes
a substantial student capability gap under matched initial conditions; the
teacher is not generally unable to climb in this synchronized environment.
Student shadow-action RMSE on teacher-controlled trajectories is0.164 (frame0)
and0.166 (frame1550), compared with0.453 and0.470 on student-controlled states.
These are errors of the student, not errors of the teacher against itself.
The different state distributions are consistent with compounding closed-loop
errors, but this comparison alone does not identify the initial cause.

This rules out output sampling noise as the sole explanation for poor frame0
or frame1550 performance: the mean policy itself fails these tests. It does
not isolate observation aliasing, recurrent memory, feature scale, or PPO/MSE
optimization as the unique cause. Fixed-start samples are not a success-rate
estimate across the full adaptive training reset distribution.

The teacher completes16/32 direct descent resets. Fifteen of its16 failures
occur within4.72s, while the other occurs at21.94s. This start therefore cannot
be treated as a guaranteed-success teacher condition. The result does not
identify whether reset perturbations, randomized geometry, contact/dynamics
states unlike those reached continuously, or another factor is responsible.
The paired student result remains
0/32 under exactly the same initial state and observation samples.

## LayerNorm-only MSE experiment

Running from scratch under:
`logs/rsl_rl/cr1_zdw0824_recurrent_visual_ln_mse_dagger_model9999/2026-09-12_10-42-07_ln_only_mse_teacher9999_env5120_iter2000_seed42`.

5120 environments,2000 updates, checkpoints every500 plus final model1999.
The old no-warmup MSE configuration is retained: fixed25% teacher/75% student,
fixed std0.1, fixed lr1e-3, one epoch and24 minibatches. Only two128-D branch
LayerNorms are added before the GRU (512 affine parameters). No PPO, critic,
entropy, velocity, reconstruction or feature-matching objective is constructed.
The full saved environment YAML and policy configuration match the old MSE run;
algorithm configuration differs only by class name. A separate task and
checkpoint data contract prevent accidental mixing with existing experiments.

Full CPU suite:347 passed,6 skipped. Real GPU training smoke:96 environments,
two updates completed. Evaluator smoke:8 environments,two steps correctly
reported as censored. The full control's first saved weights are finite and
std remains0.1. The original PPO training remains running and unchanged.

At11:07 local time the new control had reached iteration65/2000 and was still
running (PID2230357). This is a startup/health observation, not evidence that
normalization has improved climbing. Original PPO PID2148507 remains active.

See each case directory for `manifest.json`, `env.yaml`, `initial_state.pt`,
`episodes.json` and `summary.json`. Training stdout is recorded in
`outputs/training_launches/2026-09-12_ln_mse_env5120_iter2000_session.log`.
