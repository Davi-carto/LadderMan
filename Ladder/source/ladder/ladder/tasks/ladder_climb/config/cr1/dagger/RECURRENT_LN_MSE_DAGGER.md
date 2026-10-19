# LayerNorm-only recurrent visual MSE control

Task: `Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-LN-MSE-Dagger-Synced-v0`.

This is a from-scratch control for the old no-warmup mixed MSE experiment.
Only the network changes: a `LayerNorm(128)` follows each of the proprioception
and depth encoders before the GRU. The normalized network class is shared with
the PPO task, but this task never constructs a PPO algorithm or critic.

- Frozen teacher: September 9 blind `model_9999.pt`.
- Fixed 25% teacher / 75% student control from the first update.
- Action-mean MSE on both groups; fixed exploration std 0.1.
- Fixed Adam learning rate 1e-3, one epoch, 24 environment minibatches.
- Same 25-frame command-aware proprio history, 30-Hz depth, GRU/reset logic,
  rewards, reset distribution and 0-or-5ms control delay as the old MSE task.
- No PPO, value, entropy, feature, velocity or reconstruction losses.
- Initial experimental budget: 2000 updates, checkpoint interval 500, plus
  the final checkpoint `model_1999.pt` under the runner's zero-based indexing.

The task has a separate checkpoint data contract and output directory. Old
unnormalized MSE checkpoints cannot be silently loaded into this network.
Diagnostics remain under `recurrent_visual_mse_diagnostics/`, explicitly
recording the normalization and ablation. Original MSE and PPO tasks remain
registered with their existing contracts.

## Train

Do not add `--resume` for this from-scratch control. Keep quoted paths intact.

```bash
cd /home/zhangdawei/LadderMan
conda activate rl-training

python Ladder/scripts/rsl_rl/train.py \
  --task Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-LN-MSE-Dagger-Synced-v0 \
  --motion_file "Ladder/motions/zdw0824_capture1/pitchminus3_yawminus4_xminus030/37_visual_up_down_stop_command/zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_reset5mm_visual_up_down_stop_50hz_motion.npz" \
  --teacher_checkpoint "logs/rsl_rl/cr1_nonvisual_zdw0824_no_command_full_state_middle_stand_smooth_output/2026-09-09_13-56-37_train10000_scratch_delay0or5ms_phys5ms_seed42/model_9999.pt" \
  --num_envs 5120 --max_iterations 2000 --seed 42 \
  --run_name ln_only_mse_teacher9999_env5120_iter2000_seed42 --headless
```

Models are saved under `logs/rsl_rl/cr1_zdw0824_recurrent_visual_ln_mse_dagger_model9999/`.
Use the same task with the existing `play.py --fast_mode` for interactive
deterministic playback. Fast-mode playback skips reward computation, so use
the evaluation entry below when comparing returns.

## Matched first-episode evaluation

`Ladder/scripts/rsl_rl/evaluate_recurrent_dagger.py` loads only the actor and
teacher, without an optimizer or training storage. It accepts the old MSE,
normalized MSE and recurrent PPO network contracts. An example for the pinned
PPO checkpoint:

```bash
python Ladder/scripts/rsl_rl/evaluate_recurrent_dagger.py \
  --task Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-PPO-Dagger-Synced-v0 \
  --checkpoint "logs/rsl_rl/cr1_zdw0824_recurrent_visual_ppo_dagger_model9999/2026-09-12_00-37-16_ppo_mse_ln_ramp_teacher9999_env5120_iter10000_seed42/model_4500.pt" \
  --motion_file "Ladder/motions/zdw0824_capture1/pitchminus3_yawminus4_xminus030/37_visual_up_down_stop_command/zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_reset5mm_visual_up_down_stop_50hz_motion.npz" \
  --reference_frame 0 --num_envs 32 --max_steps 3500 --seed 42 \
  --controller student_mean \
  --output_dir outputs/recurrent_dagger_eval/new_comparison/frame0_student_mean \
  --headless
```

Use a new output directory on each invocation; existing results are never
overwritten. Repeat with `--controller teacher`, `fixed_std --fixed_std 0.1`,
or `checkpoint_std` to distinguish mean-policy capability from exploration.
Frame 0 tests the full up/down reference; frame 1550 tests the descending
segment. Frame 1500 is still ascending and is not a pure-down start.

Evaluation retains training randomization, contact sensors, rewards and
control delay. Deterministic means deterministic actor actions, not a
noise-free environment. `--nominal_ladder` is an explicit geometry-only
override and does not make robot resets noise-free. Initial state tensors
and hashes support matched-start checks, but later reset/noise streams need
not remain paired once policies visit different states.

Each environment contributes only its first episode. A read-only termination
hook captures post-physics state before autoreset. A clean success requires
`motion_complete` without a simultaneous failure; unfinished episodes are
reported as censored. Crossing the first downward-command frame is recorded
as an ascent-progress marker, not an independent geometric success test.
Results include full config/provenance, initial tensors, per-episode records
and summary JSON. A small sample of fixed-start evaluations does not establish
general robustness across the training reset distribution.

Verification: 347 CPU tests passed, 6 skipped; a real-GPU 96-environment,
two-update MSE training smoke completed. The saved smoke environment config
differs from the prior PPO run only in environment count (96 vs 5120).

The completed September12 model4500 deterministic evaluation is recorded in
`outputs/recurrent_dagger_eval/2026-09-12_model4500/REPORT.md`: frame0 complete
up/down succeeds30/32 for teacher and0/32 for student mean; direct descent from
frame1550 succeeds16/32 versus0/32. All cohorts use matched initial conditions
and have no censored episodes. The5120-env LayerNorm-only training has started;
its2000-update performance comparison is not yet available.
