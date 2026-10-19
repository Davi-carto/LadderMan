# Recurrent visual PPO + MSE DAgger

Task: `Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-PPO-Dagger-Synced-v0`.

This task starts a new normalized student and a new trainable critic. The
teacher is the frozen blind `2026-09-09 .../model_9999.pt`. It reuses the current
synced recurrent task's rewards, reset distribution, contacts and control delay.

| Environments | Control | Action-mean MSE | PPO, critic, GAE |
|---|---|---|---|
| Lowest 25% of IDs | Teacher deterministic mean | Yes | No |
| Remaining 75% | Student Gaussian sample | Yes | Yes |

With 5120 environments the split is 1280/3840, fixed from iteration zero.
There is no all-teacher BC warmup. All samples enter elementwise-mean action
MSE; PPO and advantage normalization select student trajectories only.

The actor consumes 25 x 85 proprioceptive history (command in every frame),
one 60 x 80 head-camera depth image, and camera freshness. The 128-D proprio
and depth features each pass through LayerNorm before the 256-D GRU. The GRU
advances on fresh 30-Hz images, the actor runs at 50 Hz, and done clears only
the affected environment's hidden state. Both imitation and PPO recompute the
full network over ordered 24-step sequences and update every student branch.
The separate critic is a 780 -> 512 -> 256 -> 128 -> 1 MLP using the existing
teacher observation tensor; that tensor never enters the student actor.

The minimized loss is
`alpha * (surrogate - entropy_coef * entropy) + value_coef * value_loss + lambda * action_MSE`.
The coefficients use the absolute, zero-based update index `t`:

| Update | PPO actor coefficient `alpha` | MSE coefficient `lambda` | Critic coefficient |
|---|---|---|---|
| 0 | 0.1 | 1.0 | 1.0 |
| 0 to 500 | Linear increase to 1.0 | 1.0 | 1.0 |
| 500 to 2000 | 1.0 | 1.0 | 1.0 |
| 2000 to 7000 | 1.0 | Linear decrease to 0.1 | 1.0 |
| 7000 onward | 1.0 | 0.1 | 1.0 |

The initial settings are:

- `ppo_initial_coeff=0.1`, `ppo_warmup_iterations=500`:
  `alpha(t) = 0.1 + 0.9 * min(t / 500, 1)`.
  This scales both the PPO surrogate and entropy term. The critic learns at
  its full coefficient from update zero; rollout remains 25% teacher / 75%
  student throughout the ramp.
- `dagger_initial_coeff=1.0`, `dagger_final_coeff=0.1`,
  `dagger_anneal_start=2000`, `dagger_anneal_iterations=5000`.
  The decline begins after update 2000 and reaches 0.1 at update 7000.
- PPO: 3 epochs, 24 environment minibatches, clip 0.2, gamma 0.99, GAE lambda
  0.95, value coefficient 1.0, entropy coefficient 0.005, grad norm 1.0.
  The effective entropy coefficient rises from 0.0005 to 0.005 with `alpha`.
- Adam starts at `3e-4`; student old/new Gaussian KL adapts the rate within
  `[1e-5, 3e-4]` around target KL 0.01. This KL is a PPO diagnostic, not a
  teacher/student distillation loss.
- Student std starts at 0.1 and is learned by PPO/entropy; MSE supervises the
  action mean. Teacher actor and its checkpoint hash remain frozen.
- No feature MSE, velocity prediction or reconstruction decoder is used.

These schedules are engineering defaults to test, rather than parameters
claimed to reproduce the paper. Coefficient values do not imply the same
ratio of PPO and imitation gradients. Annealing follows the update index;
metrics do not automatically pause it. Use the diagnostics below for manual
assessment of the schedule.
This is objective weighting, not a learning-rate ramp or a promise of 10x
smaller parameter steps: Adam can offset gradient scaling, especially for
`log_std`, which has no competing MSE gradient. The critic is not scaled by
`alpha` and is clipped separately, but shares the KL-adaptive learning rate.

## Training

Run from the project root. Keep each quoted path on one physical shell line.
Do not resume an old MSE-only checkpoint: LayerNorm, critic and optimizer state
have a new strict data contract.

```bash
cd /home/zhangdawei/LadderMan
conda activate rl-training

TASK_NAME="Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-PPO-Dagger-Synced-v0"
MOTION_FILE="/home/zhangdawei/LadderMan/Ladder/motions/zdw0824_capture1/pitchminus3_yawminus4_xminus030/37_visual_up_down_stop_command/zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_reset5mm_visual_up_down_stop_50hz_motion.npz"
TEACHER_CHECKPOINT="/home/zhangdawei/LadderMan/logs/rsl_rl/cr1_nonvisual_zdw0824_no_command_full_state_middle_stand_smooth_output/2026-09-09_13-56-37_train10000_scratch_delay0or5ms_phys5ms_seed42/model_9999.pt"

# First verify Isaac Sim reset/step and two optimizer updates on the GPU.
python Ladder/scripts/rsl_rl/train.py \
  --task "$TASK_NAME" \
  --motion_file "$MOTION_FILE" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --num_envs 96 --max_iterations 2 --seed 42 \
  --run_name ppo_dagger_ramp_gpu_smoke --headless

# After that check passes, start a fresh full training run.
python Ladder/scripts/rsl_rl/train.py \
  --task "$TASK_NAME" \
  --motion_file "$MOTION_FILE" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --num_envs 5120 --max_iterations 10000 --seed 42 \
  --run_name ppo_mse_ln_ramp_teacher9999_env5120_iter10000_seed42 \
  --headless
```

Checkpoints are saved every 500 iterations under
`logs/rsl_rl/cr1_zdw0824_recurrent_visual_ppo_dagger_model9999/`.
The existing `play.py --task "$TASK_NAME" --motion_file "$MOTION_FILE"
--checkpoint <student checkpoint> --num_envs 1 --fast_mode` plays deterministic
student actions; it restores the teacher path from checkpoint metadata and
skips loading the critic/optimizer. Use the same new task for play and resume.
Both schedules are recorded in the manifest and checkpoint `training_contract`.
An exact training resume restores the saved update index, so the ramp and
annealing continue from that point. A checkpoint saved with a different or
missing schedule is rejected for optimizer resume; actor-only play is still
allowed when the network data contract and teacher hash match.

## Diagnostics and interpretation

TensorBoard retains controller-specific reward, done fraction and termination
causes, action RMSE/correlation and per-joint errors, command/depth freshness,
hidden saturation, feature RMS, parameter changes and module gradients.
New metrics include PPO surrogate/value/entropy and their weighted terms,
old/new KL, clipping fraction, student returns, critic explained variance,
learned std, and explicit PPO student/teacher sample counts.
`ppo_coeff` records `alpha`; `weighted_ppo_surrogate_loss`,
`effective_entropy_coef`, and `weighted_ppo_actor_loss` expose the terms
actually used by the actor objective.

`ppo_preupdate_ratio_max_abs_error_probe` should be near zero before the first
minibatch update. The first minibatch also measures separate PPO-surrogate and
weighted-MSE gradients for every student branch, with their overall gradient
cosine. `weighted_ppo_{module}_grad_l2_probe` reports the PPO-surrogate gradient
after multiplication by `alpha`, allowing comparison with the weighted MSE
gradient during the ramp; the surrogate probes exclude entropy. These are
probes, not full-rollout averages. Feature RMS is measured
AFTER LayerNorm; hidden saturation/zero fractions count sampled elements, not
necessarily permanently inactive coordinates.

`recurrent_visual_ppo_diagnostics/manifest.json` records algorithm settings,
teacher/motion hashes, camera timing and environment provenance. Periodic
snapshots contain both controller groups, raw inputs, hidden states, actions,
values, returns and advantages, plus exact pre-update student weights.
`model_N.pt` is post-update N while the snapshot is pre-update N; use the
snapshot's own `policy_state_dict` when replaying its recorded actions.

CPU verification: 338 tests passed, 5 skipped; schedule boundaries, actual
actor/critic gradients, weighted entropy and resume progress are covered.
The installed RSL runner completed
two synthetic-rollout updates with the real teacher and full student network,
wrote TensorBoard/checkpoints with `alpha=0.1, 0.1018`, and restored training
at `alpha=0.1036` as well as actor-only state.
GPU physics reset/step and realistic memory/throughput still require the smoke
command above. Three encoder-backpropagation epochs cost more than pure MSE's
single epoch. These checks establish data/gradient correctness, not climbing
performance. RL does not supply an otherwise unobservable reference clock.
