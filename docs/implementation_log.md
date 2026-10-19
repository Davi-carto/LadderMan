# LadderMan Implementation Log

## 2026-10-15 - Isolated 2x reference-reward resume variant

- Registered `Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-Reference-PauseDecay-Weight2x-AMP-v0`.
  Its environment inherits the existing PauseDecay task and only multiplies the
  six reference weights by two: lower/upper body 0.30/0.30, lower/upper joints
  0.50/0.30, swing foot 0.60, anchor orientation 0.20. The baseline is unchanged.
- The runner only changes the experiment directory to
  `cr1_pause_decay_reference2x_contact_sequence_visual_amp`. Actor/critic inputs,
  PPO/AMP settings, learning-rate adaptation, reset/termination/contact logic,
  +/-3 reference window and 0.2 s grace / 0.5 s pause fade are unchanged.
- Added configuration equality/isolation and registration checks, plus a
  runtime reward test showing that shared-frame selection and pause scaling
  remain unchanged while weighted reference reward doubles.
- Verification: 33 reference-guidance tests and two IsaacSim-backed config
  tests passed. A 32-environment native `train.py` resume smoke restored the
  original model6000 and ran iterations6001 and6002. The saved model6002 has
  num_updates=6003 and amp_training_updates=6003, unchanged data/training/AMP
  contracts and finite policy/critic/discriminator parameters. Log:
  `outputs/reference2x_resume_20261015/smoke.log`.
- Smoke checkpoint is isolated under
  `logs/rsl_rl/cr1_pause_decay_reference2x_contact_sequence_visual_amp/2026-10-15_13-22-16_resume6000_env32_iter2/`.
  The full-run command resumes the ORIGINAL model6000, not that smoke output.
  `--max_iterations 4000` means 4000 additional updates, iterations6001--10000;
  5120 environments and save_interval500 are retained. Full training was not
  launched by the agent.

References checked: IsaacLab ANYmal-C `flat_env_cfg.py` reward overrides and
`reward_manager.py` dt handling; `whole_body_tracking` G1 `flat_env_cfg.py` and
tracking `mdp/rewards.py`; local `reference_guidance.py`, `train.py` resume logic,
and installed RSL-RL `runners/on_policy_runner.py` iteration semantics. No runtime
dependency on the reference project was added.

## 2026-10-15 - Depth-only closed-loop ablation before reference-weight fine-tuning

- Added a diagnostic-only script at `outputs/depth_ablation_20261015/audit_depth.py`;
  no production environment, reward, actor, optimizer, or checkpoint was changed.
- Evaluated `model_6000.pt` and `model_9999.pt` from the October 13 PauseDecay run.
  Each checkpoint used five depth conditions across the same 32 fixed fixtures:
  28 segment starts plus four additional frame-zero replicas. Randomization was
  disabled; deterministic actions controlled each independent environment, with
  its own GRU state. Only first episodes were scored, with a 2000-step cap.
- Conditions: normal, constant 1 m, constant 2 m, frozen reset history, and a
  fixed permutation of live depth histories from other normal-condition stages.
  Proprioception, privileged task inputs, rewards and termination rules were kept.
- Completed-remaining-sequence counts (out of 32), in the above order:
  model6000: `26, 0, 0, 0, 1`; model9999: `32, 0, 0, 2, 3`.
  Normal-depth frame-zero completion was 3/5 and 5/5 respectively; every corrupted
  condition was 0/5. All 320 first episodes terminated; none were censored.
- Validation: 20-step GPU smoke passed, all actions finite, exactly one scored
  terminal per environment, checkpoint SHA-256 unchanged. Same-state shadow
  inference preserved the normal GRU state and reproduced baseline actions with
  maximum errors below 4e-6. Initial non-depth observation discrepancies between
  condition groups were at most 4.83e-6 (translated environment coordinates).
- Raw actions, first-episode masks, observation snapshots, checkpoint/motion
  hashes and episode metrics are retained under
  `outputs/depth_ablation_20261015/full/model_{6000,9999}/`.
- These are nominal-geometry coverage fixtures, not IID deployment success-rate
  estimates. Five frame-zero replicas are not independent random seeds. Current
  sequence success still does not require completing the full standing motion.

References checked: IsaacLab `scripts/reinforcement_learning/rsl_rl/play.py` and
`source/isaaclab/isaaclab/envs/manager_based_rl_env.py`,
`whole_body_tracking/scripts/rsl_rl/play.py`, project
`Ladder/scripts/rsl_rl/play_transformer_vision_ppo.py`, and the earlier
`outputs/ascent_check_20261013_model3500/audit_ascent.py` diagnostic.

Reference-weight doubling remains a proposal, not an implemented change or a
launched training run. Retain phase gating, the +/-3 frame matching window and
the 0.2 s grace / 0.5 s fade on paused reference reward for that comparison.

## 2026-09-16 - Scheduled Pure Recurrent Visual MSE DAgger

- Added an independent recurrent visual MSE-DAgger task based on the latest
  25-frame proprioception, 30-Hz depth, and GRU student. It keeps PPO, critic,
  KL, feature, velocity, and reconstruction losses disabled; every rollout
  sample is trained with sequence action-mean MSE against the frozen blind
  teacher.
- Added the requested controller schedule: all environments execute the
  deterministic teacher for updates 0--499; from update 500 through 4000 the
  teacher-controlled fraction decreases linearly from 1.0 to 0.1; update 4000
  onward stays at 0.1. Schedule fields are written to diagnostics manifests
  and checkpoints and are checked on resume.
- Existing fixed 25/75, no-warmup, LN-only, PPO-DAgger and teacher-only BC
  registrations remain unchanged. The new task is
  `Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-Scheduled-MSE-Dagger-Synced-v0`.

Verification:

- [x] Modified Python files compile successfully.
- [x] CPU unit tests for recurrent mixed rollout and schedule boundaries pass
  (`2 passed`).
- [x] IsaacLab Hydra parsing reaches the new task and runner configuration;
  full environment smoke cannot run in this session because no CUDA device is
  available, and the CPU fallback lacks the remote default USD asset.

## 2026-09-01 - FP32 PPO with Auxiliary-Only Visual AMP

- Removed automatic mixed precision and GradScaler from the complete
  TransformerVision PPO actor/critic update. Actor mean/std, action log
  probability, probability ratio, entropy, adaptive KL, value prediction,
  PPO losses, backward propagation, and optimizer stepping now all run in
  FP32, matching the rollout policy's numerical path.
- Retained CUDA autocast and a dedicated `auxiliary_grad_scaler` only for the
  visual encoder's supervised auxiliary update. PPO and visual optimizers
  remain parameter-disjoint.
- Preserved the checkpoint key `grad_scaler_state_dict`, so existing complete
  visual-PPO checkpoints can still restore their scaler state; the runtime
  metric is now named `aux_grad_scaler_scale` to identify its scope.
- Left the diagnostic-only `rollout_fixed_log_prob_error`, network structure,
  observations, rewards, resets, terminations, and motion data unchanged.

References checked:

- `/home/zhangdawei/humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/algorithms/ppo.py`
- `/home/zhangdawei/humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/algorithms/ppo_ce.py`
- installed standard `rsl_rl/algorithms/ppo.py`
- `Ladder/source/ladder/ladder/rsl_rl/algorithms/transformer_vision_ppo.py`

Verification:

- [x] Modified Python files compile successfully.
- [x] All 12 TransformerVision PPO tests pass, including the CUDA test that
  verifies PPO leaves the auxiliary scaler untouched and that the visual
  auxiliary update remains finite under AMP.
- [x] A real four-environment, one-iteration headless IsaacLab smoke completed
  rollout, FP32 PPO, auxiliary AMP, logging, and checkpoint save. PPO reported
  finite loss/gradient values and zero non-finite steps. The first visual AMP
  pass produced the known recoverable startup overflow, reduced only the
  auxiliary scaler from `65536` to `32768`, and completed all 20 auxiliary
  optimizer attempts.

## 2026-08-28 - Remove Action Filtering and Reset Ramp

- Removed the remaining 0.30 s reset startup ramp from
  `RateLimitedJointPositionAction`; the current source had already removed its
  first-order alpha filter.
- The action path now applies the standard IsaacLab scale/default offset, followed
  only by the 6 rad/s target-rate clamp and the existing optional 0-2 control-step
  delay.
- Existing checkpoints trained with the earlier alpha/ramp action path retain that
  training history and require a new from-scratch run for a matched evaluation.
- In the isolated ZDW0824 full-state nonvisual task, increased each hand target
  position term from `weight=0.5, active_weight=0.3` to
  `weight=0.8, active_weight=0.4`; retained `temperature=0.01`. Historical task
  configurations and checkpoints remain unchanged.
- Tightened the same task's combined foot target-position temperature from
  `0.30` to `0.05`. Added separate left/right filtered wrong-rung force terms:
  `soft=20 N`, `scale=100 N`, `weight=-0.10`, quadratic excess capped at
  `9`. During a swing phase, where no rung is designated, contact with any rung
  is treated as wrong. This remains a reward penalty rather than termination.

## 2026-08-20 - Audit Leg Morphology Before Full ZDW 0818 Retarget

- Confirmed that the accepted earlier 501-frame chain did not lengthen the
  human shanks. Its accepted source and retargeted `human_joints` preserve
  thigh/shank lengths `0.457636/0.326162 m`. Historical
  `scale107_shank_robotlen` and `scale110_shank_robotlen_plus003` commands were
  discarded `/tmp` diagnostics and never entered the accepted chain.
- Verified the current capture has thigh/shank lengths
  `0.462674/0.360750 m`, for a combined `0.823424 m` versus CR1's
  `0.829400 m`. The meaningful mismatch is `Foot -> ToeBase`: `0.124045 m`
  in the current capture versus `0.169189 m` on CR1.
- Retained the established LAFAN correspondence (`Foot -> ankle_y_link`,
  `ToeBase -> toe_link`) and tracking-weight baseline `foot/toe/hand=4/8/3`.
- Added default-off retarget morphology settings for Foot-to-ToeBase length
  and pitch reduction. They move only the in-memory ToeBase targets; source
  files remain unchanged.
- Compared five isolated 384-frame previews. Extending the target foot segment
  to `0.150` or `0.169189 m` reduces initial knee flexion by roughly
  `3-7 deg`, but raises worst robot-ladder penetration from `6.11 mm` to
  `8.48-10.75 mm` and peak joint acceleration from `684.8 rad/s^2` to
  `867.4-915.4 rad/s^2`.
- Localized every morphology candidate's regression to the same late-preview
  contact: left ankle-x collider against ladder step 1 around frames
  `335-382`. No temporal filtering of the new human source is justified.
- After visual review, selected `0.160 m` as the next Foot-to-ToeBase candidate
  and returned to the human/ladder alignment gate before retargeting. Moved the
  reversible transformation into one shared helper used by the alignment
  viewer, derived-data preparation, and retargeting.
- Created `alignment/ladder_alignment_foot160.json` as a separate branch seeded
  from the previous accepted transform. The original alignment JSON, stage-1
  NPY, and archived BVH remain unchanged. The alignment viewer now explicitly
  supports the same X reflection used by the corrected retarget.
- Compared the human standing pitch (`-28.5/-29.0 deg`) with CR1's neutral
  ankle-to-toe pitch (`-18.97 deg`) and initialized the separate branch with a
  `10 deg` reduction in absolute tilt. Added a live `0-30 deg` viewer slider;
  the saved value is consumed identically by derived-data preparation.

References checked:

- `holosoma_retargeting/config_types/data_type.py` LAFAN-to-CR1 mapping
- `holosoma_retargeting/src/interaction_mesh_retargeter.py` tracking weights
- accepted 501-frame source/result and July retarget provenance in this log
- discarded experiment commands in the user's shell history

Verification:

- [x] Retarget-only morphology helper preserves foot positions and original
  Foot-to-ToeBase directions and produces the requested length exactly.
- [x] Modified Python files compile and the option appears in CLI help.
- [x] All five previews are finite, 384 frames, and tagged at 50 Hz.
- [x] Exhaustive 32 robot-collider by 18 ladder-box audit and 50 Hz derivative
  audit completed for all five previews.
- [x] A full preparation smoke produced 8529 finite frames with both
  Foot-to-ToeBase lengths equal to `0.160 m` within `1.7e-16 m`, retained
  heading consistency `+0.999767`, and left the source SHA-256 unchanged.
- [x] With the default `10 deg` tilt reduction, first-frame pitch becomes
  `-18.37/-18.78 deg`; all non-ToeBase keypoints are bitwise unchanged and
  horizontal direction is preserved within `1.4e-15`.
- [ ] User saves the revised `foot160` human/ladder alignment.
- [ ] Generate the separately named full input and retarget, then repeat the
  left-ankle/step-1 collision and derivative gates.

## 2026-08-19 - Preserve and Audit New ZDW 0818 Ladder Capture

- Archived `/home/zhangdawei/Downloads/爬梯子5-zdw0818.bvh` under the isolated
  `holosoma/.../demo_data/climb_ladder/zdw0818_capture5/` data tree. The source
  copy is byte-identical by SHA-256.
- Parsed 8529 frames at 200 Hz with the existing BVH reader. The hierarchy,
  parent graph, rotation channels, and required LAFAN joint names are compatible
  with the previous capture, while body proportions differ and are preserved.
- Generated only the reversible stage-1 representation: 22 global keypoints in
  meter/Z-up coordinates, with initial Hips X/Y centered. No crop, mirror,
  resampling, smoothing, hand offset, height adjustment, ladder alignment, or
  contact processing was applied.
- The full stage-1 array is finite with shape `(8529, 22, 3)`. At 200 Hz its
  largest adjacent-frame point displacement is `7.74 mm`; after a physical-rate
  stride-4 preview, the full clip's largest 50 Hz displacement is `26.87 mm`.
  These checks do not justify pre-retarget smoothing.
- The motion includes a natural standing prefix, ascent, top hold, and descent.
  Ladder alignment and the final ascent crop remain visual acceptance gates.
- Extended the BVH/ladder alignment viewer with independent X/Y/Z human
  translation controls. The viewer restores prior ladder values, defaults
  legacy alignment files to zero human translation, and saves the new offset as
  `human_translation` without mutating source motion data.
- Accepted ladder translation `(0.000, 0.300, 0.000) m`, yaw `180 deg`, and
  human translation `(-0.037, 0.010, -0.050) m`. Extended the existing BVH
  preparation path to bake `human_translation` into human keypoints while the
  existing ladder transform remains exclusive to the ladder mesh.
- Generated the isolated, untrimmed Holosoma input
  `zdw0818_capture5/holosoma_input/zdw0818_capture5_full_aligned_200hz/`.
  Its `(8529, 22, 3)` motion is exactly the stage-1 data plus the accepted
  constant offset; no temporal processing was applied.
- The aligned global toe minimum is `-0.0054565088 m`. Holosoma's existing
  height normalization would raise the human by that amount, so the next
  retarget must use `human_z_offset_after_grounding=-0.0054565088 m` to preserve
  the visually accepted human/ladder relative height exactly.

Files added:
- `holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb_ladder/zdw0818_capture5/`

Verification:
- [x] Source copy SHA-256 matches the download.
- [x] BVH parse, required-joint validation, finite check, and stage-1 shape check.
- [x] Alignment viewer compiles; legacy alignment JSON restores the saved ladder
  transform with zero human offset.
- [x] Visual human/ladder transform selected, baked, and numerically verified.
- [x] Ladder `base_platform` bottom remains at world `z=0`.
- [ ] Final ascent interval selection.
- [ ] Minimal Holosoma retarget smoke before committing a full retarget.

## 2026-08-14 - Frozen-Policy TransformerVision No-Update Diagnostic

- Added the active visual-pretraining path under
  `ladder/rsl_rl/{modules,algorithms,storage,runners}`.  It has no action
  head and does not depend on the retired local SSR prototype.
- The registered `WarpTransformerVision` environment preserves the deployed
  actor's exact `policy=[B,660]` observation contract while separately exposing
  `proprio=[B,10,85]`, Warp depth `[B,2,60,80]`, current four-limb height-map
  labels `[B,4,10,3]`, base velocity `[B,3]`, and post-action next proprio
  `[B,85]`.
- `FrozenRslPolicy` constructs installed RSL-RL 5 `MLPModel` with the original
  `[512,256,256]` ELU actor and loads the split `actor_state_dict` strictly.
  It is in evaluation mode with every parameter frozen.  The visual encoder
  runs on the resulting real robot trajectory, but its 51-D latent is not an
  input to that actor and no PPO object is constructed.
- Added `check_cr1_transformer_vision_pretrain.py`.  It saves a four-limb
  target/prediction/error dashboard, target/prediction/valid/action arrays,
  and JSON.  The diagnostic computes the auxiliary gradient but calls no
  optimizer step, and fails if either frozen-actor or visual-encoder weights
  change.

References checked:
- Installed `rl-training` RSL-RL 5 `MLPModel`, `GaussianDistribution`, and
  `VecEnv` APIs.
- `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure` recurrent data, storage,
  and auxiliary reconstruction organization.
- IsaacLab 2.1 `RslRlVecEnvWrapper` and `ManagerBasedRLEnv.step()` alignment.

Verification:
- [x] New CPU tests for TransformerVision shapes, masked MSE, future-target
  done masking, frozen split-checkpoint loading, and named vector observations
  pass (`6 passed`).
- [x] Real GPU Warp smoke test with `model_6999.pt`, 4 environments, and 3
  frozen-policy steps: policy `(4,660)`, actions `(3,4,25)`, all target shapes
  valid, visual gradient `4.642549`, and both parameter deltas `0.0`.
- [ ] Operator window/dashboard acceptance before enabling the first
  auxiliary-only optimizer step.  The initial maps are randomly initialized
  predictions and are not a measure of trained visual quality.

## 2026-08-14 - Gate E No-Update Auxiliary Diagnostic

- Added `scripts/check_cr1_ssr_auxiliary.py`, an operator-facing acceptance
  tool for the recurrent Warp SSR task. It collects a real stored rollout and
  replays exactly the same `obs_h`, two-frame depth history, initial GRU
  state, reset masks, map targets, velocity targets, and post-step
  `o_(t+1)` that the future auxiliary phase will consume.
- The tool does not invoke `LadderSSRPPO.update()`, PPO, either optimizer's
  `step()`, or target normalization. It explicitly snapshots all actor
  parameters and fails unless `parameter_change_max_abs == 0`; the JSON also
  records `optimizer_steps=0`.
- It saves a target/prediction/absolute-error dashboard for all four `10x3`
  limb maps, raw arrays, and a JSON summary. The summary includes each side's
  own valid-element denominator and MSE, target percentile/range statistics,
  velocity/future/KL and grouped map losses before and after weights
  `(2,2,2,5,1)`, future-valid count, and encoder/estimator gradient norm.
- The display remains in current relative-height meters. Its color limit is
  display-only. No clip bound, scale, or training configuration was selected
  by this diagnostic.
- Added a focused model test proving masked MSE expands the validity mask and
  divides by only its valid elements, not by invalid map pixels.

References checked:
- `humanoid_vision/.../legged_robot.py` height-target construction.
- `humanoid_vision/.../algorithm_plugin.py` decoder MSE and optional row
  weighting.

Verification:
- [x] `rl-training`: recurrent-model and local-SSR-PPO tests pass (`8 passed`).
- [x] `rl-training`: full Ladder suite reaches `124 passed, 1 skipped`.
- [ ] Three existing Warp CPU tests cannot initialize in this sandbox because
  CUDA is unavailable and Warp's cache directory is read-only.
- [ ] Operator GPU acceptance of the no-update dashboard, target ranges, loss
  balance, and gradient norm before any auxiliary optimizer step.

## 2026-08-14 - Gate D Local Recurrent Asymmetric PPO Integration

- Added the opt-in `Ladder-Climb-CR1-Geometry-DR-WarpSSRVisual-v0` task. It
  reuses the accepted Warp visual environment but has its own runner config;
  existing nonvisual, RTX visual, and feed-forward Warp visual tasks are
  unchanged.
- Added `LadderSSRPPO` through the installed RSL-RL 5
  `algorithm.class_name -> construct_algorithm()` extension point. The local
  algorithm stores the exact rollout `[o_t, z_t]` actor input, action
  distribution, critic observation, raw temporal visual inputs, current
  privileged map/velocity labels, post-step `o_(t+1)`, done flags, and the
  rollout-start GRU state.
- PPO updates only the action MLP, action standard deviation, and critic using
  the stored actor input. The visual encoder, GRU, heads, and decoders belong
  exclusively to the separate auxiliary optimizer. This preserves the sampled
  VAE latent for the PPO ratio and prevents PPO gradients entering the visual
  encoder in the first reference-aligned baseline.
- Added a dedicated vector wrapper that exposes the required named fields as
  top-level `TensorDict` tensors. RSL-RL 5's `check_nan()` cannot traverse a
  nested `TensorDict`; flattening occurs only at this runner boundary and does
  not alter the environment's named observation groups.
- Added `check_cr1_ssr_ppo.py`, a bounded GPU smoke test: finite rollout,
  stored-latent log-probability check, one PPO update with auxiliary optimizer
  disabled, checkpoint reload, and deterministic action equality.

References checked:
- Installed `rl-training` RSL-RL 5 `OnPolicyRunner`, `PPO`, `RolloutStorage`,
  and `MLPModel` source.
- `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure` PPO-CE, rollout storage,
  recurrent actor, and auxiliary-optimizer separation.

Verification:
- [x] Synthetic named-observation rollout/update/reload and standard
  `OnPolicyRunner` extension test pass (`13 passed` with related visual tests).
- [x] Full Ladder CPU collection reaches `123 passed, 1 skipped`.
- [ ] Real Warp/IsaacLab bounded smoke remains for operator acceptance. The
  three existing CPU Warp tests still cannot initialize in this sandbox because
  Warp's cache path is read-only and no CUDA device is available.

## 2026-08-14 - Gate B 85-D Temporal Proprioception Accepted

- Added joint velocities to the common SSR proprioception frame. The accepted
  order is angular velocity 3, orientation 6D 6, joint position 25, joint
  velocity 25, previous action 25, and climbing direction 1, for 85 values.
- The accepted Warp test confirms reset-filled `obs_h=(B,10,85)`, two-frame
  depth `(B,2,60,80)`, velocity `(B,3)`, four local maps `(B,4,10,3)`, and
  post-step next-proprio target `(B,85)`.

## 2026-08-14 - Gate C0 Pure PyTorch Transformer-GRU Visual Actor

- Added `ladder.models.LadderSSRVisualActor` as an isolated PyTorch module.
  It implements the selected `10-frame proprioception -> 64-D state token`,
  `two depth frames -> four-layer CNN -> 10 x 64-D spatial tokens`, two-layer
  self-attention Transformer, `704 -> 256` ELU fusion, and `GRU(256,256)`.
- The seven named heads produce velocity, four bilateral 8-D terrain latents,
  and dynamic `mu/logvar`. Only velocity, four terrain latents, and sampled
  (or deterministic mean) dynamics form the 51-D actor latent. `logvar` and
  all privileged targets are outside the actor input by construction.
- Added four independent `8 -> 64 -> 128 -> 30` map decoders for the accepted
  `10 x 3` hand/foot grids, an SSR-style future-proprioception decoder, the
  five auxiliary losses with weights `(2,2,2,5,1)`, and a diagonal action
  distribution helper. No task, camera, existing visual actor, task
  registration, or PPO code changed.
- Added CPU tests for rollout and padded sequence shapes, singleton depth
  channel canonicalization, selected-environment GRU reset, deterministic and
  sampled VAE paths, actor-input isolation from `logvar`, finite losses, and
  gradients through state and depth encoders.
- The active Gate-B term now returns the selected 85-D proprioception frame:
  angular velocity, orientation, joint position, joint velocity, previous
  action, and direction. The model still requires `proprio_dim` explicitly so
  future callers cannot silently change this contract.

References checked:
- `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/.../actor_critic.py`
- `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/.../submodule.py`
- `humanoid_vision/actor_critic.py` and `submodule.py`

Verification:
- [x] `rl-training`: new recurrent-model tests plus existing visual-model tests
  pass (`6 passed`).
- [x] Full Ladder collection reaches `120 passed`.
- [ ] Three pre-existing Warp renderer CPU tests cannot initialize in this
  sandbox because Warp 1.14 writes only to read-only
  `/home/zhangdawei/.cache/warp`, and this session has no CUDA device.

## 2026-08-13 - Gate A Gravity-Aligned Four-Limb Height-Map Diagnostic

- Added a device-safe batched world-down terrain query using runtime poses of
  CR1's 18 wood-step boxes plus its `z=-0.03 m` ground plane. It classifies
  rungs, two front rails, other ladder geometry, ground, no-hit, and invalid,
  and selects the nearest real surface.
- Added `view_cr1_limb_heightmaps.py`, a fixed-motion dashboard for left/right
  hand and foot height/class maps. Each map follows limb position only; its
  rows and columns remain aligned with world X/Y as hands and feet rotate, and
  every ray points along world `(0,0,-1)`. The dashboard shows reference-to-hit
  vertical distance and saves images, raw world hit heights, runtime ladder
  poses, and class statistics.
- The Isaac Sim viewport draws each actual hit on the corresponding ladder or
  ground surface, colored by source limb. This fixes missing initial foot
  markers, which were caused by the old ladder-only query returning 100%
  no-hit under both ground-supported feet.
- Unified all limbs to `10x3 / 0.315x0.160 m`: world-X spacing is `35 mm` and
  world-Y spacing is `80 mm`. Each side has 30 samples, so both bilateral
  terrain decoders produce 60 values.
- Nominal and randomized geometry use explicit deterministic inclination and
  spacing deltas. No observation, policy, model, task registration, reward, or
  PPO code changed; Gate B remains blocked on operator acceptance.

References checked:
- `isaaclab_v2.1.0/.../sensors/ray_caster/ray_caster.py`
- Ladder motion-command runtime ladder randomization
- `ladder/utils/wood_step_ladder.py` and `contact_schema.py`
- `humanoid_vision/.../legged_robot.py` local height-point/query flow

Verification:
- [x] Pure tensor and real 18-box nominal/randomized tests pass on CPU; the
  full Ladder suite passes (`118 passed`).
- [x] Script compiles and `--help` loads under `rl-training`.
- [ ] Operator GPU/window review of nominal and randomized motion playback.

## 2026-08-13 - Transformer Fusion and Reference Priority Corrected

- Replaced the superseded two-branch Transformer fusion with the user-selected
  path: flatten all fixed-order outputs `[B,11,64] -> [B,704]`, then apply one
  `Linear(704,256) + ELU` before `GRU(256,256)`.
- There is no state/visual split, post-Transformer visual mean, or token-wise
  64-to-256 mapping in the selected Ladder network.
- Set `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure` as the first reference
  for visual-network, recurrent data-flow, estimator/decoder, auxiliary-loss,
  rollout-storage, and PPO organization. Other `humanoid_vision` code is the
  second reference; IsaacLab remains authoritative for framework integration.
- Updated design records only. No environment, model, task, or PPO code was
  changed, and Gate A remains the next implementation step.

## 2026-08-13 - Visual Implementation Blueprint Reorganized

- Marked the final Transformer-GRU section as the authoritative design and
  separated it from the older camera-acquisition outline. The current
  feed-forward Warp visual task remains unchanged as a comparison baseline.
- Fixed the complete data flow: ten full 85-D proprioception frames, two depth
  frames, `1+10` tokens, post-Transformer fusion to 256-D, a 256-D GRU,
  seven named estimator heads, a 51-D actor latent, and 25 CR1 actions.
- Clarified that the reference global heightmap predictor is excluded while
  four training-only limb-centered ladder-map decoders remain. The maps, true
  velocity, next-proprioception, and validity fields never enter the actor.
- Recorded post-step future-target alignment, raw IsaacLab depth-history shape
  canonicalization, padded recurrent update shapes, and done-only GRU reset.
- Selected the reference-aligned first optimizer boundary: PPO uses the exact
  stored rollout actor latent, then a separate auxiliary phase re-encodes the
  trajectory. This prevents VAE re-sampling from corrupting the PPO ratio.
- Rewrote Gate A-F with explicit acceptance checks. No model, environment,
  registration, or PPO source was changed; Gate A remains the next code task.

References checked:
- `humanoid_vision/actor_critic.py` and `humanoid_vision/submodule.py`
- `humanoid_vision/rslRL-317-transformer/.../algorithms/ppo.py`
- IsaacLab `observation_manager.py`, `manager_term_cfg.py`, and
  `utils/buffers/circular_buffer.py`
- Installed `rl-training` RSL-RL `algorithms/ppo.py`, `storage/rollout_storage.py`,
  `modules/rnn.py`, and `models/rnn_model.py`

## 2026-08-13 - Visual Estimator and Local-Map Contract Refined (Old Fusion Superseded)

- Consolidated the visual-network plan into one non-conflicting specification:
  complete ten-frame proprioception history (provisional `D_p=85`), two depth
  frames, `1+10` 64-D self-attention tokens, separate state/visual `128-D`
  projections concatenated to `256-D`, and `GRU(256,256)`.
- Fixed `num_latent=51`. Seven independent linear heads produce velocity 3,
  left/right hand latents 8+8, left/right foot latents 8+8, and dynamic
  `mu/logvar` 16+16; only `[velocity, hand, foot, sampled-or-mean dynamic]`
  reaches the actor.
- This grid proposal was superseded by the unified `10x3` grids recorded in
  the Gate-A entry above.
- Recorded reference-derived initial auxiliary weights
  `(velocity, hand, foot, future, KL)=(2,2,2,5,1)`, subject to target
  normalization and gradient-scale checks before joint PPO optimization.
- No environment, model, or PPO source was changed. The next acceptance gate
  is a standalone visualization of all four privileged local maps.

## 2026-08-13 - Token Fusion and Bilateral Terrain Heads Updated (Superseded)

- Replaced fixed mean visual pooling in the selected Ladder design. After the
  two Transformer layers, map the state token `64->128`, flatten the ten visual
  tokens to `640` and map `640->128`, then concatenate to `[B,256]`; the
  temporal GRU still receives one `[B,256]` vector per control step.
- Mapping every token independently to 256 and concatenating would require a
  second fusion layer (or a token-wise GRU), so it is not the selected design.
- Preserved `num_latent=51` while making bilateral terrain semantics explicit:
  hand latent `16=8 left+8 right`, foot latent `16=8 left+8 right`. Each side
  has its own linear estimator head, local-map decoder, target, and MSE term.
- The raw estimator output remains 67 values because dynamic `mu16` and
  `logvar16` parameterize one 16-D VAE latent; only the sampled/mean dynamic
  latent enters the 51-D actor input.
- Updated grouped auxiliary weights to
  `(velocity, hand, foot, future, KL)=(2,2,2,5,1)`. Hand and foot group losses
  average left/right terms, making each side weight one.

## 2026-08-13 - Transformer/GRU Visual Network Design Frozen

- User requested a local Ladder visual actor based on
  `humanoid_vision/actor_critic.py` and `submodule.py`: ten proprioception
  frames -> `[256, 128, 64]` state token; two depth frames -> four-layer CNN
  and ten 64-D spatial tokens; eleven tokens -> two self-attention Transformer
  layers; pooled fusion -> GRU; velocity, hand-terrain, foot-terrain, and
  dynamic-VAE heads.
- This is an asymmetric PPO actor/critic path, not teacher-student
  distillation. Privileged ladder/reference/contact data remains critic-only.
- The reference global-heightmap reconstruction path is excluded. The requested
  hand/foot 16-D latents instead decode dedicated limb-centered local ladder
  maps; the latent dimensions are not direct geometry targets.
- No Ladder source code was changed in this design-only gate. The detailed
  contract and seven acceptance stages are recorded in
  `docs/learning/humanoid_vision_reference_and_ladder_plan.md`.
- References checked: `humanoid_vision/actor_critic.py`,
  `humanoid_vision/submodule.py`,
  `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/modules/actor_critic_recurrent.py`,
  `.../algorithms/ppo_ce.py`, `.../algorithms/algorithm_plugin.py`, and the
  installed `rl-training` RSL-RL recurrent model/storage APIs.

Use this file to track important implementation progress that should be easy to resume later.

Keep entries concise. For conceptual learning and code-reading notes, use `docs/learning/humanoid_rl_progress.md`.

## 2026-08-11 - Independent CR1 Warp Camera Acceptance Gate

- Added `ladder.utils.warp_camera.CR1WarpCamera`, locally adapting the
  `humanoid_vision` / `x2_vis` dynamic-link ray-query pattern without a runtime
  dependency on that project. Rays are transformed into each CR1 link frame for
  Warp mesh queries; the two sphere hands, 18 dynamic ladder OBBs, and ground
  plane use analytic intersections.
- Warp writes metric image-plane depth plus explicit hit classes:
  `BACKGROUND`, `RUNG`, `ROBOT`, `LADDER_OTHER`, and `GROUND`. Its RFM view keeps
  exactly `RUNG | ROBOT`, matching the already accepted paper mask semantics.
- Added `Ladder/scripts/view_cr1_warp_camera.py`. It creates one deterministic
  registered visual scene at a selectable fixed reference frame and shows/saves
  RTX depth, Warp depth, absolute difference, class image, rung mask, robot mask,
  RFM mask, and RFM-masked depth.
- Both renderers use the existing camera mount on `d435_head_link`, full
  `80 x 60` output, and zero crop. Warp uses the requested `87 x 58 deg` FOV;
  Isaac Sim forces square pixels for the RTX USD camera and therefore renders
  approximately `87 x 70.88 deg` at this 4:3 resolution. Crop controls are
  retained as post-render operations but are disabled by default.
- After the first GPU capture showed an empty rung mask, remove two unsafe CUDA
  assumptions from the diagnostic. Warp now launches on the current PyTorch
  CUDA stream so newly produced pose tensors are ready before Warp reads them,
  and synchronizes its device before zero-copy Torch output is consumed. Its
  camera pose is derived every frame from the actual `d435_head_link` body pose
  instead of `TiledCamera`'s initialization-time pose buffer.
- The diagnostic can replay the reference motion kinematically with `--play`:
  each frame writes the reference root and joint state, renders RTX, and updates
  all eight comparison windows. `--end_frame`, `--playback_fps`, and `--loop`
  control the selected range and playback behavior. Saved metadata now includes
  all runtime ladder object names, positions, and quaternions.
- Added gated `--warp_debug` instrumentation after repeated GPU all-background
  captures. It records values actually read inside the Warp kernel, the center
  world ray, all 18 center-ray box distances, per-stage candidate/final hit
  counts, and independent `ladder_only`, `robot_only`, and `ground_only` passes.
  The normal diagnostic and training observation path do not enable these
  atomic counters or extra render passes.
- GPU kernel diagnostics isolated the all-background failure to near-plane
  handling: all 4800 rays hit a robot mesh around the camera at about `0.005 m`.
  The old renderer selected that hit first and only later changed the pixel to
  background because it was below the `0.1 m` minimum depth, incorrectly hiding
  the ladder and ground. Rays now begin on the image near plane and query all
  geometry from there while reported depth remains relative to the optical
  center. A regression test covers a camera inside near-clipped robot geometry
  with a visible ladder behind it.
- The registered training environment remains on `TiledCamera`; camera/FOV pose
  randomization and multi-environment Warp training integration remain gated on
  operator acceptance of this diagnostic.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_camera.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/camera_data.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/utils/warp/ops.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sim/spawners/sensors/sensors_cfg.py`
- `humanoid_vision/.../legged_gym/utils/raycast.py` (`DepthCastMesh`)
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_geometry_dr_env_cfg.py`

Verification:
- [x] CPU Warp kernel compiles and executes analytic rung and robot hit tests.
- [x] Focused Warp camera tests pass in `rl-training` (`6 passed`); edited files
  compile and the playback CLI help exposes the expected controls.
- [x] CR1 parser finds 30 collision meshes and two sphere hands; all real STL
  meshes build Warp BVHs and complete a full mesh-query frame on CPU.
- [x] All 99 Ladder CPU tests pass; script CLI import and edited files compile in
  `rl-training`.
- [ ] Operator GPU re-check after device synchronization and dynamic parent-link
  pose fix: first confirm nonzero rung hits in one static frame, then confirm
  multiple horizontal rung bands, robot coverage, masked depth, and
  frame-to-frame motion during `--play --loop`.

## 2026-08-11 - Accepted RFM Migrated Into Visual Training Environment

- Reuse the accepted pre-physics CR1 link semantics in the registered
  `Ladder-Climb-CR1-Geometry-DR-Visual-v0` environment. The training camera
  requests non-colorized `class` semantics and builds RFM only from runtime
  `class=rung` and `class=robot` IDs; `BACKGROUND`, `UNLABELLED`, rails, and
  supports are excluded even when their IDs are nonzero.
- Keep the validated visual contract unchanged: one `60 x 80` metric-depth
  image, `0.1` RFM branch probability, and otherwise Gaussian noise
  (`std=0.005`) plus pixel dropout (`p=0.05`).
- Remove the unused `teacher` observation group. The registered runner is
  direct asymmetric PPO: actor groups are `visual_proprio + visual_depth`,
  while the critic alone consumes the privileged `critic` group. No
  teacher-student distillation path is present.
- This gate does not change the visual encoder, recurrent state, PPO losses,
  rewards, or long-training settings.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sim/utils.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/tiled_camera.py`
- `humanoid_vision/.../envs/base/legged_robot_vis.py`
- `humanoid_vision/.../rsl_rl/runners/on_policy_runner.py`

Verification:
- [x] Operator accepted the replay RFM view with all visible CR1 links and
  rungs retained.
- [x] Focused depth, semantics, camera, model, and RSL-RL compatibility tests
  pass (`20 passed`).
- [ ] Operator GPU smoke test of the registered visual environment: reset,
  several steps, actual PPO `visual_depth` output with shape `(1, 1, 60, 80)`,
  and a nonempty RFM keep mask.

## 2026-08-11 - Pre-Physics CR1 Link Semantic Coverage

- Root-level `UrdfFileCfg.semantic_tags` did not cover every rendered CR1
  component: Isaac Lab's spawn wrapper authors the tag only on the spawned
  root prim, while the converted CR1 keeps visual geometry under many rigid
  body links and uses instanceable meshes.
- During diagnostic asset spawning, apply `class=robot` directly to all 42
  collision-bearing CR1 link prims across the matched replay environments,
  before PhysX tensor views are created. Do not include generic `UNLABELLED`
  pixels in RFM because they are not necessarily robot geometry.
- Removed the first post-`gym.make()` implementation: operator replay proved
  that authoring semantic APIs on rigid-body prims after PhysX initialization
  invalidates the simulation view and breaks the following environment reset.
- This remains gated by `--visual_diagnostics`; normal shared `play.py` users
  and policy observations are unchanged.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sim/utils.py` (`clone` semantics)
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/assets/articulation/articulation.py`
- `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`

Verification:
- [x] Focused camera and RSL-RL compatibility tests pass in `rl-training`
  (`11 passed`).
- [x] Edited runtime files compile in `rl-training` and `wbt`.
- [x] All 32 URDF links with visual geometry have collision geometry and are
  covered by the 42-link pre-physics semantic pass.
- [x] Operator GPU replay: all visible CR1 surfaces are white in the RFM keep
  mask; rails, supports, ground, and background remain excluded.

## 2026-08-11 - Paper-Aligned RFM Replay Diagnostic

- Tag the diagnostic replay robot as `class=robot`, following Isaac Lab's
  official semantic-tag pattern.
- Change the RFM binary mask and masked-depth views to retain the union of
  runtime `class=rung` and `class=robot` IDs. Rails, supports, unlabelled
  geometry, ground, and background remain black. Keep the red RGB overlay
  rung-only so rung-label alignment can still be checked independently.
- Periodic captures now write `*_rfm_depth.png` and
  `*_rfm_keep_mask.png`, with rung IDs, robot IDs, and the RFM keep fraction
  recorded in metadata. Policy observations and training are unchanged.

References checked:
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_joint_pos_env_cfg.py`
- `docs/LadderMan.pdf`, Figure 3
- `/home/zhangdawei/humanoid_vision` visual data-flow reference

Verification:
- [x] Focused camera and RSL-RL compatibility tests pass in `rl-training`
  (`10 passed`).
- [x] Edited runtime files compile in `rl-training` and `wbt`.
- [ ] Operator GPU replay: confirm runtime semantics include `class=robot`,
  both rungs and visible robot surfaces retain depth, and all other pixels are
  black.

## 2026-08-11 - Live Rung-Masked Depth Diagnostic

- Added a fifth replay window, `CR1 camera rung-masked depth (m)`. It retains
  the raw depth color only where runtime semantic metadata reports
  `class=rung`; every non-rung pixel is black.
- Arrange the Isaac Sim fallback windows as three views on the first row
  (`RGB`, raw depth, masked depth) and two on the second row (binary mask and
  RGB overlay). Periodic captures now include `*_rung_masked_depth.png`.
- This is a diagnostic view only. It does not modify the nonvisual actor or
  approve the existing training observation's `semantic != 0` assumption.

Verification:
- [x] Camera and RSL-RL focused tests pass in `rl-training` (`9 passed`).
- [x] Edited runtime files compile in `rl-training` and `wbt`.
- [x] Operator GPU replay accepted the binary rung mask and rung-masked depth:
  only rung pixels are retained; rails, robot, and background are excluded.

## 2026-08-11 - High-Resolution Camera Diagnostics

- Raised only the replay diagnostic camera from `80x60` to true `640x480`
  RGB, metric-depth, and semantic outputs, matching the Isaac Lab camera demo
  and the accepted standalone ladder-camera script.
- Added `--camera_width` and `--camera_height`; both default to `640x480`.
  Changed display scaling from `6x` pixel enlargement to `1x`, so the default
  windows show native camera pixels.
- Kept the unaccepted training config unchanged. The mature
  `humanoid_vision` pipeline acquires D435 depth at `106x60`, crops it, and
  feeds a `60x60` tensor to its visual network; the eventual policy resolution
  must be selected separately from this inspection resolution.

References checked:
- `humanoid_vision/.../legged_gym/envs/base/legged_robot_vis_config.py`
- `humanoid_vision/.../legged_gym/envs/base/legged_robot_vis.py`
- `isaaclab_v2.1.0/scripts/demos/sensors/cameras.py`
- `Ladder/scripts/view_ladder_camera.py`

Verification:
- [x] Camera and RSL-RL focused tests pass in `rl-training` (`8 passed`).
- [x] Edited runtime files compile in `rl-training` and `wbt`; the replay CLI
  exposes the new resolution controls.
- [ ] Operator GPU replay: confirm `rgb_shape=(480, 640, 3)` and acceptable
  playback speed, window fit, depth detail, and rung-mask edges.

## 2026-08-11 - Isaac Sim Camera Display Fallback

- Confirmed runtime semantic IDs from the target scene: background `0`,
  unlabelled `1`, rung `2`, and ladder `3`. The diagnostic rung mask selects
  only ID `2` through its `class=rung` metadata.
- Keep OpenCV/Qt windows in environments with HighGUI. Isaac Sim 5.1 requires
  `opencv-python-headless`, so `rl-training` now automatically displays RGB,
  metric depth, rung-masked depth, rung mask, and overlay in five `omni.ui`
  floating windows.
- Reused Isaac Lab 2.1/2.3's `ByteImageProvider` image-update pattern; camera
  outputs, policy observations, and checkpoint loading are unchanged.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/ui/widgets/image_plot.py`
- `/home/zhangdawei/IsaacLab_v2.3.2/source/isaaclab/isaaclab/ui/widgets/image_plot.py`

Verification:
- [x] Camera and RSL-RL focused tests pass in `rl-training` (`8 passed`).
- [x] Display-backend detection selects OpenCV for `wbt` and `omni.ui` for
  `rl-training`; edited runtime files compile in both environments.
- [ ] Operator GPU replay: confirm all four in-app windows refresh while the
  policy plays and that the red rung overlay matches the visible rungs.

## 2026-08-11 - Dual RSL-RL Replay Compatibility

- Fixed replay setup diagnostics for both observation APIs: RSL-RL 2 returns
  the policy observation as a tensor, while RSL-RL 5 returns a mapping-like
  `TensorDict`.
- Detect the installed split-model API before loading
  `actor_state_dict`/`critic_state_dict` checkpoints. RSL-RL 5 now loads the
  actor state directly; RSL-RL 2 maps its `mlp.*` keys into the older
  monolithic actor-critic module and disables the absent observation
  normalizer.
- Keep standard `model_state_dict` checkpoints and replay without
  `--visual_diagnostics` unchanged.

Verification:
- [x] `play.py` and `rsl_rl_compat.py` compile in both `wbt` and
  `rl-training`.
- [x] Focused compatibility tests pass in `rl-training` (`4 passed`).
- [x] The full replay CLI, including `--visual_diagnostics`, loads in both
  environments.
- [ ] Run the checkpoint under `rl-training` on the GPU host and accept the
  scene reset/step plus RGB, depth, semantic IDs, and rung-mask windows.

## 2026-08-10 - CR1 Replay Camera and Semantic Diagnostics

Goal:
- Inspect CR1-mounted camera data while replaying the accepted nonvisual
  force-limited deployment-observation checkpoint, without changing its actor
  observation contract or policy weights.

References checked:
- Isaac Lab `TiledCamera` data and semantic-info flow:
  `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/tiled_camera.py`
- Isaac Lab camera semantic filter contract:
  `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/camera_cfg.py`
- CR1 ladder semantic tags:
  `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_geometry_dr_env_cfg.py`
- Mature visual reference fixed-rate depth/history flow:
  `humanoid_vision/.../legged_gym/envs/base/legged_robot_vis.py`

Implementation:
- Make the shared replay entry point self-contained by inserting
  Ladder/source/ladder into sys.path before task registration. This
  matches existing Ladder diagnostic scripts and fixes direct
  isaaclab.sh -p ../Ladder/scripts/rsl_rl/play.py launches.
- Add replay-time checkpoint compatibility for split-format LadderMan models
  saved as actor_state_dict / critic_state_dict. During play, only the actor
  MLP is loaded for deterministic inference; model_state_dict checkpoints
  still use the normal runner load path.
- Add a replay-only `TiledCamera` beneath `d435_head_link` through
  `--visual_diagnostics` in `Ladder/scripts/rsl_rl/play.py`. It does not alter
  the loaded task's `policy` or `critic` observation groups.
- Add RGB, metric depth, rung-mask, and RGB-plus-mask OpenCV windows, periodic
  capture files, camera intrinsics, raw semantic IDs, and semantic metadata.
- Build rung masks from Replicator `idToLabels` entries whose `class` contains
  `rung`; never infer a rung from a nonzero numeric semantic ID.
- Leave `wood_step_visual_env_cfg.py` and the training visual observation path
  unchanged pending this runtime validation. Its current nonzero-ID mask is not
  approved for training.

Verification:
- [x] `py_compile` and `compileall` pass for the replay script, diagnostic
  module, and semantic-mapping test.
- [x] The mapping helper assertion proves a nonzero `class=ladder` ID is
  excluded while a distinct `class=rung` ID is retained.
- [x] Split-format `model_6999.pt` actor mapping has zero difference from the
  run's exported `policy.pt` over a random batch of 660-D observations.
- [ ] Operator GPU replay: confirm camera mount, RGB/depth alignment, semantic
  labels, and rung mask against the checkpoint before changing visual training.

Remaining risks / next checks:
- The current world-identity mount is a starting point, not calibrated D435
  extrinsics. Use the CLI pose and convention arguments during acceptance.
- Do not promote the captured semantic mapping into `visual_depth.py` until it
  is observed in the actual target environment.
- `wbt` does not include `pytest`, so the focused test is present for the
  project test environment but was exercised here with an equivalent direct
  assertion.

## 2026-08-10 - Standalone Ladder RGB/Depth Camera Proof

Goal:
- Complete only the first visual acceptance stage: inspect live RGB and metric
  depth before mounting a sensor on CR1 or changing any policy code.

References checked:
- Isaac Lab camera demo: `isaaclab_v2.1.0/scripts/demos/sensors/cameras.py`
- Isaac Lab camera pose API: `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/camera.py`
- Existing CR1 camera config: `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_visual_env_cfg.py`
- Humanoid visual reference: `humanoid_vision/.../rsl_rl/modules/actor_critic_recurrent.py`

Implementation:
- Add `Ladder/scripts/view_ladder_camera.py`. It spawns one physical wood-step
  ladder and one `TiledCamera`, moves the camera diagonally above the ladder
  from lower to upper rungs and back, shows RGB and metric depth in separate
  OpenCV windows, and periodically saves RGB, colorized depth, and raw depth.
- Keep Gym task registration, CR1 sensor mounting, observations, rewards,
  network code, and PPO configuration unchanged.

Verification:
- [x] Script compiles with `python -m compileall` and `py_compile`.
- [x] Required runtime environment identified: `conda activate wbt` selects
  this repository's editable Isaac Lab 2.1 install, Isaac Sim 4.5, and OpenCV.
- [ ] Isaac Sim GUI acceptance: RGB/depth alignment, moving trajectory, and
  capture output must be checked by the operator before Stage 2.

Remaining risks / next checks:
- Do not launch from the `base` environment; it does not contain `isaaclab`.
- A headless start from this tool session reaches Isaac Sim but has no usable
  CUDA/Vulkan device (`Driver Version: 0`), so scene reset/step remains an
  operator-side GPU-host check.

## 2026-07-29 - Extend Right-Hand Rung-3 Dwell to Frame 195

Goal:
- Extend the right-hand rung-3 required-contact interval to frames `142–195`.

Implementation:
- Rebuilt from the original 1386-frame reference with source interval
  `[165, 166)` stretched by factor 31. The result is 1416 frames at 50 Hz;
  frame 196 enters swing.
- Regenerated the 5x5 geometry reset detector fields and switched the active
  rate-limited task to the new train-ready file. Right-hand penalties remain
  `-0.60` for off-target rung contact and `-0.40` for downward contact force.

Verification:
- [x] Right-hand rung-3 labels are exactly `142–195`; frame 196 is `rung=-1`.
- [x] Nominal reset-valid mask is `1287/1416`, frame 0 is valid; geometry
  envelope mask is `133/1416` and remains unused by relaxed reset.

## 2026-07-29 - Replay Repaired Motion Diagnostics Fallback

Goal:
- Keep reference replay usable after repair tools remove stale derived contact
  fields.

Implementation:
- `replay_ladder_reference.py` now recomputes nearest rung-target distances from
  `body_pos_w`, limb body indexes, and saved annotation geometry when
  `contact_nearest_distance` is absent.
- Missing `contact_raw_indicator` is reported as `raw=n/a`; saved
  `contact_speed` and final contact labels remain unchanged.

Verification:
- [x] Replay script compiles.
- [x] All 71 Ladder CPU tests pass.

## 2026-07-29 - Extend Right-Hand Rung-3 Dwell to Frame 175

Goal:
- Keep right-hand rung-3 contact through frame 175 and strengthen the active
  task's right-hand front/underside collision penalties.

Implementation:
- Rebuilt from the original 1386-frame train-ready reference using source
  interval `[165, 166)` with factor 11. The new motion has 1396 frames at
  50 Hz; right-hand rung-3 contact is exactly frames `142–175`, with release at
  frame 176.
- Regenerated the reset sidecar with the same 5x5 geometry grid and 0.05 m
  broad-phase margin used by the accepted reference workflow.
- The independent rate-limited task now uses this reference and sets
  `right_hand_step_surface_contact=-0.60` and
  `right_hand_downward_contact=-0.40`.

Files changed:
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_reference_rate_limited_env_cfg.py`
- `Ladder/motions/restored_prefix_v300_stride2_slow50/smooth_start/reference_repaired/right_hand_rung3_hold_to175/`

Verification:
- [x] New motion labels: right-hand rung 3 is `142–175`; frame 176 is swing.
- [x] Nominal reset-valid mask is `1267/1396`, frame 0 is valid; geometry
  envelope mask is `133/1396` and remains unused by relaxed reset.
- [ ] GPU IsaacLab reset/step smoke: requires a host with a working CUDA
  driver/device.

## 2026-07-29 - Right-Hand Rung-3 Release Hold

Goal:
- Give the right hand a few more 50 Hz samples on rung 3 before entering the
  next swing phase, and increase only the active task's right-hand rung-front/
  underside collision penalties.

Implementation:
- Time-stretched source interval `[165, 166)` by factor 4 with
  `repair_cr1_reference_jumps.py`; this adds three output frames without
  changing the 50 Hz rate, ladder pose, robot model, or IK path.
- Regenerated nominal and geometry-conditioned reset fields with
  `detect_cr1_reset_valid_frames.py` using the 18-box ladder and CR1 collision
  URDF. The current relaxed-reset task uses the nominal mask.
- The independent rate-limited task now consumes the new train-ready motion and
  uses `-0.40` for `right_hand_step_surface_contact` and `-0.20` for
  `right_hand_downward_contact`; other task variants remain unchanged.

Files changed:
- `Ladder/scripts/repair_cr1_reference_jumps.py` (reused)
- `Ladder/scripts/detect_cr1_reset_valid_frames.py` (reused)
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_reference_rate_limited_env_cfg.py`
- `Ladder/motions/restored_prefix_v300_stride2_slow50/smooth_start/reference_repaired/right_hand_rung3_hold/`

Verification:
- [x] New motion is 1389 frames at 50 Hz; right-hand rung-3 contact is held
  through three added frames and releases at new frame 169.
- [x] Nominal reset-valid mask is `1260/1389`, with frame 0 valid; geometry
  envelope mask is `133/1389` and is intentionally not used by relaxed reset.
- [x] All 71 CPU tests pass and the edited config compiles.
- [ ] GPU IsaacLab reset/step smoke: blocked in this container because no CUDA
  device/driver is available; run on the training GPU host.

## Entry Template

```md
## YYYY-MM-DD - Short Title

Goal:
- ...

References checked:
- Isaac Lab official: `...`
- isaaclab_tasks: `...`
- whole_body_tracking / BeyondMimic: `...`

Files changed:
- `...`

Verification:
- [ ] ...

Remaining risks / next checks:
- ...
```

## 2026-07-28 - Separate Reference and Physical Contact Metrics

Goal:
- Distinguish reference-required contact from contact physically measured on the robot and from contact that is both physical and on the configured ladder target.

Implementation:
- Keep `required_contact_fraction` as the reference phase occupancy metric.
- Read `ContactSensor.data.net_forces_w` on the configured limb bodies with a 5 N threshold.
- Add `required_physical_contact_fraction`, `required_target_contact_fraction`, and `swing_physical_contact_fraction`, plus per-limb physical/target metrics.
- Gate target success by the per-limb target distance thresholds in `contact_schema.py`; this is a target-contact proxy because net force is not pair-specific.

References checked:
- Isaac Lab contact sensor: `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`
- Ladder contact rewards: `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/rewards.py`

Files changed:
- `Ladder/source/ladder/ladder/utils/contact_metrics.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_env_cfg.py`
- `Ladder/scripts/rsl_rl/play.py`
- `Ladder/tests/test_contact_metrics.py`

Verification:
- [x] Modified files compile.
- [x] Contact classification, force threshold, target-distance gate, and phase gate tests pass.
- [ ] Run a GPU reset/step smoke test and confirm the new command metrics appear in TensorBoard.

Remaining risks / next checks:
- ContactSensor net force cannot identify the opposing collider; inspect `required_target_contact_fraction` together with per-limb target error and swing physical contact.

## 2026-07-28 - Filtered Ladder Surface Penalties and Root Disturbances

Goal:
- Penalize hand contact with the step front surface and the two inclined front side boards, and add a controlled motion-time root disturbance for sim-to-real robustness.

Implementation:
- Add filtered per-hand ContactSensors for `Ladder_step_*` and `Ladder_front_zigzag_side_board_*` boxes.
- Penalize step-box contact when the hand is not in the required target phase or is farther than 8 cm from its target; side-board contact is always penalized.
- Enable a conservative interval root velocity push every 5-9 s during training: up to 8 cm/s translational and 0.10 rad/s rotational delta.
- Keep play deterministic by default and add `--enable_disturbances` for explicit disturbance evaluation.

References checked:
- Isaac Lab contact sensor filtered force matrix: `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py`
- Isaac Lab interval push event: `isaaclab_v2.1.0/source/isaaclab/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

Files changed:
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/rewards.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_geometry_dr_env_cfg.py`
- `Ladder/scripts/rsl_rl/play.py`
- `Ladder/scripts/rsl_rl/train.py`
- `Ladder/tests/test_ladder_rewards.py`

Verification:
- [x] Modified files compile.
- [x] Filtered surface penalty tests pass.
- [ ] Run GPU scene creation/reset/step to verify all four filtered sensors resolve their ladder paths.

Remaining risks / next checks:
- The step penalty uses target distance as the face-vs-top discriminator because the current filtered force matrix does not expose the contact normal/point per box face.

## 2026-07-17 - Evaluate Heel-Biased CR1 Foot Retargeting

Goal:
- Map LAFAN `LeftFoot` / `RightFoot` to collision-free sole tracking points instead of ankle pivots and rerun the same 501-frame sequence.

Implementation:
- Add `left_heel_tracking_link` / `right_heel_tracking_link` at `(-0.01, 0, -0.06)` relative to each `ankle_x_link`; keep all physical collision geometry unchanged.
- Preserve toe/hand mappings, foot sticking, object non-penetration, and tracking weights. Increase the SQP trust region from `0.20` to `0.25` after the `0.20` run became infeasible at frame 93.

Verification:
- [x] Full output is `(501, 32)`, finite, and loads in the Holosoma viewer.
- [x] Physical-mesh foot penetration frames decrease from `369` to `254`, but worst foot penetration remains about `15.6 mm`.
- [x] The result is not accepted for training: knee penetration frames increase from `107` to `192`, worst knee penetration increases from `46.54` to `57.72 mm`, and leg acceleration peaks increase.

Result:
- Keep the heel-tracking output as an isolated diagnostic. Do not overwrite the prior reference until the redistributed knee error is addressed.
- The runtime mapping and heel marker bodies were subsequently removed after replay showed persistent toe-standing; the default LAFAN CR1 mapping remains `Foot -> ankle_y_link`.

## 2026-07-17 - Diagnose Holosoma and Training Collision Differences

Goal:
- Quantify the difference between Holosoma's simplified retarget collision model and the mesh collision model intended for training.
- Prepare a deterministic fixed-reference-frame IsaacLab smoke test for the accepted 486-frame motion.

References checked:
- Isaac Lab URDF importer: `isaaclab_v2.1.0/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py`, `urdf_converter.py`
- Holosoma runtime model selection and scene: `holosoma/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py`, `demo_data/climb_ladder/climing_ladder_mirror_x_trim_v150_650_down008/CR1-B2-pro_free_wrist_fixed_spherehand_w_ladder.xml`
- Existing CR1 training and detector configs: `Ladder/source/ladder/ladder/robots/cr1.py`, `Ladder/scripts/detect_cr1_reset_valid_frames.py`

Implementation:
- Generate a diagnostic 25-DOF sphere-hand CR1 URDF containing Holosoma's primitive collisions, including its ten fixed `5 mm` foot/toe spheres.
- Add a geometry regression test covering all 46 diagnostic collider types, sizes, local poses, and collision-bearing link transforms.
- Keep `CR1_CFG` and the offline reset detector on `CR1-B2-pro_free_wrist_fixed_spherehand.urdf`, whose mesh collisions are closer to the physical robot; the diagnostic primitive asset is not used for training.
- Update the fixed-frame smoke script to use the accepted motion by default and disable startup domain randomization.

Files changed:
- `assets/robots/CR1-B2-PRO-260608/generate_cr1_holosoma_collision_urdf.py`
- `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand_holosoma_collision.urdf`
- `Ladder/source/ladder/ladder/robots/cr1.py`
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`
- `Ladder/scripts/check_cr1_ladder_env.py`
- `Ladder/tests/test_cr1_collision_geometry.py`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_contactaware_initialstance_reset_validity_holosoma_collision_tol001.npz`
- `docs/implementation_log.md`

Verification:
- [x] Source-level comparison matches all 46 robot colliders and 37 non-root collision-bearing link transforms within `1e-5`.
- [x] Runtime MuJoCo comparison at the same qpos gives maximum robot collider world-position/rotation differences of `1.87e-7 m / 5.62e-7`; all 18 ladder boxes also match.
- [x] The diagnostic URDF contains only primitive collision shapes and loads successfully in MuJoCo.
- [x] Training and reset validation remain on the physical-shape mesh URDF; neither references the diagnostic primitive asset.
- [x] Python compileall and the collision geometry regression test pass.
- [ ] IsaacLab fixed-frame reset/step smoke test: this node cannot communicate with the NVIDIA driver, so IsaacSim cannot initialize a GPU/Vulkan device.

Remaining risks / next checks:
- The diagnostic primitive result (`36/486` valid frames, worst `53.74 mm`) is not a training-validity result and must not replace validation against the physical-shape mesh collision model.
- The physical-shape mesh detector result remains `37/486` strict-valid frames with the current postprocessed reference, so reference penetration still requires inspection.
- Run `Ladder/scripts/check_cr1_ladder_env.py --all_key_frames --hold_frame --steps 3 --headless` on a working IsaacLab GPU node before any training.

## 2026-07-17 - Refine Initial CR1 Ground Stance

Goal:
- Reduce the visible horizontal right-sole and root drift in frames `0-50` without changing the accepted ladder contact phases.

References checked:
- Existing Ladder contact-aware postprocessor: `Ladder/scripts/stabilize_cr1_contact_motion.py`
- Holosoma foot sticking reference already recorded in the preceding contact-aware stabilization entry.

Implementation:
- Treat the right sole as the initial ground support through frame `84`; keep full support weight through frame `69` and release it with the existing 15-frame cosine blend before the observed lift.
- Low-pass only the initial root XY trajectory at `0.5 Hz`, apply `75%` of the correction through frame `50`, and fade back to the original trajectory over 30 frames.
- Use right-leg-only IK to stabilize sole XY while preserving the original sole Z trajectory.
- Use shoulder/elbow-only IK to preserve both sphere-hand world positions after the root correction.
- Preserve trajectory-boundary contact weights instead of incorrectly blending in a contact run that already exists at frame `0`.

Files changed:
- `Ladder/scripts/stabilize_cr1_contact_motion.py`
- `Ladder/tests/test_contact_stabilization.py`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_contactaware_initialstance_motion.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_contactaware_initialstance_annotated_sphere_lhd0052_rhd005_footd0145_v035.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_contactaware_initialstance_reset_validity_strict_tol001.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_contactaware_initialstance_reset_validity_ignore_sphere_hands_tol001.npz`
- `docs/implementation_log.md`

Verification:
- [x] Frames `0-50` right-sole XY range decreases from `2.96/1.40 cm` to `0.22/0.20 cm`; cumulative XY path decreases from `10.25 cm` to `1.08 cm`.
- [x] Frames `0-50` root Y range decreases from `9.95 cm` to `8.09 cm`; root XY path decreases from `15.26 cm` to `11.87 cm` while forward motion remains.
- [x] All four contact indicators and rung-id runs are exactly unchanged after re-annotation.
- [x] Joint velocity/acceleration/jerk maxima remain `3.87/28.90/296.34`; all numeric arrays are finite and URDF position limits pass.
- [x] Contact-stabilization tests and Python compileall pass.
- [x] Ignoring sphere hands, reset-valid frames remain `100/486`; worst penetration remains `45.92 mm` at frame `413` for `left_knee_link / step_04_collision`.

Remaining risks / next checks:
- Replay the new reference and inspect frames `0-90`, especially right-foot release and arm posture.
- Strict reset-valid frames decrease from `40/486` to `37/486` because the shoulder/elbow compensation slightly changes sphere-hand collision geometry; the unresolved collision-authority mismatch still blocks training acceptance.

## 2026-07-16 - Contact-Aware CR1 Foot Stabilization

Goal:
- Remove stance-foot sliding introduced by independent root/joint low-pass filtering without erasing the reference's intentional lateral weight transfer.

References checked:
- Holosoma foot sticking constraint: `holosoma/src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py`
- Ladder offline FK and contact phase data: `Ladder/source/ladder/ladder/utils/urdf_kinematics.py`, `contact_annotation.py`
- Existing arm-only trajectory IK experiment: `Ladder/scripts/repair_cr1_sphere_hand_penetration.py`

Implementation:
- Add a CR1-specific postprocessor that anchors each foot contact run at its median sole position.
- Solve a trajectory-wide regularized root translation correction, with 15-frame cosine blending at contact boundaries.
- Apply five iterations of damped numerical IK using only each leg's six joints; smooth each leg's correction independently at `2.5 Hz`.
- Preserve root orientation, waist, arms, neck, and all sphere-hand targets.
- Increase only the left-hand annotation distance tolerance from `0.050` to `0.052 m` to absorb the verified `1.1 mm` root-correction shift and retain one continuous rung-6 phase.

Files changed:
- `Ladder/scripts/stabilize_cr1_contact_motion.py`
- `Ladder/source/ladder/ladder/utils/contact_schema.py`
- `Ladder/tests/test_contact_annotation.py`
- `Ladder/tests/test_contact_stabilization.py`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_contactaware_motion.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_contactaware_annotated_sphere_lhd0052_rhd005_footd0145_v035.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_contactaware_reset_validity_strict_tol001.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_contactaware_reset_validity_ignore_sphere_hands_tol001.npz`
- `docs/implementation_log.md`

Verification:
- [x] Stable contact cores have at most `0.53/0.11/0.13 cm` XYZ foot range.
- [x] Including landing/liftoff blends, worst XYZ foot range decreases from `5.58/3.38/2.43 cm` to `2.87/1.89/2.02 cm`.
- [x] Root correction is at most `1.82 cm`; leg correction is `0.00591 rad` RMS and `0.08134 rad` maximum.
- [x] Joint velocity/acceleration/jerk maxima remain `3.87/28.90/296.34`; no hard position or velocity limit violations.
- [x] Contact rung sets and continuous runs remain intact for both hands and feet.
- [x] Strict valid frames improve from `38/486` to `40/486`; worst penetration changes from `44.78` to `45.92 mm`, still below the unsmoothed `46.54 mm` baseline.
- [x] Contact-stabilization unit tests and Python compileall pass.

Remaining risks / next checks:
- Replay the contact-aware reference to judge whether the remaining landing/liftoff motion is visually natural.
- Root lateral range remains about `26 cm` because its dominant `0.3 Hz` component is the source's alternating weight transfer, not high-frequency jitter.
- Training-geometry penetration remains unresolved and still blocks training/reset-mask acceptance.

## 2026-07-16 - Smooth the New Sphere-Hand CR1 Reference

Goal:
- Remove the initial IK convergence jump and later root/leg/arm derivative spikes from the new 50 Hz reference.
- Preserve sphere-hand contact phases and avoid worsening training-geometry penetration.

Implementation:
- Compare six grouped zero-phase Butterworth candidates before selecting `2.5 Hz` for non-leg joints, `1.5 Hz` for leg joints, and `2.5 Hz` for root position/orientation.
- Remove the first 15 filtered frames (`0.3 s`) to eliminate the remaining endpoint convergence transient.
- Add `--trim_start_frames` and explicit smooth URDF hard-limit projection to `smooth_resample_cr1_motion.py`.
- Use an `0.008 rad` smooth saturation width instead of hard clipping so joint-limit correction does not reintroduce jerk.
- Recompute all FK/body velocities, regenerate sphere-hand contact labels, and rerun strict/hand-ignored penetration checks.

Files changed:
- `Ladder/scripts/smooth_resample_cr1_motion.py`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_motion.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_annotated_sphere_d005_footd0145_v035.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_reset_validity_strict_tol001.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_smooth_j2p5_leg1p5_root2p5_trim15_reset_validity_ignore_sphere_hands_tol001.npz`
- `docs/implementation_log.md`

Verification:
- [x] Retained `486` finite frames at `50 Hz` (`9.7 s`) with normalized body quaternions.
- [x] Maximum joint velocity/acceleration/jerk decrease from `16.50/270.03/4896.34` to `3.87/28.90/296.34` in SI time units.
- [x] Maximum leg, root, and sole acceleration are `22.36 rad/s^2`, `3.08 m/s^2`, and `5.67 m/s^2`; no frame exceeds `50 rad/s^2` joint acceleration.
- [x] No URDF hard position or velocity limit violations remain.
- [x] Contact rung sets remain left hand `{3,4,5,6}`, right hand `{4,5,6}`, and both feet `{0,1,2}`.
- [x] Worst training-geometry penetration does not increase (`46.54 mm` before, `44.78 mm` after).
- [x] Python compileall passes for Ladder source and scripts.

Remaining risks / next checks:
- Visually replay the smoothed reference before accepting its motion quality.
- This smoothing pass does not solve the Holosoma primitive-collision versus Ladder URDF-mesh mismatch; strict reset validity remains only `38/486` frames.
- Do not start training until collision geometry/penetration and reset-frame policy are resolved.

## 2026-07-16 - Use CR1 Sphere Hands as Contact Phase Points

Goal:
- Switch the new retarget contact phase/reward bodies from the fixed wrists to the physical sphere hands.
- Fit the left-hand target to the new retarget while preserving separate motion-tracking and foot-annotation semantics.

References checked:
- Current Ladder annotation and command paths: `Ladder/source/ladder/ladder/utils/contact_annotation.py`, `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- Current CR1 task wiring: `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_env_cfg.py`
- New 501-frame sphere-hand trajectories and seven-step target geometry.

Implementation:
- Use `left/right_sphere_hand_link` as CR1 contact limb bodies while retaining wrists in the motion-tracking body list.
- Set annotation sphere-center offsets to `(-0.015, +0.265, +0.045) m` and `(0, -0.270, +0.045) m`.
- Add per-limb contact distance thresholds: `0.05/0.05/0.145/0.145 m` for left hand, right hand, left foot, and right foot.
- Add separate command target offsets so training uses the sphere-center hand targets but keeps foot targets symmetric at `+/-0.18 m`; the left-foot `+0.13 m` offset remains annotation-only.

Files changed:
- `Ladder/source/ladder/ladder/utils/contact_schema.py`
- `Ladder/source/ladder/ladder/utils/contact_annotation.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_env_cfg.py`
- `Ladder/scripts/preprocess_ladder_motion.py`
- `Ladder/scripts/repair_cr1_sphere_hand_penetration.py`
- `Ladder/tests/test_contact_annotation.py`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_annotated_sphere_d005_footd0145_v035.npz`
- `docs/implementation_log.md`

Verification:
- [x] New contact fractions are `0.878/0.876/0.778/0.601`.
- [x] Contact rung sets are left hand `{3,4,5,6}`, right hand `{4,5,6}`, and both feet `{0,1,2}`.
- [x] Left-hand runs are `2-69`, `87-226`, `248-395`, and `412-495`; right-hand runs are `19-149`, `171-315`, and `334-496`.
- [x] Static compilation passes and all 14 contact/motion tests pass through direct invocation; neither available Python environment currently has `pytest` installed.

Remaining risks / next checks:
- Replay the new sphere-contact markers and visually confirm the fitted left-hand target before treating these labels as accepted.
- The previously measured training-geometry penetration and derivative spikes are unchanged; this contact-label correction does not make the reference training-ready.
- Full IsaacLab config instantiation still requires an IsaacSim AppLauncher process on a node with a working GPU/Vulkan device.

## 2026-07-16 - Convert and Audit the New 501-Frame CR1 Retarget

Goal:
- Convert the accepted Holosoma `+2 cm` full retarget into the Ladder motion contract without changing its pose or filtering it.
- Recompute contact labels and audit penetration/jitter before considering the motion for training.

Implementation:
- Override the incorrect Holosoma `fps=30` metadata with the actual `50 Hz` sampling rate.
- Use the offline CR1 URDF FK path previously verified against IsaacLab because this node has no Vulkan/CUDA device.
- Preserve the accepted robot/ladder pose by disabling the old initial-foot grounding adjustment during annotation.
- Generate strict and sphere-hand-ignored `1 mm` reset-validity sidecars against the current Ladder CR1 collision URDF, 18-box ladder, and `z=-0.03 m` ground.

Files changed:
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_motion.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_annotated_wrist_d0145_v035.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_reset_validity_strict_tol001.npz`
- `Ladder/motions/climing_ladder_cr1_trim_v150_650_up002_50fps_reset_validity_ignore_sphere_hands_tol001.npz`
- `docs/implementation_log.md`

Verification:
- [x] Converted motion has `501` frames at `50 Hz`, `25` joint columns, and `42` bodies after virtual sole materialization; all required arrays are finite.
- [x] Current wrist-based annotation produces contact fractions `0.096/0.401/0.778/0.601`; hand rungs are only `{5,6}`, while both feet retain `{0,1,2}`.
- [x] A diagnostic sphere-hand calculation recovers left-hand rungs `{3,4,5,6}` and right-hand rungs `{4,5,6}`; the cropped clip contains no right-hand rung-3 proximity.
- [x] Strict training-geometry penetration check leaves only `33/501` valid frames and finds a worst `46.54 mm` left-knee/step-4 penetration at frame `424`.
- [x] Ignoring sphere hands leaves `116/501` valid frames; ankle/step and knee/step intersections remain.
- [x] At the actual `50 Hz`, maximum joint velocity/acceleration are `16.50 rad/s` and `270.03 rad/s^2`; after excluding frames `0-4`, maxima remain `6.25 rad/s` and `118.48 rad/s^2`.

Remaining risks / next checks:
- Do not use this candidate for training yet. Decide whether the new sphere-hand retarget should also switch the contact phase point from wrist to sphere hand.
- Holosoma enforces simplified robot primitive collisions, while Ladder reset validation uses the training URDF collision meshes; align these collision models before another final retarget.
- Remove or repair the initial IK convergence jump and the later ankle/elbow/hip derivative spikes before regenerating contact labels and reset masks.

## 2026-07-16 - Trial a 2 cm Post-Grounding Human Height Offset

Goal:
- Test whether raising the complete human reference by `2 cm` relative to the fixed ladder reduces the remaining left-foot/step penetration.

Implementation:
- Add `task_config.human_z_offset_after_grounding`, defaulting to zero.
- Apply the offset after toe-height normalization so it is not canceled by preprocessing.
- Retarget the first 80 frames with the upright root, sphere-hand mapping, foot sticking, and primitive ladder collisions.

Files changed:
- `holosoma/src/holosoma_retargeting/holosoma_retargeting/config_types/task.py`
- `holosoma/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py`
- `docs/implementation_log.md`

Verification:
- [x] The saved human reference differs from the baseline by exactly `(0, 0, 0.02) m` at every point.
- [x] Worst 80-frame robot-ladder penetration drops from `13.55 mm` to `5.97 mm`.
- [x] Frames exceeding `2 mm` penetration drop from 21 to 1; only frame 71 remains above the threshold.
- [x] Torso tilt, head height, tracking errors, and joint derivative maxima remain comparable to the baseline.

Remaining risks / next checks:
- The height offset is useful but does not make collision feasibility exact; frame 71 still needs inspection or a final collision-correction pass.

## 2026-07-16 - Track Human Hands With CR1 Sphere Hands

Goal:
- Map the final LAFAN hand keypoints to the fixed CR1 sphere-hand bodies instead of the wrist bodies.

Implementation:
- Change `LeftHand/RightHand` for `lafan + cr1_wrist_fixed_spherehand` to `left/right_sphere_hand_link`.
- Add a regression test for the resolved hand mapping.

Files changed:
- `holosoma/src/holosoma_retargeting/holosoma_retargeting/config_types/data_type.py`
- `holosoma/src/holosoma_retargeting/tests/test_data_type.py`
- `docs/implementation_log.md`

Verification:
- [x] All three retargeting unit tests pass in the `retarget` environment.
- [x] The 80-frame upright diagnostic completes without infeasibility.
- [x] Mean right-hand tracking error drops from `6.92 cm` to `3.26 cm`.
- [x] Frames with more than `2 mm` robot-ladder penetration drop from 57 to 21; the previous wrist/step penetration interval is removed.
- [ ] Do not run the full 501-frame candidate yet: frames 59-79 retain left-foot/step-1 penetration up to `13.6 mm`.

Remaining risks / next checks:
- Resolve the remaining left-foot collision independently of the now-correct hand mapping.

## 2026-07-16 - Keep Retarget Root Orientation Z-Up

Goal:
- Prevent a forward-leaning ladder pose from initializing the CR1 root upside down.
- Preserve human heading while fixing the retarget root up direction to world `+Z`.

Implementation:
- Replace the torso-lean-based orientation estimate with a yaw-only frame built from the left/right hip direction and fixed world up.
- Add standard-library regression tests for a strongly forward-leaning pose and a 90-degree yaw pose.

Files changed:
- `holosoma/src/holosoma_retargeting/holosoma_retargeting/src/utils.py`
- `holosoma/src/holosoma_retargeting/tests/test_utils.py`
- `docs/implementation_log.md`

Verification:
- [x] Both orientation regression tests pass in the `retarget` environment.
- [x] The actual cropped human motion now initializes with yaw `0.46 deg` and world up exactly `(0, 0, 1)`, instead of an approximately 180-degree flipped root.
- [x] An 80-frame foot-sticking/collision retarget reduces body tilt from `96-114 deg` to `44-55 deg`; the head remains `47-58 cm` above the hips.
- [x] The 80-frame final retarget cost drops from `4.44` to `0.76`, with lower mean tracking errors for the torso, hands, feet, and toes.
- [ ] Do not run the full 501-frame training candidate yet: the corrected pose exposes wrist/step and later foot/step penetrations up to `13.7 mm` in the short diagnostic.

Remaining risks / next checks:
- Object non-penetration needs a separate fix or tuning pass; the pose correction itself must not be reverted to hide those collisions.
- After collision feasibility passes, rerun the complete 501-frame retarget and repeat pose, penetration, and derivative validation.

## 2026-07-16 - CR1 Ladder Retarget Preview With Foot Sticking and Primitive Collisions

Goal:
- Retarget the cropped/downshifted 22-joint human ladder motion to the 25-DOF wrist-fixed sphere-hand CR1.
- Enable velocity-derived foot sticking and robot-ladder non-penetration while preserving the 50 Hz motion rate.

Implementation:
- Replace the single colliding ladder mesh in the sequence-local MuJoCo scene with one visual-only mesh and 18 named primitive box collision geoms.
- Run all 501 frames with `motion_downsample=4`, `foot_sticking_tolerance=0.02 m`, `penetration_tolerance=0.001 m`, and foot/toe/hand tracking weights `4/8/3`.
- Save the result as an explicitly diagnostic preview under `demo_results/cr1_wrist_fixed_spherehand/climbing/ladder_trim_v150_650_down008_footstick_primitive_collision_preview/`.

Files changed:
- `holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb_ladder/climing_ladder_mirror_x_trim_v150_650_down008/ladder_body.xml`
- `docs/implementation_log.md`

Verification:
- [x] MuJoCo loads exactly one non-colliding ladder visual mesh and 18 colliding boxes.
- [x] All 144 collision-box corners match the 144 visual-mesh vertices within `1.1e-7 m`; the base-platform bottom is at `z=0`.
- [x] The optimizer completed all 501 frames without infeasibility; all saved values are finite.
- [ ] Reject the preview for training: worst robot-ladder penetration is `7.21 mm`, with 10 frames exceeding `2 mm`.
- [ ] Reject the preview for training: the trajectory contains joint speed/acceleration peaks of about `42.50 rad/s` and `1573.27 rad/s^2` at the actual 50 Hz rate.

Remaining risks / next checks:
- Holosoma still writes `fps=30` into the NPZ even though this downsampled sequence is 50 Hz; replay must explicitly use 50 FPS until the export metadata is fixed.
- The first-frame IK convergence jump and later collision-induced jumps must be addressed before using this output as an RL reference.
- The linearized collision constraint does not perform a final nonlinear feasibility check before saving, so post-retarget signed-distance validation remains mandatory.

## 2026-07-16 - Lower the Cropped Human and Ladder by 8 cm

Goal:
- Translate the cropped upstream human motion and its exact Holosoma ladder together by `(0, 0, -0.08) m`.
- Put the ladder base-platform box bottom at `z=0` without changing any human/ladder relative geometry.

References checked:
- Holosoma climbing input and scene asset loading: `holosoma/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py`
- Existing sequence-local scene/include structure: `demo_data/climb_ladder/climing_ladder_mirror_x/`

Implementation:
- Create a separate retarget-ready task directory containing one shifted 200 Hz human NPY and a shifted ladder OBJ.
- Copy the sequence-local ladder URDF, MuJoCo scene/include files, and materials; point `ladder_assets.xml` at the new OBJ.
- Record corrected baked-asset provenance as translation `(0.38, 0.04, 0.0)`, yaw `+90 deg`, and scale `1.0`.

Files changed:
- `holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb_ladder/climing_ladder_mirror_x_trim_v150_650_down008/`
- `docs/implementation_log.md`

Verification:
- [x] Every human point has delta `(0, 0, -0.08) m` within `7e-17 m`; shape remains `(2004, 22, 3)` and all values are finite.
- [x] Every ladder vertex has the same delta within `3e-16 m`; topology remains `144` vertices and `216` faces.
- [x] Base-platform bottom is now `z=0`; the minimum human keypoint is approximately `z=0.00138 m`.
- [x] New CR1/ladder MuJoCo scene loads successfully (`nq=32`, ladder collision geom resolved).

Remaining risks / next checks:
- The rear support-leg boxes extend `0.06044 m` below the base-platform bottom. Consequently the overall shifted mesh minimum is `z=-0.06044 m`; those legs will penetrate a `z=0` ground plane unless the ground/collision convention is revised.
- Holosoma initialization previously applied an additional human-only grounding translation. Confirm or disable that behavior when rerunning retargeting so it does not destroy the preserved human/ladder relation.

## 2026-07-16 - Trim and Inspect the Upstream Human Ladder Motion

Goal:
- Preserve Viewer frames `150-650` inclusive from the upstream 200 Hz human joint trajectory without overwriting it.
- Check the cropped skeleton against the actual Holosoma ladder geometry before smoothing and retargeting again.

References checked:
- Human/BVH preparation and Viewer frame mapping: `holosoma/src/holosoma_retargeting/holosoma_retargeting/data_utils/prepare_ladder_bvh.py`, `visualize_bvh_ladder_alignment.py`
- Retarget input loading: `holosoma/src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py`
- Ladder mesh and primitive source: `demo_data/climb_ladder/climing_ladder_mirror_x/ladder.obj`, `assets/ladders/ladder_A_aframe/wood_step_ladder_sim.urdf`

Implementation:
- Map the reviewed 50 Hz Viewer interval `150-650` back to raw 200 Hz frames `[600:2604]`.
- Save the resulting 2004-frame trajectory under `processed_human/`; future 4x downsampling yields exactly 501 frames.
- Evaluate keypoint/box signed distances and exact skeleton-segment/box intersections against all 18 ladder boxes.

Files changed:
- `holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb_ladder/climing_ladder_mirror_x/processed_human/climing_ladder_mirror_x_trim_v150_650_200hz_joint_positions.npy`
- `docs/implementation_log.md`

Verification:
- [x] Cropped data equals source frames `[600:2604]`, has shape `(2004, 22, 3)`, and contains only finite values.
- [x] The baked `ladder.obj` matches primitive boxes transformed by `(0.38, 0.04, 0.08)` and yaw `+90 deg` within `7e-9 m`.
- [x] No human keypoint penetrates a ladder box by more than `1 cm`; only toe keypoints enter step surfaces by about `4-8 mm`.
- [x] Skeleton intersections occur only on foot-to-toe segments at step 1/2; no lower-leg, arm, or torso segment crosses the ladder.
- [x] Relative to the lowest support-leg box bottom (`z=0.01956 m`), the minimum human point is `6.19 cm` above ground. Relative to the base-platform bottom (`z=0.08 m`), it is `1.41 mm` above ground.

Remaining risks / next checks:
- Decide which box bottom is the intended ground authority before changing vertical alignment; no height shift was applied in this step.
- `ladder_alignment.json` records `(-0.38, 0.04, 0.08)` and yaw `-90 deg`, but the baked OBJ has the opposite X/yaw transform. Treat the OBJ as the current authority and repair provenance before regenerating assets.
- The human input has only skeleton keypoints, not a body mesh, so this check cannot prove full body-volume non-penetration.

## 2026-07-16 - Arm-Only Sphere-Hand IK Is Not Training-Safe

Goal:
- Test whether the `851`-frame CR1 reference can remove all sphere-hand/ladder penetration over `1 mm` using only the three shoulder joints and elbow on each side.
- Keep the root, waist, legs, joint limits, contact schedule, and original trajectory smoothness intact.

References checked:
- Current CR1 kinematic/collision chain: `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`
- Existing signed-distance reset checker: `Ladder/scripts/detect_cr1_reset_valid_frames.py`
- Existing offline FK and contact regeneration: `Ladder/source/ladder/ladder/utils/urdf_kinematics.py`, `Ladder/source/ladder/ladder/utils/contact_annotation.py`

Implementation:
- Add an experimental global sequential-convex trajectory IK using contact-aware sphere/box separation constraints, `0.9` soft joint limits, and correction velocity/acceleration smoothing.
- Add hard arm derivative limits based on the accepted motion (`3 rad/s`, `35 rad/s^2`, and `1200 rad/s^3` using per-frame finite differences).
- Restrict collision constraint activation to actual penetration beyond the `1 mm` tolerance.

Files changed:
- `Ladder/scripts/repair_cr1_sphere_hand_penetration.py`
- `Ladder/source/ladder/ladder/utils/urdf_kinematics.py`
- `docs/implementation_log.md`

Verification:
- [x] Unbounded arm-only IK removes sphere-hand penetration beyond `1 mm` and changes only the eight shoulder/elbow joints; it adds no reset-invalid frames relative to the original sphere-hand-ignored mask.
- [x] Reject the unbounded candidate: maximum arm speed/acceleration/jerk rises from about `2.37/16.82/250.56` to `13.73/258.24/9492.16`, and the right-hand rung-3 contact run disappears after re-annotation.
- [x] Smooth bounded IK is infeasible on the second left-arm convexification step, even after excluding non-penetrating near-contact pairs from constraint activation.
- [x] Confirm all CR1 wrist joints are fixed in the training URDF, so there is no wrist DOF that can move the sphere while retaining the wrist contact-phase point.
- [x] Original `trim_start200_end851.npz` remains unchanged; no repaired candidate is accepted for training.

Remaining risks / next checks:
- The requested shoulder/elbow-only repair is not feasible under the stated collision and trajectory-quality constraints.
- The next decision must relax one constraint: revise the virtual sphere collision geometry/offset, allow waist/root participation in IK, or accept a different reference. Do not use the generated `sphere_hand_arm_ik_contact_aware.npz` candidate for training.

## 2026-07-16 - Remove Motion After Frame 850

Goal:
- Keep current reference frames `0-850` and remove frames `851-1012` without overwriting the accepted source motion.
- Regenerate the sphere-hand-ignored reset-validity sidecar for the shortened reference.

Implementation:
- Slice all 11 frame-aligned motion/contact arrays to `[:851]`.
- Preserve the original full-reference provenance and update its retained interval to `[200,1051)`.
- Add parent-tail-trim metadata recording the source `1013` frames and retained interval `[0,851)`.
- Re-run the `1 mm` reset penetration detector while ignoring both sphere-hand bodies.

Files changed:
- `docs/implementation_log.md`
- `Ladder/motions/climing_ladder_cr1_root3hz_leg2hz_other5hz_50fps_annotated_wrist_d0145_v035_trim_start200_end851.npz`
- `Ladder/motions/climing_ladder_cr1_root3hz_leg2hz_other5hz_50fps_annotated_wrist_d0145_v035_trim_start200_end851_reset_validity_ignore_sphere_hands_tol001.npz`

Verification:
- [x] New reference has `851` frames at `50 Hz` (`17.0 s`).
- [x] All 11 retained frame arrays are exactly equal to the parent motion `[:851]`; all numeric values are finite.
- [x] Final frame has hand contacts on rung `6`, foot contacts on rung `2`, and no detected non-sphere-hand penetration.
- [x] Contact rung sets remain hands `{3,4,5,6}` and feet `{0,1,2}`.
- [x] Reset-validity sidecar contains `565/851` valid frames (`66.39%`) and matches the shortened motion source/length.

Remaining risks / next checks:
- The new final frame is not stationary: maximum joint speed is about `0.476 rad/s`. `motion_complete` will time out immediately after it, so no artificial hold was added.
- The reset mask is still not wired into adaptive frame sampling.

## 2026-07-16 - Recheck Reset Frames Without Sphere-Hand Collisions

Goal:
- Recompute reset-valid frames while ignoring environment penetration from the two sphere-hand collision bodies.
- Leave the CR1 training URDF and the earlier strict sidecar unchanged.

Implementation:
- Add `--ignore_sphere_hands` to `detect_cr1_reset_valid_frames.py`.
- Skip `left_sphere_hand_link` and `right_sphere_hand_link` during signed-distance evaluation and record the ignored bodies in sidecar metadata.
- Generate a separately named `ignore_sphere_hands_tol001` sidecar at the same `1 mm` tolerance.

Files changed:
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`
- `docs/implementation_log.md`
- `Ladder/motions/climing_ladder_cr1_root3hz_leg2hz_other5hz_50fps_annotated_wrist_d0145_v035_trim_start200_reset_validity_ignore_sphere_hands_tol001.npz`

Verification:
- [x] Python compilation passes in both `retarget` and `wbt` environments.
- [x] Valid frames increase from `95/1013` (`9.38%`) to `593/1013` (`58.54%`).
- [x] Mask/index arrays partition all `1013` frames; valid frames have zero penetrating pairs and invalid frames have at least one.
- [x] Maximum remaining penetration is `0.04254 m` at frame `759`, between `left_knee_link` and `step_04_collision`.
- [x] Remaining invalid frames are dominated by ankle/step, knee/step, and wrist/side-board intersections.

Remaining risks / next checks:
- The detector ignores sphere-hand collisions only; the training URDF still retains physical sphere hands.
- Do not silently ignore ankle/step and knee/step intersections. Decide from replay whether they are expected support contacts or genuine bad reset poses before connecting this mask to the adaptive sampler.

## 2026-07-15 - Detect Collision-Free CR1 Reset Frames

Goal:
- Precompute which CR1 reference frames can be used for reset without penetrating the current training ground or wood-step ladder.
- Keep the accepted motion unchanged and store the result as a separate sidecar before wiring it into reset sampling.

References checked:
- Isaac Lab scene-query and articulation state APIs: `isaaclab_v2.1.0/source/isaaclab/isaaclab/sim/simulation_cfg.py`, `isaaclab_v2.1.0/source/isaaclab/isaaclab/assets/articulation/articulation.py`
- Holosoma signed-distance penetration evaluation: `holosoma/src/holosoma_retargeting/holosoma_retargeting/evaluation/eval_retargeting.py`
- Current collision assets and scene placement: `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`, `assets/ladders/ladder_A_aframe/wood_step_ladder_sim.urdf`, `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_env_cfg.py`

Implementation:
- Build a temporary combined MuJoCo model from the current CR1 URDF collision meshes, all 18 primitive ladder collision boxes, and the `z=-0.03 m` training ground.
- Place the ladder at `(0.38, 0.04, 0.0)` with `90 deg` yaw, matching the CR1 IsaacLab scene.
- Evaluate signed robot/environment geometry distance at every reference frame and mark a frame invalid when penetration exceeds `1 mm`.
- Save the mask, valid/invalid indexes, maximum depth, pair count, worst robot body/obstacle, and detector metadata in a separate sidecar.

Files changed:
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`
- `docs/implementation_log.md`
- `Ladder/motions/climing_ladder_cr1_root3hz_leg2hz_other5hz_50fps_annotated_wrist_d0145_v035_trim_start200_reset_validity_tol001.npz`

Verification:
- [x] Python compilation passes in both `retarget` and `wbt` environments.
- [x] MuJoCo FK agrees with all 40 comparable NPZ body poses: maximum position error about `2.1e-7 m`, maximum orientation error below `5e-4 rad`.
- [x] Sidecar arrays align with all `1013` motion frames and contain finite penetration depths.
- [x] At `1 mm` tolerance, only `95/1013` frames (`9.38%`) are valid; invalid runs are `45-112`, `114-162`, `177-186`, and `222-1012`.
- [x] Maximum penetration is `0.05975 m` at frame `888`, between `right_sphere_hand_link` and `step_07_collision`.

Remaining risks / next checks:
- Do not connect this mask to adaptive reset yet. The reference has sustained hand/step, hand/side-board, ankle/step, and knee/step intersections, so almost no later motion frames would remain available for reset.
- Visually inspect representative invalid frames and decide whether to correct the reference/ladder collision alignment or explicitly allow selected physical contact pairs before regenerating the final mask.
- MuJoCo imports the same URDF collision meshes and primitive ladder boxes, but final confirmation should still use IsaacSim collision reporting before long training.

## 2026-07-15 - Smooth CR1 Root and Leg Motion

Goal:
- Reduce the visible root/sole spikes and leg jitter in trimmed reference frames `50-130` without flattening the real leg swing.
- Preserve the accepted contact geometry and all earlier motion artifacts.

References checked:
- Existing grouped filter/resampling pipeline: `Ladder/scripts/smooth_resample_cr1_motion.py`
- Offline FK and finite-difference utilities: `Ladder/source/ladder/ladder/utils/urdf_kinematics.py`
- CR1 joint limits and kinematic model: `Ladder/source/ladder/ladder/robots/cr1.py`, `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`

Implementation:
- Add fourth-order zero-phase low-pass filtering for root position and root orientation at `3 Hz`.
- Filter root orientation in first-frame-relative rotation-vector space, then reconstruct unit rotations before SLERP.
- Strengthen the leg-joint cutoff from `3 Hz` to `2 Hz`; retain `5 Hz` for other joints and resample to `50 Hz`.
- Recompute CR1 FK and all joint/body linear/angular velocities, regenerate contact labels, and trim original frames `0-199`.
- Write new `root3hz_leg2hz_other5hz` artifacts without overwriting previous references.

Files changed:
- `Ladder/scripts/smooth_resample_cr1_motion.py`
- `docs/implementation_log.md`
- `Ladder/motions/climing_ladder_cr1_root3hz_leg2hz_other5hz_50fps_motion.npz`
- `Ladder/motions/climing_ladder_cr1_root3hz_leg2hz_other5hz_50fps_annotated_wrist_d0145_v035.npz`
- `Ladder/motions/climing_ladder_cr1_root3hz_leg2hz_other5hz_50fps_annotated_wrist_d0145_v035_trim_start200.npz`

Verification:
- [x] New full reference has `1213` frames at `50 Hz`; trimmed reference has `1013` frames (`20.24 s`).
- [x] All numeric arrays are finite; quaternion norm error is at most `1.2e-7`.
- [x] No CR1 URDF joint position or velocity limit violation; maximum joint speed is `3.84 rad/s`.
- [x] In trimmed frames `50-130`, root acceleration peak decreased from `11.57` to `1.35 m/s^2`; left/right sole peaks decreased from `26.78/28.36` to `5.73/2.20 m/s^2`.
- [x] Leg jerk RMS decreased from `28.15` to `17.35 rad/s^3`, while left-knee range stayed at about `1.65 rad`.
- [x] Contact rung sets remain hands `{3,4,5,6}` and feet `{0,1,2}`; transition counts remain unchanged.
- [x] Python compilation passes for smoothing, annotation, and replay scripts.

Remaining risks / next checks:
- Replay the new trimmed reference in IsaacSim. The remaining fast left-leg swing is part of the source motion; if it is still visually unacceptable, use local full-body time stretching instead of stronger low-pass filtering.

## 2026-07-15 - Trim the First 200 Reference Frames

Goal:
- Remove original frames `0-199` from the accepted annotated CR1 reference without overwriting it.

Implementation:
- Save original frames `200-1212` as `Ladder/motions/climing_ladder_cr1_leg3hz_other5hz_50fps_annotated_wrist_d0145_v035_trim_start200.npz`.
- Slice all 11 frame-aligned motion/contact arrays and preserve static metadata.
- Add source path, original frame count, and trim interval metadata.

Verification:
- [x] New reference has `1013` frames at `50 Hz` (`20.24 s`).
- [x] Every retained time-series value is exactly equal to the original `[200:]`; all values are finite.
- [x] Contact rung sets remain hands `{3,4,5,6}` and feet `{0,1,2}`.
- [x] Required reference loader fields and CR1 tensor shapes are present.

## 2026-07-15 - Add Interactive Reference Replay Controls

Goal:
- Show the currently displayed reference frame during replay and support frame-accurate trimming inspection.

References checked:
- Isaac Lab official keyboard devices and demo keyboard subscriptions.
- IsaacSim `omni.ui` window, label, slider, and button APIs.

Implementation:
- Add a compact `Reference Replay` window with frame/time/playback status, a seek slider, and playback buttons.
- Add keyboard controls: `Space` play/pause, arrows single-step, `PageDown/PageUp` jump 50 frames, `Home/End` seek first/final, and `I` print contact diagnostics.
- Continuously update the same terminal status line with the displayed frame.
- Keep headless replay functional without creating GUI/keyboard resources; manual camera control remains unchanged unless `--follow_camera` is explicitly passed.

Files changed:
- `Ladder/scripts/replay_ladder_reference.py`
- `docs/implementation_log.md`

Verification:
- [x] Python compilation passes.
- [ ] GUI replay check on an IsaacSim GPU/Vulkan session.

## 2026-07-15 - Reset Cleanly at Reference Completion

Goal:
- Keep the CR1 `25 s` episode cap and adaptive reference-frame initialization.
- Treat execution of the final reference frame as a successful time-out reset instead of resampling state inside an episode.

References checked:
- Isaac Lab official: `ManagerBasedRLEnv.step/_reset_idx`, observation/action/termination manager reset behavior.
- whole_body_tracking: adaptive frame sampling and in-command final-frame resampling.

Implementation:
- Add a `motion_complete` time-out termination that fires after the final frame has been executed.
- Clamp the advancing reference index at its final valid frame and remove in-command end-of-motion resampling.
- Disable `motion_complete` for held-reference diagnostics and during the zero-step reset state.
- Preserve the existing adaptive reference-frame sampler and `episode_length_s = 25.0`.

Files changed:
- `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/terminations.py`
- `Ladder/tests/test_ladder_terminations.py`
- `docs/learning/humanoid_rl_progress.md`
- `docs/implementation_log.md`

Verification:
- [x] Python compilation passes.
- [x] Offline tensor checks cover pre-final, final, held-reference, and zero-step reset cases.
- [ ] IsaacSim check: final-frame step returns `truncated=True`, then reset histories and consecutive-failure counters are clear.

Remaining work:
- Generate an offline `reset_valid_mask` after the revised reference is ready. Prefer signed-distance penetration screening over reading stale contact forces during command reset.

## 2026-07-15 - Add Joint Tracking and Torque Regularization

Goal:
- Add the missing LadderMan joint-position tracking reward.
- Reduce false undesired-contact penalties and add the requested standard IsaacLab torque penalty.

References checked:
- LadderMan: Appendix A.1, Tables 4-5.
- Isaac Lab official: `isaaclab/envs/mdp/rewards.py`, `isaaclab_tasks/.../velocity_env_cfg.py`, and `ImplicitActuator.compute()`.
- whole_body_tracking: tracking reward/config and motion-end resampling behavior.

Implementation:
- Add separate upper/lower joint-position tracking terms using `exp(-sum(square(q-q_hat))/temperature)`.
- Use LadderMan Table 4 values: upper weight/temperature `0.5/0.5`, lower `1.0/0.25`.
- Add IsaacLab `joint_torques_l2` with the common locomotion weight `-1e-5`.
- Raise the undesired-contact threshold from `1 N` to `5 N`.

Files changed:
- `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/rewards.py`
- `Ladder/tests/test_ladder_rewards.py`
- `docs/learning/humanoid_rl_progress.md`
- `docs/implementation_log.md`

Verification:
- [x] Python compilation passes; direct tensor formula check returns `exp(-10)` for the selected-joint fixture.
- [x] Joint-group regexes cover CR1 `12 lower + 13 upper` and G1 `12 lower + 17 upper` joints without gaps or overlap.
- [ ] Run the pytest test file when `pytest` is available in the IsaacLab environment.
- [ ] IsaacSim manager instantiation and per-term finite/value checks with the revised motion.

Remaining risk:
- CR1 uses implicit actuators, so `applied_torque` is IsaacLab's clipped PD estimate rather than a PhysX solver torque measurement.
- The current execution node has no CUDA/Vulkan device, so the manager smoke stopped before environment creation.

## 2026-07-15 - Add Contact Targets and Split Paper-Style Contact Rewards

Goal:
- Add geometry-derived contact targets back to the shared state-expert observation for the first fixed-ladder training pass.
- Remove four superseded rewards and replace the combined contact reward with separate LadderMan-style hand/foot terms.

Implementation:
- Add 12-D `target_contact_pos_b` after `contact_indicator`; policy and critic now have the same expected 592-D observation.
- Remove `motion_body_ori`, `motion_body_ang_vel`, `root_ladder_progress`, and `required_contact_error` from `RewardsCfg` and remove their unused reward functions.
- Replace `ladder_contact` with `hand_contact_position` and `foot_contact_position`.
- Each group averages its two limbs and evaluates `0.3 * c * exp(-||P-P_hat||^2 / 0.3) + (1-c)` with an outer IsaacLab term weight of `1.0`.
- Keep `mean_required_contact_error` as a diagnostic command metric only.

Files changed:
- `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/rewards.py`
- `docs/implementation_log.md`

Verification:
- [x] `python -m compileall -q Ladder/source/ladder Ladder/scripts`.
- [x] All 13 offline contact/FK/motion tests pass.
- [x] Static audit finds none of the four removed reward terms/functions.
- [x] Formula check: inactive `1.0`, active zero-error `0.3`, active squared-error equal to temperature `0.3/e`.
- [ ] Instantiate IsaacSim managers after the motion is revised and confirm policy/critic `(1, 592)` plus finite per-term rewards.

## 2026-07-14 - Align CR1 Expert Observations, Terminations, and Randomization with LadderMan

Goal:
- Give actor and critic the same state-based expert observations from LadderMan Table 3 while retaining `contact_indicator`.
- Replace immediate tracking failures with the paper's consecutive-step termination logic.
- Match the disclosed reference initialization and robot physical randomization ranges.
- Document, but do not yet implement, the Table 4 reward redesign.

References checked:
- LadderMan: Sec. 3.1, Appendix A.1, Tables 3-5.
- Isaac Lab official: `observation_manager.py`, `manager_term_cfg.py`, `termination_manager.py`, `envs/mdp/events.py`.
- isaaclab_tasks: locomotion observation histories, material/mass randomization, feet slip/air-time rewards.
- whole_body_tracking: `tracking_env_cfg.py`, `mdp/events.py`, and adaptive reference-frame initialization.

Implementation:
- Policy and critic now share the same expected 580-D observation: reference body positions/joints/phase, contact indicator, root state, 9 joint-position frames, 3 action frames, ladder-relative root position, body state/errors, and joint error.
- Contact target positions, rung ids, and ladder parameters are no longer observations; they remain internal command/reward/termination data.
- Root global pose and body local pose failures require 25 consecutive 50 Hz control steps; required-contact position loss requires 30.
- Reference reset perturbations are root position `+/-0.01 m`, root rotation `+/-0.05 rad`, and joints `+/-0.05 rad`; unlisted velocity perturbations are disabled.
- Startup randomization covers link mass `[0.9, 1.1]x`, torso COM `+/-0.02 m`, joint calibration offset `+/-0.01 rad`, static/dynamic friction `[0.3, 7.2]/[0.3, 3.6]`, and restitution `[0, 0.2]`.

Files changed:
- `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/__init__.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/events.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/observations.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/terminations.py`
- `docs/learning/humanoid_rl_progress.md`
- `docs/implementation_log.md`

Verification:
- [x] `python -m compileall -q Ladder/source/ladder Ladder/scripts`.
- [x] All 13 offline contact/FK/motion tests pass using the local manual runner.
- [x] `rewards.py` was not modified.
- [ ] Instantiate the managers in IsaacSim and confirm policy/critic shapes `(1, 580)`, identical finite observations, history reset behavior, and event body/joint resolution.

Remaining limitation:
- Per-environment ladder inclination `+/-1 degree` and rung spacing `+/-0.01 m` are not implemented. The current wood ladder is one static compound USD; changing command targets alone would make targets disagree with physical step colliders. A parameterized front-step collision representation is required before adding this randomization.
- The paper does not publish the position threshold used to classify a required contact as lost. The existing `0.45 m` geometric threshold is retained, now with the paper's 30-step persistence.

Reward status:
- Table 4 alignment is intentionally design-only in this milestone. The current reward implementation remains unchanged pending review of aggregation, temperature semantics, foot air-time sign, impact threshold, and implicit-actuator torque observability.

## 2026-07-14 - Fix Frame-0 Reset Termination and Ground Penetration

Goal:
- Resolve the two failures found by the first real GPU fixed-frame smoke at frame 0.

Observed GPU smoke results:
- [x] `Ladder-Climb-CR1-v0` creates successfully with the CR1 URDF and wood-step USD.
- [x] Policy observation shape is `(1, 160)`, critic shape is `(1, 302)`, and action shape is `(1, 25)`.
- [x] Reset joint/body/anchor errors are effectively zero and all reported tensors are finite.
- [x] The first smoke exposed soles at `-0.0265/-0.0300 m` against the old `z=0` plane and ankle-force spikes; the ground alignment has since been corrected.
- [x] The first smoke exposed an incorrect reset-time `ee_body_pos` termination and near-zero first-step body reward; reset target initialization has since been corrected.

Root causes and fixes:
- `body_pos_relative_w/body_quat_relative_w` were zero-filled until the first command update, but rewards and terminations run before that update. `_resample_command` now seeds them from the current reference immediately after reset state assignment.
- The visually accepted robot-ladder relation places the soles near `z=-0.03 m`. The CR1 task now uses a dedicated ground plane at `z=-0.03 m` while leaving the static ladder at its accepted `z=0` pose.

Files changed:
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_env_cfg.py`
- `Ladder/scripts/check_cr1_ladder_env.py`
- `docs/implementation_log.md`

Verification:
- [x] Python compilation and all 13 offline tests pass.
- [x] Frame-0 GPU smoke reports policy `(1, 160)`, critic `(1, 302)`, action `(1, 25)`, finite tensors, and effectively zero reset joint/body/anchor errors.
- [x] Reset no longer terminates immediately: `immediate_done=False` and all five termination terms are false.
- [x] The first valid step gives `motion_body_pos_asym=0.997`, confirming that the reset body target is initialized correctly.
- [x] Soles are geometrically aligned with the lowered plane: left sole is about `3.5 mm` above `z=-0.03 m`; right sole is within `1 um` of it.
- [x] No undesired body exceeds the `1 N` contact-history threshold in the first three steps.
- [ ] The reset force sample is about `1.47/1.53 kN` at the ankles, but the first simulated step settles to `155/621 N` (about `776 N` total, close to CR1's `766 N` weight). Check this transient under reference-action control before accepting the actuator setup.

Frame-0 reference caveat:
- The accepted motion starts dynamically rather than from rest: root horizontal speed is about `0.62 m/s`, and several leg joints have substantial initial velocity. Zero-action drift after reset is therefore expected and is not an actuator verdict; the oracle/PD rollout must determine whether the reference is physically trackable.

Geometry tradeoff:
- A single plane cannot simultaneously keep soles at `-0.03 m`, the ladder base at `0 m`, and both objects grounded. Lowering only the training ground preserves the user-approved robot-ladder relative height; the ladder remains static but its base is 3 cm above that plane.

## 2026-07-14 - Fixed-Frame CR1 Environment Diagnostic

Goal:
- Make CR1 environment resets reproducible at any reference frame.
- Add a one-environment diagnostic before oracle rollout or PPO training.

Implementation:
- Add `fixed_reference_frame` and `advance_reference` to `LadderMotionCommandCfg`.
- Add `Ladder/scripts/check_cr1_ladder_env.py` with deterministic reset ranges and zero-action steps.
- Add `--all_key_frames` to reset one simulator instance across frame 0, every unique contact-indicator transition, and the final frame.
- Report actor/critic observation shapes, 25-D action shape, reward/done shapes, finite checks, reset tracking errors, contact targets, sole Z, physical contact forces, reward terms, and termination terms.

Files changed:
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/scripts/check_cr1_ladder_env.py`
- `Ladder/tests/test_cr1_motion_processing.py`
- `Ladder/README.md`
- `docs/implementation_log.md`

Verification:
- [x] Python compilation passes.
- [x] Diagnostic CLI help resolves with the final grouped-filter motion as its default.
- [x] All 13 contact/motion-processing tests pass.
- [x] Contact transition frames are regression-tested for fixed-frame checks.
- [x] `--all_key_frames` resolves 28 unique frames, including frame `0` and frame `1212`.
- [x] IsaacLab scene and manager creation complete on the user's GPU/Vulkan machine for the frame-0 smoke.

Fixed frames for the first smoke pass:
- Frame `0` and final frame `1212`.
- Left hand: `228, 368, 389, 631, 661, 910, 932, 1079`.
- Right hand: `235, 244, 282, 499, 524, 776, 799, 1082`.
- Left foot: `344, 561, 597, 844, 881`.
- Right foot: `470, 701, 744, 976, 1021`.

Remaining risks / next checks:
- Frame 0 now passes the reset/shape/finite/termination gate on a working IsaacSim GPU.
- Check every recorded hand/foot transition and frame `1212` before implementing the oracle rollout.
- During the oracle rollout, inspect the frame-0 contact transient and decide whether an initial hold/ease-in segment is needed; do not change the accepted motion preemptively.

## 2026-07-14 - Stronger CR1 Leg Smoothing

Goal:
- Reduce the remaining visible leg jitter without changing the root or over-smoothing the upper body.
- Preserve the accepted contact phases and rung sets.

Implementation:
- Apply a fourth-order zero-phase Butterworth filter at `3 Hz` to the 12 leg joints.
- Keep waist, arm, and neck joints at `5 Hz`; keep root position/orientation unfiltered.
- Rebuild all body states with offline URDF FK and rerun 50 Hz contact annotation.

Files changed:
- `Ladder/scripts/smooth_resample_cr1_motion.py`
- `Ladder/scripts/replay_ladder_reference.py`
- `Ladder/tests/test_cr1_motion_processing.py`
- `Ladder/README.md`
- `Ladder/motions/climing_ladder_cr1_leg3hz_other5hz_50fps_motion.npz`
- `Ladder/motions/climing_ladder_cr1_leg3hz_other5hz_50fps_annotated_wrist_d0145_v035.npz`
- `docs/implementation_log.md`

Verification:
- [x] Early leg acceleration RMS decreases from `3.879` to `2.506 rad/s^2` versus the previous 5 Hz version.
- [x] Early leg jerk RMS decreases from `74.31` to `30.35 rad/s^3`.
- [x] Non-leg joint positions and the unannotated root trajectory are unchanged.
- [x] Leg changes versus the previous version are `0.00356 rad` RMS and `0.04419 rad` maximum.
- [x] Hand and left-foot contact indicators are identical; right foot changes by one frame only.
- [x] Contact rung sets remain hands `{3,4,5,6}` and feet `{0,1,2}`.
- [x] No joint/velocity limit violations or non-finite values.

Remaining risks / next checks:
- Replay the grouped-filter motion and confirm the visual improvement before deleting or superseding the 5 Hz comparison files.

## 2026-07-14 - CR1 Wood-Step Task and Smoothed 50 Hz Reference

Goal:
- Add `Ladder-Climb-CR1-v0` with the current CR1 URDF and wood step ladder asset.
- Match command geometry to the seven step-top targets while keeping training offsets symmetric at `+/-0.18 m`; the left-foot `+0.13 m` offset remains annotation-only.
- Remove high-frequency joint jitter and synchronize the reference with the 50 Hz environment control rate.

References checked:
- Isaac Lab official G1 task: `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py`
- whole_body_tracking command/config: `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`, `config/g1/flat_env_cfg.py`
- Existing CR1 import/replay: `Ladder/scripts/export_holosoma_cr1_motion.py`, `Ladder/scripts/replay_ladder_reference.py`

Files changed:
- `Ladder/source/ladder/ladder/robots/cr1.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/utils/urdf_kinematics.py`
- `Ladder/scripts/smooth_resample_cr1_motion.py`
- `Ladder/scripts/replay_ladder_reference.py`
- `Ladder/scripts/rsl_rl/train.py`
- `Ladder/scripts/rsl_rl/play.py`
- `Ladder/tests/test_cr1_motion_processing.py`
- `Ladder/motions/climing_ladder_cr1_smooth5hz_50fps_motion.npz`
- `Ladder/motions/climing_ladder_cr1_smooth5hz_50fps_annotated_wrist_d0145_v035.npz`

Verification:
- [x] Offline jitter analysis selected a fourth-order zero-phase Butterworth filter at `5 Hz`.
- [x] Joint acceleration RMS falls from `3.693` to `2.886 rad/s^2`; joint changes are `0.00266 rad` RMS and `0.05236 rad` maximum.
- [x] Reference resampled from `728 @ 30 Hz` to `1213 @ 50 Hz` while preserving the approximately `24.24 s` duration.
- [x] Offline URDF FK matches the existing IsaacLab export within `1.1e-6 m` position and about `0.0013 rad` orientation.
- [x] The final NPZ has finite 25-joint/42-body tensors and contact rung sets hands `{3,4,5,6}`, feet `{0,1,2}`.
- [x] Motion and command ladder parameters/origin/rung count match the seven-step wood ladder.
- [x] Python compilation and 12 contact/motion-processing tests pass.
- [ ] IsaacLab task instantiation, because the current execution node has no CUDA/Vulkan device and cannot create the simulator renderer.

Deferred checks requested by the user:
- [ ] Separate contact phase point, motion tracking point, and physical contact body.
- [ ] Inspect initial sole penetration (`z=-0.0265/-0.0300 m`) and reset contact impulses in IsaacSim.
- [ ] Single-env fixed-frame reset/step at frame 0, every hand/foot transition, and the final frame.
- [ ] Check actor/critic observation shapes, 25-D action shape, every reward/done shape, finite tensors, reset errors, required contact force, and immediate termination.
- [ ] Run reference joint-target oracle/PD rollout.
- [ ] Run a 64-128 env, 200-500 iteration short training with most randomization disabled.
- [ ] Scale environment count/training only after reward, done, and tracking metrics are sane.

Remaining risks / next checks:
- CR1 actuator gains and action scale are initial replay-derived values; calibrate them with the oracle rollout before serious training.
- The reference starts below the `z=0` ground plane and must not be accepted for physics training until the penetration check is resolved.

## 2026-07-14 - Use CR1 Wrists and Shift the Left Foot Target

Goal:
- Use `left/right_wrist_x_link` instead of the sphere-hand centers for CR1 hand contact annotation.
- Move only the left-foot target 5 cm toward world `-Y`; preserve the accepted right-foot target and phase.
- Keep replay target markers identical to the offsets used by preprocessing.

References checked:
- CR1 wrist chain and fixed-joint transforms in `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`
- Current annotation and replay implementations in `Ladder/source/ladder/ladder/utils/contact_annotation.py` and `Ladder/scripts/replay_ladder_reference.py`
- Per-frame CR1 wrist/sole positions and distance/speed logs from the 728-frame motion

Files changed:
- `Ladder/source/ladder/ladder/utils/contact_schema.py`
- `Ladder/source/ladder/ladder/utils/contact_annotation.py`
- `Ladder/scripts/preprocess_ladder_motion.py`
- `Ladder/scripts/replay_ladder_reference.py`
- `Ladder/tests/test_contact_annotation.py`
- `Ladder/motions/climing_ladder_cr1_downsample4_motion_annotated_wrist_d0145_v035.npz`
- `docs/implementation_log.md`

Verification:
- [x] Full-motion CR1 preprocessing preserves hand rungs `{3, 4, 5, 6}` and foot rungs `{0, 1, 2}`.
- [x] Wrist-body indexes resolve to `36/37`; the CR1 wrist distance threshold is `0.145 m` and speed remains `0.35 m/s`.
- [x] Contact fractions are `0.640/0.630/0.655/0.540` for left hand, right hand, left foot, and right foot.
- [x] The right-foot indicator is byte-for-byte equal to the accepted previous phase across all 728 frames.
- [x] Generated the wrist annotation NPZ and selected it as the replay default.
- [x] Python compilation and all nine contact annotation tests pass.

Remaining risks / next checks:
- Visually inspect wrist markers and the shifted left-foot target before accepting the labels for training.

## 2026-07-14 - Tune Contact Distance and Speed Thresholds

Goal:
- Label both hands on rungs `{3, 4, 5, 6}` and both feet on rungs `{0, 1, 2}`.
- Keep the existing point-distance rule without adding rung allowlists.

References checked:
- Current grounded/lowered CR1 annotated motion and per-frame distance/speed logs
- Contact implementation: `Ladder/source/ladder/ladder/utils/contact_annotation.py`

Files changed:
- `Ladder/source/ladder/ladder/utils/contact_annotation.py`
- `Ladder/scripts/preprocess_ladder_motion.py`
- `Ladder/scripts/replay_ladder_reference.py`
- `Ladder/tests/test_contact_annotation.py`
- `Ladder/motions/climing_ladder_cr1_downsample4_motion_annotated_d0125_v035.npz`
- `docs/implementation_log.md`

Verification:
- [x] Distance `0.10 m` misses right-hand rung 3 and left-foot rungs 0/1.
- [x] Distance `0.125 m` and speed `0.35 m/s` include the expected hand and foot rung sets.
- [x] Full-motion smoke yields hand rungs `[3, 4, 5, 6]`, foot rungs `[0, 1, 2]`, and fractions `0.508/0.405/0.407/0.540`.
- [x] Compared with `0.12 m / 0.30 m/s`, contact fractions increase from `0.452/0.299/0.345/0.537` without adding unexpected rungs.
- [x] Generated `climing_ladder_cr1_downsample4_motion_annotated_d0125_v035.npz` and selected it as the replay default.
- [x] Python compilation and all eight contact annotation test functions pass.

Remaining risks / next checks:
- Visually inspect representative frames before accepting the labels for training.

## 2026-07-14 - Lower Grounded Robot by 3 cm

Goal:
- Lower the complete robot trajectory by `0.03 m` relative to the previous sole-grounded result.
- Keep the ladder and all ladder contact targets unchanged.

References checked:
- CR1 contact schema: `Ladder/source/ladder/ladder/utils/contact_schema.py`
- Initial-foot alignment: `Ladder/source/ladder/ladder/utils/contact_annotation.py`

Files changed:
- `Ladder/source/ladder/ladder/utils/contact_schema.py`
- `Ladder/source/ladder/ladder/utils/contact_annotation.py`
- `Ladder/scripts/preprocess_ladder_motion.py`
- `Ladder/tests/test_contact_annotation.py`
- `docs/implementation_log.md`

Verification:
- [x] Initial lowest-sole target changes from `z=0` to `z=-0.03 m`.
- [x] Raw-motion total Z correction changes from `+0.06676098 m` to `+0.03676098 m`.
- [x] The target height is explicit in CLI/config and recorded in the annotated NPZ.
- [x] Every robot body moves by exactly `-0.03 m` relative to the previous result; ladder pose and targets are unchanged.
- [x] Full-motion smoke labels contacts as left hand `0.210`, right hand `0.045`, left foot `0.223`, right foot `0.545`.

Remaining risks / next checks:
- Regenerate d010 labels and visually verify the intended 3 cm downward placement.

## 2026-07-14 - Ground Initial CR1 Sole Points

Goal:
- Raise the exported robot trajectory so the lowest initial sole point is on the ground.
- Keep the ladder fixed at the base-platform-grounded pose.

References checked:
- CR1 virtual sole schema: `Ladder/source/ladder/ladder/utils/contact_schema.py`
- Motion annotation path: `Ladder/source/ladder/ladder/utils/contact_annotation.py`
- Holosoma grounding metadata: `climing_ladder_mirror_x/retarget_ladder_alignment.json`

Files changed:
- `Ladder/source/ladder/ladder/utils/contact_annotation.py`
- `Ladder/scripts/preprocess_ladder_motion.py`
- `Ladder/tests/test_contact_annotation.py`
- `docs/implementation_log.md`

Verification:
- [x] Raw initial sole heights are left `-0.06323 m` and right `-0.06676 m`.
- [x] CR1 preprocessing applies a constant `+0.06676098 m` robot Z translation and records it in the NPZ.
- [x] Body positions move together; body velocities, orientations, and joint states are unchanged.
- [x] Reprocessing an already grounded annotated NPZ does not apply the offset twice.
- [x] Python compilation and all eight contact annotation test functions pass.

Remaining risks / next checks:
- Regenerate d010 labels because contact distances change after robot grounding.
- Visually confirm the initial sole-ground alignment and later ladder contacts.

## 2026-07-14 - Ground Ladder by Base Platform

Goal:
- Lower the ladder again without moving the robot.
- Use the bottom face of the large `base_platform` box as the ground reference.

References checked:
- Ladder collider: `assets/ladders/ladder_A_aframe/wood_step_ladder_physics.usda`
- Shared preset: `Ladder/source/ladder/ladder/utils/ladder_presets.py`

Files changed:
- `Ladder/source/ladder/ladder/utils/ladder_presets.py`
- `Ladder/tests/test_ladder_geometry.py`
- `docs/implementation_log.md`

Verification:
- [x] `base_platform` center is local `z=0.08 m`, half-height is `0.08 m`, and its bottom is local `z=0`.
- [x] Asset translation `z=0` places that bottom face on the world ground plane.
- [x] Step-top contact targets move down by the same `0.06044322 m`; robot motion remains unchanged.
- [x] Full-motion smoke labels contacts as left hand `0.203`, right hand `0.077`, left foot `0.225`, right foot `0.554`.

Remaining risks / next checks:
- Slanted support geometry extends about `0.06044 m` below the base-platform reference and will intersect the ground.
- Regenerate d010 labels and visually inspect fixed contact frames.

## 2026-07-14 - Ground Holosoma Ladder Without Moving Robot

Goal:
- Raise the robot relative to the ladder by lowering only the ladder and its contact targets.
- Place the lowest ladder geometry point exactly on the ground plane.

References checked:
- Holosoma alignment: `climing_ladder_mirror_x/retarget_ladder_alignment.json`
- Ladder mesh: `assets/ladders/ladder_A_aframe/wood_step_ladder.obj`
- Shared tool preset: `Ladder/source/ladder/ladder/utils/ladder_presets.py`

Files changed:
- `Ladder/source/ladder/ladder/utils/ladder_presets.py`
- `Ladder/tests/test_ladder_geometry.py`
- `docs/implementation_log.md`

Verification:
- [x] Local ladder minimum is `z=-0.06044322 m`; asset translation `z=0.06044322 m` gives world minimum `z=0`.
- [x] Ladder and all step-top targets move down together by `0.01955678 m`; robot motion is unchanged.
- [x] Holosoma's recorded ladder grounding adjustment is `-0.0195567794 m`, matching the mesh-derived value.
- [x] Full-motion smoke labels grounded contacts as left hand `0.166`, right hand `0.067`, left foot `0.000`, right foot `0.199`.

Remaining risks / next checks:
- Regenerate the annotated d010 NPZ so it carries the grounded target origin and asset pose.
- Visually inspect fixed hand and foot contact frames in IsaacSim.

## 2026-07-14 - Holosoma Ladder Frame and Replay Debugging

Goal:
- Preserve the robot/ladder relationship established in the Holosoma ladder scene.
- Make contact-label replay clear enough for frame-level visual inspection.

References checked:
- Isaac Lab official: `isaaclab_v2.1.0/source/isaaclab/isaaclab/markers/visualization_markers.py`
- Existing Ladder code: `Ladder/scripts/preprocess_ladder_motion.py`, `Ladder/scripts/replay_ladder_reference.py`, `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_geometry.py`
- Holosoma source: `climing_ladder_mirror_x/ladder.obj`, `ladder_alignment.json`, retargeted qpos NPZ files
- Ladder asset source: `assets/ladders/ladder_A_aframe/wood_step_ladder.yaml`, `wood_step_ladder_physics.usda`

Files changed:
- `Ladder/source/ladder/ladder/utils/ladder_presets.py`
- `Ladder/source/ladder/ladder/utils/contact_annotation.py`
- `Ladder/scripts/preprocess_ladder_motion.py`
- `Ladder/scripts/replay_ladder_reference.py`
- `Ladder/tests/test_ladder_geometry.py`
- `Ladder/tests/test_contact_annotation.py`
- `docs/implementation_log.md`

Verification:
- [x] Mesh comparison proves the local wood-step asset matches the baked Holosoma ladder at translation `(0.38, 0.04, 0.08)` and yaw `90 deg`.
- [x] Shared CR1 preset generates seven step-top targets from origin `(0.445, 0.04, 0.27)` with increments `(0.135, 0, 0.23)`.
- [x] Full 728-frame preprocessing smoke test writes the shared target geometry and asset pose into the annotated NPZ.
- [x] Python compilation and all seven contact annotation test functions pass.

Remaining risks / next checks:
- IsaacSim GUI verification still needs a machine with a usable GPU/Vulkan device.
- Hand and foot targets still share the same lateral offset and remain point targets; visually inspect fixed contact and swing frames before accepting labels.
- Regenerate the checked-in `d010` motion after this change; the existing file contains the earlier 12-rung target metadata.

## 2026-07-14 - CR1 Sole Point Annotation Pipeline

Goal:
- Complete the CR1 sole contact point path without changing the existing discrete rung `P_hat` targets.
- Allow existing CR1 motion files to use sole points without requiring a new IsaacSim export.

References checked:
- Isaac Lab official: `isaaclab_v2.1.0/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py`, `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/imu/imu.py`
- Existing Ladder code: `Ladder/source/ladder/ladder/utils/contact_annotation.py`, `Ladder/scripts/export_holosoma_cr1_motion.py`, `Ladder/scripts/replay_ladder_reference.py`
- CR1 assets: `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`, `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.xml`, left/right ankle mesh bounds

Files changed:
- `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`
- `Ladder/source/ladder/ladder/utils/contact_schema.py`
- `Ladder/source/ladder/ladder/utils/contact_annotation.py`
- `Ladder/tests/test_contact_annotation.py`
- `docs/implementation_log.md`

Verification:
- [x] Left/right ankle mesh bounds place the sole bottom center at approximately `(0.04490, 0, -0.07521) m`; the model and schema use `(0.045, 0, -0.075) m`.
- [x] Python/XML compile and parse checks pass.
- [x] All seven contact annotation test functions pass in the WBT Python environment.
- [x] Full 728-frame preprocessing smoke test materializes both sole bodies, expands all body state arrays from 40 to 42 bodies, and resolves limb indexes `[38, 39, 40, 41]`.
- [x] Saved annotation thresholds start with `0.10 m`; rung target generation remains unchanged.

Remaining risks / next checks:
- The current container has no usable GPU/Vulkan device, so the IsaacSim URDF load/export smoke test cannot complete here.
- With the current ladder alignment and `0.10 m` threshold, the left/right sole minimum target distances are about `0.1049 m` and `0.1326 m`; this motion produces no foot contact frames. Keep the requested threshold and validate ladder/motion alignment visually before training.
- Existing checked-in annotated NPZ files remain unchanged and still contain the earlier ankle-based labels.

## 2026-07-13 - CR1 Sole Contact Points for Annotation

Goal:
- Add left/right CR1 sole contact point bodies for offline contact annotation.
- Keep ladder targets on the existing discrete rung `P_hat` points and keep the default contact distance threshold at 0.10 m.

References checked:
- Isaac Lab official: `isaaclab_v2.1.0/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py`
- Existing Ladder code: `Ladder/source/ladder/ladder/utils/contact_annotation.py`, `Ladder/scripts/export_holosoma_cr1_motion.py`, `Ladder/scripts/preprocess_ladder_motion.py`
- CR1 assets: `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`, `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.xml`, CR1 ankle mesh bounds

Files changed:
- `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`
- `Ladder/source/ladder/ladder/utils/contact_schema.py`
- `Ladder/tests/test_contact_annotation.py`
- `docs/implementation_log.md`

Verification:
- [x] `python -m py_compile Ladder/source/ladder/ladder/utils/contact_schema.py Ladder/source/ladder/ladder/utils/contact_annotation.py Ladder/scripts/preprocess_ladder_motion.py`
- [x] Parsed the CR1 URDF and verified `left_sole_contact_point` / `right_sole_contact_point` fixed joints at `0.045 0 -0.075` under the ankle roll bodies.
- [x] Manual schema/default-threshold checks with `PYTHONPATH=Ladder/source/ladder python -c ...`.
- [x] Manual annotation check verifies the new sole body names work with the existing discrete rung target distance path.

Remaining risks / next checks:
- Existing CR1 motion npz files were not regenerated and still lack the sole contact point body names; rerun export and annotation before using the labels for training/replay.
- `pytest` is not installed in the current shell, so the added test file was not run through pytest here.
- IsaacLab/IsaacSim asset-load smoke test still needs the project IsaacLab Python environment.

## 2026-06-23 - Project Agent Guidelines

Goal:
- Establish LadderMan-specific coding guidelines for IsaacLab-based robot reinforcement learning work.
- Define where durable learning notes and implementation progress should be recorded.

References checked:
- Existing global/project agent guidelines: `.codex/AGENTS.md`, `LadderMan/AGENTS.md`
- Existing learning notes: `LadderMan/docs/learning/humanoid_rl_progress.md`
- Current reference task configs: `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`, `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`

Files changed:
- `LadderMan/AGENTS.md`
- `LadderMan/docs/implementation_log.md`
- `LadderMan/docs/learning/humanoid_rl_progress.md`

Verification:
- [ ] Review the new guidelines before starting the first LadderMan implementation task.

Remaining risks / next checks:
- Convert future repeated lessons into `AGENTS.md` rules only after they prove durable.

## 2026-06-23 - Ladder Extension Skeleton and WBT Decoupling

Goal:
- Create a first-stage LadderMan IsaacLab extension under `Ladder/`.
- Implement a fixed Ladder A procedural scene, local G1 config/assets, hybrid contact target helpers, and state-based tracking MDP terms.
- Keep `whole_body_tracking` as a reference only, with no Ladder runtime inheritance/import dependency.

References checked:
- Isaac Lab official: `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/mdp/`, `isaaclab_v2.1.0/source/isaaclab/isaaclab/scene/interactive_scene.py`, `isaaclab_v2.1.0/source/isaaclab/isaaclab/sim/spawners/`
- isaaclab_tasks: `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/`, `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/`
- whole_body_tracking / BeyondMimic: `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`, `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`, `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`

Files changed:
- `AGENTS.md`
- `Ladder/README.md`
- `Ladder/source/ladder/`
- `Ladder/scripts/rsl_rl/`
- `Ladder/tests/test_ladder_geometry.py`

Verification:
- [x] `python -m compileall Ladder/source/ladder`
- [x] Standalone Ladder A geometry checks
- [x] `rg "from whole_body_tracking|import whole_body_tracking|whole_body_tracking\\." Ladder/source/ladder Ladder/scripts Ladder/tests` returns no runtime references
- [x] Local G1 URDF/mesh assets copied under `Ladder/source/ladder/ladder/assets/unitree_description`
- [x] Local RSL-RL train/play scripts require `--motion_file` and import `ladder.tasks` directly
- [ ] IsaacLab import/config instantiation in a Python 3.10 IsaacLab environment
- [ ] Reset/step smoke test with a real LadderMan climbing `motion.npz`

Remaining risks / next checks:
- Current shell uses Python 3.13 and lacks `torch`/`toml`; IsaacLab 2.1.0 validation still needs the proper Python 3.10 IsaacSim/IsaacLab environment.
- Training/replay requires a climbing reference `motion.npz` with `contact_indicator` and `contact_rung_id`; the repo currently only has walking CSV examples.
- First IsaacLab smoke test should verify G1 asset conversion from the local copied Unitree description and that ladder cuboid collisions appear in each cloned env.

## 2026-06-23 - Ladder Script Layout Fix

Goal:
- Move Ladder entry-point scripts to the standard IsaacLab extension layout where `scripts/` and `source/` are siblings.
- Keep RSL-RL helpers local to Ladder so the project does not depend on `whole_body_tracking` scripts.

References checked:
- Isaac Lab official: `isaaclab_v2.1.0/scripts/reinforcement_learning/rsl_rl/`
- Existing Ladder extension files: `Ladder/README.md`, `Ladder/source/ladder/README.md`

Files changed:
- `Ladder/scripts/rsl_rl/`
- `Ladder/README.md`
- `Ladder/source/ladder/README.md`
- `docs/implementation_log.md`

Verification:
- [x] `python -m compileall Ladder/source/ladder Ladder/scripts`
- [x] README files point to `Ladder/scripts/rsl_rl/train.py`
- [x] Ladder source/scripts/tests have no runtime imports from the WBT reference project

Remaining risks / next checks:
- IsaacLab import/config instantiation and reset/step smoke tests still require the proper IsaacLab Python 3.10/IsaacSim environment plus a real LadderMan climbing `motion.npz`.

## 2026-06-23 - First-Version Contact Annotation Scope

Goal:
- Fix the first version scope to one upward motion, fixed reference timing, and successful single-pass expert training on Ladder A.
- Add a local preprocessing path that infers `contact_indicator` and `contact_rung_id` from a WBT/IsaacLab-style motion npz.
- Keep up/down commands and phase/time scaling out of the first version.

References checked:
- LadderMan paper: hybrid motion tracking uses manually annotated contact indicators and geometry-generated target contact positions.
- IsaacLab/WBT reference: `whole_body_tracking/scripts/csv_to_npz.py`, `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- Existing Ladder code: `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`, `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_geometry.py`

Files changed:
- `Ladder/source/ladder/setup.py`
- `Ladder/source/ladder/ladder/__init__.py`
- `Ladder/source/ladder/ladder/tasks/__init__.py`
- `Ladder/source/ladder/ladder/utils/contact_schema.py`
- `Ladder/source/ladder/ladder/utils/contact_annotation.py`
- `Ladder/scripts/preprocess_ladder_motion.py`
- `Ladder/tests/test_contact_annotation.py`
- `Ladder/README.md`
- `Ladder/source/ladder/README.md`

Verification:
- [x] `python -m compileall Ladder/source/ladder Ladder/scripts`
- [x] `import ladder; import ladder.tasks` does not require IsaacLab when only preprocessing modules are needed
- [ ] Contact annotation unit tests

Remaining risks / next checks:
- Current shell lacks `numpy` and `pytest`, so contact annotation tests and script execution still need the IsaacLab/NumPy environment.
- Automatic contact labels still need visual review on a real climbing motion to tune distance/speed thresholds.
- Runtime reset/step validation still needs the IsaacLab Python 3.10/IsaacSim environment.

## 2026-06-23 - Move Preprocessing Utilities

Goal:
- Keep data preprocessing utilities outside task definitions.
- Move contact annotation and shared contact schema to `ladder/utils/`, sibling to `tasks/` and `robots/`.

References checked:
- Existing Ladder package layout under `Ladder/source/ladder/ladder/`

Files changed:
- `Ladder/source/ladder/ladder/utils/`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/scripts/preprocess_ladder_motion.py`
- `Ladder/tests/test_contact_annotation.py`
- `docs/implementation_log.md`

Verification:
- [x] `python -m compileall Ladder/source/ladder Ladder/scripts`
- [x] No imports remain from `ladder.tasks.ladder_climb.contact_*`

Remaining risks / next checks:
- Contact annotation tests still need a Python environment with `numpy` and `pytest`.

## 2026-06-23 - Handrail Stair Asset

Goal:
- Add a temporary stair-with-handrail asset for collecting and validating handrail-assisted stair climbing data before the physical ladder is ready.
- Use primitive USD geometry so IsaacSim can load static collision shapes without mesh conversion.

References checked:
- Existing repository assets: `assets/ladders/ladder_A_aframe/wood_step_ladder_physics.usda`

Files changed:
- `assets/stairs/handrail_stairs_4step/handrail_stairs_4step.usda`
- `assets/stairs/handrail_stairs_4step/handrail_stairs_4step_notes.md`
- `Ladder/scripts/view_handrail_stairs.py`
- `docs/implementation_log.md`

Verification:
- [x] Asset dimensions encoded as four steps: height `0.20 m`, depth `0.40 m`, width `1.20 m`
- [x] Handrail cylinders encoded with diameter `0.08 m`; four vertical posts of length `1.20 m`; one sloped top rail
- [ ] Load asset in IsaacSim and verify collision/visual alignment with robot contact sensors

Remaining risks / next checks:
- The first asset places the handrail only on the `+Y` side; add mirrored rails later if the data collection setup needs both sides.

## 2026-06-23 - Handrail Stair Asset Revision

Goal:
- Make handrail posts measure `1.20 m` from each corresponding stair/platform surface.
- Widen/deepen the top stair platform to `1.00 m`.
- Add three extra top-platform handrail posts and a horizontal rail.

Files changed:
- `assets/stairs/handrail_stairs_4step/handrail_stairs_4step.usda`
- `assets/stairs/handrail_stairs_4step/handrail_stairs_4step_notes.md`
- `docs/implementation_log.md`

Verification:
- [x] Top platform scale changed to represent `1.00 m` depth
- [x] All vertical handrail posts remain `1.20 m` tall with `0.08 m` diameter
- [x] Sloped stair rail and horizontal platform rail use `0.08 m` diameter cylinders

Remaining risks / next checks:
- Load in IsaacSim to visually confirm the extra platform handrail placement matches the data collection setup.

## 2026-07-01 - IsaacLab Ladder and Robot Viewer

Goal:
- Add a small IsaacLab-based viewer that loads the Ladder A USD asset and the CR1-B2 URDF robot in one scene.
- Keep this as an asset-placement check, separate from the RL task/environment.

References checked:
- Isaac Lab official: `isaaclab_v2.1.0/scripts/tutorials/00_sim/spawn_prims.py`, `isaaclab_v2.1.0/scripts/demos/bipeds.py`, `isaaclab_v2.1.0/source/isaaclab/isaaclab/sim/spawners/from_files/`
- Existing project reference: `Ladder/scripts/view_handrail_stairs.py`, `scripts/isaacgym_load_ladder_robot.py`
- Ladder asset notes: `assets/ladders/ladder_A_aframe/wood_step_ladder_notes.md`

Files changed:
- `Ladder/scripts/view_ladder_robot.py`
- `docs/implementation_log.md`

Verification:
- [x] `python -m py_compile Ladder/scripts/view_ladder_robot.py`
- [x] `python -m compileall Ladder/source/ladder Ladder/scripts`
- [ ] Open the viewer in an IsaacLab/IsaacSim Python environment.

Remaining risks / next checks:
- Current shell uses `/home/zhangdawei/miniforge3/bin/python` and cannot import `isaaclab`, so the viewer still needs a real IsaacLab environment test.

## 2026-07-01 - Ladder A Asset Dimension Update

Goal:
- Update the A-frame step ladder asset dimensions to match the requested platform/step geometry.
- Keep this scoped to the generated asset files; do not change the separate procedural ladder scaffold used by the RL task.

References checked:
- Existing asset generator: `assets/ladders/ladder_A_aframe/generate_wood_step_ladder_model.py`
- Existing asset notes: `assets/ladders/ladder_A_aframe/wood_step_ladder_notes.md`
- Procedural RL scaffold checked but not changed: `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_geometry.py`

Files changed:
- `assets/ladders/ladder_A_aframe/generate_wood_step_ladder_model.py`
- `assets/ladders/ladder_A_aframe/wood_step_ladder.yaml`
- `assets/ladders/ladder_A_aframe/wood_step_ladder.usda`
- `assets/ladders/ladder_A_aframe/wood_step_ladder_physics.usda`
- `assets/ladders/ladder_A_aframe/wood_step_ladder_sim.urdf`
- `assets/ladders/ladder_A_aframe/wood_step_ladder.urdf`
- `assets/ladders/ladder_A_aframe/wood_step_ladder.stl`
- `assets/ladders/ladder_A_aframe/wood_step_ladder.obj`
- `assets/ladders/ladder_A_aframe/wood_step_ladder.mtl`
- `assets/ladders/ladder_A_aframe/wood_step_ladder.scad`
- `assets/ladders/ladder_A_aframe/wood_step_ladder_notes.md`
- `docs/implementation_log.md`

Verification:
- [x] Regenerated ladder asset files from `generate_wood_step_ladder_model.py`
- [x] `python -m py_compile assets/ladders/ladder_A_aframe/generate_wood_step_ladder_model.py`
- [x] Confirmed 7 step cubes plus `base_platform` in `wood_step_ladder_physics.usda`
- [x] Confirmed first step top height `0.19 m` and step top heights `[0.19, 0.42, 0.65, 0.88, 1.11, 1.34, 1.57]`

Remaining risks / next checks:
- IsaacLab visual check still needs an environment where `isaaclab` imports successfully.
- The RL task currently uses its own procedural ladder geometry; synchronize it later only when that task should train against this A-frame step ladder asset.

## 2026-07-01 - Ladder A Base Platform Length Fix

Goal:
- Change the A-frame ladder base platform from step-depth-sized to a full support platform: `0.64 m` wide, `1.04 m` long, `0.16 m` high.

References checked:
- `assets/ladders/ladder_A_aframe/generate_wood_step_ladder_model.py`
- `assets/ladders/ladder_A_aframe/wood_step_ladder_physics.usda`

Files changed:
- `assets/ladders/ladder_A_aframe/generate_wood_step_ladder_model.py`
- Regenerated `assets/ladders/ladder_A_aframe/wood_step_ladder.*`
- `docs/implementation_log.md`

Verification:
- [x] Regenerated ladder asset files from the generator.
- [x] Confirmed `base_platform` scale in `wood_step_ladder_physics.usda` is `(0.32000000, 0.52000000, 0.08000000)`, giving real dimensions `0.64 m x 1.04 m x 0.16 m`.
- [x] Confirmed the scene still has 7 step cubes.

## 2026-07-01 - Remove Step Warning Strips and Align First Step

Goal:
- Remove the decorative front warning strips from the ladder asset so they cannot affect contacts.
- Align the first step front edge with the base platform front edge at `y=0`.
- Move the remaining steps consistently with the same surface offset while preserving requested step heights.

References checked:
- `assets/ladders/ladder_A_aframe/generate_wood_step_ladder_model.py`
- Generated USD primitive ranges in `assets/ladders/ladder_A_aframe/wood_step_ladder_physics.usda`

Files changed:
- `assets/ladders/ladder_A_aframe/generate_wood_step_ladder_model.py`
- Regenerated `assets/ladders/ladder_A_aframe/wood_step_ladder.*`
- `docs/implementation_log.md`

Verification:
- [x] Confirmed no `front_warning_strip` or `warning_strip` remains in generated USD/URDF/OBJ/MTL/SCAD assets.
- [x] Confirmed `base_platform` y range is `[-1.040, 0.000]`.
- [x] Confirmed `step_01` y range is `[-0.130, 0.000]`, z range is `[0.160, 0.190]`.
- [x] Confirmed the scene still has 7 step cubes.

## 2026-07-01 - Add Front Slope Measurement Guides

Goal:
- Align the front slope reference triangle with the first step front-bottom corner at `y=0, z=0.16`.
- Add visual-only colored guides for the triangle: green run, blue height, red hypotenuse.
- Encode the measured tread front-edge spacing as `0.135 m` along the y direction.

References checked:
- `assets/ladders/ladder_A_aframe/generate_wood_step_ladder_model.py`
- Generated primitive ranges in `assets/ladders/ladder_A_aframe/wood_step_ladder_physics.usda`
- Existing IsaacLab viewer: `Ladder/scripts/view_ladder_robot.py`

Files changed:
- `assets/ladders/ladder_A_aframe/generate_wood_step_ladder_model.py`
- Regenerated `assets/ladders/ladder_A_aframe/wood_step_ladder.*`
- `docs/implementation_log.md`

Verification:
- [x] `python -m py_compile assets/ladders/ladder_A_aframe/generate_wood_step_ladder_model.py`
- [x] Confirmed `step_01` front edge is at `y=0`.
- [x] Confirmed step front-edge y positions are `[0.000, -0.135, -0.270, -0.405, -0.540, -0.675, -0.810]`.
- [x] Confirmed guide lengths in `wood_step_ladder_physics.usda`: green `1.04 m`, blue `1.60 m`, red `1.908 m`.
- [x] Confirmed guide cubes have no `PhysicsCollisionAPI`.

Remaining risks / next checks:
- The front slope reference top is now at `z=1.76` because the measured `1.60 m` height starts from the platform/first-step bottom height `z=0.16`.
- Rear support leg placement was not redesigned in this pass; visually check it in IsaacSim before using this asset for training contacts.

## 2026-07-13 - CR1 Holosoma Motion Export and Contact Labels

Goal:
- Standardize the current ladder reference pipeline on the CR1 wrist-fixed sphere-hand 25 DOF asset.
- Export Holosoma `qpos` retarget data into a Ladder/WBT-style motion npz with IsaacLab body tensors.
- Generate a first automatic CR1 contact-label pass for `downsample4_half_footstick`.

References checked:
- Isaac Lab official: `isaaclab_v2.1.0/source/isaaclab/isaaclab/scene/interactive_scene.py`, `isaaclab_v2.1.0/scripts/demos/markers.py`
- whole_body_tracking / BeyondMimic: `whole_body_tracking/scripts/csv_to_npz.py`, `whole_body_tracking/scripts/replay_npz.py`
- Existing Ladder code: `Ladder/scripts/preprocess_ladder_motion.py`, `Ladder/source/ladder/ladder/utils/contact_annotation.py`, `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`

Files changed:
- `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`
- `assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.xml`
- `Ladder/scripts/view_ladder_robot.py`
- `Ladder/scripts/export_holosoma_cr1_motion.py`
- `Ladder/scripts/preprocess_ladder_motion.py`
- `Ladder/source/ladder/ladder/utils/contact_schema.py`
- `Ladder/source/ladder/ladder/utils/contact_annotation.py`

Verification:
- [x] `python -m compileall Ladder/source/ladder Ladder/scripts` in the `wbt` environment
- [x] CR1 25 DOF URDF imported by the export script and produced `joint_pos` shape `(728, 25)`
- [x] Exported `Ladder/motions/climing_ladder_cr1_downsample4_motion.npz` with finite joint/body tensors and CR1 limb body names
- [x] Generated `Ladder/motions/climing_ladder_cr1_downsample4_motion_annotated_d025.npz` with `contact_indicator`, `contact_rung_id`, `contact_nearest_distance`, and `contact_speed`
- [ ] Contact annotation unit tests, because the active `wbt` environment does not currently have `pytest`
- [ ] Reference replay visualization of limb points, rung ids, and geometry-derived `P_hat`

Remaining risks / next checks:
- The first automatic labels use a broad `0.25 m` distance threshold because CR1 foot bodies are offset from the procedural targets; replay visualization must verify whether foot contacts on rungs `[1, 2]` are meaningful.
- IsaacSim shutdown can hang after writing the export file; the saved npz was verified, but the script may need cleaner app shutdown later.
- The RL environment is still G1-based and should not consume CR1 motion until a CR1 task config is added.

## 2026-07-13 - CR1 Reference Replay Viewer

Goal:
- Add an IsaacLab replay script for annotated CR1 ladder reference motions.
- Visualize CR1 limb positions and geometry-derived contact targets so `contact_indicator`, `contact_rung_id`, and `P_hat` can be checked before training.

References checked:
- Isaac Lab official: `isaaclab_v2.1.0/scripts/demos/markers.py`
- whole_body_tracking / BeyondMimic: `whole_body_tracking/scripts/replay_npz.py`
- Existing Ladder code: `Ladder/scripts/view_ladder_robot.py`, `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_geometry.py`

Files changed:
- `Ladder/scripts/replay_ladder_reference.py`
- `docs/implementation_log.md`

Verification:
- [x] `python -m py_compile Ladder/scripts/replay_ladder_reference.py`
- [x] `python -m compileall Ladder/source/ladder Ladder/scripts`
- [x] Headless smoke without markers/ladder asset replayed 5 frames from `climing_ladder_cr1_downsample4_motion_annotated_d025.npz`
- [ ] GUI replay with ladder asset and contact markers for visual inspection

Remaining risks / next checks:
- Headless marker visualization appears to stall in this environment, so marker correctness should be checked in GUI mode.
- The displayed A-frame ladder asset and the procedural rung geometry used for `P_hat` are not guaranteed to match perfectly; visual replay should decide whether to adjust `ladder_origin`, target offsets, or the contact body choices.
- IsaacSim shutdown can still hang after successful smoke checks and may need a cleaner shutdown path later.

## 2026-07-13 - Replay Ladder Asset Orientation Fix

Goal:
- Align the displayed A-frame ladder asset with the CR1 reference motion/P_hat coordinate convention.

References checked:
- Ladder asset notes: `assets/ladders/ladder_A_aframe/wood_step_ladder_notes.md`
- Generated USD primitive positions: `assets/ladders/ladder_A_aframe/wood_step_ladder_physics.usda`
- Current motion/contact metadata: `Ladder/motions/climing_ladder_cr1_downsample4_motion_annotated_d025.npz`

Files changed:
- `Ladder/scripts/replay_ladder_reference.py`
- `Ladder/scripts/view_ladder_robot.py`
- `docs/implementation_log.md`

Verification:
- [x] `python -m py_compile Ladder/scripts/replay_ladder_reference.py Ladder/scripts/view_ladder_robot.py`
- [x] `python -m compileall Ladder/source/ladder Ladder/scripts`

Remaining risks / next checks:
- Replay now defaults the displayed ladder to `x=0.55`, `yaw=90 deg` because the asset climbs along local `-Y` while the CR1 motion/P_hat climb along world `+X`.
- GUI visual inspection should confirm whether `ladder_x`, `ladder_z`, or the procedural `ladder_origin` need further adjustment.
## 2026-07-17 - Make CR1 Annotation Grounding Explicit

- Changed `Ladder/scripts/preprocess_ladder_motion.py` to preserve the source
  robot trajectory by default instead of automatically shifting every CR1
  `body_pos_w` trajectory to the legacy `-0.03 m` sole target.
- Added the opt-in `--ground_initial_robot` flag. The existing
  `--initial_foot_target_z` argument now requires this flag so a requested
  target cannot be silently ignored.
- Kept the underlying contact-annotation grounding utility for datasets that
  require an explicit global Z correction.
- Reference checked: `whole_body_tracking/.../tasks/tracking/mdp/commands.py`
  consumes the source `body_pos_w` directly and does not apply hidden
  preprocessing-time grounding.

Verification:
- [x] `python -m py_compile` passed for the preprocessing and annotation modules.
- [x] `Ladder/tests/test_contact_annotation.py`: 10 tests passed.
- [x] Default CLI preprocessing wrote no grounding metadata and preserved all
  source body positions exactly (`max delta = 0.0 m`).

## 2026-07-17 - Export New Collision-Constrained CR1 Reference

Goal:
- Convert the 501-frame Holosoma result produced with training-aligned robot
  colliders and 18-box ladder collision into a Ladder training reference while
  preserving the source robot-to-ladder transform.

Implementation:
- Added `Ladder/scripts/export_holosoma_cr1_motion_offline.py`, which reconstructs
  Ladder joint/body states with the current CR1 training URDF and the existing
  offline FK utilities. This is the deterministic fallback when IsaacSim export
  cannot start because no GPU/Vulkan device is available.
- Exported the source at its actual 50 Hz rate without smoothing or global
  grounding, then regenerated sphere-hand/virtual-sole contact annotations.

Outputs:
- `Ladder/motions/climing_ladder_cr1_training_collision_front_clearance010_50fps_motion.npz`
- `Ladder/motions/climing_ladder_cr1_training_collision_front_clearance010_50fps_annotated.npz`

Verification:
- [x] Offline FK matches an existing IsaacLab export within `2e-6 m`.
- [x] Final shapes: joint `(501, 25)`, body `(501, 42, 3)`, contact `(501, 4)`.
- [x] Required training fields are finite; quaternion norm error is below `1.2e-7`.
- [x] Annotation preserved body positions exactly and wrote no grounding metadata.
- [x] Contact rungs: left hand `[3,4,5,6]`, right hand `[4,5,6]`, both feet `[0,1,2]`.

## 2026-07-17 - Audit New Reference Penetration and Reset Frames

- Checked the 501-frame annotated reference against the current CR1 training
  collision URDF, the 18-box ladder, and the training ground at `z=-0.03 m`.
- At a strict `1 mm` penetration tolerance, `346/501` frames (69.1%) are valid
  reset starts. The worst penetration is `11.56 mm` at frame 499 between the
  left ankle/foot mesh and step 3.
- All `155` strict-invalid frames are foot/step contacts. No hand, knee, torso,
  or ground pair exceeds `1 mm`.
- At `5 mm`, `468/501` frames are valid; at `10 mm`, `493/501` are valid.
- Sidecars:
  - `Ladder/motions/climing_ladder_cr1_training_collision_front_clearance010_50fps_reset_validity_tol001.npz`
  - `Ladder/motions/climing_ladder_cr1_training_collision_front_clearance010_50fps_reset_validity_tol005.npz`
  - `Ladder/motions/climing_ladder_cr1_training_collision_front_clearance010_50fps_reset_validity_tol010.npz`
- Remaining integration: `LadderMotionCommand` does not yet consume a reset-valid
  mask, so adaptive reset sampling still considers all motion frames.

## 2026-07-17 - Collision-Aware Smoothing of New CR1 Reference

- Rejected ordinary Butterworth candidates: strict reset-valid frames fell from
  `346` to `206-228/501`, with new hand and knee penetrations up to `31 mm`.
- Added `blend_cr1_motion_collision_aware.py`. It blends each processed frame
  toward the source only while every robot/environment collision pair stays
  below `1 mm` or does not exceed that pair's source penetration.
- Added `--leg_only_xy` to the contact stabilizer for root-preserving stance-foot
  XY correction. The accepted comparison uses conservative collision-aware
  smoothing plus bounded initial/contact stabilization.
- Output:
  - `Ladder/motions/climing_ladder_cr1_training_collision_front_clearance010_50fps_collisionaware_smooth_motion.npz`
  - `Ladder/motions/climing_ladder_cr1_training_collision_front_clearance010_50fps_collisionaware_smooth_annotated.npz`
  - `Ladder/motions/climing_ladder_cr1_training_collision_front_clearance010_50fps_collisionaware_smooth_reset_validity_tol001.npz`
- Joint acceleration RMS decreases `17.79 -> 13.62 rad/s^2`; root acceleration
  RMS decreases `4.44 -> 2.77 m/s^2`.
- Fixed-contact-phase sole XY path decreases `110.14 -> 107.13 cm`. This is a
  modest improvement because collision limits reject larger stance corrections.
- Strict reset-valid frames improve `346 -> 365/501`; worst penetration changes
  `11.56 -> 11.55 mm`, with no new collision pair exceeding `1 mm`.
- Verification: collision/contact scripts compile and four contact-stabilization
  tests pass.

## 2026-07-17 - Prepare Collision-Aware CR1 Reference for Baseline Training

- Embedded the strict `1 mm` collision-screening reset fields into a separate
  train-ready motion file; the motion, contact annotations, and collision model
  are unchanged.
- `LadderMotionLoader` now validates an optional boolean `reset_valid_mask`.
  Adaptive reset sampling distributes each bin's probability only across valid
  frames in that bin. Files without a mask remain backward-compatible and use
  all frames; `fixed_reference_frame` remains unrestricted for diagnostics.
- Added `--disable_randomization` to the local RSL-RL training entry point. It
  disables startup events and reset pose/joint perturbations for baseline runs
  without deleting the configured domain randomization.
- References checked:
  - `whole_body_tracking/.../tasks/tracking/mdp/commands.py`
  - `isaaclab_tasks/.../locomotion/velocity/config/g1/flat_env_cfg.py`

Verification:
- [x] Train-ready NPZ contains the 365/501 strict reset mask, and every
  pre-existing motion/contact field exactly matches the accepted annotated file.
- [x] All 33 Ladder CPU tests pass, including reset-mask validation, adaptive
  bin weighting, and invalid-frame exclusion.
- [x] Python compileall and the training CLI help check pass.
- [ ] IsaacLab GPU reset/step smoke test.
- [ ] 64-128 environment baseline training with randomization disabled.

Remaining risk:
- Holosoma optimized against its primitive robot colliders, while training and
  strict reset screening use the physical-shape mesh CR1 URDF by design. The
  offline mask removes known penetrating reset poses, but only the GPU smoke
  test can validate immediate PhysX contacts and terminations.

## 2026-07-17 - IsaacLab 5.1 Training-Entry Compatibility

- The GPU environment uses a newer IsaacLab whose `isaaclab.utils.io` package
  exports `dump_yaml` but no longer exports `dump_pickle`.
- Added a local pickle fallback in `Ladder/scripts/rsl_rl/train.py`; older
  IsaacLab versions still use their native helper. This only affects saving
  configuration snapshots after environment creation.
- Made `LadderGeometry` a mutable dataclass so IsaacLab 5.1 Hydra can apply its
  nested `phi/rung_spacing/...` config dictionary without a frozen-instance
  error. Its defaults and geometry formulas are unchanged.
- Updated `ContactAnnotationCfg` to use a dataclass `default_factory` for the
  now-mutable geometry default; annotation behavior is unchanged.

## 2026-07-17 - rsl-rl 5 Runner Configuration Compatibility

- The installed `rsl-rl-lib 5.0.1` expects top-level `actor` and `critic`
  `MLPModel` dictionaries, while the IsaacLab config class still emits the
  legacy `policy` ActorCritic dictionary.
- Added `Ladder/scripts/rsl_rl/rsl_rl_compat.py` and enabled it only when the
  installed package exposes the new `rsl_rl.models.MLPModel` API. Network sizes,
  activation, Gaussian initial noise, normalization, and PPO hyperparameters
  are preserved.
- The converted actor and critic both read the `policy` observation group, as
  required for the current identical actor/critic observation design.
- Training disables command/contact debug markers; replay and the dedicated
  smoke script retain visualization controls.
- Added pure CPU regression tests for the legacy-to-new runner config mapping.
- Randomization disabling now clears individual event terms while retaining the
  EventCfg object, avoiding IsaacLab 5.1's callback traceback for `events=None`.

## 2026-07-21 - CR1 Sphere-Hand Tracking and Phase-Aware Contact Rewards

- Added `left_sphere_hand_link` and `right_sphere_hand_link` to the CR1 motion
  command body list and upper-body position tracking. The accepted train-ready
  NPZ already contains both bodies, so no motion conversion is required.
- Removed the 12-D `target_contact_pos_b` term from actor observations. Added
  the existing 12-D `contact_error_b` only to the privileged critic group.
  With two additional tracked bodies, the expected actor observation changes
  from 592-D to 610-D and the critic observation is 622-D.
- Tightened only the hand contact-position reward temperature from `0.3` to
  `0.05`; the foot reward remains at `0.3`. This gives a `1/e` distance of
  `sqrt(0.05) = 22.36 cm`: the exponential kernel is `95.1%` at `5 cm`,
  `81.9%` at `10 cm`, and `71.7%` at the observed `12.9 cm` failure error.
- Added three sphere-hand contact terms for CR1:
  - swing-phase contact penalty above `5 N` (`weight=-0.2`);
  - downward world-Z force penalty below `-5 N` as a low-complexity proxy for
    hitting a horizontal step underside (`weight=-0.2`);
  - required-contact upward world-Z force reward, linear from `5 N` and
    saturated at `50 N` (`weight=0.3`).
- These phase-aware terms use the current net contact force instead of the
  six-sample history, avoiding stale contact penalties for roughly 120 ms after
  a contact-indicator transition under the default lazy sensor update.
- The underside term identifies force direction, not the exact contacted rung.
  Exact collision-pair attribution would require a filtered ladder contact
  sensor and is intentionally deferred.
- RSL-RL now maps the actor to the `policy` group and the critic to the
  privileged `critic` group. Existing 592-D checkpoints, including
  `model_6999.pt`, are incompatible and must not be resumed.

References checked:
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`

Verification:
- [x] Modified Python files compile.
- [x] All 38 Ladder CPU tests pass, including phase gating, force direction,
  threshold, saturation, and inactive-phase force exclusion.
- [x] Train-ready NPZ contains both sphere-hand body trajectories.
- [ ] GPU reset/step smoke test. The coding container cannot expose a CUDA or
  Vulkan device; rerun the existing CR1 smoke script in the GPU terminal.

## 2026-07-21 - Fixed-Start Reset Mixture and Per-Hand Reward Diagnostics

- Added a generic fixed-frame/adaptive reset mixture to `LadderMotionCommand`.
  CR1 forces exactly 20% of resets to frame 0; the remaining 80% uses adaptive
  sampling with frame 0 excluded from that branch. `fixed_reference_frame`
  diagnostics still bypass the mixture.
- Reject an enabled fixed start when the selected frame is outside the motion,
  excluded by `reset_valid_mask`, or cannot be mixed with another valid frame.
- Added the `fixed_start_episode` command metric to identify episodes created by
  the fixed-start branch.
- Split all hand rewards into left/right TensorBoard terms while preserving the
  previous total reward exactly:
  - contact-position terms use weights `0.5 + 0.5`;
  - swing and downward-contact penalties use `-0.1 + -0.1`;
  - upward-force terms retain weight `0.3` each and share the old denominator,
    the number of currently required hands.
- Added terminal metrics for left/right required-contact error and whether each
  hand is required at termination.

Current motion provenance:
- Train-ready motion metadata points to
  `holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_results/cr1_wrist_fixed_spherehand/climbing/ladder_training_collision_front_clearance010_full/climing_ladder_mirror_x_trim_v150_650_down008_original.npz`.
- Its human input is
  `holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb_ladder/climing_ladder_mirror_x_trim_v150_650_down008/climing_ladder_mirror_x_trim_v150_650_down008_joint_positions.npy`.
- That 2004-frame 200 Hz input is exactly full human frames `[600:2604]`
  shifted by `(0, 0, -0.08) m`. The untrimmed 5822-frame source is
  `holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb_ladder/climing_ladder_mirror_x/climing_ladder_mirror_x_joint_positions.npy`.
- Holosoma then downsamples by 4 to produce the 501-frame 50 Hz retarget result.

Verification:
- [x] All 42 Ladder CPU tests pass.
- [x] Split upward-force rewards sum exactly to the previous combined reward for
  zero, one, or two required hands.
- [x] Fixed-frame sampling tests cover 0%, 20%, 100%, and invalid probabilities.
- [ ] Do not start the next full training run until the restored human prefix is
  reviewed and propagated through retarget, conversion, annotation, smoothing,
  collision screening, and the embedded reset mask.

## 2026-07-21 - Restored-Prefix Human Motion Candidates

- Preserved the original `climing_ladder.bvh`, the 5822-frame mirrored human
  keypoint source, and the accepted 501-frame training chain unchanged.
- Generated three separately named 200 Hz human-keypoint candidates from the
  mirrored source, retaining its existing right-hand offset and applying the
  established world-Z `-0.08 m` human/ladder alignment convention:
  - frames `[300:2604]`: 2304 frames, or 576 frames after stride 4;
  - frames `[2604:]`: 3218 frames, or 805 frames after stride 4;
  - frames `[300:]`: 5522 frames, or 1381 frames after stride 4.
- Verified that the first two candidates concatenate exactly to the third and
  that all arrays are finite.
- Deferred deletion of rejected Holosoma/Ladder artifacts until the new
  `[300:2604]` candidate is visually accepted and the deletion scope is named
  explicitly; the current train-ready reference remains reproducible.

## 2026-07-21 - Isolated 576-Frame Restored-Prefix Retarget

- Archived the original BVH, the full mirrored 22-keypoint source, and all
  three restored-prefix candidates under
  `demo_data/climb_ladder/restored_prefix_v300/`.
- Built a self-contained `[300:2604]` Holosoma task with corrected relative
  scene paths, the training-aligned 32-collider CR1 model, and 18-box ladder.
- Reused the accepted retarget parameters and shifted all former 501-frame
  hand swing windows by 75 frames.
- Completed all 576 frames without infeasibility; output is isolated under
  `demo_results/restored_prefix_v300/` and contains finite `(576, 32)` qpos.
- Offline collision audit finds a worst `8.51 mm` foot/step penetration across
  104 frames below `-0.5 mm`; no hand, knee, or torso pair exceeds that level.
- The right sphere-hand front-plane margin misses by at most `0.54 mm` in one
  sample; the left hand satisfies the margin within numerical tolerance.
- The longer clip changes Holosoma's toe-min grounding, raising the overlapping
  501 human-target frames by `4.01 mm`. It also exposes a frame-0 IK velocity
  endpoint spike, so collision-aware Ladder smoothing remains required after
  visual acceptance.

## 2026-07-21 - Restored-Prefix Ladder Processing Candidate

- Converted the isolated 576-frame Holosoma result to the 25-joint, 42-body
  Ladder motion schema at the actual 50 Hz rate without robot grounding.
- Reused the accepted 501-frame processing parameters exactly: contact speed
  threshold `0.35 m/s`, four-frame minimum contacts, two-frame gap filling,
  Butterworth cutoffs `4/2.5/3/4 Hz` for non-leg joints, leg joints, root
  position, and root orientation, joint-limit projection, collision-aware
  blending at `1 mm`, and bounded frame `0-84` initial right-stance refinement.
- Kept every new Ladder artifact isolated under
  `Ladder/motions/restored_prefix_v300/`; the accepted 501-frame files and
  training defaults are unchanged.
- Final candidate has 576 finite frames, 25 joints, 42 bodies, and an exact
  merge of motion/contact fields with the strict reset sidecar. Joint
  acceleration RMS changes `15.64 -> 12.39 rad/s^2`, and root acceleration RMS
  changes `3.49 -> 2.50 m/s^2` after collision-limited blending.
- Strict reset screening accepts `495/576` frames. The first valid frame is 5
  and the last is 543. Worst remaining penetration is `20.51 mm` at frame 0,
  between `right_ankle_x_link` and the ground.
- The candidate must not replace the default training motion yet: the current
  CR1 reset configuration deliberately sends 20% of episodes to frame 0, but
  frame 0 is invalid in this candidate. Replay it first, then repair or replace
  the initial pose before a full training run.

Verification:
- [x] All 42 Ladder CPU tests pass.
- [x] Ladder source and scripts compile.
- [x] Final train-ready NPZ is finite and exactly preserves every field from
  its accepted annotated motion and reset-validity sidecar.
- [ ] Visual replay of the restored prefix.
- [ ] GPU reset/step smoke test after frame 0 is made valid.

## 2026-07-21 - Isaac Lab 2.1 Replay Import Compatibility

- Restored `wbt` replay compatibility by falling back from the newer
  `quat_apply_inverse` name to Isaac Lab 2.1's mathematically equivalent
  `quat_rotate_inverse` only when the newer API is unavailable.
- Observation values, dimensions, and the Isaac Sim 5.1 training path are
  unchanged.
- Verification: the `wbt` ladder-geometry import passes, the modified module
  compiles, and all 42 Ladder CPU tests pass.

## 2026-07-22 - Stride-2 Slow-50 Reference Candidate

- Retargeted the isolated human range `[300:2604]` with Holosoma
  `motion_downsample=2`, doubling every hand swing window to preserve the same
  source-time phases. The 1152-frame Holosoma run completed with cost `0.698`.
- Intentionally interpreted the result at `50 Hz` in Ladder. The resulting
  reference is `1152` frames over `23.02 s`, approximately half the original
  speed while retaining intermediate retargeted poses.
- Reused the accepted 501-frame Ladder pipeline exactly: offline FK, contact
  annotation, `4/2.5/3/4 Hz` smoothing, collision-aware blending, and initial
  right-stance stabilization. New files are isolated under
  `Ladder/motions/restored_prefix_v300_stride2_slow50/`.
- Final candidate has `1028/1152` strict reset-valid frames. Joint acceleration
  RMS changes `8.69 -> 6.92 rad/s^2`; root acceleration RMS changes
  `1.46 -> 1.08 m/s^2`. All required fields are finite and the train-ready
  merge is exact.
- Frame 0 remains invalid with about `23.3 mm` right-ankle/ground penetration;
  valid reset sampling starts at frame 8. Do not enable the existing 20% fixed
  frame-0 reset branch or change the default motion until the initial pose is
  repaired or the fixed-start policy is changed for this candidate.
- The 23-second reference also requires `episode_length_s >= 24` before
  training; the default 12-second environment was intentionally left unchanged
  pending visual acceptance.

## 2026-07-22 - Slow-50 Outlier Repair and Standing Start

- Audited the 1152-frame slow-50 candidate and traced its largest discontinuities
  to retarget clearance-window boundaries. In particular, frame 772 moved the
  root about `35 mm` laterally in one sample and returned on the next sample.
- Applied collision-aware stronger smoothing plus neighbor interpolation only
  at five confirmed one-frame outliers (`442, 604, 772, 791, 940`). Direct
  unbounded smoothing was rejected because it introduced up to `43 mm`
  sphere-hand/step penetration.
- Added a 25-frame (`0.5 s`) zero-joint CR1 standing segment. Its root keeps
  only the reference yaw (`-2.47 deg`), forces roll/pitch to zero, and uses
  root height `0.8744 m`; both sole contact points are exactly on the training
  ground at world Z `-0.03 m`. It is followed by a 50-frame (`1.0 s`) quintic
  transition into repaired source frames `8:58`. A transition-only root-Z
  clearance envelope removes temporary foot/ground penetration and is exactly
  zero during the 25-frame hold.
- Replaced the unsafe final 66-frame penetration run with a time-warped terminal
  slowdown: safe source frames `1002:1102` are expanded from 100 to 150
  intervals with velocity decreasing continuously to zero, followed by a
  24-frame stationary hold.
- Final output is `1177 @ 50 Hz` (`23.52 s`) under
  `Ladder/motions/restored_prefix_v300_stride2_slow50/smooth_start/`.
  Frame 0 is reset-valid; `1140/1177` frames pass the 1 mm screen and worst
  remaining penetration is `2.49 mm`.
- Relative to the first slow-50 candidate, maximum joint speed changes
  `12.97 -> 5.71 rad/s`, maximum joint acceleration `263 -> 134 rad/s^2`,
  maximum root speed `0.97 -> 0.37 m/s`, and maximum root acceleration
  `45.6 -> 5.49 m/s^2`.

## 2026-07-22 - Slow-50 Ground-Stance Foot Drift Repair

- Traced the visible foot/leg drift near frame 150 to an annotation gap: both
  feet are still supported by the ground, but `contact_indicator` intentionally
  describes only ladder contacts, so the earlier ladder-contact stabilization
  did not constrain this interval.
- Kept the accepted climbing trajectory and collision geometry unchanged.
  Shifted only the default standing root XY by `(0.12830, 0.00623) m`, then
  applied leg-only IK to hold the left/right sole XY positions during their
  initial ground stance. The constraints fade out before the first left/right
  foot lifts at approximately frames 228 and 374 respectively.
- The frame 100-200 sole XY path decreases from `9.25 -> 0.98 cm` on the left
  and `12.72 -> 0.89 cm` on the right. The root retains its intended `11.35 cm`
  XY motion over this interval; the legs now compensate instead of dragging
  the planted soles.
- Updated the canonical terminal motion, regenerated contact annotation and the
  strict 1 mm reset sidecar, and rebuilt the compressed train-ready file under
  `Ladder/motions/restored_prefix_v300_stride2_slow50/smooth_start/`.
- Aligned the replay ground plane with training by placing it at
  `CR1_INITIAL_SOLE_TARGET_Z = -0.03 m`. The replay previously displayed the
  plane at world Z zero even though the CR1 training environment and reset
  detector both use `-0.03 m`.

References checked:
- `Ladder/scripts/stabilize_cr1_contact_motion.py`
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`
- `Ladder/scripts/replay_ladder_reference.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_env_cfg.py`

Verification:
- [x] The first 25 frames remain a zero-joint, zero-velocity standing hold;
  both sole points are exactly at world Z `-0.03 m` and root roll/pitch are zero.
- [x] Strict reset screening remains `1140/1177` valid frames with the same 37
  invalid frames. Worst penetration remains `2.49 mm` at frame 787 between the
  right ankle and step 2; frames 100-200 are all valid.
- [x] The train-ready NPZ is finite and is an exact field-for-field merge of
  the regenerated annotation and reset sidecar.
- [x] Replay and preprocessing scripts compile; all 42 Ladder CPU tests pass.
- [x] Selected this repaired reference for the first training baseline; the CR1
  task already provides the required 25 s episode cap.

## 2026-07-22 - Select Slow-50 Reference for CR1 Training

- Selected the repaired 1177-frame slow-50 reference as the CR1 wood-step
  task default:
  `Ladder/motions/restored_prefix_v300_stride2_slow50/smooth_start/`
  `climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_train_ready.npz`.
  The training CLI still requires an explicit `--motion_file`, so each run log
  remains unambiguous and can override the task default.
- Synchronized the deterministic reset/step checker and reference replay
  defaults with the selected file. This prevents a no-argument diagnostic from
  silently checking the previous 501-frame reference.
- Kept the CR1 fallback episode timeout at `25.0 s`. The motion contains 1177
  samples at 50 Hz and reaches its last frame after `23.52 s`; the existing
  `motion_complete` term therefore ends it before the generic episode timeout.
- Preserved the 50 Hz control rate (`decimation=4`, simulation `dt=0.005 s`),
  the then-current 20% frame-0 plus 80% adaptive-valid reset mixture, all
  rewards, domain randomization ranges, PPO settings, and the physical
  robot/18-box ladder collision geometry. The reset mixture was explicitly
  changed to 20/40/40 in the later entry below.

References checked:
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/terminations.py`

Verification:
- [x] The selected NPZ is 1177 frames at 50 Hz, contains all 25 joints and 42
  bodies, embeds `1140/1177` reset-valid frames, and includes valid frame 0.
- [x] Its duration is below the 25 s episode cap and its FPS exactly matches
  the 50 Hz environment control rate.
- [x] Modified files compile and all 42 Ladder CPU tests pass.
- [ ] IsaacLab GPU reset/step smoke test. The coding container cannot expose a
  CUDA or Vulkan device; run the updated checker in the GPU terminal before
  starting the long baseline.

## 2026-07-22 - Anchor Observations and Mixed Reset Sampling

- Added `motion_anchor_pos_b` (3 values) and `motion_anchor_ori_b` (6 values)
  to the shared actor/critic observation group. This follows the local
  whole-body-tracking reference implementation and changes the expected
  concatenated dimensions from actor `610` / critic `622` to actor `619` /
  critic `631`. Existing checkpoints are intentionally incompatible; training
  must start from scratch.
- Replaced the old fixed-frame-plus-adaptive reset routing with an explicit
  mixture: `20%` fixed frame 0, `40%` uniform over non-fixed valid frames, and
  `40%` adaptive over the same non-fixed valid frames. Metrics now expose both
  the effective top-bin probability and the adaptive-only top-bin probability,
  so a concentrated adaptive distribution cannot hide behind the mixture.
- Set `motion_global_anchor_pos` tracking weight to `0.8`.
- Added reset-time base pose/velocity perturbations for CR1: XY position
  `+-0.01 m`, upward-only Z `0..0.01 m`, Euler orientation `+-0.02 rad`, linear
  velocity XY `+-0.05 m/s`, Z `+-0.02 m/s`, and angular velocity `+-0.05 rad/s`.
  `--disable_randomization` remains deterministic; `--base_reset_noise_only`
  disables physics/joint randomization while retaining these base perturbations.
- Preserved the physical CR1 collision model and 18-box ladder geometry.

References checked:
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/utils/motion_reset.py`

Verification:
- [x] Modified Python files compile.
- [x] `PYTHONPATH=Ladder/source/ladder ... -m pytest -q Ladder/tests`: `44 passed`.
- [ ] IsaacLab GPU reset/step smoke test: unavailable in this container (`no
  CUDA-capable device`). CPU fallback reached scene creation but could not
  fetch Isaac Sim's remote default USD asset.

## 2026-07-22 - Explicit Tracking-Progress Diagnostics

- Existing logs exposed absolute tracking errors, but did not directly show
  whether the robot moved with the advancing reference. Added episode-level
  command metrics for reference/robot anchor path length, net displacement,
  projected progress ratio, reference completion fraction, reference and robot
  speeds, joint speeds, and the fraction of reference-moving steps during
  which the robot anchor stayed below `0.03 m/s`.
- `progress_evaluable` marks episodes whose reference anchor moved more than
  `0.05 m`; `anchor_progress_ratio` is intentionally left un-clipped so wrong
  direction or overshoot remains visible. The movement detector uses reference
  speed `>0.05 m/s` and robot stationary speed `<0.03 m/s`.
- Fixed `reset_valid_fraction` being cleared to zero by the manager reset and
  never restored until now.
- Interactive play now prints reference and robot anchor speeds beside the
  existing absolute tracking errors.

Verification:
- [x] Modified Python files compile.
- [x] `PYTHONPATH=Ladder/source/ladder ... -m pytest -q Ladder/tests`: `44 passed`.

## 2026-07-24 - Per-Environment Local Ladder Geometry Randomization

- Preserved the accepted fixed-ladder task and data unchanged. `Ladder-Climb-CR1-v0`
  still uses the static `wood_step_ladder_physics.usda` and the original
  `...terminal_train_ready.npz`. Added a separate
  `Ladder-Climb-CR1-Geometry-DR-v0` task and a separate
  `smooth_start/ladder_geometry_dr/...terminal_train_ready_ladder_dr.npz` artifact.
  Its RSL-RL runs also use the separate `cr1_wood_step_ladder_geometry_dr`
  experiment directory, so fixed-ladder checkpoints are not mixed with DR runs.
- Replaced only the DR task's compound ladder asset with an IsaacLab
  `RigidObjectCollection` containing the same 18 primitive boxes. Collision and
  visual geometry share each cuboid prim. Per environment, reset samples ladder
  inclination `+-1 degree` and rung spacing `+-0.01 m`, then holds them fixed for
  the episode. The base platform remains grounded; step top centers exactly match
  the sampled rung centers; inclined supports rotate with the sampled inclination.
- Kept the reference motion completely unchanged. Randomization affects the ladder,
  geometry-derived contact targets, ladder-frame root observation, collision-valid
  reset selection, metrics, and debug markers. It does not rotate, scale, or move
  reference root/body poses or velocities.
- Added geometry-conditioned reset masks. The offline detector evaluates a 5x5
  inclination/spacing grid with the real CR1 collision URDF and 18 boxes. Each of
  the 16 parameter cells uses the intersection of its four vertex masks. Runtime
  first samples ladder geometry and then applies the corresponding cell mask to
  both uniform and adaptive reset sampling while preserving the 20/40/40 mixture.
- Debug visualization now includes sampled rung centers and current contact targets.
  Play diagnostics report sampled inclination/spacing. The smoke checker can retain
  ladder randomization across multiple environments and reports physical-rung versus
  command-rung synchronization error.
- `--disable_randomization` and `--base_reset_noise_only` explicitly set both ladder
  perturbation ranges to zero; this preserves their existing meanings.

References checked:
- `LadderMan.pdf`, Sec. 3.1 and Appendix A.1, especially Tables 3 and 5.
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/assets/rigid_object_collection/`
- `isaaclab_v2.1.0/source/isaaclab/test/assets/test_rigid_object_collection.py`
- `assets/ladders/ladder_A_aframe/wood_step_ladder_sim.urdf`
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`

Verification:
- [x] The 18-box parser returns 18 unique primitive boxes; zero perturbation exactly
  reproduces nominal poses; all seven sampled step top centers match command rung
  centers in CPU geometry tests.
- [x] The fixed source motion was not overwritten. The DR motion retains all source
  trajectory/contact fields; only reset detector metadata is regenerated/extended.
  All 11 loaded reference/contact arrays compare exactly equal, including the full
  joint and 42-body pose/velocity trajectories.
- [x] Nominal detection remains `1140/1177` valid with the same `2.49 mm` worst
  penetration at frame 787. The global four-corner intersection is `133/1177`, but
  conditioned cell masks retain `133-1144` frames (median `185.5`) and frame 0 is
  valid in all 16 cells.
- [x] Modified files compile and all 49 Ladder CPU tests pass.
- [ ] IsaacLab GPU reset/step smoke. The coding container reports
  `cudaErrorNoDevice`; run the updated multi-environment checker in the GPU terminal.
- [x] Fixed the first GPU scene-spawn failure: `RigidObjectCollection` children now
  use flat `{ENV_REGEX_NS}/Ladder_<box>` prim paths, matching the IsaacLab collection
  pattern and avoiding an uncreated regex parent prim at `{ENV_REGEX_NS}/Ladder`.

Known approximation:
- Long support-box dimensions are not resized per environment because collection
  shapes are shared; their centers/orientations follow the sampled state. This is
  bounded to the paper's local `+-1 degree` / `+-1 cm` expert randomization range.

## 2026-07-24 - Relaxed Reset-Mask Geometry-DR Experiment

- Added the separate `Ladder-Climb-CR1-Geometry-DR-Relaxed-Reset-v0` task and
  `cr1_wood_step_ladder_geometry_dr_relaxed_reset` RSL-RL experiment directory.
- Kept the reference and randomized 18-box ladder unchanged. The relaxed task only
  disables geometry-conditioned cell masks and uses the nominal ladder's embedded
  `1140/1177` reset-valid mask for every sampled ladder.
- Preserved the strict geometry-DR task as the default comparison. The relaxed task
  prints a startup warning because nominal-valid states may penetrate randomized
  ladders. At the `+1 degree/+1 cm` grid corner, `1007/1140` nominal-valid frames
  exceed the original `1 mm` threshold and the offline worst depth is about `60 mm`.

Verification:
- [x] Modified files compile and all 49 Ladder CPU tests pass.
- [ ] Run the GPU reset/step smoke and inspect reset contact-force transients before
  starting the relaxed-reset training comparison.

## 2026-07-24 - Relaxed Reset With 30 Percent Nominal Ladder Mixture

- Added `Ladder-Climb-CR1-Geometry-DR-Relaxed-Reset-Nominal30-v0` without changing
  the strict or fully randomized relaxed-reset tasks.
- On every environment reset, one joint Bernoulli mask assigns exactly nominal
  ladder geometry (`delta_phi=0`, `delta_spacing=0`) with probability `0.30`.
  The remaining `0.70` samples retain independent continuous uniform perturbations
  over `+-1 degree` and `+-0.01 m`.
- Added `Metrics/motion/nominal_ladder_geometry_episode` and a separate
  `cr1_wood_step_ladder_geometry_dr_relaxed_reset_nominal30` experiment directory.

Verification:
- [x] A 100,000-sample seeded check produced `0.29934` nominal samples, with both
  deltas jointly zero and the random branch retaining the complete configured range.
- [x] Modified files compile and all 52 Ladder CPU tests pass.

## 2026-07-24 - Visual Depth, Rung-Focused Masking, and CNN Task

- Added the separate `Ladder-Climb-CR1-Geometry-DR-Visual-v0` task without
  changing the fixed-ladder or geometry-DR expert tasks. The camera is attached
  to CR1's `d435_head_link` and produces metric depth plus rung-filtered semantic
  segmentation at the paper's `60 x 80` resolution.
- Added `class:rung` semantics to the existing physical/visual rung primitives.
  The preprocessing clips depth to `[0.1 m, 2.0 m]` and selects the two Figure 3
  branches independently per environment: rung-focused masking with probability
  `0.1`, or Gaussian noise `N(0, 0.005)` plus pixel dropout `0.05` with probability
  `0.9`. Invalid/masked pixels use the local value `0.0`.
- Added reset-time camera extrinsic randomization of `+-0.01 m` translation and
  `+-1.5 degrees` rotation. `--disable_randomization` and
  `--base_reset_noise_only` disable this event consistently with the other
  randomization terms.
- Added a three-layer depth encoder with paper-specified channels `[8, 8, 8]`,
  kernel size `5`, and a local `stride=2` choice because the paper does not report
  stride. It maps `(1, 60, 80)` depth to a 224-D latent, which is concatenated
  with 310-D deployable proprioception before the actor MLP
  `[2048, 1024, 512, 256, 128]`.
- Kept observation ownership explicit: the actor resolves only
  `visual_proprio + visual_depth`; the 631-D critic group remains privileged; the
  619-D `teacher` group is exposed for the later Hybrid DAgger stage but is not
  consumed by this PPO runner. Training and play automatically enable cameras for
  task IDs ending in `-Visual-v0`.
- Added `Ladder/scripts/view_cr1_visual_observations.py` to save a four-panel raw,
  Gaussian/dropout, rung-mask, and forced-RFM comparison to
  `outputs/cr1_visual_observations.png`.

References checked:
- `docs/LadderMan.pdf`, Sections 3.2-3.3, Figure 3, and Appendix A.2.
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_camera_env_cfg.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/tiled_camera.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_env_cfg.py`

Verification:
- [x] All 61 Ladder CPU tests pass, including augmentation branch isolation,
  encoder output shape, CNN backpropagation, and RSL-RL config conversion.
- [x] GPU capture returns `(1, 1, 60, 80)`, depth range `0.4391-2.0000 m`,
  dropout fraction `0.0471`, and non-empty rung-mask fraction `0.4575`; the saved
  image was inspected for camera framing and mask alignment.
- [x] A one-environment, one-iteration GPU smoke completed 24 rollout steps and
  a PPO update with the visual actor, then exited successfully.

Known boundary:
- The paper uses NVIDIA Warp for throughput. Isaac Lab 2.1's Warp ray caster only
  supports one static mesh, while this task moves an 18-object ladder collection
  independently per environment and requires semantic pixels for RFM. This first
  implementation therefore uses `TiledCamera`; renderer throughput must be
  benchmarked before selecting the full training environment count.
- Hybrid DAgger plus PPO and KL-weight annealing are not implemented in this
  visual-module milestone.

## 2026-07-27 - Local Reference Jump Repair and Rate-Limited Action Task

- Added the read-only `analyze_cr1_reference_dynamics.py` audit tool and the
  transformation-only `repair_cr1_reference_jumps.py` tool. The latter performs
  tapered local time stretching, PCHIP joint/root interpolation, quaternion SLERP,
  full URDF FK reconstruction, and categorical contact-phase time warping.
- Repaired source intervals `440:500` at 2x, `775:835` at 3x, and `968:983` at 2x,
  with 5 source frames of smooth transition at each boundary. The 50 Hz trajectory
  changes from `1177` to `1352` frames; source intervals `630:680` and `850:880`
  are not stretched.
- The three repaired intervals reduce maximum joint speed from
  `3.21/4.63/5.78` to `2.21/2.14/3.22 rad/s`. Right-hand body speed in the first
  two intervals changes from `1.51/3.20` to `0.90/1.52 m/s`; left-hand body speed
  in the final interval changes from `0.95` to `0.54 m/s`.
- Preserved the first 25 static default-pose frames exactly. Existing contact phase
  transitions map back to the same source frames within output sampling precision.
- Regenerated nominal and geometry-conditioned reset masks. Nominal validity is
  `1259/1352` with frame 0 valid; nominal worst penetration is `11.59 mm` at output
  frame 958, right sphere hand against step 7. Geometry-cell valid counts are
  `133/185.5/1266` min/median/max, and frame 0 is valid in every cell.
- Added the separate `Ladder-Climb-CR1-Reference-RateLimited-v0` task and
  `cr1_wood_step_reference_rate_limited` experiment. Existing tasks and source
  motions remain unchanged. The new task points to the repaired train-ready file
  and inherits the existing Relaxed-Reset-Nominal30 ladder behavior. Its episode
  limit is independently increased from 25 to 30 s so the 27.02 s repaired motion
  can reach `motion_complete` from frame 0.
- Added `RateLimitedJointPositionAction`: lazy initialization from measured reset
  joints, a 0.30 s reset ramp, alpha 0.5 first-order filtering, and a 6 rad/s
  (`0.12 rad/step` at 50 Hz) target-rate clamp. The value is deliberately
  conservative relative to the URDF's `19.38-23.76 rad/s`, but still requires
  replacement with measured CR1 controller command limits.
- Extended `check_cr1_ladder_env.py` with `--action_value` and an actual processed
  target-delta assertion for GPU smoke checks.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/action_manager.py`
- The same installed IsaacLab action files in the `rl-training` environment.
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `Ladder/scripts/smooth_resample_cr1_motion.py`
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`

Verification:
- [x] Final NPZ contains finite 25-joint/42-body kinematics, contact data, lineage,
  and `(4, 4, 1352)` geometry-conditioned reset masks.
- [x] All 61 Ladder CPU tests pass.
- [x] Python compilation passes for the new scripts, action term, and task config.
- [ ] IsaacLab GPU reset/step and processed-target assertion. The Codex container
  cannot enumerate CUDA; run the documented checker in the GPU terminal.

## 2026-07-27 - Coordinated Arc-Length Retiming Preview

- The first local-stretch result reduced peak speed but retained the source's dwell
  versus burst ratio. In the repaired right-hand intervals, maximum per-step hand
  displacement remained `5.57x` and `11.46x` the interval mean, so adding frames
  did not solve the visible jump.
- Added `retime_cr1_reference_by_arc_length.py`. It allocates samples by a combined
  metric containing the selected sphere-hand path, the base/two-hands/two-soles
  path, and `0.05 m/rad` times joint-space path length. This prevents hand-only
  retiming from compressing base and leg motion into the segment entrance.
- The preview retimes source `440:500` to 115 intervals, `775:835` to 170 intervals,
  and the complete left-hand release/swing/contact boundary `956:984` to 72
  intervals. It produces `1386` frames at 50 Hz without changing the source file.
- Added contact-safe interpolation semantics: a contact-to-swing interior sample is
  swing; swing-to-contact becomes contact only at the endpoint; inactive rung IDs
  are `-1`.
- Compared with the rejected local-stretch preview, maximum sphere-hand step length
  changes from `18.06/30.42/10.89 mm` to `4.89/4.46/5.08 mm`. Maximum-to-mean ratios
  are `1.50/1.68/1.21`. The left-hand entrance body-speed peak changes from about
  `1.22` to `0.254 m/s`, and the third-interval joint-speed peak is `2.42 rad/s`.
- The first 25 frames remain exactly static; source ranges `630:680` and `850:880`
  are byte-identical after lineage mapping. The existing rate-limited task still
  points to the earlier train-ready file pending visual acceptance of this preview.

Verification:
- [x] All trajectory/body arrays are finite and source time is strictly monotonic.
- [x] Contact transitions preserve their source phases with contact-safe subframe
  semantics; every inactive contact has rung ID `-1`.
- [ ] Visual acceptance in IsaacLab replay.
- [ ] Collision/reset-mask regeneration; intentionally deferred until acceptance.

## 2026-07-27 - Arc-Retimed Training Reference and Frame-0 XY Randomization

- Accepted the coordinated arc-length-retimed preview and generated a separate
  train-ready file with nominal and 5x5 geometry-grid reset metadata:
  `climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_arc_uniform_contact_safe_train_ready.npz`.
  The merged file has 1386 frames at 50 Hz. All source trajectory/contact fields
  are byte-identical to the accepted preview; only 36 reset-detector fields are new.
- Nominal collision screening accepts `1257/1386` frames and includes frame 0.
  Nominal worst penetration is `14.21 mm` at frame 967, between the right sphere
  hand and step 7. Geometry-cell valid counts are `133/185.5/1290`
  min/median/max, and frame 0 is valid in every cell.
- The independent `Ladder-Climb-CR1-Reference-RateLimited-v0` task now points to
  this train-ready motion. It retains the existing relaxed-reset Nominal30 ladder
  experiment behavior: reset selection uses the nominal mask while ladder geometry
  is randomized, and 30% of environments use nominal ladder dimensions.
- Added `fixed_start_pose_range` and a fixed-start pose-randomization probability
  to the motion command. The rate-limited task uses a 30% fixed frame-0 branch and
  a conditional `2/3` perturbation probability: expected total reset proportions
  are 10% frame-0 with exact-zero six-dimensional root pose offset, 20% frame-0
  with world X/Y in `[-0.10, 0.10] m`, pitch/yaw in `[-0.20, 0.20] rad`, the
  existing Z/roll ranges, 35% uniform valid frames, and 35% adaptive valid frames.
  Non-frame0 pose ranges, root velocity, joint noise, reference motion, and ladder
  pose are unchanged.
- Randomization-disable paths in train, play, deterministic smoke, and visual
  inspection now also clear `fixed_start_pose_range`.

References checked:
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`

Verification:
- [x] Reset-mixture tests confirm `10/20/35/35` nominal-frame0,
  perturbed-frame0, uniform-valid, and adaptive-valid proportions; exact-zero
  nominal root pose; and the requested perturbed XY/pitch/yaw bounds.
- [x] All 66 Ladder CPU tests pass and edited Python files compile.
- [ ] IsaacLab GPU reset/step smoke and processed-target rate-limit assertion.

## 2026-07-28 - Hand-to-Ladder Filtered Contact Fix

- Fixed the four per-hand ladder contact sensors. A single wildcard filter had
  matched all 7 step bodies or both front side-board bodies, while the PhysX
  contact view requires every filter expression to match exactly one body per
  environment. This made all four new collision reward terms identically zero.
- Each hand sensor now uses seven explicit step paths or two explicit front
  side-board paths. Reward weights, the 5 N force threshold, the 8 cm allowed
  step-contact distance, and the 18-box ladder collision geometry are unchanged.
- Play diagnostics report the total hand force, filtered step/side-board force,
  target error, threshold gates, and the resulting penalty decision.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`
- `isaaclab_v2.1.0/source/isaaclab/test/sensors/test_contact_sensor.py`

Verification:
- [x] Relevant CPU tests pass (`14 passed`) and edited Python files compile.
- [x] A 2-env, 1-iteration GPU train smoke completes without PhysX filter errors.
- [x] A 1-env play smoke reports step force matrices `(1, 1, 7, 3)` and
  side-board force matrices `(1, 1, 2, 3)`. At contact frame 1001 the left-hand
  total and step-filtered normal forces both report `123.22 N`.

## 2026-07-28 - Inboard Hand Targets and Step-Only Support Reward

- Moved the left/right sphere-hand targets laterally from `+0.265/-0.270 m` to
  `+0.250/-0.250 m`. With the 4.5 cm sphere radius this gives about 2.5 cm
  clearance from both inclined front side boards on the nominal ladder.
- Changed both hand contact-position temperatures from `0.05` to `0.03`.
- Added an opt-in filtered-force path to the upward hand-contact reward. Geometry-DR
  tasks now sum only each hand sensor's seven `Ladder_step_*` force-matrix entries;
  total force from a side board or another collider cannot earn upward support reward.
- Kept the old total-force behavior as the default for tasks without filtered step
  sensors. Reward weights and the 5--50 N normalization range are unchanged.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/rewards.py`

Verification:
- [x] Contact reward, contact annotation, and contact metric tests pass (`21 passed`).
- [x] Edited Python files compile.
- [x] A 2-env randomized-ladder GPU reset/step smoke at reference frame 700
  completes with finite `(2, 619)` actor and `(2, 631)` critic observations,
  finite rewards, and no contact-filter errors.

## 2026-07-29 - Surface-Aware Hand Contact Rewards

- Kept the robot and 18-box ladder collision geometry unchanged. The isolated
  `Ladder-Climb-CR1-Reference-RateLimited-v0` task now classifies filtered step
  normal forces by world-Z direction: upward top-face contacts are allowed,
  horizontal front/edge contacts and downward underside contacts are penalized
  and logged separately for each hand.
- Added per-hand filters for `/World/ground` and the ladder `base_platform` and
  retained always-on per-hand front-zigzag side-board penalties. The older broad
  swing, downward-net-force, and off-target whole-step penalties are disabled in
  this task variant so an acceptable top contact is not punished twice.
- Set both hand contact-position temperatures to `0.01 m^2`. Added a `-0.20`
  missing-support term after a 10-frame/0.2-second required-phase and post-reset
  grace period when the sphere center is within 8 cm of its target but filtered
  upward step force is at most 5 N. Forbidden ground/front/underside/side-board
  terms use weight `-0.40` and a 5 N force threshold.
- Updated play diagnostics to report `TOP_ALLOWED`, `FRONT_EDGE`, or `UNDERSIDE`,
  ground and side-board forces, and the missing-support decision.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py`
- `isaaclab_v2.1.0/source/isaaclab/test/sensors/test_contact_sensor.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`

Verification:
- [x] All Ladder CPU tests pass (`74 passed`); the focused contact
  reward/metric/termination subset passes (`21 passed`).
- [x] Edited Python files compile.
- [ ] IsaacLab GPU reset/step smoke. The available execution environment could
  not enumerate a CUDA device and another Kit process held the datastore lock;
  failure occurred before scene/config creation.

## 2026-07-30 - Fixed Frame-0 Playback With Full Reset Noise

- When `commands.motion.fixed_reference_frame` is set to `0`, mark the episode
  as a fixed-start episode so `fixed_start_pose_range` and its configured
  randomization probability are still applied. Fixed playback at other frames
  remains free of frame-0-specific reset noise.
- This keeps frame-0 playback compatible with the training configuration:
  ladder geometry and startup physics randomization remain enabled, while the
  optional interval root-velocity disturbance is explicitly enabled by the
  play command.

Verification:
- [x] All Ladder CPU tests pass (`74 passed`).
- [x] Edited Python files compile.

## 2026-07-30 - Target-Rung Force Reward and Force Calibration

- Added the isolated `Ladder-Climb-CR1-Reference-Force-Diagnostic-v0` task. The
  established rate-limited task remains unchanged. In the diagnostic task, a
  hand earns upward support reward only from the per-environment rung selected
  by `contact_rung_id`; force on another rung cannot earn that reward.
- Added per-rung filtered sensors for both feet and episode maximum metrics for
  hand top/front/underside/side-board/ground force, target-rung hand upward
  force, and foot step/target-rung force.
- Added the inactive continuous excess-force reward implementation
  `square(relu((force_max - force_soft) / force_scale))`. No unmeasured hand or
  foot limit is enabled in a training task yet.
- Extended play force calibration to report run-wide and per-episode contact
  `p50/p95/max`, wrong-rung hand force, foot contact-onset force, and foot
  single-support force. The first five control steps after every reset are
  excluded. An optional JSON limits file enables per-episode soft-limit frame
  fractions and hard-limit crossing counts without changing the policy.
- Made `omni.appwindow` a viewer-only lazy import so the same force-calibration
  play command works with the Isaac Sim headless experience.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/utils/wood_step_ladder.py`
- `Ladder/scripts/repair_cr1_sphere_hand_penetration.py`

Verification:
- [x] All Ladder CPU tests pass (`78 passed`).
- [x] Edited Python files compile.
- [x] A 64-env, 3000-step randomized/disturbed force-calibration play completed
  with 268 episode records. Target-rung upward force p95 was `155/125 N`
  (left/right); single-support foot force p95 was `709/699 N`.
- [x] A second 64-env, 3000-step run with the candidate limits completed with
  274 episode records. Frame-weighted over-soft fractions were `2.17%/0.21%`
  for left/right top force and `0.53%/0.054%` for left/right foot step force.
  Undesired-contact rates were highest for the left side board (`6.03%`) and
  right underside (`2.20%`).
- [x] Added isolated `Ladder-Climb-CR1-Reference-ForceLimited-v0`. It keeps the
  existing binary collision terms, adds capped continuous force-excess terms,
  logs per-limb over-soft fractions and hard-crossing counts, and leaves the
  actor/critic observations, reset logic, reference, and terminations unchanged.
- [x] Target-top excess uses `soft=180 N`, scale `100 N`; non-target top and
  forbidden surfaces use `soft=20 N`, scale `50 N`; foot step force uses
  `soft=900 N`, scale `300 N`. All normalized excess values are capped at `3`.
- [x] All Ladder CPU tests pass (`80 passed`) and edited Python files compile.

## 2026-07-31 - Isolated Strict Front/Side Contact Penalties

- Added the isolated task
  `Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-v0`; the existing
  force-limited task and its checkpoints are unchanged.
- Kept the filtered binary contact threshold at `5 N`.
- Increased left/right hand front-edge and diagonal side-board binary reward
  weights from `-0.40` to `-0.60`.
- Changed only the front-edge and side-board continuous excess-force terms to
  `soft=10 N`, `scale=50 N`, and weight `-0.10`. Wrong-rung top, underside,
  ground, target-top, and foot force limits remain inherited from the existing
  force-limited task.
- The play diagnostic already prints the filtered force vector, normalized
  world-Z component, surface class, and per-surface force maxima. Use those
  fields to distinguish a front-edge classifier error from a side-board
  sensor/filter issue before changing the classifier.

## 2026-07-31 - Isolated Deployment-Oriented Actor Observations

- Added `Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-v0`
  without changing the established strict-front/side task.
- Replaced only the actor input with a 660-D group: five-frame histories of
  reference joint position/velocity, projected gravity, base angular velocity,
  relative joint position/velocity, and previous actions, plus current motion
  phase and four contact indicators.
- Enabled actor corruption using the whole-body-tracking reference ranges:
  projected gravity `+/-0.05`, base angular velocity `+/-0.2 rad/s`, joint
  position `+/-0.01 rad`, and joint velocity `+/-0.5 rad/s`.
- Kept the existing 631-D privileged critic, rewards, resets, randomization,
  reference motion, and action filtering unchanged. The new experiment logs to
  `cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs`.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/mdp/observations.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/manager_term_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`

Verification:
- [x] Edited Python files compile.
- [x] All Ladder CPU tests pass (`80 passed`).
- [x] Kit-backed config instantiation confirms the requested observation order,
  history/noise settings, 660-D actor, inherited privileged critic, and unchanged
  reset-mixing probabilities.
- [ ] Reset/step and one-iteration training smoke remain to be run on the user's
  GPU host. This execution environment could not expose CUDA; CPU fallback then
  failed while resolving Isaac Sim's remote ground-plane USD.

## 2026-08-03 - Isolated Ladder Without Front Side Boards

- Added `Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-NoSideBoards-v0`.
  The existing ladder asset, tasks, and checkpoints remain unchanged.
- The new scene filters the two `front_zigzag_side_board_left/right` boxes out
  of the 18-box collection, so both the spawned visual cuboids and their
  physical collision geometry contain 16 boxes. Rungs, top platform, posts,
  rear support legs, rear cross beam, and side limit bars remain.
- The new command config keeps only the rear support legs in the inclined-box
  list. Side-board sensors/reward terms are removed rather than assigned a
  zero weight, because IsaacLab resolves `SceneEntityCfg` even for zero-weight
  reward terms. The other reference, observation, reset, disturbance, force,
  and action-rate settings are inherited unchanged.
- The new runner writes to
  `cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs_no_side_boards`.

Front-edge penalty comparison against the previous force-limited run (last 500
iterations) shows that the stricter terms were active but did not reduce the
observed collisions: left/right front-force maxima changed from `59.4/71.2 N`
to `78.4/85.1 N`, and over-soft fractions from `6.80%/3.71%` to
`8.33%/6.02%`. The binary and continuous reward magnitudes increased as
configured. This is not a controlled penalty-only ablation because the newer
run also used the deployment-oriented actor observations; the no-side-board
task provides a cleaner follow-up comparison.

References checked:
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_geometry_dr_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/reward_manager.py`
- the two TensorBoard event files listed in the comparison above

Verification:
- [x] New task config instantiated after Kit startup: 16 ladder boxes,
  no side-board sensors/reward terms, rear support legs retained.
- [x] Edited Python files compile.
- [x] All Ladder CPU tests pass (`80 passed`).
- [ ] GPU reset/step smoke remains to be run on the training host; this
  environment has no CUDA device.

## 2026-08-03 - Nearer Hand Targets and Target-Rung Missing Contact

- Added the isolated task
  `Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-NoSideBoards-NearTargets-v0`;
  the existing no-side-board task and checkpoints remain unchanged.
- Moved both sphere-hand targets `0.03 m` toward the robot along world `-X`:
  left changed from `(-0.015, 0.25, 0.045)` to
  `(-0.045, 0.25, 0.045)`, and right changed from
  `(-0.010, -0.25, 0.045)` to `(-0.040, -0.25, 0.045)`.
- Added an opt-in `target_rung_only` mode to the missing-required-contact
  penalty. The new task uses it for both hands, so upward force on the wrong
  rung cannot suppress the `-0.20` missing-contact penalty.
- Matched both hands' underside penalties to the established strict front-face
  settings: binary threshold `5 N`, binary weight `-0.60`, continuous soft
  threshold `10 N`, scale `50 N`, and continuous weight `-0.10`.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/mdp/rewards.py`
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `Ladder/source/ladder/ladder/utils/contact_force.py`

Verification:
- [x] Edited Python files compile.
- [x] Focused reward tests pass (`16 passed`).
- [x] All Ladder CPU tests pass (`81 passed`).
- [ ] Kit config/reset/step smoke remains to be run on the training host. The
  current execution environment could not initialize CUDA/Vulkan.

## 2026-08-04 - Original Target Control Task

- Added the isolated task
  `Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-NoSideBoards-OriginalTargets-v0`.
- Restored the runtime hand target offsets to left `(-0.015, 0.250, 0.045)`
  and right `(-0.010, -0.250, 0.045)`.
- Kept target-rung-only missing-contact detection and the strict underside
  force penalties, so this run isolates the target-offset change from the
  previous `NearTargets` experiment.
- The prior near-target task and its `model_6999.pt` remain unchanged.

## 2026-08-06 - Safer Frame-0 Forward Reset Range

- Added the isolated `OriginalTargets-SafeFrame0` task variant.
- Changed only the randomized frame-0 root world-X range from
  `[-0.10, +0.10] m` to `[-0.10, +0.06] m`. Positive world X moves the robot
  toward the ladder; all other reset and domain randomization stays inherited.
- Kept the completed `OriginalTargets` task unchanged so its training and
  `model_6999.pt` evaluation configuration remain reproducible.

## 2026-08-06 - Restored Rails With Whole-Robot Avoidance

- Added the isolated task `Ladder-Climb-CR1-Reference-RailsProtected-SafeFrame0-v0`;
  the completed no-side-board task and checkpoint remain unchanged.
- Restored the full 18-box ladder, including
  `front_zigzag_side_board_left/right`, and restored both boxes to the ladder
  geometry-randomization update list.
- Added one filtered contact sensor to each rail. Each sensor uses the rail as
  its single sensor body and filters `Robot/.*`, following IsaacLab's
  one-to-many contact-filter limitation. Any robot link over `5 N` receives a
  `-2.0` binary penalty per contacted rail, plus a squared force-excess penalty
  with `soft=5 N`, `scale=50 N`, clamp `3`, and weight `-0.25`.
- Raised target-rung top-force soft limits from `180 N` to `250 N`, hand support
  reward saturation from `50 N` to `100 N`, and narrowed front classification
  from `|Fz|/|F| < 0.5` to `< 0.35`. Underside classification remains unchanged.
- Added a command-level front-normal threshold so training metrics and play force
  statistics use the same `0.35` front classification as the reward. Its default
  remains `0.5`, preserving all prior task behavior.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`
- `isaaclab_v2.1.0/source/isaaclab/test/sensors/test_contact_sensor.py`
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_geometry_dr_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/rewards.py`

Verification:
- [x] Edited Python files compile and pass Ruff checks.
- [x] Focused contact/ladder tests pass (`8 passed`).
- [x] All Ladder CPU tests pass (`82 passed`).
- [ ] Kit-backed config/reset/step smoke remains to be run on the GPU host. The
  current execution container could not initialize CUDA/Vulkan before imports.

## 2026-08-06 - Per-Body Rail Contact Sensor Fix

- Preserved the failed rail-mounted sensor task and its checkpoints, and added
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-SafeFrame0-v0`.
- Replaced the two rail-mounted sensors, which produced zero reward for every
  training iteration, with one sensor on each of the CR1's 32 collision-bearing
  bodies. Each sensor filters the two rail boxes, matching the working
  robot-body-to-ladder direction and IsaacLab's one-to-many requirement.
- Added batched per-rail aggregation across the 32 sensors. A per-control-step
  cache lets the four left/right binary/excess reward terms share one force
  reduction. Penalty thresholds and weights remain unchanged.
- Added per-rail force output to interactive play diagnostics and force-stat
  collection so the signal can be validated before another long training run.

## 2026-08-07 - Thin Suspended Front Rails

- Added the isolated task
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-SafeFrame0-v0`;
  existing ladder tasks, geometry, runs, and checkpoints remain unchanged.
- Reduced only the two front side-board local depths by two thirds, from
  `0.1400 m` to `0.0467 m`. Their lengths, lateral widths, rotations, and
  left/right positions are unchanged.
- Shifted each rail center `0.0467 m` along its positive local depth normal so
  the original upward/robot-facing plane remains fixed. In task coordinates
  this is approximately `(-0.0391, 0.0, +0.0254) m`; therefore there is no
  lateral displacement.
- The procedural `RigidObjectCollection` uses the same derived box spec for
  rendering and collision, and existing ladder inclination/spacing
  randomization transforms the thinner rails together with the other inclined
  boxes. Reference motion and contact targets are unchanged.
- In this thin-rail task only, stable required hand phases no longer require
  the hand to be within `0.08 m` before missing target-rung support is
  penalized. After the existing 10-frame grace period, target-rung upward force
  at or below `5 N` receives weight `-0.40` at any hand distance.
- Reduced the whole-robot binary penalty per contacted rail from `-2.0` to
  `-1.0` for this task. The continuous rail force penalty remains unchanged at
  `soft=5 N`, `scale=50 N`, clamp `3`, and weight `-0.25`.
- Updated `train.py --checkpoint` to accept a direct existing checkpoint path,
  enabling an isolated thin-rail run to warm-start from a compatible prior
  experiment without copying checkpoints or mixing experiment directories.

Verification:
- [x] Thin-rail geometry unit tests pass.
- [x] All Ladder CPU tests pass (`84 passed`).
- [ ] Kit-backed Viewer inspection remains to be run on the GPU host.

## 2026-08-11 - Native-FOV Warp Camera Diagnostics

- Kept the RTX and Warp depth windows as independent native camera outputs.
  The RTX image remains the square-pixel `80 x 60` view, while Warp retains
  the `x2_vis`-style independent `87 x 58 deg` pinhole field of view.
- Restricted RTX-to-Warp remapping to the explicitly labeled depth-difference
  diagnostic. The remapped depth is saved separately and is not used by the
  Warp image, RFM masks, or policy observations.
- Corrected the custom Warp renderer pose source to the current
  `d435_head_link` world pose. `TiledCameraCfg.return_latest_camera_pose`
  defaults to `False`, so `camera.data.pos_w` otherwise reports the
  initialization pose and makes the Warp view appear fixed while the robot
  moves. The diagnostic has no extrinsic randomization and uses an identity
  sensor offset, so the current link pose is the matching camera pose here.
- Added left/right hand projection metadata so an empty robot mask can be
  distinguished from robot geometry that should have intersected the view.
- Confirmed that frame 0 legitimately contains no Warp robot pixels: both hand
  centers are below the `58 deg` vertical FOV. A CPU robot-only render at frame
  100 produced 868 robot pixels (763 rays hit robot meshes and 109 hit the hand
  spheres), confirming that the robot-mask ray path works when the robot enters
  the Warp view.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/camera.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/X2/x2_vis_config.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/utils/raycast.py`

Verification:
- [x] Warp-camera focused tests pass (`8 passed`).
- [x] All Ladder CPU tests pass (`101 passed`).
- [x] Edited Python files compile.
- [ ] Interactive RTX capture remains to be rerun on the GPU host; this
  execution container has no CUDA/Vulkan device.

## 2026-08-12 - Warp Training Decision

- The paper Appendix A.2 specifies NVIDIA Warp, not Isaac Sim's built-in
  renderer, for its `60 x 80` training depth. The paper does not publish the
  renderer kernel or a throughput measurement.
- The custom `CR1WarpCamera` remains a one-environment visual diagnostic, not
  a training sensor. Its kernel traces every pixel against up to 30 robot
  collision meshes, two hand spheres, and 18 ladder boxes, then synchronizes
  the Warp device before each frame is consumed.
- `humanoid_vision` `x2_vis` supplies the relevant reusable pattern:
  `DepthCastMesh` preallocates batched GPU buffers, derives all moving camera
  poses in tensors, and launches one kernel over every due environment and
  pixel without a per-frame synchronization. Warp itself does not guarantee a
  faster trainer; this batched GPU-resident design is what can make it faster.
- Implement a local batched Ladder Warp renderer before visual-policy
  training, rather than importing the Isaac-Gym-specific reference or using
  the diagnostic renderer. It must consume dynamic `RigidObjectCollection`
  poses for all 18 geometry-randomized ladder boxes and emit native depth plus
  `RUNG | ROBOT` classes for paper RFM.
- Compare the accepted batched Warp training task against `TiledCamera` with
  matched environment count, `60 x 80` output, update interval, headless mode,
  policy, and RFM. Adopt Warp only if its images/masks remain correct and its
  measured end-to-end training steps/s is higher.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/tiled_camera_cfg.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/tiled_camera.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/base/legged_robot_vis.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/utils/raycast.py`

## 2026-08-12 - Batched CR1 Warp Renderer Acceptance Tool

- Added `CR1WarpBatchCamera`, a local GPU-resident renderer patterned after the
  batching strategy in `humanoid_vision` `DepthCastMesh`. One Warp launch
  covers every requested environment and all image pixels; the renderer
  preallocates `[num_envs, 60, 80]` depth and hit-class buffers and does not
  synchronize per frame.
- The batch kernel uses the current CR1 `d435_head_link` pose for every
  environment, all current robot-link poses for self-occlusion, and every
  `RigidObjectCollection` box pose for the 18 independently geometry-randomized
  ladder parts. It returns native metric image-plane depth and the explicit
  `RUNG`, `ROBOT`, other-ladder, ground, and background classes; RFM remains
  exactly `RUNG | ROBOT`.
- Added `scripts/view_cr1_warp_batch.py`. It replays reference motion across
  256 environments by default, retains per-environment ladder inclination and
  spacing randomization, shows depth/classes/RFM/RFM-depth for one selected
  environment, saves its native tensors, and logs untimed-warmup-excluded Warp
  GPU render time plus Mrays/s and batch class fractions. It does not create an
  RTX camera or read TiledCamera output.
- This tool is intentionally not connected to the registered visual PPO task.
  It is the acceptance gate for image correctness and renderer throughput
  before adding a Warp-backed visual observation term.

References checked:
- `Ladder/source/ladder/ladder/utils/warp_camera.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_geometry_dr_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/utils/raycast.py`

Verification:
- [x] Batch-kernel CPU tests cover simultaneous per-environment robot/rung
  hits, RFM classes, requested output rows, and no per-frame
  `wp.synchronize_device()` call (`10 passed`).
- [x] Script and renderer compile.
- [ ] GPU operator check is required for moving views, varied per-environment
  ladder geometry, and measured batch render throughput.

## 2026-08-12 - Warp Visual Observation Integration (Awaiting GPU Acceptance)

- GPU operator acceptance completed for `view_cr1_warp_batch.py`: the head
  camera follows the robot and the native depth/classes/RFM/RFM-depth windows
  are correct while replaying the geometry-randomized ladder motion.
- At 256 environments and `80 x 60`, native Warp rendering measured roughly
  `1.2-7.7 ms` per batch (`160-1032 Mrays/s`, depending on visible robot and
  ladder intersection work). This is a renderer-kernel measurement only.
- Added the matched RTX baseline script
  `scripts/benchmark_cr1_tiled_camera.py`. The accepted 256-environment full
  visual step result is `206.600 ms` mean, `210.049 ms` median, `233.498 ms`
  p95, and `1239.1 transitions/s`. It includes physics, RTX/TiledCamera,
  existing RFM/noise preprocessing, and observation computation.
- Added the separate registered task
  `Ladder-Climb-CR1-Geometry-DR-WarpVisual-v0`. It has no `TiledCamera` or
  RTX semantic sensor. Its stateful `mdp.warp_visual_depth` observation term
  owns `CR1WarpBatchCamera`, samples the same reset-time camera translation
  (`+-1 cm`) and rotation (`+-1.5 deg`) ranges, derives world camera poses
  from the current `d435_head_link`, and consumes current GPU state tensors
  for robot and ladder boxes.
- Warp hit classes are processed through the same paper-aligned visual branch:
  metric clipping `[0.1, 2.0] m`; `RUNG | ROBOT` RFM with probability `0.1`;
  otherwise Gaussian depth noise `0.005 m` and `5%` pixel dropout. The output
  remains actor-compatible `N x 1 x 60 x 80`; the critic remains privileged
  through the existing asymmetric PPO observation group.
- Added `scripts/benchmark_cr1_warp_visual.py` for the required complete-step
  acceptance benchmark. It asserts no RTX sensors, validates the actor depth
  shape/range/finiteness at runtime, and reports matched `transitions/s`.
- The existing RTX visual task remains unchanged as
  `Ladder-Climb-CR1-Geometry-DR-Visual-v0`, so either backend can be compared
  or retained without altering other ladder tasks.

Verification:
- [x] New Warp RFM preprocessing test and all visual-depth unit tests pass
  (`8 passed`).
- [x] New Python modules compile and task registration imports successfully.
- [ ] GPU end-to-end Warp task/throughput acceptance is required before PPO
  training. This execution container has no CUDA/Vulkan device; existing Warp
  kernel tests cannot initialize their Warp cache/device here.

## 2026-08-12 - SSR Visual Network Study (No Runtime Change)

- Studied the local SSR paper and `humanoid_vision` X2 visual code before
  changing Ladder's visual policy. The resulting design note is
  `../humanoid_vision/SSR_visual_network_and_LadderMan_migration.md`.
- Confirmed the reference route is end-to-end asymmetric PPO, not a
  teacher-student method. SSR's actor processes depth and temporal
  proprioception; its critic separately uses privileged simulator state.
- The first future transfer boundary is the recurrent cross-modal encoder:
  depth CNN `[32,64,128]`, proprioception MLP `[512,256,128]`, and a one-layer
  GRU with 256 hidden units. RFM/noise, rewards, and actor/critic ownership
  remain Ladder-specific at this boundary.
- The SSR `36 x 36` CNN output is `128 x 1 x 1`; at Ladder's accepted
  `60 x 80` resolution it becomes `128 x 2 x 3`. The planned initial depth
  projection is therefore `768 -> 128`, not SSR's direct `128 -> 128` layer.
- Future work is gated: first a reset-filled one-frame visual lag buffer, then
  model/hidden-state smoke testing, then recurrent PPO rollout verification,
  then a controlled visual-training baseline. No Ladder source or training
  configuration changed in this study milestone.

References checked:
- `humanoid_vision/ssr Scaling Surefooted and Symmetric Humanoid.pdf`
- `humanoid_vision/.../envs/X2/x2_vis_config.py`
- `humanoid_vision/.../envs/base/legged_robot_vis.py`
- `humanoid_vision/.../rsl_rl/modules/actor_critic_recurrent.py`
- `humanoid_vision/.../rsl_rl/algorithms/ppo_mm.py`
- `humanoid_vision/.../rsl_rl/storage/rollout_storage.py`
- `Ladder/source/ladder/ladder/models/visual.py`
- installed `rl-training` RSL-RL `RNNModel` and `RNN` interfaces

Verification:
- [x] Cross-checked paper Appendix A.4 dimensions against the local recurrent
  actor implementation.
- [x] Cross-checked the depth lag/reset and recurrent rollout handling against
  the local visual environment and PPO storage.
- [x] Ladder runtime code was not changed.

## 2026-08-14 - Gate B Relative Height-Map Target Interface

- Added `visual_targets` to both CR1 RTX and Warp visual environments. It is a
  separate, non-concatenated observation group and is not listed in the actor
  or critic PPO observation groups.
- The target term batches four maps in the order
  `left_hand, right_hand, left_foot, right_foot`, with shape `[B, 4, 10, 3]`.
  Hand references use sphere-link centers; foot references use the existing
  virtual sole contact points. Rays stay world-down and the grid stays world
  X/Y aligned.
- The group exposes `relative_height` in meters and a float `hit_valid` mask.
  No-hit relative values are zero placeholders and must be masked by
  `hit_valid` in the later decoder loss.
- Added `Ladder/scripts/check_cr1_limb_heightmap_targets.py` for reset/step
  acceptance before adding the visual decoder or PPO auxiliary losses.

References checked:
- `Ladder/source/ladder/ladder/utils/limb_heightmaps.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/observations.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_visual_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_warp_visual_env_cfg.py`
- Installed IsaacLab `ObservationManager` and `RslRlVecEnvWrapper`

Verification:
- [x] Python compile check passed for all changed modules and the checker.
- [x] Existing limb-heightmap tests passed (`15 passed`).
- [ ] IsaacLab reset/step acceptance is pending on a host with CUDA/RTX and
  local Isaac assets. This container has no CUDA device and cannot open the
  configured remote ground USD on CPU.

## 2026-08-14 - Height-Map Coordinate Semantic Review

- Reviewed `humanoid_vision` `HeightCast`, `legged_robot.py`, SSR recurrent
  actor/critic, and `algorithm_plugin.py`. The ray caster returns world `z_hit`,
  while estimator/decoder MSE uses a clipped and scaled difference from a
  body/foot reference plane.
- Confirmed X2 formulas: body
  `clip(root_z - base_height_target - z_hit, -1, 1) * 10/3`; foot
  `clip(foot_center_z - 0.055 - z_hit, -0.8, 0.8) * 5`.
- Ladder Gate A remains raw-world-height diagnostics only. No training target,
  observation term, model, or PPO code changed in this review.
- CR1 virtual sole contact bodies already include the ankle-to-sole offset, so
  the planned foot target uses virtual sole `z` directly. CR1 sphere hands have
  radius `0.045 m` and the confirmed hand reference is their center.

References checked:
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/utils/raycast.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/base/legged_robot.py`
- `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/algorithms/algorithm_plugin.py`
- `Ladder/source/ladder/ladder/utils/limb_heightmaps.py`
- `Ladder/source/ladder/ladder/utils/contact_schema.py`
- `Ladder/assets/robots/CR1-B2-PRO-260608/CR1-B2-pro_free_wrist_fixed_spherehand.urdf`

User decision:
- Hand reference is the `left/right_sphere_hand_link` center.
- Raw world hit heights are visualization data; relative heights are training
  data. Normalization remains a later model-contract decision.

Implementation:
- `LimbHeightMap` now exposes raw world hit `z`, reference `z`, relative height
  in meters, geometry validity, and per-sample hit validity without changing
  the existing raw-height field.
- The Gate-A viewer displays raw world hit heights and saves a separate
  `*_relative_height_m.npy` plus `*_hit_valid.npy` for later training use.
- No actor observation, decoder, or PPO path consumes these targets yet.

## 2026-08-14 - TransformerVision GRU Rollout Continuity

- Visual-only pretraining now carries the detached GRU hidden state from one
  24-step rollout chunk into the next chunk for every continuing environment.
  A done environment alone clears its corresponding hidden-state column.
- The stored initial state is replayed during the auxiliary update, so the
  truncated 24-step backward pass matches the state used during collection.
- The frozen 660-D actor remains the only action source; no visual latent or
  PPO update is enabled by this change.

Verification:
- [x] TransformerVision unit and runner tests passed (`10 passed`).
- [ ] GPU long-training acceptance remains pending.

## 2026-08-17 - Live TransformerVision Reconstruction Dashboard

- The no-update TransformerVision acceptance script now refreshes the four-limb
  target/prediction/error dashboard once per controlled frozen-policy step.
- Dashboard creation and refresh happen before the matching `env.step(action)`;
  the final pose is paused before the window is kept open. This prevents UI
  updates from advancing physics without a newly computed robot action.
- The runner exposes an optional read-only per-step callback. Training uses the
  unchanged collection path, and the diagnostic still performs no optimizer or
  PPO update.

Verification:
- [x] Python compile check passed.
- [x] TransformerVision unit and runner tests passed (`10 passed`).
- [ ] Isaac Sim live-window acceptance remains pending on the GUI/CUDA host.

## 2026-08-17 - Interactive 3D Height-Map Overlay

- The live TransformerVision checker now places the selected environment's
  four `10x3` height maps directly on their moving world-XY limb grids.
- Target and prediction points use the same grid and valid-hit mask. Their
  world height is reconstructed as `reference_z_w - relative_height_m`.
- `H` cycles one marker set through `target -> prediction -> off`; limb colors
  remain fixed so hand/foot identity is preserved across modes.
- Target grid geometry is retained only in the diagnostic cache and is not
  added to actor or critic observations.

Verification:
- [x] Python compile check passed.
- [x] Limb-map, TransformerVision, and runner tests passed (`27 passed`).
- [ ] 3D marker placement and keyboard focus require GUI/CUDA host acceptance.

## 2026-08-17 - Safe Manual Reset in Visual Acceptance

- `R` now requests a full environment reset during the live no-update visual
  diagnostic. The request is checked both before inference and after the UI
  callback but before `env.step(action)`, so an action computed from the old
  state is never applied after reset.
- Manual reset clears the full visual GRU state. If it occurs inside a stored
  rollout, the preceding transition is marked as a sequence boundary and its
  future-observation target is excluded.
- Training collection has no reset callback and remains unchanged.

Verification:
- [x] Related unit tests passed (`28 passed`), including manual reset timing,
  GRU clearing, and recurrent sequence-boundary insertion.
- [ ] Keyboard-driven Isaac Sim reset remains pending on the GUI/CUDA host.

## 2026-08-17 - Rung-Weighted Visual Pretraining and Dependency Diagnostics

- Added exact ray-hit semantic classes to the four-limb height-map target,
  rollout storage, runner, and auxiliary replay path. Reconstruction MSE now
  weights valid rung hits 2 and other valid hits 1, normalized by the sum of
  those weights.
- Added separate rung/non-rung MSE for all maps, hands, and feet. The no-update
  checker also reports per-limb unweighted MSE, 2:1 weighted MSE, class counts,
  and rung fractions, and saves the hit-class tensor beside the captures.
- Aligned future reconstruction with the current pure-SSR architecture by
  changing its decoder input from the complete 51-D latent to only the 16-D
  stochastic dynamic latent. Checkpoints containing the retired 51-D decoder
  now fail early with a clear incompatibility message; a fresh visual-only run
  is required.
- Added deterministic depth ablations over the same stored rollout and GRU
  state: normal depth, zero depth, and depth shuffled across environments.
  Absolute MSE, increase over normal, and relative increase are reported for
  hand/foot and rung/non-rung reconstruction groups.
- Added the always-zero velocity baseline and overall/per-axis MSE, RMSE, MAE,
  target/prediction distribution, correlation, and skill-over-zero metrics.
  The frozen 660-D actor remains unchanged and the visual latent is still not
  supplied to PPO.
- Split the 85-D future-observation diagnostic into root angular velocity,
  root orientation, joint position, joint velocity, action, and
  climbing-direction fields. Depth ablations also report velocity and future
  prediction changes, in addition to terrain reconstruction changes.

References checked:
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/observation_manager.py`
- `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/modules/actor_critic.py`
- `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/algorithms/algorithm_plugin.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/base/legged_robot.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/X2/x2_vis_config.py`

Verification:
- [x] Visual/height-map/runner unit tests passed (`38 passed, 1 skipped`).
- [x] A CUDA smoke run completed one visual-only optimizer update and kept the
  actor unchanged (`actor_delta=0`).
- [x] The CUDA no-update checker emitted rung/non-rung, depth-ablation, and
  velocity-baseline metrics and saved its JSON/dashboard artifacts.
- [ ] Long-run quality must be evaluated from a fresh checkpoint because the
  future decoder input changed from 51 to 16 dimensions.

## 2026-08-17 - Field-Normalized TransformerVision Proprioception

- The shared 85-D frame builder now applies the X2-reference field scaling to
  both the ten-frame encoder history and the next-frame auxiliary label:
  angular velocity `*0.2`, 6-D orientation unchanged, joint position
  `(q-default_q)/CR1_ACTION_SCALE`, joint velocity `*0.1`, action and direction
  unchanged, followed by clipping to `[-100,100]`.
- Joint scales are resolved in the live articulation's joint-name order and
  cached on the environment. Joint positions subtract each environment's
  randomized default pose, preserving the semantics of the existing startup
  joint calibration randomization.
- Visual checkpoints now contain the data-contract identifier
  `cr1_field_normalized_proprio_rung_weighted_dynamic16_v1`. Missing/older
  contracts fail early instead of silently resuming incompatible raw-input
  weights.
- The no-update checker records normalization rules and input-history/future-
  target min, max, mean, and standard deviation for all six field groups.

References checked:
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/X2/x2_vis_config.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/base/legged_robot.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/mdp/observations.py`
- `Ladder/source/ladder/ladder/robots/cr1.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/events.py`

Verification:
- [x] Visual/height-map/runner regression tests passed (`39 passed, 1 skipped`).
- [x] CUDA two-environment visual-only optimizer smoke test completed with
  finite losses and `actor_delta=0`.
- [x] CUDA diagnostic loaded the new-contract checkpoint and reported finite
  normalized field ranges for both history and next-frame targets.

## 2026-08-17 - Height-Map Cache Refresh and Future Decoder ELU

- Replaced the unbounded limb-height target cache with a manager term whose
  shared result is keyed by `common_step_counter` and invalidated on reset.
  The relative-height, valid-mask, and hit-class observation terms still share
  one ray query inside a control step, but the next step cannot reuse stale
  geometry.
- Added the requested ELU between the future decoder's `128 -> 85` projection
  and final `85 -> 85` projection.
- Bumped the visual checkpoint contract to
  `cr1_field_normalized_proprio_rung_weighted_dynamic16_future_elu_v2`; older
  visual-only checkpoints fail early and require a fresh run.
- Added an authoritative current-design note at
  `docs/learning/cr1_transformer_vision_current_design.md` and marked the older
  migration plan as historical where proposals conflict.

References checked:
- `humanoid_vision/.../envs/base/legged_robot.py` (X2 SSR actor observation and
  future-observation timing)
- `humanoid_vision/.../rsl_rl/modules/actor_critic.py` and `submodule.py`
  (implicit-only future decoder and ELU MLP construction)
- `isaaclab_v2.1.0/.../envs/manager_based_env.py` and
  `manager_based_rl_env.py` (reset lifecycle and `common_step_counter`)

Verification:
- [x] Focused TransformerVision/runner/cache/height-map regression tests passed
  (`33 passed`).
- [x] A one-environment IsaacLab visual-only optimizer smoke test completed,
  loaded all six observation groups, produced finite losses, saved a v2
  checkpoint, and kept `actor_delta=0`.
- [x] A two-environment three-step height-map smoke test showed changing target
  ranges on every step (`max: 1.1886 -> 1.1859 -> 1.1782 -> 1.1699`), proving
  that the former first-frame cache is no longer retained across steps.

## 2026-08-18 - TransformerVision Asymmetric PPO Integration

- Added a new upward-only PPO task without changing the accepted visual-only
  environment or scripts. The new actor receives the current normalized 85-D
  frame and the 51-D TransformerVision latent; the 631-D privileged critic
  remains separate.
- Matched the humanoid visual PPO update boundary: rollout stores the exact
  sampled 136-D actor input and current GRU state, PPO never resamples the
  latent, and a separate optimizer replays complete temporal sequences for
  visual auxiliary losses.
- GRU state is persistent across 24-step chunks, cleared per environment only
  after done/reset, and detached at chunk boundaries. Auxiliary replay starts
  from the stored rollout state and applies each done before the next input.
- Fresh runs require an accepted visual checkpoint; full PPO checkpoints save
  and restore actor, visual encoder, critic, PPO optimizer, and auxiliary
  optimizer. Native recurrent play resets the policy GRU on automatic or
  manual environment reset.
- Safe hold and down-climbing are deliberately not fabricated in this task.
  They remain blocked on the planned command diagnostic and separately
  recorded downward reference motion.

References checked:

- `humanoid_vision/.../rsl_rl/modules/actor_critic_recurrent.py`
- `humanoid_vision/.../rsl_rl/algorithms/ppo_ce.py`
- `humanoid_vision/.../rsl_rl/storage/rollout_storage.py`
- `humanoid_vision/.../rsl_rl/runners/on_policy_runner.py`
- installed RSL-RL 5 `runners/on_policy_runner.py` and `algorithms/ppo.py`

Verification:

- [x] The complete Ladder test suite passed (`150 passed, 1 skipped`).
- [x] Eight real Warp environments completed 24 rollout steps, one PPO update,
  one auxiliary update, and saved a full checkpoint with finite losses.
- [x] A second real run restored that full checkpoint without supplying the
  visual-pretraining checkpoint and completed another update.
- [x] A 512-environment run completed 12,288 transitions, two 256-environment
  recurrent auxiliary mini-batches, and reported zero rollout fixed-latent
  log-probability error without OOM.
- [x] Native headless play loaded the full PPO checkpoint and executed the
  recurrent policy path.
- [ ] Long upward training and native GUI playback remain user acceptance.
- [ ] Three-command safe hold/down integration remains pending new motion data.

## 2026-08-18 - Accepted Visual Baseline Archive and Task Comparison

- Preserved the accepted fresh 5000-iteration visual-only checkpoint and its
  original frozen actor without moving or overwriting either result.
- Added `baseline_snapshot_2026-08-18` beside the visual checkpoint. It contains
  a compressed snapshot of Ladder source/scripts/tests, Conda environment
  exports, SHA-256 hashes, and exact inspection/reproduction commands.
- Recorded the observation/reward comparison in
  `docs/learning/cr1_visual_ppo_observation_reward_comparison.md`.
- Confirmed from saved `env.yaml` files that the July visual-pretraining base
  and August thin-rail task use the same 660-D nonvisual actor contract and the
  same 631-D critic. The August task changes rail geometry, contact sensing,
  reset range, and contact rewards.
- Confirmed that the current visual PPO task still inherits the July reward and
  scene configuration, while replacing the 660-D actor input with current
  proprioception 85 plus visual latent 51.
- Identified a geometry migration constraint: PhysX, Warp rendering, and limb
  height-map queries must all receive the thin-rail box specification together;
  changing only environment inheritance would render/label the wrong geometry.

References checked:

- the July 31, August 10, and August 18 saved environment YAML files
- CR1 strict-front/side, rail-protected, per-body rail, and thin-rail configs
- CR1 TransformerVision environment/data/target configs
- X2 `x2_vis_config.py` privileged terrain-height observation design

Verification:

- [x] Visual checkpoint, summary, frozen actor, and source archive hashes were
  written to the baseline `SHA256SUMS` file.
- [x] The archived environment records Python 3.11.15, PyTorch 2.7.0+cu128,
  Gymnasium 1.2.0, and the explicit `rl-training` Conda package set.
- [x] No active task, reward, actor, critic, or training code was changed by
  this comparison/archive step.

## 2026-08-18 - Independent ThinRails Scratch Visual PPO Task

- Added the independent task
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-SafeFrame0-WarpTransformerVision-PPO-v0`.
  The accepted visual-only task and the earlier July-geometry visual PPO task
  remain registered and unchanged.
- Kept the action actor contract exactly at current proprioception 85 plus
  TransformerVision latent 51 (`136 -> 512 -> 256 -> 128 -> 25`).
- Extended only the privileged critic: existing 631, current joint velocity
  25, sampled ladder parameters 4, and clean four-limb relative height maps
  `4 x 10 x 3 = 120`, for a verified total of 780.
- Moved the ThinRails box tuple to the shared geometry configuration. PhysX,
  Warp depth rendering, auxiliary limb-map labels, and critic limb maps now
  resolve the same canonical `THIN_RAIL_WOOD_STEP_BOX_SPECS` selection. Runtime
  validation rejects a ladder whose object-name set differs from that tuple.
- Added an explicit scratch initialization mode. A fresh run of this task
  randomly initializes actor, critic, and TransformerVision and rejects
  `--vision_checkpoint`. A complete checkpoint may still resume the same run.
  The accepted visual-only `model_5000.pt` is preserved only for later
  inspection and comparison.

References checked:

- IsaacLab manager observation configuration and RSL-RL runner integration
- the August ThinRails scene/reward configuration
- CR1 geometry-DR ladder construction, Warp camera, and limb-height query
- the existing Ladder TransformerVision PPO algorithm and recurrent runner

Verification:

- [x] Full Ladder test suite passed (`151 passed, 2 skipped`).
- [x] Eight real IsaacLab/Warp environments instantiated with actor input 136
  and critic input 780, completed a 24-step rollout, PPO update, recurrent
  auxiliary update, and wrote `model_0.pt` with finite losses.
- [x] Saved `env.yaml` records the new critic fields and `thin_rails` for Warp,
  auxiliary targets, and critic maps; `agent.yaml` records scratch
  initialization and an empty visual-pretraining checkpoint.
- [x] Added a separate no-update preflight task with the latest trained
  ThinRails 660-D actor observation. The PPO training task does not compute or
  expose that legacy policy group.
- [x] Extended the existing diagnostic dashboard with a geometry-only mode:
  it displays the processed Warp depth actually sent to the CNN, the four true
  relative height maps, and target points in the Isaac viewport. Random visual
  predictions are hidden; `H` toggles target points and `R` requests reset.
- [x] A two-environment, three-step real preflight produced policy 660, critic
  780, depth history `2x60x80`, limb maps `4x10x3`, zero done transitions, and
  exactly zero actor/visual parameter changes.
- [ ] GUI geometry inspection and long training remain staged user-acceptance
  steps.

## 2026-08-18 - ThinRails Scratch Visual PPO Learning-Rate Validation

- The first 4096-environment scratch trial became unstable at iteration 41.
  Although its configured PPO learning rate was `5e-4`, the adaptive schedule
  was applied after every one of the 5 x 4 PPO mini-batch updates and reached
  its `1e-2` cap during the first iteration. At iteration 41, policy KL rose to
  `1.80`, value loss to `241.44`, and rollout log-probability replay error to
  `0.569`; the run was stopped and is not an approved resume source.
- Changed only `CR1ThinRailsTransformerVisionPPORunnerCfg` to a fixed PPO
  learning rate of `5e-4`. The earlier checkpoint-initialized visual PPO task
  remains adaptive, and the separate auxiliary optimizer remains unchanged.
- A new 4096-environment, 100-iteration scratch validation completed naturally
  in `2026-08-18_14-43-35_fixed_lr_validation100` and saved `model_99.pt`.
  Across all 100 iterations, learning rate stayed at `5e-4`, policy KL stayed
  between `1.86e-4` and `7.09e-4`, and value loss stayed finite without a
  late spike.
- Auxiliary learning remained active: total loss fell `3.635 -> 1.440`,
  velocity skill over the zero predictor rose `0.018 -> 0.916`, hand-map MSE
  fell `0.307 -> 0.0273`, foot-map MSE fell `0.0211 -> 0.00344`, and future
  loss fell `0.544 -> 0.273`.
- The final checkpoint contains actor plus visual encoder (state MLP, depth
  CNN, two Transformer layers, fusion, GRU, estimator heads and decoders), the
  780-D critic, PPO optimizer, and auxiliary optimizer.

Verification:

- [x] Full Ladder test suite passed (`151 passed, 2 skipped`).
- [x] The corrected run crossed the former iteration-41 failure point and
  completed 100 iterations with finite PPO and auxiliary losses.
- [x] Checkpoint structure and all 100 TensorBoard scalar samples were audited.

## 2026-08-19 - Reference-Aligned Visual PPO Stability Constraints

- Rechecked X2 visual PPO configuration and its RSL-RL implementation. X2
  starts PPO and CE/visual optimization at `5e-4`, adapts only the PPO
  optimizer from mini-batch KL, clips unscaled gradients to `0.1`, clips
  actions to `[-10,10]`, clips observations to `[-100,100]`, bounds explicit
  estimates to `[-10,10]`, and uses CUDA autocast plus `GradScaler`.
- Replaced the ThinRails fixed PPO schedule with the same KL thresholds
  (`desired_kl=0.01`) and a Ladder-specific `[1e-5,1e-3]` range. The visual
  auxiliary optimizer remains fixed at `5e-4`; entropy remains `0.005`.
- Enabled ThinRails action clipping at `10` and final network-input clipping at
  `100`. The wrapper clips only proprio history/current proprio, processed
  depth, and critic input. It deliberately leaves all simulator supervision
  labels unchanged.
- Audited the new critic fields. Limb height maps are relative heights in
  meters, with prior real runs approximately `-0.4` to `1.65`; ladder geometry
  parameters are also order-one physical values. Neither needs a new scaling
  transform. The final clip is only an outlier guard. Raw current joint
  velocity is protected by the final critic clip.
- Clipped the explicit velocity estimate used in the 51-D actor latent to
  `[-10,10]`, matching X2. Added one CUDA GradScaler shared by the separate PPO
  and visual optimizers, saved/restored its state, and restored adaptive PPO
  learning rate from the optimizer while enforcing the configured bounds.
- Added logged PPO/visual maximum gradient norms, non-finite gradient-step
  counts, and current GradScaler scale. A non-finite CUDA gradient is skipped
  by GradScaler; CPU/full-precision validation still fails immediately.

References checked:

- `humanoid_vision/.../X2/x2_vis_config.py`
- `humanoid_vision/.../legged_gym/envs/base/legged_robot.py`
- `humanoid_vision/.../rsl_rl/algorithms/ppo.py`
- `humanoid_vision/.../rsl_rl/algorithms/ppo_ce.py`
- `humanoid_vision/.../rsl_rl/modules/actor_critic.py`

Verification:

- [x] TransformerVision unit suite passed (`25 passed`), including a real CUDA
  autocast/GradScaler PPO plus visual update.
- [x] IsaacLab application config assertions confirmed adaptive `5e-4`, bounds
  `[1e-5,1e-3]`, auxiliary `5e-4`, gradient limit `0.1`, action clip `10`,
  observation clip `100`, and GradScaler enabled.
- [x] A real 32-environment ThinRails Warp run completed two iterations and
  saved complete checkpoints in
  `isaaclab_v2.1.0/logs/rsl_rl/cr1_transformer_vision_ppo_thin_rails_scratch/2026-08-19_10-05-57_adaptive_amp_constraints_metrics_smoke`.
  PPO KL was `0.0089/0.0035`, fixed-input log-probability error stayed below
  `3e-4`, PPO reached but did not exceed `1e-3`, and visual LR stayed `5e-4`.
  GradScaler skipped one initial PPO overflow, reduced its scale to `32768`,
  and then reported zero non-finite PPO/visual steps on iteration 1.
- [ ] A 4096-environment staged run must cross the prior iteration-41 and
  iteration-2033 failure regions before this is approved for an unattended
  long run.

## 2026-08-19 - ThinRails Visual Input and Dynamics Randomization

- Changed only the independent ThinRails scratch visual-PPO task. The accepted
  frozen-policy visual pretraining and its `model_5000.pt` inspection path keep
  their previous input contract.
- Centered the two encoder depth frames as `(depth_m - 1.0) * 1.0` after the
  existing Warp depth augmentation. The diagnostic depth frame remains metric;
  only the CNN history is centered.
- Added input-only uniform proprioceptive sensor noise: angular velocity
  `+/-0.5 rad/s` before its `0.2` scale, root orientation perturbation
  `+/-0.05 rad` per axis, joint position `+/-0.01` after per-joint
  normalization, and joint velocity `+/-1.5 rad/s` before its `0.1` scale.
  This matches X2's observation-space joint-position noise rather than using
  one raw-radian error for CR1 joints with different action scales. The clean
  future-proprioception target is unchanged.
- Added independent per-environment horizontal and vertical Warp FOV offsets
  in `[-1,1]` degrees and extended the batched ray kernel to consume per-env
  pinhole tangents. FOV is resampled on environment reset.
- Added a `0..2` policy/control-step delay before the existing action filter
  and rate limiter. Delay state and lag are reset independently for each done
  environment.
- Added the X2 motor formula to CR1's implicit actuators. One shared per-joint
  motor-strength sample in `[0.8,1.2]` multiplies both nominal stiffness and
  damping, followed by independent Kp and Kd factors in `[0.9,1.1]`. This is
  equivalent to X2's post-PD motor multiplier before effort saturation.
  Factors are resampled on environment reset and every 8 seconds, matching
  X2's reset and interval behavior.
- Scaled only the newly added privileged critic joint velocity by `0.1`.
  Ladder parameters and relative limb height maps remain order-one metric
  fields, and the wrapper still applies the final `[-100,100]` outlier guard.
- Matched the humanoid CE data-reuse schedule with five passes over four
  recurrent environment groups. Each logical visual batch uses 256-env
  gradient-accumulation micro-batches, yielding exactly 20 visual optimizer
  steps without requiring 1024 depth sequences in memory at once.
- Kept `dynamic_logvar` unconstrained. CUDA GradScaler remains the numerical
  guard; its skipped-step count/scale plus log-variance mean/min/max, maximum
  implied standard deviation, and maximum absolute dynamic mean are logged.

References checked:

- X2 `x2_vis_config.py` and `legged_robot.py` depth, noise, FOV, lag, motor,
  observation clipping, and reset logic
- humanoid RSL-RL `ppo_ce.py` five-epoch/four-mini-batch CE update
- IsaacLab `envs/mdp/events.py::randomize_actuator_gains` and implicit-drive
  gain write APIs
- IsaacLab `utils/buffers/delay_buffer.py`

Verification:

- [x] Full Ladder test suite passed under a running IsaacLab application
  (`164 passed`).
- [x] IsaacLab configuration instantiated the rate-limited action with delay
  `0..2`, critic joint scale `0.1`, and auxiliary schedule `5 x 4`.
- [x] A four-environment real Warp smoke run completed one 24-step rollout,
  all PPO updates, and exactly 20 visual optimizer steps after the corrected
  normalized joint noise and reset-time motor event. Its saved config is
  `isaaclab_v2.1.0/logs/rsl_rl/cr1_transformer_vision_ppo_thin_rails_scratch/2026-08-19_11-14-13_reference_motor_noise_smoke`.
- [x] The smoke run's first visual pass produced one AMP overflow; GradScaler
  skipped it, reduced scale from `65536` to `32768`, and the iteration
  completed. This remains a long-run metric to monitor while log-variance is
  intentionally unclamped.

## 2026-08-19 - ZDW 0818 Capture 5 Full-Clip Retarget Gate

- Preserved the complete 8529-frame, 200 Hz BVH and converted it to the
  accepted human/ladder alignment without cropping or temporal processing.
  The complete clip is intentionally retained so a later vision task can
  annotate upward, hold, and downward commands.
- Retargeted all frames to CR1 at the physical 50 Hz rate, producing 2133
  robot frames. The stored human sequence matches the aligned source sampled
  by four to floating-point precision; no source smoothing or corrective
  offsets were added.
- Added exact retarget checkpoint/resume support plus two opt-in foot-sticking
  infeasibility fallbacks. This capture used a one-frame transition-foot
  release at frame 1444 and a temporary 2-to-3 cm sticking tolerance at five
  explicitly recorded frames. Defaults remain unchanged for other motions.
- Replaced the retarget writer's hard-coded 30 Hz metadata with explicit
  `output_fps` configuration. The current artifact is stored and played at
  50 Hz.
- Audited derivatives and all active CR1 training colliders against all 18
  ladder boxes. The source is smooth, but the first robot solve has IK branch
  jumps at frames 21, 121, and especially 655 (right-hip-z peak
  `64.36 rad/s`). The deepest real ladder penetration is `14.70 mm`, primarily
  ankle versus step; an earlier apparent 18 cm elbow/rail penetration was a
  sign error and is actually positive clearance.
- Confirmed the solver currently returns the least-violating candidate when
  an SQP frame has no feasible candidate. Therefore this output remains a
  visual-review artifact and is not labeled train-ready.

References checked:

- Holosoma `examples/robot_retarget.py`
- Holosoma `src/interaction_mesh_retargeter.py`
- `demo_data/climb_ladder/restored_prefix_v300/README.md`
- `Ladder/scripts/export_holosoma_cr1_motion.py`
- `Ladder/scripts/preprocess_ladder_motion.py`

Verification:

- [x] Retarget configuration and implementation files compile in the
  `retarget` environment.
- [x] Tyro exposes `--output-fps`; result metadata reports 50 Hz.
- [x] Full result is finite with qpos `(2133,32)` and human joints
  `(2133,22,3)`.
- [x] The debug viewer loads the complete result and reports the derivative
  and reference-tracking summaries.
- [ ] Visually inspect the flagged robot-frame intervals before applying any
  solver-local repair or generating training contact/command annotations.

## 2026-08-20 - Fix New-Capture Retarget Handedness and Backward Heading

- Reproduced the backward-facing CR1 numerically. In the first full retarget,
  robot local `+X` and the horizontal robot-to-ladder direction had dot product
  `-0.947` (about `161 deg` apart).
- Rejected an initial `180 deg` robot-yaw workaround: after 30 solver warmup
  steps the root returned to roughly `-94 deg`, with forward-to-ladder dot
  `-0.999`. The yaw option changes only the initial guess and does not correct
  the source coordinate convention.
- Identified the fixed pipeline cause. The legacy Y-up conversion
  `(X,Y,Z)->(X,Z,Y)` has determinant `-1`. Earlier successful LadderMan data
  then applied an X reflection, resulting in the proper right-handed transform
  `(X,Y,Z)->(-X,Z,Y)`. The new capture initially omitted that required step.
- Added explicit, default-off `mirror_x` support to `prepare_ladder_bvh.py` and
  a mandatory printed hip-versus-toe heading-consistency diagnostic. Negative
  consistency indicates a handedness/180-degree-heading failure before any
  expensive retarget begins.
- Generated a separate, complete mirrored input without overwriting the source
  BVH or rejected first input. Only X was reflected; Y/Z, timing, alignment
  translation, and the accepted human/ladder height are unchanged.
- A 43-frame corrected smoke retains root yaw near `+89 deg`, gives
  forward-to-ladder dot `0.99994`, lowers final retarget cost from the forced-yaw
  test's `1.94` to `0.712`, and lowers maximum joint speed from `3.56` to
  `1.87 rad/s`.

Verification:

- [x] Coordinate-heading regression test passes together with the existing
  human-orientation tests (`3 tests`).
- [x] Corrected 8529-frame input is finite and has heading consistency
  `+0.999767`.
- [x] Corrected 43-frame retarget is finite, complete, and stores 50 Hz.
- [ ] User visual acceptance of the corrected short retarget.
- [ ] Full corrected retarget; the rejected left-handed full result must not be
  exported to Ladder training data.

## 2026-08-20 - Complete TransformerVision PPO Playback and Diagnostics

- Added the standalone `Ladder/scripts/rsl_rl/play_transformer_vision_ppo.py`
  for complete visual-PPO checkpoints. The older frozen-policy visual
  pretraining inspector remains unchanged and continues to serve its original
  checkpoint format.
- Playback controls the robot with the recurrent visual actor and obtains the
  action plus its matching visual output in a single encoder/GRU evaluation;
  diagnostics therefore cannot advance the GRU a second time.
- Added a live processed-Warp-depth and four-limb height-map dashboard. Rows
  are left hand, right hand, left foot, and right foot; columns are target,
  prediction, and absolute error. `H` cycles the world-space point overlay
  through target, prediction, and off; `R` resets the environment and actor
  GRU; `P/N/I/Q` provide pause, single step, immediate log, and quit controls.
- Added setup, periodic, termination, episode, and summary diagnostics for
  observation shapes, depth processing, actions, velocity prediction versus
  a zero predictor, dynamic latent statistics, rung/non-rung reconstruction,
  next-proprio fields, tracking errors, physical limb forces, and termination
  causes. JSONL events, a summary JSON, and dashboard PNG captures are saved
  under `outputs/cr1_transformer_vision_ppo_play/<checkpoint-name>/`.

Verification:

- [x] TransformerVision PPO unit tests pass (`10 passed`), including agreement
  between the diagnostic visual output and the action generated from it.
- [x] A two-environment headless CUDA smoke run loaded `model_7999.pt`, stepped
  the environment, and produced all 30 aggregate diagnostic metrics.
- [x] A one-environment Isaac Sim GUI smoke run created the fallback Isaac Sim
  dashboard window, updated world-space height-map markers, saved dashboard
  PNGs, and exited cleanly after five steps.
- [x] Fixed the playback `R` reset ordering: a reset request received during
  an Isaac render update now keeps the timeline live and prevents execution
  of another action from the old episode. Reset progress and elapsed time are
  logged, stale dashboard state is cleared, and the recurrent actor state is
  cleared only after fresh reset observations are available. Headless and GUI
  tests both reset after step 5 (about 0.3 seconds) and continued through step
  10 with refreshed depth, height maps, reference frame, and GRU state.
- [x] Corrected the complete-PPO dashboard depth orientation. Warp depth is
  already in camera row order and is now displayed without the vertical flip
  reserved for limb-map visualization. This is a display-only correction;
  the depth tensor supplied to the CNN and all trained parameters are
  unchanged. A GUI smoke capture shows the ladder upright with the ground at
  the bottom while the accepted limb-map panel orientation remains unchanged.

## 2026-08-20 - Deterministic Dynamic Decoder and Physical-State Target

- Replaced the active dynamic VAE branch with a deterministic
  `Linear(256,16)` latent and a `16 -> 64 -> 128 -> 59 -> 59` ELU decoder.
  Active training no longer creates log-variance, samples dynamic noise, or
  adds KL to the auxiliary objective.
- Kept every encoder input and policy/critic observation unchanged. Storage
  still records the clean 85-D next proprioceptive frame, while the loss
  selects only root angular velocity 3, root orientation 6, joint position 25,
  and joint velocity 25. Action 25 and climb direction 1 are no longer future
  reconstruction targets.
- Preserved the prior implementation as
  `VariationalTransformerVisionEncoder`. It retains the old parameter names,
  85-D decoder, stochastic reparameterization, and KL loss. Complete-v2 PPO
  playback detects the checkpoint contract and constructs this legacy encoder
  before strict state loading.
- Added a new data contract so old VAE training cannot be accidentally resumed
  into the deterministic network. Updated pretraining/full-PPO diagnostics to
  report deterministic latent magnitude and only the supervised future fields.

References checked:

- `humanoid_vision/.../rsl_rl/modules/actor_critic.py`
- `humanoid_vision/.../rsl_rl/algorithms/ppo_ce.py`
- `Ladder/.../mdp/observations.py`
- Ladder TransformerVision module, pretrainer, PPO, runner, and diagnostics

Verification:

- [x] Active encoder emits latent 51 and next physical state 59 without VAE
  heads or KL.
- [x] Legacy encoder emits next proprio 85 and retains VAE/KL behavior.
- [x] Full Ladder test suite passes (`158 passed, 3 skipped`); the focused
  visual/pretrainer/PPO subset passes (`28 passed`).
- [x] The accepted `model_7999.pt` policy state loads strictly into the legacy
  actor with no missing or unexpected parameters.

## 2026-08-20 - User-Aligned Foot160/Pitch10 Full Retarget

- Generated a new isolated Holosoma input from the preserved 8529-frame,
  200 Hz capture using the user-saved ladder/human alignment, the corrected
  right-handed X mirror, a `0.160 m` Foot-to-ToeBase target, and a `10 deg`
  pitch reduction. The archived BVH and earlier alignment branches remain
  unchanged.
- Ran exactly one full retarget at 50 Hz with the previously disclosed fixed
  configuration: stride 4, warmup 30, foot/toe/hand weights `4/8/3`, 1 mm
  object non-penetration, 2 cm foot sticking with 3 cm fallback, and no
  self-collision or ladder-clearance extension. No smoothing, filtering,
  interpolation, cropping, local repair, or alternate parameter test was run.
- Preserved the user-aligned absolute human/ladder height by applying the exact
  `+0.009852227488 m` post-grounding cancellation. Stored retarget human joints
  equal the prepared input sampled at `[::4]` exactly.
- The result is complete and finite at `(2133, 32)` qpos and 50 Hz. Configured
  foot-sticking fallback was required only at source frames `822` and `1626`;
  there were no penetration fallbacks or transitioning-foot releases.
- Marked the output review-only rather than train-ready. Peak robot joint speed
  is `8.85 rad/s` (right elbow, frame 731), peak joint acceleration is
  `746.0 rad/s^2` (right ankle-y, frame 1313), and the exhaustive training
  collider/18-box audit finds `14.47 mm` maximum penetration at frame 1022
  (left ankle-x versus step 3). These intervals require operator visual review
  before any approved solver or trajectory change.

Verification:

- [x] Prepared motion shape, finite values, heading, morphology, and source
  hash checked.
- [x] Full retarget complete flag, frame count, FPS, finite values, quaternion
  norm, and model joint limits checked.
- [x] Full derivative and 32-by-18 collision audits completed read-only.
- [ ] Operator visual review and decision on whether any retarget adjustment is
  warranted.

## 2026-08-20 - Second User Alignment Full Retarget

- Preserved the first full retarget and generated a separate v2 input/output
  after the operator changed only the human alignment from
  `(0.040, 0.010, -0.035) m` to `(0.040, 0.000, -0.030) m`. Ladder pose,
  X-mirror correction, Foot-to-ToeBase length `0.160 m`, and pitch reduction
  `10 deg` remain unchanged.
- Re-ran the full 50 Hz retarget once with exactly the previously disclosed
  fixed parameters. No smoothing, filtering, local repair, alternate weight,
  or exploratory parameter change was performed.
- The separate v2 result is finite and complete at `(2133, 32)`, matches its
  prepared human input sampled at `[::4]` exactly, and respects model joint
  limits. Foot-sticking fallback occurred at source frames `1498` and `1785`;
  there were no penetration fallbacks or transitioning-foot releases.
- Read-only derivative audit: maximum root step `14.61 mm`, root angular speed
  `1.831 rad/s`, joint speed `7.62 rad/s`, and joint acceleration
  `723.6 rad/s^2`.
- Read-only 32 robot-collider by 18 ladder-box audit: maximum penetration
  `12.82 mm` at frame `1135` (right ankle-x versus step 3); frames deeper than
  1/5/10 mm are `1684/368/79`, improved from the first alignment's
  `1739/833/281` but still unsuitable for automatic train-ready acceptance.

Verification:

- [x] Input hash, shape, finite values, heading, exact translation delta,
  morphology, and grounding cancellation checked.
- [x] Output hash, complete flag, frame count, FPS, finite values, quaternion
  norm, exact sampled-human equality, and joint limits checked.
- [x] Full derivative and exhaustive training-collider/ladder audits completed.
- [ ] Operator visual review of the v2 artifact and approval before any repair
  or training-data conversion.

## 2026-08-20 - Third Alignment Retarget with Recorded Toe Pitch

- Generated isolated v3 input/output after the operator raised the human by
  `10 mm` and changed Foot-to-ToeBase pitch reduction from `10 deg` to zero.
  Ladder pose, human X/Y, corrected X mirror, and `0.160 m` ToeBase target
  length remain unchanged.
- Ran one full 50 Hz retarget with the same fixed v2 solver/constraint
  configuration. No smoothing, filtering, interpolation, local repair, or
  alternate parameter experiment was performed.
- The result is finite and complete at `(2133, 32)`, respects joint limits,
  and required 3 cm foot-sticking fallback at frames `601`, `605`, and `1241`.
  No penetration fallback or transitioning-foot release occurred.
- Derivative audit reports `16.70 mm` maximum root step, `1.847 rad/s` maximum
  root angular speed, `8.14 rad/s` maximum joint speed, and
  `805.7 rad/s^2` maximum joint acceleration. Both joint extrema are dominated
  by right ankle-x motion around frames `1233-1236`.
- Exhaustive collision audit reports `10.95 mm` maximum penetration at frame
  `1285` (right ankle-x versus step 3) and `1709/745/10` frames deeper than
  1/5/10 mm. V3 improves the deepest contacts relative to v2 but worsens the
  prevalence of 5 mm contacts and the peak ankle derivatives, so recorded toe
  pitch is not accepted as a jitter fix.

Verification:

- [x] Saved alignment, source hash, v3 input shape/finite values, mirror,
  Foot-to-ToeBase length, and pitch checked.
- [x] Output hash, complete flag, frame count, FPS, finite values, quaternion
  norm, and joint limits checked.
- [x] Foot-sticking metadata, full derivative audit, and exhaustive 32-by-18
  collider audit completed read-only.
- [ ] Operator visual review before any approved post-retarget repair.

## 2026-08-20 - ZDW0818 Capture 5 IsaacLab Export, World Alignment, and Smoothing Preview

- Accepted the v3 Holosoma retarget as the immutable source and exported all
  `2133` frames through the current CR1 IsaacLab articulation at the source
  `50 Hz`. No crop, resampling, hold insertion, filtering, or contact labels
  were applied to the raw export.
- Added `ladder.utils.motion_world_transform.align_motion_ladder_frame` and
  `Ladder/scripts/transform_cr1_motion_world_frame.py`. The reusable transform
  maps a motion between explicit source/target ladder poses and applies the
  same yaw rotation to world positions, orientations, linear velocities, and
  angular velocities while leaving joint-space fields unchanged.
- Mapped the Holosoma ladder `(0, 0.30, 0), yaw=180 deg` to the current
  geometry-DR ladder `(0.38, 0.04, 0), yaw=90 deg`, equivalent to
  `x'=y+0.08`, `y'=-x+0.04`. Applied the operator-approved `-5.6457434 mm`
  robot Z translation so the lowest frame-0 virtual sole reaches the task
  ground at `z=-0.03 m`.
- Generated a review-only fourth-order zero-phase Butterworth preview at the
  unchanged `50 Hz`: non-leg joints `6 Hz`, all leg joints `4 Hz`, root
  position/orientation `5 Hz`. World-alignment lineage is preserved through
  the smoothing stage.
- The preview reduces whole-joint acceleration RMS from `12.295` to
  `4.024 rad/s^2` and leg acceleration RMS from `16.559` to
  `4.162 rad/s^2`. Right ankle-x peak speed/acceleration fall from
  `8.14/805.7` to `1.09/15.7 rad/s` and `rad/s^2`. Joint-position change is
  `0.00955 rad` RMS and `0.14118 rad` maximum.
- Critical body-position deltas are small for 95% of frames: left/right hands
  `2.06/2.23 mm`, left/right soles `4.51/5.64 mm`. The largest right-hand
  correction is `30.46 mm` at the original frame-1943 jump. No joint limits
  are violated, quaternion norms remain valid, and first/last joint changes
  remain below `0.00022 rad`.
- An in-memory contact-candidate comparison changed only `5/4` left/right hand
  indicator frames and no rung IDs. Foot candidate counts increase because
  the retargeted high-frequency ankle motion no longer repeatedly violates
  the `0.35 m/s` annotation speed gate; no contact labels were written yet.

References checked:

- `whole_body_tracking/scripts/csv_to_npz.py`
- `Ladder/scripts/export_holosoma_cr1_motion.py`
- `Ladder/scripts/export_holosoma_cr1_motion_offline.py`
- `Ladder/scripts/smooth_resample_cr1_motion.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`

Verification:

- [x] New world-transform unit test passes in `rl-training`.
- [x] IsaacLab versus source joint positions and offline FK checked.
- [x] Aligned and smoothed artifacts are finite, 2133 frames, 50 Hz, and
  preserve the approved ladder-frame and grounding metadata.
- [x] Dynamics, endpoint, end-effector deviation, and URDF joint-limit audits
  completed.
- [ ] Operator visual review before accepting the preview or writing contact,
  command, collision-validity, or train-ready fields.

## 2026-08-20 - ZDW0818 Capture 5 No-Z-Shift Smoothing and Contact Preview

- Preserved the v3 Holosoma retarget and all earlier IsaacLab artifacts. A new
  isolated processing branch removes the previously applied `-5.6457434 mm`
  robot-world Z translation; `world_alignment_robot_translation` is exactly
  `(0, 0, 0) m`, so the manually selected human/ladder height is restored.
- Extended `smooth_resample_cr1_motion.py` with a backward-compatible ankle
  cutoff. The new 2133-frame, 50 Hz preview uses fourth-order zero-phase
  Butterworth filtering at `5 Hz` for non-leg joints, `3 Hz` for hip/knee leg
  joints, `2 Hz` for the four ankle joints, and `4 Hz` for root position and
  orientation. No arm IK, path routing, per-frame IK repair, crop, resampling,
  or frame insertion was applied.
- Ankle acceleration RMS falls from `25.139` to `2.083 rad/s^2`; the maximum
  central-difference ankle acceleration falls from `202.64` to
  `19.58 rad/s^2`. Whole-joint acceleration RMS falls from `12.295` to
  `3.472 rad/s^2`, and no URDF joint limit is violated.
- Added a data-local `--limb_target_offsets` annotation option without changing
  the shared CR1 defaults. This preview uses left/right sphere-hand offsets
  `(-0.015, +0.210, +0.045) m` and `(-0.010, -0.220, +0.045) m`. With ladder
  origin Y `+0.04 m`, hand target world Y is `+0.25/-0.18 m`, respectively;
  this moves the old targets inward by `4 cm` left and `3 cm` right.
- Nominal signed-distance audit improves frames deeper than `1/3/5 mm` from
  `1835/1022/746` in the unsmoothed no-Z-shift export to `1690/937/504` in the
  preview. These counts include intended hand/foot support overlap and are not
  all unwanted collisions.
- The preview is not train-ready: the source right sphere hand jumps
  `83.32 mm` between frames 1942 and 1943. Ordinary temporal smoothing spreads
  that jump over adjacent frames and creates a `30.01 mm` nominal penetration
  at frame 1942 between `right_sphere_hand_link` and `step_05_collision`
  (`36` frames exceed `10 mm`, `5` exceed `20 mm`). No arm-only IK was run per
  operator instruction. Under the existing `+/-1 deg`, `+/-1 cm` 3-by-3
  geometry audit, the worst corner is `59.87 mm` at frame 921 between the left
  sphere hand and step 7, so geometry-randomized reset masks also remain
  review-only.

Artifacts:

- no-Z-shift alignment: `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/05_postprocess_v2_no_z_smooth5_leg3_ankle2_root4/01_world_aligned_no_z/`
- smoothing preview: `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/05_postprocess_v2_no_z_smooth5_leg3_ankle2_root4/02_smoothing_preview/`
- contact preview: `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/05_postprocess_v2_no_z_smooth5_leg3_ankle2_root4/03_contact_preview/`
- 3-by-3 collision audit sidecar: `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/05_postprocess_v2_no_z_smooth5_leg3_ankle2_root4/04_collision_audit/`

Verification:

- [x] All 2133 frames remain finite at 50 Hz and preserve the 42.64 s duration.
- [x] World-alignment metadata records zero robot translation.
- [x] Joint limits, dynamics, end-effector deviations, contact targets, and
  nominal/geometry-envelope penetrations checked.
- [x] Existing motion-processing and world-transform tests pass (`6 passed`).
- [ ] Operator visual review of ankle motion and frames around 1939-1947.
- [ ] Do not package this preview for training until the right-hand transition
  is accepted or a non-IK repair method is explicitly approved.

## 2026-08-20 - Base-Grounded Ladder and CR1 Default-Standing Start Preview

- Preserved all prior ZDW0818 artifacts and created an isolated
  `06_base_grounded_default_start/` branch. The replay/training ground is at
  world `z=-0.03 m`, while the ladder base-platform cube spans local
  `z=[0, 0.16]`; therefore the ladder asset and the source robot trajectory are
  both transformed by exactly `-0.03 m` in world Z. The base-platform bottom
  is now coincident with the ground. Contact target origin Z changes from
  `0.19` to `0.16 m`, and the saved ladder asset translation is
  `(0.38, 0.04, -0.03) m`.
- Added `add_cr1_default_standing_start.py`, following the accepted old
  training-reference start design: 25 frames (`0.5 s`) of zero-joint CR1
  default standing with zero root roll/pitch and source yaw, followed by a
  50-frame (`1.0 s`) quintic transition into source frames `0:50`. The default
  feet are centered on the new source frame-0 foot midpoint rather than using
  a hard-coded world XY.
- A uniform `-0.03 m` robot shift would place the captured standing feet below
  the ground. As in the old training start, a root-Z clearance envelope is
  therefore applied during ground support. This version checks the actual
  left/right ankle-x collision-mesh vertices as well as virtual sole points,
  not only the sole centers. The required/applied correction is stored per
  frame, peaks at `33.159 mm`, and is constrained to change by at most
  `1 mm/frame`. No joint, leg, arm, or per-frame IK is used.
- The final preview contains `2158` frames at `50 Hz` (`43.14 s`). The first
  25 frames have exactly zero joint/root velocity. Over the default-to-motion
  transition, peak central-difference joint speed/acceleration is
  `2.171 rad/s` / `6.856 rad/s^2`, and root speed/acceleration is
  `0.094 m/s` / `0.365 m/s^2`. Global root acceleration remains below
  `3.262 m/s^2` after slope-limiting the clearance envelope.
- Updated the offline reset detector to read ladder asset pose and target
  origin from motion annotation metadata. This prevents auditing a lowered
  motion against the stale Z-zero ladder pose. Nominal 1 mm screening accepts
  `789/2158` frames; the 3-by-3 geometry envelope accepts `341/2158` and frame
  0 is valid in every sampled geometry. The high invalid count includes
  expected support contact overlap and is not a train-ready decision by
  itself.
- Existing arm-path problems remain intentionally unresolved. The nominal
  worst penetration is `26.98 mm` at frame 1518 between the right sphere hand
  and step 7; geometry corners still reach `59.87 mm`. No arm IK or hand-path
  repair was run.

Artifacts:

- aligned ladder/robot: `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/06_base_grounded_default_start/01_world_aligned_base_grounded/`
- grouped smoothing: `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/06_base_grounded_default_start/02_smoothing_preview/`
- default-standing start: `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/06_base_grounded_default_start/03_default_start_preview/`
- annotated replay preview: `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/06_base_grounded_default_start/04_contact_preview/`
- collision sidecar: `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/06_base_grounded_default_start/05_collision_audit/`

References checked:

- `Ladder/motions/restored_prefix_v300_stride2_slow50/smooth_start/climing_ladder_cr1_f300_2604_slow50_smooth_start_terminal_motion.npz`
- `Ladder/source/ladder/ladder/robots/cr1.py`
- `Ladder/scripts/replay_ladder_reference.py`
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`

Verification:

- [x] Base-platform bottom and ground are both at world `z=-0.03 m`.
- [x] First 25 frames are zero-joint, fixed-root, zero-velocity standing.
- [x] Actual foot collision meshes remain at or above the ground clearance
  envelope; the default/transition interval has no 1 mm ground penetration.
- [x] Contact/ladder metadata, transition dynamics, nominal collision, and
  3-by-3 geometry-envelope collision audits completed.
- [x] Existing world-transform/motion-processing tests pass (`6 passed`).
- [ ] Visual review before any command labels or train-ready packaging.

## 2026-08-21 - ZDW0818 Ladder-Top Default Stand and Command Labels

- Added `Ladder/scripts/insert_cr1_top_default_stand.py` as a separate,
  reproducible post-processing step. It inserts 100/100/100 frames of quintic
  transition, zero-joint ladder-top default standing, and transition back at
  source frame 1100. The original prefix is unchanged, and the original suffix
  is shifted by exactly 300 frames without changing joint/body pose values.
- The ladder-top default root pose is computed from URDF FK by matching the
  current two-sole midpoint and lowest sole height. The ground-level default
  root pose is not reused at the ladder top. No IK or additional smoothing is
  applied.
- Added per-frame `climb_direction`: `+1` for frames `0:1250`, `-1` from frame
  1250 onward, exactly as requested. The current runtime command state machine
  is intentionally unchanged.
- Added `zero_command_valid_mask` and component masks. Natural candidates need
  all four annotated limb contacts, base speed `<=0.10 m/s`, base angular speed
  `<=0.30 rad/s`, maximum joint speed `<=1.00 rad/s`, and a stable run of at
  least five frames. Ground default frames `0:25` and ladder-top default frames
  `1200:1300` are explicit candidates. The preview has 150 natural and 275
  total legal zero-command frames.
- Extended reference replay diagnostics to print climb direction and the
  natural/ground/top zero-command eligibility flags when `I` is pressed.

Artifact:

- `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/07_top_default_command_preview/zdw0818_capture5_v3_full_50hz_top_default100_transition100_command_annotated_preview.npz`

References checked:

- `Ladder/scripts/add_cr1_default_standing_start.py`
- `Ladder/source/ladder/ladder/utils/contact_annotation.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/observations.py`

Verification:

- [x] 2458 finite frames at 50 Hz; source pose prefix/suffix unchanged.
- [x] Frames `1200:1300` are fixed zero-joint default standing.
- [x] Command boundary and all component masks have the expected shapes/ranges.
- [x] Contact-annotation and CR1 motion-processing tests pass (`15 passed`).
- [ ] Operator visual review; preview is not yet train-ready.

### 2026-08-21 - Per-contact-segment zero-command selection

- Preserved the first threshold-based command preview and generated a separate
  `v2_segment_min_base_speed` artifact.
- Replaced threshold filtering with the requested deterministic rule: split
  all four-limb-contact frames into contiguous runs and select exactly the
  minimum `base_link` world linear-speed frame in each run. Ground frames
  `0:25` and top-default frames `1200:1300` remain explicitly valid.
- The reference contains 20 four-limb contact runs and therefore 20 natural
  zero-command keyframes. The final mask contains 145 frames including the 125
  default-standing frames. Segment bounds, selected frames, and selected
  speeds are stored in the NPZ for audit.
- No absolute velocity threshold is part of this rule. Selected frames 523,
  1701, and 1940 still have base speeds around `0.135`, `0.154`, and
  `0.110 m/s`; they remain marked by design and require visual review.
- Revalidated exact source prefix/suffix preservation, command boundary,
  default hold, segment argmin selection, and existing contact/motion tests
  (`15 passed`).

### 2026-08-21 - Same-rung contact release hysteresis

- Preserved v1/v2 command previews and generated a separate
  `v3_contact_hysteresis` artifact. Strict hand entry distances remain
  `0.052/0.050 m`; a bracketed gap is filled only while the nearest rung is
  unchanged, hand distance stays within `0.080 m`, and speed stays at or below
  `0.35 m/s`. Feet retain their original `0.145 m` entry/release distance.
- Filled right-hand frames `512:523`, `701:710`, and `925:948`, and left-hand
  frames `1685:1698`. This makes right-hand contact continuous over the
  operator-reported `508-523` interval without filling true swing phases.
- Added `contact_hysteresis_filled_indicator` and replay diagnostics that print
  raw/hysteresis/final contact state separately. Contact configuration defaults
  release distance to the existing entry distance, so older callers retain
  their former behavior unless they explicitly request hysteresis.
- Four-limb contact segments reduce from 20 to 18 after false splits merge;
  natural keyframes are `467, 508, 599, 631, 728, 757, 859, 890, 998, 1525,
  1682, 1723, 1814, 1848, 1940, 1965, 2050, 2076`, and the final legal mask
  contains 143 frames including default stands.
- Verified that all changed final contact bits equal the separately recorded
  hysteresis-filled bits (56 limb-frames total), no foot contacts changed, and
  v2/v3 joint/body poses, velocities, commands, and default masks are bitwise
  identical. Contact and CR1 motion-processing tests pass (`16 passed`).

### 2026-08-21 - Terminal stand and safe zero-command task

- Extended `insert_cr1_top_default_stand.py` with an independent terminal
  default-standing append: 50-frame quintic transition plus 25-frame fixed
  zero-joint hold. Generated the v4 stop-command preview with 2533 frames at
  50 Hz; terminal hold frames `2508:2533` have zero joint/root velocity.
- Added the default-off `RUN -> SEEK -> HOLD` planner to
  `LadderMotionCommand`. While SEEK is pending, the policy continues to see
  the recorded `+1/-1` effective direction and the reference advances. At the
  selected legal frame, the reference freezes, the effective command becomes
  zero, and all reference velocity properties become zero. External requests
  hold indefinitely until `resume_climb_command()` is called.
- Added an isolated TransformerVision stop-command task. Its corrected mixture
  is 5% direct reset at one of the 18 natural contact-segment minima, 5%
  scheduled zero request, and 90% ordinary motion. The inherited 30% frame-0,
  35% uniform-valid, and 35% adaptive reset runs first; only the direct-hold
  branch replaces that frame. Scheduled requests retain their original reset,
  sample a random future request time, and choose the first geometry-valid
  legal stop at or after it. Training holds last 1--3 seconds before resuming.
- Increased this task's episode limit from the inherited 30 s to 60 s. The new
  reference lasts 50.66 s, so frame-0 episodes can now execute the full up/down
  sequence and a maximum three-second training hold before timeout.
- Added the effective command to actor history, auxiliary next-proprioception,
  and the privileged critic without changing the actor's 85-D per-frame data
  contract. Existing tasks still use their previous constant/direction path.
- Added a grounded ThinRails geometry set for this task only, keeping physical
  boxes, visual boxes, Warp rays, height-map labels, contact targets, and the
  v4 reference's 3 cm ladder offset synchronized.
- Added visual-play controls: `0` requests the next safe stop and `C` resumes;
  automatic headless checks can use `--zero_command_at_step` and
  `--resume_command_at_step`. Logs report effective command, state, and target
  frame.
- Verification: Python compilation; 40 selected unit tests; one full
  reset/rollout/PPO-update smoke iteration; and a 4096-environment reset/state
  check. It sampled 202/212/3682 direct/seek/normal episodes, retained
  30.17%/33.98%/35.85% fixed/uniform/adaptive sampling among non-direct
  episodes, represented all 18 direct-hold targets, and reached every
  scheduled geometry-valid target with zero effective command and reference
  velocity.

References checked:

- Isaac Lab in-hand orientation command term
- whole-body-tracking motion command term
- existing Ladder motion command, TransformerVision observations, and
  ThinRails geometry/configs

### 2026-08-21 - V4 strict reset-validity audit

- Extended `detect_cr1_reset_valid_frames.py` with an explicit front
  side-board depth fraction. A value of `1/3` reproduces the grounded
  ThinRails collision boxes used by the active StopCommand task; the default
  remains `1.0` for all prior detector calls.
- Ran the established strict rule on all 2533 v4 frames: MuJoCo signed geometry
  distance, current CR1 collision URDF, 18 boxes, ground `z=-0.03 m`, 1 mm
  tolerance, 5 cm broad phase, no ignored sphere hands, and a 3-by-3
  `+/-1 deg`, `+/-1 cm` geometry grid.
- Nominal validity is `836/2533` (`33.00%`) with frame 0 valid. The four-corner
  envelope is `388/2533` (`15.32%`); the four conditioned cells contain
  `820`, `392`, `511`, and `388` valid frames.
- Nominal invalidity is dominated by ankle/step overlap: right/left ankle are
  the worst body on 745/735 frames. Worst depth is `26.98 mm` at frame 1818,
  `right_sphere_hand_link` against step 7.
- All 18 natural zero-command stop frames fail the strict 1 mm screen, as does
  the 100-frame ladder-top default hold. Only the 25 ground and 25 terminal
  default-hold frames overlap the zero-command mask and strict mask. The
  sidecar is intentionally not embedded because the current direct-hold branch
  would otherwise have no legal natural target.
- Saved the independent audit at
  `Ladder/motions/zdw0818_capture5/v3_foot160_pitch0/07_top_default_command_preview/v4_terminal_default_stop_command/reset_audit/zdw0818_capture5_v4_stop_command_thinrails_reset_audit_3grid_tol001.npz`.
- Verification: detector compilation and 22 wood-step/reset utility tests.

### 2026-08-21 - DZK 0821 capture 2 stage-1 preparation

- Archived only `/home/zhangdawei/Downloads/爬梯子08212-dzk0821.bvh` under
  `holosoma/.../demo_data/climb_ladder/dzk0821_capture2/`; the neighboring
  capture 3 file is excluded.
- Verified 10380 finite frames at 200 Hz, the expected 72-channel hierarchy,
  and all 22 required LAFAN joints.
- Generated the untrimmed, unsmoothed, right-handed, initial-Hips-XY-centered
  stage-1 keypoints with shape `(10380, 22, 3)`. The X-reflected heading
  consistency is `+0.999943`.
- Seeded only the known ladder pose for the manual alignment viewer. Human
  translation and capture-specific morphology remain unapproved; Holosoma
  retargeting is blocked on visual alignment acceptance.
- Recorded source/prepared hashes, motion statistics, morphology, and the next
  gate in the capture-local `README.md`.
- Added a capture-alignment `human_y_rotation_degrees` control. The viewer
  rotates the full motion about world Y through the first-frame Hips, persists the value in
  the alignment JSON, and `prepare_ladder_bvh.py` applies the identical
  rotation before human translation. Existing JSON files default to zero rotation.
- Verification: five preparation/morphology/rotation unit tests, Python compilation,
  and loading the new alignment JSON through the viewer parser.
- Accepted alignment: human translation `(+0.015, 0, +0.010) m` and world-Y
  rotation `-0.1 deg`; ladder remains at `(0, 0.3, 0)`, yaw `180 deg`.
- Generated a baseline input with the recorded `0.137995 m` Foot-to-ToeBase
  length and verified exact equality with the viewer transform.
- Completed 2% and 20% baseline retarget previews at 50 Hz using the established
  `4/8/3` tracking weights and `1 mm / 2 cm / 3 cm` collision/sticking settings.
  The 20% output has 519 finite frames and no solver fallback. Its worst signed
  penetration is `1.004 mm`; peak joint speed/acceleration are `5.747 rad/s`
  and `189.75 rad/s^2` at left elbow frame 416. Full retargeting remains blocked
  on visual review.
- At the operator's request, completed the full baseline retarget with the same
  fixed parameters. The output contains 2595 finite 50 Hz frames, is marked
  complete, exactly preserves the aligned source samples, and used no solver
  fallback or transition release.
- Full-motion audit: endpoint mean errors are `32.46/32.12 mm` at the ankles,
  `7.54/7.45 mm` at the toes, and `28.73/33.13 mm` at the sphere hands. Peak
  joint speed is `6.726 rad/s` at frame 1951; peak joint acceleration is
  `245.13 rad/s^2` at frame 2556; peak root step is `15.97 mm` at frame 1441.
  The worst 32-by-18 robot/ladder penetration is `1.017 mm` at frame 1608,
  right sphere hand against step 7, with no frame deeper than 2 mm.
- The full NPZ SHA-256 is
  `319ecfaa413f51a450e53cd488098aa88b0b5b8a42bdeaac654501a318a32a10`.
  It remains blocked on operator visual acceptance before IsaacLab conversion.
- Full-motion visual review found a genuine `3--4 cm` sole-marker gap over the
  ladder rungs. Measured human ToeBase targets were themselves `44--49 mm`
  above the rung, so the post-grounding source-height restoration was retained
  by the robot instead of producing support contact. The full baseline output
  is rejected as a training reference.
- Generated an isolated 20% comparison with post-grounding Z offset `0.0` and
  otherwise identical parameters. Its step-1 support gap improves from about
  `41 mm` to `12.8/11.2 mm` for left/right feet. Both hands still contact the
  intended step-4/step-5 rungs, no fallback occurs, and no penetration exceeds
  2 mm. This comparison is blocked on operator visual review before deciding
  whether another contact-height correction is needed.
- The operator saved a v2 alignment with human `+0.030 m` world-Y translation
  and `0.160 m` Foot-to-ToeBase length. Generated an isolated 10380-frame input
  that matches the alignment viewer to floating-point precision, then completed
  a separate 20% zero-offset retarget with no fallback.
- V2 increases median step-depth overlap from `37.8/52.4 mm` to
  `85.0/98.6 mm` for left/right feet while retaining step-4/step-5 hand contact.
  The vertical support gap remains `12.9/11.6 mm`. Worst penetration is
  `1.496 mm` at right wrist/step 5 with no frame deeper than 2 mm. The output is
  held at the visual-review gate before any full retarget.
- The operator then increased human world-Y translation from `+0.030 m` to
  `+0.040 m`. Isolated this as v3, regenerated the full source input, and
  completed another 20% zero-offset retarget without solver fallback.
- V3 improves step-depth overlap to `95.3/109.4 mm` (`47.4%/51.9%`) but causes
  right-ankle/step-1 penetration above 2 mm on frames 497--506, peaking at
  `2.966 mm` on frame 506. Peak speed/acceleration are `5.838 rad/s` and
  `201.47 rad/s^2`; maximum root step is `14.44 mm`. V3 is blocked from a full
  retarget pending visual review and contact correction.

## 2026-08-21 - DZK0821 Capture 2 V3 Full Retarget

- Ran the operator-requested full v3 retarget in a separately named output
  after explicitly preserving the preview's known ankle/step contact warning.
  The aligned input, earlier previews, and prior full baseline remain
  unchanged.
- Used the established fixed configuration: 50 Hz output from 200 Hz source,
  30 warmup frames, zero post-grounding offset, `0.160 m` Foot-to-ToeBase,
  zero pitch reduction, foot/toe/hand weights `4/8/3`, 1 mm penetration
  tolerance, and 2 cm foot sticking with a 3 cm fallback.
- The output contains 2595 finite frames, is marked complete, matches the
  expected grounded source samples to `2.22e-16 m`, has normalized root
  quaternions, and respects all model joint limits. SHA-256 is
  `8477196e18c78b615c03dec33841ca149759cb4694213b74945d547f7a975ed2`.
- The 3 cm foot-sticking fallback was needed at source frames `1045`, `1323`,
  `1338`, and `1381`. No penetration fallback or transition release occurred.
- Read-only derivative audit: maximum root step `17.215 mm`, root angular
  speed `2.805 rad/s`, joint speed `8.910 rad/s`, and joint acceleration
  `816.92 rad/s^2`; both joint extrema occur at right ankle-x near frame 1381.
- Read-only 32-by-18 collision audit: maximum penetration `14.264 mm` at frame
  1209 (`24.18 s`), right ankle-x versus step 3. Frames deeper than
  `2/5/10 mm` number `1303/914/118`. This output is review-only and is not
  approved for training.

Verification:

- [x] Full solver run completed and saved the expected frame count.
- [x] NPZ flags, finite values, source sampling, quaternion norm, and joint
  limits checked.
- [x] Full derivative, endpoint-error, and exhaustive robot/ladder collision
  audits completed read-only.
- [ ] Operator playback review and decision on the substantial ankle/rung
  collisions before any training-data conversion.

## 2026-08-21 - V4 Contact-Target Sync and No-Command Training Task

- Synchronized the v4 task contact offsets with the reference metadata:
  left/right hand Y `+0.21/-0.22 m`, left/right foot Y `+0.13/-0.18 m`.
  The existing StopCommand task now uses these values instead of inherited
  targets from the older motion.
- Added an independent
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-WarpTransformerVision-V4-NoCommand-v0`
  task and `cr1_transformer_vision_v4_no_command` experiment directory. The
  StopCommand task remains available as a separate version.
- The no-command task disables direct-hold and seek-stop sampling, omits the
  critic command term, and replaces the visual proprio/auxiliary command slot
  with a constant zero compatibility pad. The established 85-D encoder data
  contract is retained, but no up/down/stop information reaches the policy.
- Reference files checked: IsaacLab manager-based observation/config patterns,
  `whole_body_tracking/tasks/tracking/tracking_env_cfg.py`, and the existing
  Ladder ThinRails/TransformerVision/StopCommand configurations.
- Verification: Python compilation, final Hydra configuration inspection, and
  an 8-environment one-iteration IsaacLab PPO smoke test. The saved config has
  both stop probabilities at zero, no critic command, zero-command input mode,
  and the four synchronized target offsets.
- Added a separate nominal-geometry reset-filtered derivative for this
  NoCommand task. A `10 mm` maximum penetration rule excludes 35 frames in 9
  short runs and retains 2498/2533 frames, frame 0, all 168 zero-command-valid
  frames, and all 18 natural stable frames. The original v4 NPZ is unchanged;
  the task points to the filtered derivative. Geometry-conditioned reset masks
  remain disabled because applying the fixed reference pose to the randomized
  geometry corners rejects normal support frames.

## 2026-08-24 - V4 No-Command Nonvisual PPO Baseline

- Added the independent task
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-V4-NoCommand-NonVisual-v0`
  to test whether the current task design is learnable without vision.
- Kept the visual v4 task's filtered 2498-frame reference, grounded ThinRails
  PhysX geometry, four limb target offsets, 60-second episode, rewards,
  terminations, reset distribution, physics randomization, motor variation,
  pushes, and 0--2-step control delay. Replaced only the observation contract
  and training runner.
- The baseline uses the established deployable nonvisual policy observation
  `(660,)`, privileged critic `(631,)`, 25 actions, and the standard RSL-RL
  `OnPolicyRunner`. It has no Warp renderer, depth history, height-map labels,
  Transformer, GRU, auxiliary optimizer, or 51-D visual latent.
- Isolated output under `cr1_nonvisual_v4_no_command` and set checkpoint saving
  every 400 iterations. Existing visual tasks and checkpoints are unchanged.

Verification:

- [x] Python compilation passed for the task, registration, runner, and test.
- [x] A real two-environment IsaacLab reset/step check produced only policy
  `(2,660)` and critic `(2,631)` observations with finite values.
- [x] A real eight-environment, one-iteration rollout and standard PPO update
  completed and saved a finite smoke checkpoint.
- [ ] Long-run task completion remains the operator acceptance gate.

## 2026-08-25 - ZDW0824 Capture 1 Retarget Intake

- Archived `/home/zhangdawei/Downloads/patizi_1-0824.bvh` under the independent
  `demo_data/climb_ladder/zdw0824_capture1` tree. The archived copy is
  byte-identical to the source; the earlier `zdw0818_capture5` chain remains
  unchanged.
- Audited 10944 frames at 200 Hz with all 22 required LAFAN keypoints present,
  finite coordinates, and a 72-channel compatible hierarchy.
- Generated the stage-1 `(10944, 22, 3)` motion using only centimetre/Y-up to
  metre/Z-up conversion, initial-Hips XY centering, and the established X
  reflection. Heading consistency is `+0.999708`.
- Did not inherit the earlier capture's 0.160 m Foot-to-ToeBase adaptation and
  did not apply crop, downsampling, filtering, smoothing, hand offsets, holds,
  or contact annotations. The new source Foot-to-ToeBase length is about
  0.147215 m and remains unchanged pending operator review.

Verification:

- [x] Source/archive SHA-256 match.
- [x] Prepared shape, dtype, finite values, frame rate, skeleton compatibility,
  heading consistency, and basic temporal displacement statistics checked.
- [x] Operator human/ladder alignment saved to
  `zdw0824_capture1/alignment/ladder_alignment.json` before any retarget.
- [x] Generated the isolated aligned 200 Hz Holosoma input. It is exactly the
  stage-1 motion plus the accepted `(+0.018, 0.000, -0.045) m` translation,
  keeps the source 0.147215 m Foot-to-ToeBase geometry, and has SHA-256
  `ee63fe21e652608e9b37db6f94b921fcaea9b76dcb5dcad5ec5cd036780794eb`.
- [x] Ran the disclosed single 20% 50 Hz preview without solver or morphology
  variants. It produced 548 finite, complete frames with the correct heading
  (`forward_to_ladder=+0.995781`) and no fallback events.
- [ ] Operator review frames 400--440. The first left-foot ascent has a
  `5.329 mm` left-ankle/step-1 penetration at frame 418 and left ankle-x peaks
  of `9.450 rad/s` and `885.91 rad/s^2` near frames 419--428.
- [ ] Decide whether to retain the existing preview or rerun it with the exact
  subset grounding cancellation. Fraction-before-grounding made this preview
  uniformly `1.989 mm` lower than the accepted alignment; the corrected full
  50 Hz cancellation value is independently `-0.0114702784 m`.
- [x] At the operator's request, ran one complete 2736-frame retarget using
  the same fixed preview parameters and the exact full-sampled grounding
  cancellation `-0.011470278368214473 m`. No solver-weight, morphology,
  smoothing, clearance, or yaw variant was introduced. The finite, complete
  50 Hz result has SHA-256
  `72614c2b8cbea3a36866ad0cc7f97c823331acef7fe00ba9896170d2f0a21699`.
- [x] Verified that stored human joints equal the accepted aligned input
  sampled by four to `2.22e-16 m`, all model joint limits are respected, and
  no penetration fallback or transitioning-foot release occurred. The
  pre-approved 3 cm foot-sticking fallback was used at source frame 1880 only.
- [ ] Full-result visual/rewrite gate: left ankle-x reaches `9.450 rad/s` at
  frame 428 and `920.591 rad/s^2` around frame 738. Frame 1461 contains a
  retarget-only root branch jump of `47.172 mm` in one interval, despite only
  `2.225 mm` Hips motion in the accepted human source.
- [ ] Full signed-distance review: maximum robot/ladder penetration is
  `15.505 mm` at frame 1305 (left ankle-x versus step 3); frames deeper than
  `1/2/5/10 mm` number `2121/1684/1031/191`. Confirm the optimization and
  audit collision sets before authorizing localized repair or IsaacLab
  conversion.

## 2026-08-25 - ZDW0824 Capture 1 Leg-Only Smoothing Preview

- Preserved the complete Holosoma result and exported its 2736 frames through
  the current IsaacLab CR1 articulation at 50 Hz. The raw export differs from
  Holosoma joint/root positions by at most `1.19e-7 rad` / `1.77e-7 m`.
- With explicit operator approval, extended `smooth_resample_cr1_motion.py`
  using default-off options that preserve non-leg joints and root trajectories
  and interpolate explicitly listed isolated solver-outlier frames. Existing
  script behavior remains unchanged when the options are absent.
- Generated one independent preview: frame 1460 is replaced by the midpoint
  of frames 1459/1461 (root quaternion uses SLERP), hip/knee joints use a
  fourth-order zero-phase 3 Hz Butterworth filter, and ankle joints use 2 Hz.
  No resampling, cropping, joint-limit projection, contact annotation, or
  train-ready packaging was applied. Root and non-leg joints are bit-identical
  to the raw export outside frame 1460.
- Forward-difference leg acceleration RMS falls from `78.468` to
  `2.817 rad/s^2`; ankle acceleration RMS falls from `128.116` to
  `1.919 rad/s^2`. Leg acceleration samples over `50/100/200 rad/s^2` fall
  from `2322/1444/832` to `2/0/0`. The root's `47.172 mm` spike is removed;
  the new maximum root step is `17.722 mm`.
- Left/right hand positions outside frame 1460 change by at most about
  `0.0011/0.0010 mm` (IsaacLab versus offline-FK numerical difference). At the
  repaired frame they change by `23.09/18.76 mm`. Foot-position changes have
  about `8.1 mm` 95th percentile and `25.82 mm` maximum.
- The signed-distance audit improves frames deeper than `2/5/10 mm` from
  `1684/1031/191` to `1651/959/188`, but shallow `>1 mm` frames change from
  `2121` to `2136` and the worst penetration increases from `15.505 mm` at
  frame 1305 to `17.316 mm` at frame 1566 (right ankle-x versus step 3).

Verification:

- [x] Script compilation and all seven CR1 motion-processing tests pass.
- [x] Preview is finite, remains 2736 frames at 50 Hz, respects all URDF joint
  limits, and records the preservation/interpolation settings in metadata.
- [x] Generated a Viser-only replay bridge in `03_viser_replay_bridge/`.
  Its MuJoCo-ordered qpos exactly reconstructs the approved preview root and
  joints, while `human_joints` is copied unchanged from the complete Holosoma
  retarget. The bridge records both source paths and SHA-256 hashes and does
  not introduce another trajectory-processing step.
- [ ] Operator visual review, especially frame 1566, before any contact-aware
  repair, parameter change, world alignment, annotation, or train-ready step.

## 2026-08-25 - Human Yaw Control for BVH/Ladder Alignment

- Added an independent human `yaw (world Z) deg` control to the BVH/ladder
  alignment viewer. It rotates the complete trajectory about the first-frame
  Hips and is distinct from the existing world-Y pitch control.
- The value is saved as `human_yaw_degrees`, restored by the viewer, and
  applied by `prepare_ladder_bvh.py` when generating a Holosoma input. Existing
  alignment JSON files remain compatible and default to zero yaw.
- Added zero-yaw and 90-degree transform tests. Script compilation, all seven
  preparation tests, JSON save/restore checks, and a Viser startup smoke test
  pass. No ZDW0824 alignment value or motion artifact was changed.

## 2026-08-25 - ZDW0824 Yaw-Minus-2 Retarget Comparison

- Preserved the original accepted alignment/input/retarget and generated an
  independent candidate using the operator-saved human yaw `-2 deg` and human
  X translation `0 m`. The ladder geometry, human Z trajectory, morphology,
  and temporal sampling are unchanged.
- Ran only the approved first-20% retarget with the original fixed settings:
  50 Hz, 30 warmup frames, foot/toe/hand weights `4/8/3`, 1 mm penetration,
  2 cm foot sticking with 3 cm fallback, and the same post-grounding Z offset.
- The 548-frame result is finite, complete, within joint limits, and uses no
  fallback or transition release. Base yaw improves from `97.143` to
  `95.217 deg`, root angular-speed maximum from `1.892` to `1.404 rad/s`, and
  final cost from `0.77414` to `0.76596`.
- The left-ankle speed/acceleration spike and worst collision remain at the
  same frames and nearly the same magnitudes. Penetration over 1 mm increases
  by 11 shallow frames, while counts over 2/5/10 mm remain `104/7/0`.
  Therefore the alignment is a heading improvement, not a collision or
  smoothing fix. Full retarget remains gated on operator visual review.

## 2026-08-25 - ZDW0824 Pitch-Minus-3 / Yaw-Minus-4 Retarget Comparison

- Preserved both earlier candidates and generated an independent input from
  the operator-saved human X `-0.030 m`, world-Y rotation `-3 deg`, and
  world-Z yaw `-4 deg`. Human Z remains `-0.045 m`; no morphology, temporal,
  smoothing, or ladder-geometry change was introduced.
- Verified the 10944-frame input exactly equals the stage-1 motion under the
  saved transforms. Ladder visual and 18-box collision files are
  byte-identical to the yaw-minus-2 candidate.
- Ran one 20% / 548-frame, 50 Hz retarget with the same fixed solver settings
  and the exact subset grounding cancellation `-0.015547589182178467 m`.
  The output is finite, complete, within joint limits, and has no fallback
  events. Stored human joints reproduce the aligned sampled input to
  `2.22e-16 m`.
- Base yaw improves from `95.217` to `92.717 deg`, root angular-speed maximum
  from `1.404` to `1.369 rad/s`, and final cost from `0.76596` to `0.76256`.
  The left-ankle spike remains at frames 411--412 (`9.369 rad/s`,
  `919.58 rad/s^2`). Worst penetration is `5.859 mm` at frame 412,
  left ankle-x versus step 1; counts over `1/2/5/10 mm` are `249/51/14/0`.
- Full retarget remains gated on operator visual review around frames
  400--430.

## 2026-08-25 - ZDW0824 Pitch-Minus-3 / Yaw-Minus-4 Complete Retarget

- Completed the operator-approved 2736-frame, 50 Hz retarget with the same
  fixed preview parameters and exact full-clip grounding cancellation
  `-0.017654418872559807 m`. No morphology, smoothing, filtering, solver
  weight, clearance, self-collision, or ladder-geometry adjustment was made.
- Output SHA-256 is
  `11e443eee2978354d5330ecfd1bc73cf333b5c167920541dcdcc8a4c780cbd19`.
  The result is finite and complete, all joint limits are respected, and
  stored human joints match the aligned 200 Hz source sampled by four to
  `4.44e-16 m`.
- No penetration fallback or transition release occurred. The pre-approved
  3 cm foot-sticking fallback was used at source frames
  `420, 1339, 1354, 1517, 1551, 1576, 1800, 1839, 2164`.
- The old frame-1461 retarget-only root jump is removed (`1.602 mm` now).
  Maximum root step is `14.812 mm` into frame 2248 and closely matches the
  source Hips step of `14.763 mm`. Frame-0 base yaw is `92.685 deg` and
  maximum root angular speed is `1.911 rad/s`.
- Left ankle-x jitter remains: maximum speed/acceleration are
  `9.369 rad/s` / `919.58 rad/s^2` around frames 411--412. The 32-by-18
  collision audit finds a `13.329 mm` maximum at frame 1304, right ankle-x
  versus step 3, with `2114/1551/1036/101` frames deeper than
  `1/2/5/10 mm`. Maximum hand/wrist depth is `1.852 mm` at frame 1589.
- [ ] Keep this immutable result as a review artifact. Perform operator visual
  review before authorizing any independent leg/ankle smoothing or collision
  repair; do not label it train-ready yet.

## 2026-08-25 - ZDW0824 Raw Ladder Export and Canonical World Alignment

- Exported the approved 2736-frame Holosoma qpos to a separately named Ladder
  raw motion using offline URDF FK. The output has 25 joints and 42 bodies at
  50 Hz; no smoothing, annotation, interpolation, or IK was applied. SHA-256:
  `f96527d1d45159fe3e9c2847a34812bf34ae14d8f264966a0b4ea1db9fd06485`.
- Rigidly mapped the source ladder pose `(0, 0.3700000047683716, 0)`, yaw
  `-180 deg`, to the standard task pose `(0.38, 0.04, 0)`, yaw `90 deg`, with
  zero extra robot translation. Aligned SHA-256:
  `a0f6ff29d63172ea6d39e64f1ebf6bc108a9f643c1eddadd6eac48b8deb5b88c`.
- Joint positions/velocities remain bit-identical. All 18 ladder collider
  centers agree with the canonical July scene within `1.63e-12 m`; transformed
  body positions/orientations agree to float32 precision. The collision audit
  preserves the `13.329 mm` frame-1304 right-ankle/step-3 maximum and the
  `1551/1036/101` counts above `2/5/10 mm`.
- Updated `replay_ladder_reference.py` so `--skip_markers` accepts an
  unannotated raw motion rather than requiring fabricated contact fields.
  Annotated replay behavior is unchanged. Python compilation passes; an
  IsaacLab headless smoke loaded the standard ladder and wrote frames 0--1
  from the 2736-frame raw motion before manual shutdown.
- [ ] Operator visual review of the world-aligned, unfiltered raw motion.
  Stop before contact annotation and smoothing.

## 2026-08-25 - ZDW0824 Ladder-Base Grounding Transform

- At the operator's direction, excluded the earlier V4 environment convention
  from this capture's reference comparison and grounded the physical ladder by
  mapping its pose from `(0.38, 0.04, 0)`, yaw `90 deg`, to
  `(0.38, 0.04, -0.03)`, yaw `90 deg`.
- Applied the same rigid `-0.03 m` world-Z transform to every robot body so the
  robot/ladder relative trajectory is unchanged. Joint positions and velocities
  are bit-identical; body X/Y are unchanged; all arrays remain finite at 2736
  frames and 50 Hz.
- Preserved the preceding aligned raw motion. The independent grounded raw
  artifact is
  `Ladder/motions/zdw0824_capture1/pitchminus3_yawminus4_xminus030/`
  `03_ladder_grounded_raw/`
  `zdw0824_capture1_pitchminus3_yawminus4_xminus030_`
  `ladder_grounded_raw_50hz_motion.npz`, SHA-256
  `cf3489d1123fa724e89ffcff215dcad1a914b1ff3e7ec52b4f3d066fece4b44b`.
- [ ] The grounded raw artifact still requires an independently approved
  leg/ankle smoothing pass, contact-offset fitting, contact annotation, and
  reset-validity generation. No smoothing or IK was applied in this step.

## 2026-08-25 - ZDW0824 Grounded Leg Smoothing and Ground Audit

- Generated an independent 50 Hz joint-space smoothing artifact using 3 Hz
  hip/knee and 2 Hz ankle fourth-order zero-phase filters. Root and non-leg
  joints are preserved, and no IK, outlier interpolation, annotation, or
  train-ready packaging is included. SHA-256:
  `f07d65851bddacc5feb5384c952dd70b5eaf43f54001d79358ab479e366ae17e`.
- Joint/leg/ankle acceleration RMS changes from `9.007/11.658/17.053` to
  `4.425/2.775/1.879 rad/s^2`. Root and sphere-hand trajectories are exactly
  unchanged.
- Exact foot collision-mesh evaluation against the `z=-0.03 m` task ground
  finds frame-0 left/right penetration of `55.9/46.4 mm`. The principal
  initial below-ground runs are frames `0--371` and `0--516`; the first
  left/right ladder-foot contacts occur near frames `401/556`, while hand
  contact begins near frame `219`.
- A whole-body root-Z clearance pass is rejected for this stage because it
  would displace already-contacting hands by about `28--34 mm` during the
  initial mixed-support interval. A ground-only leg IK mode that preserves
  root, torso, arm, sphere-hand, and ladder trajectories is proposed but has
  not been executed pending operator approval.

## 2026-08-25 - ZDW0824 Hybrid Ground-Clearance Repair

- Extended `stabilize_cr1_contact_motion.py` with a default-off hybrid
  ground-clearance mode. Existing contact stabilization remains unchanged.
  The new mode evaluates the actual ankle collision meshes, shares common
  clearance with root Z only under a disclosed schedule, and applies bounded
  leg-only IK to the residual. It never calls arm IK.
- Rejected intermediate approaches that used pure leg IK, unlimited two-sided
  anticipation, binary correction gating, or unconstrained zero-phase filter
  tails. These respectively caused large standing-pose changes, correction
  during ladder support, acceleration spikes, or deeper step penetration.
  The accepted implementation uses 15-frame look-ahead plus a 15-frame
  quintic activity window derived from each exact ground-penetration run.
- Generated the separate 2736-frame, 50 Hz artifact under
  `05_ground_clearance_hybrid/`, SHA-256
  `fb83997f5719f4bc6d331acd00837ca981da2decb5826fce73011f5ef3dab4e4`.
  Settings are ground `-0.03 m`, margin `7 mm`, and root schedule
  `160/219/2494/2554`.
- Both feet have zero residual ground penetration. Root XY/orientation,
  non-leg joints, and protected-frame (`219--2494`) sphere-hand positions are
  unchanged. Root-Z correction is at most `62.582 mm`; joint correction
  RMS/maximum are `0.05213/0.50386 rad`.
- Leg acceleration RMS changes from `2.77476` to `2.76459 rad/s^2`; the
  existing `49.52 rad/s^2` maximum and counts above `20/30/40 rad/s^2` do not
  increase. The nominal ladder audit changes `>1/2/5/10 mm` frames from
  `2249/1596/976/127` to `2240/1596/976/127`, preserves the `13.548 mm`
  maximum, and has no frame worse by more than `0.5 mm`.
- The preceding 3 Hz smoothing introduced a separate `0.005541 rad`
  left-hip-X lower-limit overshoot near frame 1150. The hybrid pass preserves
  rather than enlarges it. Visual review, explicit limit-repair approval, and
  formal contact annotation remain pending.

Verification:

- [x] Script compilation and 16 relevant motion-processing tests pass.
- [x] Official output arrays exactly match the audited v7 candidate and are
  finite.
- [x] Exact ground and nominal 18-box ladder audits use the grounded ladder
  pose `(0.38, 0.04, -0.03)`, yaw `90 deg`.

## 2026-08-27 - ZDW0824 Collision-Protected Non-Leg Smoothing Preview

- Diagnosed the remaining high-frequency motion in the ground-clearance
  artifact. The largest residual accelerations are in unsmoothed shoulders,
  elbows, and waist rather than the already filtered ankles. Added a
  default-off `--preserve_leg_joints` option to
  `smooth_resample_cr1_motion.py` and exact same-rate root-sample preservation.
- Generated an intermediate 5 Hz non-leg candidate under
  `06_nonleg_smoothing_5hz/`. It keeps root and all hip/knee/ankle joints
  unchanged and reduces all-joint acceleration RMS from `4.422` to
  `3.129 rad/s^2`, but it is rejected as a final artifact because 107 frames
  worsen nominal penetration by more than 1 mm and the worst increase is
  `5.144 mm` at a sphere-hand/step event.
- Updated `blend_cr1_motion_collision_aware.py` to use the current 18-box
  collision builder, explicit ladder/ground poses, an optional legacy
  sidecar, candidate-processing provenance, and float32-quantized collision
  checks matching the saved NPZ. The unquantized implementation was rejected
  after an independent audit exposed a float32-only wrist/side-board contact.
- Generated the separately stored collision-protected preview under
  `07_nonleg_smoothing_5hz_collision_aware/`, SHA-256
  `ccf0010d0ed91b55ddd8af4086907e0046d32d89fdec30b895f44afb1a567ba8`.
  Mean smoothing blend is `0.253`; 1987/2736 frames remain below 0.1 because
  contact-rich intervals conservatively fall back toward stage 05.
- All-joint acceleration RMS changes from `4.422` to `4.205 rad/s^2` and jerk
  RMS from `139.96` to `127.03 rad/s^3`; the `73.2 rad/s^2` maximum is not
  materially reduced. Left/right sphere-hand displacement has 95th percentile
  `2.120/2.284 mm` and maximum `11.373/12.258 mm` relative to stage 05.
- Root and leg joint samples are preserved; foot poses differ only by offline
  FK roundoff (`5.8e-5 mm`). Both foot collision meshes remain above the
  `z=-0.03 m` ground. The existing `0.005541 rad` left-hip-X limit excess is
  unchanged rather than enlarged.
- The independent nominal audit keeps the `13.548 mm` frame-1147
  right-ankle/step-2 maximum. Counts deeper than `1/2/5/10 mm` change from
  `2240/1596/976/127` to `2089/1596/976/127`; no frame worsens by 0.1 mm, and
  a second pairwise audit finds zero violations of the source-or-1-mm bound.
  This remains a visual-review preview, not a train-ready reference.

Verification:

- [x] 19 focused motion-processing and collision-blending tests pass.
- [x] Independent float32 nominal and pairwise collision audits pass.
- [x] Ground clearance, joint limits, root/leg preservation, hand displacement,
  and derivative statistics checked.

## 2026-08-27 - ZDW0824 Groupwise Full-Body Smoothing Preview

- Extended `blend_cr1_motion_collision_aware.py` with an opt-in `groupwise`
  mode. It processes left arm, right arm, torso, right leg, and left leg as
  disjoint joint groups while retaining the stage-05 root trajectory. No arm
  trajectory IK or contact-target editing is used. The existing `global` mode
  remains the default.
- Replaced the slow sparse bounded least-squares implementation of the same
  convex alpha objective with bounded L-BFGS-B. On the saved stage-09 bounds,
  old/new alpha RMS and maximum differences are `5.2e-5/2.9e-4`, while the
  objective is marginally lower and solve time falls to about `0.09 s`.
- Compared full-body Butterworth candidates. The accepted preview uses 4 Hz
  non-leg, 2.5 Hz hip/knee, and 1.8 Hz ankle cutoffs. A stronger 3.5/2.2/1.5 Hz
  candidate is retained only for comparison because it improves jerk RMS by
  just `1.31 rad/s^3` while increasing right-foot p95/max displacement from
  `3.88/12.64 mm` to `6.11/23.16 mm`.
- Generated the selected preview under
  `10_fullbody_groupwise_collision_aware/`, SHA-256
  `522a855ed85f31feef2c4f210836d669c9c87641d69614a2fd40d25f9582eeb4`.
  Mean retained alpha is `0.365`; left/right arm, torso, right-leg, and
  left-leg means are `0.332/0.348/0.259/0.521/0.363`.
- All-joint acceleration RMS/maximum change from `4.422/73.22` to
  `3.764/59.57 rad/s^2`; jerk RMS changes from `139.96` to
  `105.64 rad/s^3`. Left/right sphere-hand p95 displacement is
  `3.11/3.32 mm`; left/right sole p95 displacement is `2.44/3.88 mm`.
- Independent float32 pairwise audit finds zero source-or-1-mm violations and
  zero ground penetration. The maximum ladder penetration remains the
  stage-05 `13.548 mm`; counts deeper than `1/2/5/10 mm` change from
  `2240/1596/976/127` to `1985/1579/959/125`. The existing
  `0.005541 rad` left-hip-X limit excess is unchanged.

Verification:

- [x] 22 focused motion-processing and collision-blending tests pass.
- [x] Independent reloaded-NPZ pairwise collision and ground audit passes.
- [x] IsaacLab loads the 2736-frame motion and replays frames 0--1 at 50 Hz.
- [ ] Operator visual review and contact annotation remain pending; this is
  not yet a train-ready artifact.

## 2026-08-27 - ZDW0824 Support-Constrained Leg Smoothing Diagnostic

- Diagnosed remaining visible leg motion as a combination of joint jitter and
  support-foot drift. Stage-03/04/10 mean inferred support-path rates are
  approximately `0.058/0.121/0.120 m/s`; stage-04 independent joint filtering
  is the main source of the slow foot wandering.
- Added `transfer_cr1_joint_delta.py` to transfer an explicitly recorded joint
  processing delta between matching motions, and
  `optimize_cr1_support_motion.py` to solve a bounded support-foot tracking
  correction with magnitude, velocity, acceleration, and jerk regularization.
  The optimizer changes only leg joints and enforces the source joint-limit
  excess envelope inside the solve.
- Reconstructed a coordinated, ground-clear baseline under stage 16 and a
  common-4-Hz leg candidate under stage 17. The selected stage-18 constrained
  result uses weights `300/1/5/60/120`; support-foot error p95/max is
  `3.02/9.17 mm`, path rate is `0.0891 m/s`, leg jerk RMS is
  `73.01 rad/s^3`, and exact ankle collision meshes have zero ground
  penetration. Root and sphere-hand trajectories are unchanged.
- Restored the stage-10 upper body exactly and applied existing groupwise
  source-or-1-mm collision protection. Mean retained right/left-leg alpha is
  only `0.497/0.353`, so the stage-20 collision-safe result improves support
  path rate only from `0.1204` to `0.1075 m/s` and horizontal core ranges from
  `26.4/23.6` to `21.0/21.0 mm`; all-joint jerk RMS increases from `105.64` to
  `110.22 rad/s^3`.
- Stage 20 is retained for operator visual comparison but is not promoted to
  train-ready. A stronger solution must include sole pose/collision geometry
  in the same trajectory objective rather than relying on post-hoc collision
  blending.

Verification:

- [x] New scripts compile; 23 focused motion-processing tests pass.
- [x] Bounded solver does not enlarge the existing left-hip-X limit excess.
- [x] Stage-20 groupwise collision protection passes the stage-10
  source-or-1-mm contract; exact ground penetration is zero.
- [ ] Operator visual comparison of stage 10 and stage 20.

## 2026-08-27 - ZDW0824 Contact-Phase-Aware Collision Smoothing Preview

- Extended `blend_cr1_motion_collision_aware.py` with a default-off intended
  support allowance. It consumes explicit contact annotations and relaxes a
  pair only for the required sphere hand/ankle on its exact target-rung top.
  MuJoCo environment-side closest points and closest axes classify the box
  face; wrists, knees, wrong rungs, rails, ground, and non-top faces remain
  strict.
- The configured `3 mm` allowance is relative to the stage-05 pair depth,
  because intended support can already have nonzero collision-geometry depth.
  An independent final audit reports zero invalid frames, zero non-support
  regressions, 2593 relaxed pairs over 1578 frames, and a maximum relative
  increase of `2.958 mm`.
- Added a source joint-envelope guard to prevent blending from enlarging any
  pre-existing URDF limit excess. Stage 21 preserves the existing
  `0.005541 rad` left-hip-X excess exactly.
- Generated the separate stage-21 artifact under
  `21_contact_phase_aware_fullbody_collision_aware/`, SHA-256
  `1435f62fc5a5af9ed40f52b392720466d550579aa9db38f14dbd63677012ca47`.
  Mean retained left/right-arm, torso, right-leg, and left-leg coefficients
  are `0.761/0.780/0.607/0.981/0.809`.
- Full-trajectory joint acceleration RMS changes from stage 10's `6.511` to
  `5.051 rad/s^2`; jerk RMS changes from `489.329` to
  `348.685 rad/s^3`. Remaining peak events are associated with strict wrist,
  knee, and side-board constraints, while the unchanged root trajectory still
  carries its original Cartesian jitter.

References checked:

- `detect_cr1_reset_valid_frames.py` collision model and pair construction
- `ladder/utils/wood_step_ladder.py` 18-box names, poses, and dimensions
- ladder-climb runtime contact-surface classification in `mdp/rewards.py`

Verification:

- [x] Script compilation and 18 focused tests pass.
- [x] Reloaded float32 NPZ passes the contact-aware pairwise audit.
- [x] Joint-limit envelope and annotation/ladder-pose consistency checked.
- [ ] Operator visual review; stage 21 is not yet train-ready.

## 2026-08-27 - ZDW0824 Root-Only Collision-Aware Smoothing Preview

- Generated a 5 Hz root-only candidate from stage 21 with
  `smooth_resample_cr1_motion.py`. All 25 joint samples are bit-identical;
  only root translation/orientation changes.
- Added the default-off `--collision_limit_file` option to
  `blend_cr1_motion_collision_aware.py`. The blend source can now remain
  stage 21 while pairwise allowed depths are still measured against stage 05,
  preventing repeated processing from cumulatively adding support slack.
- Generated stage 22 under `22_root5_collision_aware/01_collision_aware/`,
  SHA-256
  `8529c9195b221919fbc9c79b52be8339874c9402321a76380a31db37eeedecd3`.
  Mean retained root alpha is `0.621`; 834 frames are below `0.1`.
- Root linear/angular jerk vector RMS changes from
  `503.451/539.304` to `293.305/299.766`; five-body Cartesian jerk RMS changes
  from `866.383` to `433.389`. Joint positions and joint jerk remain exactly
  unchanged from stage 21.
- Independent reloaded-NPZ audit reports zero invalid frames, zero ground
  penetration, and zero non-support collision regressions. Required-contact
  limb displacement has `2.9--3.3 mm` p95 and `9.02 mm` maximum.

Verification:

- [x] Script compilation and 18 focused tests pass.
- [x] Candidate joints are bit-identical to stage 21.
- [x] Collision limit remains stage 05 plus only the existing target-top
  support allowance.
- [ ] Operator visual review; collision fallback still leaves substantial
  root jerk, so stage 22 is not train-ready.

## 2026-08-27 - ZDW0824 Bounded-Penetration Stage-23 Preview

- Added a default-off `--ladder_depth_slack` option to
  `blend_cr1_motion_collision_aware.py`. It applies only to 18-box ladder
  collision pairs and explicitly excludes ground; target-top support uses the
  larger of this general allowance and its existing support allowance.
- Generated an independent stage-23 branch with `2 mm` general ladder slack,
  `3 mm` intended-support slack, and stage 05 as the immutable collision-limit
  motion. The groupwise joint candidate retains mean coefficients
  `0.978/0.988/0.883/0.987/0.986`; the following root-only 5 Hz blend retains
  mean alpha `0.951`, with only 3 frames below `0.1`.
- The final artifact is under
  `23_relaxed2_fullbody_root5/02_root5_collision_aware/`, SHA-256
  `838a39dac050be686299eace0bfb7713c858af387497630f279ee6963edf146a`.
  Joint jerk RMS is `118.514 rad/s^3`, root linear jerk vector RMS is
  `118.053 m/s^3`, and five-body Cartesian jerk vector RMS is
  `222.064 m/s^3`.
- Independent reloaded-NPZ audit reports zero contract violations and zero
  ground penetration. General ladder relaxation affects 848 pairs over 619
  frames, with p95/max added depth `1.349/1.999 mm`; target support reaches its
  configured `3.000 mm` maximum. There are 654 nominal frames with no exact
  collision deeper than `1 mm`.

Verification:

- [x] Script compiles and 19 focused tests pass.
- [x] Ground exclusion and ladder-pair slack have a focused unit test.
- [x] Serialized output passes the independent Stage05-relative 2/3-mm audit.
- [x] IsaacLab headless replay loads the final 2736-frame file and advances frames 0--1 at 50 Hz.
- [ ] Operator visual review; stage 23 is not yet marked train-ready.

## 2026-08-27 - ZDW0824 Stage-24 Standing-Transition Preview

- Added `optimize_cr1_standing_transition.py`, a local sparse trajectory
  optimizer for root translation and both CR1 legs. It tracks both ankle
  collision-body poses and regularizes correction position, velocity,
  acceleration, and jerk without changing root orientation or non-leg joints.
- Applied it only over frames 80--218 with full tracking over 100--190. The
  final collision-aware artifact is under
  `24_standing_transition_root_leg/01_collision_aware/`, SHA-256
  `6e300235bc04c2eb6f25399b9f6601758a779c27be7a30412b64ab6cf8722069`.
- Core left/right foot XYZ ranges reduce from
  `[32.15,13.25,26.77]/[40.71,16.05,28.14] mm` to
  `[2.63,3.12,2.48]/[3.43,2.77,2.13] mm`; maximum one-frame translation
  reduces from `5.13/6.04` to `1.20/1.07 mm`.
- Root correction p95/max is `13.00/13.31 mm`; leg correction absolute
  p95/max is `0.0723/0.1443 rad`. The local leg jerk peak is
  `117.6 rad/s^3`, below the unchanged `315.2 rad/s^3` right-elbow peak.
- Collision-aware alpha is exactly 1 for all frames. Local ground overlap is
  at most `0.616 mm`, within the selected 1-mm preview tolerance. Non-leg
  joints and frames outside the local window remain exactly unchanged.

Verification:

- [x] Script compilation and 29 focused motion-processing tests pass.
- [x] Joint limits, finite arrays, frame boundaries, and exact foot collision
  geometry checked after float32 serialization.
- [x] Stage05-relative ladder collision audit retains the full candidate.
- [x] IsaacLab headless replay loads the 2736-frame output and advances frames.
- [ ] Operator visual review of frames 70--240 and the complete motion.

## 2026-08-27 - ZDW0824 Stage-25 Contact and Reset Packaging

- Recomputed contact annotations from the operator-accepted stage-24 body
  trajectory with the stored ladder pose, limb target offsets, entry/release
  thresholds, and temporal cleanup settings. No trajectory or target offset was
  changed. Final left/right hand and left/right foot contact counts are
  `1312/1656/1816/1506`; only `13/7/5/9` indicator frames differ from the stale
  stage-24 annotations, all at contact-run boundaries.
- Re-ran the stage-05-relative collision audit using the new indicators and the
  existing `2 mm` general ladder and `3 mm` intended-support allowances. Every
  frame retains alpha `1.0`; joint, root/body, velocity, and contact arrays are
  bit-identical between the reannotated and collision-audited files.
- Generated strict nominal and 3-by-3 geometry-conditioned reset fields with
  the full ladder asset, no ignored sphere hands, and `1 mm` penetration
  tolerance. Nominal validity is `654/2736` and includes frame 0. The
  four-corner `+/-1 degree`, `+/-1 cm` envelope is `443/2736` and is diagnostic
  only; training should use the nominal mask for the fixed reference pose.
- Saved the train-ready artifact under
  `25_final_contact_reannotation/02_reset_validity/`, SHA-256
  `1f209409263c0133e62d23b1ef9ea00840e3a33e20d6367b4358455d13c2ed27`.

Verification:

- [x] Contact reannotation changes only six derived contact arrays.
- [x] Collision re-audit retains the complete accepted trajectory.
- [x] Embedded reset fields preserve all motion and contact arrays exactly.
- [ ] Final marker replay and task-config integration.

## 2026-08-27 - ZDW0824 Stage-26 Finite-Tread Contact-Phase Preview

- Added `ContactPhaseAnnotationCfg` and `annotate_ladder_contact_phases` to
  `contact_annotation.py`. The additive annotation uses the finite tread top
  rectangle, sphere-bottom/sole points, per-limb entry/release thresholds,
  lower-clearance bounds, same-rung hysteresis, and temporal cleanup. It has
  no speed gate and does not modify stable support contacts.
- Added `annotate_cr1_contact_phases.py` to reproduce the Stage-26 labels from
  any compatible contact-annotated CR1 NPZ. The CLI reads the stored ladder
  target geometry rather than applying a separate preset.
- Updated `replay_ladder_reference.py`: red denotes existing stable support,
  orange denotes surface phase only, and gray remains inactive. Diagnostics
  now print both contracts, the phase rung, finite-surface distance, and
  signed top-plane clearance.
- Generated `26_surface_contact_phase/zdw0824_capture1_stage26_surface_contact_phase_preview_50hz_motion.npz`,
  SHA-256 `625aeaa9b03bbb1c9dad52121d1c3b1b3ea797d15bc9f17df529ee3bee8ec532`.
  Phase counts are `2055/2031/1865/1540`, including
  `743/375/49/43` phase-only frames.

Verification:

- [x] 14 focused contact-annotation tests pass.
- [x] All 140 Stage-25 fields are array-identical in Stage 26.
- [x] Stage-25 SHA remains `1f209409263c0133e62d23b1ef9ea00840e3a33e20d6367b4358455d13c2ed27`.
- [x] Headless IsaacLab replay loads the Stage-26 file and prints the new diagnostics.
- [ ] Operator visual approval before any task/reward/termination integration.

## 2026-08-27 - ZDW0824 Stage-27 Direction-Aware Hand Phase Preview

- Added optional `DirectionalContactPhaseCfg` processing on top of the finite
  tread annotation. Selected limbs follow an explicit ascent/neutral/descent
  timeline, cannot re-enter a completed rung in the same direction, and leave
  a reached surface using confirmed distance-rate or outward-normal motion.
- Configured only `left_hand/right_hand` for this artifact. Foot analysis found
  correctly ordered `0-1-2-1-0` runs with only 4--9 frames of release tail, so
  every foot phase field remains bit-identical to Stage 26.
- Updated replay markers with a purple `legacy support only` state. This makes
  stable target-point labels that are intentionally rejected by the new phase
  visible during review instead of letting red support markers hide them.
- Generated `27_directional_contact_phase/zdw0824_capture1_stage27_directional_hand_contact_phase_preview_50hz_motion.npz`,
  SHA-256 `1f9a5ce4b9c55ba4c6b64379b728ec6cb2f615945b44fb44578d8c9c2b0b06cb`.

Verification:

- [x] 15 focused contact-annotation tests pass.
- [x] All 140 Stage-25 fields and both Stage-26 foot phase columns remain exact.
- [x] Stage-27 loads in IsaacLab and prints direction-aware frame diagnostics.
- [ ] Operator review of frames 1080--1280, 1420--1840, and the complete motion.

## 2026-08-27 - ZDW0824 Stage-28 Continuous Same-Tread Phase Preview

- Changed directional departure from immediate release-bound failure to a
  five-frame gap grace. At 50 Hz, up to `0.1 s` of jitter or small limb motion
  remains part of the current tread phase; a sixth consecutive departure frame
  confirms release. Approaching the next ordered tread still switches phase
  immediately, and completed-tread locking remains active.
- Short non-directional gaps are now closed whenever both sides belong to the
  same tread, including transient frames outside the geometric release window.
  This applies the requested continuity rule to feet without changing their
  already-continuous labels in this reference.
- Generated
  `28_continuous_contact_phase/zdw0824_capture1_stage28_continuous_contact_phase_preview_50hz_motion.npz`,
  SHA-256 `b211a2fc9946c6be5c0c35beb270559988cfcde86b7d16230097235bfd70c8c3`.
  Relative to Stage 27, it adds `32/28/0/0` left-hand/right-hand/left-foot/right-foot
  phase frames and removes none. Final counts are `1895/1960/1865/1540`.

Verification:

- [x] 16 focused contact-annotation tests pass.
- [x] All common fields except the four directional phase/departure fields remain exact.
- [x] Every limb/tread occurrence is one continuous phase run.
- [x] Headless IsaacLab replay loads Stage 28 and advances frames.
- [ ] Operator visual review before task integration.

## 2026-08-27 - ZDW0824 Stage-29 Stable-Support Subset Invariant

- Made stable support a hard subset of the geometric contact phase. Whenever
  `contact_indicator` is active, `contact_phase_indicator` is now forced active
  and inherits the same `contact_rung_id`, including directional neutral frames
  and ordinary foot-phase boundaries.
- Generated
  `29_support_subset_contact_phase/zdw0824_capture1_stage29_support_subset_contact_phase_preview_50hz_motion.npz`,
  SHA-256 `7e7e7e81b71be665b997b7baa7552d133609c0f308d47803bc883458b7c3c4d8`.
  Final phase counts are `1899/1965/1865/1549`.
- Frame 246 is explicitly verified as left/right hand support and phase on rung
  3. Across all 2736 frames, stable-support-without-phase and support/phase
  rung-ID mismatch counts are both `0/0/0/0`.
- Changed replay color priority so every active contact phase is yellow/orange,
  including stable-support frames. Purple now exclusively exposes an invalid
  legacy-support-only state; red no longer hides the phase marker.

Verification:

- [x] 16 focused contact-annotation tests pass, including support during a directional neutral segment.
- [x] The stable-support subset and same-rung invariants pass for all four limbs.
- [ ] Operator visual review; marker occlusion must be distinguished from missing array labels.

## 2026-08-27 - ZDW0824 Stage-30 Left-Hand Recontact Bridge

- Preserved the non-directional finite-surface phase as a fallback envelope for
  directional recontact repair. When one base-surface run contains two
  directional phase runs on the same tread and no intervening tread, the full
  base run is restored. This uses a later stable-support return as evidence
  that the intermediate reference lift is an annotation/tracking defect rather
  than an intended swing.
- The left-hand top-tread phase is now continuous on rung 6 from frame
  `1087--1265`, adding 121 phase frames relative to Stage 29. Frame 1150 is
  explicitly active on rung 6. Right hand and both feet remain unchanged.
- Generated
  `30_recontact_bridged_contact_phase/zdw0824_capture1_stage30_recontact_bridged_contact_phase_preview_50hz_motion.npz`,
  SHA-256 `b48ebbb9328c0040a8fad5383507b03ae92ce54f7527905b0617b96659d7734e`.
- Moved only the replayed foot target markers `0.10 m` heelward along the
  horizontal ladder outward normal (world `-X` here). Saved target coordinates,
  diagnostics, motion, and training data are unchanged.

Verification:

- [x] 17 focused contact-annotation tests pass, including directional recontact bridging.
- [x] Stable support remains a phase subset with matching rung IDs for all limbs.
- [x] Stage-30 phase changes are exactly `121/0/0/0` added frames and no removals.
- [ ] Operator visual review of frames 1080--1280 and foot marker visibility.

## 2026-08-27 - ZDW0824 Stage-31 Runtime Contact Contract and Full-State PPO

- Promoted the operator-approved finite-tread phase to the runtime
  `contact_indicator/contact_rung_id` keys. The duplicate final
  `contact_phase_indicator/contact_phase_rung_id` keys are omitted; Stage 30
  remains the immutable source containing both reviewed contracts.
- Generated
  `31_train_ready_contact_phase_contract/zdw0824_capture1_stage31_train_ready_contact_phase_50hz_motion.npz`,
  SHA-256 `4db41141a37e80935e882f99243ced400c6555ca2d052ccac27f8555ea77851f`.
  Runtime contact counts are `2020/1965/1865/1549`; the existing nominal reset
  mask remains `654/2736`.
- Added the independent task
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-ZDW0824-NoCommand-FullState-v0`.
  It uses the Stage-31 reference, its exact ladder geometry and target offsets,
  no camera or command/stop mechanism, and identical 780-D policy/critic state
  groups. Existing V4, visual, and historical tasks are unchanged.

References checked:

- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_transformer_vision_thin_rails_ppo_env_cfg.py`

Verification:

- [x] Stage-31 runtime labels exactly equal Stage-30 reviewed phase labels.
- [x] No motion/reset/common field changed during promotion.
- [x] 17 focused contact-annotation tests pass.
- [x] A four-environment, one-iteration Isaac Sim smoke run completed with
  780-D actor, 780-D critic, 25-D action, and all managers active.

## 2026-08-27 - ZDW0824 Stage-33 Relaxed Reset Coverage

- Recomputed nominal and geometry-conditioned reset masks against the current
  grounded ThinRails collision model, using the existing three-by-three
  `+-1 deg`/`+-1 cm` ladder grid and a `5 mm` reset-only penetration tolerance.
- Generated
  `33_train_ready_contact_phase_reset5mm/zdw0824_capture1_stage33_train_ready_contact_phase_reset5mm_50hz_motion.npz`,
  SHA-256 `41c2e4efa0fc9945d6399bbf98dca4433400e7c411d45a97aff981ee5da369d0`.
  Nominal reset coverage is now `1775/2736` (`64.88%`), while frame 0 remains
  valid throughout the sampled geometry grid. Motion, velocity, and runtime
  contact arrays are unchanged from Stage 31.
- The newly accepted `2--5 mm` states are predominantly intended tread
  support: `625/635` additions over the 2 mm mask have the indicated hand/foot
  on its specified target tread. Four-limb-contact reset states increase to
  `599` frames.
- Updated the ZDW0824 no-command full-state task to Stage 33 and set fixed
  frame-0 root-position perturbations to `x,y in [-0.06, 0.06] m`. Hand/foot
  target offsets remain unchanged pending visual review.

References checked:

- `Ladder/scripts/detect_cr1_reset_valid_frames.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_zdw0824_no_command_full_state_env_cfg.py`

Verification:

- [x] Stage-33 indicator, rung-ID, joint, velocity, body-position, and body-orientation arrays match Stage 31.
- [x] The task configuration compiles and resolves the Stage-33 motion path.
- [x] A four-environment, one-iteration full-randomization smoke run completed
  with reset-valid fraction `0.6488`, 780-D actor/critic observations, and 25-D actions.

## 2026-08-27 - ZDW0824 Stage-34 Default Start and Refitted Limb Targets

- Fitted target offsets from the accepted reference's finite-tread contact
  phases, then rounded them to conservative points inside the tread:
  left/right hand `(0.000, +/-0.230, 0.045) m`, left foot
  `(-0.040, 0.110, 0.000) m`, and right foot
  `(-0.050, -0.160, 0.000) m`. Contact-phase median target errors change from
  `5.10/4.47/4.09/10.15 cm` to `3.29/3.07/3.01/4.96 cm` for
  left hand/right hand/left foot/right foot.
- Extended `add_cr1_default_standing_start.py` so all existing frame-indexed
  fields remain aligned when an annotated motion receives a standing prefix.
  The Stage-34 reference uses the July workflow's 25-frame zero-joint grounded
  hold and 50-frame quintic transition. No arm IK or hand-path design is used.
- Kept the operator-reviewed Stage-33 runtime
  `contact_indicator/contact_rung_id` contract exactly, shifted by the new
  25-frame prefix. The prefix has no ladder contact; runtime contact counts
  remain `2020/1965/1865/1549`.
- Recomputed the three-by-three `+-1 deg`/`+-1 cm` reset audit with the current
  ThinRails geometry and 5-mm reset tolerance. The final artifact is
  `34_default_standing_start_new_targets_reset5mm/`
  `zdw0824_capture1_stage34_train_ready_default_start_new_targets_reset5mm_50hz_motion.npz`,
  SHA-256 `64650dc3ae2aba6a8e7412bb0e4022f2f8fb3140a15709b8aefedc5954683721`.
  It contains 2761 frames at 50 Hz; all 25 default hold frames are reset-valid,
  and nominal reset coverage is `1800/2761` (`65.19%`).
- Updated the independent no-command full-state task to use Stage 34 and its
  new target offsets. Existing Stage 33 and all earlier artifacts remain
  unchanged.

References checked:

- `Ladder/scripts/add_cr1_default_standing_start.py`
- `Ladder/scripts/preprocess_ladder_motion.py`
- `Ladder/scripts/annotate_cr1_contact_phases.py`
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_zdw0824_no_command_full_state_env_cfg.py`

Verification:

- [x] First 25 frames have zero joint, root-linear, and root-angular velocity;
  their joint positions are exactly the CR1 zero-joint default pose.
- [x] Stage-34 runtime contact labels after frame 25 are array-identical to
  Stage 33, and all 182 fields have consistent frame dimensions.
- [x] 25 focused contact-annotation and motion-processing tests pass.
- [x] A four-environment, one-iteration full-randomization IsaacLab smoke run
  completed with 780-D actor/critic observations, 25-D actions, and runtime
  reset-valid fraction `0.6519`.

## 2026-08-28 - ZDW0824 Stage-36 Ladder-Top Default Stand

- Added `insert_cr1_middle_default_stand.py` for one reproducible operation:
  insert a zero-joint CR1 default stand between two source frames without IK
  or modifying any source pose. At source frame 1400 it inserts 100 frames of
  quintic transition in, 100 frames of hold, and 100 frames of quintic
  transition out. The descent boundary shifts from 1425 to 1725.
- Applied the approved world-Z profile only to the inserted interval:
  `0 -> 5 mm`, fixed `5 mm`, then `5 -> 0 mm`, using the same quintic weights.
  Source frame 1399 and shifted source frame 1400 retain zero offset and all
  source joint/body/contact arrays are bitwise preserved outside the insert.
- Rebuilt the finite-tread contact phase while preserving the reviewed source
  `contact_indicator/contact_rung_id` contract. The default hold requires both
  feet on rung 2 and neither hand; all 100 hold frames have zero joint speed.
- Recomputed the existing three-by-three `+-1 deg`/`+-1 cm` ThinRails reset
  audit at 5-mm tolerance. Nominal valid frames are `1970/3061` (`64.36%`):
  ascent `949`, top interval `183`, and descent `838`. The new interval adds
  35/100/35 valid transition-in/hold/transition-out frames, increasing the top
  region by 170 valid reset states without changing ascent or descent counts.
- Registered the separate 90-second task
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-ZDW0824-`
  `NoCommand-FullState-MiddleStand-v0` and a separate experiment directory.
  The Stage-34 task and motion remain unchanged.

Artifact:

- `Ladder/motions/zdw0824_capture1/pitchminus3_yawminus4_xminus030/`
  `36_middle_default_standing_100_zplus5mm/`
  `zdw0824_capture1_stage36_train_ready_middle_default100_zplus5mm_reset5mm_50hz_motion.npz`
- SHA-256 `e160464961d2656d239dd43fccde388ab6654901a8297d5f815d6a6858136909`

References checked:

- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `Ladder/scripts/add_cr1_default_standing_start.py`
- `Ladder/scripts/insert_cr1_top_default_stand.py`
- `Ladder/scripts/annotate_cr1_contact_phases.py`
- `Ladder/scripts/detect_cr1_reset_valid_frames.py`

Verification:

- [x] 3061 finite frames at 50 Hz (`61.22 s`) with the original 2761 source
  poses and reviewed contact labels preserved exactly around the insertion.
- [x] The approved quintic Z-offset profile and zero-speed hold were checked.
- [x] Full 3-by-3 reset collision audit completed; all 100 hold frames are
  nominal reset-valid and frame 0 remains valid in all geometry cells.
- [x] A four-environment, one-iteration full-randomization IsaacLab smoke run
  completed with 780-D actor/critic observations, 25-D actions, and runtime
  reset-valid fraction `0.6436`.

## 2026-08-28 - Stage-36 Two-Anchor Reset Mixture

- Extended the reset sampler with an optional secondary fixed start while
  preserving the existing single-anchor path for all earlier tasks.
- Configured only the Stage-36 MiddleStand task to sample `20%` frame 0,
  `10%` frame 1500, `35%` uniform-valid frames, and `35%` adaptive-valid
  frames. Frame 1500 is the first frame of the 100-frame ladder-top default
  hold, leaving the full two-second hold before descent begins.
- Restricted the frame-1500 anchor to nominal-ladder environments. The
  conditional selection probability is adjusted so it remains `10%` of all
  resets while the configured nominal-ladder share remains `30%`.
- Added `secondary_fixed_start_episode` logging. Secondary starts retain the
  ordinary reset pose perturbation and do not inherit frame 0's larger
  fixed-start perturbation envelope.
- Temporarily disabled the Stage-36 frame-0 root-pose perturbation by setting
  `fixed_start_pose_randomization_probability=0.0`. Velocity, joint-state,
  dynamics, and non-frame-0 reset randomization remain enabled.

References checked:

- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `Ladder/source/ladder/ladder/utils/motion_reset.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_reference_rate_limited_env_cfg.py`

Verification:

- [x] 19 reset-sampling tests pass, including a 100,000-sample distribution
  test for the requested `20/10/35/35` mixture and nominal-ladder eligibility.
- [x] A 512-environment IsaacLab reset/step smoke test completed with 780-D
  actor/critic observations and 25-D actions. The observed one-batch reset
  split was `19.1/8.4/36.1/36.3%`, and every frame-1500 start used nominal
  ladder geometry.

## 2026-08-30 - Real-Time Fast Play and Smooth-Output Variant

- Added `play.py --fast_mode` for inference-oriented playback. It removes only
  play-time work that does not affect actor output: reward evaluation, 38
  reward-only filtered contact sensors, duplicate critic observations, command
  debug markers, policy export, automatic force diagnostics, and training-log
  metric updates. Physics collisions, the primary contact sensor, actor
  observations, terminations, ladder/physics/reset randomization, and optional
  root disturbances remain active.
- Added periodic performance reporting with wall-clock Hz, compute-only Hz,
  real-time factor, and the fraction of control cycles exceeding 20 ms.
- Kept the accepted Stage-36 task unchanged. Registered a separate
  `MiddleStand-SmoothOutput-v0` task and experiment directory with the existing
  `action_rate_l2=-0.1` plus IsaacLab's `joint_acc_l2=-2.5e-7`, so future
  training can learn smoother realized motion under full randomization without
  changing old-checkpoint reproduction.

References checked:

- `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/mdp/rewards.py`
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/actions.py`

Verification:

- [x] Fast-play syntax and task registration compile.
- [x] The separate smooth-output task completed a two-environment full-DR
  reset/step smoke test with 780-D actor/critic observations and 25-D actions.
- [x] A one-environment CPU/headless run reached `64.7-68.0 Hz` unthrottled;
  with `--real-time`, it held `49.4-49.6 Hz` (`0.988-0.993x` real time).
- [x] On the existing checkpoint near frame 0, the new acceleration term was
  about `-0.0005` to `-0.0072` per control step, below the existing action-rate
  term (`-0.006` to `-0.018`) and far below primary tracking rewards.

## 2026-08-30 - Validated Task Design Synchronized to Stop-Command Vision

- Added a separate visual task,
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-`
  `ZDW0824-WarpTransformerVision-StopCommand-ValidatedTask-v0`; the existing visual
  and nonvisual task registrations remain unchanged.
- Preserved the existing TransformerVision observation groups, depth input,
  auxiliary targets, and all up/down/stop runtime settings. The ladder
  geometry and limb target offsets now come from the validated ZDW0824
  Stage-36 task rather than the older ZDW0818 visual reference.
- Synchronized the validated nonvisual task design: the full 47-term reward
  set (including per-foot wrong-rung force penalties), 90-second episode,
  rate-limited actions, actuator/dynamics/ladder randomization, and the five
  termination terms.
- Adapted the validated `20/10/35/35%` reset mixture to the visual reference:
  frame 0 is the primary anchor and frame 1500 is the first frame of its
  100-frame ladder-top default-standing hold. The secondary anchor remains
  nominal-ladder-only and frame-0 root-pose perturbation remains disabled.
- Added a reusable command-only annotation step and Stage-37 artifact. All 187
  Stage-36 fields are array-identical; 30 command/audit fields are appended,
  producing 217 fields without changing any pose, contact, velocity, or reset
  array. Frames `0:1550` command up and frames `1550:3061` command down.
- Selected one minimum-base-speed frame from each of the 18 contiguous
  four-limb-contact segments. The initial default stand, ladder-top default
  stand, and existing geometry-valid terminal stable-ground run
  `2991:3061` are also legal zero-command states.
- Added optional `zero_command_direct_hold_frames` support to the motion
  loader. Older motions still fall back to their 18 natural stop frames. The
  Stage-37 file includes geometry-valid ground candidates so the existing 5%
  direct-hold branch remains valid for every randomized ladder cell; no
  command-state transition or probability was changed.

References checked:

- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_zdw0824_no_command_full_state_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_transformer_vision_stop_command_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/scripts/annotate_cr1_climb_commands.py`

Verification:

- [x] Python compilation and task registration succeeded.
- [x] All 187 Stage-36 fields match exactly in Stage 37; frame 0 and frame
  1500 remain valid reset anchors.
- [x] Config contract checks confirmed unchanged visual observations and
  stop-command settings, with rewards/actions/events/terminations equal to the
  validated nonvisual task.
- [x] Three pure-data command-annotation regression tests pass.
- [x] A 64-environment, one-iteration headless TransformerVision PPO smoke
  run completed with 25-D actions, the expected visual observation groups,
  all 47 reward terms, all five termination terms, geometry-conditioned reset
  masks, and reset-valid fraction `0.6436`.
- [x] After assigning the final ZDW0824-specific task ID and experiment name,
  a four-environment one-iteration registration smoke test also completed;
  the PPO gradient was finite and the task loaded Stage 37 successfully.

## 2026-08-30 - ZDW0824 Visual Up/Down Smooth-Output Task

- Added a separate visual task,
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-`
  `ZDW0824-WarpTransformerVision-UpDown-SmoothOutput-v0`; the existing
  stop-command visual task remains unchanged.
- Reused the Stage-37 command annotation because its recorded direction is
  already strictly `+1` before frame 1550 and `-1` afterward. Disabled stop
  training and set both direct-hold and scheduled-seek episode probabilities
  to zero, so this task never presents a zero command during training.
- Switched the task to the accepted SmoothOutput reward set. It retains all
  47 validated tracking/contact/safety terms and adds IsaacLab's
  `joint_acc_l2` at weight `-2.5e-7`.
- Kept visual observations, privileged critic inputs, Stage-36 robot motion,
  ladder geometry, contact targets, 90-second episode, `20/10/35/35%` reset
  mixture, actions, randomization, and terminations unchanged.

References checked:

- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_zdw0824_no_command_full_state_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_transformer_vision_stop_command_validated_task_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`

Verification:

- [x] Python compilation passed.
- [x] All 26 pure command/reset/data tests passed; the Isaac-Sim-only config
  test remains guarded outside an AppLauncher process.
- [x] A 64-environment, one-iteration headless TransformerVision PPO smoke
  run completed with 48 rewards, including `joint_acc_l2=-2.5e-7`.
- [x] Runtime stop-requested, seeking, holding, target, direct-hold, and
  scheduled-seek metrics were all exactly zero.

## 2026-08-31 - TransformerVision Fast-Play Actor-Only Observations

- Fixed `play.py --fast_mode` for custom TransformerVision PPO tasks whose
  runner config intentionally leaves standard `obs_groups` unset.
- The fast path now computes only `transformer_vision_inputs` and packs the
  three deployable actor tensors. Training-only height-map labels, root-linear-
  velocity labels, future-state labels, and privileged critic observations
  remain unchanged in training and normal play but are skipped in fast play.

Verification:

- [x] Python compilation passed and all 11 TransformerVision PPO tests passed.
- [x] A two-step headless fast-play smoke test loaded `model_6000.pt`, reset
  the real IsaacLab task, produced the expected `(85)`, `(10,85)`, and
  `(2,60,80)` actor observations, ran inference, and stepped cleanly.

## 2026-08-31 - TransformerVision Four-Limb Task-Goal Latent

- Added the independent task
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-`
  `ZDW0824-WarpTransformerVision-UpDown-TaskGoal-v0`. Existing visual tasks,
  checkpoints, observations, rewards, resets, terminations, randomization,
  and the up/down command mechanism remain unchanged.
- Replaced only the new variant's future-physical-state latent with a
  supervised 16-D task latent. The head is an ELU MLP
  `256 -> 128 -> 64 -> 16`, not a single linear layer. Twelve outputs predict
  four limb goals in the robot `body` frame; four outputs are contact logits,
  whose sigmoid probabilities enter the actor. The complete actor input stays
  136-D.
- Precomputed each limb's current support rung or next scheduled landing rung
  from the reference contact sequence. Swing limbs therefore retain a useful
  next target; only frames with no remaining ladder contact receive an invalid
  position mask.
- Added privileged supervision using masked Huber loss (`beta=0.05 m`) for
  limb goals and BCE-with-logits for desired contact state. Both first-version
  task losses use weight `1.0`; existing velocity and four height-map loss
  weights are unchanged. Raw phase and rung IDs were not added to the actor.
- Added rollout storage, wrapper packing, shape validation, checkpoint data
  contract, and auxiliary metrics for the new labels while retaining the old
  deterministic and variational contracts.

References checked:

- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/observations.py`
- `Ladder/source/ladder/ladder/rsl_rl/modules/transformer_vision.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_transformer_vision_up_down_smooth_output_env_cfg.py`

Verification:

- [x] Python compilation passed for all changed modules and configs.
- [x] 37 TransformerVision network, PPO, wrapper, runner, and observation
  tests passed; one Isaac-Sim-only test remained intentionally skipped.
- [x] A four-environment, one-iteration headless smoke run completed rollout,
  PPO update, task-goal auxiliary backpropagation, logging, and checkpoint
  save. Runtime labels were `(12) + (4) + (4)`, the actor remained 136-D, and
  every task loss/metric was finite.

## 2026-08-31 - Frozen Full-State Policy Task-Goal Visual Pretraining

- Extended the existing auxiliary-only `train_transformer_vision.py` path to
  select the new `task_goal` visual contract while retaining the deterministic
  future-state contract as its default.
- Added a pretraining-only task that exposes the successful nonvisual policy's
  780-D full-state observation alongside the unchanged visual inputs and
  privileged task-goal labels. This extra `policy` group is not added to the
  deployable visual PPO task.
- The frozen actor loader now infers its observation dimension from
  `actor_state_dict/mlp.0.weight`, allowing both the historical 660-D actor and
  the current 780-D full-state actor. Actor parameters remain excluded from the
  optimizer and are checked for zero change after every update.
- Added temporal storage and replay support for four base-frame limb goals,
  four goal masks, and four contact indicators. Task-goal pretraining uses the
  same masked-Huber and BCE-with-logits losses as visual PPO.

Verification:

- [x] Python compilation and 34 focused TransformerVision tests passed; one
  Isaac-Sim-only config test remained guarded.
- [x] A real two-environment, two-step, one-update headless smoke run loaded
  the successful Stage-36 `model_9999.pt`, observed policy shape `(780,)`,
  produced finite task-goal losses, changed visual parameters, and measured
  exactly zero frozen-actor parameter change.

## 2026-09-01 - Phase-Conditioned Limb Tracking Reward

- Added a ZDW0824-only phase-conditioned limb-position reward. During a
  required-contact phase it tracks the limb's target point; otherwise it
  tracks the corresponding hand-sphere or ankle rigid body in the reference
  motion. The two branches have equal scale: each hand term has weight `0.32`
  and the averaged two-foot term has weight `0.30`. Hand target/reference
  temperatures are `0.01`/`0.03`; foot temperatures are `0.02`/`0.02`.
  The historical ladder reward and older task variants are unchanged.
- Tightened the foot target temperature from `0.05` to `0.02`, tightened the
  body-linear-velocity tracking standard deviation from `1.0` to `0.5`, and
  capped the ZDW0824 foot wrong-rung normalized excess at `1.5`. The ZDW0824
  upper-body position standard deviation was tightened from `0.35 m` to
  `0.25 m`; the lower-body value remains `0.18 m`.
- Split upper- and lower-body joint tracking into sharp and broad kernels
  without changing their maximum combined rewards. This preserves precise
  tracking near the reference while avoiding an almost-zero learning signal
  for larger initial errors.
- Added the diagnostic-only `--uniform_reset_sampling` play option. Training
  reset probabilities are unchanged. For all-frame diagnostics, a temporary
  motion copy with every frame marked reset-valid was used; the training
  motion was not modified.

References checked:

- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/rewards.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_zdw0824_no_command_full_state_env_cfg.py`

Verification:

- [x] All 19 focused ladder reward tests passed and changed Python modules
  compiled successfully.
- [x] The real IsaacLab visual task instantiated with all 50 reward terms and
  loaded `model_6500.pt` successfully.
- [x] Reward-enabled play sampled 113 observations from seven randomized
  starts spanning all ten equal-length reference-motion intervals.
- [x] After equalizing the limb reward branches, real-task play sampled 28
  observations from seven starts spanning frames 120 through 2760. Contact
  and swing rewards stayed within the intended `0.32` hand and `0.30` foot
  maxima, and all runs completed without reward-configuration errors.
- [x] The pre-existing 5120-environment training process remained running;
  it does not hot-reload these on-disk reward changes.

## 2026-09-02 - State-Locked Next-Stable Whole-Body Visual Semantics

- Extended only the independent up/down task-goal visual variant with a
  state-locked next-stable whole-body target. The reward/reference cursor,
  reset mixture, terminations, randomization, contact rewards, and robot motion
  data are unchanged; no IK or reference-motion rewrite was introduced.
- Reused the Stage-37 reference's 21 accepted direct-hold frames as ordered
  stable anchors. Initial targets are selected from the reset frame without
  crossing the `+1/-1` command boundary. The active target advances only after
  five consecutive steps satisfy joint-pose, root-orientation, physical
  contact, base-speed, and joint-speed completion checks.
- Corrected current/next limb-goal generation to search separately inside
  contiguous ascent and descent runs. This removes the previous hand-label
  leakage from late ascent frames into descent contacts.
- Added privileged auxiliary labels for the next stable 25-D normalized joint
  pose, 6-D root orientation, and validity mask. Raw phase, rung ID, and target
  frame remain absent from actor observations.
- Retained the existing 16-D four-limb goal/contact head and added an
  independent motion MLP `256->128->64->16` plus decoder
  `16->64->128->31`. Masked Huber losses supervise the 25 joint and 6
  orientation outputs at weight `1.0` each. The task-goal actor now consumes
  `85 + 51 + 16 = 152` values; all older TransformerVision variants remain
  136-D with their existing checkpoint contracts.
- Wired the labels and variable actor-input dimension through frozen-policy
  visual pretraining, joint visual PPO, rollout storage, wrappers, checkpoint
  contract validation, inference, and fixed-log-prob diagnostics.

References checked:

- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/observations.py`
- `Ladder/source/ladder/ladder/rsl_rl/modules/transformer_vision.py`
- `Ladder/source/ladder/ladder/rsl_rl/algorithms/transformer_vision_ppo.py`

Verification:

- [x] Stage-37 data checks confirmed 3061 frames, one direction switch at
  frame 1550, 21 valid stable anchors, no cross-direction target lookup, and
  finite joint/orientation labels for every frame.
- [x] Python compilation passed for every changed command, observation,
  network, storage, wrapper, runner, and algorithm module.
- [x] The complete non-simulator suite passed: 206 tests, with three
  Isaac-Sim-only tests skipped because this shell does not expose the
  `isaaclab` Python package.
- [x] A four-environment, one-iteration headless AppLauncher training smoke
  completed rollout, PPO, and visual auxiliary updates and saved `model_0.pt`.
  The new motion supervision was finite and active (`motion_joint=0.327`,
  `motion_root_orientation=0.307`, valid fraction `1.0`), and the PPO update
  remained finite. One of 20 auxiliary micro-batches was skipped by AMP in
  this tiny-batch smoke, matching the pre-existing four-environment task-goal
  smoke behavior; the prior 5120-environment run recorded zero skipped
  auxiliary updates.
## 2026-09-02 - Nonvisual CR1 Hybrid DAgger + PPO

- Added an isolated `config/cr1/dagger/` task implementation.  The student
  receives five flattened 84-D deployable proprioception frames (420-D total):
  root angular velocity, root orientation 6-D, joint position, joint velocity,
  and previous action.  Reference motion, phase, contact annotations, depth,
  recurrent modules, and visual encoders are excluded.
- Reused the accepted ZDW0824 MiddleStand SmoothOutput task dynamics, rewards,
  resets, and randomization.  Separate 780-D full-state observation groups feed
  the frozen teacher and privileged critic.
- Added a feed-forward student, split-format RSL-RL teacher-checkpoint loader,
  rollout storage, TensorDict environment wrapper, and PPO update augmented by
  annealed teacher action-mean MSE.  Environment actions always come from the
  student, preserving PPO's on-policy rollout contract.
- Added the task
  `Ladder-Climb-CR1-ZDW0824-NoCommand-NonVisual-Dagger-v0` and the required
  `--teacher_checkpoint` training argument.  Student-training checkpoints and
  teacher checkpoints remain separate.

References checked:
- `docs/LadderMan.pdf`, Sec. 3.2 and Appendix A.2
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_zdw0824_no_command_full_state_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/observations.py`
- `Ladder/source/ladder/ladder/rsl_rl/algorithms/transformer_vision_ppo.py`
- `isaaclab_v2.1.0/source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py`

Verification:
- [x] Modified Python files compile.
- [x] The actual 780-D `model_4000.pt` teacher actor loads, produces 25-D
  actions, and has no trainable parameters.
- [x] `PYTHONPATH=Ladder/source/ladder ... -m pytest -q Ladder/tests`:
  `211 passed, 3 skipped`.
- [ ] IsaacLab reset/step smoke: the available runtime combines IsaacLab 2.1
  with Isaac Sim 4.5 and fails in `AppLauncher` because PhysX does not expose
  `SETTING_BACKWARD_COMPATIBILITY`.  Confirm the 420/780/780 observation shapes
  in the version-matched training environment before a long run.

## 2026-09-02 - Command-Aware Visual CR1 Hybrid DAgger + PPO

- Added the independent task
  `Ladder-Climb-CR1-ZDW0824-UpDown-Visual-Dagger-v0` on the Stage-37 motion.
  Stage-37 was verified byte-identical to the teacher's Stage-36 joint, body,
  contact, and reset arrays; it only adds the deployable direction annotation.
- The visual student receives five 85-D proprioception frames and two 60x80
  Warp depth frames.  Every proprioception frame includes exactly one binary
  command value: `+1` upward or `-1` downward.  The flattened 425-D history and
  deterministic 35-D visual latent form the 460-D actor input.
- Added a five-frame deterministic TransformerVision encoder with the separate
  contract `cr1_dagger_visual_5x85_depth2_velocity_heightmap35_v2`.  Its
  auxiliary heads estimate only base linear velocity and current-limb terrain
  maps. The 16-D dynamic latent, next-proprio decoder/target, task-goal,
  reference joint pose, reference root orientation, phase, contact annotation,
  and motion-latent paths are not configured.
- PPO and action-head DAgger use fixed rollout latents.  A sequence replay
  applies the same teacher action loss to the visual encoder, while the
  existing reference-free visual auxiliary losses provide direct perception
  supervision.  Teacher and critic retain separate 780-D privileged inputs.

Verification:

- [x] Stage-37 direction contains only `-1/+1` (`1511/1550` frames).
- [x] Stage-36 and Stage-37 joint/body/contact/reset arrays are identical.
- [x] Visual student tests cover 5x85 input, 460-D actor input, binary command
  preservation, frozen teacher labels, rollout collection, and joint
  PPO/DAgger/visual-auxiliary update.
- [x] Full non-simulator suite: `217 passed, 3 skipped`.
- [ ] Real reset/step smoke remains blocked by the available IsaacLab 2.1 /
  Isaac Sim 4.5 PhysX API mismatch described above.

## 2026-09-02 - Isolated Nonvisual Ten-Frame Tracking Penalty

- Added a stateful, reset-aware ten-frame reference-history penalty to a new
  nonvisual A/B task. The accepted MiddleStand SmoothOutput baseline, visual
  tasks, motion data, observations, resets, randomization, and terminations are
  unchanged.
- The term maintains only four scalar component costs per environment per
  history frame. It compares joint pose, aligned full-body position plus root
  pose, full-body linear velocity, and robot/reference motion increments using
  normalized Huber costs. Component shares are `0.30/0.30/0.20/0.20`; the
  overall reward weight is `-0.20`.
- Histories are cleared per environment on reset. The warm-up uses one through
  nine actual post-reset samples rather than repeating the reset observation,
  and the motion-increment component starts from the second real sample.
- Registered
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-ZDW0824-NoCommand-FullState-MiddleStand-SmoothOutput-HistoryPenalty-v0`
  under the separate experiment
  `cr1_nonvisual_zdw0824_no_command_full_state_middle_stand_history_penalty`.
- Compared the resolved reward YAML against the latest visual TaskGoal task.
  Its 50 shared reward terms, functions, weights, and parameters are identical;
  the nonvisual A/B task has exactly one additional term: the ten-frame history
  penalty. A config regression assertion now preserves this relationship.

References checked:

- `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/reward_manager.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/manager_term_cfg.py`
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/rewards.py`

Verification:

- [x] Changed Python modules compile.
- [x] Complete non-simulator suite: `214 passed, 3 skipped`.
- [x] A 64-environment, two-iteration headless IsaacLab smoke completed and
  saved `model_0.pt` and `model_1.pt`.
- [x] A second 64-environment, two-iteration smoke using the synchronized final
  config completed PPO updates and checkpoint writes without non-finite values.
- [x] At history length ten, all four diagnostics were finite. The random
  policy produced joint/body/velocity/delta costs of
  `0.8964/0.4876/0.4405/0.8172`; their weighted composite was about `0.69`, or
  about `-0.14` before IsaacLab's reward-`dt` integration.

## 2026-09-03 - Motion-Delta-Only Nonvisual A/B Task

- Added a separate nonvisual task and experiment that keep only the ten-frame
  motion-delta component active. The full-history task and its existing runs
  remain reproducible and unchanged.
- Set the total reward weight to `-0.04`. Observed motion-delta costs were
  approximately `0.48-0.72` (recent mean about `0.63`), so the typical raw
  contribution is about `-0.025`. This preserves the previous effective
  coefficient of the delta component (`-0.20 * 0.20`) rather than increasing
  it fivefold.
- Registered
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-ZDW0824-NoCommand-FullState-MiddleStand-SmoothOutput-MotionDeltaPenalty-v0`
  under
  `cr1_nonvisual_zdw0824_no_command_full_state_middle_stand_motion_delta_penalty`.
- Absolute joint-pose, body-pose, and body-velocity component weights are zero.
  Their metrics remain diagnostic only and cannot affect optimization.
- After comparing the isolated delta contribution with the former complete
  history penalty, increased the motion-delta-only task weight from `-0.04` to
  `-0.10`. At the recent mean delta cost of `0.59`, its raw contribution is
  about `-0.059` (`-0.00118` after 20-ms reward integration).
- The motion-delta-only task disables reset-valid filtering. Its reset mixture
  remains frame 0 `20%`, frame 1500 `10%`, uniform `35%`, and adaptive `35%`,
  but both random branches now admit every reference frame. Other nonvisual
  and visual tasks continue to use their embedded geometry-conditioned masks.

Verification:

- [x] Changed Python modules compile.
- [x] Complete non-simulator suite: `217 passed, 3 skipped`.
- [x] A 16-environment, two-iteration headless IsaacLab smoke instantiated the
  new task, completed reset/rollout/PPO updates, and saved checkpoints. The
  measured random-policy delta cost was `0.4665`; the integrated episode log
  showed a finite `reference_motion_history_huber` contribution.
- [x] A second 16-environment, two-iteration smoke after enabling all-frame
  reset sampling reported `reset_valid_fraction=1.0000` and completed PPO
  updates without non-finite values.

## 2026-09-03 - Pure Visual DAgger Distillation Baseline

- Added a separate pure-distillation task and experiment; the existing hybrid
  DAgger+PPO run remains unchanged.
- The student uses nine 85-D deployable proprioceptive frames, two 60x80 depth
  frames, and the paper's three-layer `[8, 8, 8]`, kernel-5 CNN. CNN features
  feed the actor directly; there are no reconstruction, height-map, velocity,
  Transformer, GRU, or decoder losses/modules.
- Student rollouts are retained: every visited student state is labeled online
  by the frozen full-state teacher. The only update is
  `2 * KL(student || teacher) + 1 * penultimate_feature_MSE`; the teacher and
  environment rewards receive no gradients, and every student parameter is in
  the distillation optimizer.
- The student penultimate feature and teacher final hidden feature are both
  256-D. The teacher checkpoint's learned Gaussian standard deviation is used
  in the distribution KL.
- The pure task retains the established reset-valid filtering and reset
  mixture: frame 0 `20%`, frame 1500 `10%`, valid-frame uniform `35%`, and
  valid-frame adaptive `35%`. An initial all-frame experiment was reverted
  after it introduced too many teacher out-of-distribution reset states.
- Both distillation terms are reduced by their feature dimensions before the
  configured `2:1` weighting: diagonal-Gaussian KL is averaged across the 25
  actions and batch, while feature MSE is averaged across the 256 hidden
  features and batch.

References checked:

- `docs/LadderMan.pdf`, Sec. 3.2 and Appendix A.2.
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/dagger/visual_algorithm.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`

Verification:

- [x] DAgger unit suite: `11 passed`.
- [x] A two-environment, one-iteration headless IsaacLab smoke collected 48
  student transitions and completed finite KL/feature-only updates.

## 2026-09-04 - Pure Visual DAgger BC Warmup and Mixed Rollout

- Added a 500-update behavior-cloning warmup to the pure visual DAgger runner.
  During warmup, every environment executes the frozen teacher's deterministic
  action mean while retaining both the action KL and penultimate-feature MSE.
- After warmup, exactly 25% of the parallel environments execute teacher means
  and the remaining 75% execute deterministic student means. Every visited
  state is still labeled online by the teacher; no intervention trigger or RL
  loss was added.
- Separated rollout behavior from the KL distribution: the learned policy
  standard deviation remains part of KL distillation but is no longer sampled
  as environment action noise. Added `rollout_teacher_fraction` logging.
- The existing `update_iteration` checkpoint field controls the phase after a
  resume, so a resumed run does not repeat BC warmup.

Verification:

- [x] DAgger unit suite: `12 passed`.
- [x] Tests cover all-teacher warmup, exact 25/75 mixed action selection, both
  retained distillation losses, and update-iteration checkpoint restoration.

## 2026-09-04 - TaskGoal v1 Playback Compatibility

- Added a playback-only legacy TaskGoal encoder selection for checkpoints with
  data contract `cr1_field_normalized_proprio_rung_weighted_task_goal12_contact4_v1`.
  It restores the original 51-D visual latent and 136-D actor input without
  constructing the later v2 stable-motion branch.
- `play_transformer_vision_ppo.py` now detects the checkpoint contract before
  runner construction. Current v2 training remains unchanged; strict state
  loading is retained.

Verification:

- [x] The complete `model_7000.pt` policy state loads strictly with matching
  v1 contract and 136-D actor input.
- [x] TransformerVision focused tests pass (`28 passed`).
- [x] A two-step headless IsaacLab playback loaded the checkpoint and stepped
  the TaskGoal environment without a data-contract error.
## 2026-09-05 - Legacy TransformerVision PPO Resume Compatibility

- Extended the training resume path to inspect TransformerVision checkpoint
  contracts before constructing the policy. Legacy TaskGoal-v1 and variational
  checkpoints now select the same exact legacy encoders already used by the
  dedicated playback script, allowing optimizer-complete training continuation
  without changing their actor topology.
- Verified `train.py` compilation and the focused TransformerVision suite:
  `27 passed, 2 skipped`.

## 2026-09-06 - Teacher-Only Visual BC Diagnostic Experiment

- Added the independent `Ladder-Climb-CR1-ZDW0824-UpDown-TeacherOnly-Visual-BC-v0`
  task under the existing CR1 `dagger/` folder. Every environment executes the
  frozen teacher mean permanently. The original BC500 + mixed 25/75 task remains
  available; student architecture, KL + feature MSE, adaptive valid-frame resets,
  randomization, and action/termination contracts are unchanged.
- Fixed held-out environment suffix is excluded from every optimization epoch.
  Defaults: 4096 total environments, 64 validation environments, 10000 updates,
  checkpoint/diagnostic-sample interval 500. Uses the validated stage36 teacher
  `model_9999.pt`, with stage37 up/down motion and a freshly initialized student.
- Added sample-weighted action/feature metrics, KL mean/std decomposition,
  per-joint and direction/phase/reset/moving-label diagnostics, previous-action
  baseline, explicit teacher episode metrics, per-episode termination records,
  bounded raw rollout snapshots, checksummed input/source provenance, and a
  copy-safe launch script retaining full console logs. See `dagger/TEACHER_ONLY_BC.md`.
- Teacher-only checkpoints carry a rollout-mode guard for resume; resumed runner
  iteration is aligned with the algorithm's next-update counter.
- References checked: official IsaacLab distillation config, G1 registration,
  ManagerBasedRLEnv/TerminationManager; whole_body_tracking train/tracking config;
  current pure DAgger implementation. No new whole_body_tracking runtime imports.

Verification:

- [x] Existing DAgger + new CPU tests: `24 passed`. They check permanent teacher control, retained
  two losses, all student subnetworks updating with frozen teacher, held-out
  gradient isolation, checkpoint mode, metric arithmetic and pre-reset attribution.
- [x] Actual IsaacLab GPU smoke: 8 envs, 3 updates, with a test-only 0.12 s episode
  horizon to exercise auto-reset and logging. Logged 32 completed episodes,
  3 metrics rows, 3 raw snapshots and model_0/1/2. Teacher fraction stayed 1.0;
  each rollout had 48 train and 16 validation samples. TensorBoard and checkpoint
  reload checks passed. Formal training retains the original episode horizon.
- [x] Separate 8-env, one-update GPU resume smoke used the normal episode horizon.
  Resumed model_2 into model_3 with update counter 4; standard TensorBoard,
  BC JSONL and snapshot index all aligned at 3. Raw samples confirmed previous
  actions exactly equal teacher outputs on non-reset steps, and depth refreshes.
- [x] `bash -n` passes for `train_teacher_only_visual_bc.sh`.
- At initial implementation handoff, formal 4096-env/10000-update training had
  not been started by the agent.

## 2026-09-06 - Started 5120-Environment Teacher-Only BC

- User requested 5120 environments and explicitly authorized starting training.
  Updated the launcher default/run name and experiment guide to 5120, retaining
  10000 fresh-student updates, save interval 500, seed 42, stage36 teacher9999,
  stage37 motion, original episode horizon, and all-teacher deterministic control.
- Split: env 0–5055 train, env 5056–5119 validation. Each 24-step iteration logs
  121344 training and 1536 validation samples; validation never enters optimization.
- Started detached tmux session `teacher_bc_5120_20260906`; Python PID `928796`.
  Run: `logs/rsl_rl/cr1_zdw0824_up_down_teacher_only_visual_bc/2026-09-06_01-10-23_teacher_only_bc_teacher9999_5120env_10000iter`.
  Console: `logs/teacher_only_bc_launch/2026-09-06_01-10-13_928791.log`.
- Startup verification: completed iterations 0–3, all BC metrics finite,
  rollout teacher fraction exactly 1.0, correct train/validation sample counts,
  model_0 and raw rollout_000000 snapshot saved, episode JSONL and manifest present.
  GPU memory at check: 22146 MiB used, 24657 MiB free. Training left running.
- References rechecked: IsaacLab G1 registration, whole_body_tracking training
  runner/config persistence, and current teacher-only algorithm/config.

## 2026-09-07 - Same-State Shadow Check of Teacher-Only BC Student

- Added standalone `scripts/rsl_rl/diagnose_visual_student.py` and CR1 `dagger/`
  `shadow_config.py` / `shadow_analysis.py`. Existing training and play behavior
  remain unchanged. Student or frozen teacher executes deterministic means;
  the other network supplies shadow predictions without interventions/updates.
- Two explicit evaluation configurations: existing play disable-randomization
  behavior, and clean sensor/camera/motor/no-delay behavior. Kept action scales,
  rate limit, collision geometry, physics and terminations unchanged.
- Saved full actor inputs/actions/features, actual delayed/rate-limited targets,
  pre/post physical state and termination labels, configuration/input hashes,
  source archive, CSV/JSON summaries and console logs. Captured post-action data
  before automatic reset using an instance-local read-only termination hook;
  avoided RecorderManager's extra observation/history computation.
- References checked: official IsaacLab RSL-RL play, Anymal play corruption
  configuration, installed ManagerBasedRLEnv step/TerminationManager/RecorderManager;
  whole_body_tracking play; local observations/actions/commands/terminations and
  pure DAgger model/wrapper. No whole_body_tracking runtime dependency added.

Verification and results:

- CPU configuration/summary tests: 14 passed. Combined DAgger/teacher-only/shadow
  regression tests: 33 passed.
- GPU smoke: 2 environments, 3 steps; finite/correct input shapes and reference
  progression, zero clean delay, frozen weights, correctly censored episodes.
- Four GPU cases: 8 environments each, seed 42, frame 0, first episode only,
  max 3200 steps. Teacher finished all 3061 reference frames in 8/8 cases in
  each condition. Student failed in 8/8 cases in each condition: 75–124 steps
  normal, 104–110 steps clean. No required-contact termination in these starts.
- Student same-state shadow action RMSE on teacher trajectories: 0.019403
  normal / 0.016389 clean; on student trajectories: 0.960922 / 1.007460.
  A shared reset transient decays under teacher control but grows under student
  control. This supports a closed-loop imitation failure without identifying a
  unique cause among missing information, feedback dependence and fitting errors.
- Within each condition, controller runs have identical initial states and
  network inputs/outputs. All traces finite, weights unchanged, actual delays
  and 0.12-rad-per-step target limit verified. All evaluation processes exited.
- Artifacts: `outputs/teacher_student_shadow/2026-09-07_frame0_check/`.
  Detailed findings: `docs/learning/teacher_only_bc_shadow_check_2026-09-07.md`.

## 2026-09-07 - Distillation Reference Audit and Independent Warp Camera Clock

- Audited installed RSL-RL 5.0.1 and current upstream distillation, normalization,
  loss reduction, sampling and recurrent replay. Compared Extreme Parkour's
  structured adaptation, actor initialization, heading supervision, GRU reset
  and 10-Hz camera path. Findings and interpretation boundaries are recorded in
  `docs/learning/distillation_rsl_parkour_audit_2026-09-07.md`.
- Added `update_period` to the shared Warp observation term; new Warp visual /
  TransformerVision task configurations default to 1/30 s, while policy and
  proprio history remain 50 Hz. Fractional-phase scheduling at policy boundaries
  yields average30Hz with20/40ms capture intervals, not exact asynchronous exposure.
- Hold processed images/noise between captures; insert depth history only on new
  frame IDs. Immediately refresh reset environments and refill their history,
  preserving continuing environments and the global clock. Repeated observation
  queries within one step do not resample or shift histories. Renderer uses its
  existing batched env_ids interface; actor dimensions and preprocessing unchanged.
- Main train/play, visual pretrain/check, and shadow diagnostics restore the
  visual checkpoint's saved camera period. Missing old fields retain legacy
  policy-rate sampling; absent config warns. CLI `--depth_update_hz` overrides
  timing (`0` means policy rate). Fresh runs persist effective params/env.yaml.
  RTX observations and nonvisual tasks are excluded from the Warp-only override.
- Reference order checked: IsaacLab sensor_base/config timestamp/update/reset,
  official sensor/task examples, whole_body_tracking tracking config and play,
  existing LadderMan Warp/history/GRU/storage/play implementations. No runtime
  dependency on whole_body_tracking added. Loss, normalization, networks, rewards,
  teacher, control rates and reference-frame/adaptive reset sampling not changed.
- Verification: 95 CPU regression tests passed, one CUDA-only test skipped in
  sandbox; 10 changed production Python files compile. Independent timing review
  and 11 targeted timing tests passed. Real GPU smoke:2envs,50policy steps,
  30camera captures/env, held history bitwise unchanged, same-step env0-only reset
  refresh/refill verified while env1 remains unchanged; correct finite input
  shapes and no unexpected episode termination. Temporary smoke-script reset
  inference-mode boundary was corrected; no production fix was needed for it.
- All GPU checks finished; no training started. Temporary smoke source/log:
  `/tmp/ladder-depth-check-Oo9RQI/`. Detailed report linked above.

## 2026-09-07 - Retired Superseded ZDW0824 Nonvisual Task Variants

- Kept only the registered ZDW0824 nonvisual PPO tasks `MiddleStand-v0` and
  `MiddleStand-SmoothOutput-v0`. Removed the Stage-34, full-history-penalty,
  motion-delta-only, and untrained nonvisual DAgger task registrations and their
  dedicated environment/runner configurations.
- Made `MiddleStand` the direct Stage-36 full-state environment rather than
  inheriting through the retired Stage-34 task. Visual DAgger runners now inherit
  directly from the retained SmoothOutput runner, so visual task registrations
  and behavior remain available.
- Preserved the phase-conditioned limb tracking contract in both retained tasks:
  required-contact limbs track ladder target points, while non-contact limbs
  track their corresponding reference-motion rigid-body positions. SmoothOutput
  additionally retains `joint_acc_l2=-2.5e-7`.
- Deleted the three retired RSL-RL experiment directories and their checkpoints,
  TensorBoard events, and saved parameter files (about 732 MiB total). Motion
  preprocessing artifacts were retained for provenance.
- Verification: changed Python files compile; reward/DAgger CPU regressions pass
  (`27 passed`). Registry and experiment-directory audits find only the two
  retained nonvisual variants, with the retained `model_9999.pt` (MiddleStand)
  and `model_4000.pt` (SmoothOutput) checkpoints still present. An IsaacLab
  reset/step smoke could not reach task setup in the tool sandbox because CUDA
  was unavailable, so the real simulator smoke remains to be run on the host.

## 2026-09-07 - Updated Retained ZDW0824 Blind-Policy Reset and Failure Envelope

- Updated the retained MiddleStand base (and therefore SmoothOutput) to sample
  `10%` from frame 0, `10%` from frame 1500, `40%` uniformly from all motion
  frames, and the remaining `40%` adaptively over all frames. Embedded reset
  validity masks are disabled for these two blind-policy tasks only.
- Tightened only the sustained root-orientation failure threshold from `1.2`
  to `0.8 rad`; the local-body orientation threshold remains `1.2 rad`.
- Relaxed the shared ZDW0824 body-linear-velocity tracking standard deviation
  from `0.5` to `0.75 m/s`. Phase-conditioned limb tracking and the
  SmoothOutput joint-acceleration penalty are unchanged.
- Verification: reset/reward CPU regressions pass (`38 passed`), the IsaacLab
  config test module is skipped outside SimulationApp, and the changed Python
  files compile. Source assertions confirm the retained blind tasks use the
  requested probabilities and root-orientation threshold.
- Re-enabled frame-0-only pose randomization for `70%` of frame-0 starts using
  the reduced envelope: `x/y=+/-0.03 m`, `z=[0, 0.005] m`,
  `roll=+/-0.01 rad`, and `pitch/yaw=+/-0.05 rad`. The middle anchor and other
  reset branches are unaffected.

## 2026-09-08 - Added the upstream DR02 Standard robot model

- Added the DeepRoboticsLab DR02 Standard MJCF, URDF, and USD assets under
  `assets/robots/DR02-STANDARD/`, alongside the existing DR02 Pro-derived
  `CR1-B2-PRO-260608/` assets. No task or default robot configuration changed.
- Pinned the import provenance to upstream commit
  `549eb9ac08ab0674ad388202f0a276a1b140833b` and retained its BSD 3-Clause
  license. Deduplicated the byte-identical MJCF/URDF mesh copies without
  changing either model file.
- Verified that all 22 MJCF and 22 URDF mesh references resolve, the MJCF and
  URDF XML parse successfully, and the imported files match their upstream
  source hashes. The USD package and its four relative configuration layers
  were retained unchanged; no simulator smoke was needed because the model was
  added as an unused asset only.

## 2026-09-08 - Added read-only lower-body jitter diagnostics to play

- Added an opt-in env-0 diagnostic for hip, knee, and ankle joint position,
  velocity, control-rate acceleration, reference errors, raw-action deltas, and
  processed-target deltas. It writes per-joint/aggregate RMS and absolute P95
  summaries plus the worst 25-frame intervals to JSON, and every frame/joint
  sample to a CSV sidecar. Policy outputs, physics, rewards, and training code
  are unchanged.
- Fixed `play.py --disable_randomization` to also disable the two actuator
  strength randomization events, matching the option's documented meaning and
  permitting comparable fixed-frame playback.
- Verified `play.py` compiles and completed full frame-0 playbacks of the
  retained SmoothOutput `model_4000` and `model_9999` checkpoints. Both covered
  reference frames 0 through 3060 and ended only on motion completion.

## 2026-09-08 - Strengthened SmoothOutput lower-body acceleration regularization

- Replaced the shared all-joint `joint_acc_l2=-2.5e-7` term with disjoint
  groups: upper-body joints retain `-2.5e-7`, while hip, knee, and ankle joints
  use `-1.0e-6`. The existing `action_rate_l2=-0.1` and all tracking, contact,
  reset, termination, and randomization settings are unchanged.
- The SmoothOutput reward config is intentionally shared by the retained
  nonvisual and visual tasks; the immediate experiment remains a nonvisual
  continuation from `model_9999`.
- Verification: changed sources compile. The config-only pytest module skips
  without SimulationApp as expected; a real one-environment, one-step GPU smoke
  instantiated both acceleration terms and exited successfully.

## 2026-09-08 - Added lower-body raw-action second-difference regularization

- Added a stateful SmoothOutput reward for
  `sum((a[t] - 2*a[t-1] + a[t-2])^2)` over the 12 lower-body actions, with
  weight `-0.05`. This directly discourages alternating policy-output changes
  while leaving constant-rate action ramps unpenalized.
- The first two steps after each per-environment reset are masked so cleared
  action history cannot create a false startup penalty. The existing
  action-rate, realized joint-acceleration, and robust reference-velocity terms
  remain unchanged.
- The retained `model_9999` frame-0 diagnostic had lower-body raw-action
  second-difference RMS `0.0439` and P95 `0.0908`; at the configured weight its
  average clean-play contribution is about `-0.00115` before RewardManager
  time-step scaling.

## 2026-09-08 - Added robust lower-body reference-velocity regularization

- Verified that the retained Stage-36 reference is internally consistent and
  already low-pass filtered: lower-body velocity energy above 5 Hz is about
  `0.0135%`, while the prior deterministic `model_9999` playback produced
  roughly four times the reference acceleration RMS/P95. The remaining visible
  jitter is therefore not explained by high-frequency reference noise alone.
- Added a SmoothOutput-only lower-body velocity-error penalty with a
  `0.15 rad/s` deadband and `0.5 rad/s` Huber transition. It averages across
  the selected lower-body joints and uses weight `-0.05`, so small reference
  imperfections are ignored and large errors grow linearly rather than
  dominating PPO. Existing action-rate and acceleration terms are unchanged.
- The shared SmoothOutput reward config applies this term consistently to the
  retained nonvisual and visual tasks.

## 2026-09-08 - Added DR02 Standard retargeting model and 0824 mapping

- Added a self-contained Holosoma DR02 Standard model directory based on the
  pinned upstream model. The source URDF/MJCF remain unchanged; derived
  retargeting files expose named fixed hand spheres, four sole reference points
  per foot, and one toe point per foot using the established Pro geometry.
- Re-parented each existing Standard hand collision sphere instead of adding a
  duplicate. Reference geoms have zero density, retaining the official
  `67.575345 kg` MJCF mass. The model remains 21 actuated joints with
  `nq=28`, `nv=27`, and 21 actuators.
- Registered `dr02_standard` and a LAFAN mapping for the aligned 0824 data. The
  human-to-robot semantic links match the successful fixed-wrist Pro mapping;
  Standard-specific qpos limits/cost indices account for the absent waist-x,
  waist-y, neck, and wrist joints.
- Verification: Python compilation and the new config/mapping/model tests pass;
  MuJoCo and yourdfpy both load the derived models and expose all 12 reference
  links. A 3-frame 0824 climbing smoke after 30 warmup solves produced finite
  `(3, 28)` qpos at 50 Hz, remained at least `0.3007 rad` inside source joint
  limits, and used no penetration or foot-sticking fallback.
- Smoke artifact:
  `holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_results/zdw0824_capture1/dr02_standard/climbing/smoke_3frames/`.
  The complete unittest discovery is still blocked by the environment's missing
  optional `pytest` dependency; all discovered unittest-only cases passed.
- Completed the full aligned 0824 capture with the same Standard mapping and
  contact settings: 10,944 source samples were downsampled by 4, then solved
  with 30 warmup frames to produce 2,736 frames at 50 Hz (54.72 s). The final
  `(2736, 28)` qpos and `(2736, 22, 3)` human arrays are finite; root quaternion
  norm error is below `1.1e-12`. There are no source-joint-limit violations, no
  penetration fallback, and no foot-contact transition release. Foot sticking
  used its configured 3 cm fallback instead of 2 cm at source frames 873, 895,
  and 904. The worst source-limit margin is `0.1257 rad`; manual-bound excess is
  below `8e-12 rad` numerical tolerance. The maximum consecutive joint change
  is `0.1922 rad` at the right ankle-x joint; the 99th percentile over all
  frame/joint changes is `0.0506 rad`.
- Full artifact:
  `holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_results/zdw0824_capture1/dr02_standard/climbing/full_pitchminus3_yawminus4_xminus030_50hz_fixedparams_exact_grounding/zdw0824_capture1_full_aligned_mirrorx_pitchminus3_yawminus4_xminus030_200hz_original.npz`.
- Audited every full-result frame against all 30 enabled Standard robot
  collision geoms and all 18 ladder boxes using final nonlinear MuJoCo signed
  distances. The worst overlap is `16.217 mm` at frame 1913 between
  `left_ankle_x_link` and ladder step 2. Frames deeper than `1/2/5/10 mm`
  number `2034/1625/1529/501`; these counts include intended hand and foot
  support. The raw trajectory's maximum forward-difference joint speed is
  `9.609 rad/s` at right ankle-x into frame 1475, maximum joint acceleration
  is `913.290 rad/s^2` at left ankle-x around frame 410, and maximum root
  translation step is `18.589 mm` into frame 2248. The corresponding human
  Hips step is `14.763 mm`, so this is not the old Pro retarget-only root
  branch jump.
- Exported an immutable Ladder-format raw motion using the Standard retarget
  URDF: 2,736 frames, 21 joints, and 36 bodies. All arrays are finite; exported
  joints reproduce source qpos within float32 rounding (`5.96e-8 rad`). Full
  offline URDF FK agrees with MuJoCo at every frame/body within `1.86e-7 m`
  and `1.99e-7 rad`. Raw SHA-256 is
  `05c5d15ea082ba5b8f752b76ec29925f0efd188ed9b155acff641ed1d0b513f2`.
- Reused the accepted 0824 rigid ladder-frame transform in a separate
  world-aligned raw artifact, mapping source `(0, 0.3700000048, 0)`, yaw
  `-180 deg`, to canonical `(0.38, 0.04, 0)`, yaw `90 deg`; SHA-256 is
  `8e1d8d020d10ffe9680742d461fc062c2a5da75fe6419e352a1890807fa661ef`.
  A third immutable raw artifact applies the established `-0.03 m` joint
  robot/ladder grounding translation; SHA-256 is
  `7b7a5b4ddad4366d6cc953a4f74313422a672300d7085152a7e0564d6286f061`.
  Joint positions and velocities are bit-identical across all three stages.
- The Pro grounding correction cannot be accepted unchanged for Standard:
  after the shared `-0.03 m` transform, the physical left/right foot boxes
  begin `85.95/74.35 mm` below the task ground. Left-foot below-ground runs are
  frames `0--374` and `2383--2735`; right-foot runs are `0--524` and
  `2242--2735`. The aligned Pro and Standard base/hip/knee/ankle positions are
  otherwise within roughly millimetres at frame 0. The additional depth comes
  from collision geometry: Standard uses a `0.30 x 0.11 x 0.06 m` foot box
  centered `0.045 m` below ankle-x, while the Pro training asset uses its foot
  mesh. Stage 03 is retained only as a raw baseline and is not train-ready;
  Standard-specific ground-clearance repair and visual review are required
  before contact annotation.
- Ladder artifacts are under
  `Ladder/motions/zdw0824_capture1/dr02_standard/pitchminus3_yawminus4_xminus030/`
  in `01_raw_holosoma_export/`, `02_world_aligned_raw/`, and
  `03_ladder_grounded_raw/`.

## 2026-09-09 - Complete playback control-delay randomization switch

- `play.py --disable_randomization` now zeros both action control-delay bounds
  before environment creation. This closes the remaining 0/20/40-ms reset
  variation identified in `outputs/reset_delay_probe_20260909/findings.md`.
- Added `[PLAY][DELAY]` reporting for environment 0 at startup, after manual
  and automatic resets, and on the `I` information request. Reports use the
  actual delay-buffer lag and convert policy steps to milliseconds using
  `env.step_dt`. They are also emitted in fast mode.
- Without the flag, the configured per-reset delay sampling is retained.
  Changes are limited to playback; training configurations and rewards are
  unchanged.
- References checked: IsaacLab v2.1 RSL-RL `play.py`, Anymal-C rough PLAY
  configuration, `utils/buffers/delay_buffer.py`, whole_body_tracking's
  `scripts/rsl_rl/play.py`, and the local rate-limited action implementation.
- Verification: Python compilation passed. Two real one-environment GPU
  play smoke runs, using model_11999 and the last reference frame to exercise
  automatic reset, each ran six steps, two manual resets and one information
  request. With the flag, every report was 0 steps / 0 ms and the action delay
  buffer was absent. Without it, reports covered 0/1/2 steps / 0/20/40 ms.
  Assertions covered all reporting paths; both processes exited successfully.

## 2026-09-09 - Correct control delay to 0--5 physical milliseconds

- The user authorized implementation and revised the earlier deferred
  0--2-ms target to **0--5 ms**. The shared CR1 ThinRails visual PPO task now
  sets `sim.dt=0.001`, `decimation=20`, `render_interval=20`, and delay bounds
  0--5 physics steps. The 0824 blind MiddleStand/SmoothOutput and visual
  descendants inherit this. The generic LadderEnvCfg and older no-delay
  baseline remain at 5-ms physics / decimation 4.
- Moved the existing IsaacLab DelayBuffer from policy-rate `process_actions`
  to physics-rate `apply_actions`. Sampling remains uniform per environment
  at reset, fixed within an episode. Scale, offset and the 6-rad/s target
  rate limit still run once per 20-ms policy step; only target delivery is
  delayed. The existing implicit actuator model is unchanged.
- Reset clears only selected histories. Until the first delayed target
  arrives, hold the measured new reset pose instead of leaking old targets
  or allowing DelayBuffer's empty-history fallback to bypass the delay.
  No alpha filter or startup fade was introduced.
- Play now labels lag as `control_delay_physics_steps` and converts to ms
  with `env.physics_dt`. Its disable-randomization option still eliminates
  the delay buffer. The shadow diagnostic also labels units as physics
  steps. Rewards, observations, camera timing, reference motion, reset-frame
  sampling and other randomization settings are unchanged.
- References checked: IsaacLab v2.1 `manager_based_rl_env.py`,
  `envs/mdp/actions/joint_actions.py`, `actuators/actuator_pd.py`
  (`DelayedPDActuator`), `utils/buffers/delay_buffer.py`, the official
  locomotion `velocity_env_cfg.py`, whole_body_tracking's `tracking_env_cfg.py`,
  and the installed IsaacLab 2.3.2 buffer implementation. No runtime dependency
  on whole_body_tracking was added.
- Verification: compilation passed; **38 CPU tests passed**, including exact
  0/1/2/3/4/5-ms arrival, policy-rate target limiting, partial reset isolation,
  per-episode sampling, play units, shadow switches and camera/history timing.
  Tests execute the complete action class with both reference and installed
  IsaacLab buffer implementations. **5 configuration tests passed** inside
  the actual Isaac Sim runtime, covering current blind and visual descendants.
- Three one-environment GPU play probes loaded the existing model_11999.
  Each ran six policy steps / 120 physics applications, two manual resets,
  six automatic last-frame resets and one info request. The normal-mode probe
  retained reward computation and checked finite observations/rewards and done
  shapes. The fast enabled probe explicitly covered all six lag values and
  checked actual articulation targets at every physics application. The fast
  disabled probe verified buffer absence and zero delay throughout. All exited
  successfully. Fast probe logs:
  [enabled](../outputs/control_delay_20260909/fast_enabled.log) and
  [disabled](../outputs/control_delay_20260909/fast_disabled.log).
- These are bounded timing/reset smoke checks, not a full-climb stability or
  throughput result. Old weights remain loadable, but now run under the new
  physics/delay configuration. Increasing physics substeps from 4 to 20 adds
  simulation work while preserving simulated policy/reference timing. No long
  training was started and no existing model was changed.

## 2026-09-09 - Restore 5-ms physics with two-point 0/5-ms delay

- At the user's request after the 1-ms setup proved too slow, changed the
  shared CR1 ThinRails task to `sim.dt=0.005`, `decimation=4`, and
  `render_interval=4`. Its current blind MiddleStand/SmoothOutput and visual
  descendants inherit the change. Policy/reference/reward timing remains
  20 ms (50 Hz).
- Delay bounds are now 0--1 **physics steps**, i.e. 0 or 5 ms, each sampled
  with probability 50% per environment at reset and held for the episode.
  Kept the corrected physics-rate `apply_actions` delivery and reset-history
  clearing unchanged. No fractional interpolation, alpha filtering, reward,
  observation, actuator, camera, reset-frame or other randomization changes.
- References rechecked: official IsaacLab v2.1 `manager_based_rl_env.py`,
  `DelayedPDActuator` in `actuators/actuator_pd.py`, official locomotion
  `velocity_env_cfg.py`, and whole_body_tracking's `tracking_env_cfg.py`.
  The latter two use 5-ms physics with decimation 4.
- Verification: compilation and **48 CPU tests passed**; added the two-point
  four-substep delivery regression and tested reset sampling and play's
  physical-time conversion at 5 ms. **5 configuration tests passed** inside
  Isaac Sim, including the blind and visual task descendants.
- A one-environment GPU probe loaded model_11999 and ran six control steps
  with rewards enabled. Verified exactly **24 physics applications**, actual
  articulation target delivery at lag 0 and 1, finite observations/rewards,
  correct done shapes, two manual resets, six last-frame automatic resets,
  and the info-report path. Reports showed 0/5 ms, not 0/20 ms; exit code 0.
  [Probe log](../outputs/control_delay_20260909/two_point5ms_enabled.log).
  This checks timing/reset correctness, not full-climb success or throughput.
- Supplied a fresh blind SmoothOutput training command: 6144 environments,
  10000 iterations, save every 500, seed 42, Stage36 motion, run name
  `train10000_scratch_delay0or5ms_phys5ms_seed42`. No resume/checkpoint flag;
  no long training was launched and no existing training process was stopped.

## 2026-09-09 - Synchronized visual MDP and same-input DAgger control

- Added separate synchronized visual DAgger registrations instead of changing
  the historical visual tasks. Both new environments inherit the current
  Stage-36 blind `MiddleStandSmoothOutput` environment directly, so rewards,
  reset mixture, termination thresholds, domain randomization, 5-ms physics,
  50-Hz policy rate, and the per-episode 0/5-ms control delay stay identical.
  Only the observation groups are replaced. The visual variants select the
  Stage-37 motion file because it adds the up/down annotations required by the
  deployable observation; all shared motion/reference/contact arrays were
  checked against Stage 36. Pure visual distillation still disables rewards.
- Added `Ladder-Climb-CR1-ZDW0824-SameInput-Dagger-Control-v0` as a positive
  control. It creates no camera and exposes one 780-D full-state `policy`
  tensor. Student and teacher both read that exact group, with identical
  `780 -> 512 -> 256 -> 256 -> 25`, ELU, unnormalized MLP contracts. Runtime
  assertions reject a dimension, group, normalization, or value mismatch.
- The control uses RSL-RL's standard `DistillationRunner` and action-mean MSE:
  the randomly initialized student samples every environment action; the
  frozen teacher supplies deterministic online labels at those student-visited
  states. No PPO, visual encoder, feature loss, KL loss, teacher rollout, or
  mixed control is active. Student exploration std is fixed at 0.1 because the
  deterministic MSE does not train the distribution parameter.
- Strictly load the teacher's `actor_state_dict`, save its path and SHA-256,
  reject resume from a different teacher, and restore the next update index so
  an iteration is not repeated. Play accepts only checkpoints containing
  `student_state_dict`, loads only the student, and rejects accidentally passing
  the teacher RL checkpoint.
- Record the immutable task/motion/timing/reset/optimizer contract, per-update
  action RMSE/MAE/max error, all 25 per-joint RMSEs, action magnitudes,
  exploration RMS, clipped gradient norm, parameter delta, student rollout
  fraction, and shared-input max difference. Every 500 updates a raw sample of
  observations, teacher/student means, executed student action, reward, and
  done flags is saved. Environment and agent YAML/pickle plus normal episode,
  reward, and termination metrics remain enabled.

References checked:

- `isaaclab_v2.1.0/source/isaaclab_rl/isaaclab_rl/rsl_rl/distillation_cfg.py`
- installed IsaacLab AnyMal-D same-observation reference
  `isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_distillation_cfg.py`
- installed RSL-RL 5 `runners/distillation_runner.py`,
  `algorithms/distillation.py`, and `models/mlp_model.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- the current blind Stage-36 environment and its saved `env.yaml`/`agent.yaml`

Verification:

- [x] Changed Python files compile successfully.
- [x] RSL compatibility and DAgger CPU suite: **14 passed**. Tests cover one
  shared tensor, online teacher labels, actual student actions, student-only
  parameter updates, frozen teacher weights, provenance, resume update count,
  diagnostics, and config conversion.
- [x] A real two-environment, one-update GPU smoke loaded the current blind
  `model_9000.pt`; the 780-D inputs were exactly equal, student rollout fraction
  was 1, all values were finite, the student changed, and model/manifest/sample
  artifacts were saved.
- [x] The distilled smoke checkpoint completed a real one-environment fast-play
  reset/step run using its student state only.
- [x] A one-environment Isaac Sim instantiation of the synchronized pure visual
  task confirmed depth `60x80`, two depth-history frames, nine proprio-history
  frames, 780-D teacher input, latest reset/termination settings, and successful
  checkpoint construction. These are bounded plumbing checks, not evidence that
  either student has learned to climb.

## 2026-09-10 - Visual action-MSE DAgger after the same-input control

- Added the independent task
  `Ladder-Climb-CR1-ZDW0824-UpDown-Visual-MSE-Dagger-Synced-v0`; historical
  visual KL/feature, teacher-only BC, mixed-rollout, and hybrid PPO tasks were
  not changed. The new task inherits the current blind
  `MiddleStandSmoothOutput` MDP and keeps its rewards for diagnostics. It swaps
  only the observations to the deployable visual contract and selects the
  Stage-37 up/down annotation motion.
- The student consumes nine 85-D proprioceptive frames and two `60 x 80` Warp
  depth frames. The command is the final scalar of every proprio frame; no
  reference phase, target pose/contact, height-map label, velocity target, or
  teacher feature is exposed. The Warp camera remains at 30 Hz and its history
  term inserts only fresh frames and refills both histories at reset.
- The student is the existing direct-CNN policy: three 8-channel stride-2
  convolutions produce a 224-D visual feature, which is concatenated with the
  765-D proprio history and passed through
  `989 -> 2048 -> 1024 -> 512 -> 256 -> 25` with ELU. There is no decoder,
  reconstruction head, velocity head, feature target, or recurrent state.
- Added separate action-MSE algorithm and storage. Every environment executes
  a stochastic student action with fixed standard deviation 0.1. At the same
  student-visited state, the frozen privileged teacher supplies its
  deterministic action mean. The sole optimized loss is elementwise-mean
  `MSE(student_mean, teacher_mean)`; PPO, action KL, teacher-feature MSE, and all
  auxiliary visual losses are absent. Twenty-four rollout steps and 24 shuffled
  mini-batches give 24 optimizer updates per iteration and a 5120-sample batch
  when training with 5120 environments.
- The selected teacher is
  `model_9999.pt` from the 2026-09-09 10k blind SmoothOutput run. Its strict
  actor contract is `780 -> 512 -> 256 -> 256 -> 25`, with no actor observation
  normalization. SHA-256:
  `dc03d17436781a3fd24eae746fb340ce0da3b4f133be62b56cf1f9ee64f59dba`.
  Checkpoints persist the teacher path/hash and reject resume/play with a
  different teacher.
- Diagnostics record the immutable task/motion/reset/timing/camera/optimizer
  contract, training and pre-update rollout action errors, all 25 per-joint
  RMSE and correlations, action magnitudes/correlation/cosine, exploration,
  reward/done rates, command balance, depth statistics/freshness, reset-filled
  history frequency, optimizer steps, and per-module gradient and parameter
  delta norms. Every 500 updates saves raw proprio/depth/teacher observations,
  teacher/student means, executed action, reward, and done.
- References checked: current blind and synchronized visual CR1 configs;
  IsaacLab/RSL-RL `OnPolicyRunner`, PPO config schema, official distillation
  runner/action-MSE implementation; the existing pure visual wrapper/history
  cache; and the accepted same-input positive control.
- Verification: changed files compile; the DAgger CPU suite reports **10
  passed** and the synchronization/camera suites report **23 passed, 1
  skipped**. A CPU integration check strict-loaded the real `model_9999.pt`,
  produced 25-D labels, completed an optimizer update, changed the CNN encoder,
  and confirmed that no KL/feature/PPO/reconstruction loss was emitted. A real
  IsaacLab launch parsed the new environment/runner and selected 30-Hz depth,
  but the current execution container exposes no CUDA device, so it stopped
  before PhysX/Warp environment creation. A GPU reset/step smoke remains before
  launching the long 5120-environment run.

## 2026-09-10 - Recurrent visual MSE-DAgger with camera-clock state updates

- Added the independent task
  `Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-MSE-Dagger-Synced-v0`;
  the prior two-depth-frame feed-forward MSE control remains unchanged. The
  recurrent task still inherits the latest blind `MiddleStandSmoothOutput` MDP
  and uses the Stage-37 up/down annotation motion.
- Every student proprioceptive history now contains 25 complete 85-D frames.
  The up/down/stop command remains index 84 in **every** frame, so the 50-Hz
  history covers 480 ms without a separate privileged phase signal. One current
  `60 x 80` head-camera depth exposure is supplied instead of treating two
  images as independent CNN channels.
- Added an explicit per-environment `depth_updated` term to the Warp history
  cache. It is true for a new 30-Hz exposure and for a reset environment's first
  image, remains stable across duplicate observation queries in the same policy
  step, and is false while the camera image is held. The visual GRU updates only
  where this mask is true; the actor still emits actions at 50 Hz.
- The student uses a reference-derived `1 -> 32 -> 64 -> 128` depth CNN, a
  `2125 -> 512 -> 256 -> 128` proprio encoder, and a 256-D GRU over their fused
  features. A 256-D recurrent projection and the current 85-D proprio frame feed
  the existing `2048 -> 1024 -> 512 -> 256 -> 25` actor. There are no decoder,
  reconstruction, velocity, teacher-feature, KL, or PPO losses.
- Recurrent DAgger storage never flattens or shuffles the time axis. It batches
  complete 24-step trajectories by environment, saves the rollout-start hidden
  state, applies `done[t]` before processing `t+1`, and backpropagates through
  each continuous sequence. Overlapping 25-frame histories are reconstructed
  from the rollout-start history plus current frames, including reset refill,
  instead of storing all overlapping history tensors.
- Training and play maintain one hidden state per environment. Automatic and
  manual resets clear only the affected environments; held camera frames do not
  advance the GRU. Checkpoints use a new strict data contract and do not contain
  transient episode hidden state. Diagnostics add fresh-camera fraction, hidden
  RMS/max/reset fraction, sequence length, module gradients/parameter deltas,
  raw hidden snapshots, the complete task/camera/reset contract, and immutable
  teacher provenance.
- The new runner defaults to the latest retained blind teacher,
  `2026-09-09_13-56-37_train10000_scratch_delay0or5ms_phys5ms_seed42/model_9999.pt`;
  an explicit CLI teacher may still override it. Resume and play continue to
  recover and verify the teacher path/hash stored by the student checkpoint.
- References checked: IsaacLab/RSL-RL 5 `RNNModel`, `RNN`, recurrent PPO reset
  handling, and distillation runner; X2 `x2_vis_config.py`,
  `x2_vis_sparse_config.py`, and `actor_critic_recurrent.py`; existing LadderMan
  Warp camera/history timing tests and synchronized visual DAgger task.
- Verification: all changed Python files compile; complete `Ladder/tests` reports
  **309 passed, 5 skipped**. Dedicated tests prove per-environment reset, held
  image behavior, exact sequence/step equivalence, 25-frame reconstruction over
  done boundaries, and nonzero gradients/updates through the depth CNN,
  proprio encoder, GRU, recurrent projection, actor body, and action head. A CPU
  integration smoke strict-loaded the retained 780-D blind
  `2026-09-09 .../model_9999.pt`, generated 25-D labels, updated a two-step GRU
  sequence, and strict-restored its new checkpoint contract. Isaac Sim parsed
  the registered task, runner, and 30-Hz camera configuration; the reset/step
  GPU smoke could not run because this execution container exposes no CUDA GPU.

## 2026-09-10 - Recurrent visual mixed-rollout DAgger

- Added the independent task
  `Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-Mixed-MSE-Dagger-Synced-v0`.
  It reuses the recurrent visual environment, network, teacher, observation
  contract and sequence action-MSE loss. The existing 100% stochastic-student
  rollout task remains unchanged as the comparison baseline, including for the
  training process that was already running when this source change was made.
- The new controller schedule uses 100% deterministic teacher action means for
  iterations 0--499. From iteration 500 onward, the lowest 25% of environment
  IDs remain teacher-controlled and the other 75% execute stochastic student
  actions. With 5120 environments this is a fixed 1280/3840 split; it is not
  per-step switching or numerical action interpolation.
- Both networks are evaluated in every environment. The student GRU therefore
  continues to consume the deployable observation and update on fresh camera
  frames even where the teacher advances the simulation. The frozen teacher's
  action mean labels every teacher- and student-visited state, and every stored
  sample enters the same unweighted sequence MSE update. Reset handling and
  camera freshness semantics are unchanged.
- Added controller-aware diagnostics: actual teacher/student rollout fractions,
  action RMSE on each state distribution, per-step reward and done fraction for
  each controller, and each termination cause as a fraction of that
  controller's dones. Raw snapshots select both controller groups after warmup
  and record environment IDs, controller mask, candidate student action and
  action actually sent to the environment. The manifest records the complete
  schedule. Checkpoints persist the warmup/fraction and reject resume under a
  different schedule, while historical student-only checkpoints remain
  compatible with the original task.
- Reference checked: the existing LadderMan pure visual mixed DAgger controller
  and fixed-prefix schedule in `dagger/pure_algorithm.py`, its all-environment
  storage in `dagger/pure_storage.py`, and the current recurrent freshness,
  hidden-reset and sequence-storage implementation. No MDP, reward, reset,
  camera, teacher or action convention was changed.
- Verification: changed files compile; the full CPU suite reports **310 passed,
  5 skipped**. Tests cover full-teacher warmup, fixed 25/75 switching, exact
  selected actions, split reward/done/termination metrics, all-sample training,
  schedule provenance and mismatched-resume rejection. A separate CPU
  integration strict-loaded the real blind `model_9999.pt`, executed a
  two-step warmup-to-mixed transition, completed the recurrent optimizer update,
  and strict-restored the resulting checkpoint. A new Isaac Sim GPU reset/step
  smoke remains unavailable in this tool container because CUDA is not exposed.
  An AppLauncher-only registry/config probe was also attempted, but this
  container's Isaac runtime exited with signal 11 before task parsing; actual
  Isaac process startup therefore remains to be checked in the user's GPU
  terminal before launching the long run.

## 2026-09-11 - Recurrent visual mixed DAgger without BC warmup

- Added the independent task
  `Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-Mixed-NoWarmup-MSE-Dagger-Synced-v0`.
  It changes one experimental variable relative to the September 10 mixed
  task: `bc_warmup_iterations=0`. From iteration zero, the lowest 25% of
  environment IDs execute deterministic teacher means and the remaining 75%
  execute stochastic student actions. With 5120 environments the fixed split
  is 1280 teacher / 3840 student. The teacher, student network, 30-Hz camera,
  recurrent state/reset semantics, MDP, optimizer and action-mean MSE are
  unchanged. The old 500-iteration-warmup task and its checkpoints remain
  registered and unchanged.
- Added online representation diagnostics needed to detect the collapse seen
  in the warmup run without post-hoc checkpoint probing. TensorBoard now
  records RMS and maximum absolute values for proprio features, depth
  features, their GRU input, the GRU candidate, recurrent projection and final
  actor feature. It also records GRU-candidate and selected-hidden saturation
  fractions plus the selected hidden state's exact-zero fraction. Periodic raw
  snapshots retain the corresponding activation tensors for both fixed
  controller groups. The diagnostic manifest explicitly distinguishes a
  no-warmup mixture starting at iteration zero from warmup-then-mix control.
- Reference checked: the fixed-prefix mixed controller in
  `dagger/pure_algorithm.py`, the existing recurrent sequence/freshness/reset
  implementation in `recurrent_visual_model.py` and
  `recurrent_visual_mse_storage.py`, and IsaacLab's per-environment recurrent
  reset convention. No runtime dependency on a reference project was added.
- Verification: changed sources compile and the complete CPU test suite reports
  **310 passed, 5 skipped**. Tests cover the unchanged masked GRU step/sequence
  equivalence, activation-tensor capture, scalar diagnostic bounds, mixed
  action selection and the new task registration/config contract. A new Isaac
  GPU reset/step smoke remains to be performed in the training terminal because
  CUDA is not exposed in the tool container.

## 2026-09-12 - Normalized recurrent visual PPO plus MSE DAgger

- Added independent task
  `Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-PPO-Dagger-Synced-v0`.
  The frozen teacher remains the September 9 blind `model_9999.pt`. A fresh
  student uses per-branch 128-D LayerNorm before the camera-rate GRU; a fresh
  `780 -> 512 -> 256 -> 128 -> 1` privileged critic reuses the teacher's
  observation tensor without exposing it to the actor. Rewards, reset,
  camera timing, contact geometry and control delay use the synced task.
- From iteration zero, fixed 25% teacher environments enter MSE only; fixed
  75% student environments enter MSE and PPO. GAE, value targets, advantage
  normalization, entropy and old/new-policy KL exclude teacher trajectories.
  Timeout bootstrapping follows installed RSL-RL and excludes simultaneous
  true failures. Stored raw observations, actual actions, old probabilities,
  values and hidden states support full sequence recomputation. PPO gradients
  reach proprioception, depth CNN and GRU as well as the actor head.
- PPO starts immediately with 3 epochs/24 minibatches and adaptive learning
  rate `3e-4` bounded by `[1e-5, 3e-4]`. Action MSE coefficient starts at 1.0,
  decays linearly during updates 1000--5000 and remains at 0.1. This coefficient
  is experimental; independent PPO/MSE gradient probes document its effective
  influence. Student action std starts at 0.1 and is learned. No auxiliary
  reconstruction, velocity or feature-matching loss is restored.
- Diagnostics retain split-controller/per-joint metrics and add PPO/value/
  entropy losses, KL/clip fraction, return/value estimates, critic explained
  variance, learned std and separate objective gradients. Periodic snapshots
  now include the exact pre-update student weights for action replay; regular
  checkpoints are post-update. A new checkpoint contract persists critic,
  optimizer, learning rate, annealing state and training semantics. Existing
  MSE tasks are unchanged. Train, resume and deterministic student play route
  through the new algorithm; play does not load the critic or optimizer.
- References checked: IsaacLab H1 PPO config and termination manager, installed
  RSL-RL `algorithms/ppo.py` and `runners/on_policy_runner.py`, reference
  `whole_body_tracking/.../g1/agents/rsl_rl_ppo_cfg.py`, LadderMan PDF section
  3.2, and existing recurrent DAgger and detached-feature visual PPO paths.
- Verification: actual IsaacLab configclass/RSL configuration instantiation,
  validation and serialization passed. Full CPU suite: **325 passed, 5 skipped**.
  New tests cover numerical GAE, teacher-data isolation including NaNs, masked
  step/sequence equivalence, PPO-only encoder/GRU gradients, learned std,
  frozen teacher, checkpoint/actor-only restoration and exact snapshot replay.
  Installed RSL runner with the real teacher and full actor completed two CPU
  synthetic-rollout updates and TensorBoard/checkpoint restoration; pre-update
  ratio error was about `3.8e-6`. This is not a physics/performance evaluation.
  CUDA is unavailable here; GPU reset/step and throughput remain unverified.
  No new long training was launched. Commands and diagnostic definitions are in
  `dagger/RECURRENT_PPO_DAGGER.md`.

## 2026-09-12 - PPO actor ramp and delayed distillation annealing

- Updated only the recurrent visual PPO-DAgger task's objective schedule:
  `alpha=0.1` at update 0, linearly reaching 1 at update 500; action-MSE
  `lambda=1` through update 2000, decreasing to 0.1 at update 7000 and then
  staying there. The actor uses `alpha * (surrogate - entropy_coef * entropy)`;
  critic value loss remains unscaled and actor/critic clipping remains separate.
  Fixed 25% teacher / 75% student control, teacher `model_9999.pt`, observations,
  LayerNorm/GRU, camera timing, rewards, resets and delays are unchanged.
- Added raw/weighted PPO gradient diagnostics, applied PPO and entropy
  coefficients, weighted actor loss and schedule metadata. Checkpoints retain
  the schedule parameters and absolute update index; full optimizer resume
  rejects changed or missing schedule fields, while actor-only play remains
  compatible. Startup messages and the training README show the actual schedule.
  This is an iteration-based experimental schedule, not automatic performance
  gating and not a paper-provided timetable. Objective scaling is not an Adam
  step-size guarantee; critic and actor still share the adaptive learning rate.
- References checked: IsaacLab `isaaclab_rl/rsl_rl/rl_cfg.py` and G1 velocity
  `agents/rsl_rl_ppo_cfg.py`; `whole_body_tracking` G1 tracking PPO config;
  installed RSL-RL `algorithms/ppo.py` and `runners/on_policy_runner.py`;
  existing LadderMan recurrent PPO/DAgger implementation and PDF section 3.2.
- Verification: real IsaacLab configclass/RSL configuration `validate()` and
  `to_dict()` passed. Full CPU suite: **338 passed, 5 skipped**. Tests verify
  schedule boundaries, weighted entropy/total loss, actual actor and unscaled
  critic gradients, checkpoint progress and mismatched-schedule rejection.
  The installed RSL runner with the real teacher and full actor completed two
  CPU synthetic-rollout updates, logging alpha `0.1, 0.1018`, and restored at
  `0.1036`; pre-update probability-ratio error remained about `3.8e-6`.
  These checks do not establish climbing performance. CUDA remains unavailable
  in this tool environment, so GPU physics/throughput still need a smoke test.
  No long training was launched.

## 2026-09-12 - LayerNorm-only MSE control and fixed-start evaluation

- Added independent task
  `Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-LN-MSE-Dagger-Synced-v0`.
  It reuses the normalized network, but the algorithm is the original pure
  MSE path: fixed 25/75 control, fixed std 0.1, lr 1e-3, one epoch and 24
  minibatches. No critic, PPO, entropy or auxiliary losses are constructed.
  The only network additions are two 128-D LayerNorms (512 affine parameters).
  Original tasks and checkpoint contracts remain unchanged; the control has
  its own strict contract, task registration and train/play routing.
- Added `scripts/rsl_rl/evaluate_recurrent_dagger.py` and tensor-only first-
  episode accounting. The evaluator loads actor weights without training
  storage/optimizers, supports mean/teacher/fixed-std/checkpoint-std control,
  retains training randomization and 0-or-5ms delay, and records true terminal
  state before autoreset. It saves hashes, full environment config, initial
  tensors, episode records, clean success, censored outcomes and phase progress.
  Evaluation uses frame0 (full up/down) and frame1550 (descent), not frame1500.
- References checked: IsaacLab G1 gym registration and RSL play script,
  installed termination/step ordering, whole_body_tracking RSL play, existing
  LadderMan play/diagnose_visual_student hooks, MSE/PPO students and motion37
  command/reset semantics. No runtime reference-project dependency was added.
- Verification: **347 CPU tests passed, 6 skipped**. A host-GPU 96-env,
  two-update training smoke completed and an 8-env/two-step evaluator smoke
  correctly classified all unfinished episodes as censored. The full control
  was then started with 5120 envs, 2000 updates and save interval500 under
  `logs/rsl_rl/cr1_zdw0824_recurrent_visual_ln_mse_dagger_model9999/2026-09-12_10-42-07_ln_only_mse_teacher9999_env5120_iter2000_seed42`.
  `model_0.pt` is finite with std still0.1. Saved full environment config is
  identical to the old no-warmup MSE run; algorithm config differs only by
  class name and policy config is identical. Original PPO training is retained.
- Pinned PPO `model_4500.pt` evaluations are stored under
  `outputs/recurrent_dagger_eval/2026-09-12_model4500/`.
  Frame0 teacher/student initial state and observation hashes match exactly.
  All four first-episode GPU evaluations completed with no censored episodes:
  frame0 teacher30/32 versus student-mean0/32; frame1550 teacher16/32 versus
  student-mean0/32. Both starts have identical teacher/student initial tensor
  and environment-config hashes. Frame0 mean maximum root rise is0.623m for
  teacher versus0.0352m for student. Student frame683 failure clustering follows
  a reference contact transition; `required_contact` is a sustained target-
  distance condition, not contact-force loss. Details and caveats are saved in
  `outputs/recurrent_dagger_eval/2026-09-12_model4500/REPORT.md`.
  At11:07 the LayerNorm control had reached iteration65/2000 and remained
  running. Its final performance comparison is pending; the completed
  deterministic evaluations do not establish LayerNorm training efficacy.

## 2026-09-12 - Resume LayerNorm-only MSE from 2000 to 10000 updates

- The initial5120-env control completed2000 updates and saved `model_1999.pt`
  at17:50. At the user's request, resumed this model with8000 additional updates,
  preserving save_interval500, frozen teacher9999, fixed25/75 control,
  action-mean MSE, lr1e-3 and fixed std0.1. No algorithm or environment code was
  changed. No old training process was running at the pre-launch check.
- New run:
  `logs/rsl_rl/cr1_zdw0824_recurrent_visual_ln_mse_dagger_model9999/2026-09-12_18-48-54_ln_only_mse_resume1999_to9999_env5120_seed42`.
  Source is the10:42:07 run's `model_1999.pt`; stdout is
  `outputs/training_launches/2026-09-12_ln_mse_resume1999_env5120_add8000_session.log`.
  Training PID at launch is2297346. The original run is preserved.
- Checked `train.py`, `recurrent_visual_mse_algorithm.py` and installed RSL-RL
  `runners/on_policy_runner.py`: completed-update count2000 is restored as the
  next iteration, and8000 means additional updates. Logging correctly starts
  at2000/10000. Checkpoints are2000,2500,...,9500 and final9999.
- Verified new `model_2000.pt`: update_iteration2001, all34 Adam parameter
  states at step48024 (continued from48000), lr0.001, fixed std0.1 and finite
  policy weights. Full saved environment YAML exactly matches the source run.
  Agent YAML differs only in resume flag, checkpoint, run name and additional
  iteration budget. Training continues beyond the first restored update.
- Resume restores model/optimizer/training count, not the previous physics
  scene, adaptive-sampler history or online GRU state. Those restart with fresh
  episodes while adaptive reset remains enabled; early resumed metrics can
  therefore differ temporarily from the source run's final window.

## 2026-09-13 - LN-only MSE completes 10000 updates, with rare raw-action spikes

- The18:48:54 resumed run completed at06:24 and saved `model_9999.pt`.
  Final update count10000, Adam step240000, finite policy/optimizer states,
  lr0.001, fixed std0.1 and25% teacher control were verified. No training
  process remained at the11:07 check. No algorithm or parameters were changed
  during this analysis, and no new physics evaluation was launched.
- Student-only100-update windows show gradual improvement, not a decisive
  task breakthrough:4900--4999 reward0.05755, motion-complete fraction22.52%,
  action RMSE0.2124;8900--8999 gives0.06010/25.79%/0.1974;
  9900--9999 gives0.06077/24.56%/0.1948. Completion is weighted by done counts
  across random-reset episodes, not clean full-reference success. The teacher
  and overall reset/command mixture remain broadly stable.
- Full-history inspection exposes rare severe spikes missed by selected-window
  summaries. At9873, student rollout mean reward=-25.3484, action RMSE=27.4636,
  max absolute student-teacher action error=15048.62, training MSE=394.6437,
  and actor-MLP feature maximum=168686.9. This one negative-reward iteration
  makes the true9800--9999 mean reward=-0.066855; the other199 average0.060188.
  Last100 rewards recover, so this is not sustained late-stage collapse, but
  the outlier must not be discarded when assessing reliability. Related severe
  events occurred at2393/2394,3273/3274,4118/4119,6583/6584/6590,9400/9401.
- Action-rate and lower-body action second-difference penalties dominate the
  reward spikes; torque, joint-acceleration and contact terms do not show a
  matching general blow-up. Checked `mdp/actions.py`, `mdp/rewards.py`,
  `mdp/observations.py`, the recurrent student/MSE algorithm, and official
  reward-manager/runner semantics. The rate limiter bounds processed joint
  target increments, not raw policy actions. Wrapper and action clipping are
  disabled in the saved config. Encoder LayerNorm does not cover the actor's
  direct current-proprio input, which includes previous action. That feedback
  path and preceding large optimizer updates are suspects, not proven causes.
- GRU saturation remains low (0.0778% in the final100), unlike the old unnormalized
  model's78.58% around2000 updates. Finite tensors and healthy GRU statistics
  therefore do not establish reliable actor output. Every500-update/16-env
  snapshots miss the spike trajectories, preventing attribution to a specific
  triggering observation, environment or joint. Further work should separate
  fixed-start mean-policy capability from rare-action-tail diagnosis.

## 2026-09-13 - Independent end-to-end PPO plus velocity/heightmap reconstruction

- Added task `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-ZDW0824-WarpTransformerVision-EndToEnd-Reconstruction-v0`.
  `CR1EndToEndVisualPPOEnvCfg` directly inherits the retained latest blind
  `CR1ZDW0824NoCommandFullStateMiddleStandSmoothOutputEnvCfg`. Only observation
  groups and the equivalent Stage37 command-annotated motion change. Compared
  against both the current blind config and the September09 13:56:37 saved
  training YAML, including all reward/reset/action/termination/randomization
  fields. Saved runtime-expanded prim paths and the training entrypoint's
  two debug-visualization overrides are normalized only for this comparison.
  Every original Stage36 motion array equals its Stage37 counterpart.
- Latest MDP remains physics5ms/decimation4, 0-or-5ms action delay, target-rate
  limiter6rad/s, latest upper/lower joint-acc and lower-body smoothness rewards,
  original terminations and full DR. Reset fixed/secondary/uniform/adaptive
  remains10/10/40/40%, frame0 pose-randomization probability0.7, and
  `use_reset_valid_mask=False`, exactly as the retained blind environment.
  Motion semantic-progress supervision stays disabled. No blind or DAgger
  environment/reward/action implementation was changed.
- New `EndToEndVisualActor` consumes25x85 proprio history (command in every
  frame) and two60x80 head-depth exposures at30Hz. CNN/state-token encoder,
  two Transformer layers and GRU256 produce velocity3 + four8-D limb-map
  latents. Current85 + latent35 enters the512/256/128 action MLP ->25 actions.
  GRU updates at every50Hz policy step using held camera data when necessary.
  The privileged critic uses the same780-D group and512/256/128 MLP as blind
  PPO. There is no teacher, imitation loss, reference-motion decoder,
  free dynamic latent, VAE KL or next-frame reconstruction.
- New `EndToEndVisualPPO` replays complete raw24-step sequences with saved
  rollout-start hidden and reset-before masks. Pending observations/labels
  are cloned before env.step can mutate them. Transformer dropout is zero;
  rollout/update/play use FP32 with TF32 disabled in the new entrypoint branch.
  One Adam updates policy (including visual encoder/decoders) and critic:
  PPO surrogate + value -0.005 entropy + (2 velocityMSE +2 handMSE +2 footMSE).
  Each hand/foot term averages its two limb-map MSEs; rung pixels have weight2
  versus1 for other valid pixels, including in the denominator. Logical4
  minibatches x5 epochs use32-env sequence microbatches with exact global
  denominators. InitialLR3e-4, adaptive sum-over-actions KL target0.01,
  LR bounds1e-5..1e-3, actor/critic gradient clip1, trainable std initialized1
  without the old0.2 floor. The decoders receive reconstruction gradients;
  the entire action-producing visual path receives PPO gradients.
- Added independent checkpoint data/training contracts, optimizer/LR/update
  restoration and correct next-iteration resume handling. Train and existing
  `play_transformer_vision_ppo.py` support the new task; old tasks/checkpoint
  loading behavior remains intact. Playback handles the absent dynamic/future
  heads and retains heightmap overlays.
- Diagnostics: TensorBoard and per-update JSONL include raw/weighted losses,
  PPO/aux per-module gradient norms, ratio/cosine, KL/clip fraction, learning
  rate, pre-update probability consistency, action tails and critic quality.
  Every500 updates stores up to16 envs' complete raw rollout with matching
  pre-update policy/critic weights. Params include actual env/agent YAML/PKL,
  motion hash, selected implementation snapshots/hashes and timing metadata.
  Snapshots are diagnostic subsets, not exhaustive rare-event capture.
- Verification: CPU suite372 passed/7 skipped; all5 real Isaac config/motion
  tests passed. GPU96-env/two-update smoke completed with finite gradients and
  checkpoints0/1. PPO-only gradient probes for CNN/Transformer/GRU were
  respectively0.04635/0.03936/0.04767 on the first probe; pre-update ratio
  error3.8147e-6. Four-env headless playback loaded model1, ran80 steps and
  reset at40 successfully. This is an implementation check, not a trained
  policy performance claim. Artifacts: `outputs/end_to_end_visual_ppo_smoke.log`,
  `outputs/end_to_end_visual_env_check_final.log`, and
  `outputs/end_to_end_visual_ppo_play_smoke/model_1/summary.json`.
- References checked: official IsaacLab `rl_cfg.py`, G1 velocity PPO config
  and InteractiveScene prim-path expansion; installed RSL-RL PPO, recurrent
  storage and OnPolicyRunner; `whole_body_tracking` G1 PPO/observations; local
  detached TransformerVision PPO and recurrent visual PPO tests. No runtime
  dependency on whole_body_tracking was introduced.
- Started the requested fresh trial after all checks:5120 envs,10000 updates,
  save_interval500, seed42, full training randomization. Run:
  `logs/rsl_rl/cr1_zdw0824_end_to_end_visual_reconstruction_ppo/2026-09-13_16-08-30_scratch_e2e_ppo_reconstruction_env5120_iter10000_seed42`.
  Stdout:
  `outputs/training_launches/2026-09-13_end_to_end_visual_reconstruction_env5120_iter10000.log`.
  PID2468200 (managed exec session3668). No earlier training process was
  stopped or checkpoint overwritten. Updates0--2 completed; model0 has finite
  policy weights, num_updates1 and97 initialized Adam parameter states.
  Large-run pre-update ratio error7.6294e-6, nonzero PPO CNN/Transformer/GRU
  gradients, KL0.00824--0.00957 and LR3e-4--6.75e-4. Iteration times after
  initialization are about23.3s. These first updates validate execution, not
  climbing capability. Periodic saves are0,500,...,9500 and final9999 under
  RSL-RL's zero-based model numbering. Training is left running.

## 2026-09-14 - Single-environment end-to-end visual PPO playback

- Fixed `play_transformer_vision_ppo.py` failing before checkpoint load when
  the training configuration retained four logical minibatches but playback
  created only one environment. End-to-end playback now caps only its local
  runner copy to the playback environment count; the training configuration
  and algorithm's strict training-time validation remain unchanged.
- End-to-end playback now explicitly loads only the actor. It does not restore
  the unused critic, optimizer, or update index, so the playback-only minibatch
  adjustment cannot conflict with the checkpoint's immutable training
  contract. Existing TransformerVision checkpoint loading is unchanged.
- References checked: the installed RSL-RL `OnPolicyRunner` play/load path,
  the local end-to-end algorithm training-contract loader, and the existing
  actor-only recurrent visual playback branches.
- Verification: changed files compile; end-to-end actor/wrapper/PPO/config
  tests report **26 passed, 1 skipped**. A CPU regression additionally loads a
  four-environment/four-minibatch saved actor strictly into a
  one-environment/one-minibatch playback instance and verifies every policy
  tensor. A new GPU playback was not launched while the active 5120-environment
  training process was using the device.

## 2026-09-14 - Oracle contact-goal visual PPO diagnostic and execution cleanup

- Added independent task
  `Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-ZDW0824-WarpTransformerVision-EndToEnd-OracleContactGoal-v0`.
  It inherits the current end-to-end environment. Apart from the new observation
  group, its serialized configuration equals the baseline: no changes to
  physics, action delay/rate limiting, rewards, reset sampling or reference clock.
- The actor receives current85 + visual latent35 + oracle16. The extra16 are
  masked body-frame reference limb goals12 followed by desired contact4, ordered
  left hand/right hand/left foot/right foot. Goals are current/next scheduled
  ladder targets, not joint reference poses. The raw group includes validity
  and normalized reference rung IDs for inspection, but these are NOT additional
  actor inputs. This is privileged diagnostic input, not a deployment policy.
- All rollouts remain current-policy PPO. CNN/Transformer/GRU/velocity/map
  heads, auxiliary losses, critic, optimizer and schedules are unchanged.
  Oracle values are cloned with the transition, stored in actor_inputs[...,120:136],
  and used alongside recomputed visual latents in PPO sequence replay. Play's
  actor-only conversion retains the oracle input and warns about privilege.
  Baseline stays120D and ignores extra oracle fields; distinct checkpoint data
  contracts reject baseline/oracle mixing, including actor-only loads.
- Run manifests now record actor privilege/input width. Diagnostic manifests
  record the exact input layout and desired-versus-measured contact semantics.
  Existing TensorBoard/JSONL losses, gradient probes, source/config hashes and
  every500-update pre-update raw snapshots remain enabled; oracle16 is included
  in those snapshots. Raw validity/rung metadata is not separately persisted.
- Architecture-preserving speed cleanup: combine14 per-microbatch logging
  scalars into device-side float64 sums and transfer once per logical batch;
  cache the full reset-before mask once per update. Keep immediate nonfinite
  checks, logical denominators, backpropagation, clipping and Adam steps intact.
  No measured long-run speedup is claimed. Defaults remain32-env microbatches;
  benchmark64/128 next, before considering reset-aware GRU fusion or a separate
  lighter CNN+history-MLP+GRU architecture. Height-map removal is a supervision
  experiment, not a demonstrated main speed optimization.
- Verification: CPU suite379 passed/7 skipped; all7 real Isaac configuration/motion
  tests passed after app startup. Oracle tests cover missing/shape
  errors, input sensitivity, baseline isolation, cached-camera/done replay,
  invalid-goal masking and checkpoint contracts. The logging/cache regression
  matches the old Python scalar sums exactly and checks one reset-mask creation.
  An8-env/4-step GPU smoke with0.06s test-only episodes exercised autoresets,
  finite rewards/actions, one PPO update and actor inference; pre-update ratio
  error0 and nonzero PPO gradients to CNN/Transformer/GRU. This verifies the
  implementation, not task learning. Full training has NOT been started, and
  the existing5120-env baseline was not stopped or modified in memory.
- References checked: official IsaacLab RSL play/config and ANYmal registration
  patterns; `whole_body_tracking/.../tracking/tracking_env_cfg.py` observations;
  local end-to-end runner/storage and retained blind MDP; PyTorch's performance
  tuning guide on unnecessary CPU/GPU synchronization. No reference-repo runtime
  dependency was introduced.

Suggested fresh diagnostic launch (5120 envs,10000 updates, save every500):

```bash
cd /home/zhangdawei/LadderMan
conda activate rl-training
python Ladder/scripts/rsl_rl/train.py \
  --task Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-ZDW0824-WarpTransformerVision-EndToEnd-OracleContactGoal-v0 \
  --motion_file Ladder/motions/zdw0824_capture1/pitchminus3_yawminus4_xminus030/37_visual_up_down_stop_command/zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_reset5mm_visual_up_down_stop_50hz_motion.npz \
  --num_envs 5120 \
  --max_iterations 10000 \
  --run_name scratch_oracle_goal12_contact4_env5120_seed42 \
  --headless \
  agent.save_interval=500
```

Use the independent experiment directory
`logs/rsl_rl/cr1_zdw0824_end_to_end_visual_oracle_contact_goal_ppo/`.
Do not pass a teacher or visual pretraining checkpoint. Compare learning curves
and deterministic climbing/contact performance at matched training updates,
starts and randomization. An oracle improvement motivates testing the command
interface, not immediate replacement with a network guessing reference phase.

## 2026-09-16 - Recurrent visual DAgger numeric-contract audit

- Audited the active scheduled recurrent MSE-DAgger path against
  `mdp/observations.py`, `utils/visual_depth.py`, the synchronized CR1
  environment, the retained blind teacher configuration and the local action
  wrapper. The student does not use running mean/std: proprio is field-scaled
  (angular velocity `0.2`, joint velocity `0.1`, joint position divided by
  `CR1_ACTION_SCALE`) and clipped to `[-100,100]`; metric depth is clipped to
  `[0.1,2.0] m`, centered at `1.0 m`, with unit scale; actions remain in the
  normalized joint-position action coordinates used by both teacher and env.
- Added an explicit normalization contract to the recurrent DAgger diagnostic
  manifest. Teacher running normalization remains disabled deliberately so the
  780-D input exactly matches the `model_9999.pt` checkpoint contract.
- Added fail-fast checks for non-finite values and out-of-contract proprio/depth
  ranges, plus non-finite student/teacher action means, MSE loss and gradients.
  No new running normalizer or silent `nan_to_num` was added to the policy path.
- Verification: recurrent DAgger and LayerNorm ablation tests report
  **23 passed, 1 deselected**; changed Python files compile successfully.
  The new scheduled 5120-env run was started after the audit as
  `normalization_audit_scheduled_5120`; initialization completed and the first
  update finished without triggering numeric-contract checks.

## 2026-09-17 - Privileged recurrent visual DAgger information-sufficiency ablation

- Added an independent task
  `Ladder-Climb-CR1-ZDW0824-UpDown-Recurrent-Visual-Privileged-MSE-Dagger-Synced-v0`
  under `config/cr1/dagger/`. It inherits the synchronized visual DAgger MDP,
  teacher checkpoint and sequence action-MSE update; existing DAgger tasks are
  unchanged.
- The student keeps its 25x85 proprio history, 30-Hz 60x80 depth encoder and
  256-D GRU visual feature, then concatenates a 28-D oracle input before the
  action MLP: next-stable four-limb ladder target positions in the robot anchor
  frame (12-D meters), actual ladder contact flags (4-D from the existing
  rung-filtered contact sensors), and four end-effector linear velocities in
  the robot anchor frame scaled by 0.1 (12-D). No target-validity/reference
  contact bit or rung ID is exposed. This is explicitly non-deployable and
  tests information sufficiency only.
- Next-stable targets are taken from the first stable anchor at or after the
  current reference frame, not from the current reference pose. The new command
  properties keep this target available even when motion-semantic progress is
  disabled for the task. Actual contact does not use the reference contact bit;
  it reads the per-limb rung-filtered force matrices.
- Storage clones the privileged vector for every transition and replays it in
  the original environment/time order. Diagnostic snapshots/manifests record
  the oracle tensor and its layout. The controller schedule matches the latest
  requested privileged-ablation schedule: 100% teacher through update 1000, a
  linear decrease to 10% teacher at update 5000, then 10% teacher. Every
  visited state is labelled by the frozen teacher. Play/evaluation recognize
  the new data contract and retain all four filtered contact sensors in fast
  mode.
- References checked: Isaac Lab lift-task relative-frame observations,
  `ArticulationData.body_link_lin_vel_w`, `ObservationManager` concatenation
  and contact-sensor force matrices; `whole_body_tracking` relative-body
  observation transforms; local `LadderMotionCommand`, motion semantics,
  rung-filtered sensors and synchronized recurrent DAgger
  wrapper/storage/model. No `whole_body_tracking` runtime dependency was
  introduced.
- Verification: changed Python files compile; the new model/storage tests plus
  existing DAgger/model tests report **27 passed, 1 skipped**. A 24-environment
  GPU smoke run completed one 24-step rollout and optimizer update, wrote
  `model_0.pt`, and confirmed observation shapes `25x85`, `1x60x80`, and `28`.
  The action MLP input is `85 + 256 + 28 = 369`; checkpoint tensors are finite.
  The sampled oracle ranges were target position `[-1.629, 0.936] m` and scaled
  limb velocity `[-0.0495, 0.0480]`.
- Added per-update TensorBoard diagnostics for overall/per-limb actual-contact
  fractions, any-limb contact coverage, per-limb target distance, and per-limb
  physical speed. They are reduced on device and transferred together, do not
  participate in the loss, and make an all-zero contact signal or bad scale
  visible during training rather than only in occasional snapshots.
- A second 24-environment GPU smoke run completed after adding these metrics.
  The 24-step rollout measured a `0.5699` mean contact-bit fraction and `0.8663`
  any-limb contact fraction; per-limb contact fractions were left/right hand
  `0.4375/0.3351` and left/right foot `0.7917/0.7153`. This confirms that the
  four actual-contact channels are live rather than constant zero. Mean/max
  physical limb speed were `0.155/3.390 m/s`. No long training was started.
- After the first long run suffered an abrupt finite-value collapse around
  update 158, added the same two branch normalizers used by the existing
  recurrent LN control: `LayerNorm(128)` after the proprio encoder and after
  the depth projection, before their concatenation enters the GRU. Learning
  rate remains `1e-3`, the 1000--5000 controller schedule is unchanged, and
  checkpoint cadence remains 500 updates as requested. The new LayerNorm
  contract is `...30hz_layernorm_gru256_..._v2` and logs use the separate
  experiment directory
  `cr1_zdw0824_privileged_recurrent_visual_ln_mse_dagger_model9999`.
  The legacy unnormalized class/contract remains available to the bounded
  evaluator so old diagnostic checkpoints are not silently loaded as LN
  policies.
- Verification after the LayerNorm change: **28 passed, 2 skipped** in the
  focused CPU suite, followed by a successful 24-environment GPU rollout and
  optimizer update. The saved checkpoint retains learning rate `1e-3` and the
  `1000 -> 5000 -> 10%` schedule, contains exactly four new LayerNorm
  parameters, and is finite. Real rollout snapshots show per-sample feature
  means below `3e-8` in magnitude and branch RMS values of about `0.999` for
  both proprioception and depth.

## 2026-09-21 - Oracle end-to-end PPO uses GRU256 and explicit support state

- Changed only the independent oracle end-to-end visual PPO task; the retained
  120-D reconstruction baseline and its checkpoints remain compatible. The
  oracle action head now consumes current proprioception85, the complete
  recurrent visual feature256, and a32-D privileged support interface, for373
  inputs total. It no longer consumes the compressed velocity/map latent35.
- The32-D oracle is ordered left hand/right hand/left foot/right foot and
  contains next-stable target positions in the robot anchor frame12, measured
  end-effector positions in the same frame12, rung-filtered actual-contact
  flags4, and desired contacts at that next stable support frame4. Invalid
  terminal next-stable contacts are zero. Target and measured positions use
  the identical `subtract_frame_transforms` anchor pose.
- Raw sequence replay stores the373-D action input and reconstructs the oracle
  suffix as32-D during PPO. The data contract, diagnostics, train/play notices
  and strict checkpoint isolation were updated. Auxiliary velocity and four
  limb-map losses remain enabled; PPO reaches CNN, Transformer and GRU directly
  through the256-D feature.
- References checked: IsaacLab locomotion contact-sensor reward patterns,
  `whole_body_tracking` robot-body relative observations, and the existing
  Ladder privileged DAgger next-stable target/contact implementation. No
  reference-project runtime dependency was introduced.
- Verification: changed modules compile; focused actor/oracle/wrapper/PPO tests
  report **32 passed**, with the config/observation check **1 passed, 1
  skipped** outside an AppLauncher. An8-environment real GPU smoke completed a
  24-step rollout and one PPO update. The saved action layer is `[512,373]`,
  all actor inputs are finite, both contact groups are binary and nonconstant,
  pre-update probability replay error is zero, and PPO gradient probes are
  nonzero for CNN, Transformer and GRU. Artifact:
  `logs/rsl_rl/cr1_zdw0824_end_to_end_visual_oracle_contact_goal_ppo/2026-09-21_10-16-51_smoke_oracle32_gru256_env8`.

## 2026-09-21 - Oracle Visual PPO Action Safety Envelope

- Kept the existing per-joint target rate limiter at 6 rad/s and added the
  accepted full-state PPO action bound, `clip_actions=10`, only to the Oracle
  end-to-end visual runner.  Clipping occurs before `env.step`, so IsaacLab's
  raw/previous-action state is bounded as well as the command sent downstream;
  the baseline visual runner remains unchanged for comparison.  The PPO
  diagnostics now record `rollout_action_safety_clip_fraction`, allowing a
  distinction between rare safety clipping and persistent policy saturation.
- Confirmed that the visual encoder still consumes the full 25x85
  command-aware proprioceptive history: it is encoded as a state token and
  fused with the depth tokens before the Transformer and 256-D GRU.
- Audited the Stage-37 motion semantics.  It contains 21 direct stable anchors
  (18 four-limb ladder-contact anchors plus initial, top-stand, and terminal
  ground anchors).  The current Oracle task does not state-lock reference
  progression; its next-stable goal is a feed-forward label derived from the
  continuously advancing 50-Hz reference.

References checked:

- `isaaclab_v2.1.0/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/actions.py`
- `Ladder/source/ladder/ladder/utils/motion_semantics.py`

## 2026-09-21 - Stable-Anchor-Gated Oracle Reference

- Added `--stable_frames_only` to `replay_ladder_reference.py`.  For the
  Stage-37 motion it cycles through the 21 `zero_command_direct_hold_frames`
  rather than traversing all 3061 reference frames; the existing contact and
  target markers, frame slider, pause, and single-step controls remain active.
- Enabled reference gating only for the Oracle end-to-end visual task.  The
  reference advances normally between anchors and freezes when it reaches an
  anchor.  Position, orientation, and all reference velocity targets share the
  same clock; reference joint/body velocities are zero while frozen.
- Strict release requires the four rung-contact bits to exactly equal the
  anchor annotation, every required contact to be near its target rung, no
  non-limb or rail danger contact, bounded root height/orientation error, and
  bounded root linear/angular speed for 4 consecutive 50-Hz control steps.
  Contact bits and target-contact completion share the existing pair-filtered
  hand/ankle rung sensors; the main danger-contact sensor explicitly excludes
  those legal contact bodies (including ankle geometry versus sole target
  points), preventing valid foot support from being classified as danger.
- After 2.0 s the relaxed condition still requires every newly placed limb on
  its correct target and forbids unexpected rung contact, but permits one old
  support to flicker.  At 5.5 s an independent `stable_anchor_timeout`
  termination resets the environment.  The legacy dense-reference
  `body_local_pose` and `required_contact` terminations are disabled only in
  this goal-gated Oracle task so they cannot preempt the new timeout.
- Increased only the Oracle episode horizon to 180 s.  The original reference
  is 61.2 s and the worst-case 21x5.5-s anchor budget is 176.7 s, so the global
  time limit cannot silently override the per-anchor rule.
- Added per-anchor metrics for hold time, strict/relaxed completion, safety,
  contact matching, target contact, and timeout failure.

References checked:

- `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/command_manager.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `Ladder/source/ladder/ladder/utils/motion_semantics.py`

Verification:

- [x] Python compilation succeeds for the command, termination, config, and
  replay modules.
- [x] Focused pure-data/config/actor/PPO tests pass (`45 passed, 1 skipped`).
- [ ] Full IsaacLab reset/step smoke remains pending because the current Codex
  execution environment reports no CUDA device; the attempted smoke reached
  task/runner parsing before failing at CUDA environment allocation.

## 2026-09-21 - Ground-aware stable-support anchors supersede stop keyframes

- Kept the legacy ladder-only `contact_indicator/contact_rung_id` unchanged so
  existing teacher policies, tracking rewards and checkpoints retain their
  original contract. Added a separate stable-support contract: `-1` means no
  support, `0` means ground/base platform, and `1..7` map to legacy ladder
  rungs `0..6`.
- Stable phases now require all four limbs to have support and split whenever
  the four-limb support-ID tuple changes. Runs shorter than three frames are
  discarded. A representative must pass the per-limb support-surface distance
  test and keep every end-effector below 0.10 m/s; the frame minimizing the
  maximum four-limb speed is selected. Stage 37 contains 26 four-limb support
  anchors (13 ascending and 13 descending), plus frame 0, middle standing
  frame 1500, and terminal frame 3060 as explicitly typed stable anchors.
- Ground distance is measured from the hand-sphere bottom and the saved ankle
  collision-mesh minimum, not from the virtual sole marker. The ground-surface
  tolerance is 0.015 m; ladder contacts use the finite-tread release distance.
- The replay viewer, next-stable observations and Oracle reference gate all
  consume the same saved `stable_support_anchor_frames`. Ground targets use
  reference limb positions at the anchor; ladder support ID `k` resolves
  through the randomized ladder geometry as legacy rung `k-1`.
- Added pair-filtered left/right foot ground sensors to the Oracle scene; the
  existing hand ground sensors are reused. The Oracle actual-contact input is
  now ground-or-rung support, and its checkpoint data contract advances to v3.
  The episode horizon is 210 s, above `61.2 + 26 * 5.5 = 204.2 s`.
- Regenerated the Stage-37 artifact from Stage 36. All 187 Stage-36 fields and
  legacy command arrays are preserved; the stable-support contract now also
  stores anchor kinds/full-support flags and a complete reset-segment
  partition. The three special anchors are allowed to have only their actual
  ground/rung supports; they are not incorrectly validated as four-limb
  rung-contact frames.
- The Oracle reset sampler selects a stable-support segment first and then
  sample a geometry-valid frame inside it. Adaptive failure weighting is
  aggregated at the segment level, uniform sampling chooses an eligible
  segment uniformly, and the existing fixed/secondary/adaptive mixing and
  geometry masks remain active. The Oracle config enables this mode; other
  tasks retain the old frame sampler by default.
- Decoupled replay rendering from motion playback. The GUI and keyboard now
  refresh at 60 Hz by default even when stable anchors advance at 1 Hz, and
  robot/marker buffers are rewritten only when the selected frame changes.

Verification:

- [x] Focused annotation/motion/config/gate/actor/PPO tests pass
  (`48 passed, 1 skipped`).
- [x] Changed Python modules compile.
- [ ] Full GPU reset/step smoke remains pending because this execution context
  cannot communicate with the NVIDIA driver.

## 2026-09-21 - Policy-only lightweight visual PPO encoder

- Replaced the end-to-end visual PPO encoder with the requested two-branch
  policy model: `25x85 -> 256 -> 128` proprioception, a
  `2x60x80 -> Conv(16,32,64) -> AdaptivePool -> Linear128` depth branch, and
  a reset-aware `GRU(256,256)` over their concatenated feature.
- Kept the Oracle action network exactly `373 -> 512 -> 256 -> 128 -> 25`.
  PPO gradients still update the proprioceptive branch, depth branch, GRU and
  action network jointly.
- Removed the reconstruction observation groups, decoder heads, reconstruction
  targets and auxiliary losses from this training path. The rollout storage
  does not allocate velocity, height-map or next-frame reconstruction buffers
  for the policy-only algorithm.
- Increased `sequence_micro_batch_envs` from 32 to 64. With 5120 environments,
  four logical minibatches and five epochs, this reduces sequence microbatch
  launches from 800 to 400 per update without changing logical minibatch loss
  normalization or optimizer-step count.
- Advanced the model/training data contract and gave the new runs separate
  `policy_only` experiment names; reconstruction-era checkpoints are rejected
  instead of being partially loaded into the incompatible encoder.

References checked:

- `isaaclab_v2.1.0/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`
- `/home/zhangdawei/humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/modules/actor_critic_recurrent.py`
- `Ladder/source/ladder/ladder/rsl_rl/modules/transformer_vision.py`

Verification:

- [x] Policy, PPO, wrapper, Oracle and legacy TransformerVision regression
  tests pass (`64 passed, 1 skipped`).
- [x] New Oracle policy has 1,420,482 parameters versus 1,810,364 previously;
  the action MLP remains 358,937 parameters.
- [ ] Full IsaacLab reset/step smoke is pending because the existing PID
  661099 is still running the old, already-loaded reconstruction-era model.

## 2026-09-22 - Stable-support-aligned Oracle rewards

- Kept the stable-anchor gate and its three-stage thresholds unchanged:
  confirmation `4` control steps, relaxed completion after `2.0 s`, and
  timeout/reset after `5.5 s`.
- Changed only the Oracle visual PPO contact-position rewards to use the saved
  ground-aware `stable_support_contact_indicator` and
  `stable_support_rung_id` contract. Support ID `0` resolves to the saved
  ground target, IDs `1..N` resolve to randomized ladder rungs, and `-1`
  remains inactive. The retained blind task still uses its previous rewards.
- Added signed potential progress toward the pending stable anchor. It averages
  the valid four-limb target distances, returns `(d_previous-d_current)/dt`,
  uses reward weight `2.0`, and emits zero on reset or target-frame changes so
  reference jumps cannot create false reward. The signed term is intentionally
  not clipped, so moving away cannot be made cheaper than returning by
  exploiting asymmetric saturation.
- Added a one-shot completion bonus with weight `1.0`. The event is the exact
  existing gate `release` tensor rather than a duplicated approximation; the
  reward is divided by control `dt` so IsaacLab's reward-manager integration
  produces exactly one bonus unit per completed stable anchor.
- Added command metrics/properties for the active stable-support contract,
  pending target frame, terminal mean target distance, one-tick completion
  event, and per-episode stable-anchor completion count.

References checked:

- `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/reward_manager.py`
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/humanoid/mdp/rewards.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`

Verification:

- [x] Reward, stable-contact, motion-semantic, reset, actor, Oracle, and PPO
  focused tests pass (`76 passed`).
- [x] Changed Python modules compile.
- [ ] Full IsaacLab config/reset/step smoke remains pending because this
  execution context cannot access a CUDA device; the attempted headless app
  launch failed before configuration import during GPU initialization.

## 2026-09-22 - Rung-aware Oracle support interface

- Replaced the preliminary Oracle actor state with the requested interface:
  current proprioception 85, the recurrent proprioception/depth feature 256,
  next-stable four-limb target positions 12, and three 12-D support relations
  for actual physical support, current-stage expected support, and next-stable
  support. The action MLP architecture is unchanged; its input is now 389-D.
- Kept raw support semantics `-1=none`, `0=ground`, and `1..N=ladder rung`.
  Pair-filtered force matrices identify the actual surface by strongest force;
  filter paths are mapped explicitly to rung IDs rather than assuming column
  order. Desired current/next IDs come from the saved stable-support motion
  annotations.
- Encoded every limb relation as `[has_support, is_ground,
  relative_rung/num_rungs]`, sharing the lowest positive next-stable rung as
  the origin across all three groups. This removes integer scale and sentinel
  ambiguity while preserving who is above or below whom.
- Retained target XYZ for all four limbs at the pending stable frame, including
  limbs that will not be supporting there, and zero-masked the full target only
  when no pending target exists. Advanced the Oracle checkpoint data contract
  to v5; older 32-D
  Oracle checkpoints are rejected instead of partially loading into the 389-D
  policy.
- Did not change the stable-anchor gate, reset segmentation, rewards, PPO,
  visual encoder, GRU, action head layers, or camera timing.

References checked:

- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py`
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/mdp/observations.py`
- `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
- Existing LadderMan stable-support gate, contact-force, and recurrent visual
  actor implementations.

Verification:

- [x] Focused contact, support encoding, actor, wrapper, recurrent replay,
  PPO, reward, motion semantic, and reset tests pass (`92 passed`).
- [x] Changed Python modules compile.
- [ ] Full IsaacLab config/reset/step smoke remains pending because this
  execution context cannot initialize the Isaac Sim/CUDA runtime.

## 2026-09-28 - Event-driven support task and SSR-aligned visual AMP

- Added independent `Visual-ContactTask-AMP-v0` and
  `Visual-ContactSequence-AMP-v0` tasks. The former trains the 28 annotated
  stable-support transitions independently; the latter advances a goal only
  after measured contact, target proximity, low limb/root velocity, upright
  posture, and four consecutive valid control steps. Both use the 53-D
  Oracle task vector, the existing light CNN+proprio+GRU256 policy, and task
  rewards without continuously advancing reference tracking.
- Added SSR-style 5-frame/stride-2 least-squares AMP: the 71-D local state,
  512/256/128 discriminator, gradient penalty, rolling normalizer, independent
  optimizer, bounded style reward, expert stable-hold augmentation, reset-safe
  history, AMP metrics, snapshots, and checkpoint contract.
- Fixed a PhysX GPU ground-contact failure confined to the event tasks:
  `/World/ground` pair-filtered forces are zero despite hundreds of newtons
  of ankle net force. Ground ID 0 now additionally requires net ankle force,
  no rung-filter contact and foot height within 0.12 m of ground. The
  checkpoint data contract is versioned to `ground_net_v2`; pre-fix smoke
  checkpoints cannot be resumed silently.
- Event completion is consumed once. Signed distance progress has no
  per-step clamp, preventing an approach/retreat reward loop. Playback
  disables legacy reference-clock 0/C stop/resume controls for these tasks.
  Runs save source hashes, motion SHA, AMP contract and the complete 28-task
  plan with ground-contact threshold.

References checked: IsaacLab 2.1's
`source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/humanoid_amp_env.py`
and contact sensor implementation; installed IsaacLab 2.3 command/reward
update order; `whole_body_tracking` tracking command pattern; local SSR
`rsl_rl/modules/amp_discriminator.py`, `rsl_rl/algorithms/ppo_ce.py` and
`rsl_rl/utils/utils.py`; existing LadderMan stable-support command and
filtered contact identity code.

Verification: 27 targeted CPU tests pass, including SSR discriminator and
normalizer numerical comparisons and event single-consumption/timeout tests.
Both task variants completed 8-environment, 2-update IsaacLab PPO+AMP smoke
runs with nonzero depth-CNN and GRU gradients. The final `ground_net_v2`
contract also completed a one-update save and actor-only playback. Fixed-frame
playback verified ground ID 0 at frame 0, rung hand IDs 4/5 at frame 373,
and left-foot rung ID 1 alongside right-foot ground ID 0 at frame 456.
These are execution checks, not evidence that a trained policy climbs.

## 2026-09-28 - All-frame contact-task resets and measured rung-ID actor input

- Event-task reset now samples uniformly from frames 0..3059, maps each frame
  through the motion file's saved stable-support segment ID, and keeps that
  frame's joint/root velocities. Frame 3060 is excluded because no next
  stable target exists. In sequence mode, successful advancement starts the
  next task at its new source anchor without advancing a reference clock.
- Task deadlines now use the larger of 5.5 s and 1.25 times the remaining
  reference interval plus 0.5 s. For full-segment starts this gives 7.64 s
  (task 0), 7.08 s (13), 8.94 s (14), and 7.16 s (27); the single-task episode
  limit is 10 s and the worst-case sequence deadline sum is 162.82 s < 170 s.
  Posture failure requires 10 consecutive steps; non-limb/rail collision
  requires 20. Timeout, fall, and collision are logged separately.
- Added a 9-way one-hot measured support identity for each limb (none, ground,
  rung 1..7) while retaining measured contact bits. The task vector is 89-D,
  actor action-MLP input 430-D, critic input 160-D; a new checkpoint data
  contract prevents loading 53-D task checkpoints as if they were compatible.
- Confirmed against local SSR: its AMP reward coefficient is fixed at 1.0,
  with no linear reward-weight warmup. LadderMan's 0-to-0.2/500-update ramp
  remains an intentional early-discriminator precaution, not SSR parity.

References checked: installed IsaacLab 2.3 command resampling/reset order;
existing Ladder motion loader's saved reset-segment IDs and velocity reset;
local SSR `x2_vis_sparse_mamp_config.py` and `legged_robot_vis_sparse_mamp.py`.

Verification: 31 focused CPU tests pass; both single-task and sequence-task
8-environment, one-update IsaacLab PPO+AMP smoke runs exited successfully.
The single-task run recorded actor input 430 and non-anchor reset frames.
Short random-policy smoke runs still produce early falls and no successful
climb; they validate integration only, not learnability.

## 2026-09-28 - Raw support-ID command instead of one-hot contact features

- Replaced the event-task actor's binary measured-contact and 9-way one-hot
  fields with three raw four-limb IDs: physically measured support,
  `stable_support_rung_id` at the sampled reset frame (current expectation),
  and `stable_support_rung_id` at the next stable anchor. All use
  `-1=none, 0=ground, 1..7=rung`. The sampled-frame expectation stays fixed
  until a successful task transition; the reference clock remains disabled.
- Kept all four target-frame limb XYZ positions even when a limb's next
  support ID is `-1`. The task vector is 57-D, actor action-MLP input 398-D,
  critic input 128-D. Checkpoint data contract v4 rejects the one-hot v3
  checkpoint rather than silently interpreting its features differently.
- Checked the motion NPZ: `stable_support_rung_id` contains the requested
  ground-aware numbering on every frame. Its `contact_rung_id` field uses
  different numbering and cannot stand in for this annotation.

Verification: 31 focused CPU tests pass, including exact three-vector ID
layout and non-contact-limb target XYZ; an 8-environment one-update IsaacLab
PPO+AMP smoke run exited successfully and recorded actor input 398. No long
training was started; learnability remains unproven.

## 2026-09-29 - Continuous support progression and AMP training audit

- Changed the default `CR1ContactTaskAMPEnvCfg` and `ContactTaskCommandCfg`
  to continuous episodes. Random motion-frame resets remain; intermediate
  successes update the next goal in place and preserve robot/GRU state.
  Final-goal success or failure ends the episode. The horizon is 170 s;
  per-goal deadlines, reward weights, actor layout and AMP losses are unchanged.
  `CR1ContactSequenceAMPEnvCfg` is an explicit alias of this behavior. An
  intentional old single-task control requires overrides for both
  `single_task_episode=true` and `episode_length_s=10`.
- Added start-task, remaining-goal, completion-fraction and sequence-success
  metrics. Preserved terminal metrics before IsaacLab auto-reset for playback
  episode JSON. Kept per-task attempts/successes/failures and completed count.
- Fixed playback termination logging for env IDs above zero: IsaacLab's
  `get_active_iterable_terms(env_idx)` already selects the environment and
  returns a singleton value, so it must be read at index zero.
- Audited task/AMP numerical units and local SSR. Discriminator formula and
  rolling-normalizer parity do not cover SSR's separate per-reward-group
  value/GAE/advantage normalization. LadderMan currently uses a scalar
  reward sum with a single critic. At update 7279 the AMP weighted per-step
  reward was 0.0001735 versus task reward 0.2947, with near-perfect
  discrimination and only 52.85% valid AMP windows. Saved sample features
  were finite with negligible clipping; window-to-next-critic-state error
  was exactly zero. Recommendations and limitations are in the learning log.

References checked: IsaacLab `manager_based_rl_env.py`,
`managers/command_manager.py`, installed `managers/termination_manager.py`,
official `humanoid_amp_env.py`, local `whole_body_tracking/.../mdp/commands.py`,
LadderMan contact/goal/reset code, SSR `amp_discriminator.py`,
`algorithm_plugin.py`, `ppo.py`, `ppo_ce.py`, `storage/rollout_storage.py`,
`legged_robot.py`, and `x2_vis_sparse_mamp_config.py`.

Verification: 34 focused CPU tests pass, including consecutive goal completion,
no robot-state reset on advancement, final-goal termination, suffix-relative
completion fraction and multi-env termination logging. An 8-env/two-update
PPO+AMP smoke run completed under
`logs/rsl_rl/cr1_event_contact_task_visual_amp/2026-09-29_10-16-09_smoke_continuous_contact_audit/`
with `single_task_episode=false` and horizon 170 s. Subsequent playback checks
exercise the latest terminal-metric changes:

- Old model_7000, 8 random-start envs, 750 steps: 98 completed episodes;
  transition counts {1: 92, 2: 6}; 13 suffix successes (12 start task 27,
  one starts task 26), 66 posture failures, 19 collision failures, no deadline
  failures. No episode completes more than two transitions.
- Old model_7000, 4 frame-0 envs, 400 steps: all 12 finished episodes complete
  only task 0 and fail in task 1 (4 posture, 8 collision); no sequence success.

Evaluation artifacts are in `outputs/contact_sequence_audit_20260929_random_verified/`
and `outputs/contact_sequence_audit_20260929_frame0/`. Evaluation is deterministic
and nominal, so these are bounded diagnostic results, not training success
estimates or proof of learnability. No long training was launched.


## 2026-09-29 - SSR-style grouped AMP advantages and support-release reward

- AMP PPO now stores task and style rewards in two channels, emits two critic values, computes GAE separately for each channel, normalizes each channel over the complete rollout, and combines the normalized actor advantages as `1.0 * A_task + 0.2 * A_amp`. AMP style reward is no longer added to the task scalar before GAE; the configured warmup is disabled (`0`) for SSR parity.
- Rollout storage validates multi-channel reward shapes and keeps raw task rewards as a separate diagnostic channel. The training contract records `value_channels=2`, so old single-value AMP checkpoints cannot be resumed as if they had the new semantics.
- Added `support_match` (weight `+0.5`) using exact measured-vs-next-stable support IDs. IDs are `-1` no contact, `0` ground, and `1..7` ladder rungs; therefore a `-1` target is positively rewarded only when that limb is actually released.

References checked: local SSR PPO storage/algorithm integration and LadderMan contact-task command/reward code.
Verification: 35 focused CPU tests pass. The real IsaacLab 8-env smoke run could not start because the installed IsaacLab/Isaac Sim pair lacks `omni.physx.bindings._physx.SETTING_BACKWARD_COMPATIBILITY`; this is an runtime installation mismatch, not a task/PPO assertion.
## 2026-09-29 - Time-sampled visual/proprioception history

- End-to-end visual PPO now uses a fixed-capacity per-environment device ring buffer for camera, proprioception, and command samples. Camera exposures follow the configured 30 Hz clock; repeated policy calls reuse the same frame and a full buffer overwrites the oldest sample.
- The end-to-end contract is `10x85` proprioception frames sampled every `0.1 s` and `3x60x80` depth frames sampled at approximately `t-0.4 s`, `t-0.2 s`, and `t`, returned oldest-to-newest. The camera history is selected by timestamps, not by the last three policy calls.
- Reset clears the selected environment's three rings, timestamps, cursors, and recurrent history; partial reset cannot leak a previous episode into either actor input or command history.
- Actor/CNN/storage/playback paths infer the configured history lengths. The checkpoint contract is `cr1_end_to_end_visual_policy_only_proprio10x85_depth3x60x80_time_sampled_30hz_cnn16_32_64_prop128_depth128_gru256_v3`; old 25-frame/2-depth-frame checkpoints are intentionally incompatible.

References checked: IsaacLab observation-manager stateful term lifecycle, existing `warp_visual_depth` camera clock, `transformer_vision_ppo_wrapper.py`, end-to-end actor/storage/PPO modules, and the CR1 end-to-end environment/agent configs.

Verification: 12 depth timing tests (including the explicit 10 Hz/30 Hz contract), 55 visual/PPO/AMP tests, 31 configuration/storage/runner tests, and static compilation checks pass. A full IsaacLab environment launch remains dependent on the installed Isaac Sim PhysX bindings.

## 2026-09-30 - Command-ring reset fix and verified continuous AMP startup

- Corrected the preceding reset claim: command samples were cleared, but their
  timestamps, write cursor, and count were not. Cleared samples could therefore
  be selected as valid zero commands for up to the 0.9 s history span. Reset now
  clears all command metadata as well as the data. A same-step partial reset
  preserves the global sampling clock so untouched environments are not written
  a second time.
- Added regression coverage for both direction signs, all/partial resets,
  same-step/next-step observation calls, and wrapped rings. The new checks failed
  before the fix and now pass. AMP rollout/update/checkpoint tests cover the new
  10x85/3-depth contract in addition to the legacy shapes.
- The intended new experiment is explicitly
  `Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-AMP-v0`, not the basic
  `WarpTransformerVision-EndToEnd-Reconstruction-v0` task. Its saved configuration
  confirms `advance_reference=false`, `single_task_episode=false`, actor input
  398, two value channels, independently normalized task/style advantages with
  weights 1.0/0.2, and no AMP warmup. Intermediate successes advance the goal
  without resetting robot/GRU state. The actor still uses oracle contact/task
  fields; this is not yet a fully deployable visual policy.
- Startup diagnostics now describe the actual grouped-advantage AMP behavior.
  AMP manifests record history parameters, the training contract, reference
  advancement and episode mode; basic visual manifests no longer hardcode 25
  proprioception frames. No reward, gate or learning-rate settings changed here.

References checked: IsaacLab `managers/observation_manager.py`,
`utils/buffers/circular_buffer.py`, official `direct/humanoid_amp/humanoid_amp_env.py`,
`whole_body_tracking/.../mdp/commands.py`, and LadderMan history/contact/AMP code.

Verification: static compilation and 71 focused tests pass. Activating
`rl-training` normally (without the old forced IsaacLab PYTHONPATH) also allowed
an actual 8-env, two-update PPO+AMP smoke run to exit successfully:
`logs/rsl_rl/cr1_event_contact_sequence_visual_amp/2026-09-30_10-34-50_smoke_contact_sequence_resetfix_0930/`.
Both updates have finite logged losses, four discriminator updates and nonzero
PPO CNN/GRU gradients; checkpoints and source hashes were saved. Thus the earlier
PhysX startup failure was not reproduced in this launch environment. The run had
nonfatal file-watch warnings; the filesystem had 521 GiB available. This verifies
execution, not policy quality or 5120-env throughput. No long training launched.

## 2026-10-08 - Event-driven support phases for the contact-sequence task

- Added `SupportContactPhases` with four per-limb states: `prepare` (keep the
  source support), `release` (detach the old support), `move` (flight), and
  `acquire` (establish and maintain the target support). The expected contact
  ID is now derived from this state, not from a reference clock: source ID in
  `prepare`, `-1` during release/move, and target ID during acquire.
- Reset seeds the phase from the annotation at the sampled motion frame. A
  reset already in flight or already on the target does not receive a
  retroactive release/contact event. Release and contact milestones require
  consecutive confirmation and are paid once; established supports remain
  protected while brief sensor dropouts are tolerated.
- The phased reward set removes the old continuous reach/contact/support-match
  rewards and uses signed target-distance progress, release/contact events,
  phase contact error, support-loss, slip, wrong-contact, danger, completion,
  and failure terms. Intermediate task completion advances the next support
  goal in place; robot state and recurrent state are not reset. Final success
  remains reserved for the terminal goal.
- Registered `Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-AMP-v0`.
  The phased task observation is 61D (legacy contact task remains 57D), making
  the actor input 402D = proprio85 + GRU256 + task61. Old 398D checkpoints are
  intentionally incompatible. The PPO wrapper validates both contracts and
  rejects any other contact-task width.
- Added Isaac-free regression tests for phase monotonicity, one-shot events,
  reset milestone semantics, and post-acquisition support protection.

References checked: LadderMan `utils/contact_task.py`, contact-task MDP/config,
end-to-end visual actor/PPO wrapper, the motion NPZ support annotations, and
the existing SSR-style grouped-advantage AMP implementation.

Verification: modified files pass `compileall`; the new four-test CPU phase
suite passes. The full AMP test module cannot be collected in the plain shell
because it requires IsaacLab/Omniverse `pxr` bindings; no claim of a full
IsaacLab environment smoke run is made here. No long training was started.

## 2026-10-08 - Body-goal shaping and contact-phase stall audit

- Registered the separate experiment
  `Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-AMP-v0`.
  It adds the next stable frame's `body` anchor position as a fixed segment
  goal. This is a torso-link position, not a mass-weighted whole-body center
  of mass. The goal includes the environment origin translation and switches
  only on reset or segment completion; the reference clock remains disabled.
  Non-nominal ladder geometry is rejected for this goal implementation.
- Added a low-weight signed body-progress reward. With distance `d` in meters,
  define `e = max(d - 0.10, 0)`; the per-step reward contribution is
  `0.5 * (e_previous - e_current)`. The MDP term divides by control dt because
  IsaacLab's reward manager multiplies by dt. The weight is constant across
  phases, avoiding an artificial potential jump at a phase change. Standing
  still pays nothing, retreat is negative, and approach inside the 10 cm
  deadband pays nothing. Reset/goal-switch baselines cannot generate a jump
  reward. No body-position completion gate was added.
- Added the 3-D body goal error in the current torso frame to the actor's
  task input: task 64D, actor `85 + 256 + 64 = 405D`, critic `64 + 71 = 135D`.
  Network, wrapper, training manifest, playback and checkpoint contracts all
  recognize the new width. Old 57D/398D and phased 61D/402D contracts remain
  distinct; their checkpoints cannot be resumed directly in the 405D task.
- Fixed three shared phased-task stall risks: sustained early detachment in
  PREPARE no longer requires recontacting the obsolete source; physical
  detachment can advance RELEASE even when safety conditions disallow a bonus;
  another active flying/unconfirmed landing limb no longer creates a circular
  wait for PREPARE. Required established supports still constrain safe release.
  Sensor flicker tolerance, consecutive safe-event confirmation and one-shot
  milestone budgets remain. Unsafe/early releases do not receive delayed
  bonuses. These fixes also affect the existing phased task; only the body
  goal/reward/input addition is isolated to the new experiment.
- Added body target distance, cumulative signed progress, mean body error at
  completed goals, and phase residence-time diagnostics. Completion error is
  an event mean, not diluted by non-completion steps; read it together with
  `task_completed_count`. Phase residence times accumulate active-limb-average
  occupancy. Terminal copies preserve these diagnostics through auto-reset.
- Preserved the existing safety/completion thresholds and shared segment
  deadline: `max(5.5 s, 1.25 * remaining reference duration + 0.5 s)`.
  Phase changes do not renew this deadline. Intermediate goal completion
  preserves the robot/GRU state; only final success ends the sequence. AMP
  still has two value/GAE channels and separately normalized advantages with
  task/style weights 1.0/0.2. Learning rate and visual hidden architecture
  were not changed.

References checked: IsaacLab `managers/reward_manager.py` (weight/dt),
`envs/manager_based_rl_env.py` and `managers/command_manager.py` (step/reset
ordering), the official `direct/humanoid_amp/humanoid_amp_env.py`,
`whole_body_tracking/.../mdp/rewards.py`, and LadderMan motion-anchor,
contact-task and grouped-advantage AMP code. No runtime dependency on
`whole_body_tracking` was added.

Verification: static compilation passed. The expanded CPU regression run has
**149 passed, 1 skipped** across contact-task/phase, actor/oracle/wrapper/PPO,
contact annotation/stabilization, camera timing and recurrent storage tests.
Using the repository's IsaacLab 2.1 source resolves the preceding plain-shell
CPU collection limitation. Tests cover reward sign/dt/deadband, fixed goals,
selected-environment reset, body-frame invariance, phase stalls and one-shot
events, finite PPO/AMP updates and checkpoint contracts. A scripted ideal
contact sequence visits all 29 anchors/28 transitions and ends only at the
last goal; this is not a physical rollout or proof of learnability.

The skipped module requires a full Isaac Sim config runtime. In the current
execution environment there are no NVIDIA devices and CUDA is unavailable,
so a real 8-env reset/step/short-training smoke test remains outstanding.
No long training was launched. Next physical validation should check phase
occupancy/timeouts, per-segment success, support losses and body completion
error before a large run. Contact-only completion can still advance a
release-only segment before the torso reaches its goal; this experiment
deliberately measures that risk instead of adding an unverified hard gate.

## 2026-10-10 - Foot-only wrong-surface and swing-pose penalties

User playback identified excessive toe-up during foot lifting and obstruction at
the next rung (particularly tasks 06 and 10). The BodyGoal reward set did not
include the older front/underside contact penalties: rung identity alone cannot
distinguish stepping on a rung from hitting its front or underside.

- Added the isolated task
  `Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-AMP-v0`
  and experiment `cr1_phased_body_goal_foot_safety_contact_sequence_visual_amp`.
  It inherits the 405-D BodyGoal actor and adds three negative reward rates.
  Hand behavior, raw contact IDs, goal progression, completion/termination,
  PPO/AMP configuration and learning rate remain unchanged. The actor/checkpoint
  contract remains v7, allowing an existing BodyGoal checkpoint as a starting
  point. The original task remains available for baseline playback.
- Each foot receives `-3/s` for front/side contact and `-3/s` for underside
  contact. Contacts must exceed 5 N; each category is a bounded binary per-foot
  value, independent of how many rungs are touched or how large the force gets.
  Source and target rungs have no exemption. All ladder steps are included;
  hands and ground are excluded. One foot hitting one wrong surface for 0.1 s
  costs 0.3 reward units. Terms return rates and IsaacLab multiplies by dt.
  Terminal collision steps are still penalized; reset states are masked.
- Reused `filtered_surface_force_max`: the per-foot/per-rung normal-force
  direction separates upward, horizontal and downward forces with normal-Z
  fraction 0.5. Horizontal includes front/back/side contacts. This is an
  approximate face classifier when multiple faces contact the same foot/rung
  pair; it does not provide an exact contact-point label.
- Added swing-pose shaping at weight `-1/s` per foot, capped at one unit of
  penalty per foot before applying weight/dt. CR1 ankle-link +X points toward
  the toe, +Y laterally and +Z above the sole. The world-up vector in the foot
  frame gives signed toe elevation and lateral edge tilt, independent of yaw.
  Free toe pitch range is **-55 to +15 degrees** (positive means toe up);
  lateral tilt tolerance is **20 degrees**. Angular excess is divided by
  30 degrees and squared; upside-down feet are additionally penalized before
  the per-foot cap. A 30-degree toe-up angle alone costs 0.25/s.
- The pose term applies to active feet during RELEASE/MOVE/unconfirmed ACQUIRE,
  including contact before detachment, and to physically airborne active feet
  even in PREPARE. It uses the issued phase before the next-phase update.
  Established support is excluded unless the foot loses contact again.
  Inactive support feet and all hands are excluded from the pose term.
- Checked the motion and mesh before choosing limits. The foot meshes extend
  further along +X, and the initial standing ankle frame is level. Annotated
  swing toe pitch is approximately -37 to -6 degrees for the left foot and
  -46 to -3 degrees for the right foot. Maximum absolute lateral swing tilt is
  about 14 degrees. A symmetric flat-foot penalty would oppose these successful
  reference poses; the selected asymmetric band leaves them unpenalized.
- Logged per-foot front/underside contact exposure, peak force, swing duration,
  peak positive toe elevation/lateral tilt, toe-up excess duration and integrated
  pose error. These cumulative/peak metrics survive auto-reset in terminal
  episode records. Playback prints and serializes current foot forces/angles
  and the pose mask. Training manifests record axes, thresholds, units, weights
  and scope; source snapshots now include `utils/contact_force.py`.

References checked: IsaacLab `managers/reward_manager.py` (dt),
`sensors/contact_sensor/contact_sensor_data.py` (filtered normal forces),
`envs/mdp/rewards.py` (orientation/contact shaping), Spot
`manager_based/locomotion/velocity/config/spot/mdp/rewards.py` (swing clearance),
`whole_body_tracking/.../mdp/rewards.py` (body orientation), and LadderMan
`mdp/rewards.py::step_surface_orientation_contact_penalty`, contact-task code,
the actual CR1 URDF/STL and motion quaternion arrays. No reference-project
runtime dependency was added.

Verification: **104 CPU tests passed, 1 Isaac Sim config module skipped** in
two disjoint suites; static compilation passed. Tests exercise actual command
methods with synthetic sensors, including hand isolation, target-rung contact,
top/noise rejection, both feet, bounded force response, reward sign/dt, resetting
environment masks, asymmetric pitch, yaw invariance, inversion, all relevant
phases, failed-step penalties, refresh caching, unchanged baseline completion
and reference swing-pose compatibility. Actor/wrapper/oracle regression tests
also pass. NVIDIA driver access is unavailable in this execution environment;
real config import/reset/step and a short GPU training run remain unverified.
No training was launched.

This is the requested penalty-only experiment: it does not yet make valid
top-face support an additional completion condition, nor add swing waypoints.
Negative rewards discourage wrong contact but cannot guarantee its absence.
Check task 06/10 success and timeout rates jointly with wrong-face exposure
and swing toe-up excess before increasing penalty weights further.

## 2026-10-12 - Offline AMP score check for successful versus failed swing behavior

The end-of-run AMP discriminator from `foot_safety_scratch_env5120_iter10000_seed42`
was evaluated offline with the same 71-D feature map, five-frame/stride-two
window, running normalizer, and clipped LSGAN reward used during training.
This was a read-only check; no code or checkpoint was changed.

- Kinematic motion-reference windows scored `0.844` mean style reward
  (`0.825` median).
- A previously recorded deterministic visual-teacher rollout, with 8/8
  successful episodes in its own teacher-only evaluation, scored `0.935` mean
  (`0.959` median). On the two difficult upward transitions, its mean scores
  were `0.930` for task 06 and `0.930` for task 10.
- The latest FootSafety policy diagnostic snapshot scored only `0.123` mean
  over valid windows (`0.024` median). The task-06 sample averaged `0.300` and
  the task-10 sample averaged `0.109` in PREPARE and `0.016` in RELEASE.

This supports the limited claim that the current AMP discriminator assigns
higher scores to the known reference/successful-teacher motion than to the
observed student motion. It does not yet prove that AMP ranks two student
actions correctly, because the snapshot contains no complete success label for
each transition. The teacher trace also comes from the older teacher-only
visual task, so it is a physically feasible-motion reference rather than an
exact FootSafety-task ablation. The next decisive test is to score paired
successful and failed student/teacher transitions from the same task and
stage; increasing the AMP weight should wait for that comparison.

## 2026-10-12 - Phase-bounded whole-body reference reward experiment

Implemented the independent task
`Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-Reference-AMP-v0`
with experiment name `cr1_reference_guided_contact_sequence_visual_amp`.
The old FootSafety task is retained. Actor405, visual/GRU architecture, AMP,
task/AMP two-value PPO, physical contacts, support completion and safety
terminations are unchanged. No weight curriculum or third value was added.

References inspected: IsaacLab `managers/reward_manager.py` and
`envs/manager_based_rl_env.py` for dt/order, the official reach-task reward
utilities, `whole_body_tracking/tasks/tracking/mdp/rewards.py`, and the local
ZDW0824 SmoothOutput teacher reward config/functions. There is no new runtime
dependency on whole_body_tracking.

Implementation:

- `utils/reference_guidance.py` builds phase-compatible frame ranges once
  from stable-support IDs. Startup checks all per-limb sequences are constant
  or source -> no-contact -> target (including free-hand/release-only cases).
  The 28 current segments satisfy this contract.
- A reward-only clock starts at the random reset frame, advances by
  `motion.fps * step_dt`, and is bounded by the intersection of all limbs'
  phase ranges. It never changes `time_steps`, support IDs or task completion.
  PREPARE ends before release; RELEASE allows preparation and at most five
  frames of post-release anticipation; MOVE reaches the reference landing;
  ACQUIRE allows five frames of landing anticipation. Established contact
  excludes earlier flight frames. RELEASE's preparation range is necessary
  for the reference's sequential two-hand releases.
- Physical phase progress can align the clock forward past obsolete poses;
  it never rewinds. An empty shared range gives zero imitation reward, logs
  a conflict, and leaves the physical task active until its normal deadline.
  Matching within +/-5 frames selects ONE weighted-best whole-body frame for
  all six terms. The winning frame does not advance the clock.
- Six rate weights: lower/upper body position .15/.15, lower/upper joint pose
  .25/.15, swing ankle position .30, body-anchor orientation .10. Teacher
  stds/temperatures are .18/.25 m, .8/1.0 summed squared radians, .02 m^2,
  and .35 rad respectively. Body position alignment preserves reference Z
  and follows robot XY/yaw. Both sides of foot comparisons are ankle links,
  not a mixture of ankle and sole markers. Joint groups resolve by names
  against the motion loader's runtime-reordered joint data.
- Rewards evaluate the issued reference before phase advancement; command
  update prepares the next reference after rewards. Cached refresh and an
  advance-step guard prevent duplicate updates. Reset and goal changes clear
  guidance state without resetting other environments or moving the robot.
- The new variant alone uses `max(10 s, 2 * remaining reference duration + 1 s)`
  per goal, with no subphase deadline resets, and a 320 s episode limit.
  Full-sequence budgets sum to about 290.04 s (before control-step rounding).

Logging includes per-term rewards/errors, valid-match state, reference/matched
frames, cumulative conflict/hold time, phase skips and weighted imitation
return. Error metrics are zero when invalid; interpret them with validity.
The run manifest stores phase ranges, weights, tolerances, alignment, deadlines
and source snapshots. Playback logs reference timing, pause reasons and ankle
targets; `--show_reference_guidance` optionally draws body/ankle point markers.
Detailed point rendering is off by default; training uses batched tensor
operations and retains only one selected pose per environment.

Validation: 85 CPU tests passed; the Isaac Sim config-import test module was
skipped because its runtime is unavailable. Coverage includes every one of
the 3060 nonterminal reset frames, all 28 contact sequences, sequential hand
release, conflicting phase ranges, shared-window selection, partial reset,
goal handoff, 500-step timeout, dt scaling, name-based joint groups, translation
alignment, existing contact/foot regressions and PPO+AMP update regressions.
Python compilation passed. These are tensor/synthetic-sensor checks, not
physical rollout success. No CUDA device or /dev/nvidia device is available
in this execution environment, so actual config/reset/step and short GPU
training remain unverified. No training was launched.

The new clock is not an actor observation. This controlled experiment adds
soft supervision, not a demonstrated solution to hidden reference intent.
The six positive terms total at most 1.10 reward units/s; increased holding
return without improved task06/task10 success is a failure mode to monitor.

## 2026-10-12 - Three-frame matching window and real GPU smoke test

Reduced the matching radius from +/-5 to +/-3 frames (seven candidates,
+/-0.06 s at 50 Hz). The separate release/landing anticipation remains five
frames; reward weights, deadlines and network/value architecture are unchanged.
Updated the config default, utility default and tests. CPU regression result:
86 passed, including a new assertion that narrowing the matching window does
not change anticipation.

The previous GPU limitation was the restricted execution context: an approved
outside-sandbox check found an NVIDIA RTX5880-Ada-48Q with sufficient free
memory. A real headless run then completed successfully, exit code 0:

- Task: `Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-Reference-AMP-v0`.
- Run: `logs/rsl_rl/cr1_reference_guided_contact_sequence_visual_amp/2026-10-12_14-58-46_reference_w3_smoke_env32_iter10_seed42`.
- 32 environments, 10 PPO iterations, 24 steps/env/iteration, seed42,
  randomization disabled for startup validation; 7680 environment steps total.
- Checkpoints `model_0.pt`, `model_5.pt`, `model_9.pt` saved.
- Saved config confirms window3/lead5, 10 s minimum goal deadline,
  2x remaining duration + 1 s, 320 s episode limit, actor405 and two values.
- All ten diagnostic records have finite numeric values. Final PPO KL .01430,
  depth-encoder gradient norm .01429 and GRU gradient norm .04377; all six
  reference terms appear in the reward log. Reward/reset and PPO/AMP update
  paths executed without a training traceback.
- Iterations took 4.54 s initially and roughly 3.1 s after warmup. These
  small-batch timings do not predict 5120-environment throughput.

Console log: `/tmp/ladder-reference-w3-smoke-20261012.log`; durable numeric
records and configuration/source snapshots are in the run directory. Isaac
reported nonfatal change-watch ENOSPC warnings, while filesystem checks showed
516 GB free; this was not evidence of a full training disk. No OS watch limits
were changed. Expected unsupported ground contact-filter warnings also occurred.

This completes startup/config, physical stepping/reset and short-training
verification, not policy-quality evaluation or proof of all real transitions.
No long training is left running. Instantaneous `guidance_valid`/error metrics
at terminal states are masked; do not interpret those episode-end averages as
overall imitation coverage or tracking accuracy.

## 2026-10-13 - Release-stall diagnostic (no training changes)

Ran the Reference task with 20 distinct fixed reset frames and three frozen
actors: reference-guided model_5500, previous FootSafety model_5500 and
model_9999. Each ran 600 control steps, nominal/no-randomization, deterministic
actions, the SAME new environment/rewards/10 s deadlines. These are targeted
fixtures, not independent success-rate trials. No optimizer updates occurred.
The training process was already absent before the diagnostic; it was not killed.

References checked: IsaacLab `manager_based_rl_env.py` (reward before auto-reset),
`reward_manager.py` (weight/dt), locomotion contact reward examples,
`whole_body_tracking/scripts/rsl_rl/play.py`, and LadderMan's normal visual play,
contact-force/phase and reference-guidance paths. Runtime setup reused LadderMan
play helpers; whole_body_tracking is not imported. Instrumentation exists only
in `outputs/release_check_20261013/`, not production environment code.

- 54 phase/reference/foot-safety unit tests passed. All 36,000 simulated env
  steps completed; reward component sums matched totals within 1.91e-6.
  No sampled case stayed in RELEASE with four confirmed no-contact readings.
- New actor from frames 373 (left foot 0->1), 686 (1->2), 2001 (3->2),
  and 996 (2->3) held the source contact until the 10 s timeout. Mean force
  during held-source RELEASE was about 372, 358, 291 and 365 N, versus 5 N
  contact threshold; release counters stayed zero in the late dwell interval.
- Previous model_5500 completed the same initial tasks at frames 373/2001 in
  .42/.48 s; model_9999 in .38/.42 s. Thus those failures are not a universal
  impossibility imposed by the new gate. Completion is not proof of good style.
- New actor initialized airborne at frames 397/408 and 2049/2060 completed
  the initial task, while frame 721 still failed. Reset provides favorable
  pose/velocity; this does not prove an entire swing or obstacle bypass learned.
- Dwell reference clocks were 397/721/2049/1034; matched frames remained
  394/718/2046/1031 (the earliest allowed matching frame). Reference rates during
  held-source RELEASE were .875/.779/.810/.536 reward units/s. Net task rates
  were +.673/+.566/+.575/+.309, excluding the terminal event.
- For the 10 s failed first goals at 373/686/2001, reference sums were
  +8.632/+7.736/+8.041; all other task terms INCLUDING failure totaled
  -4.182/-4.261/-4.434. Net returns remained +4.451/+3.475/+3.607. This
  directly establishes a positive dwell incentive in nominal deterministic
  playback; stochastic training samples can have negative net rewards.
- Correct old-model completions still earned more: at frame 373, old5500
  received +6.334 (discounted +5.234), versus stalled new5500 +4.451
  (discounted +1.101). Do not claim stalling is globally optimal or that a
  causal training ablation has been performed.
- Kinematic counterfactual only: freezing reference at release-1 while scoring
  against the saturated RELEASE window yielded 1.01-1.08 out of max 1.10/s
  across the five inspected foot tasks. This tests reward discrimination, not
  physical feasibility of the frozen reference pose.
- Ground geometry is at z=-.03, NOT zero. Source-support height/XY bounds for
  held feet at 373/686/2001 were consistent with contact (about 1 mm from top)
  plus hundreds of N force. Bounds are not exact signed mesh penetration.
  Historical reset-valid annotation marks frames 996 and 2043 invalid; the
  decisive 373/686/2001 cases are marked valid. All-frame reset still includes
  annotated invalid frames; this secondary risk was not changed.

Evidence: `outputs/release_check_20261013/{metadata.json,reference.pt,new5500.pt,
old5500_same_env.pt,old9999_same_env.pt,summary.json,fixtures.csv,geometry.json}`.
Scripts alongside the evidence reproduce capture and analysis. Primary next
ablation: bound/decay reference pose payout during unresolved release stalls,
then assess phase-reference target informativeness. AMP, penalties, deadlines,
actor inputs, and training checkpoints remain unchanged; no new training started.

## 2026-10-13 - Reference pause-reward decay ablation

Added the independent `...-FootSafety-Reference-PauseDecay-AMP-v0` task and
`cr1_pause_decay_reference_contact_sequence_visual_amp` experiment. The original
Reference task retains its previous rewards for comparison and old playback.

- A monotone reference-clock stall keeps full shaping for 0.2 s, then fades
  linearly over 0.5 s; at 0.7 s all six reference terms are exactly zero.
  This is pause-conditioned payout, not per-step clipping or a training curriculum.
- Scaling happens AFTER the shared weighted-best frame is selected. The +/-3
  frame window, six kernels and weights, physical error metrics, actor405,
  critic135/two value channels, PPO, AMP and learning-rate settings are unchanged.
- Only actual clock advance or goal/reset renews the grace period. Contact
  flicker, matched-frame changes, and unsafe->safe transitions cannot renew it.
  Unsafe stalls continue aging. Per-env counters reset independently. Aging
  is prepared once after rewards/phase updates for the next physics step.
- All existing non-reference task terms, release/acquire gates, 10 s minimum
  goal budget, duration-scaled deadlines and 320 s episode length are unchanged.
  No forced advancement and no early failure merely because shaping faded.
- TensorBoard adds `guidance_pause_age_s`, `guidance_reward_scale`,
  `guidance_raw_weighted_reward_integral`, `guidance_suppressed_reward_time_s`.
  Existing `guidance_weighted_reward_integral` now records applied payout.
  Play records raw/applied rate and multiplier; `contact_task_plan.yaml` records
  all decay parameters and renewal semantics.

References checked: IsaacLab `managers/reward_manager.py` (rates multiplied by
dt once), `envs/manager_based_rl_env.py` (reward/reset/command ordering), official
`manager_based/locomotion/velocity/velocity_env_cfg.py` reward conventions;
`whole_body_tracking/tasks/tracking/mdp/rewards.py` and LadderMan's
`ladder_climb/mdp/rewards.py` for the existing teacher kernels. No new runtime
coupling to the reference project.

Verification:

- 102 tests passed across reference guidance, contact phases, foot safety and
  task/AMP tests. Standalone config module skips without IsaacSim; the new task
  was instantiated in the real simulation smoke below. Added config parity and
  registration regression checks for execution inside IsaacSim.
- Real GPU smoke: 32 envs x 10 iterations, exit 0; all JSON diagnostic scalars
  finite, vision/GRU PPO gradients nonzero, saved model_0/model_5/model_9.
  Run: `logs/rsl_rl/cr1_pause_decay_reference_contact_sequence_visual_amp/2026-10-13_10-40-49_pause_decay_smoke_env32_iter10_seed42/`.
- Fixed frame373 playback with the OLD reference model_5500: while left foot
  stays in RELEASE on ground, clock397/match394 stay unchanged. At pause age
  0.44 s, multiplier .52; from age .94 s onwards sampled applied reference rate
  is zero despite raw rate around .87-.89/s. This verifies removal of dwell
  payout, NOT acquisition of a better policy. No old checkpoint was modified.
  At step500 it still fails at the original 10 s deadline, not earlier; the
  process finishes 520 steps, exit0. Evidence:
  `outputs/reference_pause_decay_20261013/fixed373/model_5500/{events.jsonl,summary.json}`.

Full training has NOT been launched. Recommended next comparison is scratch,
5120 envs, 10000 iterations, seed42, save every500. Evaluate stage transitions
and failures as well as returns: removing the positive dwell incentive does not
by itself provide an obstacle-avoiding swing trajectory.

### 2026-10-13: model3500 ascent failure diagnostic (no training changes)

Tested the running PauseDecay run's model_3500 with 12 nominal deterministic
fixtures for 600 steps: six ascent anchors and six mid-swing controls. Captured
physics/contact/phase/reward data before auto-reset and goal handoff. Reused
the previous release audit, following IsaacLab ManagerBasedRLEnv/RewardManager
ordering and the local playback setup; no production logic or weights changed.
Evidence and reproducible scripts: `outputs/ascent_check_20261013_model3500/`.

- Left foot 1->2 (frame686): RELEASE 0.24-10 s, old support force mean351.9 N,
  no airborne reading after settling, millimetric final-second foot motion;
  timeout. Left2->3/frame996 similarly retains347.6 N and times out, but that
  reset has a5.7 mm penetration annotation and is only corroborating evidence.
- Left0->1/frame373 physically releases at0.38 s, enters MOVE at0.44 s, then
  falls at1.80 s; it is no longer the old10 s ground-release stall.
- Right0->1,1->2,2->3 anchors fall in0.40/0.46/0.42 s. The latter two have
  underside-force peaks38.6/162.3 N, but tilt-failure conditions begin BEFORE
  those contacts. Do not label underside collision the proven cause of falls.
  Frame1142 also has an8.7 mm initial-penetration annotation.
- Mid-swing408/562/873/1038 complete the initial goal. Frame727 has front-force
  peak124 N and collision termination. These favorable reset states do not
  demonstrate a learned full swing or an unbiased success rate.
- Mid-swing1195 reaches target contacts[7,7,3,3] but waits in ACQUIRE until
  timeout: final-second left-hand goal distance5.86-5.90 cm exceeds5.2 cm.
  This fixture has an8.4 mm initial-penetration annotation; treat it as an
  example of the completion criterion, not a population-frequency estimate.
- All actions/rewards finite; reward-sum residual<1e-6; zero confirmed-airborne
  samples improperly held in RELEASE. Applied reference rate is zero in the
  long stalls. Removing paused reward payout did not eliminate the behavior.

Evaluation exited0 and released GPU memory. Training PID440098 remained active.

### 2026-10-14: PauseDecay training improves after the model3500 audit

Read-only review through iteration6965 (09:33 Asia/Shanghai); training PID440098
still running, latest saved checkpoint model6500. No environment, reward,
optimizer, or process changes. Reproducible window analysis and results:
`outputs/training_review_20261014/{analyze_windows.py,summary.json}`.

Comparing200-iteration windows3200-3399 versus6700-6899: completed stages per
episode1.684->10.514; logged sequence success11.75%->69.56%; fall75.96%->16.77%;
timeout8.75%->1.48%; collision3.64%->12.27%. Improvement develops after4500:
completed stages2.341 at4500,4.505 at5000,6.844 at5500,8.973 at6000. Latest
window is still8.4% above6500-6699, not a demonstrated plateau. Reset frame
means remain near1526-1530 for the endpoint windows; task attempts/successes/
failures reconcile, and summed per-task successes equal completed-stage count.

Ascent success/attempt ratios in the latest window: left0->1/1->2/2->3 =
99.57/94.03/96.44%; right =94.61/93.18/93.37%. These are ratios of logged
reset-batch means, mixing random initial frames and carried-over goals, NOT
independent fixed-anchor tests or frame0 whole-sequence success. The model3500
playback findings cannot be assumed unchanged for model6500 without replay.

Quality remains unresolved: latest logged toe-up>15deg time/swing-time ratios
are39.5% left and33.1% right; AMP expert/policy scores remain about+.923/-.924,
with policy accuracy99.87%. No numeric nonfinites; PPO KL~.0133 and depth/GRU
gradient probes nonzero. Task completion improvement is not proof of reference
style fidelity or deployable visual-only perception (actor still has oracle goals).

### 2026-10-14: model6500 playback exposes missing standing completion semantics

Read-only code/reference/playback audit; no running training or environment
changes. Current relevant source hashes match the run manifest. Evidence:
`outputs/play_pause_decay_20261014/model_6500/events.jsonl`.

- One frame0 episode succeeds after813 steps/28 tasks. Final guidance frame2827
  is233 frames before motion end3060; body-target distance remains0.163 m.
- At step500 the task is14 (regrasp), not13 (release/stand), with actual
  contacts[7,7,3,3]. Right-hand target error0.05111 m exceeds its0.050 m
  tolerance, sufficient to prevent completion at that sample. The sparse log
  does not establish how long this condition persists or exact task13 timing.
- Completion ignores free-hand target positions, torso goal and stand pose.
  Task13 can hand off to14 immediately on release; task27 can terminate on
  release before returning to the final stand pose. This is a task-definition
  gap, not evidence that increasing AMP alone would fix it.
- Reference ascent airborne ankle pitch ranges[-37.0,-7.1]deg left and
  [-46.2,-13.0]deg right; this play reports maximum swing toe-up39.2/41.7deg.
  Foot position tracking is not sole-orientation tracking.
- Inspected local SSR AMP_Discriminator, PPO/storage and original AMP paper.
  Current task/style advantages are separately normalized, then weighted1/.2.
  In the stored iteration6500 sample,367 valid style windows have mean0.083,
  std0.122,24.5% zero reward: not a completely constant/absent style signal,
  but this small sample cannot validate within-stage quality ranking.

### 2026-10-16: Toe-up phase/geometry/reference-eligibility audit

No production environment, reward, checkpoint, or training-process changes.
Added bounded diagnostic scripts under outputs/toe_up_audit_20261016. Checked
IsaacLab reward_manager.py and envs/mdp/rewards.py, project contact_task.py,
utils/contact_task.py, utils/reference_guidance.py and the runtime CR1 URDF.
The three phase/reference source hashes match the baseline training manifest.

Played latest PauseDecay model9999 and Weight2x model10000 with deterministic
means, no randomization, seven reset fixtures [0,373,514,686,824,996,1142].
Snapshots are after reward computation but before reset/command advancement;
analysis uses only each fixture's first episode. Both tests exited0 after
522/540 steps; all fixtures reached the existing contact-sequence success
condition (not proof of final upright standing). No optimizer or AMP training
was instantiated. All actions finite; weighted reward-sum residual<1.5e-6.

- ACQUIRE-before-acquired toe-excess samples:140/153; every sample has the
  explicit foot-pose penalty mask. Reference joint reward is not MOVE-only.
- Upward toe-excess samples:334/358; pause scale<1 in1/0, scale=0 in0/0.
  Reference invalid in89/12, with danger reward active in78/11 of those.
- Upward airborne samples in PREPARE:260/330 and255/352. Every handoff resets
  task_steps, so task_settle_steps=10 blocks phase advancement for0.2s even
  though the policy is already moving. Some PREPARE->ACQUIRE jumps bypass MOVE.
- Full-start right1->2 toe peak24.3deg baseline,32.7deg Weight2x; both ACQUIRE.
  At the Weight2x peak the pose penalty rate is-0.348, joint-reference rate
  +0.169, fade scale1. Geometry: sole marker-0.27cm relative to target top,
  front mesh marker+6.93cm; level sole at identical ankle position-3.90cm.
  This is a kinematic comparison, not a closed-loop leveling intervention.
- Active regularization includes torque squared(-1e-5), action rate(-0.1),
  lower action second difference(-0.05), upper/lower acceleration(-2.5e-7/-1e-6)
  and soft joint limits(-10). No explicit abs(torque*velocity) energy term.

Reproduction: audit_toe.py accepts normal play CLI, --num_envs7 and a positive
--max_steps<=1500; use each checkpoint's matching task and a fresh output_dir.
analyze_toe.py regenerates foot_steps.csv and analysis.json from trace.pt and
reference.pt. No production fixes applied pending the user's choice.

### 2026-10-16: Make contact settling physical-reset-only

User authorized repairing the extra stage/physical-motion delay. Production
change is confined to mdp/contact_task.py: add a per-env `_reset_steps` counter,
clear it in `_resample_command`, age once in cached post-physics `refresh`, and
use it for phase/completion settling instead of per-goal `task_steps`. Goal
handoffs retain reset age. Keep task deadlines, safety checks, four-step event
and completion confirmation, reward weights, foot constraints, AMP and actor
input/checkpoint contract unchanged. No training or checkpoint writes.

References checked: IsaacLab manager_based_rl_env.py reset/step ordering,
command_manager.py, mdp/commands/pose_2d_command.py and velocity_env_cfg.py;
whole_body_tracking/tasks/tracking/mdp/commands.py; local contact phases and
reference guidance. RSL on_policy_runner.py and train.py confirm randomized
initial episode_length_buf, so that buffer cannot replace a true reset age.

Validation:
- Added four command regressions for handoff release, per-env reset-age
  isolation/randomized episode age, early flight versus contact flicker, and
  consecutive completion/event confirmation. Updated two sequence tests for
  reset-only settling. These six tests failed against the old implementation.
- Added a reference-integration regression: handoff reaches MOVE by step8 and
  subsequent reward matches the flight reference without another10-step wait.
- 113 focused command/phase/reference/foot/config tests pass in the real Isaac
  runtime; log `outputs/phase_handoff_fix_20261016/tests_final.log`. The initial
  broader historical config suite also exposed two unrelated pre-existing
  failures; those tests and their production targets were not changed.
- Seven-env deterministic model10000 smoke exited0 after553 steps. All seven
  first episodes met the existing contact-sequence success condition. Runtime
  assertions checked10 resample calls (8 partial),203 handoff envs, repeated
  refresh caching and untouched counters for non-reset envs. These counts
  include later auto-reset episodes, whereas policy metrics use first episodes.
- Actions finite; reward-sum residual<1.5e-6; old checkpoint loads unchanged.
  Frame0 ascent MOVE is recognized at goal steps8-9 instead of13. Toe-up is
  still present and some old-policy peaks increased; no quality improvement
  or deployment robustness is claimed without retraining/evaluation.

Artifacts: `outputs/phase_handoff_fix_20261016/{run_tests.py,smoke.py}` and
`reference2x/model_10000/{reset_handoff_checks.json,trace.pt,analysis.json,
foot_steps.csv,episodes.json}`. Smoke only instruments this process and reuses
the earlier toe audit; no production monkeypatch or training run is left active.

### 2026-10-16: Reference-1x resume with control regularization 2x and small work cost

User requested a quality fine-tune from the original reference-1x `model_9999.pt`,
not the later reference-2x experiment. Added an independent registered task:
`Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-Reference-PauseDecay-Reg2x-Energy-AMP-v0`.
Its environment/runner inherit the reference-1x PauseDecay baseline. The new
experiment directory is `cr1_pause_decay_reference_reg2x_energy_contact_sequence_visual_amp`.

Only reward changes relative to the current baseline:
- action_rate_l2: -0.1 -> -0.2;
- action_second_difference_l2_lower: -0.05 -> -0.1;
- joint_acc_l2_upper: -2.5e-7 -> -5e-7;
- joint_acc_l2_lower: -1e-6 -> -2e-6;
- joint_torque_l2: -1e-5 -> -2e-5;
- new joint_mechanical_power: -5e-5 * sum(abs(applied_torque * joint_vel)).

The power function returns watts for all robot joints, with SceneEntityCfg
selection supported. RewardManager multiplies by control dt exactly once.
It is a mechanical-work proxy, not electrical energy; no cross-joint positive/
negative cancellation, reference matching or phase gating is used for this cost.
Static-support torque remains covered by joint_torque_l2. Soft limits (-10),
contact/safety rewards, six reference-1x rates, task/AMP advantage weights,
learning-rate setup and actor/critic input contracts remain unchanged. The
physical-reset-only settling fix from the preceding entry remains active.

References checked: IsaacLab `envs/mdp/rewards.py::joint_torques_l2`,
`managers/reward_manager.py::compute`, humanoid `mdp/rewards.py::power_consumption`,
velocity_env_cfg.py, and whole_body_tracking tracking/mdp/rewards.py. The official
humanoid power term assumes torque actions; our position-target PD action needs
applied_torque instead. No runtime whole_body_tracking dependency was added.

Validation:
- 29 reward unit tests pass, including two new power tests for per-joint abs,
  applied versus computed torque, zero velocity, joint selection, and no inner dt.
- 3 real-runtime isolated-config tests pass for PauseDecay, reference2x and
  Reg2xEnergy. Whole config equality after undoing the intended changes confirms
  unchanged observations, reset/phase/termination, reference, AMP and runner.
- 8-env reset plus 24 deterministic steps pass shape/finite checks. Then the
  native runner completes 2 updates, restoring original model9999 actor, critic,
  AMP discriminator/normalizer and both optimizer states. Start updates10000,
  end10002, final small-test model10001; checkpoint tensors finite.
- Checked 72 reward computations: negative power rate equals -5e-5*P, and
  summed weighted rates times dt equal total task reward. Mean sampled power
  485.56W, added rate -0.02428; this tiny test does not measure policy quality.
- Earlier seven-fixture nominal baseline traces calibrate mean643.28W,
  prospective added rate -0.03216 versus reference rate+0.34544. Not a training-
  distribution or success-rate estimate.

Artifacts: `outputs/reg2x_energy_resume_20261016/validate_resume.py`,
`validation_pass.log`, `validation.json`; successful short run:
`logs/rsl_rl/cr1_pause_decay_reference_reg2x_energy_contact_sequence_visual_amp/2026-10-16_16-11-56_smoke_resume9999_reg2x_energy_env8_iter2`.
Two initial checks exposed diagnostic-only issues (serialized callable assertion,
then manual reset outside inference_mode); both were corrected in the tests/harness,
not production. The failed second process was terminated after hanging on exit.
No full training launched. Proposed user command resumes the original model9999,
not the tiny-test output, for 4000 additional updates, 5120 envs, saves every500.
Original checkpoint and historical task variants are retained unchanged.

### 2026-10-16: Correct the regulation fine-tune to a 2500 + 2000 update schedule

User clarified that the five control regularizers must start at1x, linearly
reach2.5x after2500 additional PPO updates, and remain2.5x for another2000.
This supersedes the previous fixed2x/4000-update command. Added isolated
`Reference-PauseDecay-RegRamp-Energy-AMP-v0` environment/runner (full prefix
`Ladder-Climb-CR1-ZDW0824-Visual-ContactSequence-Phased-BodyGoal-FootSafety-`).
Runner default max_iterations4500;5120env supplied by launch; save_interval500.
The old fixed2x task is retained but is not the corrected experiment.

Environment starts with all five reference-1x baseline control weights. The
mechanical-power rate stays -5e-5, not ramped. Reference, AMP, safety, limits,
policy/critic architecture, observation/data contracts and learning-rate setup
are unchanged. At multiplier2.5, action_rate=-.25, lower action second
difference=-.125, upper acceleration=-6.25e-7, lower acceleration=-2.5e-6,
applied torque squared=-2.5e-5.

Implementation references: IsaacLab envs/mdp/curriculums.py::modify_reward_weight,
RewardManager.get_term_cfg/set_term_cfg, velocity_env_cfg curriculum wiring;
whole_body_tracking tracking_env_cfg curriculum pattern; existing local AMP
update metrics/checkpoint hooks and RSL on_policy_runner rollout lifecycle.
The official curriculum is environment-step/reset oriented and does not retain
the start of a resumed fine-tune. This opt-in schedule therefore uses saved PPO
num_updates and the official RewardManager weight API before each rollout.

EndToEndVisualAMPPPO accepts optional regulation_reward_schedule; defaultNone
preserves historical behavior. With k completed updates since schedule start,
multiplier=1+1.5*min(k/2500,1). Set absolute base_weight*multiplier, never compound
an already-scaled weight. A legacy checkpoint starts the schedule at its saved
num_updates. A scheduled checkpoint preserves start_update/config and rejects
changed/disabled schedules on training resume; actor-only playback is exempt.
Snapshot/checkpoint metadata and rollout-aligned metrics retain the schedule.
TensorBoard: Loss/regulation_multiplier, Loss/regulation_elapsed_updates,
Loss/regulation_weight_<term>. From model9999, the origin is updates10000:
global rollout12500 begins the2.5x plateau; global14499 is the final rollout.

Verification:78 tests pass in the real Isaac runtime, including9 schedule tests,
the existing AMP/SSR parity and checkpoint suite,29 reward tests and4 isolated
config tests. Boundary checks cover elapsed0,1,1250,2499,2500,4499,9000; validate
mid-ramp/plateau restores, no compounding, fixed within-rollout rates, correct
metrics and unchanged energy/reference/limits/AMP weights.
Eight-env reset +24 deterministic steps then2 native PPO/AMP updates pass with
the original reference-1x model9999.72 reward computations check applied weights,
power scaling and reward totals. After saving/reloading, origin remains10000
and next multiplier is1.0012 rather than1.0. All checkpoint network tensors finite.
Real manager boundary values [1,1.75,2.5,2.5] verified at offsets[0,1250,2500,4500];
this is a mechanism check, not4500 updates of training or a quality evaluation.

Artifacts:outputs/reg_ramp_resume_20261016/{validate_resume.py,validation_pass.log,
validation.json}. Successful short run:
logs/rsl_rl/cr1_pause_decay_reference_regramp_energy_contact_sequence_visual_amp/
2026-10-16_16-23-05_smoke_resume9999_regramp_energy_env8_iter2.
Initial diagnostic-only inference_mode mistakes were corrected in tests/harness;
one failed validation process was stopped. The final short run exited normally.
No full training launched; corrected command still resumes original model9999,
not any small-test checkpoint, with4500 additional updates.
