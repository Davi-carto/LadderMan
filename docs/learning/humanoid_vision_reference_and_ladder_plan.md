# Humanoid Vision Reference and LadderMan Plan

> This file preserves the reference study and staged migration plan. Some
> dimensions below are historical proposals. The authoritative description of
> the currently implemented network and data contract is
> `docs/learning/cr1_transformer_vision_current_design.md`.

## Purpose and Scope

This note records the visual-policy design reference from
`/home/zhangdawei/humanoid_vision` and defines the gated implementation path
for CR1 ladder climbing. It is a planning and code-reading record, not an
acceptance of the existing visual code or a request to start visual training.

Every implementation stage below stops at its acceptance gate. Do not start
the next stage until the operator has run the stated check and approved the
result.

## Current LadderMan Snapshot

The camera and Warp observation path have passed their operator checks; the
policy network remains a prototype that will be replaced through the gates in
this document:

| Area | Current artifact | Status |
| --- | --- | --- |
| Registered tasks | `Ladder-Climb-CR1-Geometry-DR-Visual-v0` and `...-WarpVisual-v0` | RTX baseline and faster native batched Warp path are present. |
| Sensor | Head-mounted D435-style view, `60 x 80`, `87 x 58 deg`, metric depth and RFM classes | Moving Warp images, self-occlusion, RFM, geometry randomization, and 256-env throughput were operator checked. |
| Actor inputs | `visual_proprio` `(N, 310)` and metric `visual_depth` `(N, 1, 60, 80)` | Deployable-only by intent. |
| Critic inputs | Separate `critic` group containing expert/reference state | Correct asymmetric ownership in principle. |
| Augmentation | Clip `[0.1, 2.0] m`, Gaussian `0.005`, dropout `0.05`, rung-focused masking probability `0.1` | Matches the LadderMan paper's two mutually exclusive branches. |
| Actor model | Three-layer `[8, 8, 8]`, kernel `5` CNN plus MLP | Old feed-forward prototype only; it does not implement the frozen Transformer-GRU plan below. |
| Visualization | Standalone RTX/Warp camera viewers and batched Warp replay/benchmark scripts | Camera preparation and acquisition gate complete. |

The new visual actor deliberately differs from the old expert actor. It must
never receive reference-body state, reference joints, ladder geometry, target
contacts, contact errors, or any other simulator-only signal. Those values may
remain in the `critic` group or auxiliary training targets only.

## References Reviewed

For the visual network and learning algorithm, the implementation reference
priority is:

1. `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure`;
2. the other relevant code and papers under `humanoid_vision`;
3. Ladder-owned adaptation.

Within the first reference, inspect `modules/actor_critic.py`,
`modules/actor_critic_recurrent.py`, `modules/submodule.py`,
`algorithms/ppo_ce.py`, `algorithms/algorithm_plugin.py`,
`storage/rollout_storage.py`, and the runner together. Reuse their named
encoding dictionaries, recurrent rollout/update handling, auxiliary target
storage, decoder/loss organization, inference use of the VAE mean, and
asymmetric PPO separation wherever they fit the frozen Ladder contract.

This priority concerns visual architecture and learning semantics. IsaacLab's
official source remains the required reference for ManagerBasedEnv lifecycle,
observation history, tensor shapes, task registration, and the installed
RSL-RL extension API. Ladder must locally adapt the research implementation;
it must not import the reference project or patch `site-packages` at runtime.

### LadderMan Paper and Local Code

- `docs/LadderMan.pdf`, Section 3.2, Section 3.3, Figure 3, Appendix A.2.
  It specifies a `60 x 80` depth input, depth range `[0.1, 2.0] m`, Gaussian
  noise `N(0, 0.005)`, dropout `0.05`, camera extrinsic perturbation of
  `+/- 0.01 m` and `+/- 1.5 deg`, and RFM on 10 percent of images.
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_visual_env_cfg.py`.
  Existing sensor and actor/critic observation split.
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/observations.py` and
  `Ladder/source/ladder/ladder/utils/visual_depth.py`.
  Current conversion from Isaac Lab NHWC camera tensors to NCHW depth,
  clipping, noise/dropout, and RFM.
- `Ladder/source/ladder/ladder/models/visual.py`.
  Existing feed-forward visual actor prototype.
- `Ladder/scripts/view_cr1_visual_observations.py`.
  Existing offline four-panel depth diagnostic.

### `humanoid_vision` / SSR

- `LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/base/legged_robot_vis_config.py`
  and `envs/X2/x2_vis_config.py`.
  Depth camera geometry, cropping/downsampling, depth update rate/lag,
  camera randomization, and a separate privileged critic.
- `.../legged_gym/envs/base/legged_robot_vis.py`.
  Initializes depth history after reset, updates a fixed-rate depth buffer,
  exposes it separately from proprioception, and visualizes latest/history
  depth images.
- `rslRL-317-rq-humanoid_ssr_pure/rsl_rl/modules/actor_critic_recurrent.py`.
  A CNN encodes depth, an MLP encodes proprioception history, and a GRU fuses
  the two streams. The GRU hidden state is reset per environment.
- `.../rsl_rl/runners/on_policy_runner.py`, `algorithms/ppo_mm.py`, and
  `storage/rollout_storage.py`.
  The runner forwards depth separately, PPO stores depth frames and recurrent
  hidden states, and the minibatch path reconstructs padded trajectories.
- `ssr Scaling Surefooted and Symmetric Humanoid.pdf`, Section 3.1 and
  Appendices A.4-A.6.
  SSR uses asymmetric actor-critic, a `36 x 36` depth image, CNN/MLP/GRU
  fusion, one-frame visual lag, and small camera pose/FoV randomization.

### Isaac Lab 2.1

- `isaaclab_v2.1.0/scripts/demos/sensors/cameras.py`.
  Official RGB/depth camera launch and image-grid reference.
- `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_camera_env_cfg.py`.
  Official manager-based `TiledCameraCfg` plus image observation pattern.
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/camera_cfg.py`.
  Camera conventions: ROS is forward `+Z`, OpenGL is forward `-Z`, and world
  is forward `+X`. Camera convention must be checked on images, not guessed.
- `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/observation_manager.py`.
  Observation history is available but image history changes tensor rank; it
  cannot be added to a CNN policy without an explicit data-contract test.
- `isaaclab_v2.1.0/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`
  and `rl_cfg.py`.
  The local training integration must carry a visual actor input separately
  from privileged critic observations. It is an integration reference, not a
  reason to choose distillation.

No visual implementation was found in `whole_body_tracking`; it is not a
runtime dependency for this work.

## What To Reuse

| Reference idea | LadderMan decision |
| --- | --- |
| Asymmetric actor-critic | Primary training route. Actor stays deployable; critic retains privileged reference/geometry state. |
| Depth separate from proprioception | Reuse immediately. Do not flatten raw images into the actor vector. |
| Reset-filled image history and explicit sensor latency | Reuse after the single-frame visual baseline works. A reset must populate the buffer to avoid undefined first-frame inputs. |
| CNN + proprioception-history + GRU | Primary network reference after the single-frame contract is accepted. It requires recurrent rollout-storage verification. |
| Camera position/orientation/FoV randomization | Reuse gradually after calibrated raw images are accepted. Use the LadderMan `+/- 1.5 deg` rotation bound first; add FoV perturbation only after a fixed intrinsics test. |
| Rendering with robot self-occlusion | Preserve in the renderer validation. The paper's Warp training route is the target backend, but only after a local batched renderer handles the geometry-randomized ladder and native RFM classes correctly. `TiledCamera` remains the visual-validation and throughput-baseline backend. |
| SSR 36 x 36 input | Do not copy initially. Ladder rungs are thin and the LadderMan task specifies `60 x 80`; reduce resolution only through an accuracy/throughput ablation. |
| SSR MoE actor, symmetry architecture, foothold imagination, and motion-prior stack | Do not include in the first visual policy. They add independent algorithmic risk and cannot validate camera integration. |
| LadderMan RFM and VFM deployment idea | Keep RFM in the Ladder-specific data design. Defer Fast-FoundationStereo integration until simulated policy, real sensor capture, and deployment latency are each validated. |

`humanoid_vision` uses Isaac Gym and a custom RSL-RL fork. Its code is a
behavioral and architectural reference only. LadderMan must use Isaac Lab APIs
and local code; it must not import `legged_gym` or the reference RSL-RL fork at
runtime.

### Warp Throughput and Ladder Migration

The LadderMan paper, Appendix A.2, states that its `60 x 80` training depth is
rendered with NVIDIA Warp rather than Isaac Sim's built-in renderer. It does
not publish a renderer benchmark or the kernel implementation. The relevant
reusable implementation pattern is therefore `humanoid_vision` `DepthCastMesh`,
not the current single-scene CR1 Warp diagnostic.

Warp improves throughput when the complete visual path is specialized and
batched: current body poses and camera transforms are GPU tensors; one kernel
launch covers `rendered_envs x height x width` rays; static mesh acceleration
structures and `[max_envs, height, width]` depth buffers are reused; and the
result stays on the GPU for augmentation and the policy. This bypasses USD
camera updates, Omniverse/RTX rendering, Replicator semantic annotation, and
per-camera renderer bookkeeping. It only replaces sensing, not PhysX stepping,
so a speedup is expected only when visual rendering is a meaningful share of
the end-to-end training time.

The existing `CR1WarpCamera` intentionally does not meet this contract: it has
one camera and one output image, loops over all 18 ladder boxes and 30 robot
collision meshes per pixel, and synchronizes the GPU after every frame for
diagnostic consumption. It must remain a correctness viewer, not be inserted
into the trainer.

`x2_vis` cannot be imported directly because it is coupled to Isaac Gym
terrain coordinates and task classes. Its local adaptation must instead:

1. Preallocate batched CR1 camera, depth, hit-class, ladder-pose, and robot-link
   pose tensors on the IsaacLab device. Derive each camera pose from the current
   `d435_head_link` pose plus that environment's randomized extrinsics.
2. Launch one Warp kernel for all due environments at each visual update. Use
   the runtime `RigidObjectCollection` poses for the 18 individually randomized
   ladder OBBs; unlike `x2_vis` terrain, Ladder's ladder cannot be one static
   shared mesh because inclination and rung spacing change per environment.
3. Return native metric depth and native classes (`RUNG`, `ROBOT`, other ladder,
   ground, background). Build the paper RFM directly from `RUNG | ROBOT`, so a
   training Warp task needs no RTX semantic-segmentation buffer.
4. Preserve robot self-occlusion with CR1 mesh queries. A simplified occluder
   set may be benchmarked only after the full-collision version is visually
   accepted; it cannot silently replace the accepted image contract.
5. Keep rendering, masking, noise, dropout, clipping, NCHW conversion, CNN,
   and PPO tensors on GPU. Do not synchronize per image; synchronize only at a
   true consumer boundary if IsaacLab/PyTorch stream semantics require it.

The next gated visual stage is a standalone *batched* Warp renderer proof at
the eventual training resolution and update rate. Its acceptance requires
correct moving-camera images, nonempty rung/robot classes where expected, and
correct RFM across fixed motion frames and per-environment ladder randomization.
Only then create a Warp-backed training task and benchmark it against the
current `TiledCamera` task under identical environment count, `60 x 80` output,
visual update period, RFM requirement, headless setting, policy, and GPU.
Record renderer milliseconds, end-to-end environment steps/s, GPU memory, and
depth/mask statistics. Use Warp for training only when both image correctness
and end-to-end throughput improve; `TiledCamera` remains the fallback.

## Gated Implementation Plan

Stages 0-4 below are the historical camera-acquisition plan and have already
passed their operator checks. Stages 5-8 are the earlier high-level network
outline. If they conflict with the frozen network contract later in this
document, the `2026-08-13 - Frozen Ladder Visual-Network Plan` section and its
Gate A-F sequence are authoritative.

### Stage 0: Freeze the Nonvisual Baseline

Goal: define the nonvisual baseline and comparison target before touching vision.

Implementation boundary:

- Record the exact registered task, motion file, checkpoint, action dimension,
  evaluation seeds, ladder-geometry distribution, and success/failure metrics.
- Save a short deterministic replay at fixed reference frames and a
  geometry-randomized evaluation summary.
- Identify which current critic/reference observations are unavailable on the
  deployed robot.

Acceptance gate:

- The operator can replay the frozen nonvisual checkpoint.
- We have a baseline success rate, completion time, termination breakdown, and
  contact metrics on nominal and geometry-randomized ladders.

Do not: change rewards, reset distributions, action scaling, contacts, or the
baseline checkpoint during visual work.

### Stage 1: Live RGB and Depth Camera Proof

Goal: make sensor placement observable before putting any camera data in a
policy.

Implementation boundary:

- Create a standalone one-ladder scene and a moving `TiledCamera`, based on
  the official Isaac Lab camera demo. The camera follows the ladder diagonally
  from above; this stage deliberately has no robot or Gym task.
- Enable RGB and `distance_to_image_plane` from the same camera pose and show
  them in two live windows with a fixed metric depth color range.
- Keep the normal Isaac Sim viewport open and save periodic RGB, colorized
  depth, and raw metric-depth captures.

Acceptance gate:

- RGB shows an upright ladder from the expected diagonal-above moving view.
- Depth is finite, uses meters, respects the selected range, and shows ladder
  rungs at the same pixels as RGB.
- The operator inspects the lower, middle, and upper sections of the camera
  trajectory. A capture of each is kept with the run output.

Stop condition: any camera frame convention, field of view, or depth
definition ambiguity. Resolve it visually before proceeding.

### Stage 2: Calibrated Deterministic Sensor Contract

Goal: turn the accepted image into a repeatable sensor contract.

Implementation boundary:

- Mount the accepted camera on the selected CR1 D435 link, then lock image
  size at `60 x 80`, depth type at `distance_to_image_plane`, range at
  `[0.1, 2.0] m`, and the accepted intrinsics/extrinsics.
- Verify explicit non-colorized rung and robot semantic IDs and the paper RFM
  keep mask against RGB/depth.
- Record camera pose, intrinsics, dtype, NHWC source shape, NCHW actor shape,
  min/max depth, invalid-pixel fraction, and rung-pixel fraction.
- Add automated checks for three fixed reference frames. The checks test data
  semantics and dimensions, not pixel-perfect rendered values.

Acceptance gate:

- Repeated fixed-seed runs preserve tensor shape and bounded statistics.
- The semantic rung mask covers rungs rather than rails, ground, or robot links.
- The RFM keep mask retains rungs and visible robot surfaces while excluding
  rails, supports, unlabelled geometry, ground, and background.
- The chosen mount has enough upcoming-rung coverage for the climbing horizon.

### Stage 3: Vectorized Observation Pipeline Without a Visual Policy

Goal: prove that camera rendering and observation management work at realistic
parallel scale while the acting policy remains nonvisual.

Implementation boundary:

- Put raw, unaugmented NCHW depth in a named observation group separate from
  proprioception and critic groups.
- Start at 1 environment, then 16, then a conservative training-scale count.
- Reset arbitrary subsets of environments and verify image tensors, sensor
  update periods, rewards, dones, and all group shapes after each reset/step.
- Measure steps per second and GPU memory with and without the camera.

Acceptance gate:

- No stale/uninitialized depth after partial resets.
- Actor-side visual groups contain no privileged/reference terms; the critic
  shape and baseline behavior match Stage 0.
- Throughput and memory data identify a safe initial visual environment count.

### Stage 4: Depth Preprocessing and Robustness Ablation

Goal: validate image processing independently from policy learning.

Implementation boundary:

- Enable the LadderMan mutually exclusive augmentation branches: RFM on 10
  percent of images, otherwise Gaussian noise plus dropout.
- Add camera extrinsic randomization only after the fixed camera passes, then
  check `+/- 0.01 m` translation and `+/- 1.5 deg` rotation.
- Keep an unaugmented diagnostic copy outside the actor path for visualization
  and statistics. Do not leak the rung mask to the actor.

Acceptance gate:

- Observed branch frequencies, noise scale, dropout rate, depth bounds, and
  camera offsets match configuration over a large sampled batch.
- Training does not begin until augmented images remain recognizably aligned
  with the ladder and raw images remain inspectable.

### Stage 5: Multimodal Asymmetric PPO Data Path

Goal: reproduce the `humanoid_vision` data flow: visual information is
processed separately and combined with proprioception in an asymmetric PPO
actor-critic, without teacher-student distillation.

Implementation boundary:

- Extend the local PPO runner, actor-critic interface, and rollout storage
  only as required to pass depth separately from proprioception.
- Keep privileged reference/geometry state in the critic only. The actor
  receives only deployable proprioception, climbing direction, and depth.
- Prove one collected rollout, one PPO update, and one checkpoint reload with
  task rewards. Do not add a teacher action loss, KL imitation loss, or DAgger
  rollout path.

Acceptance gate:

- One rollout/update/reload completes with finite observations, actions,
  rewards, values, advantages, and PPO losses.
- Depth encoder parameters receive nonzero finite gradients, while a deliberate
  privileged actor input fails a data-contract test.

### Stage 6: Recurrent Visual Asymmetric PPO Actor

Goal: locally adapt the mature `humanoid_vision` visual network without
creating a runtime dependency on its Isaac Gym or RSL-RL packages.

Implementation boundary:

- Encode the accepted depth input with a CNN, proprioception history with an
  MLP, and fuse them with a GRU, following
  `rsl_rl/modules/actor_critic_recurrent.py`.
- Add an explicit reset-filled depth buffer and one control-step visual lag.
  Record the update timestamp; do not rely on an undocumented image-history
  tensor layout.
- Store depth, done masks, and recurrent hidden states through the PPO
  minibatch path. Reset hidden state only for done environments.

Acceptance gate:

- A partial-reset test proves correct depth-buffer contents, visual lag, and
  GRU reset masks.
- One recurrent PPO rollout/update has the expected tensor shapes and finite
  gradients.

### Stage 7: Visual Asymmetric PPO Training Baseline

Goal: train the accepted visual actor with task rewards while preserving the
Stage 0 environment, reward, and action contracts.

Implementation boundary:

- Start from fixed camera intrinsics/extrinsics and the Stage 0 task reward.
- Run a short PPO job before any visual domain randomization; log reward
  components, termination causes, value loss, policy loss, entropy, KL, and
  depth statistics separately.
- Evaluate exactly on the Stage 0 nominal and geometry-randomized suites.

Acceptance gate:

- Short PPO runs remain finite and checkpoint/replay successfully.
- The visual actor has a nontrivial successful rollout and a documented
  comparison with the Stage 0 nonvisual baseline.

### Stage 8: Visual Robustness and Sim-to-Real Validation

Goal: close the gap only after simulated visual performance is established.

Implementation boundary:

- Capture synchronized real D435 RGB/stereo/depth data at the accepted mount.
- Compare real depth against the simulator's units, FoV, crop, latency,
  invalid pixels, and rung visibility. Introduce Fast-FoundationStereo only
  after measuring its latency and output alignment.
- Export the deployable actor and test state reset, depth preprocessing, and
  end-to-end inference timing on target hardware.

Acceptance gate:

- The runtime receives the same NCHW contract as the simulator and does not
  require critic, segmentation, or simulator data.
- Inference plus sensing latency is measured against the policy control period.

## 2026-08-13 - Frozen Ladder Visual-Network Plan

This section is the authoritative specification for the next implementation.
It primarily adapts `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure`, then
draws on `humanoid_vision/actor_critic.py`, `submodule.py`, and the other
related files in that project. It is not a byte-for-byte copy and not
teacher-student distillation.
The reusable reference choices are ELU MLP/CNN blocks, 64-D tokens, two
self-attention layers, independent linear estimation heads, VAE
reparameterization, local-map decoders, future-proprioception reconstruction,
and asymmetric PPO ownership. The Ladder-specific changes are two depth frames,
ten spatial visual tokens, a post-Transformer GRU, and separate hand- and
foot-centered ladder maps.

The reference's global `32 x 16` heightmap reconstruction,
`HMapPresEncoder`, and U-Net refinement are excluded. This does **not** remove
terrain reconstruction altogether: the four requested limb-centered ladder
maps remain training-only auxiliary targets. The accepted feed-forward
`LadderVisualActorModel` and registered `...-WarpVisual-v0` task are kept
unchanged as comparison baselines; the recurrent task gets a separate config
and registration.

### Authoritative end-to-end data flow

```text
deployable observations
  proprioception history obs_h [B,10,85]
  two processed depth frames   [B,2,60,80]
                 |
                 v
  state MLP -> 1 x 64-D token
  depth CNN -> 10 x 64-D tokens
                 |
                 v
  11-token, two-layer self-attention Transformer
                 |
  all 11 output tokens, fixed order -> flatten [B,704]
                            |
                  Linear(704,256) + ELU
                            |
                            v
                  GRU(256 -> hidden 256)
                            |
                            v
  named heads: velocity 3, LH 8, RH 8, LF 8, RF 8,
               dynamic mu 16, dynamic logvar 16
                            |
             actor latent [B,51]
                            |
           concat current o_t [B,85]
                            |
              actor MLP -> 25 actions

training-only labels
  true body-frame linear velocity, four local ladder maps,
  next proprioception frame, map-valid masks
                            |
             auxiliary reconstruction losses

privileged critic observation ----------------> critic MLP -> value
```

The seven estimator heads expose named tensors rather than one anonymous
67-D vector. This makes it mechanically testable that `dynamic_logvar` is used
by KL/reparameterization but never reaches the actor.

### Deployable input contract

One proprioception frame is:

```text
o_t = [base angular velocity 3,
       root orientation 6D 6,
       joint positions 25,
       joint velocities 25,
       previous action 25,
       climbing direction 1]                         # D_p = 85
```

The 25-joint count is statically confirmed by
`ladder.robots.cr1.CR1_ACTUATED_JOINT_NAMES`; Gate B still asserts all runtime
shapes and the joint ordering. The ten-frame input is ten complete,
chronologically ordered copies of this
same vector, not the current mixed `joint_pos_history=9` plus
`action_history=3` observation. True base linear velocity is deliberately
absent because it is the explicit three-dimensional estimation target. Motion
reference, contact state, body state, ladder geometry, and local-map targets
are also absent from the actor.

```text
obs_h = [o_(t-9), ..., o_t]       [B, 10, 85]
depth = [d_(t-1), d_t]            [B, 2, 60, 80]
```

On reset, all ten proprioception slots receive the first valid `o_t`, and both
depth slots receive the first valid depth image. There must be no data from the
previous episode. Both histories advance once per policy control step.
Oldest-to-newest order is fixed, so the actor's current proprioception is
`o_t = obs_h[:, -1]`; a duplicate current-proprioception observation group is
unnecessary.

IsaacLab's history buffer adds a history axis without removing the Warp
renderer's singleton image channel. If the raw manager field is
`[B,2,1,60,80]`, the model must assert that singleton and canonicalize it to
`[B,2,60,80]` before the CNN. It must never flatten the image history into the
proprioception vector. No extra visual lag is introduced in the first
implementation; sensor lag is a later, separately accepted robustness change.

### Encoder, fusion, and recurrence

```text
obs_h [B,10,85]
  -> flatten [B,850]
  -> MLP 850 -> 256 -> 128 -> 64, ELU
  -> state token [B,1,64]

depth [B,2,60,80]
  -> four convolution stages, channels 16 -> 32 -> 64 -> 128, ELU
  -> preserve a spatial feature map
  -> AdaptiveAvgPool2d(2,5)
  -> ten 128-D spatial features
  -> shared Linear(128,64) + ELU
  -> visual tokens [B,10,64]

token sequence                               [B,11,64]
  -> 2 x TransformerEncoderLayer(
         d_model=64, nhead=1, dim_feedforward=256,
         dropout=0.1, norm_first=False)
  -> fixed order: state, visual_1, ..., visual_10
  -> flatten all Transformer outputs                 [B,704]
  -> Linear(704,256) + ELU                           [B,256] = x_t
  -> GRU(input_size=256, hidden_size=256, layers=1)
  -> recurrent state                             [B,256] = g_t
```

The two Transformer blocks are ordinary self-attention, following
`rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/modules/submodule.py`: all
eleven tokens participate as Q, K, and V. This is not a
separate cross-attention block with state as Q and terrain as K/V. Unlike the
reference's fixed mean visual pooling, the Ladder variant retains every
Transformer output. The fixed sequence order makes the flattened 704-D vector
unambiguous: state occupies the first 64 values, followed by the ten visual
tokens in the CNN pool's fixed row-major spatial order. One shared
`Linear(704,256) + ELU` performs cross-modal fusion before the GRU. There are
no separate state/visual projection branches and no token averaging after the
Transformer.

The previously proposed `state 64 -> 128` plus `visual 640 -> 128` branches
are superseded. Mapping each token independently to 256 is also not the
selected design; the only post-Transformer projection is the single
`704 -> 256` layer.

The exact four convolution kernel/stride/padding tuples remain a shape-test
parameter. The reference's valid convolutions collapse a `60 x 80` image too
far to provide ten meaningful spatial locations, so those tuples cannot be
copied literally. The first smoke-test candidate is `k/s/p = 5/2/2,
3/2/1, 3/2/1, 3/1/1`, followed by the fixed `2 x 5` adaptive pool. The output
contract, not this candidate tuple, is frozen here.

### Estimation heads and latent

Keep the heads shallow, following `actor_critic.py`. There is no shared head
MLP beyond `g_t`; each estimate is an independent linear projection:

```text
velocity_head:       Linear(256, 3)   -> v_hat
hand_left_head:      Linear(256, 8)   -> z_hand_left
hand_right_head:     Linear(256, 8)   -> z_hand_right
foot_left_head:      Linear(256, 8)   -> z_foot_left
foot_right_head:     Linear(256, 8)   -> z_foot_right
dynamic_mu_head:     Linear(256, 16)  -> mu_dyn
dynamic_logvar_head: Linear(256, 16)  -> logvar_dyn
```

During training:

```text
z_dyn = mu_dyn + exp(0.5 * logvar_dyn) * epsilon
```

Deterministic inference uses `z_dyn = mu_dyn`. The seven heads emit 67 numeric
parameters during training (`3+8+8+8+8+16+16`), but the actor latent remains:

```text
z_hand = [z_hand_left, z_hand_right]              [B,16]
z_foot = [z_foot_left, z_foot_right]              [B,16]
latent = [v_hat, z_hand, z_foot, z_dyn]           [B,51]
num_latent = 3 + (8+8) + (8+8) + 16 = 51
```

The 67 values are not all actor features. `mu_dyn` and `logvar_dyn` are the
mean and log-variance of one 16-D dynamic latent distribution. The actor needs
one 16-D sample during training (`z_dyn = mu + sigma * epsilon`) or the 16-D
mean at deterministic inference; `logvar_dyn` is used for VAE uncertainty and
KL regularization, not concatenated as another actor input.

The action branch is separate from these estimator heads:

```text
[current o_t 85, latent 51] -> actor MLP -> 25 CR1 joint actions
privileged critic observation -> critic MLP -> value
```

The actor-MLP widths are not part of this encoder decision; select them in the
model smoke test using the current CR1 and reference configurations. The local
map decoders are training-only and are removed from deployment export.

There are two temporal mechanisms with different jobs. `obs_h` supplies an
explicit ten-control-step short history at every time step. The GRU hidden
state carries information beyond that window across policy steps. Rollout
inference uses `[B,...]` inputs and hidden state `[1,B,256]`; PPO updates use
padded `[T,B_trajectory,...]` sequences plus masks and the saved initial hidden
state. The CNN and Transformer may temporarily flatten the leading trajectory
dimensions, but those dimensions must be restored before the GRU. Only done
environments reset their hidden state.

### Auxiliary decoders

There is no reference global-heightmap reconstruction or `HMapPresEncoder` in
the Ladder actor. There are still two requested local reconstruction branches:

```text
z_hand_left [B,8]  -> hand_left_decoder  MLP 8 -> 64 -> 128 -> 135
z_hand_right [B,8] -> hand_right_decoder MLP 8 -> 64 -> 128 -> 135
  -> reshape each to [B,15,9]                  # fixed left/right order

z_foot_left [B,8]  -> foot_left_decoder  MLP 8 -> 64 -> 128 -> 55
z_foot_right [B,8] -> foot_right_decoder MLP 8 -> 64 -> 128 -> 55
  -> reshape each to [B,11,5]                  # fixed left/right order

[v_hat, z_hand, z_foot, z_dyn] [B,51]
  -> MLP 51 -> 64 -> 128 -> D_p
  -> Linear(D_p,D_p)
  -> predicted next single-frame proprioception [B,D_p]
```

Use ELU between decoder linear layers. The local-map decoders end with a linear
layer because their targets are signed normalized values. The extra final
`D_p -> D_p` layer in the future decoder follows the reference
`latent_decoder_final_layer`. The future target is `o_(t+1)`, not a ten-frame
history, a 310-D mixed-history vector, or a reference observation. Transitions
that reset an environment do not contribute to future reconstruction loss.
Because `o_(t+1)` exists only after applying `a_t`, it is attached in
`process_env_step()` from the returned post-step observation. A done
transition's returned reset observation is not a valid future target and is
masked out.

### Hand and foot local-map contract

These targets are privileged collision-geometry maps used only for auxiliary
training. They are neither camera depth nor the RFM semantic mask. Each map is
centered at the current limb position in a gravity-aligned horizontal plane:

- local `x/y`: fixed to world X/Y; the limb contributes only the grid origin;
- ray direction: world `(0, 0, -1)` regardless of limb or ladder inclination;
- raw map value: nearest ladder-or-ground collision hit in world `z`;
- processed map value: limb reference `z - hit_z`, then clipped and normalized.

The initial grids are:

| target | grid per limb (`x x y`) | physical coverage | spacing | decoder output |
| --- | --- | --- | --- | --- |
| hand | `10 x 3` per side | `0.315 m x 0.160 m` | `0.035 m x 0.080 m` | `2 x 30 = 60` |
| foot | `10 x 3` per side | `0.315 m x 0.160 m` | `0.035 m x 0.080 m` | `2 x 30 = 60` |

All four limb maps now share one grid contract. Ten X samples retain both
boundaries over `0.315 m`, giving nine `0.035 m` intervals; three Y samples
retain both boundaries over `0.160 m`, giving two `0.080 m` intervals.
Left/right ordering and row-major `x,y` flattening are fixed.

The target includes all 18 ladder collision boxes and the CR1 ground plane;
the nearest world-down hit wins. The debug view labels rung, front rail, other
ladder, and ground separately. This matches SSR's terrain-height query and
gives the initial ground-supported feet real height samples. The validity mask
is reserved for non-finite/query failures. No-hit handling, normal clip range,
and final normalization are accepted only after visualizing nominal and
randomized ladders. Actual hit points are drawn on the physical terrain in the
Isaac Sim viewport.

### Losses and starting coefficients

```text
L_velocity = MSE(v_hat, v_true_body)
L_hand     = 0.5 * [masked_MSE(hand_left_hat, hand_left_target)
                   + masked_MSE(hand_right_hat, hand_right_target)]
L_foot     = 0.5 * [masked_MSE(foot_left_hat, foot_left_target)
                   + masked_MSE(foot_right_hat, foot_right_target)]
L_future   = MSE(o_next_hat, o_next)
L_KL       = mean(-0.5 * sum(1 + logvar_dyn
                              - mu_dyn^2 - exp(logvar_dyn), dim=-1))

L_aux = w_v * L_velocity
      + w_hand * L_hand
      + w_foot * L_foot
      + w_future * L_future
      + w_KL * L_KL
```

The initial coefficient proposal is
`w_velocity=2, w_hand=2, w_foot=2, w_future=5, w_KL=1`. Because `L_hand` and
`L_foot` average their two sides, the grouped weights are equivalent to one
unit for each left/right hand and foot map. Keep every coefficient in
configuration. Do not enable the combined objective until each target is
normalized, each masked loss divides by its own valid-element count, and the
unweighted losses and encoder gradient norms have been logged. This prevents
the larger Ladder map output sizes from silently changing the effective task
balance.

### Reference-aligned PPO and auxiliary update boundary

The first training implementation follows the reference's two-phase update,
rather than silently changing its optimization semantics:

1. During rollout, run the recurrent visual encoder once, sample one
   `z_dyn`, form the 51-D latent, and form `[o_t, latent]` for the action MLP.
   Store that exact actor input and its action-distribution parameters, along
   with raw histories, privileged labels, done flags, and the GRU state.
2. During the PPO phase, recompute action probabilities from the stored actor
   input. This fixes the sampled dynamic latent for the PPO ratio; re-sampling
   it inside the update would make the ratio change for reasons unrelated to
   policy parameters.
3. During the auxiliary phase, replay the stored trajectories through the
   encoder in temporal order and optimize the five weighted terms above. Log
   all unweighted and weighted losses and the encoder gradient norm.

Thus "joint PPO plus auxiliary training" means that both phases run in each
training iteration. `L_aux` is not added to rewards or advantages. In this
reference baseline the encoder/estimator/decoders are trained by the auxiliary
phase and the action/value networks by PPO. An end-to-end variant that also
backpropagates PPO through the visual encoder is a later ablation, not an
unrecorded change to the first baseline.

The installed `rl-training` RSL-RL already stores recurrent hidden states,
splits and pads trajectories, supplies masks, and resets done environments.
Reuse those facilities. Add only the local algorithm/storage fields required
for the four maps, velocity target, next-frame target, and exact stored actor
input; do not edit `site-packages` and do not import `humanoid_vision` at
runtime.

### Gated implementation sequence

1. **Gate A - four local-map targets only.** Add a standalone fixed-motion
   diagnostic. It reads runtime CR1 limb and randomized ladder poses and shows
   left/right hand maps, left/right foot maps, rung/rail/no-hit/invalid classes,
   axes, limb centers, rung locations, shapes, ranges, and valid fractions.
   Feet use the ankle-link pose plus the contact-schema local sole offset; the
   virtual sole names are not articulation body names. Verify nominal and
   geometry-randomized frames and confirm that `0.08 m` rungs are never skipped.
   Do not change the acting policy, observations, or PPO.
2. **Gate B - named temporal data contract only.** Create a separate recurrent
   Warp visual environment while preserving the existing visual baseline.
   Expose `obs_h=[B,10,85]`, canonical depth `[B,2,60,80]`, privileged true
   velocity, four maps/masks, and the post-step `o_next` target. Verify term
   order, oldest/newest order, first-frame fill, arbitrary partial resets,
   finite values, target timing, and actor/critic/auxiliary ownership. Continue
   acting with the existing policy during this gate.
3. **Gate C - pure PyTorch network only.** Implement the state/depth token
   encoders, two-layer Transformer, `704->256` fusion, `GRU(256,256)`, seven named
   heads, four map decoders, future decoder, and 51-D actor input. Test rollout
   and padded-sequence modes, exact shapes, CNN candidate geometry,
   selected-environment hidden reset, stochastic and deterministic VAE paths,
   finite losses, nonzero gradients, and proof that `logvar`/privileged targets
   cannot enter the actor. No PPO training yet.
4. **Gate D - recurrent asymmetric PPO mechanics.** Locally adapt the installed
   RSL-RL algorithm around its existing recurrent storage. Add the minimum
   auxiliary fields and reference-style two-phase update. Verify one rollout,
   PPO ratio consistency with the fixed rollout latent, one recurrent
   minibatch, done masks, `o_next` alignment, checkpoint/reload including GRU
   state behavior, and deterministic play/export. No long training.
5. **Gate E - auxiliary objectives.** Disable PPO parameter updates and first
   train only the encoder/heads/decoders on replayed short rollouts. Display
   predicted versus target maps for all four limbs, velocity error, future
   proprioception error, KL statistics, every weighted/unweighted loss, and
   gradient norms. After operator approval, run a short finite job with both
   PPO and the auxiliary phase enabled at weights `(2,2,2,5,1)`.
6. **Gate F - controlled training comparison.** Compare the nonvisual baseline,
   old feed-forward visual prototype, and Transformer-GRU policy using the same
   Warp images, ladder distribution, rewards, resets, motion, environment count,
   and evaluation seeds. Only after this gate add visual latency/FoV/noise
   ablations or sim-to-real changes.

No gate advances until its operator acceptance check passes. The immediate
next implementation is Gate A only; model, environment observation, task
registration, and PPO code remain unchanged until the four local maps are
visually approved.

## Immediate Next Work

Run and visually accept the implemented Gate A standalone local-map diagnostic
on both nominal and explicitly randomized ladder geometry. Check all four limb
maps during fixed CR1 motion playback, including rung/rail/no-hit/invalid
classes, axes, centers, coverage, and valid fractions. It does not change the
acting policy, environment rewards, PPO implementation, or registered training
task. Gate B does not begin until these four maps and their physical coverage
are approved.

Parameters deliberately left for the corresponding acceptance gate are the
four CNN kernel/stride/padding tuples, actor MLP widths, local-map normal clip
and no-hit normalization values, and target normalization statistics. The
interfaces and tensor dimensions above are fixed; these numerical choices are
not to be guessed and buried in the implementation.

### Planned Ladder-owned files by gate

This is an ownership map, not permission to implement later gates early:

| gate | Ladder-owned artifact | responsibility |
| --- | --- | --- |
| A | `scripts/view_cr1_limb_heightmaps.py` | Fixed-motion nominal/randomized four-map diagnostic and saved metadata. |
| A | `ladder/utils/limb_heightmaps.py` | Batched ladder-frame target query shared later by observations; no policy code. |
| B | new CR1 recurrent Warp environment config | Named 10-frame proprioception, two-frame depth, and training-only target groups without changing the old task. |
| B | minimal MDP observation terms | Current deployable frame, true body velocity, and four map/validity tensors. Future alignment remains algorithm-side. |
| C | new recurrent visual model module under `ladder/models/` | Token encoders, Transformer, GRU, named heads, decoders, actor distribution, hidden-state and export contract. |
| C | focused tests under `Ladder/tests/` | Rollout/sequence shapes, resets, VAE modes, losses, gradients, and actor-input isolation. |
| D | Ladder-local PPO/rollout extension module | Reference-style stored actor input, auxiliary fields/two-phase update, save/load, and logging. |
| D | new agent config and gym registration | Opt-in recurrent task only; existing nonvisual and feed-forward visual entries remain intact. |

Do not modify the installed RSL-RL package. Its `construct_algorithm()` already
supports local actor, critic, and algorithm callables through configuration,
so Ladder-specific behavior stays inside the Ladder package.

## Risks To Keep Visible

- Keep the already accepted camera pose/FoV/depth contract fixed while the
  policy network is integrated; do not mix camera changes with model changes.
- Warp is the selected training renderer based on the measured 256-env
  throughput. RTX/TiledCamera remains the independent visual baseline.
- Image history changes the observation tensor rank. The CNN/MLP/GRU actor
  requires explicitly tested image and recurrent-storage contracts.
- Local-map targets are privileged training labels and must never become actor
  inputs. Deployment consumes proprioception and depth only.
- The primary route is the `humanoid_vision` design: depth processing plus
  asymmetric PPO. Teacher-student distillation and DAgger are not in scope
  unless explicitly requested as later ablations.
