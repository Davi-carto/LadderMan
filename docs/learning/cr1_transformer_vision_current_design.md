# CR1 TransformerVision Current Implemented Design

## Status and authority

This document records the implemented CR1 visual pretraining and opt-in
upward-climbing PPO contract. It supersedes older proposed dimensions and decoder sketches in
`humanoid_vision_reference_and_ladder_plan.md`. The older file remains useful
for reference-study history and staged acceptance decisions.

The visual-only path remains available unchanged:

- the established 660-D nonvisual actor controls the robot and is frozen;
- TransformerVision is trained only by auxiliary estimation/reconstruction;
- its 51-D latent is not connected to the frozen actor or PPO;
- this is asymmetric-learning preparation, not teacher-student distillation.

Separate upward PPO tasks connect the 51-D latent to a new actor. The latest
independent ThinRails task starts that actor, critic, and TransformerVision
from scratch; it does not consume the accepted visual-only checkpoint. Safe
hold and downward commands are still pending the command lifecycle diagnostic
and separately recorded downward motion, so the current command field remains
fixed to `+1`.

## End-to-end ownership

```text
660-D deploy observation -> frozen actor -> 25-D action -> IsaacLab environment
                                                        |
                    +-----------------------------------+------------------+
                    |                                   |                  |
              10 x 85 proprio                     2 x 60 x 80 depth   privileged labels
                    |                                   |                  |
                    +---------- TransformerVision ------+                  |
                                        |                                  |
                                  51-D latent                         auxiliary losses
                                  (not in actor)
```

The critic observation is still produced by the environment but is unused by
visual-only pretraining. Height maps, ray-hit classes, valid masks, true base
linear velocity, and the future proprioception target never enter the actor.

## Timing

- physics step: `0.005 s`;
- action decimation: `4`;
- control and visual sample period: `0.02 s` (`50 Hz`);
- proprioception history: 10 consecutive samples, oldest to newest;
- depth history: 2 consecutive samples, oldest to newest;
- reset fills every history slot with the first valid current sample;
- default rollout chunk: 24 control steps (`0.48 s`).

## Warp depth acquisition

- parent body: `d435_head_link`;
- URDF transform from `body`: position `(0.11871, 0.0175, 0.5096) m`, RPY
  `(0, 0.61086, 0) rad`;
- direct render resolution: `80 x 60`; no high-resolution resize and no crop;
- horizontal/vertical FoV: `87/58 deg`;
- ThinRails PPO only: independent per-environment horizontal and vertical FoV
  offsets `+/-1 deg`, resampled at reset;
- image-plane depth interval: `[0.1, 2.0] m`;
- per-reset camera translation perturbation: each axis `+/-0.01 m`;
- per-reset camera RPY perturbation: each axis `+/-1.5 deg`;
- Warp hit classes: background, rung, robot, ladder-other, and ground;
- robot collision meshes/spheres, current randomized ladder boxes, and ground
  all participate in the ray query.

The semantic/hit-class image is not a network channel. It is used only to
construct the training-time rung-focused depth branch.

### Mutually exclusive depth augmentation

Per environment and rendered frame:

1. probability `0.1`: retain only exact rung and robot pixels, set everything
   else to zero, with no Gaussian noise/dropout in this branch;
2. probability `0.9`: retain full depth, add Gaussian noise with standard
   deviation `0.005 m`, then independently set `5%` of pixels to zero.

The accepted visual-only CNN continues to receive metric values in `[0,2.0]`,
including invalid zeros. The ThinRails PPO CNN history instead applies the X2
centering `(depth_m - 1.0) * 1.0` after augmentation, giving nominal
`[-1,1]`. This changes only encoder input; diagnostic depth remains metric.

## One normalized 85-D proprioception frame

```text
[0:3]    body-frame root angular velocity x 0.2
[3:9]    Ladder root orientation 6-D, unchanged
[9:34]   (joint position - randomized default position) / CR1 action scale
[34:59]  joint velocity x 0.1
[59:84]  previous action, unchanged
[84:85]  climb direction (+1), unchanged
```

The concatenated frame is clipped to `[-100,100]`. The same builder and field
scales produce both history frames and the next-frame target. In the ThinRails
PPO task only, history input receives uniform sensor noise before these scales:
angular velocity `+/-0.5 rad/s`, orientation `+/-0.05 rad` per axis through a
valid quaternion perturbation, joint position `+/-0.01 rad`, and joint
velocity `+/-1.5 rad/s`. The future target remains clean.

### Why the orientation field is 6-D

The 6-D field is the first two columns of the valid root rotation matrix. The
two columns are orthonormal and determine the missing third column by their
cross product, so no orientation information is discarded. This is a common
neural-network rotation representation because it varies continuously with the
physical rotation. Euler angles have wrapping/singularity problems, while a
unit quaternion has the equivalent pair `q` and `-q`; selecting one sign moves
rather than eliminates the resulting boundary. Requiring all four quaternion
components to be positive is invalid because only a simultaneous global sign
flip preserves the represented rotation. Requiring only `w >= 0` is possible,
but creates a sign-selection boundary at 180 degrees.

This is a Ladder-specific choice, not an exact X2 SSR copy. X2 uses 3-D
projected gravity. Projected gravity describes tilt but is unchanged by a
rotation around the world gravity axis, so it deliberately omits absolute yaw.
The 6-D matrix field retains the complete root orientation, including yaw.

The 6-D orientation is the first two columns of the root rotation matrix. It
is a Ladder-specific input choice already used by Ladder observation code. It
is not copied from the current X2 SSR observation: X2 uses 3-D projected
gravity in its actor observation. SSR network code accepts the configured
observation vector and does not require this 6-D representation.

## Four privileged limb-height targets

Shape and order:

```text
[B,4,10,3] = left hand, right hand, left foot, right foot
```

- both hands use the sphere-hand link center;
- both feet use the ankle pose plus local sole offset `(0.045,0,-0.075) m`;
- every grid has 10 world-X points over `0.315 m` (`0.035 m` spacing);
- every grid has 3 world-Y points over `0.160 m` (`0.08 m` spacing);
- limb pose supplies only the origin; grid axes do not rotate with the limb;
- rays start `2 m` above the grid and travel world `-Z` up to `4 m`;
- queried surfaces are rung, rail, other ladder boxes, and ground;
- display value is world hit Z;
- training value is `limb reference Z - world hit Z`, in raw meters;
- no-hit and invalid samples are removed by `hit_valid`.

The three manager outputs (relative height, valid mask, hit class) share one
ray query only within the same `common_step_counter`. The cache is invalidated
on every observation-term reset. This prevents the old first-frame target from
being reused throughout a trajectory or after manual reset.

## TransformerVision network

### Proprioception branch

```text
[B,10,85] -> flatten 850
 -> Linear(850,256) + ELU
 -> Linear(256,128) + ELU
 -> Linear(128,64)
 -> one state token [B,1,64]
```

### Depth branch

```text
[B,2,60,80]
 -> Conv 2->16,   k5 s2 p2 + ELU       [B,16,30,40]
 -> Conv 16->32,  k3 s2 p1 + ELU       [B,32,15,20]
 -> Conv 32->64,  k3 s2 p1 + ELU       [B,64,8,10]
 -> Conv 64->128, k3 s1 p1 + ELU       [B,128,8,10]
 -> AdaptiveAvgPool(2,5)               [B,128,2,5]
 -> ten row-major 128-D cells
 -> shared Linear(128,64) + ELU
 -> ten visual tokens [B,10,64]
```

### Attention, fusion, and recurrence

```text
[state token, ten visual tokens] [B,11,64]
 -> two TransformerEncoderLayer blocks
    d_model=64, nhead=1, FFN=64->256->64,
    dropout=0.1, post-norm
 -> flatten all eleven outputs [B,704]
 -> Linear(704,256) + ELU
 -> one-layer GRU(input=256, hidden=256)
```

This is full self-attention: all eleven tokens participate as Q, K, and V. It
is not state-Q/visual-KV cross-attention and does not average visual tokens.

The GRU owns one hidden vector per environment. Continuing environments carry
it across rollout chunks; only done/reset environments clear it. Gradients are
truncated at the 24-step rollout boundary even though the detached hidden
value continues into the next chunk.

### Estimation heads and 51-D latent

```text
GRU 256 -> velocity 3
        -> left-hand terrain 8
        -> right-hand terrain 8
        -> left-foot terrain 8
        -> right-foot terrain 8
        -> dynamic mean 16
        -> dynamic log-variance 16
```

Training samples `z = mu + exp(0.5*logvar)*epsilon`; deterministic collection
uses `z=mu`. The heads emit 67 parameters, but the deployable latent contains
only velocity, four 8-D terrain values, and one 16-D dynamic sample/mean:
`3+8+8+8+8+16=51`. Log-variance is used only by sampling and KL.
`dynamic_logvar` is intentionally not clamped: this matches the humanoid
reference, while KL, GradScaler scale, and skipped non-finite visual steps are
logged as the numerical safeguards and diagnostics. Logs also expose
log-variance mean/min/max, maximum implied standard deviation, and maximum
absolute dynamic mean so an unsafe trend is visible before adding a clamp.

### Decoders

Each limb independently uses:

```text
8 -> Linear 64 + ELU -> Linear 128 + ELU -> Linear 30 -> [10,3]
```

Only the 16-D dynamic latent predicts future proprioception:

```text
16 -> Linear 64 + ELU
   -> Linear 128 + ELU
   -> Linear 85 + ELU
   -> Linear 85
```

The ELU after the first 85-D projection is part of data contract
`cr1_field_normalized_proprio_rung_weighted_dynamic16_future_elu_v2`.
Pre-v2 visual checkpoints must not resume this architecture.

## Losses

```text
L = 2 L_velocity + 2 L_hand + 2 L_foot + 5 L_future + 1 L_KL
```

- velocity: MSE against raw body-frame root linear velocity in `m/s`;
- hand: average of independent left/right hand map losses;
- foot: average of independent left/right foot map losses;
- map weighting: exact rung hit `2`, other valid hit `1`, invalid `0`, divided
  by the sum of weights;
- future: elementwise MSE over the normalized next 85-D frame, excluding done
  transitions;
- KL: standard diagonal-Gaussian KL summed over 16 dimensions and averaged
  over valid samples.

Logged diagnostics include per-limb map errors, hand/foot rung and non-rung
errors, velocity zero-prediction baselines, per-axis velocity skill, and normal
versus zero/shuffled-depth ablations.

## Visual-only runner and optimizer

Collection is driven exclusively by the deterministic frozen actor. The
visual encoder runs in evaluation mode without gradients to advance its GRU.
Storage preserves `[T,B,...]` sequences, the rollout-start hidden state, dones,
and future-valid masks.

Optimization runs only over TransformerVision parameters:

- Adam, default learning rate `1e-3`;
- gradient norm clipped to `1`;
- environment mini-batches retain all 24 time steps;
- default large run `4096` environments and `256` environments per mini-batch
  produces 16 optimizer steps per collected iteration;
- Transformer dropout and stochastic dynamic sampling are active during the
  update;
- frozen actor parameters are snapshotted and required to have exactly zero
  change.

## Upward asymmetric PPO connection

The opt-in PPO task removes the old 660-D policy observation. Its action input
is exactly:

```text
current normalized proprio frame 85 + sampled TransformerVision latent 51
                                                              = 136
```

The actor MLP is `136 -> 512 -> 256 -> 128 -> 25` with ELU hidden
activations. A state-independent log-standard-deviation parameter defines the
training action distribution. The original July-geometry visual PPO task has
a separate `631 -> 512 -> 256 -> 128 -> 1` privileged critic. The independent
ThinRails task keeps the actor unchanged and uses a
`780 -> 512 -> 256 -> 128 -> 1` critic composed of:

```text
established privileged state       631
current joint velocity x 0.1        25
ladder phi/spacing/length/width       4
clean limb relative maps 4x10x3     120
---------------------------------------
total                               780
```

Rollout and update follow the recurrent `humanoid_vision` boundary:

- one GRU hidden vector is carried for every environment and only a done/reset
  environment clears its own vector;
- rollout samples the 16-D dynamic latent and stores the exact
  `[current proprio, latent]` used for the action;
- PPO reuses that stored 136-D input, so its probability ratio cannot change
  because of a second GRU pass or a newly sampled latent;
- PPO updates only the action MLP, action log standard deviation, and critic;
- a separate auxiliary optimizer replays complete 24-step sequences from the
  stored rollout-start GRU state, applies done-aligned resets, and updates only
  TransformerVision with the same `(2,2,2,5,1)` objective;
- the auxiliary schedule is five passes over four recurrent environment
  groups, exactly 20 optimizer steps per iteration; 256-environment physical
  micro-batches accumulate gradients inside each logical group;
- the hidden value carried into the next rollout is detached, so gradients are
  truncated at the rollout boundary but temporal state is not discarded.

The ThinRails scratch task now uses the humanoid reference's per-mini-batch KL
adaptation with Ladder-specific bounds. PPO starts at `5e-4`; KL above `0.02`
divides the rate by `1.5`, KL between zero and `0.005` multiplies it by `1.5`,
and the final range is `[1e-5, 1e-3]`. The separate visual optimizer remains
fixed at `5e-4`. The `1e-3` upper bound replaces both unsafe extremes already
observed in Ladder: the unchanged reference rule reached `1e-2` and failed
near iteration 41, while a fixed `5e-4` long run could not reduce its step size
when policy KL first expanded near iteration 2033.

Both optimizers use CUDA autocast and `GradScaler`, with the scaler state saved
in complete PPO checkpoints. Actor/critic and visual gradients are unscaled
before the common `0.1` norm limit. Logs expose the maximum unscaled gradient
norm, non-finite gradient steps skipped by the scaler, and current scaler
scale. Exploration keeps the existing entropy coefficient `0.005`, initial
standard deviation `1.0`, and minimum standard deviation `0.2`.

The new ThinRails runner clips actions sent to the environment to `[-10,10]`
and applies a final `[-100,100]` guard only to actual network inputs: current
and historical proprioception, processed depth, and privileged critic input.
Like the humanoid reference, it does not apply another global clip after the
85-D current frame and 51-D latent have been concatenated.
Auxiliary relative-height targets, valid masks, hit classes, velocity targets,
and future-proprioception labels are not modified by that wrapper. The 85-D
proprioceptive frame is already field-normalized and clipped; processed depth
is already bounded to `[0,2]`. Limb maps remain relative heights in meters and
normally occupy a small range, so they require no additional normalization
before entering the critic. The newly added critic joint velocity is scaled by
`0.1`; the explicit three-dimensional velocity estimate
entering the actor latent is independently clipped to `[-10,10]`, following
the humanoid visual estimator.

The earlier July-geometry PPO task still requires the accepted visual-only
checkpoint for a fresh run. The independent ThinRails task instead initializes
the actor, critic, and visual encoder from scratch and rejects a separate
visual-pretraining checkpoint. Complete PPO checkpoints contain all three
networks and both optimizer states and can resume normally.

For ThinRails, PhysX, Warp, auxiliary limb maps, and critic limb maps all use
the canonical `THIN_RAIL_WOOD_STEP_BOX_SPECS`. The established visual-only and
July-geometry PPO tasks retain the standard box tuple, so previous checkpoints
and inspection commands remain reproducible.

The separate ThinRails preflight task adds the established 660-D policy group
only so the trained `model_8498.pt` actor can move the robot during visual
inspection. It runs no optimizer and is not the PPO training environment. Its
geometry-only dashboard shows the processed Warp depth and clean target maps;
untrained predictions are deliberately hidden.

## Active randomization while the frozen actor collects data

The TransformerVision task inherits the existing nonvisual training
environment. Active perturbations include:

- ladder: 30% nominal; otherwise inclination delta `+/-1 deg` and rung-spacing
  delta `+/-0.01 m`;
- reset reference pose: position `+/-0.01 m`, RPY `+/-0.05 rad`, joint
  position `+/-0.05 rad`;
- startup link mass scale `[0.9,1.1]`;
- startup torso COM offset `+/-0.02 m` on each axis;
- startup default-joint offset `+/-0.01 rad`;
- static friction `[0.3,7.2]`, dynamic friction `[0.3,3.6]`, restitution
  `[0,0.2]`;
- root velocity pushes every `5-9 s`: linear X/Y `+/-0.08 m/s`, Z
  `+/-0.03 m/s`, angular XYZ `+/-0.10 rad/s`;
- frozen actor's original 660-D observation noise remains active;
- visual proprioception has no extra ObservationManager noise;
- depth has the dedicated masking/noise/dropout and camera-pose randomization
  described above.

The independent ThinRails PPO task additionally uses centered depth, the
input-only proprioceptive noise listed above, FOV `+/-1 deg`, a per-environment
`0..2` control-step action delay, normalized joint-position noise `+/-0.01`,
and X2-equivalent implicit-actuator gains. A shared motor-strength factor in
`[0.8,1.2]` multiplies both P and D while independent Kp/Kd factors use
`[0.9,1.1]`; these are resampled on reset and every 8 seconds. The old
visual-only collector does not inherit these changes.

## Current-action semantics in future prediction

At time `t`, visual history contains previous action `a_(t-1)`. The frozen
actor then produces `a_t`, and the returned next 85-D target contains `a_t`.
The future decoder input remains only the sampled 16-D dynamic latent:

```text
visual + proprio history -> dynamic z 16 -> future decoder -> next 85-D frame
                                                               [includes a_t]
```

This exactly follows the current `rslRL-317-rq-humanoid_ssr_pure` structure:
its `prop_decoder` receives only `encodings["implicit"]`, while the future
observation assembled after the environment step contains `self.actions`, the
action just executed. The current action must therefore remain a label field,
not an extra decoder input. Ladder's frozen actor sees a richer 660-D input, so
the action portion may be intrinsically harder for this smaller visual encoder
to predict; the per-field future metrics must expose that error rather than
silently changing the SSR objective.

## Main implementation files

- `ladder/tasks/ladder_climb/config/cr1/wood_step_transformer_vision_env_cfg.py`
- `ladder/tasks/ladder_climb/config/cr1/wood_step_transformer_vision_thin_rails_ppo_env_cfg.py`
- `ladder/tasks/ladder_climb/config/cr1/wood_step_transformer_vision_data_cfg.py`
- `ladder/tasks/ladder_climb/mdp/observations.py`
- `ladder/utils/warp_camera.py`
- `ladder/utils/visual_depth.py`
- `ladder/utils/limb_heightmaps.py`
- `ladder/rsl_rl/modules/transformer_vision.py`
- `ladder/rsl_rl/storage/transformer_vision_storage.py`
- `ladder/rsl_rl/algorithms/vision_pretrain.py`
- `ladder/rsl_rl/runners/frozen_policy_vision_runner.py`
- `ladder/rsl_rl/modules/transformer_vision_actor.py`
- `ladder/rsl_rl/storage/transformer_vision_ppo_storage.py`
- `ladder/rsl_rl/algorithms/transformer_vision_ppo.py`
- `ladder/rsl_rl/runners/transformer_vision_ppo_wrapper.py`
- `scripts/rsl_rl/train_transformer_vision.py`
- `scripts/rsl_rl/train.py`
