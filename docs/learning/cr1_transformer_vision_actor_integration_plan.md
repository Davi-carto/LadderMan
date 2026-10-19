# CR1 TransformerVision Actor Integration Plan

## Status and isolation

The upward-climbing network and recurrent asymmetric-PPO plumbing in this plan
are now implemented. The accepted visual-only checkpoint is preserved at
`2026-08-17_16-33-26_fresh_5000/model_5000.pt`, but the selected independent
ThinRails PPO task starts actor, critic, and TransformerVision from scratch.

The visual-pretraining task, its 85-D builder/history, Warp renderer, auxiliary
losses, training script, and diagnostic script remain unchanged. PPO uses a
new task configuration and new Ladder-owned RSL-RL files, so visual-only
checkpoint inspection remains reproducible.

All final modules, storage, algorithms, and runners remain under
`ladder/rsl_rl/{modules,storage,algorithms,runners}` and do not use `SSR` in
their new names. The humanoid project is a reference, not a runtime dependency.
The method is visual auxiliary learning plus asymmetric PPO, not distillation.

## Accepted actor input boundary

The final actor receives only one current proprioceptive frame, not the old
660-D observation with five-frame histories:

```text
current body state 84:
    body angular velocity 3
    root orientation 6
    joint position 25
    joint velocity 25
    previous action 25

effective climb command 1:  down=-1, hold=0, up=+1
--------------------------------------------------
current o_t                                      85

TransformerVision latent                         51
--------------------------------------------------
actor input                                      136
```

The visual encoder still receives ten consecutive 85-D frames and two depth
frames, and its GRU retains longer context. That temporal processing belongs
inside TransformerVision and is not duplicated at the action MLP input.

The current visual-only run has the same 85-D shape, with its final direction
field fixed to `+1`. Its weights remain structurally loadable. During commanded
policy training, the same scalar position becomes the effective command and
TransformerVision is fine-tuned on `-1/0/+1` trajectories.

The mature 660-D `model_6999.pt` policy remains the trajectory collector for
visual pretraining. It cannot be transferred as an equivalent final actor:
its first layer and information contract differ, it contains observation
histories/reference inputs, and it was never trained to respond to the three
commands. The final 136-D commanded visual actor therefore starts as a new
policy.

The critic remains asymmetric. The selected ThinRails task uses the established
631-D privileged state plus current joint velocity 25, clean four-limb
relative height maps 120, and ladder parameters 4, for 780 dimensions.
Privileged terrain labels and current ladder parameters never enter the actor
directly.

The current joint-training safeguards are based on the X2 visual PPO while
retaining a Ladder-specific learning-rate ceiling: PPO uses KL-adaptive
`5e-4` within `[1e-5,1e-3]`, the visual auxiliary optimizer stays at `5e-4`,
both use CUDA autocast plus `GradScaler`, and both unscaled gradient norms are
limited to `0.1`. Environment actions are clipped to `[-10,10]`; network
inputs receive a final `[-100,100]` guard; and the explicit velocity latent is
clipped to `[-10,10]`. Relative limb maps remain meter-valued data rather than
being rescaled merely because they were added to the critic.

## What X2 does when the command becomes zero

The X2 visual reference uses current proprioception plus its velocity command
as the actor observation; the long observation history is consumed separately
by the encoder. Its command configuration uses a 10-second resampling period,
15% zero-command probability, and 0.1 linear/angular zero thresholds.

It avoids an unsafe instantaneous stop in two ways:

1. At a mid-episode command resample on a hard terrain, a robot with a nonzero
   sampled motion command is not allowed to turn that sample into zero. The
   zero choice is rejected and motion stays nonzero (the sampled nonzero value
   need not equal the previous command). Zero is allowed at reset, when already
   stopped, or on simpler terrain.
2. When a zero command is accepted, `GaitManager` does not freeze a swing foot
   immediately. It marks that foot as stopping, lets its current phase advance
   to the stance center, and only then holds it in stance. This avoids a sudden
   foot-phase discontinuity.

For a zero command, X2 also sets target heading to the current heading, so
zero translation does not accidentally request a turn. Tracking rewards then
request zero velocity. The base project contains a stand-still posture term,
although the main `x2_vis_config.py` reward list does not enable that term;
some sparse variants do.

This reference mainly avoids issuing a stop at a bad time. It is not a general
emergency-stop controller for an arbitrary asynchronous user command.

## Ladder safe command state machine

For ladder climbing, `0` means **finish the current unsafe limb transfer and
hold at the next stable contact state**. It must not mean zero motor action,
reset, or instant freezing of the reference phase.

Maintain two per-environment values:

```text
requested_command: latest external request in {-1,0,+1}
effective_command: command currently given to actor in {-1,0,+1}
```

Transitions:

- `hold -> up/down`: begin only from an accepted stable support state;
- `up/down -> hold`: if already stable, latch hold; otherwise keep the previous
  effective direction and enter `stop_pending`;
- `stop_pending -> hold`: switch to zero only when the current limb transfer
  has completed and the stable-support test passes;
- `up <-> down`: never reverse directly in a swing/transfer phase; pass through
  a stable hold first.

The actor receives `effective_command`, because that describes what it must do
on the current control step. `requested_command` and `stop_pending` remain in
the command manager and diagnostics, not hidden privileged actor hints.

Initial stable-stop candidates use the motion's four-limb contact schedule:
the current training motion contains 408 frames where all four reference
contact indicators are active. A candidate becomes an actual safe stop only
after checking current physical contact and limb/base motion as well. Exact
force/velocity thresholds must be selected in a standalone diagnostic before
training.

Once hold is latched:

- stop advancing the ladder reference at that accepted safe frame;
- do not clear the GRU and do not reset the environment;
- reward low body/limb velocity, maintained current rung contacts, low slip,
  smooth actions, and bounded effort;
- do not pull the robot toward a generic standing joint pose, because that can
  detach hands or feet from the ladder.

A normal zero command and an emergency hardware stop are different mechanisms.
The latter belongs to a lower-level safety controller and is not learned PPO
behavior.

The command used to produce action `a_t` must also be the command field in that
transition's future-observation label. A newly requested external command is
applied before the next actor decision; the future decoder is not asked to
predict an unpredictable command change. This matches the timing intent of the
X2 reference, which assembles its future target before command resampling.

## Up and down data

The present motion and visual-only pretraining cover upward climbing. Downward
control is not obtained merely by setting the scalar to `-1`. The accepted
decision is to collect a separate human downward-climbing motion rather than
reverse the upward trajectory. Before commanded PPO, process and validate this
new recording with the same authoritative motion contract:

- match CR1 joint/body names, coordinate conventions, units, and control rate;
- reconstruct root/body/joint positions, orientations, and velocities without
  discontinuities;
- create and inspect four-limb `contact_indicator` and `contact_rung_id` labels;
- add smooth start/terminal hold segments and identify all-four-contact safe
  hold intervals;
- retarget to the exact ladder geometry, then regenerate collision-safe reset
  masks;
- replay the complete downward motion with contact/force/penetration diagnostics
  before using it for rewards, resets, or the `-1` command.

Training command transitions should initially occur only at accepted stable
frames. A later curriculum may request stop at arbitrary motion phases while
the state machine delays the effective transition safely.

## Integration gates

Current status:

- Gate 0: passed for the upward `model_5000.pt` checkpoint;
- Gate 1: pending; safe hold and down-to-hold require the command lifecycle
  diagnostic and the separately recorded downward motion;
- Gate 2: passed in pure PyTorch tests; the selected scratch task deliberately
  does not load the accepted visual-only checkpoint;
- Gate 3: upward native-checkpoint play is wired; three-command acceptance is
  pending Gate 1 data and command state machine;
- Gate 4: passed with an eight-environment real Warp rollout, one PPO update,
  one recurrent auxiliary update, checkpoint save, and checkpoint resume;
- Gate 5: only the one-iteration upward smoke has run. Long upward PPO and all
  three-command training remain user-run acceptance stages.

### Gate 0 - Accept the visual-only checkpoint

Check a complete upward climb, resets, four target/predicted limb maps,
rung/non-rung errors, velocity skill against zero, per-field future error, and
normal/zero/shuffled-depth ablations. Select the accepted checkpoint rather
than automatically choosing the final iteration.

### Gate 1 - Command lifecycle diagnostic

In a separate task, visualize requested/effective command, pending state,
reference phase, desired/actual four-limb contacts, and stop-frame selection.
Test up-to-hold and down-to-hold without an actor/PPO implementation. Accept
only when no swing limb is frozen in mid-transfer and hold does not remove a
support contact.

### Gate 2 - Pure combined network

Build a Ladder-owned recurrent actor-critic:

```text
10 x 85 proprio + 2 x 60 x 80 depth -> TransformerVision -> latent 51
[current o_t 85, latent 51]          -> actor MLP        -> actions 25
privileged current observation       -> critic MLP       -> value 1
```

For the selected independent ThinRails task, randomly initialize all three
networks. Test all shapes, deterministic dynamic decoding, per-environment GRU
reset, gradients, and checkpoint reload without an environment or PPO update.
The original VAE encoder remains available only as a named legacy comparison
path for accepted checkpoints.

### Gate 3 - No-update environment integration

Run all three effective commands with every network parameter frozen. Verify
live depth/latent/command changes, complete episodes, delayed safe stopping,
holding, restarting, direction changes through hold, reset behavior, and
checkpoint replay. No optimizer step is allowed.

### Gate 4 - RSL-RL recurrent asymmetric PPO plumbing

Store the exact `[o_t,latent]` actor input used during rollout. PPO recomputes
action probabilities from that stored 136-D tensor and must not rerun the
recurrent visual encoder inside the ratio calculation.

Each iteration uses two optimizer phases:

1. PPO updates actor, action standard deviation, and privileged critic. The
   first baseline does not backpropagate PPO through TransformerVision.
2. The auxiliary phase replays temporal visual data and updates the CNN,
   proprio MLP, Transformer, GRU, estimator heads, and decoders with the
   accepted velocity/hand/foot/future `(2,2,2,5)` objective. The active
   deterministic dynamic branch has no KL term.

Auxiliary losses are not rewards or advantages. Accept one small real Warp
rollout/update with finite losses, correct recurrent boundaries, intended-only
parameter changes, and checkpoint reload.

Implemented runtime order:

1. rollout runs TransformerVision once with the persistent per-environment GRU
   state, computes the deterministic dynamic latent, and stores the exact
   136-D actor input;
2. after `env.step`, only done environments clear their GRU columns;
3. PPO updates the action MLP, log standard deviation, and privileged critic
   from the stored 136-D input, without rerunning TransformerVision;
4. the auxiliary phase selects environment mini-batches but retains all 24
   ordered time steps, the stored rollout-start hidden state, and resets before
   the observation following each done;
5. the auxiliary optimizer alone updates the CNN, proprio MLP, Transformer,
   GRU, estimator heads, and decoders. The detached live hidden state continues
   into the next rollout, matching the reference project's truncated recurrent
   rollout behavior.

The new runtime files are:

- `ladder/rsl_rl/modules/transformer_vision_actor.py`
- `ladder/rsl_rl/storage/transformer_vision_ppo_storage.py`
- `ladder/rsl_rl/algorithms/transformer_vision_ppo.py`
- `ladder/rsl_rl/runners/transformer_vision_ppo_wrapper.py`
- `ladder/tasks/ladder_climb/config/cr1/wood_step_transformer_vision_ppo_env_cfg.py`

### Gate 5 - Short and full commanded training

First run a short three-command training and verify that the actor learns safe
hold/restart behavior without destroying contact stability. Then run the full
PPO plus auxiliary schedule and compare normal vision against zero latent,
zero depth, shuffled depth, and a nonvisual command-policy baseline on nominal
and randomized ladders.

Deployment uses the same preprocessing/histories, dynamic mean, persistent GRU
state, safe command state machine, and per-environment reset. Auxiliary
decoders and privileged targets are not needed to produce actions.

## Primary references

- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/X2/x2_vis_config.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/base/legged_robot.py`
- `humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/utils/gait_manager.py`
- `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/modules/actor_critic.py`
- `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure/rslrl-317/rsl_rl/algorithms/ppo_ce.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/commands.py`
- `Ladder/source/ladder/ladder/rsl_rl/modules/transformer_vision.py`
