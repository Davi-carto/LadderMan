# Reset-dependent jitter diagnosis — 2026-09-09

## Confirmed primary finding

`play.py --disable_randomization` disables the listed material/mass/pose/motor
events and reset perturbations, but does not zero the action control-delay range.
The retained blind SmoothOutput task inherits `min_control_delay=0` and
`max_control_delay=2`. The action term samples an integer delay on every reset;
at step_dt=0.02 this means 0, 20 or 40 ms, retained until the next reset.

Runtime audit of the actual play configuration found:

- No active randomization events.
- No observation-manager noise in either policy or critic.
- Deterministic MLP inference, with Identity observation normalization.
- Implicit actuators, and an action-level 0..2-step delay buffer.
- Five consecutive fixed-frame resets had exactly identical root/joint
  state and actor input tensors, but different sampled delays.

## Controlled experiment

Used the actual fast-mode play setup, 9 environments, frozen checkpoints,
fixed starts 0, 1500 and 1800, and assigned delays 0/1/2 to three environments
each. Repeated frame 0 after the other starts. Assignments rotated between
frame trials. Each rollout lasted 350 policy steps (7 s). Statistics exclude
the first 25 steps (0.5 s) and any samples after first termination. No case
terminated during its 350-step test: 72 environment/rollout observations.
This is a short-window comparison, not a complete-motion success test.

Mean of per-environment, 12-lower-joint acceleration RMS (rad/s²), computed
as control-rate finite differences of actual joint velocities:

| Start frame | Delay (ms) | model_9999 | model_11999 |
|---|---:|---:|---:|
| 0 | 0 | 4.890 | 2.802 |
| 0 | 20 | 25.000 | 8.637 |
| 0 | 40 | 33.120 | 12.689 |
| 1500 | 0 | 9.149 | 2.983 |
| 1500 | 20 | 27.623 | 12.430 |
| 1500 | 40 | 34.119 | 12.635 |
| 1800 | 0 | 11.940 | 7.750 |
| 1800 | 20 | 22.031 | 10.983 |
| 1800 | 40 | 27.542 | 12.601 |

Frame-0 joint position error RMS stayed near 0.05 rad across the delay groups,
despite much larger acceleration at longer delays. Good average position
tracking and surviving a rollout therefore do not imply smooth motion.

This controlled comparison supports the user's impression that model_11999
handles jitter better. Earlier single-rollout model rankings did not record
or control this remaining randomized delay, so they cannot establish that
model_9999 is intrinsically smoother.

The delay is applied in simulation control steps; slow GUI wall-clock execution
is a different issue and does not create this 0/20/40-ms sampling.

## Secondary finding: reset first-step transients

A separate zero-delay probe used repeated starts at frame 0, interleaved with
40-step rollouts from frames 1500 and 1800. Checked both cached state and the
PhysX articulation view directly. At repeated frame-0 resets the following
matched exactly: root pose/velocity, joint pose/velocity, link poses/velocities,
actor observations, and first actor output. The first applied joint targets
also matched exactly.

Nevertheless, the first physical step differed from the initial trial, with
maximum differences over the 9 environments/all joints up to 0.0567 rad in
joint position and 2.95 rad/s in joint velocity. Thus displayed/reset state
alone does not establish identical first-step physics. Contact reconstruction,
solver state, or another internal physics state are candidates, not confirmed
individual causes. These differences should not be dismissed as proven tiny
floating-point roundoff.

Stored processed/drive target buffers still contained previous values just
after reset, but the first action replaced them with identical new targets
before integration. Therefore their presence alone is not evidence that old
actions were applied. Contact sensor data queried immediately after reset also
reflected differences; that readout alone does not prove a physical residual
force was reapplied.

The main experiment already excludes 0.5 s of startup and still shows a strong
delay effect. Both findings can coexist: initial contact transients and weak
closed-loop tolerance to delayed commands.

## Implications

1. Make playback's disable-randomization option cover control delay and display
   the sampled delay; this would fix the misleading evaluation condition.
2. Compare policies at explicitly fixed 0/20/40-ms delay before deciding on
   further reward/controller changes. Keep realistic training randomization
   as a separate design choice.
3. Continue analyzing residual zero-delay startup/contact transients if they
   remain visibly troublesome; current evidence does not justify attributing
   all jitter to a single stale action buffer or to the reference motion.

No production scripts, task configurations, rewards or model weights were
modified for this diagnosis. Only isolated diagnostic processes and artifacts
were created. The testing script is `/tmp/ladder_reset_delay_probe.py`.

## Evidence and source locations

- `audit.json`: runtime configuration and repeated reset audit (most recent probe).
- `summary.json`: 72 per-environment rollout summaries.
- `model_*_frame*_trial*.npz`: lower-joint traces, initial states/observations,
  assigned delay and naturally sampled delay before the diagnostic override.
- `startup_summary.json`, `startup_trial*.npz`: zero-delay first-step probe.
- `Ladder/scripts/rsl_rl/play.py:1552`: disable-randomization implementation.
- `Ladder/source/ladder/ladder/tasks/ladder_climb/mdp/actions.py:88`: delayed
  action processing and rate limiting; line 109 resets and re-samples delay.
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_transformer_vision_thin_rails_ppo_env_cfg.py:157`:
  inherited control-delay configuration.
- Checked IsaacLab v2.1 reference reset/action/history utilities and the actual
  installed IsaacLab 2.3.2 implementations of ManagerBasedEnv,
  ManagerBasedRLEnv, Articulation, ObservationManager and SimulationContext;
  installed RSL-RL MLPModel defaults to deterministic inference.
