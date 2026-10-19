# CR1 visual PPO observation and reward comparison

Date: 2026-08-18

This note compares the accepted 5000-iteration visual-pretraining environment,
the latest trained nonvisual environment, and the visual-latent PPO
environments. The decision at the end of this review has now been implemented
as a separate task.

## Compared runs

1. Accepted visual-only pretraining:
   - visual checkpoint: `logs/rsl_rl/cr1_transformer_vision_pretrain/2026-08-17_16-33-26_fresh_5000/model_5000.pt`
   - frozen actor: `logs/rsl_rl/cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs/2026-07-31_14-01-06_deploy_obs_from_scratch_to7000/model_6999.pt`
   - environment: `CR1ReferenceForceLimitedStrictFrontSideDeployObsEnvCfg` plus Warp visual inputs and auxiliary labels.
2. Latest trained nonvisual variant found in the log tree:
   - `logs/rsl_rl/cr1_wood_step_reference_rails_protected_per_body_thin_rails_safe_frame0/2026-08-10_11-32-45_from_scratch_extend_to8500`
   - environment: `CR1ReferenceRailsProtectedPerBodyThinRailsSafeFrame0EnvCfg`.
3. Current visual PPO smoke configuration:
   - `logs/rsl_rl/cr1_transformer_vision_ppo_up/2026-08-18_10-11-46`
   - environment: `CR1TransformerVisionPPOEnvCfg`, which still inherits the July 31 strict-front/side task rather than the August thin-rail task.

“Latest” here means the most recent nonvisual log/config. It does not by itself prove that this checkpoint has the best behavior; playback/contact statistics still decide that.

## Actor observation comparison

### July 31 and August 10 nonvisual actor: identical 660-D contract

Both saved environments use the same nine terms:

| Term | Frames | Dimension |
|---|---:|---:|
| reference joint position | 5 | 125 |
| reference joint velocity | 5 | 125 |
| projected gravity | 5 | 15 |
| root angular velocity | 5 | 15 |
| joint position relative to default | 5 | 125 |
| joint velocity relative/default convention | 5 | 125 |
| previous action | 5 | 125 |
| motion phase | 1 | 1 |
| desired four-limb contact indicator | 1 | 4 |
| total |  | 660 |

The August task changed the ladder and rewards, not the actor observation contract.

### Current visual PPO actor: 136-D

The new actor does not consume the old 660-D group. The wrapper takes the final frame from the visual proprioceptive history:

| Current 85-D proprio field | Dimension |
|---|---:|
| body-frame root angular velocity, scaled by 0.2 | 3 |
| root rotation matrix first two columns | 6 |
| joint position error from default, divided by action scale | 25 |
| joint velocity, scaled by 0.1 | 25 |
| previous action | 25 |
| climb-direction command, currently fixed to +1 | 1 |
| total current proprio | 85 |

The actor concatenates this with the 51-D visual latent:

```text
current proprio 85 + visual latent 51 = actor input 136
```

The 51-D latent is velocity 3, four limb-terrain latents of 8 dimensions each, and dynamic latent 16.

Important difference: the current 136-D actor has no explicit reference joint position, reference joint velocity, motion phase, or desired contact phase. The visual GRU may infer progress from motion history, but its auxiliary objectives do not explicitly supervise reference phase. Meanwhile the PPO rewards are still reference-motion tracking rewards. This is a task-design decision, not a harmless dimensional reduction.

### Actor decision needed before long PPO training

Two coherent designs exist:

1. Reference-driven climbing, closest to the successful nonvisual task:
   - keep current proprio 85 and visual latent 51;
   - add current reference joint position 25, current reference joint velocity 25, motion phase 1, and desired contact indicator 4;
   - no five-frame actor history is needed because the visual encoder/GRU owns temporal history;
   - resulting actor input would be 191-D.
2. Command-only autonomous climbing:
   - retain 136-D and eventually allow command values +1, 0, and -1;
   - do not expect strict imitation rewards to be an ideal final objective, because a direction scalar alone does not state which hand/foot phase should occur next;
   - redesign motion progression, hold behavior, and rewards together rather than silently dropping the reference signal.

For the next upward-only PPO stage, reference-driven 191-D is the lower-risk option because it preserves the information used by the successful nonvisual policy while removing only redundant five-frame actor history. Command-only control should be a separate accepted stage.

## Current critic observation: 631-D

The July 31 nonvisual task, August 10 thin-rail task, and current visual PPO task all use the same `CriticObservationCfg`. The current critic has no visual depth and no visual latent.

| Privileged term | Meaning | Dimension |
|---|---|---:|
| reference body positions in reference-root frame | 16 tracked bodies x 3 | 48 |
| reference joint positions | desired joints | 25 |
| reference anchor position error | desired root translation | 3 |
| reference anchor orientation error | 6-D rotation | 6 |
| motion phase | normalized reference time | 1 |
| desired contact indicator | desired contact for four limbs | 4 |
| root angular velocity | current body-frame angular velocity | 3 |
| root orientation | 6-D world orientation | 6 |
| joint-position history | 9 x 25 | 225 |
| action history | 3 x 25 | 75 |
| root relative to ladder | axial/lateral/normal or configured frame | 3 |
| current tracked-body positions | 16 x 3 | 48 |
| current tracked-body linear velocities | 16 x 3 | 48 |
| root linear velocity | current body-frame velocity | 3 |
| tracked-body position errors | current minus reference | 48 |
| tracked-body velocity errors | current minus reference | 48 |
| joint-position errors | current minus reference | 25 |
| four-limb target-position errors | 4 x 3 | 12 |
| total |  | 631 |

The critic is already substantially more privileged than the actor, but it lacks several quantities directly relevant to the new visual/rail task.

## Recommended critic additions

### Strongly recommended for the first visual PPO task

1. True four-limb relative height maps: `4 x 10 x 3 = 120`.
   - The actor receives estimated terrain information through its 51-D latent.
   - The critic should receive the clean simulator height maps, not noisy depth.
   - This follows the X2 visual reference: its privileged observation explicitly contains the true body/foot terrain height maps that the visual network reconstructs.
2. Current joint velocity: 25.
   - The current actor observes it, but the critic does not explicitly receive it.
   - Selected body linear velocities and joint-position history are not an exact replacement for all joint velocities.
3. Sampled ladder parameters: 4.
   - inclination, rung spacing, rung length, and rung width;
   - especially important because inclination and spacing are randomized.

These three additions would change the critic from 631-D to 780-D.

### Recommended when adopting the latest rail-contact rewards

Add compact clean contact state rather than raw sensor histories:

- four physical limb-contact flags;
- four target-rung-contact flags or normalized contacted-rung IDs;
- four limb contact-force vectors or equivalent top/front/underside components;
- left/right whole-body rail-contact flags and maximum rail-force values.

Exact dimension should be fixed only after choosing the latest reward set. The reason is direct: the latest value target contains missing-contact, surface-direction, force-excess, and whole-body rail penalties, while the current critic sees desired contacts and position errors but not the actual contact/force event that generated those rewards.

### Optional later, not required in the first merge

- applied joint torques, because torque penalty contributes to reward;
- randomized mass, COM, and friction values, if value learning remains noisy;
- target rung IDs, if the compact contact representation does not already contain them.

Do not add noisy depth to the critic. Clean local height maps and exact ladder/contact state are the appropriate privileged equivalents.

## PPO reward comparison

Adding TransformerVision did not change the PPO reward terms. Visual reconstruction, velocity, future-proprioception, and KL losses are auxiliary neural-network losses and are not environment rewards.

### Shared core rewards

Both the July and August tasks retain the same main weights:

- global anchor position: `+0.8`;
- global anchor orientation: `+0.4`;
- asymmetric tracked-body position: `+1.0`;
- tracked-body linear velocity: `+0.6`;
- upper joint tracking: `+0.5`;
- lower joint tracking: `+1.0`;
- left/right hand target position: `+0.5` each;
- foot target position: `+1.0`;
- left/right hand upward support force: `+0.3` each;
- action rate: `-0.1`;
- joint torque: `-1e-5`;
- joint limit: `-10.0`;
- undesired body contact: `-0.1`.

Both also retain hand front/underside/ground/missing-contact penalties and hand/foot force-excess penalties. Several older broad hand-contact terms remain configured with zero weight.

### August thin-rail changes relative to the July baseline

- ladder front rails are reduced to one third of their previous depth;
- old hand-only side-board contact and force terms are removed;
- four whole-robot rail terms are added:
  - left/right rail binary contact: `-1.0` each;
  - left/right rail force excess: `-0.25` each;
- hand underside binary penalty becomes `-0.6` instead of `-0.4`;
- hand underside force-excess penalty becomes `-0.1` instead of `-0.05`;
- underside soft force limit becomes 10 N instead of 20 N;
- missing required hand contact becomes `-0.4` instead of `-0.2`, is tied to the original target rung, and does not require the hand first to be near the target;
- acceptable target-top hand force rises from 180 N to 250 N;
- upward support-force saturation rises from 50 N to 100 N;
- top-normal/front classification fraction becomes 0.35 instead of the older 0.5 convention;
- fixed frame-0 start x range changes from `[-0.10, 0.10]` to `[-0.10, 0.06]`.

The current visual PPO task still uses the July reward/scene values, not these August changes.

## Geometry compatibility resolution

The accepted visual-only task continues to use `WOOD_STEP_BOX_SPECS`. The new
independent August task uses canonical `THIN_RAIL_WOOD_STEP_BOX_SPECS` for all
geometry consumers rather than changing the old task in place.

The same selected tuple is now passed to:

- the PhysX ladder scene;
- `CR1WarpBatchCamera`;
- auxiliary four-limb height-map sampling/classification;
- privileged critic four-limb height maps.

The implemented task is
`Ladder-Climb-CR1-Reference-RailsProtected-PerBody-ThinRails-SafeFrame0-WarpTransformerVision-PPO-v0`.
It keeps the 136-D actor unchanged, extends the critic from 631 to 780 with the
recommended 120+25+4 fields, inherits the latest ThinRails rewards, and starts
the actor, critic, and visual network from scratch. It deliberately does not
initialize from the accepted visual-only checkpoint.

## References checked

- `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_reference_force_limited_strict_front_side_deploy_obs_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_reference_rails_protected_per_body_thin_rails_safe_frame0_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_reference_rails_protected_per_body_safe_frame0_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_reference_rails_protected_safe_frame0_env_cfg.py`
- `Ladder/source/ladder/ladder/tasks/ladder_climb/config/cr1/wood_step_transformer_vision_ppo_env_cfg.py`
- the three saved `params/env.yaml` files named above
- `/home/zhangdawei/humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317/legged_gym/envs/X2/x2_vis_config.py`
