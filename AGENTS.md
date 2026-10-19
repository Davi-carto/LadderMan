# LadderMan AGENTS.md

Project rules for coding agents working on LadderMan, an IsaacLab-based robot RL project.

## Core Conduct

- State assumptions before non-trivial coding. If the requirement is ambiguous and the difference matters, ask.
- Keep changes surgical: no speculative features, unrelated cleanup, or one-off abstractions.
- For multi-step work, give a short plan with verification checks.
- Every changed line should trace to the current task.

## Reference-First Workflow

Before implementing or changing environments, MDP terms, rewards, observations, actions, assets, robot configs, terrains, training configs, or scripts, check the closest references in this order:

1. Isaac Lab official source: `isaaclab_v2.1.0/`
2. Similar tasks: `isaaclab_v2.1.0/source/isaaclab_tasks/`
3. Existing project reference: `whole_body_tracking/`
4. BeyondMimic-style changes already present in this repo
5. The minimum LadderMan-specific implementation

When proposing or summarizing important changes, name the reference files checked.

References are for learning, not runtime coupling. New IsaacLab RL projects may study
and locally adapt code from `whole_body_tracking`, but LadderMan implementation code
must not inherit from `whole_body_tracking` classes or require `whole_body_tracking`
imports at runtime. Prefer inheritance only from IsaacLab APIs; copy or reimplement the
small reference-derived pieces inside the new project when needed so the reference repo
can be removed without breaking the project.

## IsaacLab Conventions

- Prefer IsaacLab's existing managers, config classes, MDP utilities, wrappers, math utilities, and task patterns over custom code.
- Use standard IsaacLab structure: scene/assets, commands, actions, observations, rewards, events, terminations, curriculum, gym registration, agent configs, and scripts should live where IsaacLab examples put them.
- Before creating new task folders, inspect the IsaacLab project/template layout and `isaaclab_tasks` registration patterns.
- Add custom MDP functions only when official IsaacLab or existing project functions cannot express the behavior cleanly.
- Custom tensor code must be batched over envs, device-safe, and avoid Python per-env loops unless justified.

## RL Design Rules

- Keep actor observations separate from privileged critic observations. Do not leak simulator-only state into a deployable actor.
- Treat action scale, decimation, joint limits, default pose, contact body names, contact sensors, and ladder collision geometry as high-risk changes.
- Each reward or termination change must have a clear behavioral purpose and checked sign, scale, units, `dt` handling, and reward-hacking risk.
- Add domain randomization gradually with physically plausible ranges.
- Do not diagnose a training failure as a PPO problem until configs, shapes, resets, rewards, terminations, contacts, and assets have been checked.

## Verification

Use staged checks before long training:

1. Import/config instantiation after config changes.
2. Reset/step smoke test after environment changes.
3. Observation/action/reward/done shape checks after MDP changes.
4. Short training only after shapes, rewards, resets, and terminations are sane.

If a check cannot run in the current environment, say why and record the remaining risk.

## Notes and Progress Tracking

- Write durable conceptual learning to `docs/learning/humanoid_rl_progress.md`.
- Track important implementation milestones in `docs/implementation_log.md`.
- Promote only stable, repeatedly useful lessons into this file.
- Keep this file concise; it is a working checklist, not a session transcript.
