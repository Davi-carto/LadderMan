# Humanoid RL Learning Progress

## Current Focus
- Project: LadderMan visual climbing policy
- Milestone: frozen-policy TransformerVision auxiliary-data acceptance
- Current question: do the velocity, four local height-map, and next-proprio
  targets have healthy scale and reconstruction progress before the visual
  latent is connected to the actor?
- Last updated: 2026-08-14

## Roadmap
- [x] M0: Isaac Lab manager-based RL skeleton
- [ ] M1: Motion data format and replay
- [ ] M2: MotionCommand / command manager integration
- [ ] M3: Observation, reward, termination design
- [ ] M4: PPO training loop and diagnostics
- [ ] M5: BeyondMimic-specific motion tracking reproduction
- [ ] M6: LadderMan hybrid contact tracking extension
- [ ] M7: Multi-expert training and robustness
- [ ] M8: Export/deployment understanding

## Session Log
### 2026-08-20 - Separate Anatomical Leg Length From Foot-Marker Mismatch
- A retargeted robot's persistent knee flexion does not by itself prove the
  recorded human legs are too short. Compare each mapped rigid segment first.
  For the new capture, thigh plus shank already matches CR1 within `6 mm`,
  while `Foot -> ToeBase` is `45 mm` shorter.
- The mapped endpoint and its weight can redistribute IK error through the
  whole leg. Extending only the retarget-time toe target straightened the
  knees, but worsened a left-ankle/step collision and joint acceleration.
  Morphology changes therefore need the same penetration and derivative gate
  as temporal repairs; visual posture alone is insufficient.
- Keep morphology adaptation reversible and off by default. Preserve the raw
  capture and diagnose the localized collision constraint before smoothing or
  rewriting otherwise high-quality source motion.
- When a mapped endpoint has a fixed geometric offset on the robot, compare
  both length and neutral inclination. Here the human standing Foot-to-ToeBase
  pitch was about `10 deg` steeper than CR1's ankle-to-toe geometry. Reducing
  absolute pitch by that measured offset preserves motion variation better
  than multiplying every frame's angle by an arbitrary scale.

### 2026-08-18 - Visual Latent Connected to Asymmetric PPO
- The new upward actor receives one current normalized 85-D proprioceptive
  frame and the sampled 51-D TransformerVision latent. The old 660-D frozen
  policy remains only part of visual-pretraining history and is not an input or
  teacher for the new policy.
- The humanoid reference's optimizer boundary is retained: rollout stores the
  exact combined actor input; PPO updates the actor and privileged critic from
  that fixed tensor; a separate visual optimizer replays the recurrent inputs
  and auxiliary targets. PPO therefore does not rerun the GRU or resample the
  dynamic latent inside its probability ratio.
- GRU state is carried per environment across rollout chunks. A done clears
  only that environment before its reset observation is processed. Auxiliary
  replay uses the stored chunk-start state and done-aligned sequence resets;
  gradients are truncated at the 24-step boundary.
- Pure tests, one real Warp PPO/auxiliary update, full checkpoint save, and
  full checkpoint resume passed. Hold/down commands remain a separate stage
  because the safe command lifecycle and downward human motion are not yet
  available.

### 2026-08-14 - Frozen Policy Before Visual Training
- The nonvisual actor is retained exactly as the existing RSL-RL 5 split
  checkpoint: it consumes only its established 660 deployable observations
  and produces 25 deterministic actions.  It is not a teacher and no
  distillation loss exists.  Its purpose in this stage is to provide realistic
  climbing trajectories while visual data are collected.
- TransformerVision is an independent RSL-RL-style module/algorithm/storage/
  runner path.  Its 51-D latent is deliberately absent from the frozen actor.
  The actor, visual inputs, privileged labels, and critic observations stay as
  named, separately validated TensorDict fields.
- A transition stores visual input and current velocity/map labels from
  `o_t`, then records next proprio from `o_(t+1)`.  If the action terminates
  an environment, that replacement reset frame is stored but its future loss
  is masked out; the GRU resets before the subsequent real transition.
- A no-update diagnostic may backpropagate to measure whether the visual
  losses reach the encoder, but it must leave both the frozen actor and visual
  encoder parameters unchanged.  Only after the dashboard and loss scales are
  accepted should an auxiliary-only optimizer step be added; PPO and actor
  latent integration remain later phases.

### 2026-08-14 - Local Recurrent PPO Boundary
- The first Ladder visual PPO baseline is Ladder-local and uses the installed
  RSL-RL runner's algorithm-construction hook, not a modified `site-packages`
  package. A dedicated vector wrapper flattens only the named SSR runner fields
  because the current RSL-RL NaN checker does not support nested TensorDicts.
- During rollout, PPO receives and stores the exact sampled actor input
  `[o_t,z_t]`. Its update trains only the action MLP, action standard deviation,
  and privileged critic. The temporal visual encoder and all estimator/decoder
  branches are trained only in the separate auxiliary phase, matching the
  selected SSR-style two-phase optimization boundary.
- Gate D's real Warp smoke runs one finite PPO update with the auxiliary phase
  disabled. Map/velocity/future target normalization and actual auxiliary
  optimization remain Gate E work.

### 2026-08-14 - Gate C0 Recurrent Visual Model
- The Ladder-owned pure PyTorch model now follows the chosen SSR-derived
  architecture: a 64-D state token, ten 64-D depth tokens, two self-attention
  layers, flat `704 -> 256` fusion, and a 256-D GRU. It does not import the
  reference project at runtime and does not use teacher-student distillation.
- Its seven heads expose velocity, separate left/right hand and foot latents,
  and dynamic `mu/logvar`. The actor receives exactly 51 latent values;
  `logvar`, four map targets, map-validity masks, true velocity, and future
  proprioception labels are available only to the separate auxiliary-loss API.
- The four local-map decoders use the accepted common `10 x 3` contract, not
  older `15 x 9` or `11 x 5` proposals. Sequence masks prevent padded steps
  from advancing the GRU or contributing to auxiliary losses.
- The selected temporal frame is 85-D: angular velocity, orientation, joint
  position, joint velocity, previous action, and direction. The model requires
  this dimension explicitly, and the environment history and next-frame target
  share the same ordered frame definition.

### 2026-08-11 - Dynamic CR1/Ladder Warp Rendering Boundary
- Isaac Lab 2.1's official `RayCasterCamera` cannot implement the required
  LadderMan RFM scene directly: its ray caster accepts one static mesh, does not
  update dynamic meshes, and does not produce semantic segmentation. CR1 has
  independently moving link geometry and the geometry-DR ladder has 18 moving
  cuboids, so the renderer must query all dynamic candidates and retain the
  nearest hit class.
- The reusable `x2_vis` design is to keep each robot mesh in link-local
  coordinates, transform the world ray by the current inverse link pose, and
  call `wp.mesh_query_ray()`. LadderMan locally applies that design to all 30
  CR1 collision meshes, uses analytic spheres for both hands, and uses analytic
  OBBs for all ladder parts so randomized ladder poses can be consumed directly.
- Isaac Sim enforces square pixels for its USD camera. At `80 x 60`, an `87 deg`
  horizontal FOV therefore implies about `70.88 deg` vertical, even if a
  conflicting vertical aperture is configured. The Warp renderer can still use
  independent `87 x 58 deg` axes like `x2_vis`, but this RTX view is then a
  geometry/orientation reference rather than an exact per-pixel depth oracle.
- Isaac Lab 2.1's `TiledCamera` updates image buffers without updating cached
  `data.pos_w` and `data.quat_w_world`; its `reset()` only resets timestamps and
  does not recompute those pose buffers. RTX still follows the USD hierarchy.
  A dynamic Warp consumer should therefore derive camera pose from the current
  parent-link state (plus configured local extrinsics) every frame.
- A Warp CUDA launch is asynchronous. When Warp writes through zero-copy Torch
  aliases and Python immediately consumes those tensors, explicitly synchronize
  the Warp device or use a correctly shared stream. The input direction matters
  too: if PyTorch has just produced pose tensors, launch Warp on
  `wp.stream_from_torch()` so it cannot read them before PyTorch finishes. Isaac
  Lab's official `utils/warp/ops.py` currently keeps an output synchronization
  before returning ray-cast tensors.
- Near clipping must be part of ray construction, not only final depth
  validation. If a head-mounted camera begins inside or within `min_depth` of a
  robot collision mesh, selecting that first hit and then mapping it to
  background incorrectly occludes every farther ladder hit. Shift each ray
  origin to the image near plane (`min_depth / forward_cosine`) before querying,
  then add that ray distance back when reporting optical-center depth.
- Cropping and low-resolution rendering are separate operations. The current
  gate renders the full `80 x 60` at `87 x 58 deg`; optional crop changes only
  the output tensor extent and is disabled until a later training decision.

### 2026-08-10 - Visual Reference Review and Gated Plan
- The local `humanoid_vision` SSR implementation is a mature behavioral
  reference, but it uses Isaac Gym and a custom RSL-RL fork. LadderMan must
  locally adapt its camera/data-flow principles through Isaac Lab APIs rather
  than importing the reference project at runtime.
- Keep the LadderMan paper's `60 x 80` rung-sensitive depth contract and RFM
  for the first visual actor. Reuse SSR's asymmetric observation ownership,
  reset-filled visual buffer, explicit visual lag, CNN/proprioception/GRU
  architecture, and recurrent rollout-storage checks in later gated stages.
- Visual work begins with a live one-environment RGB-plus-metric-depth camera
  proof, then a vectorized raw-observation check, and then the separate-depth
  asymmetric PPO data path. CNN/proprioception/GRU fusion and temporal
  recurrence remain subsequent, separately accepted stages. Details:
  `humanoid_vision_reference_and_ladder_plan.md`.

### 2026-07-29 - Contact Dwell Must Follow Reference Time
- When a contact phase ends one frame before a stable hand release, stretching
  the source interval immediately before the categorical transition preserves
  the reference contact label while spreading the release motion over more
  50 Hz samples. Reusing a uniform global slowdown would also slow unrelated
  phases and does not address a localized release timing problem.
- Reset validity must be regenerated after timeline edits because every
  frame-indexed field, including collision diagnostics and valid masks, changes.

### 2026-07-22 - Anchor State and Reset Distribution
- The local whole-body-tracking pattern uses the reference anchor displacement
  and orientation in the robot anchor frame as explicit observations. For this
  task they are shared by actor and critic: `motion_anchor_pos_b` is 3D and the
  first two rotation-matrix columns in `motion_anchor_ori_b` are 6D.
- Reset sampling should distinguish the effective episode distribution from
  the adaptive component. The current CR1 policy uses `0.20` frame 0,
  `0.40` uniform valid non-frame-0, and `0.40` adaptive valid non-frame-0.
  This prevents the previous adaptive top-bin concentration from becoming the
  de facto reset distribution.
- Reset-time base perturbation is deliberately small and upward-only in Z to
  avoid introducing ground penetration while still preventing a policy from
  overfitting one exact base state. The deterministic CLI switch disables it;
  a separate base-noise-only switch keeps it while disabling other domain and
  joint reset randomization.
- Absolute tracking error alone is insufficient to identify a frozen policy.
  Compare reference/robot anchor path length and projected progress, and use
  the stationary-during-reference-motion fraction as a direct freeze signal.
  The latter is only meaningful together with `progress_evaluable`, because a
  static reference segment should not be treated as a failed movement.

### 2026-07-15 - Collision-Safe Reference Initialization
- Lifecycle decision: keep the CR1 25 s episode cap and adaptive reference-frame sampling, but terminate as a successful time-out after executing the final frame. IsaacLab's normal reset path clears observation/action histories and the stateful termination counters.
- Reset-frame filtering: online collision rejection inside `MotionCommand._resample_command()` is unreliable because contact-sensor buffers are not refreshed until physics advances. An offline frame-validity mask is deterministic, cheap at training time, and can be combined with adaptive sampling without per-environment Python retry loops.
- Available checker: Holosoma already evaluates per-frame signed penetration with MuJoCo `mj_collision` plus `mj_geomDistance` and a `0.01 m` tolerance. It can provide the first reset mask for the CR1/ladder trajectory; IsaacSim should still verify boundary frames because its robot and ladder collision shapes are the training authority.
- Validity semantics: expected hand/foot surface contact is allowed; only penetration beyond tolerance, ground penetration, or unsafe non-contact-body collision should invalidate a frame. The mask should be generated against the exact fixed-ladder pose and versioned in the NPZ.

### 2026-07-15 - Reference Lifecycle and Single-Reference Adaptation
- Episode lifecycle: LadderMan specifies three tracking/contact failure conditions but does not publish the episode duration, reset-frame sampling algorithm, or final-reference-frame behavior. Whole Body Tracking uses a fixed 10 s episode, samples reference frames adaptively at reset, and internally resamples/writes robot state when the reference reaches its final frame.
- Reference data: the paper explicitly collects one OptiTrack climbing sequence on Ladder A (`65.5 deg`, `24.8 cm`) and reuses it for every ladder expert. It does not document the human-to-robot retargeting implementation, but the training observation contains robot joint targets, so the expert necessarily consumes one robot-space retargeted trajectory rather than a raw human skeleton.
- Geometry adaptation: the original robot-space pose remains an intentionally imperfect motion target on other ladders. LadderMan recomputes end-effector targets from each physical ladder, relaxes upper-body tracking, supplies ladder-relative state, adds contact/root task rewards, and trains a separate expert for each target `(inclination, spacing)` with local ladder randomization. The paper does not disclose collision-aware reference warping or projection at reset.
- Torque observability: IsaacLab's implicit actuator estimates torque with its configured PD law and clips it to actuator limits because PhysX does not expose the actual implicit-drive solver torque. Standard `joint_torques_l2` therefore regularizes that estimate.
- Files studied: `LadderMan.pdf` Sec. 3.1, Appendix A.1, Fig. 10, Tables 1 and 3-5; whole-body tracking command/config; IsaacLab actuator and reward sources.

### 2026-07-14 - LadderMan Expert Policy Environment
- Learned: LadderMan does not require a dynamically initialized reference frame to remain stable under zero action. Root/body failures persist for 25 control steps and required-contact loss persists for 30, giving the policy a recovery window.
- Observation design: the state-based expert receives reference body pose/joints/phase, proprioceptive histories, and privileged ladder/body/error state. The paper does not document an asymmetric actor/critic observation split, so the CR1 expert now gives both networks the same inputs.
- Reward design: lower-body tracking has larger weight and lower temperature than upper-body tracking; hand/foot contact-position terms then adapt end-effectors to geometry-derived targets. Table temperatures are denominators in `exp(-E/tau)`, not directly equivalent to the current code's `std` parameter in `exp(-E/std^2)`.
- Randomization: paper ranges cover reference pose/joints, link mass, torso COM, joint offset, contact material, and local ladder inclination/spacing. Physical ladder geometry and command targets must be randomized together.
- Key implementation constraint: the current CR1 wood ladder is a static compound USD. Rung-spacing randomization requires parameterized step colliders; target-only randomization would train against nonexistent surfaces.
- Files studied: `LadderMan.pdf` Sec. 3.1 and Appendix A.1; Isaac Lab observation/termination/event managers; `whole_body_tracking` tracking config and events.

### 2026-06-23 - LadderMan Agent Workflow Guidelines
- Learned: LadderMan implementation should follow a fixed reference chain before coding: Isaac Lab official source, similar `isaaclab_tasks`, existing `whole_body_tracking`, BeyondMimic-style modifications, then the minimum LadderMan change.
- Files studied:
  - `LadderMan/AGENTS.md`
  - `.codex/AGENTS.md`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
- Isaac Lab baseline: New robot RL task work should follow Isaac Lab's manager-based config and extension/task organization instead of inventing local structure.
- Project modification: LadderMan now records durable coding rules in `AGENTS.md`, conceptual learning in this file, and implementation progress in `docs/implementation_log.md`.
- Key insight: The development workflow itself is part of the project architecture: reference-first implementation, official API reuse, note feedback, and progress tracking reduce RL debugging ambiguity.
- Exercise / checkpoint: Before the first LadderMan task implementation, identify the closest Isaac Lab official task, the closest `isaaclab_tasks` example, and the closest `whole_body_tracking`/BeyondMimic reference.
- Result: Workflow guidelines written.

### 2026-06-17 - Learning System Setup
- Learned: The study should follow the order of implementing a real humanoid RL project from scratch, not a passive file tour.
- Files studied:
  - `whole_body_tracking/README.md`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
  - `whole_body_tracking/scripts/rsl_rl/train.py`
- Isaac Lab baseline: Isaac Lab manager-based RL tasks are organized around scene, commands, actions, observations, rewards, terminations, events, curriculum, and gym registration.
- Project modification: BeyondMimic replaces the usual velocity command with a motion-tracking command that advances through a reference motion and resets robot state near reference frames.
- Key insight: The important implementation bridge is `MotionCommand`, not PPO itself.
- Exercise / checkpoint: Explain the `ManagerBasedRLEnv.step()` order before reading more BeyondMimic code.
- Result: Pending.

### 2026-06-17 - Version Baseline Created
- Learned: Comparisons should be pinned to Isaac Lab 2.1.0 because `whole_body_tracking` declares Isaac Sim 4.5.0 / Isaac Lab 2.1.0.
- Files studied:
  - `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
  - `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/command_manager.py`
  - `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- Isaac Lab baseline: Locomotion velocity tasks use `UniformVelocityCommandCfg` and velocity-tracking rewards.
- Project modification: BeyondMimic uses `MotionCommandCfg`, reference-motion observations, and body/root tracking rewards.
- Key insight: Future lessons should use the pattern `Isaac Lab baseline -> BeyondMimic modification -> reason and reproduction steps`.
- Exercise / checkpoint: Locate the exact point where Isaac Lab calls `command_manager.compute(dt=...)`.
- Result: Pending.

### 2026-06-18 - Hermes Learning Workflow Setup
- Learned: The long-term learning workflow should be maintained in project notes plus a custom Hermes skill, not only in chat memory.
- Files studied:
  - `whole_body_tracking/README.md`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/__init__.py`
  - `whole_body_tracking/source/whole_body_tracking/setup.py`
  - `whole_body_tracking/source/whole_body_tracking/config/extension.toml`
  - `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
  - `isaaclab_v2.1.0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py`
- Isaac Lab baseline: Standard locomotion velocity tasks are packaged as manager-based tasks with `MySceneCfg`, `CommandsCfg`, `ActionsCfg`, `ObservationsCfg`, `EventCfg`, `RewardsCfg`, `TerminationsCfg`, and robot-specific config subclasses registered through Gym.
- Project modification: BeyondMimic keeps the IsaacLab manager-based and extension-template skeleton, but replaces the command concept from `UniformVelocityCommandCfg` to `MotionCommandCfg`, adds motion data loading, reference motion state, reference-body observations, motion imitation rewards, adaptive sampling, and robot-specific tracked body subsets.
- Skill created: `humanoid-rl-learning` at `/home/zhangdawei/.hermes/skills/mlops/humanoid-rl-learning/SKILL.md`.
- Key insight: The learning method itself should be standardized: `from-scratch need -> IsaacLab baseline -> BeyondMimic modification -> why it matters -> reproduction checkpoint -> user feedback -> note/skill update`.
- Exercise / checkpoint: Start the first real lesson from `ManagerBasedRLEnv.step()` and map the precise lifecycle position of `MotionCommand`.
- Result: Setup complete; first technical lesson pending.

### 2026-06-18 - Lesson 1: ManagerBasedRLEnv Step Lifecycle
- Learned: A manager-based IsaacLab RL step is not simply `obs -> policy -> reward`. The environment first processes and applies action through decimated physics steps, then computes termination and reward using the post-physics state, resets terminated environments, updates command terms, applies interval events, and finally computes observations.
- Files studied:
  - `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
  - `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/manager_based_env.py`
  - `isaaclab_v2.1.0/source/isaaclab/isaaclab/managers/command_manager.py`
  - `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/mdp/observations.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- Isaac Lab baseline: `ManagerBasedRLEnv.step()` calls `termination_manager.compute()` before `reward_manager.compute(dt)`, resets terminated envs through `_reset_idx()`, then calls `command_manager.compute(dt)`, interval events, and `observation_manager.compute()`. `_reset_idx()` resets managers in an order where `command_manager.reset(env_ids)` happens after observation/action/reward/curriculum reset and before event/termination reset.
- Project modification: BeyondMimic implements `MotionCommand` as a custom `CommandTerm`. During command reset it samples a reference motion frame and writes the robot root/joint state to simulation. During command compute it advances `time_steps`, updates reference body poses aligned to the robot anchor, and exposes these tensors to observations and rewards through `env.command_manager.get_term(command_name)`.
- Key insight: `MotionCommand` is the bridge between IsaacLab's standard manager lifecycle and motion imitation. It is not just an observation provider; it owns reference motion state, reset-time initialization, adaptive sampling, metrics, and the aligned target body poses used by rewards.
- Exercise / checkpoint: Explain why `command_manager.compute(dt)` is after reset and before observation computation, then predict what would break if observations were computed before the command update.
- User checkpoint answer: `command_manager.compute(dt)` is after reset to avoid continuing to advance useless command data for already-failed robots, and before observation because observation needs the new command information.
- Teaching refinement: The answer is correct. Add the nuance that reset creates a new episode state, so command compute after reset also aligns the newly sampled reference command with the newly reset robot before observations are returned to the policy.
- User additional focus:
  - Data/action path: policy action enters `env.step(action)`, is processed by `action_manager.process_action()`, converted by `action_manager.apply_action()` into asset-side targets, then actually reaches PhysX through `scene.write_data_to_sim() -> articulation.write_data_to_sim() -> root_physx_view.set_dof_xxx(...) -> sim.step()`.
  - Reward scaling: `RewardManager.compute(dt)` uses `value = reward_term(...) * weight * dt`, making reward closer to a physical-time integral and reducing reward-scale drift when control frequency changes.
  - CommandTerm abstraction: custom command terms mainly define stored data plus `command`, `_update_metrics()`, `_resample_command()`, `_update_command()`, and optional debug visualization hooks.
  - No step modification needed: IsaacLab's existing `CommandManager` / `CommandTerm` framework is already enough to define BeyondMimic-style `MotionCommandTerm`.
  - Resample call sites: `_resample_command()` is called from `CommandTerm.reset() -> _resample()` during failure and timeout resets; from `CommandTerm.compute()` if `time_left <= 0`; and from `MotionCommand._update_command()` when `time_steps >= motion.time_step_total`.
  - BeyondMimic disables normal time-based command resampling with `resampling_time_range=(1e9, 1e9)`, so meaningful resampling is mostly reset-time and motion-end-time.
  - Adaptive sampling nuance: `_adaptive_sampling()` records failed bins using `termination_manager.terminated`, so timeout resets do not count as hard failed motion phases.
  - Signal separation: `body_pos_relative_w` is used by reward and termination but not directly fed to actor policy because it is a high-dimensional world-frame/aligned dense teacher signal, overlaps with reference joint command, and is better used as evaluator/supervisor than actor input.
- Result: User feedback captured; next lesson should keep explicitly tracing call sites, data flow, and tensor/signal ownership rather than only explaining high-level concepts.

### 2026-06-18 - Lesson 2: Motion Data Format and Replay
- Learned: BeyondMimic's motion preprocessing is not just CSV-to-NPZ conversion. It writes retargeted generalized coordinates into an IsaacLab `Articulation`, renders/updates the scene without physics stepping, then records IsaacLab-computed joint and body kinematics as the training-time reference tensor contract.
- Files studied:
  - `whole_body_tracking/scripts/csv_to_npz.py`
  - `whole_body_tracking/scripts/replay_npz.py`
  - `whole_body_tracking/scripts/rsl_rl/train.py`
  - `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- Isaac Lab baseline: IsaacLab `InteractiveScene` and `Articulation` can be used outside RL training as a kinematic preprocessing/validation tool: write root/joint state to sim, call render/update instead of physics stepping, then read `robot.data.*` tensors.
- Project modification: BeyondMimic stores `motion.npz` with keys `fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`. Training and replay both consume the same NPZ through `MotionLoader`, keeping one reference-data contract across preprocessing, visual validation, and RL.
- Data ownership:
  - `csv_to_npz.py` owns raw CSV loading, interpolation to output FPS, velocity estimation, IsaacLab kinematic replay, NPZ writing, and W&B artifact upload.
  - `replay_npz.py` owns visual validation by downloading a W&B artifact, loading `motion.npz`, writing motion root/joint states into G1, and rendering without physics stepping.
  - `train.py` owns downloading the motion artifact and injecting the local path into `env_cfg.commands.motion.motion_file` before `gym.make()` creates the environment.
  - `MotionLoader` owns loading the finalized NPZ tensors onto the training device and selecting the tracked body subset through `body_indexes`.
- Key insight: A motion-tracking project needs a reference dataset that is already expressed in the simulator robot's state conventions. Storing only root pose and joint positions is insufficient because rewards, terminations, observations, metrics, and reset logic need body poses and velocities in consistent IsaacLab tensor conventions.
- Exercise / checkpoint: Explain why `csv_to_npz.py` uses `sim.render()` and `scene.update()` but not `sim.step()`, and why `motion.npz` stores both joint-space and body-space data.
- Result: Lesson delivered; awaiting user checkpoint answer and feedback.

### 2026-07-24 - Visual Observation Ownership and Augmentation
- Learned: LadderMan's final visual policy does not replace proprioception with
  depth. Its deployable actor input is a split contract: 310-D proprioceptive
  history and climbing direction plus a `(1, 60, 80)` metric-depth tensor. The
  privileged expert/reference state must remain outside the actor even when it is
  retained as a critic observation group.
- Paper detail: Figure 3 represents two mutually exclusive per-image processing
  branches. Rung-focused masking is selected with probability `0.1`; the remaining
  `0.9` receives Gaussian noise `N(0, 0.005)` and independent pixel dropout `0.05`.
  Applying noise to every image before RFM is not equivalent because it also
  corrupts the masked branch.
- Figure 3's RFM image retains both ladder rungs and visible robot surfaces.
  Simulator semantics must therefore use explicit `class=rung` and `class=robot`
  labels; treating every nonzero or `UNLABELLED` pixel as foreground would also
  retain rails and unrelated geometry.
- For a converted multi-link URDF, a semantic tag on the articulation's spawn
  root may not label every rendered link. Validate the mask at link level and,
  when needed, author the robot class on each link during asset spawning,
  before PhysX tensor views are created. Editing rigid-body prim semantics
  after environment initialization invalidates those views. Do not treat
  `UNLABELLED` as robot.
- Sensor choice: Isaac Lab's `RayCasterCamera` matches the paper's Warp-rendering
  direction, but v2.1 supports only one static mesh. The geometry-DR ladder is an
  18-object moving `RigidObjectCollection`, and RFM needs semantic rung IDs, so
  `TiledCamera` is the correct first functional backend despite lower expected
  throughput.
- Data path: `TiledCamera` produces NHWC metric depth and non-colorized semantic
  IDs; the observation term clips and augments them into NCHW; RSL-RL's CNN actor
  encodes depth to 224-D and concatenates it with proprioception; the critic reads
  only its separate 631-D privileged group.
- Local choices that require future evidence are explicit: invalid pixels use
  depth `0.0`, and the paper's three convolution layers use `stride=2` because
  stride/padding are not reported. Full-scale camera throughput and the later
  visual asymmetric PPO objective remain separate validation stages.
- References studied:
  - `docs/LadderMan.pdf`, Sections 3.2-3.3, Figure 3, and Appendix A.2
  - `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py`
  - `isaaclab_v2.1.0/source/isaaclab/isaaclab/sensors/camera/tiled_camera.py`
  - `Ladder/source/ladder/ladder/tasks/ladder_climb/ladder_env_cfg.py`
- Result: Image capture, CNN backpropagation, visual task initialization, and one
  complete PPO rollout/update all pass. The visual policy is trained directly
  with asymmetric PPO: depth and deployable proprioception belong to the actor,
  while simulator-only state belongs only to the critic. This project does not
  use teacher-student distillation.

## Decisions
- Date: 2026-08-03
  Decision: Evaluate required hand support on the indicator-selected rung only.
  Reason: Summing upward force over all filtered rungs lets a hand press the
  wrong rung while avoiding the missing-contact penalty, even though the
  positive support reward already requires the correct rung.
  Consequence: New tasks can opt into target-rung-only missing-contact logic;
  existing tasks retain their previous behavior for checkpoint reproducibility.

- Date: 2026-07-29
  Decision: Classify allowed and forbidden hand contacts from filtered normal-force
  direction rather than target distance alone.
  Reason: IsaacLab's filtered contact matrix contains world-frame normal force per
  contacted step. Horizontal step tops produce upward force, while front/edge and
  underside contacts produce horizontal and downward forces. A hard distance gate
  cannot distinguish these faces and encouraged the policy to avoid all contact.
  Consequence: The rate-limited expert task allows top-face contact, independently
  penalizes front/edge, underside, ground/base-platform, and side-board contact,
  and penalizes missing upward support only after a required-phase grace period.

- Date: 2026-07-27
  Decision: Apply the large real-world initial-position uncertainty only to the
  fixed ground-standing frame-0 reset branch.
  Reason: A `+/-0.10 m` world-XY error models imprecise physical placement at task
  startup, but applying it to ladder-mounted random-frame resets would create
  unrelated contact penetrations and unreachable contact states.
  Consequence: Total reset probability is 10% frame-0 with exact-zero root pose
  offset, 20% frame-0 with `+/-0.10 m` XY and `+/-0.20 rad` pitch/yaw noise, 35%
  uniform valid frames, and 35% adaptive valid frames. The reference and ladder
  are never moved by the robot reset-pose randomization.

- Date: 2026-07-27
  Decision: Treat offline reference repair and online policy-target protection as
  separate layers.
  Reason: Local time stretching can reduce source trajectory jumps but cannot
  prevent a learned policy from producing a new target discontinuity. Conversely,
  runtime filtering cannot repair incorrect contact timing or unsafe reference
  geometry.
  Consequence: Future references use the replay -> dynamics audit -> tapered local
  stretch or coordinated arc-length retiming -> FK/reset-mask regeneration
  workflow, while new policies train with a separate rate-limited action task.
  Uniform time stretch is insufficient for dwell-plus-burst data because it
  preserves the bad displacement distribution. Contact labels follow source-time
  lineage and are not re-inferred from slowed swing velocity.

- Date: 2026-07-24
  Decision: Keep the single expert reference motion fixed during local ladder geometry randomization.
  Reason: LadderMan states that one captured reference is reused across expert ladder configurations; ladder-specific contact targets are generated from the target ladder geometry. Moving the reference itself also corrupts the ground-standing initialization.
  Consequence: Inclination/spacing randomization changes physical/visual ladder boxes, contact targets, ladder-frame observations, and reset validity, while reference joint/root/body trajectories remain byte-identical.

- Date: 2026-07-24
  Decision: Select reset-valid frames conditionally on each environment's sampled ladder geometry.
  Reason: A single conservative four-corner intersection leaves only early ground frames because fixed-reference poses that are collision-free for one rung spacing can penetrate another. One global mask would destroy useful mid-motion initialization for near-nominal geometries.
  Consequence: Offline screening stores a 2-D geometry grid and conservative per-cell masks; runtime samples geometry first and masks both uniform and adaptive reset distributions per environment.

- Date: 2026-08-03
  Decision: Treat nonphysical ladder geometry removal as a separate scene/task variant, not an edit to the established asset.
  Reason: The front diagonal side boards were producing contacts that do not exist on the real ladder, but changing the shared 18-box asset would invalidate comparisons with prior checkpoints. The no-side-board scene filters those boxes at the same collection boundary used for physical and visual spawning, while retaining the rear support legs and all reference data.
  Consequence: New training can measure true step-front contacts without side-board interference; old tasks remain reproducible. A stricter collision reward must be judged from force metrics, not reward magnitude alone, and comparisons that also change actor observations are not penalty-only ablations.

- Date: 2026-08-06
  Decision: Detect whole-robot contact with each ladder rail by making the rail the single contact-sensor body and filtering `Robot/.*`.
  Reason: IsaacLab filtered contact reporting supports one sensor body against many filtered bodies, but not multiple sensor bodies against multiple filtered bodies. A robot-wide filtered sensor would therefore be invalid for two rails.
  Consequence: Two rail-mounted sensors cover every robot link without per-link reward terms or Python loops; the rail cuboids must enable contact reporting, and reward/log surface thresholds must share the same configuration.

- Date: 2026-06-23
  Decision: Use `LadderMan/AGENTS.md` for durable project coding rules, `docs/learning/humanoid_rl_progress.md` for conceptual learning, and `docs/implementation_log.md` for important implementation progress.
  Reason: LadderMan work needs both long-term rules and lightweight progress tracking across IsaacLab, BeyondMimic, and robot RL experiments.
  Consequence: Future substantial changes should update notes/logs alongside code when they create new understanding or important implementation milestones.

- Date: 2026-06-17
  Decision: Clone Isaac Lab v2.1.0 source into `isaaclab_v2.1.0/`.
  Reason: Keep a local, version-pinned baseline for comparing BeyondMimic changes.
  Consequence: Future explanations can cite local IsaacLab files directly.

- Date: 2026-06-17
  Decision: Create a custom learning skill named `humanoid-rl-learning`.
  Reason: The learning project will span many sessions and needs a consistent teaching/progress protocol.
  Consequence: Future sessions should use the skill and update this progress file.

- Date: 2026-06-18
  Decision: Use Hermes user-local skill storage for the learning skill.
  Reason: This conversation runs on Hermes Agent, so the active skill path is `/home/zhangdawei/.hermes/skills/...`, not a Codex skill folder.
  Consequence: Future Hermes sessions can load `humanoid-rl-learning` directly.

## 2026-08-11 - Native Depth Views Versus Comparison Images

- A native RTX depth image and a native Warp depth image do not need identical
  border content when their pinhole fields of view differ. At `80 x 60`, the
  square-pixel RTX camera measured `87 x 70.88 deg`, while the reference Warp
  design independently specifies `87 x 58 deg`.
- Missing robot pixels at one frame are not sufficient evidence of a semantic
  or ray-intersection failure. Project the relevant robot geometry into that
  camera's own FOV first; frame 0's CR1 hands lie below the Warp image, while a
  later in-view frame produces robot hits normally.
- Reprojection is appropriate for a derived renderer-comparison diagnostic,
  but it must not replace, distort, or feed back into either native depth image
  or the policy observation.

## 2026-08-12 - SSR Recurrent Visual Policy Boundary

- SSR uses end-to-end asymmetric PPO, not teacher-student distillation: its
  deployable actor receives depth plus temporal proprioception, while its
  critic receives training-only privileged state.
- The initial Ladder visual-policy transfer should be only SSR's recurrent
  cross-modal encoder: depth CNN `[32, 64, 128]`, proprioception MLP
  `[512, 256, 128]`, and one-layer GRU with 256 hidden units. Its recurrent PPO
  sequence storage, padding masks, and done-only hidden-state reset are part of
  the same required contract.
- Retain the accepted Ladder-specific Warp `60 x 80` image and `RUNG | ROBOT`
  RFM/noise branch. SSR's direct `36 x 36` CNN projection cannot be copied:
  the same valid convolutions produce `128 x 2 x 3` for Ladder, requiring an
  initial `768 -> 128` projection.
- Defer SSR's MoE, auxiliary decoders/VAE, symmetry-equivariant networks,
  imagined foothold guidance, AMP, and multiple critics until the recurrent
  visual baseline is separately accepted.
- Detailed source-to-design notes and staged acceptance gates are in
  `../humanoid_vision/SSR_visual_network_and_LadderMan_migration.md`.

## 2026-08-13 - Ladder Transformer-GRU Estimator Contract

- This decision supersedes the August 12 proposal to defer auxiliary
  decoders/VAE. The selected Ladder design now includes the reference-style
  velocity, terrain-latent reconstruction, future-proprioception, and dynamic
  VAE objectives in the first recurrent visual policy, while still excluding
  SSR MoE, symmetry, foothold imagination, AMP, and multiple critics.
- Ten-frame proprioception means ten chronologically ordered copies of one
  complete deployable observation vector. It must not be approximated by
  independently configured joint and action histories with different lengths.
- Seven estimator heads produce 67 training parameters: velocity 3, left/right
  hand 8+8, left/right foot 8+8, dynamic mean 16, and dynamic log-variance 16.
  The deployed actor
  still consumes exactly 51 latent values because dynamic mean/log-variance
  parameterize one sampled (or deterministic-mean) 16-D latent.
- The hand and foot 16-D values are latent codes, not direct terrain labels.
  Each is split into semantically separate left/right 8-D codes, with an
  independent decoder and MSE for each side. All four limbs use the accepted
  `10x3` grid over `0.315x0.160 m`, with `35 mm` world-X and `80 mm` world-Y
  spacing; the bilateral hand and foot decoder outputs are both 60.
- The terrain query includes all physical ladder boxes and the CR1 ground
  plane, selecting the nearest world-down hit. This matches SSR terrain-height
  semantics and ensures initial ground-supported feet have valid targets.
  Only a true query gap or failed/non-finite geometry lacks a terrain value.
  These local maps remain training-only and never enter the deployable actor.
- Detailed network shapes, initial loss coefficients, and Gate A-F acceptance
  order are frozen in `humanoid_vision_reference_and_ladder_plan.md`.
- Transformer fusion preserves all eleven output tokens. Their fixed-order
  `[B,11,64]` tensor is flattened directly to `[B,704]` and passed through one
  `Linear(704,256) + ELU` before `GRU(256,256)`. The earlier separate
  state/visual `128+128` branches are superseded; there is no post-Transformer
  token averaging.
- Auxiliary weights are `(velocity, hand, foot, future, KL)=(2,2,2,5,1)`, with
  hand and foot group losses averaging their left/right map errors.
- The final design keeps the accepted feed-forward Warp visual task as a
  baseline and adds a separate recurrent task. IsaacLab may expose two depth
  frames as `[B,2,1,60,80]`; the model asserts and removes only the singleton
  image-channel dimension to obtain the frozen `[B,2,60,80]` CNN input.
- `o_(t+1)` is a post-step auxiliary label, not an observation available at
  time `t`. It is attached to the pending transition in `process_env_step()`;
  done transitions mask it because the returned observation has already reset.
- Initial training follows the reference's two optimizer phases. PPO consumes
  the exact stored `[o_t, latent_t]` from rollout so VAE re-sampling cannot
  perturb the probability ratio; the auxiliary phase replays raw histories in
  temporal order to train the encoder, named heads, and decoders. End-to-end
  PPO gradients through the visual encoder are deferred to an explicit ablation.
- The installed RSL-RL already provides recurrent hidden-state storage,
  trajectory padding/masks, and done resets. Ladder should extend only the
  auxiliary transition fields and must not patch `site-packages` or import the
  reference project at runtime.
- Visual-network and learning-algorithm implementation must first reuse the
  patterns in `humanoid_vision/rslRL-317-rq-humanoid_ssr_pure`, especially its
  actor-critic, recurrent encoder, submodules, CE-PPO/helper plugins, rollout
  storage, and runner. Other `humanoid_vision` files are the second reference.
  IsaacLab official code remains authoritative for framework integration.

## 2026-08-13 - Gate A Gravity-Aligned Limb Height Maps

- CR1 local terrain targets query the 18 runtime wood-step collision-box poses
  from `scene["ladder"]`. Nominal `LadderGeometry` positions are insufficient
  because inclination and spacing randomization rewrites every environment's
  physical box poses.
- Each horizontal map is centered on the current limb position. Hands use
  sphere-link centers; feet use each `ankle_x` pose plus local sole offset
  `(0.045, 0, -0.075)`. The limb supplies only this origin: map rows are fixed
  to world X and columns are fixed to world Y, independent of every limb
  rotation.
- Diagnostic classes distinguish rung, front rail, other ladder geometry,
  ground, no-hit, and invalid. The query includes all 18 ladder boxes plus the
  infinite CR1 ground plane and selects the nearest physical surface.
- Batched ray/OBB intersections start `2 m` above the limb reference height,
  point along world `(0,0,-1)`, and select the first surface hit within `4 m`.
  This intentionally matches the SSR gravity-aligned convention rather than
  following hand, foot, or ladder inclination.
- The raw result is world hit `z`; the diagnostic panel shows
  `limb_reference_z - hit_z`. Gaps remain valid and store raw `0 m` only as a
  diagnostic placeholder; final no-hit encoding, clipping, and normalization
  remain gated before becoming training targets.
- The Isaac Sim viewport displays every real hit on its terrain surface, offset
  upward by `1 cm` for visibility. Colors identify the map source: left hand
  cyan, right hand blue, left foot green, and right foot yellow.
- All four accepted windows are `10x3`, with world-X coverage `0.315 m` at
  `0.035 m` spacing and world-Y coverage `0.160 m` at `0.080 m` spacing.

## 2026-08-13 - SSR Privileged Height-Map Geometry

- The SSR height maps are privileged geometric ray-cast targets, not images
  derived from the depth camera. The reference builds local sample grids,
  transforms them to world positions, and uses one batched Warp launch against
  the terrain triangle mesh to return world-space hit heights.
- Its body-centered map has `18x10=180` points: local forward `x` spans
  `[-0.5, 1.2] m` and local lateral `y` spans `[-0.45, 0.45] m`, both at
  `0.1 m` spacing. The grid is centered at the root position and rotated by
  base yaw only. Roll and pitch do not tilt the grid.
- Each foot has a `10x5=50` point map: foot-local `x` spans
  `[-0.1125, 0.1125] m` at `0.025 m` spacing and `y` spans
  `[-0.05, 0.05] m` at `0.025 m` spacing. Its center follows the foot center
  and its in-plane orientation follows foot yaw only; foot roll and pitch are
  discarded.
- All ray origins are placed `2 m` above their base/foot reference height and
  all rays point along world `(0, 0, -1)`. `HeightCast` returns world terrain
  `z`; one unified caster batches the body, base, left-foot, right-foot, and
  optional auxiliary groups. It uses a `4 m` maximum ray distance and stores
  `-1e6` for no hit.
- The network reconstructs processed relative values rather than raw world
  `z`. The body target is
  `clip(root_z - base_height_target - terrain_z, -1, 1) * (10/3)`.
  Each foot target is
  `clip(foot_center_z - 0.055 - terrain_z, -0.8, 0.8) * 5`.
  Therefore zero means terrain at the relevant reference plane; positive means
  terrain below it, and negative means terrain above it.
- The body target is decoded from a 16-D latent to 180 values and trained with
  row-weighted MSE. The configured row weights linearly decrease from the
  first `x=-0.5 m` row to the last `x=1.2 m` row. The two foot maps are
  flattened left then right, decoded from a shared 16-D latent to 100 values,
  and trained with ordinary MSE. The simulator debug drawing instead plots raw
  world hit points, so it must not be confused with the normalized targets.
- SSR uses yaw-only local grids, but the Ladder contract intentionally uses a
  moving world-XY grid: only the limb origin moves. It still performs one
  batched world-down geometric query, keeps raw world hit `z` separate from
  the processed relative target, and visualizes hit points on the physical
  surface. Ladder additionally retains semantic surface classes because the
  SSR terrain caster has no semantic classes.
- Gate A's current `no-hit=0 m` is diagnostic only and must not become the
  training target: in the reference convention, zero means a surface exactly
  on the expected contact plane. The Ladder target still needs an explicit
  limb/contact-plane-relative gap definition plus clipping/scaling before
  Gate B.

## Open Questions
- [ ] How exactly does `ManagerBasedRLEnv.step()` order action processing, physics, reward, termination, reset, command update, and observation computation?
  Why it matters: This determines where a custom motion-tracking command can safely inject state and target updates.
  Next check: Read `isaaclab_v2.1.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`.

- [ ] What is the full tensor contract of BeyondMimic `motion.npz`?
  Why it matters: A from-scratch implementation must validate data before training.
  Next check: Read `MotionLoader` and `csv_to_npz.py`.

- [ ] Which BeyondMimic files are standard Isaac Lab extension-template plumbing, and which are real project innovations?
  Why it matters: It prevents studying boilerplate as if it were research contribution.
  Next check: Compare `tracking_env_cfg.py` with Isaac Lab's velocity locomotion config.

## Implementation Notes
- Area: Skill and progress maintenance
  Notes: The active Hermes skill is installed at `/home/zhangdawei/.hermes/skills/mlops/humanoid-rl-learning/SKILL.md`.
  Failure modes: Forgetting to update progress after long lessons; resuming from chat memory instead of project notes; treating extension boilerplate as research contribution.
  Debug checks: Start each future session by reading this file and naming the next entry point.

## Next Session Entry Point
- Start from: `MotionLoader` inside `whole_body_tracking/tasks/tracking/mdp/commands.py`.
- Read first: `whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py` lines around `MotionLoader`, `MotionCommand.__init__`, `_resample_command()`, and `_update_command()`.
- Do next: Trace how one `motion.npz` timestep becomes actor command, reset state, reward target, termination target, and debug visualization target.

## 2026-08-14 - SSR Height-Map Value Semantics and Ladder Gate-B Contract

- SSR `HeightCast` returns the world-space hit coordinate `z_hit` from a
  world-down ray. A miss is `-1e6`; this raw value is not sent directly to the
  estimator loss.
- The body target is
  `clip(root_z - base_height_target - z_hit, -1, 1) * (10/3)`.
  The foot target is
  `clip(foot_center_z - ankle_to_foot_distance - z_hit, -0.8, 0.8) * 5`.
  The decoder therefore learns signed, scaled vertical clearance relative to a
  reference plane, not world terrain height and not a limb-tilted distance.
- `legged_robot.py` places these processed tensors in `est_nominal`; the SSR
  PPO plugin applies MSE directly between decoder output and that stored target
  (`algorithm_plugin.py:314-341`). Raw hit-point debug plots must not be
  compared numerically with the decoder target without the same transform.
- Ladder Gate A intentionally preserves raw world `z_hit` in
  `limb_heightmaps.py` for geometry diagnostics. Gate B must derive a separate
  training target and never overwrite the raw field.
- CR1 feet use virtual `left/right_sole_contact_point` bodies built from ankle
  pose plus `(0.045, 0, -0.075)`; that point is already the sole/contact
  reference, so the Ladder foot target should use `z_sole_contact - z_hit` and
  must not subtract another `0.075 m`.
- CR1 hand links are collision spheres with radius `0.045 m`. The current
  contact schema's `+0.045 m` target offset defines the hand reference as the
  sphere center. The user confirmed this sphere-center reference for Ladder.

### Gate-B target contract (confirmed hand reference)

Keep, for every limb and frame:

1. `raw_hit_z_w`: world-space hit height for diagnostics and replay truth.
2. `reference_z_w`: sole contact point for feet; sphere center for hands.
3. `relative_height_m = reference_z_w - raw_hit_z_w`.
4. `target = clip(relative_height_m, bounds) * scale` for the selected Ladder
   decoder normalization.

The map remains a moving world-XY grid with world-down rays. No limb rotation
changes the grid or ray direction. No-hit samples need an explicit validity
policy before entering MSE; Gate-A's zero placeholder is diagnostic only.

User decision on 2026-08-14: CR1 hand maps use the sphere center as their
reference height. Raw world hit heights remain for the viewer; relative heights
in meters are the training labels. No clipping or scale is selected yet.

## 2026-08-14 - SSR Auxiliary Target Scaling Reference and Gate-E Rule

- The mature SSR reference does not reconstruct raw world hit `z`. It first
  computes a signed vertical clearance and then applies a task-specific clip
  and scale before MSE: body map `clip(clearance, -1, 1) * 10/3`; foot map
  `clip(clearance, -0.8, 0.8) * 5`. The CE plugin compares decoder output to
  those processed values directly. Its optional body-map row weighting changes
  the MSE emphasis but does not redefine target units.
- Ladder's four maps are different local targets and currently keep
  `reference_z - hit_z` in meters. It would be incorrect to silently copy the
  reference's two unequal body/foot bounds, because Ladder hand and foot maps
  share one requested `10x3` contract and the actual CR1 target distribution
  has not yet been accepted.
- Gate E therefore first uses no-update diagnostics. It must report each
  limb's valid element count independently, target range/percentiles,
  unweighted and weighted auxiliary losses, and auxiliary gradient norm. PNG
  colors are display-only; clipping and normalization become a separate,
  operator-approved target-contract change after these measurements.

## 2026-08-14 - Gate B Target Plumbing

The first training-side interface now mirrors the SSR separation without
changing PPO: `visual_targets` carries only simulator supervision targets,
while `visual_proprio`, `visual_depth`, and `critic` remain unchanged. The
four maps have shape `[B, 4, 10, 3]` and order left/right hand then left/right
foot. Each target is the reference point's world `z` minus the world-down ray
hit `z`; the raw hit height remains available to the viewer. `hit_valid` is a
separate float mask for future masked MSE. The next gated step is a decoder
that consumes the recurrent visual representation and reads this group (or an
equivalent rollout extra) without adding targets to actor input.

## 2026-08-14 - Recurrent Rollout Boundary Rule

For a visual GRU trained with short RSL-RL rollout chunks, the hidden state
must carry across chunks for continuing environments and reset only on that
environment's done signal. Backpropagation may still be truncated to each
rollout chunk, which bounds memory without turning the recurrent model into a
set of unrelated short sequences.

## 2026-08-17 - Controlled Live Visual Diagnostics

An Isaac Sim diagnostic window must refresh inside the controlled action loop:
compute the current action and visual prediction, update the dashboard, and
then call `env.step(action)`. After the requested rollout ends, pause the
timeline before keeping the UI alive. Calling `simulation_app.update()` on a
playing timeline after action generation has stopped can advance robot physics
without fresh policy actions and make a healthy policy appear to collapse.

For a relative-height prediction, the corresponding diagnostic world point is
`z_world = z_limb_reference - relative_height`; X/Y must come from the exact
world-aligned sampling grid used to create the target. Target and prediction
overlays must share that grid and the target's valid-hit mask so an apparent
offset cannot be introduced by visualization geometry.

A manual reset received by a live recurrent diagnostic must be consumed before
the next `env.step`: reset the environment and its observation histories, clear
the recurrent hidden state, discard any action computed from the pre-reset
observation, and insert a done boundary into stored replay data.

## 2026-08-17 - New SSR Future-Observation and Scaling Semantics

- The current `rslRL-317-rq-humanoid_ssr_pure` future-observation decoder takes
  only the stochastic `implicit` latent. Explicit velocity and terrain
  estimates are not concatenated into this decoder input. Its KL term is the
  standard diagonal-Gaussian KL to a unit normal prior, averaged over samples.
- The future-observation label is not raw, and it is not multiplied by one
  universal scale. `legged_robot.py::_observation_vector_assemble()` applies
  field-specific transforms: angular velocity and joint velocity use their
  observation scales, commands use `commands_scale`, joint position error is
  divided by per-joint `dof_scale`, projected gravity and the action field are
  kept as assembled, and the complete vector is finally clipped to the
  configured observation limit.
- Ladder's current 85-D visual-proprio vector has a different contract and is
  not yet field-normalized in the same way. This should be treated as a
  separate target-contract decision instead of silently changing an existing
  training run.
- The exact SSR path does not feed the newly executed action into the future
  decoder. The current history contains the previous action; the dynamic
  `implicit` latent alone predicts the full post-step observation assembled
  after simulation, whose action field is the newly executed action. Ladder
  keeps the same semantics. Its frozen actor sees a richer 660-D input, so the
  action field can be harder to infer, but this must be diagnosed with
  per-field error instead of changing the decoder input.

## 2026-08-17 - Visual-Use and Baseline Diagnostics

- A terrain reconstruction score alone cannot prove that the depth image is
  used, because proprioceptive history and recurrent memory can correlate with
  ladder position. Evaluate the same stored rollout and the same initial GRU
  state under normal depth, all-zero depth, and depth shuffled between
  environments. A meaningful error increase under zero and shuffled depth is
  direct evidence that the learned prediction depends on visual input.
- Velocity estimation must be compared with the zero predictor. Report MSE,
  RMSE, MAE, target/prediction mean and standard deviation, correlation, and
  `skill = 1 - model_MSE / zero_MSE`, both overall and per axis. Positive skill
  is better than always predicting zero; near-zero or negative skill is not.
- Rung emphasis uses exact world-down ray-hit semantics. Valid samples whose
  first hit is class `RUNG` receive weight 2; other valid hits receive weight
  1. Rung and non-rung errors are also reported separately for all limbs,
  hands, and feet so a lower weighted loss cannot hide degradation in either
  category.

## 2026-08-17 - CR1 85-D Proprioception Normalization Contract

The CR1 visual encoder input history and future-proprioception label must be
generated by the same single-frame function. The accepted field contract,
adapted from X2 visual normalization, is:

1. body-frame root angular velocity multiplied by `0.2`;
2. 6-D root rotation unchanged because its matrix entries are already bounded;
3. joint position expressed relative to each environment's randomized default
   position, then divided by that joint's CR1 action scale;
4. joint velocity multiplied by `0.1`;
5. raw policy action and binary climb direction unchanged;
6. the concatenated 85-D vector clipped to `[-100, 100]`.

Subtracting `robot.data.default_joint_pos` is important: Ladder randomizes the
joint calibration offset at startup and applies the same offset to the action
term. Using absolute joint position would incorrectly expose that calibration
randomization as posture. Per-joint action scales must be resolved by
articulation joint name, not assumed to follow a hard-coded array order.

This changes data meaning without changing tensor shape. Checkpoints trained on
raw 85-D frames are therefore semantically incompatible even when PyTorch layer
shapes still match, and must not be resumed.

## 2026-08-17 - Current TransformerVision Contract Audit

- Added `cr1_transformer_vision_current_design.md` as the authoritative record
  of the implemented camera, augmentation, normalized 85-D frame, temporal
  histories, limb maps, Transformer/GRU, estimator heads, decoders, losses,
  runner, optimizer, and inherited randomization.
- Corrected attribution of the root-orientation input: Ladder uses the first
  two rotation-matrix columns (6-D), while current X2 SSR actor observations
  use 3-D projected gravity. The 6-D field is a Ladder design choice, not an
  exact SSR copy. The two orthonormal columns determine the third and retain
  full orientation including yaw; projected gravity retains tilt but is
  invariant to world-yaw rotation. Quaternion sign canonicalization does not
  provide a globally continuous alternative.
- Future reconstruction now applies ELU after `Linear(128,85)` and before the
  final `Linear(85,85)`. The visual data contract is v2 and older visual
  checkpoints cannot resume.
- Fixed limb-height target caching: outputs share one ray query only within a
  common control step, advance on the next step, and invalidate on reset.
- Corrected the future-action interpretation: matching current SSR, Ladder's
  future decoder takes only dynamic `z`; the complete next-frame label contains
  the just-executed action. The current action is not to be added as a decoder
  input.
- Before the cache fix, both the displayed target heights and the world sample
  grid came from the same stale cache, and predictions were compared and drawn
  on that same grid. Agreement between those displays therefore did not prove
  that the target followed the current limb pose. The fixed cache is keyed by
  control step and invalidated on reset.

## 2026-08-17 - Commanded Visual Actor Boundary and Safe Stop

- Corrected the proposed final actor boundary: the action MLP receives only one
  current 85-D frame plus the 51-D TransformerVision latent. The ten-frame
  proprioceptive history and GRU context remain inside TransformerVision and
  are not duplicated in the actor input. The old 660-D five-history actor is a
  visual-data collector, not the final commanded actor.
- Preserve the 85-D shape by replacing its constant `+1` direction field with
  one effective climb-command scalar: down `-1`, hold `0`, up `+1`. The active
  upward-only visual pretraining remains untouched and structurally compatible;
  command variation is learned later during commanded PPO/auxiliary training.
- X2 avoids sudden zero commands on hard terrain during mid-episode resampling,
  and its gait manager lets a swing foot reach stance center before holding it.
  This is avoidance plus phase-safe completion, not an instantaneous arbitrary
  emergency stop.
- Ladder will use requested versus effective command and a `stop_pending`
  transition. A zero request during limb transfer keeps the previous effective
  direction until a reference all-contact candidate is confirmed by current
  physical contacts/motion, then freezes the reference at that safe frame and
  holds existing rung contacts. Direction reversal must pass through hold.
- The geometry-DR training motion contains 408 all-four-reference-contact
  frames, providing candidate stop intervals. Exact actual-contact and velocity
  thresholds still require a standalone visualization/diagnostic gate.
- Downward climbing will use separately collected human motion, not a reversed
  upward trajectory. It must be retargeted and regenerated with downward
  velocities, four-limb contact/rung labels, safe hold intervals, and
  collision-safe reset masks before enabling command `-1`.
- No training, environment, rendering, loss, runner, or diagnostic code was
  changed while the visual-only 5000-iteration job was active; only the future
  integration plan and learning record were updated.

## 2026-08-18 - ThinRails Visual PPO Geometry and Privileged Critic

- Geometry-dependent simulation and labels must be selected atomically. The
  new ThinRails task uses one canonical box tuple for PhysX collision/visual
  boxes, Warp depth rays, auxiliary height-map rays, and critic height-map
  rays. This prevents a policy from seeing or reconstructing rails that do not
  match the contact geometry.
- The deployable actor is unchanged at 136 dimensions. Only the asymmetric
  critic gains simulator truth: raw current joint velocity 25, ladder
  inclination/spacing/rung-length/rung-width 4, and clean relative limb maps
  120. Its new total is 780 dimensions.
- Visual-only pretraining remains an accepted diagnostic baseline, but the
  chosen ThinRails visual-PPO experiment intentionally starts all trainable
  networks from random parameters. Scratch initialization is an explicit
  configuration contract, not merely the absence of a command-line path.
- For the new ThinRails scratch task, using the humanoid reference's adaptive
  PPO rule unchanged caused the nominal `5e-4` rate to compound across 20
  mini-batch updates and hit `1e-2` in one iteration. A 4096-environment run
  then failed at iteration 41. This task now holds PPO at `5e-4`; a separate
  100-iteration scratch validation crossed that point with policy KL below
  `7.1e-4` and finite value/auxiliary losses. The older visual PPO task retains
  its adaptive setting, so this is a task-specific stability decision rather
  than a global change to the Ladder RSL-RL algorithm.

## 2026-08-19 - X2 PPO Constraints Applied to ThinRails

- X2's nominal `5e-4` is not fixed: actor/critic learning rate is adapted after
  every mini-batch using KL thresholds `0.005/0.02`, while its CE/visual
  optimizer remains fixed at `5e-4`. X2 also uses gradient norm `0.1`, action
  clip `10`, observation clip `100`, explicit-estimate clip `10`, minimum
  policy standard deviation `0.2`, and CUDA autocast/GradScaler.
- Ladder now retains that separation but limits adaptive PPO to
  `[1e-5,1e-3]`. The lower ceiling is deliberately Ladder-specific: exact X2
  adaptation previously reached `1e-2` too quickly, whereas the later fixed
  run could not lower its rate when KL expanded during long training.
- The final observation guard applies only to tensors entering actor/encoder
  or critic. Four-limb map labels remain relative metric heights, and the
  critic's clean maps do not need a new normalization because their real range
  is already order one. Simulator targets and masks are never globally
  clipped as if they were deployable observations.
- GradScaler overflow is now observable rather than silent through scaler
  scale and non-finite-step metrics. The first 32-environment smoke iteration
  skipped one PPO step at scale `65536`, settled at `32768`, and completed the
  following iteration with zero skipped PPO or visual steps.

## 2026-08-19 - X2 Sensor Randomization Adapted to Ladder

- X2 centers depth after augmentation with `(depth - depth_mean) * depth_scale`;
  Ladder's new ThinRails task now uses mean `1.0` and scale `1.0`, mapping its
  nominal `[0,2] m` encoder range to `[-1,1]`. Metric depth is retained for
  visualization and labels.
- X2 adds uniform noise to its scaled deployable observation. Ladder preserves
  the same physical noise magnitudes for angular and joint velocity before
  their field scales, producing `+/-0.1` and `+/-0.15` in the final 85-D
  frame. Joint-position noise is instead added after division by each joint's
  action scale, so every joint receives X2's uniform `+/-0.01` network-space
  perturbation. Ladder's 6-D rotation is perturbed by a small quaternion
  rotation rather than adding arbitrary noise to matrix entries.
- Input noise belongs only to the actor/encoder stream. Velocity, limb maps,
  critic truth, and next-proprioception reconstruction labels stay clean.
- X2 clips its environment observation before encoding and separately clips
  its explicit velocity estimate. It does not apply a second global clip after
  concatenating actor observation and learned latent. Ladder keeps the same
  boundary: inputs and critic are guarded at `100`, velocity latent at `10`,
  but the combined 136-D actor tensor is not clipped again.
- X2 multiplies computed torques by a per-joint motor-strength tensor and also
  applies independent Kp/Kd factors. With an implicit PD drive, the equivalent
  pre-saturation gains are `Kp_nominal * motor_strength * Kp_factor` and
  `Kd_nominal * motor_strength * Kd_factor`. Ladder uses the same shared
  `[0.8,1.2]` motor sample for both terms, independent `[0.9,1.1]` gain
  samples, and resamples on reset and every 8 seconds.
- A Gaussian latent's `logvar` determines standard deviation through
  `exp(0.5 * logvar)`. Very large values can overflow sampling/KL; very negative
  values can collapse stochastic variation. Clamping prevents extremes but
  changes the learned posterior and produces zero gradients beyond the clamp.
  Because the reference leaves it unconstrained, Ladder also leaves it
  unconstrained for now and relies on GradScaler plus logged KL/non-finite
  counts. A clamp is a separate intervention only if long-run evidence demands
  it.
- The humanoid CE head is optimized inside five epochs times four mini-batches.
  Ladder's PPO and visual optimizers own disjoint parameters, so their physical
  ordering is not coupled. Ladder performs the same 20 logical recurrent visual
  updates after PPO, using smaller gradient-accumulation micro-batches to keep
  the 80x60 two-frame input within memory.

## 2026-08-20 - Deterministic Dynamic Latent and Reduced Future Target

- The previous Ladder branch followed the humanoid reference's VAE path:
  `GRU -> mu/logvar -> sampled z(16) -> decoder -> next proprio 85`, with an
  additional KL loss. Deployment used `mu`, while auxiliary training sampled
  `z`.
- New Ladder training uses `GRU -> Linear(256,16) -> deterministic z ->
  decoder`. There is no log-variance head, random sampling, or KL term in the
  active estimator. The actor latent remains 51-D, so the actor input remains
  current proprio 85 plus latent 51 equals 136.
- The encoder input and environment observations remain unchanged. The clean
  next-frame label is still collected as an 85-D frame, but the decoder and
  MSE select only its first 59 physical-state values: root angular velocity 3,
  root orientation 6, joint position 25, and joint velocity 25.
- The newly applied action 25 and climb-direction command 1 are excluded from
  reconstruction. They are control/command fields rather than the robot's
  next physical state and should not consume the 16-D dynamics capacity.
- The exact original implementation remains available as
  `VariationalTransformerVisionEncoder`, including the old parameter names,
  full 85-D decoder, reparameterization, and KL loss. The complete accepted
  v2 PPO checkpoint can therefore still be loaded strictly for inspection.

References checked:

- humanoid `rsl_rl/modules/actor_critic.py` VAE implicit head and decoder
- humanoid `rsl_rl/algorithms/ppo_ce.py` future-observation and KL losses
- Ladder `mdp/observations.py` 85-D normalized proprioceptive layout
- Ladder TransformerVision module, pretrainer, PPO, storage, and playback
## BVH handedness must be checked before humanoid retargeting

For LadderMan's legacy BVH preparation, swapping Y-up coordinates with
`(X,Y,Z)->(X,Z,Y)` is a reflection (`det=-1`), not a proper 3-D rotation. The
established following X reflection is therefore required to obtain the
right-handed transform `(X,Y,Z)->(-X,Z,Y)`. Omitting it can make the retargeted
robot face backward even when the human/ladder position overlay looks correct.

A reusable pre-retarget check is to compare the horizontal forward direction
derived from the labeled left/right hips with the mean `Foot->ToeBase`
direction. Their normalized dot product should be positive and normally close
to `+1`; a value near `-1` exposes the handedness/180-degree ambiguity. Do not
hide this failure with an initial robot-yaw offset because an unconstrained IK
warmup can return to the wrong branch.

## Safe stop commands need requested and effective states

A mid-motion zero request should not immediately expose command zero and
freeze an arbitrary swing frame. Keep the current up/down command effective
while a planner seeks the next annotated four-limb stable keyframe. Only when
that keyframe is reached should the effective actor command switch to zero,
the reference frame freeze, and reference joint/body velocities become zero.

The hold behavior must be represented directly in training, not inferred from
moving-reference episodes. Stop behavior should remain a small branch layered
on the established reset distribution, not replace most resets with short
pre-stop clips. The current first mixture is 5% direct reset at uniformly
sampled stop poses, 5% scheduled requests that preserve the original reset,
and 90% uninterrupted motion. Finite training holds teach both maintaining
and leaving a stop; deployment holds remain indefinite until an explicit
resume. Keep this planner outside the deployable actor observation so
simulator-only target frame indexes do not leak into the policy.

Episode duration must be audited whenever a longer bidirectional reference is
introduced. A 2533-frame trajectory at 50 Hz lasts 50.66 s; inheriting the old
30 s limit silently prevents a frame-0 episode from completing. The stop task
therefore uses 60 s, which also accommodates its maximum three-second finite
training hold.

## Reference contact metadata and task targets are one contract

Contact indicators can be correct while training still fails if the task
reconstructs target points using offsets from an older reference. Compare the
motion's `contact_annotation_limb_target_offsets` against the resolved Hydra
environment before every new-reference long run. A 3--5 cm mismatch is already
most of a 5 cm hand target gate and creates a direct conflict between motion
tracking and contact-position rewards.

For a no-command ablation, every information path must be removed together:
actor history, privileged critic, auxiliary next-proprio target, and the
stop-state/reset sampler. A fixed zero compatibility field preserves an
established network shape without leaking a command and avoids turning a
behavioral ablation into an unrelated architecture change.

When the goal is only to remove exceptionally bad resets, use a permissive
nominal-geometry penetration threshold rather than the strict deployment
collision tolerance. For the v4 reference, `10 mm` removes only 35 visibly
deep ankle/hand intersections while preserving 98.62% temporal coverage and
all stable/default poses. This reset mask controls initialization only; the
excluded frames remain in the sequential reference and can still be traversed
during an episode.

## Ground repair must respect manipulation contact phases

Retargeting against ladder geometry does not guarantee compatibility with the
task ground. A later rigid robot/ladder translation can preserve every ladder
contact while increasing ground penetration by the same translation. Diagnose
the ankle collision mesh against the actual task plane rather than relying on
a virtual sole point.

Large start/end ground errors should not be repaired with pure leg IK: near a
straight-leg configuration, vertical foot motion is first-order singular and
can require large artificial hip/knee changes. A safer offline repair shares
the common two-foot clearance with root Z only outside protected hand-contact
phases and uses leg-only IK for the residual. Root correction must be exactly
zero while hand/ladder targets are active; no arm IK is needed.

Temporal gating is part of collision correctness. Unlimited symmetric
clearance envelopes or zero-phase filter tails can leak a ground correction
into neighboring ladder-support frames. Derive per-foot activity windows from
the exact ground-penetration runs, use finite look-ahead, and blend with a
quintic window whose endpoint velocity and acceleration are zero. Validate
ground penetration, ladder penetration, joint limits, protected hand paths,
and joint derivatives together; passing only one of these checks is
insufficient.

## Post-contact smoothing must be checked at output precision

Filtering upper-body joints after contact-aware motion processing can reduce
derivative noise while moving a sphere hand or wrist several millimetres into
a ladder box. Preserve accepted root and leg trajectories, audit the full
robot/environment collision set, and smoothly blend unsafe frames back toward
the accepted source instead of applying a global filter unconditionally.

Collision acceptance must use the same numerical precision as the saved
reference. A float64 state that is safe near a box edge can select a different
MuJoCo contact pair after float32 serialization. Quantize every tested blend
state to output precision before the final collision query, then independently
reload and audit the written file. A smoother derivative average is not by
itself sufficient: report how much smoothing survives collision protection
and whether the worst acceleration events actually improve.

## Use groupwise collision protection when one contact blocks unrelated smoothing

A single framewise blend coefficient is overly conservative for a whole-body
reference: one foot near a step edge can force both arms and the torso back to
the noisy source pose. Split non-overlapping joints into left/right arms,
torso, and left/right legs, apply each group against the full current robot
state, and keep the original source-or-tolerance collision limit for every
robot/environment pair. Recheck the complete float32 pose after the final
group so sequential group updates cannot hide a later regression.

Groupwise processing is still joint-space filtering, not Cartesian trajectory
design or IK. Report per-group retained blend coefficients because a low torso
coefficient can remain the limiting factor even when aggregate derivatives
improve. Choose filter strength using both jerk reduction and hand/foot pose
displacement; a slightly lower jerk is not worthwhile when a stance foot moves
another centimetre.

## Joint smoothing can create low-frequency support-foot drift

Lower joint acceleration does not imply a better contact trajectory. Filtering
the joints of a kinematic chain independently can remove compensating joint
motion and double the Cartesian path travelled by a nominally supporting foot,
even when every joint uses a similar low-pass filter. Separate the diagnosis
into joint derivatives, low-frequency support-foot range/path length, and
high-frequency Cartesian residual before adding more filtering.

When contact fidelity matters, optimize smoothing and support-foot tracking in
one trajectory objective. A sequential low-pass-plus-IK pipeline exposes a
hard cutoff-frequency tradeoff and can create transition jerk. Enforce joint
limits inside the solve, then audit the complete sole collision mesh: holding
one virtual sole point does not prevent ankle rotation from moving another
shoe vertex through the ground or a ladder step. Post-hoc collision blending
is a safety net, but a low retained leg coefficient means collision geometry
must be moved into the optimization objective before the result is promoted.

## Intended support must not be treated as an accidental collision

A strict source-or-tolerance collision rule can reject desirable smoothing
throughout a ladder support phase, because the sphere hand or ankle is already
touching or geometrically overlapping its commanded rung. Relaxation must be
conditioned on all parts of the contact contract: the indicator is required,
the collision body is the corresponding physical limb, the rung ID matches,
and the environment-side closest point/axis identifies the box top face.

Express the allowance as additional depth relative to the accepted source
pair, not only as an absolute depth. Preserve strict limits for wrong rungs,
swing limbs, wrists/knees, front and underside faces, rails, and ground. Store
the per-frame relaxed-pair count and independently audit the serialized
float32 output so a more permissive support rule cannot silently admit an
unrelated collision.

## Keep the collision-limit baseline separate from the blend source

Repeated collision-aware processing may need to blend from the latest accepted
motion while preserving limits defined by an earlier audited baseline. Using
the latest motion as both roles can cumulatively add an intended-support slack
on every processing pass. Record and load the pair-limit motion separately,
validate that the current source already satisfies it, and continue evaluating
all candidates against that original contract.

Root filtering can strongly reduce rigid whole-body jitter with millimetre and
sub-degree pose changes, but per-frame collision fallback may reintroduce much
of the source jerk when feasible alpha bounds are fragmented. A safe result can
still improve visual motion, but repeated low-pass-plus-fallback passes have a
clear limit. Further improvement requires a temporally constrained trajectory
that moves tangentially around the blocking geometry rather than relaxing
wrist, knee, swing-limb, or wrong-surface collisions.

## Use bounded penetration only with a fixed baseline and reset gating

For imitation references, demanding zero overlap at every frame can preserve
large derivative spikes even when shallow overlap is acceptable during rollout
and collision-free frames are selected separately for reset. A small general
ladder allowance can be useful, but ground must remain strict and the limit
must always be measured against one immutable audited motion. Otherwise each
processing pass can silently add another allowance.

Report both the depth increment and how broadly it is used. A millimetre-scale
maximum can still affect many frames, so it is not enough to quote only the
worst penetration. Keep the relaxed reference as a visual-review artifact
until the operator confirms that reduced jitter outweighs the additional
shallow contacts, then regenerate reset-valid masks from the selected final
file rather than assuming every frame is a safe initialization.

## Diagnose stance vibration in Cartesian space before adding joint filtering

A visually vibrating stance can persist after joint-space acceleration and
jerk have been reduced. Compare the captured Foot/ToeBase trajectories, the
raw retargeted collision-body poses, and each filtered artifact. A permissive
per-frame foot-sticking bound can amplify marker noise during framewise
retargeting, while independently filtering hip, knee, and ankle joints can
break the compensating joint coordination that originally kept the foot
approximately fixed.

For a short, known double-support interval, a bounded local trajectory solve
over root translation and both legs can be more surgical than another global
low-pass pass. Track the complete ankle collision-body pose, regularize the
correction through third differences, and use smooth zero-correction window
boundaries before the first hand contact. Audit both Cartesian foot drift and
leg jerk: a nearly fixed foot obtained through high joint jerk is not a useful
reference. Keep any intentional sub-millimetre ground overlap explicit and do
not silently promote the preview to reset-valid training data.

## Recompute derived contacts after spatial trajectory optimization

Contact labels may be preserved through a pure time warp, but they become
stale after root, joint, or body trajectories are spatially optimized. Once
the final motion is accepted, rerun the same contact geometry and hysteresis
parameters on that final body trajectory. Verify that all non-contact arrays
remain exact and inspect changed contact frames as segment-boundary deltas
before using the result for rewards, intended-support collision relaxation, or
reset screening.

Run collision auditing again after reannotation because the indicator and rung
ID determine which overlaps are considered intended support. Generate reset
masks only from this final contact-audited artifact. Geometry-envelope masks
can be much stricter than nominal masks when one fixed reference is evaluated
against randomized ladder corners; keep them as diagnostics unless the reset
logic explicitly transforms the reference pose for each geometry.

## Separate geometric approach phase from stable support contact

A single fixed target-point distance is useful for rewarding accurate hand or
foot placement, but it is not a complete contact-phase detector. A limb can
remain over a valid part of a finite tread top while moving laterally away from
that one point. Detect an independent approach phase from distance to the
finite top rectangle and signed plane clearance, using the physical hand
surface or sole point rather than the body-link origin.

Do not silently replace the stable support indicator when it already drives
force rewards, missing-contact penalties, and required-contact terminations.
Preview the geometric phase as a separate field and visualize both contracts.
Only after visual review should the phase be connected to observations or used
to split approach, stable-support, and release behavior.

## A contact phase needs departure intent and travel order

Entry/release distance hysteresis can still retain an old tread while a limb
has intentionally started its swing, especially when the limb remains close
to the surface. Once a limb has reached a small surface core, combine outward
surface motion with a short confirmation window to end the phase, then lock
that rung for the remainder of the current travel direction. Upward travel may
only advance to a higher rung and downward travel to a lower rung.

For a reference containing ascent, a top turnaround, and descent, store the
direction boundaries explicitly instead of inferring them from noisy
instantaneous base velocity. Apply directional refinement only to limbs that
show the failure: preserving already-correct foot phases avoids turning a
hand-specific annotation fix into an unnecessary full-contact rewrite.

## Contact departure needs temporal grace

Do not permanently close a contact phase on the first release-threshold
failure. Retargeted end effectors can briefly leave the geometric window due
to jitter while remaining in the same physical support episode. Keep the
current tread active for a short, explicit gap allowance, cancel departure if
the limb returns, and only lock the tread after sustained departure. Preserve
an immediate ordered transition when the next tread is reached so temporal
grace cannot block legitimate climbing progress.

When stable support and a broader approach/contact phase coexist, enforce the
semantic invariant `stable support => contact phase` explicitly after all
geometric and directional refinements. Also copy the stable rung ID into the
phase rung ID. Relying on thresholds to satisfy this relationship implicitly
leaves boundary and mode-transition exceptions that can create contradictory
training labels.

If a directional phase ends but the same limb later regains stable support on
the same tread without visiting another tread, the intermediate reference
motion can represent retargeting drift rather than intended release. Preserve
the broader finite-surface phase as a fallback envelope: if it continuously
connects the pre-gap phase and the later stable-support anchor, label the whole
envelope as required contact. This lets training correct the reference defect
without globally loosening contact thresholds or weakening true releases.

## Promote one reviewed contact contract before training

During annotation review it is useful to retain both conservative stable
support and a broader finite-surface contact phase. Once the broader phase is
accepted as the desired training behavior, the train-ready artifact should
expose one unambiguous runtime contract. Write that phase to the established
`contact_indicator/contact_rung_id` keys and preserve the two-contract preview
as the immutable source file. This keeps existing loaders, rewards, metrics,
and terminations synchronized without teaching them two competing meanings.

## Use a symmetric full-state policy as the nonvisual oracle baseline

Before introducing vision, actor and critic can intentionally share the same
complete state observation. For the current CR1 task this is the 631-D expert
critic state plus current joint velocity, ladder parameters, and four local
limb height maps, for 780 dimensions total. The height maps are queried from
known simulator geometry and are therefore an oracle state signal rather than
a camera input. A later visual task should restore actor/critic asymmetry:
retain the full state for the critic and replace actor-only oracle geometry
with deployable visual/history features.

## Inserted stop poses must be audited against reset collision tolerances

A kinematically stable default pose is not automatically a valid reset pose.
Matching virtual sole height to a support tread can still place the actual foot
collision mesh slightly deeper than the offline reset tolerance. Audit the
inserted hold before task integration. Prefer a small, smooth root-height
correction restricted to the synthetic interval over globally loosening the
reset tolerance, because a global tolerance change also admits unrelated
source collisions. Blend the correction to exactly zero at both source-frame
boundaries so the accepted reference remains unchanged.

## Policy checkpoints are tied to the action-processing contract

First-order target filtering and reset ramps change both response time and the
effective action-to-joint-target map seen during learning. When the accepted
reference is already smooth, keep the action contract minimal: apply the measured
joint-target rate limit directly, and use reward terms to teach smooth behavior.
Removing a filter or startup ramp requires a new from-scratch run; evaluating an
older checkpoint through the changed action term is useful only as a diagnostic,
not as a matched comparison.

## Separate simulated time from wall-clock playback speed

IsaacLab advances a fixed amount of simulated time per environment step even
when a single-environment Viewer cannot compute that step before its 20-ms
deadline. Slow wall-clock playback therefore does not change the reference FPS
or the training transition dynamics; it changes only how long the run takes to
watch or train. GPU simulation is optimized for batched throughput and can be
slower than CPU for one environment because kernel-launch and synchronization
overhead dominate. Measure both compute Hz and wall-clock Hz. For deployment-
rate inspection, remove critic/reward/log-only inference work and prefer CPU
single-environment physics when it provides enough headroom, while preserving
the actor observations, physical contacts, randomization, and terminations.

## Key reset anchors must respect randomized scene compatibility

When a long bidirectional reference has stable starts at both ends of a motion
subtask, give each important start an explicit reset probability instead of
depending on uniform valid-frame sampling. Keep enough uniform and adaptive
mass to cover transitions and failure regions. A synthetic anchor calibrated
to nominal geometry should only be selected in compatible environments; use a
conditional selection probability to preserve both the requested total anchor
share and the overall domain-randomization distribution. Track primary and
secondary anchors separately so training logs expose the actual curriculum.

## Supervise visual latents with task semantics when dynamics prediction is weak

A compact recurrent visual latent is more useful to a ladder controller when
its decoder is supervised by the variables that select the next contact:
four limb goals in the robot base frame and four desired-contact states. Keep
the continuous goal coordinates and binary contact logits as separate decoder
outputs even when their 16 values are concatenated for the actor. This permits
metric regression with a masked robust loss and classification with
BCE-with-logits, instead of forcing both semantics through one undifferentiated
next-state reconstruction objective. During swing, supervise the next scheduled
landing target rather than a zero vector; reserve the position mask only for
frames where no future target exists. Phase and rung IDs may remain privileged
label-generation metadata and need not become deployable actor observations.

## Do not supervise whole-body motion semantics with an unobservable clock

Adding a compact `motion_head` whose decoder reconstructs reference joint pose
and root orientation creates a supervision ambiguity during early policy
learning. The current `LadderMotionCommand._advance_reference()` increments
`time_steps` once per control step unless a stop hold is active. Consequently,
when the robot is stuck, blocked by the ladder, or simply stands still, its
camera and proprioceptive inputs can remain almost unchanged while a target
such as `reference[time_steps + H]` continues to change. A recurrent network
can only satisfy this inconsistent state-to-target mapping by using its hidden
state as a wall-clock timer. That behavior is fragile under real-robot stalls,
slips, variable execution speed, and reset timing, so pure time-indexed future
pose reconstruction is not an acceptable deployable motion-semantic contract.

Do not repair this by selecting the globally nearest reference pose to the
current robot pose. Unrestricted nearest-pose matching can reward remaining in
an old standing pose, jump between similar ascent/descent configurations, move
the selected frame backward, and produce discontinuous labels. The command
direction, contact order, and previous progress state must constrain any
alignment.

The first implemented solution separates the legacy time reference cursor
from a state-locked motion progress target:

- `time_steps` remains the existing environment/reference cursor until the
  wider reward and command design is deliberately revisited.
- A per-environment stable-target frame is initialized from the sampled reset
  frame and is used only for whole-body motion-semantic supervision. The 21
  accepted direct-hold frames define ordered anchors: the initial stand, 18
  stable ladder-support poses, the ladder-top stand, and the final ground
  stand.
- Target construction is bounded by each contiguous command-direction run.
  Upward labels cannot search into descent, and descent starts from its own
  first future stable anchor.
- The target frame is held while the robot has not achieved the active
  transition. It advances only to the next anchor in the same direction after
  five consecutive control steps satisfy joint-pose RMS error (`<=0.20 rad`),
  root-orientation error (`<=0.25 rad`), base linear/angular speed
  (`<=0.10 m/s`, `<=0.50 rad/s`), joint-speed RMS (`<=1.0 rad/s`), and all
  reference-required physical target contacts. Terminal anchors self-hold.
- A separate ELU MLP `256->128->64->16` produces the compact motion latent.
  Its ELU decoder `16->64->128->31` predicts the next stable reference joint
  pose (25 values in the visual proprioception normalization) and reference
  root orientation (6D two-column rotation representation). Masked Huber loss
  supervises both outputs. The compact 16-D motion latent, not privileged
  labels or frame/rung IDs, enters the actor.
- If the robot is stuck, the observable input and its active whole-body target
  remain consistent rather than asking an unchanged observation to reconstruct
  a clock-advancing pose. The existing command scalar remains part of each
  85-D proprioceptive frame and disambiguates ascent from descent.
- Direction and prior segment state disambiguate similar poses during ascent
  and descent. Ground-truth dataset phase/rung values may generate training
  labels, but deployable actor input should use command, perception, physical
  contacts, and recurrently inferred/event-driven progress rather than the
  simulator's ground-truth frame index.

The same state-locked labels are now carried by both the visual-only frozen-
policy pretraining path and joint visual PPO. The first version intentionally
predicts only the next stable pose; it does not add the optional local
short-horizon decoder. The legacy reward/reference cursor also remains
time-based. Those are separate future experiments rather than hidden changes
inside this implementation.

## Use a reset-aware temporal penalty to reward motion, not only pose snapshots

A per-frame imitation reward can accept a sequence whose individual poses are
roughly correct while its short-term motion is hesitant, stalled, or locally
out of order. For the nonvisual full-state baseline, add an isolated temporal
tracking experiment that compares the latest ten real control frames with the
same ten time-aligned reference frames. Store per-step tracking costs in a
batched ring buffer and clear only the resetting environments. During the first
nine post-reset steps, average only the history that actually exists; never pad
the buffer by repeating the reset pose, because that invents a stationary
trajectory and creates a false motion target.

Use dimensionless normalized Huber costs rather than another exponential
reward. Exponential kernels approach zero at large error and then provide
little distinction between a bad trajectory and a much worse one. Huber stays
quadratic near the reference but becomes linear for large normalized error, so
it preserves an optimization signal without allowing isolated outliers to
dominate like raw squared error. Combine joint pose, aligned body pose, body
linear velocity, and frame-to-frame tracking-delta costs. The delta term is
equivalent to comparing robot motion increments against reference increments;
it must be disabled for the first real frame after reset because no preceding
sample exists.

This term improves local temporal coherence while the legacy reference cursor
continues to advance with the environment timestep. It does not solve the
unobservable-clock ambiguity described above and should not be treated as a
replacement for state-locked visual motion semantics. Validate it first as a
separate nonvisual A/B task before deciding whether to copy it into a visual
task.

The follow-up A/B keeps only the motion-increment component active. Its
normalized Huber cost was approximately `0.48-0.72` during the first full-
history run and approximately `0.63` in the recent training window. Use an
overall weight of `-0.10`, deliberately 2.5 times the component's former
effective coefficient (`-0.20 * 0.20 = -0.04`) while remaining below the
former four-component penalty. At the recent mean cost of about `0.59`, this
is about `-0.059` before IsaacLab reward-`dt` integration and `-0.00118` per 20-ms
control step. The disabled absolute pose and velocity costs may remain as
diagnostics, but they do not contribute to the reward.

For the same isolated motion-delta experiment, the two random-reset branches
sample from all reference frames rather than the embedded reset-valid mask.
The two fixed anchors remain at 20% and 10%, while uniform-all and adaptive-all
remain at 35% each. This deliberately exposes the policy to poor or colliding
reference initializations and therefore should not be silently propagated to
the validated baseline or visual tasks.

## DAgger rollouts must be generated by the student when sharing data with PPO

For hybrid DAgger and PPO, query the privileged teacher at every state visited
by the student, but execute the student's sampled action in the environment.
Mixing teacher-executed transitions into the same PPO rollout breaks the
on-policy likelihood-ratio contract.  If a random student cannot reach useful
states, use a separate behavior-cloning warm-up before starting joint PPO and
DAgger updates.

## Treat direction commands as deployable intent, not reference phase

An up/down visual climbing policy may receive a binary `-1/+1` direction
chosen by the task planner.  The command should be present in every temporal
proprioception frame consumed by both the state and visual fusion paths, while
reference frame index, phase, target contact annotation, and future reference
pose remain excluded.  During training a motion file may align the command to
demonstration segments, but deployment supplies the same binary intent without
exposing the demonstration clock.

## Separate teacher-distribution fitting from student closed-loop control

A useful distillation control experiment executes the frozen teacher mean in
every training environment for the whole run, while training the unchanged
deployable student with action KL and penultimate-feature MSE. This is online
behavior cloning on teacher trajectories, not student-rollout DAgger. It removes
the feedback loop where an untrained student immediately loses contacts and
changes the training-state distribution into mostly failures.

Reserve entire environment trajectories from optimization, rather than randomly
splitting neighboring frames. Report student action-mean error separately from
Gaussian-standard-deviation KL and hidden-feature error. Also compare with a
baseline that repeats the previous actual action: under teacher control that
input is the previous teacher action, whereas in deployment it is the student's
own previous action. Low mean error without beating this baseline can hide a
policy that mostly persists in its current action.

Environment returns, lifetime, contact failures, and motion-completion markers
belong to the rollout controller, here the teacher. They do not measure student
autonomous performance. Low held-out teacher-trajectory error followed by poor
student-only playback indicates a closed-loop sensitivity/distribution problem;
high error on teacher trajectories instead motivates checking optimization,
capacity, input/label timing, and privileged-feature targets. Neither result
alone proves an observation is fundamentally unobservable. Held-out envs using
the same motion and a shared adaptive sampler do not establish unseen-motion
generalization.

For reset attribution, save the reference cursor and episode identity before
stepping: IsaacLab auto-resets inside step and returns the next episode's
observations. Keep a separate observed-step count because the standard runner
randomizes initial episode timeout counters. Record prediction metrics before
optimization and label this timing explicitly relative to saved checkpoints.

## Diagnose closed-loop imitation with a same-state shadow teacher

Run the frozen student and teacher on the same observed physical state, but
allow only one of them to control each evaluation. Compare teacher-controlled
and student-controlled first episodes from identical initial conditions. Save
the real post-action state before auto-reset, without adding observation/history
computations. A RecorderTerm can itself add an observation computation in
ManagerBasedRLEnv.step; instrumentation must not alter the policy's input timing.

The teacher-only BC frame-zero check on 2026-09-07 found a small initial error
and a shared reset-transient error spike. Under teacher control the student's
shadow prediction error then decreased; under student control it grew rapidly
before large tracking errors and termination. Removing sensor/camera noise,
motor randomization, and delay did not rescue the student. This is evidence of
insufficient closed-loop imitation, not proof that the deployable observation
is fundamentally unobservable or that one specific network/loss is at fault.

Treat action errors in actual per-joint units: this task has nonuniform action
scales and a 0.12-rad-per-control-step target rate limit. Separate body-anchor
tracking metrics from base-link root termination, and force-threshold contact
proxies from the distance-and-duration condition used by required-contact
termination. A successful teacher trajectory can have proxy contact misses;
those misses alone do not demonstrate a grasp failure.

See [the bounded shadow-check report](teacher_only_bc_shadow_check_2026-09-07.md)
for the experimental scope, artifacts, and interpretation limits.

## Distinguish normalization, recurrent reset, and camera clocks

RSL-RL's official action-mean MSE/Huber distillation is not equivalent to our
teacher-variance-weighted KL plus penultimate feature MSE. Averaging each loss
over dimensions does not make its coefficient a gradient-contribution ratio.
Running observation normalization is optional, but when present its saved
statistics are part of the teacher and must be loaded and frozen. The current
teacher has none; our MLP-only loader is not safe for future normalized teachers.

Recurrent state detachment preserves memory values; it is not an episode reset.
Clear only terminated environments before consuming their new observations,
retain continuing environments across rollout chunks, and replay with stored
initial state and correctly shifted done masks. Random motion-frame reset can
still have genuine cold-start ambiguity despite correct hidden-state handling.
Successful public code must be audited rather than assumed to cover all reset
edges: the released Extreme Parkour visual path lacks explicit done-based GRU
clearing, while our current older recurrent path implements it.

Separate camera exposure, history insertion, recurrent updates and policy ticks.
At 50 Hz, nine proprio frames span 0.16 s; two adjacent policy-rate depth frames
span only 0.02 s. New Warp tasks now target 30 Hz using a fractional-phase clock
on policy boundaries (20/40 ms intervals), not exact asynchronous exposure.
Hold both pixels and augmentation between captures, insert depth history only
for fresh frames, and refresh/refill reset environments without contaminating
other histories. Lowering camera rate trades temporal spacing against freshness;
it is not a free replacement for memory. Restore old checkpoint timing explicitly.

See [the RSL/Parkour audit and timing change](distillation_rsl_parkour_audit_2026-09-07.md).

## Re-optimize after removing upstream kinematic degrees of freedom

When a target robot removes joints upstream of tracked end effectors, copying or
deleting coordinates from a successful trajectory is not a valid retargeting
method. DR02 Standard shares all 21 retained joint names, axes, and limits with
the Pro model, but lacks Pro's waist roll and pitch. Those missing transforms
move both hands, so a projected Pro pose can preserve leg coordinates while
still shifting hand positions substantially.

Reuse the robot-independent aligned human keypoints, contact annotations,
geometry, and tuned solver weights, then solve again against the reduced model.
Expose fixed reference bodies with stable semantic names for hands, soles, and
toes, but do not give them mass or new degrees of freedom. Validate the reduced
model's joint order and qpos width before any long retargeting run.

Near-identical kinematics do not imply interchangeable collision processing.
In the aligned 0824 Standard and Pro raw motions, the base, hip, knee, and ankle
positions agree at roughly millimetre scale, yet the grounded Standard feet are
about 3 cm deeper. The cause is the collision representation: the official
Standard ankle-x link uses a `0.30 x 0.11 x 0.06 m` box centered `0.045 m`
below the ankle, whereas the Pro training asset uses its foot mesh. Therefore
world alignment can be reused as a rigid transform, but ground clearance,
robot-ladder signed distances, contact target offsets, and reset-valid masks
must be recomputed with the target variant's actual collision geometry.

## Match reference action delay in milliseconds, not step counts

Latest decision on 2026-09-09: after finding the 1-ms physics setup too slow,
the user authorized **5-ms physics with two-point 0/5-ms delay**. Current
delayed CR1 ThinRails tasks, including the 0824 blind MiddleStand/SmoothOutput
and visual descendants, use `sim.dt=0.005`, `decimation=4`, and delay bounds
`min_control_delay=0`, `max_control_delay=1` in **physics steps**. Thus each
reset samples 0 or 5 ms with equal probability, not 0 or 20 ms. This supersedes
the earlier deferred 0/1/2-ms target and the implemented six-point 0--5-ms setup.
Each environment samples once per reset and retains its lag for that episode.
Policy, reference, and reward clocks remain 50 Hz; render interval is 4
physics steps. Camera refresh/history and perceptual delay are not changed.

The local X2 reference uses `sim.dt=0.001`, `decimation=20`, and
`lag_timesteps=2`. Its `LeggedRobot.step()` calls `_compute_torques()` inside
every physics substep, where the action history advances. Its delay range is
therefore 0--2 ms, even though policy actions update at 50 Hz. Reference files:
`legged_gym/envs/X2/x2_vis_config.py` and
`legged_gym/envs/base/legged_robot.py` under
`/home/zhangdawei/humanoid_vision/LeggedGym-317-rq-humanoid_ssr_pure/leggedgym-317`.

LadderMan originally used `sim.dt=0.005`, `decimation=4`, also giving a 50 Hz
policy. This matches the local IsaacLab v2.1 locomotion
`velocity/velocity_env_cfg.py` and whole_body_tracking's
`tracking/tracking_env_cfg.py` baselines. The July 22 implementation log
already records this timing; it was not introduced by the 0824 reference.
The inspected records do not establish a project-specific 1-ms-versus-5-ms
physics comparison or a demonstrated optimal timestep for ladder climbing.

The August 19 implementation log explicitly added a 0--2 **policy-step**
delay. Previously, `RateLimitedJointPositionAction.process_actions()` advanced
that buffer once per 20-ms policy step, producing 0/20/40 ms rather than the
reference's 0/1/2 ms. The corrected implementation advances IsaacLab's
`DelayBuffer` in `apply_actions()`, which ManagerBasedRLEnv calls once per
physics substep, following the timing used by official `DelayedPDActuator`.
It retains the existing actuator model rather than switching implicit drives
to explicit PD. Scale, offset and the 6-rad/s target limit still run once per
policy update; the bounded target is then delivered after the sampled delay.

Reset clears only the selected environments' histories. Until their first
new command arrives, the targets hold the newly sampled reference reset pose;
the buffer's empty-history fallback must not silently bypass the first delay.
This is not an action filter or startup fade. Play reports the actual lag in
physics steps and milliseconds; `--disable_randomization` still removes the
delay buffer entirely.

The generic LadderEnvCfg and older no-delay baseline keep their original
5-ms/4-substep timing. Current delayed tasks have returned to that timing too,
while retaining the physics-rate delay fix. This removes the extra 16 physics
steps per policy update introduced by the temporary 1-ms setup. Actual
throughput still needs measurement; total training time need not scale exactly
with the physics-step count. Existing weights remain loadable, but changed
delay semantics mean this is not a bit-identical replay of old training.

A fixed 5-ms physics grid represents 0 and 5 ms exactly, not intermediate
1--4-ms arrival times. Mixing old and new targets within a step would be a
fractional-delay approximation with smoothing effects, not an exact transport
delay, and is not used. Training at the two endpoints does not guarantee
robustness at every intermediate delay; that can be evaluated separately with
small 1-ms-physics runs if needed.

## Use an exact-input DAgger run as a positive control

When visual distillation fails, first remove observability and architecture as
variables. Give a fresh student and the accepted teacher the same full-state
tensor, in the same order and scale, with the same feed-forward architecture.
Let only the stochastic student act, query the deterministic teacher mean at
every state visited by that student, and regress the student mean with ordinary
elementwise-mean MSE. This is still DAgger: the defining property is online
expert labeling of the learner's state distribution, not teacher control.

The strictest implementation should expose one physical observation group to
both models rather than separately constructing two equal-width groups. Check
the actual model-input tensors for exact value equality, freeze and hash the
teacher, prove that the executed action is the student sample, and prove after
each update that the student changed while the teacher did not. Save raw samples
as well as aggregate losses so an apparently good scalar curve can be audited.

If this positive control learns stable closed-loop climbing, it validates the
basic teacher loading, observation ordering/scaling, online label timing,
student rollout, action convention, optimizer, checkpoint, and student play
paths under the tested contract. It does **not** validate deployable visual
observability, command inference, depth timing, recurrent resets, visual
encoding, KL weighting, feature matching, or mixed teacher/student control.
Those variables must then be restored one at a time.

If it cannot learn despite action error reaching a low plateau, the failure can
no longer be attributed to missing visual/reference information. Investigate
the distillation plumbing, optimizer behavior, action-space sensitivity, and
closed-loop accumulation directly. Conversely, early poor episode reward is
not by itself a failed test: a random student deliberately visits bad states,
so judge convergence using held checkpoints, deterministic student play, and
the action-error trajectory together. This control is diagnostic and uses
privileged 780-D input; it is not a deployable policy.

## Restore visual observability with one controlled distillation change

After an exact-input action-MSE DAgger control succeeds, the clean next
experiment changes only the student's observation and encoder. Keep the same
frozen teacher, student-only stochastic rollout, online label timing, action
convention, action-mean MSE, optimizer scale, reset distribution, and task MDP.
Do not simultaneously restore teacher-feature matching, Gaussian KL, auxiliary
reconstruction, PPO, BC warm-up, or teacher/student action mixing; otherwise a
failure cannot be attributed to the visual observation/architecture boundary.

For this comparison, retain environment rewards and termination metrics only as
closed-loop diagnostics; they must not backpropagate. Log pre-update rollout
error separately from shuffled-batch training error, plus action correlation,
per-joint errors, visual-encoder gradient/update norms, command balance, camera
freshness, and reset-filled histories. A successful same-input control followed
by failed visual MSE-DAgger narrows the problem to deployable observability,
history/camera timing, visual fusion/capacity, or conflicting teacher labels.
It does not by itself identify which one, and adding more loss terms before this
comparison converges would obscure that diagnosis.

## A camera-rate GRU requires sequence training and an explicit freshness clock

For the recurrent visual control, every one of the 25 proprioceptive history
frames must retain the deployable up/down/stop command. At 50 Hz this history
covers 480 ms. The head depth camera remains 30 Hz, and one current image is
held between exposures; the policy must not reinterpret held copies as new
visual observations.

Carry an explicit per-environment `depth_updated` mask from the camera/history
term to the student. Update the visual GRU only for fresh or first-after-reset
images, while the actor continues to run at 50 Hz from current proprioception
and the held recurrent visual state. Clear only each done environment's hidden
state. This keeps recurrent time tied to the real camera clock and avoids
different effective memory speeds from the alternating 20/40-ms simulator
sampling pattern.

Recurrent DAgger minibatches must preserve time. Batch complete rollout chunks
by environment, save the hidden state at the start of each chunk, and apply
`done[t]` before evaluating observation `t+1`. Flattening and randomly shuffling
individual transitions can train the feed-forward heads, but it cannot train
the GRU to reproduce its rollout computation. Step-by-step inference and batched
sequence evaluation should be numerically equal under the same freshness and
done masks before starting a long run.

## Mixed DAgger changes the visited states, not the expert label

In both student and teacher rollouts, evaluate the student on its deployable
observation and the frozen teacher on its privileged view of the same physical
state. Store the teacher action mean as the target in both cases. The only
difference is the action sent to the environment: a student-controlled state
transitions under the stochastic student action, while a teacher-controlled
state transitions under the deterministic teacher mean. All collected samples
can therefore share the same unweighted action-imitation loss.

Teacher-controlled environments improve expert-trajectory coverage. An early
student may terminate before reaching release, swing, target-rung contact and
body-advance states; online expert labels at its failed states cannot create
examples of later states that were never visited. A persistent teacher subset
supplies those later examples, while the student subset still supplies the
learner-induced deviations required to address covariate shift.

For recurrent policies, keep controller ownership fixed by environment rather
than resampling it every policy step. Continue advancing the student hidden
state from its own observations even where the teacher controls physics, and
clear that hidden state on the same per-environment resets. Report teacher and
student rollout performance separately: aggregate reward and termination
statistics become mixtures and may improve even if the deployable student does
not. Mixed rollout repairs state coverage, but it cannot resolve genuinely
unobservable teacher decisions or conflicting labels by itself.

## Long expert-only warmup can collapse an unnormalized recurrent student

Teacher rollout is useful for covering late expert states, but an extended
all-teacher BC phase is not automatically a safe initialization for a recurrent
visual student. In the 500-update warmup experiment, action imitation on the
narrow expert distribution became excellent while the proprio encoder grew to
thousand-scale features and the GRU settled into hard-saturated, nearly
constant coordinates. At the switch to 75% student control, the learner saw a
new state distribution and produced a large action shock; later optimization
could remap downstream features but could not recover gradients through the
saturated encoders and GRU.

The clean diagnostic is to remove only the all-teacher prefix while retaining
a persistent fixed teacher subset. Starting at iteration zero with 25% teacher
and 75% student environments preserves expert-trajectory coverage and exposes
the encoder to learner-induced deviations before an expert-state shortcut can
become entrenched. Keep the same MDP, teacher, network, learning rate and MSE so
the outcome isolates warmup scheduling rather than conflating several fixes.

Action error and episode return alone cannot reveal this failure mode. Record
the RMS/max magnitude of proprio and depth features, fused recurrent input,
GRU candidate/hidden state, recurrent projection and actor feature, together
with GRU saturation, exact-zero hidden fraction and module gradient/update
norms. Split control-performance metrics between teacher- and student-driven
environments; aggregate episode metrics are a 25/75 mixture and are not the
deployable student's performance.

## Hybrid recurrent PPO needs learner-controlled trajectories and full gradients

Adding a reward term to the loss name does not make an imitation runner use RL.
It needs sampled actions, old action probabilities, value estimates, rewards,
termination/truncation handling and return/advantage computation. In a fixed
teacher/student mixture, teacher-controlled states remain useful for action
imitation, but their returns evaluate teacher control. Keep them out of the
student PPO surrogate, critic targets and advantage normalization. Select
student samples before numerical loss operations; multiplying by a mask after
an invalid calculation can still propagate NaNs.

A recurrent visual actor must recompute its CNN, proprio encoder and GRU from
raw ordered observations during PPO optimization. Feeding cached detached
visual features trains only the downstream actor and cannot teach perception
from rewards. Before changing weights, replayed student action probabilities
must match collection probabilities (ratio approximately one) with the same
initial hidden state, camera freshness and done masks. Prove PPO-only gradients
reach every encoder independently of imitation gradients.

LayerNorm before a GRU addresses feature scale; PPO addresses long-horizon
action outcomes. They target different failure mechanisms. MSE coefficient
values are not effective gradient ratios against PPO; record both objectives'
gradients on shared parameters and their agreement. A privileged critic may
use reference information during training without putting that information
into the deployed actor, but neither critic nor RL creates an unobservable
phase signal for that actor.

Diagnostic samples must identify their parameter timing. A rollout-start
snapshot from iteration N and the usual post-update `model_N.pt` do not share
weights. Save the pre-update policy with occasional raw snapshots to make
recorded action replay an actual correctness check, rather than confusing a
normal optimizer update with an inference mismatch. Likewise, hidden-state
saturation and zero fractions describe sampled elements; they do not by
themselves identify a fixed fraction of permanently dead neurons.

## Loss warmup is not teacher-only rollout warmup

Keep controller ownership separate from objective scheduling. A fixed 25%
teacher / 75% student mixture can collect learner-induced deviations from
iteration zero while the PPO actor objective gradually gains weight. Unlike
an all-teacher prefix, this does not abruptly change who controls physics.
Scale the entropy term with the PPO surrogate, but leave critic fitting active
on student returns. Restore the objective schedules using the saved update
index and validate their configuration when resuming an optimizer.

Loss coefficients are not parameter-step ratios. Inspect both raw and weighted
objective gradients on shared actor parameters, and actual parameter deltas.
Adam can offset a uniform gradient scaling; in particular, action log-std has
no action-mean MSE gradient to compete with its PPO/entropy gradient. Keeping
the critic loss unscaled and clipping it separately does not imply a separate
learning-rate schedule when actor and critic share the same optimizer.

LadderMan section 3.2 specifies joint PPO and annealed student-to-teacher KL,
but not a numeric warmup or decay timetable. Our MSE schedule is an experiment,
not a reproduction of undisclosed paper hyperparameters. Evaluate student-only
task performance and representation stability before deciding whether the
planned decay is appropriate; a lower imitation error alone is not sufficient.

## Match evaluation starts and capture terminal state before autoreset

Deterministic evaluation means executing the actor mean; it need not disable
reset randomization, observation noise, or control delay. Keep those conditions
explicit and compare teacher/student initial state and observation hashes.
Matched initial seeds do not guarantee identical later noise streams after
policies cause different resets. Count one first episode per environment so
early failures are not replaced by additional easier random-reset episodes.

IsaacLab can reset robot and command state before returning from `step`.
Capture terminal root state and reference frame after termination computation
but before autoreset, and require completion without simultaneous failure.
Report unfinished episodes as censored, not successes. In motion37, frame0
starts the complete up/down sequence and frame1550 starts descent; frame1500
still has an upward command. A reference phase boundary is a progress marker,
not by itself a geometric success criterion.

When normalization and RL are introduced together, use a normalization-only
MSE control before attributing changes in performance to RL. Keep rollout
ownership, noise, optimizer settings, teacher, and MDP identical to the old
MSE baseline; an independent task and checkpoint contract preserve that test.

Interpret termination names from their implementations. The current
`bad_required_contact_consecutive` checks whether any reference-required
hand/sole marker stays more than 0.45 m from its world-space rung target for
30 policy steps (0.6 s); it does not read contact force. Concentrated failures
30 steps after a reference contact transition can indicate failure to follow
the new target, not necessarily loss of a previously established physical
contact. Combine terminal frames with actual root/limb progress before calling
longer survival a climbing improvement or attributing failure to exploration.

## Finite activations and rate-limited joints do not rule out raw-action tails

Distinguish the policy's raw output, the processed joint target and the actual
robot motion. A target slew-rate limiter can keep physical joint motion bounded
while raw actions become enormous. Rewards using raw-action differences can
then produce huge squared penalties without matching joint-acceleration or
torque spikes. Encoder LayerNorm and gradient clipping do not directly bound
the actor output or Adam parameter changes; a current-observation shortcut
can also bypass encoder normalization.

Inspect full-history extremes as well as comparable stage averages. Report
both the actual mean and the frequency/magnitude of outliers; an outlier-excluded
mean explains typical behavior but cannot replace the true result. A low final
loss or recovery after a spike does not establish that the rare failure mode
has disappeared. Periodic small snapshots may miss the triggering trajectory,
so temporal correlation alone cannot distinguish an optimizer jump, an unusual
observation, an expert label, or action-history feedback as the initial cause.

## End-to-end visual PPO requires raw-observation replay, not just optimizer membership

The old TransformerVision PPO stores a detached visual latent and trains the
action MLP on that fixed input; the encoder is trained separately by auxiliary
losses. Adding encoder parameters to that optimizer does not reconnect the
computation graph. For end-to-end PPO, store the raw proprio/depth sequences,
rollout-start recurrent state, reset boundaries, sampled actions and old
log probabilities, then recompute CNN, Transformer, GRU, latent heads and action
MLP with gradients enabled. Keep privileged critic inputs and reconstruction
labels outside the actor forward path. Reconstruction-only decoders have no
PPO gradient unless their outputs actually feed the actor; their supervision
still updates the shared encoder through the latent heads.

Sequence replay must represent the same policy that collected the data. Clone
observation buffers before env.step can mutate history in place. Disable
Transformer dropout consistently in rollout, replay and play; deterministic
latent selection alone does not disable attention dropout. Use done[t] to reset
before t+1, and replay the saved initial hidden state. Validate the pre-update
probability ratio against one, including resets and repeated camera frames,
and test PPO-only gradients with reconstruction and value losses disabled.
The September 13 GPU smoke reproduced ratios within 3.82e-6.

Camera rate and recurrent update rate are separate choices. The new
reconstruction baseline samples head depth at 30 Hz while updating its fused
proprioceptive/visual GRU at the 50 Hz policy rate. Held depth frames do not
represent new camera exposures, but new proprioception still enters the GRU.
This differs intentionally from the previous fresh-camera-only DAgger GRU.

Use one optimizer for the shared end-to-end objective. Accumulate microbatch
loss numerators using the full logical batch's denominators, including the
per-limb weighted count of valid heightmap pixels; averaging separately
normalized microbatches is wrong when their valid-pixel counts differ. Record
PPO and weighted-reconstruction gradient norms separately: connected gradients
do not imply balanced influence. In the initial small GPU smoke, auxiliary
encoder gradients were 25--49 times the PPO gradients; this is an early
optimization diagnostic, not evidence of task success or failure.

## Separate perceived support state from commanded contact intent

A four-limb contact bit vector cannot identify which rung supports each limb.
Pairing each contact bit with a rung association gives a richer support mode:
relative hand/foot heights, same-rung support, and inter-limb rung differences.
It still does not specify full joint posture or uniquely choose the next limb
to move. Repeated ladder geometry can make the absolute rung number ambiguous
from a local depth view even when relative geometry is clear.

LadderMan already loads `contact_indicator` and `contact_rung_id` from the
reference motion. Both describe the reference schedule, not actual physical
contact after the policy deviates. `task_goal_rung_id` selects the current
support rung or next scheduled rung while swinging, within the command segment.
Do not train a perception head against these labels and call its output an
estimate of measured contact. Actual-contact labels need force/contact sensing
and rung association at the policy's visited state; deployed inputs need
onboard sensing or an estimator, not simulator truth.

The first controlled experiment adds oracle body-frame goal12 and desired
contact4 directly to the current visual PPO actor (120 -> 136 dimensions).
Invalid goals are zeroed; rung IDs and validity bits are not extra actor inputs.
This preserves the agreed 16-D diagnostic, but a future general interface
should expose validity explicitly rather than relying on a zero-goal sentinel.
Keep the MDP, reference advancement, reconstruction losses and visual network
unchanged. This privileged experiment asks whether compact task intent helps;
success would not prove deployability, sufficient visual perception, or that
the actor actually needs the camera. Failure would not alone disprove the
interface without checking optimization and target validity/coverage.

For a later deployable design, prefer local rung geometry and actual support
estimation over reconstructing a dense map merely because it is available.
Keep desired contact and target selection on a separate onboard command/event
planner. A possible interface is goal_b12 + goal_valid4 + desired_contact4 +
actual_contact4; a rung-based alternative needs tracked relative rung offsets,
not ungrounded absolute IDs. Treat rung identities as categorical associations
(or relative up/same/down choices), not raw integer MSE regression. Event-driven
goals must eventually be aligned with reference advancement and reward targets;
changing just the actor inputs does not change a clock-driven tracking task.

## Optimize sequence execution before removing useful supervision

The September14 end-to-end run spends about81% of iteration time learning
(20.07s versus4.62s collecting, last100 logged iterations at inspection).
With5120 envs, five epochs, four logical batches and32-env microbatches, each
update runs800 complete24-step microbatches and19,200 one-step GRU calls. CNN,
Transformer and history MLP already batch the time/environment dimensions;
the recurrent reset-aware loop remains stepwise.

Safe first changes are to reduce detached logging scalars on-device before
one CPU transfer per logical batch, and cache the immutable reset-before mask
once per update. They preserve loss normalization, gradient paths, optimizer
steps and reset semantics. Next benchmark64/128-env microbatches with memory
peaks and pre-update probability replay checks. Larger microbatches preserve
the objective but need not be bitwise identical. GRU fusion must respect each
environment's reset boundary; cached detached visual latents would break the
end-to-end PPO experiment. Removing the four small map decoders is not a major
network-compute saving; the current privileged critic also consumes heightmaps,
so removing auxiliary labels alone does not remove the environment ray query.

## Fixed normalization contract for recurrent visual DAgger

The scheduled recurrent MSE-DAgger student intentionally does not use running
mean/std observation normalization. Its deployable inputs use fixed, reproducible
scales: root angular velocity `0.2`, joint velocity `0.1`, joint position error
divided by `CR1_ACTION_SCALE`, bounded rotation/action fields, and a final
proprioceptive clip of `[-100,100]`. Warp metric depth is clipped to
`[0.1,2.0] m`, centered at `1.0 m`, and therefore reaches the CNN as
`[-0.9,1.0]` (invalid/dropout pixels are `-1.0`). The frozen teacher receives
the raw 780-D observation contract used during its training; adding a new
normalizer would invalidate the checkpoint. Action MSE is computed directly in
the normalized joint-position action coordinates. Runtime finite/range checks,
finite loss checks and non-finite gradient errors are part of the contract so a
bad sensor value or exploding update stops immediately instead of silently
poisoning the rollout.

## Privileged-input DAgger is an information-sufficiency upper bound

The privileged recurrent visual DAgger ablation keeps the current learned
depth/proprio/GRU visual feature and adds simulator-only task signals before the
student action MLP: next-stable four-limb target positions in the robot anchor
frame (12-D), rung-filtered actual contact flags (4-D), and four end-effector
velocities in that frame (12-D, scaled by 0.1). The 28-D oracle deliberately
omits target-validity/reference-contact bits and rung IDs. It uses the same
frozen teacher and action-MSE update. Its current controller schedule is 100%
teacher through update 1000, linearly decreasing to 10% at update 5000, then
holding 10%. Thus a large gain isolates missing task information rather than
changing the imitation objective. It is not a deployment architecture. A failure still
leaves optimization, normalization, sensor timing and target alignment as
possible causes; a success only motivates building an onboard estimator for
the same compact interface.

The first 5120-environment run fitted the teacher well through update 156 but
then underwent a finite optimizer collapse: encoder activations exploded and
the final ELU actor feature became an all-`-1` constant, eliminating every
upstream gradient. The follow-up therefore normalizes the 128-D proprio and
depth features separately immediately before the GRU. This is the established
local LayerNorm control and changes neither learning rate (`1e-3`), controller
schedule, loss, nor checkpoint cadence.

## Stable support is a surface state, not just a ladder-contact bit

For ladder transitions, a foot on the floor is already a valid support even
though the legacy ladder-only contact label is false. Stable-state discovery
must therefore distinguish no support from ground support and rung support.
Use a separate categorical support ID (`-1` none, `0` ground, `1..N` ladder
rungs) and split phases on changes of the complete four-limb support tuple.
Selecting one representative from each sufficiently long tuple run captures
the actual contact sequence without duplicating consecutive frames. Do not
silently shift the legacy rung IDs used by trained policies and rewards; store
the ground-aware contract separately and make every next-stable consumer read
the same saved anchors.

The stable-frame list also includes three typed boundary poses: frame 0, the
middle standing frame 1500, and terminal frame 3060. These are not falsely
treated as four-limb rung contacts: their saved support IDs describe the real
feet-on-ground/rung supports and inactive hands remain `-1`. Reset segmentation
uses the same anchors, with segment `i` covering `[anchor_i, anchor_(i+1)-1]`;
the last segment includes the terminal frame 3060. Thus 29 stable boundaries
produce 28 real motion segments instead of a spurious terminal-only segment.
In the Oracle task, reset first
chooses an eligible segment and then samples a geometry-valid frame inside it;
adaptive failure weighting is summed per segment while uniform reset chooses an
eligible segment uniformly. This keeps adaptive reset while aligning reset,
playback, target generation, and stable-anchor gating to one motion contract.

## Use a policy-only visual encoder for the Oracle information experiment

The Oracle contact-goal experiment is intended to test whether compact task
intent is sufficient, not whether height-map reconstruction can be learned at
the same time. Its visual policy therefore uses the smaller established
recurrent pattern: 25 command-aware 85-D proprioceptive frames become 128-D,
two 60x80 depth frames become 128-D through three convolutions, and their
256-D concatenation enters a GRU-256. The GRU feature goes directly to the
unchanged action MLP together with current proprioception and the 32-D Oracle
input. The newer support-identity experiment replaces that preliminary 32-D
input with a 48-D interface: next-stable target XYZ for four limbs plus actual,
current-expected, and next-stable support relations. There is no Transformer,
reconstruction decoder, velocity prediction,
height-map prediction or next-frame loss in this branch; PPO is the only loss
and reaches every policy module.

Increasing sequence microbatches from 32 to 64 environments does not change
which rollout samples enter a logical PPO minibatch, its normalization, or the
20 optimizer steps per update. It only accumulates the same logical gradient
in 20 rather than 40 recurrent forward/backward calls per logical minibatch,
halving launches from 800 to 400 per update. Numerical results need not be
bitwise identical because floating-point reduction order changes, and peak
activation memory rises approximately with the microbatch size. The inspected
GPU had ample memory headroom, so 64 is the appropriate first setting; 128
should only be tested after measuring the new peak and iteration time.

## Align dense contact credit with the stable-state transition contract

If a reference gate defines success with ground-aware stable support while the
dense reward still follows a ladder-only contact label, PPO receives
contradictory credit: feet on the floor may be required by the gate but treated
as swing limbs by the reward. The Oracle visual task therefore uses the same
saved support indicator and categorical surface ID for both target selection
and dense contact-position tracking. The integer ID is never regressed; it only
selects the ground or randomized rung geometry used to construct a continuous
3-D target.

Sparse gate release alone is also a weak learning signal. A signed potential
term based on reduction of mean distance to the pending stable target gives
dense credit for approaching it and symmetric negative credit for moving away.
Its history must be reinitialized on episode reset and whenever the pending
anchor changes, otherwise reset/target jumps become false progress. The
completion bonus should consume the gate's actual one-tick release event, not
reimplement its strict/relaxed/safety conditions; that keeps reward semantics
and reference advancement exactly aligned.

## Encode ladder support identity without treating rung numbers as magnitudes

A binary four-limb contact vector cannot distinguish two stable poses whose
limbs are all in contact but on different rungs. For an Oracle sufficiency
experiment, expose three separate relations: measured physical support,
support expected at the active reference frame, and support required at the
next stable anchor. Their raw identity convention is `-1` for no contact, `0`
for ground, and `1..N` for ladder rungs from bottom to top. The measured ID is
selected from pair-filtered contact forces and must never be copied from the
reference label.

Do not feed those integers as a bare scalar. For each limb use
`[has_support, is_ground, relative_rung/N]`. The rung coordinate is measured
from the lowest positive rung in the pending stable support tuple, and all
three relations use that same origin. This keeps the scale bounded, retains
vertical ordering, makes the representation invariant to an absolute rung
offset, and prevents a valid relative rung `-1` from being confused with the
raw no-contact sentinel because `has_support` remains explicit. The pending
four-limb XYZ targets remain continuous positions in the robot anchor frame.
An inactive limb still retains its reference position at the target stable
frame; only an environment with no pending target receives an all-zero XYZ
target.

## Event-driven contact goals with an SSR-style AMP motion prior

The first Oracle experiment converts each pair of consecutive ground-aware
stable anchors into one support task. The 29 anchors produce 28 tasks: 24
single-limb changes and four simultaneous two-hand changes. Reset samples
uniformly from every motion frame with a successor anchor (frames 0..3059),
uses the saved stable-support segment ID to select the next stable target,
and restores the sampled frame's reference velocities. Only the terminal
frame is excluded because it has no next target. The actor receives
direction, active-limb and maintained-support masks, four local target/actual
limb positions, local limb velocities, and three four-limb support-ID vectors:
measured physical contact, the sampled reference frame's expected contact,
and the next stable frame's contact. All use the raw shared convention
`-1=none, 0=ground, 1..7=rung`; there is no one-hot or separate binary
contact input. The reference expectation is held at the sampled frame until
the event-driven goal advances; it is not a running reference clock. Target
XYZ is retained for all four limbs, including those whose next support ID is
`-1`. With the existing 25x85 proprioceptive history and 30 Hz depth through
CNN+GRU256, the explicit task vector is 57-D (398-D action-MLP input).
The target is held until four consecutive physically correct, near, slow,
and root-safe steps. After two seconds only a brief old-support sensor drop
can be waived; a wrong rung or missing new contact cannot. Each task deadline
is `max(5.5 s, 1.25 * remaining reference duration + 0.5 s)` from its sampled
frame. A posture fall needs 10 consecutive control steps (0.2 s); a dangerous
non-limb/rail contact needs 20 (0.4 s). Their failure metrics are separate
from the deadline metric. Single-task training ends on completion; the
separate sequence task advances to the next support goal. Reference poses
seed resets and supply AMP expert examples, but no longer act as a
clock-following reward target.

Task reward supplies signed distance change, target reach/contact, maintained
support, slip/wrong-contact/danger penalties, and one-shot completion/failure.
Distance change is not clipped: clipping positive and negative steps
independently lets a slow approach plus fast retreat yield a false net gain.
IsaacLab multiplies reward terms by dt, so progress and event bonuses are
divided by dt before passing to RewardManager. The reward for a completed goal
is computed before the command switches; the first step after a switch has
zero distance-change reward.

AMP is trained alongside PPO, using SSR's local 71-D body-motion features,
five frames separated by two control steps, 512/256/128 ReLU discriminator,
least-squares expert/policy targets +1/-1, expert-input gradient penalty 10,
and rolling 100-batch normalization (variance epsilon 1e-4, clip 10).
The policy sees only the scalar AMP reward through PPO. Deliberate differences
from SSR are one ladder prior rather than terrain-specific MAMP, 20% repeated
zero-velocity stable holds in expert windows, rejection of reset-crossing
policy windows, and a style-reward weight ramp from 0 to 0.2 over 500 updates.
The SSR task config uses a fixed AMP reward coefficient of 1.0 and computes
the reward directly; it does **not** use this linear ramp. Our ramp is a local
precaution against the untrained discriminator's initial reward, not an SSR
requirement. SSR's and LadderMan's reward units differ, so setting both
coefficients to the same number would not make them equivalent.
These choices must be checked by no-AMP and hold-rate ablations if learning
fails; a short smoke test cannot establish policy quality.

PhysX GPU does not return pair-filtered contact force for `/World/ground`.
At frame 0 the feet had roughly 400-1500 N net ankle force while the filtered
ground force was zero; without a fallback the actor and gate saw both feet as
unsupported. The event task alone now labels a foot as ground ID 0 when ankle
net force exceeds the existing threshold, no rung-filter force exceeds it,
and the local foot point is within 0.12 m of the world ground height. Rung
identity still comes from its pair-filtered force. Playback at frame 0
reported `[-1,-1,0,0]`; at frame 373 it recovered hand IDs 4 and 5 while
keeping the two ground feet at 0. At frame 456 it preserved left-foot rung
ID 1 and right-foot ground ID 0. This heuristic needs re-evaluation if
ground height or platform geometry is randomized.

The task remains an Oracle information-sufficiency experiment: the four
target positions and measured contact flags currently come from simulation.
Real deployment requires a target planner from depth and contact estimation
from onboard sensing. The design isolates whether those compact signals make
the climb learnable before building their estimator. Success would not by
itself validate sim-to-real sensing.

## 2026-09-29 - Local support success is not continuous climbing

The run `2026-09-28_10-49-56_allframe_raw_support_ids_amp_8000_env5120`
used `single_task_episode=True`. Its final logged update (7279) reports
99.66% task-success terminations and a mean episode length of 17.66 control
steps (0.3532 s). This measures reaching one target from a random point
inside its segment. It does not test whether the resulting whole-body state
can start the next transition. Neither this metric nor a successful smoke
test establishes continuous climbing.

The default ContactTask and explicit ContactSequence environments now both
continue from the measured terminal pose of each intermediate goal. There is
no reset or robot-state write at a goal switch. The goal index advances once;
the next observation contains the new goal, and visual/GRU state persists.
Only the final support goal is successful termination. Falls, persistent
dangerous contacts, and missed per-goal deadlines still fail the episode.
Random reset still samples every nonterminal motion frame with its reference
joint/root velocities. It identifies the containing segment and targets the
next stable anchor. Reference time does not advance independently of success.
The 170 s global limit exceeds the 162.82 s sum of this motion's full-segment
deadlines. The old single-task ablation is available explicitly with both
`env.commands.motion.single_task_episode=true env.episode_length_s=10`.

New episode metrics distinguish `task_completed_count`,
`sequence_start_task_index`, `sequence_remaining_tasks`,
`sequence_completion_fraction`, and `sequence_success`. The fraction is
relative to goals remaining at reset, not all 28 goals. Successful completion
after resetting in the final segment must never be called a full climb.
Playback preserves terminal metrics before auto-reset and writes them per
episode under `contact_sequence` in `events.jsonl` / `summary.json`.

A deterministic, nominal-physics evaluation of model_7000 in the corrected
continuous task (8 random-start environments, 750 steps / 6000 transitions)
finished 98 episodes. 92 completed exactly one support conversion and six
completed two; none finished more than two. Of 13 suffix-completion episodes,
12 started in task 27 and one in task 26. The other 85 ended in posture failure
(66) or persistent collision (19), with no per-goal deadline failures. This
is direct evidence of a local-transition skill that does not compose into a
climb, rather than evidence that the network learned literally nothing.
The evaluation uses deterministic actions and disabled physical/sensor
randomization, so its rates are not interchangeable with stochastic training
rates. Artifacts: `outputs/contact_sequence_audit_20260929_random_verified/model_7000/`.

A second deterministic check starts all four environments at frame 0 for
400 control steps (1600 transitions). All 12 finished episodes complete
exactly the first support conversion, then fail in task 1: four posture
failures and eight persistent collisions, zero deadline failures and zero
sequence completions. These repeated nominal starts are not independent
randomized trials, but confirm that the current policy's weakness is the
handoff into the second segment. Artifacts:
`outputs/contact_sequence_audit_20260929_frame0/model_7000/`.

### Reward and task-design audit

- The event-task formulation defines observable intent: local limb targets,
  actual/current-expected/next support IDs, active and maintained limbs, and
  direction. Its 57-D task command plus current proprio85 and GRU256 gives a
  398-D actor input. Simulation supplies these oracle quantities; the visual
  estimator/planner deployment problem remains separate.
- Signed target-distance reduction (weight 2 per meter), physical correct
  rung contact, maintained support, slip/wrong-contact/collision penalties,
  one-shot +5 completion and -2 failure form a coherent support-transition
  objective. Event and distance-change terms divide by dt before IsaacLab
  multiplies by dt; there is no accidental 50x shrinking of these events.
  Old-goal reward is evaluated before goal advancement, and the first step
  after a goal switch cannot earn spurious distance progress.
- Contact requires force >5 N and the correct rung, with hand proximity
  5.0/5.2 cm, foot proximity 14.5 cm, limb speed <=0.25 m/s, safe low-speed
  root, and four consecutive eligible steps. This is a useful transition
  gate but does not certify a long-lived load-bearing grasp or a good
  whole-body pose. Continuous continuation now tests the consequences.
- Release targets (`rung_id=-1`) are included as actor XYZ inputs, but current
  progress/reach rewards mask them out and completion only checks release.
  There is no objective for where the free hand finally rests. This needs an
  explicit free-limb target/posture objective if a particular standing/hand
  configuration is required; AMP alone has not provided it in this run.
- Reach (0.5/s) and active-contact (1/s) give positive reward while holding.
  This can encourage staying near a goal when progressing is difficult.
  It is a design risk, not the observed main failure here: the continuous
  evaluation failed by falls/collisions, not deadlines. Reassess with
  per-goal dwell time before adding a time penalty or changing scales.
- Upright cosine >=0.5 and root height >=0.4 m are failure/safety limits, not
  body-pose shaping. Support-relative body clearance/orientation and a free
  hand target can be added without reintroducing a clock-following pose.
  A world-root ascent reward must not be applied indiscriminately because
  half of the task is descending.

### AMP formula parity does not imply equal PPO influence

The discriminator architecture, least-squares labels, expert-input gradient
penalty, five-frame/stride-2 windows, and rolling normalization follow SSR.
CPU parity tests cover discriminator reward/gradient penalty and normalizer
wraparound. The saved iteration-7000 sample has finite normalized inputs,
0.0048% normalization clipping, and exact alignment between each valid AMP
window's latest state and the next stored critic AMP state (max error 0).
These checks found no numerical or temporal alignment bug in that sample.

However, SSR's integration into PPO differs materially:

- SSR `storage/rollout_storage.py::compute_returns` computes a value/GAE and
  normalizes advantages separately for each reward group. Its
  `algorithms/ppo.py::get_surrogate_loss` combines those normalized advantages
  with group weights (task 1.0, AMP 0.2 in the inspected X2 config).
- LadderMan adds `0.2 * step_dt * style_reward` to the scalar task reward and
  then uses one critic and one GAE. The same written coefficient 0.2 therefore
  does not imply the same style influence. Multiplying by dt is consistent
  with IsaacLab reward-rate units; the mismatch is the relative scale against
  event bonuses and the absence of independently normalized style advantage.
- SSR does not ramp this AMP group weight. LadderMan's 500-update ramp is a
  local choice. At update 7279 it has long been fully enabled, so the ramp is
  not an explanation for the late plateau.

At update 7279 the expert/policy discriminator means are +0.906/-0.910,
classification accuracies are 100%/99.96%, and the valid-window fraction is
52.85%. The mean style reward on valid windows is 0.0821; the weighted reward
averaged over all steps is 0.0001735, compared with task reward 0.2947 per
step (about 0.059% by signed mean magnitude; this is NOT an exact policy
gradient contribution). Short episodes discard almost half the AMP windows
because real history after reset must accumulate. In the saved sample, 28.6%
of valid windows get zero style reward; joint-velocity RMS is 2.049 rad/s
versus 0.402 in the reference (different state distributions, diagnostic only).
The discriminator learned to distinguish the policy very well; the policy
did not learn the reference motion distribution. High discriminator accuracy
is not a policy-quality score.

The next AMP experiment should first make task/style influence separately
measurable, preferably with SSR-style separate value/GAE normalization, then
compare no-AMP and AMP in the same continuous environment. Simply extending
the old single-task run, enlarging the discriminator, or copying SSR's raw
coefficient cannot resolve these design mismatches. These are recommendations;
the present patch changes episode progression and diagnostics, not PPO/AMP
losses or task reward weights.


### 2026-09-29: SSR-style AMP integration and support-release shaping

- Keep task and AMP rewards as separate channels. For each channel, use its own critic output, GAE, and rollout-wide normalization; only then apply the actor surrogate weights (`task=1.0`, `AMP=0.2`). A scalar reward sum followed by one GAE changes the effective meaning of the AMP coefficient.
- `support_match` compares each measured support ID with the next stable support ID. Exact ID equality handles all cases in one representation: `-1` means release/no contact, `0` ground, and `1..7` rung. This complements `support_loss` (maintain old support) and `contact`/`reach` (move active limbs), so lifting a limb is not penalized when the next stable state explicitly requires `-1`.
- The reward is a task-state shaping signal, not a replacement for the completion gate: the gate still requires proximity, speed, safety, and temporal confirmation.

### 2026-09-29: Time-based sensor history contract

- Treat camera, proprioception, and command histories as timestamped per-environment rings. The camera runs at 30 Hz and stores only new exposures; the policy samples the nearest available frames at `t-0.4 s`, `t-0.2 s`, and `t` (chronological order).
- Proprioception and command are sampled independently every `0.1 s`, ten frames covering the preceding `0.9 s`; they are not the latest ten policy calls. The current proprio frame remains the last sampled frame.
- Reset must clear all three rings and recurrent state for only the reset environments. Repeating the current frame after reset is valid warm-up behavior; using pre-reset samples is not.
- The end-to-end actor contract is now `proprio [B,10,85]` plus depth `[B,3,60,80]`, with CNN channels `16/32/64`, 128-D proprio/depth branches, and GRU-256. Input dimensions are inferred by actor/storage/playback, while old checkpoint contracts remain rejected.

### 2026-09-30: Reset validity metadata, not only sensor samples

Clearing a timestamped ring's data without clearing its timestamps, count and
cursor leaves zeroed entries selectable as real history. For a direction command,
zero is a meaningful stop signal, not a harmless missing value. The command ring
had this bug: reset could create a false stop-to-climb history lasting 0.9 s.
Reset now clears all metadata and warm-starts history from the new episode's
first sample. Regression tests must inspect the command values after reset,
including buffer wraparound and direction changes; checking proprio/depth alone
does not establish isolation. A same-step partial reset must also leave other
environments' write cursors and histories unchanged.

Task names that share a visual encoder do not imply the same learning problem.
Check the saved manifest for event-driven progression, continuous episodes, AMP
activation and value channels. For the current contact-sequence experiment these
are `advance_reference=false`, `single_task_episode=false`, and `value_channels=2`.
Reference data still supplies reset states, support goals and AMP expert windows;
only its time-driven tracking clock is removed. Successful short execution tests
do not prove that the policy can learn the complete ladder sequence.

### 2026-10-08: Dynamic contact-phase command

The contact annotation contains both contact changes and dwell intervals. A
repeated ID means that the limb should keep that support; it does not mean that
the whole robot must stop. The phased task therefore tracks each active limb
through four states: `prepare` keeps the current source support, `release`
requests detachment, `move` permits flight, and `acquire` requests the target
support and then protects it. The actor's expected contact is updated from this
execution state rather than from reference frame time.

For every limb, support IDs use `-1` for no contact, `0` for ground, and positive
values for rungs. The actor receives measured IDs, the dynamic expected IDs, the
next stable IDs, local next-target positions, active/protected flags, direction,
and a four-value phase code. This is a 61-D task field appended to the current
proprioception and GRU feature (402-D actor input). A sampled reset initializes
the phase from the reference annotation but does not replay release/contact
bonuses that happened before reset. Each milestone is confirmed over consecutive
steps and paid once; after acquisition, a sustained loss is penalized while a
short contact-sensor dropout is tolerated.

The phased reward removes continuous reach/contact/support-match bonuses that
could reward waiting at a goal. It uses signed distance reduction, one-shot
release and acquisition events, phase contact error, protected-support loss,
slip, wrong-rung contact, danger, and completion/failure terms. Intermediate
completion changes only the task goal and preserves the robot and GRU state;
only the final goal terminates success. The current implementation is an oracle
diagnostic: the phase/contact fields are simulator annotations and are not yet a
deployable visual estimate.

Validation status: the phase machine has four Isaac-free regression tests and
passes static compilation. Full IsaacLab tests require the Omniverse `pxr`
runtime and were not run in the plain shell. A short environment smoke run is
still required before any 5120-environment long training; old 398-D checkpoints
must not be loaded into the new 402-D task.

### 2026-10-08: Observable body goals and phase advancement without circular waits

Limb contact IDs specify the support relation but not the torso's progression.
A next-stable-frame torso goal is a useful additional oracle command: feed its
3-D error in the current torso frame to the actor and reward small signed
distance reduction. Do not add a hidden world-position target only to the
reward, recreating the original unobservable-intent problem. The current
motion stores link origins, not mass-weighted whole-body COM, so this experiment
uses the `body` anchor and must not be described as COM control.

The implemented shaping is `0.5 * (e_previous - e_current)`, where
`e = max(distance_m - 0.10, 0)`. It stays small and constant across all phases;
there is no dwell bonus and no phase-dependent potential jump. Retreat costs
reward; a closed excursion cancels in the undiscounted shaping sum (this is
not a claim of discount-invariant potential shaping). The body goal is fixed
within a support segment and advances only on task completion. Its input
extends task61/actor402 to task64/actor405, under a separate checkpoint contract.

Phase tracking and safe reward eligibility are different questions. If an
active limb has physically detached, refusing to recognize that because root
velocity is temporarily too high leaves the command stale. Update the physical
phase after sensor confirmation, but keep unsafe detachment unrewarded and
retain safety penalties. Similarly, a sustained early detachment in PREPARE
must not demand recontacting an obsolete rung forever. A second active limb
already in flight must not create a circular prerequisite for preparation;
required established supports still matter. Phase labels express contact
intent, not a controller that freezes the whole robot or requires visiting
every intermediate label.

These fixes remove specific logical stalls, not every possible learned waiting
behavior. The shared segment deadline remains
`max(5.5 s, 1.25 * remaining reference duration + 0.5 s)` and is not restarted
on phase changes. Thus an unfinished phase eventually fails/reset instead of
waiting indefinitely. Continuous goal switching, finite deadlines, one-shot
events and progress rewards make a coherent training problem, but they do not
prove dynamically feasible weight transfer or full-sequence policy learning.
AMP supplies a motion prior, not a contact planner or a stability guarantee.

The first body-goal experiment deliberately adds no hard torso completion gate.
Check `body_completion_error_m` together with `task_completed_count`, phase
residence times, per-segment successes/failures and protected-support losses.
In particular, release-only goals may complete before torso/free-limb return
is finished. If that occurs in physics, the completion criterion needs an
evidence-based revision; silently forcing every phase to reach an exact
reference posture risks reintroducing stalls. Final-body approach can also
conflict with a necessary initial counter-movement, hence its low weight.

Reference still provides resets, discrete goals and AMP expert data; no
time-indexed tracking trajectory is restored. The goal/contact inputs remain
oracle fields. Real deployment needs an observable support estimator and a
planner for torso/limb goals; adding vision does not make these fields deployable.

Validation: 149 CPU tests passed, one Isaac Sim config module skipped, and
static compilation passed. Tensor tests include all 28 reference transitions,
timeouts and PPO/AMP updates, not physical execution. The current execution
environment has no usable CUDA device, so a real reset/step and short-training
smoke test is still required before large-scale training. No long run started.

### 2026-10-10: Swing-foot posture and wrong-surface contact are separate signals

Correct rung identity does not establish correct contact geometry. A foot
touching the target rung's underside can still have the correct contact ID.
The foot-only reward experiment therefore penalizes horizontal/downward filtered
normal forces against all rungs, including the target, while preserving the
hand contact logic. Upward force is an approximate top-face cue, not proof of
an adequate sole support patch; summed multi-face forces can obscure an edge
collision. Raw IDs must continue to report physical identity rather than be
silently rewritten to mean successful support.

Match posture shaping to the observed failure phase. Excessive toe-up was seen
during lifting, so the new term covers release, flight and unconfirmed landing,
including early detachment before the phase machine catches up. Penalizing only
the established stance cannot teach the preceding swing to clear the rung.
Do not require every swinging foot to stay flat: this motion's successful foot
swings have roughly 6-37 degrees (left) and 3-46 degrees (right) of toe-down.
The initial experiment permits pitch [-55, +15] degrees and lateral tilt within
20 degrees, penalizing excess smoothly with a per-foot cap. Positive pitch is
defined geometrically as toe above heel, not the sign of an ankle joint angle.

Strong aversion to collisions does not imply an arbitrarily large coefficient
or immediate termination. Excessive penalties can make staying still preferable
to trying a difficult lift. The current rates are -3 per offending foot/second
for each wrong-surface category and at most -1 per swinging foot/second for pose.
Duration, force peaks, swing angle peaks and per-task success/timeout must be
read together: fewer collisions without more successful foot transfers may be
avoidance of motion rather than improved clearance. These penalties specify
which behavior is undesirable; they cannot by themselves guarantee discovery
of an outward-then-upward foot path. Space-based waypoints remain a separate
follow-up if successful exploration stays rare.

The new FootSafety task preserves actor405 and existing task/AMP contracts for
a controlled reward comparison. It adds penalties only; an explicit top-face
completion gate has not been introduced. Tests verify the tensor logic and
reference-pose tolerance (104 passed, one runtime-dependent module skipped),
but actual physical clearance and learned behavior still require GPU playback
and short training validation.

### 2026-10-12: AMP score validation

An offline score audit used the saved discriminator, normalizer and AMP window
contract from the 10,000-iteration FootSafety run. Reference windows averaged
0.844 style reward. A separate deterministic visual-teacher rollout that
completed 8/8 recorded episodes averaged 0.935, including 0.930 on each of
the difficult left-foot upward transitions. The latest student diagnostic
windows averaged 0.123; task 06 averaged 0.300 and task 10 averaged 0.109 in
PREPARE and 0.016 in RELEASE.

Therefore AMP is not disconnected and it does prefer the available reference
and successful-teacher motion. The audit does not establish that it ranks a
correct student swing above an incorrect student swing, because the saved
short diagnostic window has no per-transition success label. The teacher trace
also belongs to the earlier teacher-only visual task. A controlled paired
success/failure evaluation is required before treating a larger AMP weight as
the fix; the current evidence points to missing phase-conditioned spatial
guidance as the main gap.

### 2026-10-12: Time-aligned reference reward should be staged and separately valued

The current FootSafety task has no direct time-indexed reference-tracking
reward. Reference motion supplies stable-support goals, target positions and
AMP expert windows; `advance_reference=False` prevents a continuously running
reference clock. The task reward remains event/progress based, while AMP is a
phase-free style prior.

A later reference-tracking term is plausible, but a global clock must not be
turned on abruptly. Use a local clock for the current support transition,
allow a small reference-frame window, and gate its advance by actual task
progress/safety. Ramp its policy weight only after the difficult transitions
show nontrivial success; keep task completion and support safety active.

The PPO design can carry three independently estimated value channels:
`V_task`, `V_amp`, and `V_reference`, with independently normalized GAEs and
actor weight `A_task + w_amp A_amp + w_reference A_reference`. A separate
reference critic prevents return scale and bootstrapping interference, but it
does not make the reference term behavior-neutral: any positive
`w_reference` changes the actor gradient. Setting that weight to zero in early
training is the way to preserve the original task policy, followed by a
measured ramp rather than a fixed late-iteration jump.

### 2026-10-12: First whole-body reference trial keeps the existing two-value PPO

The selected experiment does not implement the earlier proposed reference
weight schedule or third critic. Six fixed-weight tracking terms are added
directly to task reward: lower/upper body position, lower/upper joint pose,
swing ankle position and torso orientation. AMP remains the separate second
reward/value channel. The original event task, contact authority and actor
inputs are retained, so imitation can influence the same actor gradient and
compete with task completion; there is no independence guarantee.

Reward guidance uses its own phase-bounded local clock, not the task's frozen
reset/reference index. All body/joint terms must use one common reference
frame. The matching window is now +/-3 frames at 50 Hz (+/-0.06 s; reduced
from +/-5 after the initial implementation). Selecting
each limb's easiest frame separately would reward an invented whole-body pose.
The matching frame only selects rewards and never awards completion or moves
the clock. No phase-compatible candidate means no imitation reward that step,
not a forced task advance or immediate reset.

Do not freeze a release target at the old stance until the policy has already
discovered how to detach. Limited post-release anticipation supplies lift
guidance. Conversely, RELEASE must retain room for preparatory weight transfer:
the reference sometimes releases two hands at different times while the
physical phase machine can request both releases together. All actual limb
phases constrain a shared clock; established support cannot match past flight.
Synthetic replay of all contact sequences checks for logic deadlocks, but
cannot establish dynamically feasible full-body execution.

The new variant gives each support goal at least 10 s, extended to twice its
remaining reference duration plus 1 s for long segments. Subphase changes do
not renew this budget, and the full-episode limit is 320 s. More time allows
correction, but positive pose rewards also make waiting potentially valuable.
Track difficult-transfer success, duration, falls, reference pauses/conflicts
and reward composition together; a higher tracking reward alone is not success.
Fine reference progress is still absent from the actor input, so observability
remains a limitation of this reward-only ablation.

Implementation has 85 passing CPU tests, including all reset frames and all
28 contact sequences, with Isaac Sim config tests skipped. Physical simulation
and short training have not been verified in the current GPU-less environment.

Follow-up on 2026-10-12: narrowing the matching window to +/-3 frames leaves
the separate five-frame release/landing anticipation unchanged. The revised
CPU suite passed 86 tests. With approved execution outside the restricted
GPU-less context, the actual Reference task completed a 32-env, 10-iteration
GPU smoke run, with finite diagnostics, nonzero visual/GRU PPO gradients and
all six rewards active. Actor405 and the two-value task/AMP layout were verified
in saved configuration. This validates the runtime path, not climbing skill;
the run ended normally and no long training was started.

### A phase-bounded reference can still reward an unresolved release

The 2026-10-13 deterministic physics audit found real source-foot loading,
not merely a stuck state-machine counter: left-foot tasks held roughly
291-372 N until timeout while the desired contact was -1. Four-frame release
recognition worked in the inspected traces. Old actors could complete the
ground->first-rung and third->second-rung fixtures in the same new environment.

Pausing reference progress does not pause positive pose rewards. A held-source
trajectory in the new task accumulated about 7.7-8.6 reference reward over a
failed 10 s goal, outweighing roughly 4.2-4.4 units of other costs including
the terminal failure. Window matching repeatedly selected the earliest allowed
frame; similar whole-body poses can score highly without actual unloading.
Therefore explicit desired contact plus reference pose reward does not by
itself make leaving the old support worthwhile locally.

This does not prove that waiting is globally optimal: actual successful old
policy trajectories had higher total and discounted returns. Nor do successful
airborne-reset cases prove a complete swing was learned; reset supplies pose
and velocity. The actionable distinction is positive unproductive dwell versus
useful reference-guided progress. Test bounded/no-progress-conditioned reference
payout before strengthening AMP or all penalties, preserving legitimate support
maintenance and avoiding time-forced physical contact transitions.

Evidence and exact limitations are in `outputs/release_check_20261013/` and the
implementation log. No reward, termination, contact threshold, phase transition,
or learning-rate changes were made during this diagnostic.

### Pause-conditioned reference payout, not a shorter exploration deadline

The next controlled ablation keeps the full-body reference kernels and task
logic, but pays reference shaping only while its local clock makes progress
or is briefly paused: full payout for 0.2 s of pause, linearly reduced to zero
over the next 0.5 s. A genuine clock advance or completed-goal handoff restores
shaping; sensor flicker and re-matching a pose do not. Existing support, safety,
progress and milestone rewards keep operating, and the robot retains its full
goal deadline to try releasing after reference shaping has stopped.

This removes repeated positive reward for an unresolved reference pose without
forcing the physical task forward in time. Fixed-start old-policy playback
confirmed zero applied reward while raw pose similarity remained high. It does
not establish that the policy will discover unloading or obstacle clearance:
that requires a fresh training comparison and per-stage evaluation. Preserve
the original task as the baseline; do not conflate fading with a per-step cap,
delayed reference curriculum, stronger AMP, or shortened terminal deadlines.

The model3500 follow-up distinguishes three different failures: physically
loaded feet retained in RELEASE, early unloading accompanied by loss of balance,
and ACQUIRE waiting despite correct rung IDs because an old hand is outside its
position tolerance. In the first case, paused reference payout was already zero;
removing a dwell incentive is not equivalent to learning weight transfer. In the
second, bad-surface force may follow loss of balance rather than cause it. In the
third, contact identity alone is insufficient under the existing completion
rule. Keep nominal fixed-start fixtures separate from random-reset training rates
and from continuous climbing with a warmed recurrent history. See
`outputs/ascent_check_20261013_model3500/summary.json` for the bounded evidence.

## Contact-sequence success is not completion of the standing motion

The model6500 frame0 playback on2026-10-14 completed28 support transitions in
813 steps, but its final reward-reference clock was2827 rather than3060.
Current completion checks position/speed only for limbs whose target ID is
nonnegative. Free hands only need ID-1; torso height, posture and return-hand
position are shaping terms, not completion requirements. Consequently top
hand release can hand off immediately to regrasp, and final hand release can
terminate before standing upright. Correct support IDs do not encode standing
up or holding that posture. These need explicit task completion/hold semantics,
with a visible standing intent, not merely larger reference or AMP weights.

Full-body reference rewards are soft rate terms (upper bound1.10/s before
pause decay); this playback received5.746 reference reward versus140 from28
completion events alone. The existing foot position target does not directly
constrain sole orientation, and joint-group tracking can trade local errors
against other objectives. Preserve the release-stall safeguard when improving
quality; do not confuse an intentional standing hold with stalled RELEASE.

AMP discrimination accuracy alone cannot establish that its policy signal is
unused: task and style advantages are independently normalized and mixed1:0.2.
Weight ablations should use a fixed evaluator or matched-stage motion examples,
since the jointly trained discriminator's score scale changes during learning.

## A connected visual branch is not evidence of useful vision; test it in closed loop

On 2026-10-15, depth-only interventions on the trained PauseDecay policies
separated network sensitivity from task performance. With all nonvisual inputs
retained, model9999 completed the remaining contact sequence from 32/32 fixed
fixtures with normal depth, versus 0/32 for a constant 1 m or 2 m image, 2/32
for frozen reset images, and 3/32 for images from mismatched stages. Normal
depth completed all five frame-zero replicas; none of the corruptions did.
Model6000 scored 26/32 normally and 3/5 from frame zero, so unchanged training
after 6000 had already produced a meaningful baseline improvement.

Same-state counterfactual actions used exactly the same proprioception, task
inputs and pre-action GRU hidden state; a normal-input identity control agreed
within 4e-6. Closed-loop tests separately evolved each condition's recurrent
state. Do not zero the entire fused GRU feature to test vision: that would also
remove the proprioceptive-history contribution.

This establishes strong dependence on correctly updating visual input under
the tested nominal ladder, not geometric understanding or robustness to new
ladders. Constant images are out of distribution; frozen and mismatched real
images strengthen the evidence but still break normal sensor/state consistency.
The actor continues to receive 64D privileged contact/goal/phase information.
These fixtures are not independent deployment trials, and sequence completion
still uses the existing permissive standing completion semantics.

For a reference-weight fine-tuning experiment, compare matched additional
updates from the same checkpoint, not only total reward (whose definition has
changed). Uniformly scaling all six reference weights preserves their shared
candidate-frame preference, unlike selectively changing the weights. Keep the
pause-decay safeguard and other task/AMP settings fixed first. Bigger weights
cannot by themselves add a missing sole-orientation target or make final
standing posture a requirement of task completion. Full test evidence is in
`outputs/depth_ablation_20261015/full/model_9999/summary.json` and its model6000
counterpart; no reward changes or training were started during this audit.

The subsequent user-requested Weight2x variant preserves the original task and
changes only the six reference weights. A bounded resume test verified that
the original model6000 restores PPO/AMP state and continues at iteration6001.
The task and reference rewards still share one value channel; AMP retains its
separate channel. A transient change in task-value error is expected after
changing the reward objective and is not itself evidence of a broken resume.

## Diagnose toe-up with physical phase, geometry and reward eligibility separately

The 2026-10-16 per-step audit of PauseDecay model9999 and Weight2x model10000
found that ACQUIRE does not disable the foot-pose penalty. Across seven nominal
first-episode fixtures per model, all 140/153 upward toe-excess samples in
ACQUIRE before acquired confirmation retained that penalty. Of 334/358 upward
toe-excess samples, only 1/0 had any pause fade and none had zero fade scale.
Some reference rewards were instead invalidated by safety gating. Do not infer
that reference shaping is absent merely from its pause-decay design.

A separate timing gap matters: task_settle_steps=10 gates phase updates after
EVERY goal handoff, not only a physical reset. The actor can already move during
those 0.2 s. In these traces, 260/330 and 255/352 upward airborne foot samples
still had PREPARE as the issued phase. Physical ascent transitions took roughly
0.32-0.42 s versus reference segment durations of 1.58-1.90 s. Some feet jumped
from PREPARE directly to ACQUIRE. The explicit airborne foot-pose penalty still
operates, but reference_swing_foot is zero in PREPARE and joint-reference targets
can remain in the old-support part of the motion. Actual unloading recognition
and contact/reward confirmation should not be conflated with reset settling.

In model10000's frame0 right-foot 1->2 transition, toe-up reached32.7deg in
ACQUIRE with reference scale1 and the pose penalty active. The sole marker was
0.27cm below the target top plane and a front-bottom mesh marker6.93cm above it;
leveling the foot at the same ankle position would put the sole3.90cm below the
plane. These are geometric markers/kinematic comparisons, not contact points
or a causal ablation. They support investigating insufficient ankle clearance
and inclined edge landing rather than only increasing pose-reward weights.
The current effort term is squared applied torque, not mechanical energy;
joint acceleration/action smoothness and joint limits also remain active.
Evidence: outputs/toe_up_audit_20261016/{reference1x/model_9999,
reference2x/model_10000}/{trace.pt,foot_steps.csv,analysis.json}. These overlapping
fixtures are diagnostic coverage, not independent deployment trials.

The same-day repair separates physical-reset age from per-goal age. A dedicated
`_reset_steps` counter is cleared only when `_resample_command` actually resets
the robot; `_set_task_goals` does not clear it. Do not substitute IsaacLab's
`episode_length_buf`: RSL deliberately randomizes that buffer at training start.
Both the phase-update gate and completion-eligibility gate use the reset-only
settling period. Per-goal deadlines, debounce, four-step safe confirmation,
reward weights and reference clock rules are unchanged.

With unchanged Weight2x model10000, seven nominal first episodes still completed
the existing contact sequence. In frame0 ascent, MOVE began at goal steps8-9
instead of13 (one old transition skipped MOVE entirely). Airborne PREPARE
samples fell from255/352 to164/349 across the overlapping fixtures; they are not
zero because the original contact confirmation delay remains. Toe-up did not
improve in this no-update replay, and some peaks increased. Removing a stale
phase freeze is a consistency fix, not a trained clearance/pose correction.
Evidence: `outputs/phase_handoff_fix_20261016/reference2x/model_10000/`.

## Quality fine-tuning: control regularization is not a clearance objective

On 2026-10-16, add an isolated reference-1x fine-tuning task from the original
`2026-10-13_11-03-28_reference_pause_decay_scratch_env5120_iter10000_seed42/model_9999.pt`.
Double action-rate, lower-body action-second-difference, upper/lower joint
acceleration and applied-torque-squared penalties. Keep soft joint limits,
contact safety, reference rates, AMP advantage weight and network unchanged.
The earlier physical-reset-only phase-settling repair remains in effect; this
experiment is therefore not a reward-only comparison against the historical
rollout collected before that repair.

Add a small mechanical-work proxy: the reward function returns
`P = sum_i abs(applied_torque_i * joint_velocity_i)` in watts. RewardManager
applies `-5e-5 * P * control_dt` once. Absolute value must be per joint, not
after summation, to avoid cancellation of driving and braking work. These are
PD-applied torques, not actor position-target actions. This is not electrical
energy, copper loss or a regenerative-braking model; static support still has
zero mechanical power and is regularized by the retained torque-squared term.

The earlier seven-fixture nominal reference-1x diagnostic measured mean power
643.28 W, implying a new mean penalty rate of -0.03216; the same traces had a
reference reward rate of +0.34544. This calibrates an intentionally small
starting weight, not an estimate of training-wide power or future improvement.
Both older task variants remain unchanged, and the new task has its own log
directory. Check task completion and phase stalls alongside smoother actions
and power: excessive effort penalties can favor not moving. Less jitter is not
proof that toe clearance, foot pitch or the final standing motion is fixed.

The user corrected this fine-tuning plan: do not start with doubled costs.
Use the independent `Reference-PauseDecay-RegRamp-Energy-AMP-v0` variant instead.
For `k` completed PPO updates since starting this fine-tune, the five original
control-regularizer weights use `multiplier = 1 + 1.5 * min(k / 2500, 1)`.
Train 4500 additional updates: 2500 ramping from1x to2.5x, then2000 at2.5x.
The mechanical-power coefficient stays fixed at -5e-5; reference, AMP and safety
remain fixed. The initial fixed-2x variant is retained only as a separate option,
not the requested schedule.

Reward scheduling must use the actual resumed PPO update counter, not episode
age, parallel-environment count, or the checkpoint filename. An old checkpoint
starts a new schedule at its saved `num_updates`; a scheduled checkpoint retains
the original start counter. Apply absolute baseline-weight times multiplier
before collecting each rollout, with no changes inside it. Record elapsed
updates, multiplier and each applied weight beside PPO metrics, and preserve
the schedule/config in checkpoints. This avoids reset-dependent curricula,
double multiplication and restarting the ramp after an interruption.
