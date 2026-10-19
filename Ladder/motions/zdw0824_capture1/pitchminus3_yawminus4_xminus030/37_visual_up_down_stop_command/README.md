# Stage 37: Visual up/down/stop command annotations

Source:

`../36_middle_default_standing_100_zplus5mm/zdw0824_capture1_stage36_train_ready_middle_default100_zplus5mm_reset5mm_50hz_motion.npz`

The reusable tool `Ladder/scripts/annotate_cr1_climb_commands.py` copies all
187 Stage-36 fields exactly and appends 30 command/diagnostic fields. It does
not modify robot poses, velocities, contact phases, or reset-validity data.

## Inherited reset audit fields

Stage 36 already contains 37 fields whose names begin with `reset_`. Stage 37
copies them unchanged. They are stored motion metadata, not 37 policy
observations and do not increase the actor or critic input dimensions.

- Nominal-ladder frame verdict and collision audit (7):
  `reset_valid_mask`, `reset_valid_frame_indexes`,
  `reset_invalid_frame_indexes`, `reset_max_penetration_depth`,
  `reset_penetration_pair_count`, `reset_worst_robot_body`, and
  `reset_worst_environment_geom`.
- Whole-randomization-envelope summary (3):
  `reset_valid_mask_ladder_geometry_envelope`,
  `reset_valid_frame_indexes_ladder_geometry_envelope`, and
  `reset_invalid_frame_indexes_ladder_geometry_envelope`.
- Geometry ranges and per-corner/grid/cell diagnostics (16): fields 11--26,
  from `reset_ladder_inclination_delta_range` through
  `reset_ladder_geometry_envelope_worst_environment_geom`.
- Detector provenance and thresholds (11): fields 27--37, from
  `reset_detection_source_file` through
  `reset_detection_ignored_robot_bodies`.

At runtime, reset sampling primarily consumes `reset_valid_mask`, the two
geometry-grid axes, and `reset_ladder_geometry_cell_valid_mask`. The index,
penetration, worst-collision, and detector-provenance fields exist so an
excluded frame and the offline detector configuration can be audited.

- Frames: 3061 at 50 Hz.
- Up command: frames `0:1550`.
- Down command: frames `1550:3061`.
- Natural stop keyframes: one minimum-base-speed frame from each of the 18
  contiguous four-limb-contact segments.
- Additional legal stop ranges: initial default stand `0:25`, ladder-top
  default stand `1500:1600`, and the existing terminal stable-ground interval
  `2991:3061`.
- Direct-hold candidates additionally include geometry-valid representatives
  of those three ranges, so every randomized ladder cell has a valid target.

Artifact:

`zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_reset5mm_visual_up_down_stop_50hz_motion.npz`

SHA-256: `f0ab0814b30e991edfd6e097d586e67d62ca81f598ca03d649447e090ea42038`
