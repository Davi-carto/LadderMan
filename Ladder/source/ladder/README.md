# Ladder

Isaac Lab extension for the LadderMan first-stage expert task.

The current implementation is intentionally narrow:

- locally implement CR1 robot/action/motion-tracking components adapted from IsaacLab and `whole_body_tracking` references;
- keep Ladder runtime code independent from `whole_body_tracking`;
- load the fixed CR1 wood step ladder used by the reference replay;
- expose ladder geometry and hybrid contact target helpers for state-based expert training;
- preprocess one upward reference motion into `contact_indicator` and `contact_rung_id` labels;
- register `Ladder-Climb-CR1-v0` while retaining the earlier G1 task.

The first version intentionally excludes up/down commands and phase/time scaling. Its target is one annotated upward motion, fixed timing, and a successful single-pass expert on Ladder A.

Annotate a WBT/IsaacLab-style motion first:

```bash
python Ladder/scripts/preprocess_ladder_motion.py \
  --input_file /path/to/raw_motion.npz \
  --output_file /path/to/ladder_motion.npz
```

Use the local RSL-RL entry points rather than `whole_body_tracking` scripts:

```bash
./isaaclab_v2.1.0/isaaclab.sh -p Ladder/scripts/rsl_rl/train.py \
  --task Ladder-Climb-CR1-v0 \
  --motion_file /path/to/ladder_motion.npz
```
