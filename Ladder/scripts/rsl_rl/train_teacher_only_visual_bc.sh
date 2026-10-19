#!/usr/bin/env bash
# Activate rl-training first. Extra arguments override the defaults below.
set -euo pipefail
ladder_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ladder_repo_root"

motion_file="Ladder/motions/zdw0824_capture1/pitchminus3_yawminus4_xminus030/37_visual_up_down_stop_command/zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_reset5mm_visual_up_down_stop_50hz_motion.npz"
teacher_checkpoint="logs/rsl_rl/cr1_nonvisual_zdw0824_no_command_full_state_middle_stand/2026-08-28_18-07-30_stage36_two_anchor_20_10_35_35_frame0_nominal_to10000/model_9999.pt"
mkdir -p logs/teacher_only_bc_launch
launch_log="logs/teacher_only_bc_launch/$(date +%Y-%m-%d_%H-%M-%S)_$$.log"
echo "Console log: $ladder_repo_root/$launch_log"

python -u Ladder/scripts/rsl_rl/train.py \
  --task Ladder-Climb-CR1-ZDW0824-UpDown-TeacherOnly-Visual-BC-v0 \
  --motion_file "$motion_file" \
  --teacher_checkpoint "$teacher_checkpoint" \
  --num_envs 5120 \
  --max_iterations 10000 \
  --seed 42 \
  --run_name teacher_only_bc_teacher9999_5120env_10000iter \
  --headless \
  agent.save_interval=500 \
  "$@" 2>&1 | tee "$launch_log"
