# CR1 TransformerVision 5000-iteration accepted baseline

This directory freezes the code and software-environment record associated with the accepted visual-only pretraining result. It exists so later task-environment changes do not erase the ability to inspect this result.

## Fixed result

- Visual checkpoint: `../model_5000.pt`
- Training summary: `../summary.json`
- Training log: `../train.log`
- Frozen nonvisual actor: `/home/zhangdawei/LadderMan/logs/rsl_rl/cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs/2026-07-31_14-01-06_deploy_obs_from_scratch_to7000/model_6999.pt`
- Task: `Ladder-Climb-CR1-Reference-ForceLimited-StrictFrontSide-DeployObs-WarpTransformerVision-v0`
- Data contract: `cr1_field_normalized_proprio_rung_weighted_dynamic16_future_elu_v2`
- Training: 4096 environments, 24 control steps per iteration, 5000 fresh iterations, environment minibatch 256, learning rate 0.001, gradient clipping 1.0.

The checkpoint contains `model_state_dict`, `optimizer_state_dict`, `iteration=5000`, the data-contract name, and the frozen-actor checkpoint path. The frozen actor was unchanged during training.

## Frozen files and environment

- `ladder_visual_source_2026-08-18.tar.gz`: current `Ladder/source/ladder/ladder`, `Ladder/scripts`, and `Ladder/tests`, excluding Python caches.
- `rl-training-environment.yml`: no-build Conda environment export.
- `rl-training-explicit.txt`: exact Conda package URLs.
- `SHA256SUMS`: integrity hashes for the source archive, visual checkpoint, summary, and frozen actor.

Recorded runtime:

- Conda environment: `rl-training`
- Python: 3.11.15
- PyTorch: 2.7.0+cu128
- CUDA build: 12.8
- Gymnasium: 1.2.0
- GPU: NVIDIA RTX5880-Ada-48Q, 49152 MiB
- NVIDIA driver: 570.172.18

## Inspect the accepted result live

This runs the old frozen actor, loads the accepted visual checkpoint without optimizer updates, and updates the depth/target/prediction dashboard while the robot moves:

```bash
cd /home/zhangdawei/LadderMan
conda activate rl-training

TERM=xterm ./isaaclab_v2.1.0/isaaclab.sh -p Ladder/scripts/check_cr1_transformer_vision_pretrain.py \
  --checkpoint /home/zhangdawei/LadderMan/logs/rsl_rl/cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs/2026-07-31_14-01-06_deploy_obs_from_scratch_to7000/model_6999.pt \
  --vision_checkpoint /home/zhangdawei/LadderMan/logs/rsl_rl/cr1_transformer_vision_pretrain/2026-08-17_16-33-26_fresh_5000/model_5000.pt \
  --num_envs 1 --steps 300 --display_env 0 \
  --device cuda:0
```

Dashboard controls:

- `H`: cycle 3-D overlay through target, prediction, and hidden.
- `R`: manually reset the environment and the recurrent state.

## Reproduce visual-only training

Use a new output directory. Do not point this command at the accepted run directory.

```bash
cd /home/zhangdawei/LadderMan
conda activate rl-training

TERM=xterm ./isaaclab_v2.1.0/isaaclab.sh -p Ladder/scripts/rsl_rl/train_transformer_vision.py \
  --headless --device cuda:0 \
  --checkpoint /home/zhangdawei/LadderMan/logs/rsl_rl/cr1_wood_step_reference_force_limited_strict_front_side_deploy_obs/2026-07-31_14-01-06_deploy_obs_from_scratch_to7000/model_6999.pt \
  --num_envs 4096 --rollout_steps 24 --iterations 5000 \
  --learning_rate 0.001 --max_grad_norm 1.0 \
  --mini_batch_envs 256 --save_interval 500 \
  --output_dir /home/zhangdawei/LadderMan/logs/rsl_rl/cr1_transformer_vision_pretrain/REPRODUCE_WITH_NEW_NAME
```

## Important scope

This baseline is visual-only pretraining under the July 31 `StrictFrontSideDeployObs` environment. It is not the later visual-latent PPO task, and it is not based on the August 10 thin-rail task. Future changes to actor observations, critic observations, rewards, rail geometry, or contact sensors must use a new task/config class and must not overwrite this baseline.
