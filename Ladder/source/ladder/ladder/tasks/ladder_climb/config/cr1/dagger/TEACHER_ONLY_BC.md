# Teacher-only visual BC 对照实验

## 启动

```bash
cd /home/zhangdawei/LadderMan
conda activate rl-training
bash Ladder/scripts/rsl_rl/train_teacher_only_visual_bc.sh
```

从零训练，5120 环境、10000 次更新、每 500 代保存一次。固定使用原 stage36
`2026-08-28_18-07-30_stage36_two_anchor_20_10_35_35_frame0_nominal_to10000/model_9999.pt`
teacher 和 stage37 up/down motion。不加载旧 student。模型编号从 0 开始，最终是
`model_9999.pt`。启动脚本支持追加标准 train.py / Hydra 参数覆盖默认值。

新 task：`Ladder-Climb-CR1-ZDW0824-UpDown-TeacherOnly-Visual-BC-v0`。
原 Pure-Visual-Dagger task 的 BC 预热 + 25%/75% 混合策略保持可用。

## 保持不变与实验变量

- 每一步、每个环境只执行 frozen teacher 的动作均值；不采样 student，也不在后期切换控制器。
  这是 teacher-distribution behavior cloning，不是 student-rollout DAgger。
- 网络不变：student 使用 9×85 本体历史（含上下 command）和 2×60×80 深度历史；
  CNN 特征直接与本体历史拼接输入 MLP，无 reference actor 输入，无 decoder/重建/速度预测分支。
- 损失不变：`2 * mean(KL(student || teacher)) + mean((feature_s-feature_t)^2)`。
  KL 对样本和 25 个动作维度平均；feature 对样本和 256 维平均，不额外标准化/白化。
  feature 是各自最终 action Linear 之前的 ELU 输出。2:1 是系数比，不是实际 loss 或梯度占比。
  CNN、MLP、action head、log_std 都由这两项损失更新，teacher 无梯度，无 RL loss。
- 保留现有 valid-frame / adaptive reset、随机化、动作处理和终止条件，不修改任务设计。
- 5120 环境全部由 teacher 控制，其中 env 0–5055 用于训练，5056–5119 为固定留出验证环境。
  验证环境的整条 episode 不进入 minibatch；不随机拆分相邻帧。验证仍使用同一 motion
  和共享 adaptive sampler，因此不是跨 motion 泛化测试。

## 数据与解释

运行目录：`logs/rsl_rl/cr1_zdw0824_up_down_teacher_only_visual_bc/<run>/`。

| 位置 | 内容 |
| --- | --- |
| `params/` | 实际环境与训练配置 YAML/pickle |
| TensorBoard `BC/train/*`, `BC/validation/*` | 采集时、更新前的 student 拟合指标；两组使用相同参数时点 |
| `teacher_bc/metrics.jsonl` | 每代全部 BC 指标、逐关节 RMSE、上下行/停/初始历史/动作变化/10 段 reference phase 的样本数和误差 |
| `teacher_bc/episodes.jsonl` | 每条结束 episode 的 env、train/validation、episode ID、起始帧、终止前 reference 帧、长度、全部终止原因、接触/跟踪/几何等现有环境指标 |
| `teacher_bc/snapshots/` | 第 0 代及每 500 代保存最多 2 个训练 + 2 个验证环境的整段 rollout：输入、teacher obs/标签、student 均值/feature、两者 std、reference 接触、done/终止原因 |
| `teacher_bc/manifest.json`, `sources.tar.gz` | teacher/motion 文件 SHA256、动作实际关节顺序、网络结构、版本、训练/验证分组、源码快照和时间语义 |
| `logs/teacher_only_bc_launch/*.log` | 启动脚本完整控制台输出，包括启动错误 |

不保存所有环境每一帧原始深度，避免巨量磁盘占用；逐代统计和全部结束 episode 保留，
原始数据为有明确 env/step 标记的定期小批快照，不是全量离线 replay 数据集。

关键读法：

- `action_kl_mean_part` 与 `action_kl_std_part` 分开观察。std 拟合改善不等于动作变准。
- 同看 `action_rmse`、`feature_mse`、`previous_action_baseline_mse`；baseline 直接复制
  `proprio_history[:, -1, 59:84]` 的上一实际动作。`previous_action_baseline_skill > 0`
  表示动作 MSE 优于此基线，0 表示相当。baseline MSE 为零时不定义 skill。
- 分组一定看 `count`；空组只记录 count，不把缺数据解释成零误差。
- `optimizer/*` 是原优化器的 minibatch loss 均值，和更新前的 rollout 指标不是同一个时点。
  梯度日志明确为最后一个 minibatch 的裁剪后范数，不是整代最大值，也不是两项 loss 的独立梯度。
- 环境寿命、接触和完成率全是 **teacher** 的表现。`motion_complete` 是一项终止标记；
  多项终止可能同时触发，各原因比例不必相加为 1，不能直接当 student 成功率。
- 原始输入/标签/reference 帧在 `env.step()` 前记录，done/终止原因在 step 后记录。
  环境在 step 中自动 reset；`last_pre_action_frame` 不是 reset 后的新帧。
  `pre_action_metrics/*` 来自现有 command metrics，可能比 reference cursor 滞后一帧。
- `observed_length` 是本进程实际执行步数；`environment_length` 包含 runner 最初随机化的
  timeout 计数。reset 初期分组用自维护步数，不用随机化的计数。
- 每代 BC 指标与快照为第 i 次优化前，`model_i.pt` 为该次优化后。
  续训仅接受 teacher-only checkpoint，并对齐 runner 与 BC 计数；不恢复物理环境状态。

## 后续独立 student 检查

teacher 验证误差低不代表 student 能自主攀爬。需要另开 `play.py`，使用原
`Ladder-Climb-CR1-ZDW0824-UpDown-Pure-Visual-Dagger-v0` task 加新的 student checkpoint；
该播放入口调用 student inference，不调用训练时的 teacher 控制器。不要使用新训练 task
做单环境播放（它需要留出验证环境）。还需传原 teacher checkpoint、stage37 motion。
可使用 `--num_envs 1 --reference_frame 0 --disable_randomization --max_steps 5000`。

本实验不自动把 student 评估 rollout 加入训练。当前 `play.py --fast_mode` 的视觉分组
可能跳过深度缓存更新，独立视觉诊断不要加该参数；也不加 `--real-time` 主动限速。

## 验证与参考

CPU：`PYTHONPATH=Ladder/source/ladder OMP_NUM_THREADS=1 python -m pytest
Ladder/tests/test_dagger.py Ladder/tests/test_teacher_bc_metrics.py
Ladder/tests/test_teacher_only_bc.py -q`（执行时合并成一行）。

实现前检查了 IsaacLab 的 `rsl_rl/distillation_cfg.py`、G1 task 注册、
`manager_based_rl_env.py` / `termination_manager.py`，以及
`whole_body_tracking/scripts/rsl_rl/train.py`、`tracking_env_cfg.py` 和现有 pure DAgger。
只沿用 IsaacLab/RSL-RL 与本地 Ladder 代码，不引入 whole_body_tracking 运行时依赖。
