# Teacher-only BC student：同状态影子对照检查（2026-09-07）

## 检查对象与边界

- Student：`cr1_zdw0824_up_down_teacher_only_visual_bc/2026-09-06_01-10-23_teacher_only_bc_teacher9999_5120env_10000iter/model_9999.pt`。
- Teacher：stage36 `2026-08-28_18-07-30_stage36_two_anchor_20_10_35_35_frame0_nominal_to10000/model_9999.pt`。
- Motion：stage37 `zdw0824_capture1_stage37_train_ready_middle_default100_zplus5mm_reset5mm_visual_up_down_stop_50hz_motion.npz`，3061 帧，50 Hz。
- 四组对照，每组 8 个环境，seed=42，固定 reference frame=0，只统计各环境第一个 episode，最多 3200 个动作步；未观察到终止的样本标为截尾，不当作成功。
- 无训练、无优化器、无 teacher 接管或动作混合。指定 controller 执行确定性均值；另一网络只对同一个实际状态给出影子输出。Student 的输入仍是 9×85 本体历史与 2×60×80 深度历史，teacher 输入为 780 维特权信息，二者输入并不相同。

`normal` 复现原 `play.py --disable_randomization` 的环境配置：关闭基础物理/初始姿态/梯子几何随机化，但仍保留本体噪声、深度增强、Warp 相机扰动、电机随机化和 0–2 步延迟。`clean` 额外关闭这些扰动，保留动作尺度、6 rad/s 目标速度限制、物理、碰撞、任务和终止阈值。这里的 clean 不保证 GPU 物理逐位确定性，也不表示比训练分布更有代表性。

## 采集实现与复核

独立入口：`Ladder/scripts/rsl_rl/diagnose_visual_student.py`。没有修改训练或现有 play 分支。

输入来自一次正常 reset/step 的输出，影子查询不额外调用 observation manager。通过该环境实例的 termination compute 只读 hook，在物理步结束、自动 reset 和 reference 推进之前保存真实 post-action 状态。没有启用会额外计算观测的 RecorderTerm。每步断言 pre/post reference frame 一致，且保存的 done 与 env.step 返回值一致。

输出根目录：`outputs/teacher_student_shadow/2026-09-07_frame0_check/`。

每组保存：

- `manifest.json`：输入文件 SHA256、网络、观测维度、动作关节顺序、接触阈值、配置改动和剩余扰动。
- `env.yaml`、`action_metadata.pt`：实际配置、动作尺度/偏置、目标变化上限及环境原点。
- `trace.pt`：逐步完整 student/teacher 输入、均值、256 维 feature，pre/post 位姿、速度、关节、接触力、目标接触误差，以及真实延迟、延迟后动作、限速后关节目标和各项终止标签。
- `steps.csv`、`summary.json`：首 episode 有效样本、逐关节误差、分阶段统计及描述性阈值越界时刻。
- `console.log`：该组实际仿真运行的完整控制台输出。
- 根目录 `diagnostic_sources.tar.gz` 保存本次采集/分析脚本与单测源码。

解释时必须区分：

1. `anchor_*` 跟踪指标对应 `body`，而 `root_pose` 终止使用 `base_link`。不能把 anchor 误差套进 root 阈值。固定零几何偏移情况下，可用 NPZ 的 `body_pos_w/body_quat_w[reference_frame, base_link]` 与 trace 中减去环境原点的真实 root 位姿离线恢复 root 误差。
2. `physical_contact` 是肢体净力超过阈值的代理，并非特定碰撞对接触。当前 `required_contact` 终止实际是 required 肢体到目标距离 >0.45 m 持续 30 步，不是力阈值缺失。
3. root 终止是 base_link 位置误差 >0.5 m 或姿态误差 >1.2 rad 持续 25 步。normal/student 的全部有效样本已离线重建，结果与保存的 root/required-contact 终止标签完全一致。
4. 两个 controller 的后续轨迹不是同一状态序列；“同状态比较”是各自轨迹内的 student 与 teacher 输出比较。相同 seed 也不能保证分叉/不同 reset 后的随机噪声逐项配对。
5. 未记录每步实际电机增益；normal 与 clean 一次性关闭多种扰动，只能检验扰动是否为必要原因，不能单独估计某一类扰动的因果贡献。

## 验证

- 新增配置/统计 CPU 测试：14 passed。
- 连同既有 DAgger 与 teacher-only BC 测试：33 passed。
- 2 环境、3 步 clean/student GPU smoke：输入维度正确、数据有限、reference 为 0/1/2、delay=0，两套网络权重未变；两个 episode 正确标为截尾。
- normal/student trace 已确认：本体历史里的上一动作与实际上一 student 输出完全一致；非 reset 步的 post→下一 pre 真实 root 状态完全一致。
- 四组均正常完成，原始张量有限，两套网络权重均未改变；实际延迟后的动作与按记录 delay 查询历史 controller 输出完全一致。实际关节目标每步最大变化约 0.120000005 rad，符合 0.12 rad 限制（浮点舍入差）。
- normal 与 clean 各自的 student/teacher 两个控制器实验，在第一个动作之前，真实 root/joint、全部网络输入、student/teacher 输出逐项完全相同。
- 全部评估进程已正常退出；没有启动新训练。

实现参考：官方 IsaacLab RSL-RL play、Anymal play corruption 配置、installed `ManagerBasedRLEnv.step` / `TerminationManager` / `RecorderManager`；`whole_body_tracking/scripts/rsl_rl/play.py`；本项目 `mdp/actions.py`、`mdp/observations.py`、`mdp/commands.py`、`mdp/terminations.py` 与 pure DAgger 网络/配置。未引入 whole_body_tracking 运行时依赖。

## 四组结果

| 配置 | 实际控制器 | 完成参考序列 | 首 episode 时长 | 同状态动作 RMSE | Feature MSE |
|---|---|---:|---:|---:|---:|
| normal | teacher | 8/8 | 61.22 s | 0.019403 | 0.000955 |
| normal | student | 0/8 | 1.50–2.48 s | 0.960922 | 1.064567 |
| clean | teacher | 8/8 | 61.22 s | 0.016389 | 0.000680 |
| clean | student | 0/8 | 2.08–2.20 s | 1.007460 | 1.279615 |

完成指触发 `motion_complete` 且没有同时触发失败项，不表示所有时刻接触代理都满足。每条轨迹的 RMSE 是该轨迹上两个网络对同一个实际状态预测的动作均值差，不是把不同轨迹上同一时刻的动作直接相减。表中全程误差受存活时长和访问状态影响，因此还需下面的分阶段比较。

### reset 瞬态之后，误差是否收敛

以下为原始动作单位，按有效样本和 25 个动作维度共同计算 RMSE；不是弧度，也不是 KL。每列按一基动作步标注，分别对应 trace 的 `[0:9]`、`[9:50]`、`[50:100]`。

| 配置/控制器 | 第 1–9 步 | 第 10–50 步 | 第 51–100 步 |
|---|---:|---:|---:|
| normal/teacher | 0.06322 | 0.01425 | 0.01261 |
| normal/student | 0.10586 | 0.64937 | 1.07532 |
| clean/teacher | 0.05920 | 0.01076 | 0.01021 |
| clean/student | 0.09346 | 0.65289 | 1.23329 |

normal/student 的一部分环境在第 100 步之前已终止，后面的 reset 样本没有混入分桶。

normal 第一个动作的误差为 0.02861。第二个动作时，teacher 轨迹上的 student 影子预测也有 0.16275 的短暂误差峰值，student 轨迹上为 0.17419。因此，不能把第二步尖峰视为 student 自主失败独有现象；区分两者的是之后是否收敛。

- normal 在第 0.50 s（零基 action_step=25）：teacher 轨迹上的误差已经回到 0.01805，而 student 轨迹上的误差为 0.56466。
- normal 在第 0.98 s：两者分别为 0.01192 和 0.98120。
- clean 在第 0.50 s：两者分别为 0.01279 和 0.64746。
- normal/student 大多数环境在约 0.28–0.40 s 后 anchor 姿态误差超过 0.1 rad，约 0.52–0.72 s 后 anchor 位置误差超过 5 cm，动作失配在此前已经开始。

在第 10–50 步，student 轨迹上的最大逐关节动作 RMSE 是 `waist_x_joint`（normal 约 1.594，clean 约 1.620）；早期还包括双踝、膝和肘。实际动作尺度按关节分别为约 1.134375、0.428125、0.15625，不能统一乘 0.25 来解释误差。

### 本次并非先丢失要求的梯上接触

normal/student 最终是 6 个 `root_pose`、2 个 `body_local_pose`；clean/student 各 4 个。两组 student 的全部有效动作步中，reference 的四肢 required-contact 标签都是 false，尚未进入要求这些接触的阶段。因此这次 frame-zero 失败不能归因为“已经丢失 reference 要求的手脚接触，才走向失稳”。

这不等于机器人没有任何地面接触。它只描述当前 reference 的 required 标记。成功 teacher 轨迹上也有大量 force-proxy miss，进一步说明不能把该统计和任务终止条件混为一谈。

## 可以确定与尚不能确定的事

这次检查可以确定：

1. 该 teacher checkpoint 在当前任务、动作执行链与两种检查配置下均能完成参考序列；环境或动作执行链并非整体不可用。
2. Student 确实能在 teacher 访问的状态上较好拟合其输出，低离线误差不是“student 已经学会自主攀爬”的证据。
3. Student 自主控制后的预测误差迅速放大，早于大的姿态/位置偏离；移除配置中的噪声、相机/电机扰动与延迟仍然失败，说明它们不是必要原因，也不是单独关闭即可解决的问题。
4. 本次观测中上一动作归属正确，标签/参考帧和 post-action 采集没有跨 reset 错位，实际延迟和限速执行符合配置。

还不能单独确定：

- 缺少 reference/真实状态导致的可观测性限制、动作历史自反馈、reset 初期样本权重、网络的局部纠偏能力、feature MSE 约束，各自贡献多大。这些仍需受控消融，不应仅凭这次结果任选一个作为唯一根因。
- Teacher 在 student 已经严重偏离的状态上实际接管是否还能救回：这次只有影子输出，没有执行接管。
- 其他 reset 帧、未见 motion 或实机的表现。每组 8 个固定帧起点是局部诊断，不是广泛成功率评估。

更准确的总结是：**当前 student 学到了专家轨迹上的动作拟合，但尚未学成稳定的自主闭环控制。** 下一步应优先区分 reset 瞬态后的纠偏不足与持续的自主状态分布问题，而不是仅凭 teacher-only 的低 loss 继续增加训练代数。此检查没有擅自修改训练方案。
