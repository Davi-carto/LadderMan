# 蒸馏、GRU reset 与深度刷新时序核查（2026-09-07）

本次审查 RSL-RL、Extreme Parkour 和本地实现；只修改深度采样时序及其配置/兼容性，不修改蒸馏 loss、网络、teacher、reward、动作或 reference/reset 采样策略，不启动长训练。

## 1. 我们不是 RSL-RL 官方蒸馏的逐项复刻

安装版本为 `rsl-rl-lib 5.0.1`。对照源码：

- [RSL-RL v5.0.1 distillation.py](https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/algorithms/distillation.py)
- [当前 main distillation.py](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/distillation.py)
- [MLPModel](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/models/mlp_model.py)、[EmpiricalNormalization](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/modules/normalization.py)

官方默认对确定性 action mean 做 MSE，也支持 Huber；没有 KL、feature MSE、PPO、输出标准化或 teacher std 加权。收集时请求 student 随机输出（仅当配置了动作分布时采样），teacher 在同一 pre-action 状态给确定性标签。只有 student 被优化，teacher 保持 eval。每轮清空 rollout，不要求永久保存整个 DAgger 数据集。

官方递归训练保持序列次序，按 `done` 清理隐藏状态，并按 `gradient_length` 截断反传；反传前是各步 batch/action-mean loss 的累加，不能把相同学习率/梯度裁剪数值直接解释成与我们的 flat-minibatch 优化等效。

当前 pure/teacher-only 实际为：

`L = 2 * mean_batch,joint KL(student || teacher) + mean_batch,feature (h_student-h_teacher)^2`。

动作 25 维、feature 256 维都已经做维度平均。KL 公式与 PyTorch 分布 KL 数值一致；其均值误差项仍隐含 `1/(2*sigma_teacher_j^2)` 权重。当前 teacher 的 std 范围 0.160816–0.328797，最大隐含权重比约 4.18；这是探索 std，不是物理控制敏感度。系数 2:1 不是梯度贡献 2:1。

当前 pure 分支实际执行确定性 action mean；student std 由 KL 学习，但不用于 rollout 采样。若未来改用官方 action MSE，独立的全局 std 不会从均值 MSE 获得梯度，需要同步明确探索设置及 std 梯度诊断，不能只替换一行 loss。feature MSE 的去留仍是独立消融，不在本次修改范围。

旧 hybrid 的 BC 项是 action-mean MSE，加 PPO 和视觉辅助学习；当前 pure 的随机展平 minibatch 对无递归 CNN+MLP 合法，不能原样用于未来 GRU。

## 2. 归一化：区分固定尺度、运行统计和 loss reduction

官方运行观测标准化是可选项，默认关闭。启用后 teacher 使用保存且冻结的统计，student 使用自己的统计，二者不因输入不同而共享均值方差。`v5.0.1` 在采集每步更新 student 统计，当前 main 改到优化后从整段 rollout 更新；必须区分版本。

本次实际检查的 stage36 teacher9999 checkpoint，actor 只有 `mlp.*` 和 `distribution.std_param`，没有 normalizer buffers。保存配置中遗留 `empirical_normalization=true`，但实际 actor 开关为 `actor_obs_normalization=false`。所以当前 teacher loader 没有漏掉必需的标准化统计。

当前 student 输入并非未缩放：

- 角速度 ×0.2；关节位置 `(q-q_default)/逐关节动作尺度`；关节速度 ×0.1。
- 姿态、上一实际动作、上下指令拼接后，本体输入截断到 ±100。
- Warp 原始处理输出是米制深度，有效深度限制到 [0.1,2.0]，丢失/RFM 像素可能为 0。
- Pure 继承 thin-rails 输入配置，编码前还做 `(depth-1.0)*1.0`；无效像素因此为 -1。不要将相机米制缓存误认为最终 CNN 输入，也不要误称为 running z-score。
- 没有运行 mean/std、BatchNorm；当前 student train/play 使用相同预处理。

**实际兼容风险但不是当前失败原因：** `dagger/student_teacher.py` 仅提取 MLP 权重，会忽略未来 teacher checkpoint 中的 `obs_normalizer.*`。若改用启用了运行标准化的 teacher，应完整加载预处理或明确拒绝该 checkpoint。此审查没有擅自改变 loader/loss 行为。

## 3. Extreme Parkour 如何缩小输入差距

核查官方公开版本 `d2ffe27ba59a3229fad22a9fc94c38010bb1f519`：

- [Actor 分支](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/rsl_rl/rsl_rl/modules/actor_critic.py)
- [视觉 runner 与权重加载](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/rsl_rl/rsl_rl/runners/on_policy_runner.py)
- [视觉/适应损失](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/rsl_rl/rsl_rl/algorithms/ppo.py)
- [部署导出](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/legged_gym/legged_gym/scripts/save_jit.py)

| 信息 | 对应替代路径 |
|---|---|
| 地形 scan | 深度 CNN＋本体＋GRU 输出相同接口的 32 维 latent |
| 质量/摩擦等隐式特权信息 | 本体历史 encoder 替代 privileged encoder 的 latent |
| 线速度等显式状态 | estimator；部署导出明确用预测覆盖显式状态输入 |
| 当前/下一目标方向 | 视觉 GRU 额外输出 2 维 yaw，使用单独方向监督 |

Student 控制 backbone 从已加载的 teacher actor 复制权重。视觉训练时双方使用 history adaptation 路径，不是重新删除所有决策接口后从头学习整个控制器。当前视觉更新使用 action 差向量 L2 norm 均值＋yaw 差向量 L2 norm 均值，不是平方 MSE，也不是 KL；收集的 scan latent 没有在当前 runner 中启用直接匹配损失。

公开实现有边界：训练时仅对 `delta_yaw_ok` 的环境用预测方向替换真值，其他状态仍用真值；非 JIT 仿真路径还保留显式速度块，部署导出另行用 estimator 覆盖。不能笼统宣称所有训练/播放路径从第一步就完全移除了特权输入。

## 4. GRU reset：不能把 detach 当清零

参考：[深度 GRU](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/rsl_rl/rsl_rl/modules/depth_backbone.py)、[play](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/legged_gym/legged_gym/scripts/play.py)、上述 runner。

Extreme Parkour 发布代码中，hidden 仅初始化为 None；训练分段后 `detach().clone()` 保留数值。视觉训练和播放路径未找到按 episode done 清零视觉 GRU 的代码。因此公开参考不能被当作完整的 reset 正确性模板。其深度 buffer 在全局每 5 步刷新时才检查 episode_age<=1，非对齐 reset 还存在保留上一 episode 图像的边界风险。

本地旧 GRU+Transformer 当前实现：

- `dagger/visual_algorithm.py` 跨 rollout 保留 hidden，记录 chunk-start hidden；step 后仅清零 done 环境。
- `transformer_vision_ppo_storage.py::reset_before_mask` 用 `reset_before[1:]=dones[:-1]`，在新 episode 第一条观测前 reset。
- `transformer_vision.py::_run_gru` 在序列回放中执行 reset/mask。
- 两条 play 路径都有相应自动/手动 reset 处理。历史上跨 chunk/手动 reset 的修正见 implementation_log 与 learning 文档。

相关 reset/sequence/hidden 测试 6 passed。未发现当前代码漏掉整套 done reset；这不等于覆盖所有端到端行为。随机 reference 帧起步时 hidden=0、历史重复当前帧，确实缺少此前运动信息，这是冷启动，不应用保留旧 episode hidden 来修补。当前最新纯 CNN 蒸馏根本没有 GRU，不能用 GRU reset 解释其失败。

## 5. 深度刷新修改及边界

参考实际频率：

- Extreme Parkour：物理 dt=.005、decimation=4，策略 50 Hz，深度每 5 策略步刷新即 **10 Hz**；视觉 CNN/GRU 仅新图时更新，中间保持 latent/yaw。[配置](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/legged_gym/legged_gym/envs/base/legged_robot_config.py)
- Robot Parkour Learning Go2：`refresh_duration=1/10`，深度 **10 Hz**，另有 .08–.142 s 视觉延迟。[配置](https://github.com/ZiwenZhuang/parkour/blob/main/legged_gym/legged_gym/envs/go2/go2_distill_config.py)

没有统一必须 30 Hz 的规则。本次按用户提出的实机目标将共享 Warp 新任务默认设为 `update_period=1/30`；策略和本体仍为 50 Hz。

修改前两帧深度间隔是 **.02 s**；此前 **.16 s** 指的是 9 帧本体历史 `(9-1)/50`，不是深度窗口。

实现为策略边界采样的多速率时钟：`floor(step * policy_dt / camera_period)` 保留小数相位。50 Hz 下，新图落在 step 2,4,5,7,9,10…，稳态平均 30 Hz，实际采集间隔 20/40 ms。这不是物理子步上精确每 33.333 ms 曝光的异步硬件模型，也没有新加传输延迟。禁止将50/30简单取整成每2步（那是25Hz）。

- 未到采集时刻：图像及其噪声/RFM处理结果保持不变，depth history 不推进；proprio history 每策略步推进。
- Reset 环境立即单独重采，重复该新图填满其历史，不影响其他环境缓存和时钟。Reset 额外曝光不计入稳态标称频率。
- 同一步重复查询不重拍、不重采噪声、不重复推进历史。
- 暴露 `_cr1_visual_depth_frame_id` 与 `_cr1_visual_depth_capture_step`，便于检查采集来源；不改变 actor 输入维度。
- 新/旧深度尺寸保持2×60×80，本体9×85；teacher 输入不变。
- 旧 GRU 仍按策略频率融合新本体与保持的深度；本次未改为 Extreme Parkour 的“仅新图更新GRU”，两种网络接口不能混同。

兼容性：主训练/播放、纯视觉预训练/检查和 shadow 诊断按视觉 checkpoint 的 `params/env.yaml` 恢复周期；缺字段视为旧策略频率，缺配置明确警告。CLI `--depth_update_hz 30` 可覆盖，`10` 可对照低频，`0` 恢复策略频率。新训练保存实际周期。RTX observation 不注入 Warp 专用参数，盲走 teacher 不受影响。

## 验证与参考顺序

修改前检查：IsaacLab `sensors/sensor_base_cfg.py`、`sensor_base.py` 的 update_period/缓存/reset；官方 locomotion `velocity_env_cfg.py` 与 camera 配置；`whole_body_tracking/.../tracking_env_cfg.py` 和 play；本地 Warp renderer 的 batched env_ids 接口、历史与 GRU/存储/播放代码。没有新增 whole_body_tracking 运行时依赖。

CPU 回归：**95 passed, 1 skipped**（沙箱中 CUDA 不可用导致 GPU-only 项跳过）。覆盖时钟计数、10/30/50 Hz、小数相位、同一步重复查询、噪声保持、单环境/同一步 reset、历史不串 episode、旧 checkpoint 周期恢复、RTX 跳过及现有蒸馏/视觉模块。修改的 10 个生产 Python 文件编译通过。

真实 GPU 冒烟通过：2 个环境，stage36 teacher9999 执行，stage37 frame0、clean 配置，50 个策略步，无优化器。每环境恰好 30 次新采集（不含初始/reset），前六次在2,4,5,7,9,10步；未刷新时两帧历史逐值不变。随后同一步仅 reset env0，只有其 frame_id 加1、两帧填成 reset 后新图，env1 历史逐值不变。本体 [2,9,85]、深度 [2,2,60,80]，输出有限，过程中无意外终止。进程已退出。

初始临时 GPU 测试把手动 reset 放到 inference_mode 外，导致已有 command metric 的 inference tensor 原地更新报错；改正测试上下文后通过，没有为此修改生产代码。启动过程中有 Isaac Sim watch/资源警告，不影响上述断言。临时验证脚本与完整日志：`/tmp/ladder-depth-check-Oo9RQI/smoke.py`、`smoke.log`。
