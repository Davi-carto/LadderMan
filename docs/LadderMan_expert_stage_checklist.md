# LadderMan Expert 阶段执行检查清单

这份清单用于边实现边勾选。建议每完成一个 milestone，就保存一次配置、视频和日志。

## A. 环境与仓库

- [ ] 准备 Linux 训练机器，推荐 Ubuntu 22.04。
- [ ] 确认 NVIDIA driver 正常：`nvidia-smi` 可用。
- [ ] 安装 Isaac Sim 4.5.0。
- [ ] 安装 Isaac Lab 2.1.0。
- [ ] 跑通 Isaac Lab 自带 RL 示例。
- [ ] 克隆 `HybridRobotics/whole_body_tracking`。
- [ ] 安装 whole_body_tracking：
  ```bash
  python -m pip install -e source/whole_body_tracking
  ```
- [ ] 按 README 下载 Unitree description assets。
- [ ] 跑通 `Tracking-Flat-G1-v0` 的 play 或 train。

## B. 参考代码阅读

- [ ] 阅读 `README.md`。
- [ ] 阅读 `tracking_env_cfg.py`，理解 scene、observation、reward、termination。
- [ ] 阅读 `mdp/rewards.py`，理解 Gaussian tracking reward。
- [ ] 阅读 `mdp/commands.py`，理解 reference motion command 和 adaptive sampling。
- [ ] 阅读 `mdp/observations.py`，理解 policy/critic observation。
- [ ] 阅读 `mdp/terminations.py`，理解 tracking failure termination。
- [ ] 阅读 `mdp/events.py`，理解 domain randomization。
- [ ] 阅读 `config/g1/agents/rsl_rl_ppo_cfg.py`，确认 PPO 参数。
- [ ] 阅读 `robots/g1.py` 和 `robots/actuator.py`，理解 G1 PD/action scale。

## C. 机器人模型

- [ ] G1 能在 IsaacSim/IsaacLab 中加载。
- [ ] 关节名和 action 维度正确。
- [ ] PD controller 可用。
- [ ] contact sensor 可读取手、脚、身体接触。
- [ ] 手部 collision 检查完成。
- [ ] 脚部 collision 检查完成。
- [ ] shin/forearm/torso collision 没有明显异常。
- [ ] 记录当前 robot asset 来源和修改内容。

## D. 程序化梯子

- [ ] 实现 `generate_ladder(...)`。
- [ ] 实现 `get_rung_pose(rung_id, ladder_geometry)`。
- [ ] 支持参数 `phi`。
- [ ] 支持参数 `rung_spacing z`。
- [ ] 支持参数 `rung_length l`。
- [ ] 支持 rung shape：至少 cuboid。
- [ ] 支持 rung shape：可选 cylinder。
- [ ] 支持 rung friction 设置。
- [ ] Ladder A 可生成：`65.5 deg, 24.8 cm, 60 cm, 8 cm`。
- [ ] 梯子横杆 collision 正常。
- [ ] 可视化每根 rung 的编号。

## E. Reference Motion

- [ ] 确定 reference motion 来源：mocap / teleop / IK keyframe。
- [ ] 保存 `q_ref[t]`。
- [ ] 保存 `qdot_ref[t]`。
- [ ] 保存 root pose / velocity。
- [ ] 保存 body local poses。
- [ ] 保存 body velocities。
- [ ] 保存 phase `t/T`。
- [ ] 标注 `contact_indicator[t, limb]`。
- [ ] 标注 `contact_rung_id[t, limb]`。
- [ ] 四个 limb 均覆盖：left hand, right hand, left foot, right foot。
- [ ] reference replay 可视化通过。
- [ ] reference 不明显穿透梯子。
- [ ] reference 手脚接触时序符合直觉。

## F. Contact Target

- [ ] 实现 `get_target_contact_position(ladder_geometry, phase, limb)`。
- [ ] `c_hat=0` 时不追 contact target。
- [ ] `c_hat=1` 时能找到正确 rung id。
- [ ] left/right hand offset 正确。
- [ ] left/right foot offset 正确。
- [ ] 不同 `phi` 下 P_hat 仍在横杆上。
- [ ] 不同 `z` 下 P_hat 仍映射到正确横杆。
- [ ] 调试可视化 P_hat 小球。
- [ ] 调试可视化 limb 当前位置到 P_hat 的误差线。
- [ ] 保存一段 P_hat 可视化视频。

## G. MDP Observation

- [ ] 加入 reference motion observation。
- [ ] 加入 phase observation。
- [ ] 加入 proprioception。
- [ ] 加入 previous action。
- [ ] 加入 pelvis/root relative to ladder。
- [ ] 加入 body pose error。
- [ ] 加入 body velocity error。
- [ ] 加入 joint position error。
- [ ] actor observation 和 critic privileged observation 区分清楚。
- [ ] observation normalization 正常。
- [ ] 检查 observation 无 NaN/Inf。

## H. Reward

- [ ] 实现 body pose tracking reward。
- [ ] 实现 body velocity tracking reward。
- [ ] 实现 joint tracking reward。
- [ ] 实现 lower-body stricter tracking。
- [ ] 实现 relaxed upper-body tracking。
- [ ] 实现 hand contact tracking reward。
- [ ] 实现 foot contact tracking reward。
- [ ] 实现 root / climbing task reward。
- [ ] 实现 action rate penalty。
- [ ] 实现 joint limit penalty。
- [ ] 实现 joint velocity / torque penalty。
- [ ] 实现 slip / impact penalty。
- [ ] 分项记录 reward curves。
- [ ] 单独测试每个 reward 数值范围。

## I. Termination

- [ ] time out termination。
- [ ] root pose error termination。
- [ ] body local pose error termination。
- [ ] base roll/pitch termination。
- [ ] required ladder contact lost termination。
- [ ] undesired collision termination。
- [ ] joint/torque safety termination。
- [ ] 记录 termination reason distribution。

## J. Domain Randomization

- [ ] 第一版关闭大部分随机化。
- [ ] 加入 mass scale randomization。
- [ ] 加入 torso COM randomization。
- [ ] 加入 joint offset randomization。
- [ ] 加入 friction randomization。
- [ ] 加入 restitution randomization。
- [ ] 加入 initial root pose randomization。
- [ ] 加入 initial joint randomization。
- [ ] 加入 `phi ± 1 deg` local perturbation。
- [ ] 加入 `z ± 0.01 m` local perturbation。
- [ ] 加入 rung size / shape randomization。

## K. 训练 Milestones

### Milestone 0：环境

- [ ] G1 + 梯子场景可 reset。
- [ ] 能 step 1000 步不崩。
- [ ] contact sensor 正常。
- [ ] viewer 中梯子和机器人位置正确。

### Milestone 1：Reference Replay

- [ ] 能回放 reference。
- [ ] 能显示 contact phase。
- [ ] 能显示 target rung id。
- [ ] 能显示 P_hat。

### Milestone 2：单梯子基础 tracking

- [ ] Ladder A 固定。
- [ ] 仅 motion tracking reward 能收敛。
- [ ] tracking error 下降。
- [ ] robot 不快速摔倒。

### Milestone 3：Contact Tracking

- [ ] contact reward 加入后训练稳定。
- [ ] hand/foot contact error 下降。
- [ ] required contact loss 减少。
- [ ] 接触阶段 limb 靠近目标横杆。

### Milestone 4：Hybrid Tracking

- [ ] lower body tracking 更严格。
- [ ] upper body tracking 更放松。
- [ ] 小幅改变梯子几何时，手能适应横杆位置。
- [ ] 策略不是死板复现 reference 世界坐标。

### Milestone 5：局部随机化 expert

- [ ] `phi ± 1 deg` 训练通过。
- [ ] `z ± 1 cm` 训练通过。
- [ ] friction randomization 训练通过。
- [ ] 局部邻域 success rate 达到可接受水平。

### Milestone 6：多 expert

- [ ] 训练 `(55 deg, 20 cm)` expert。
- [ ] 训练 `(65.5 deg, 24.8 cm)` expert。
- [ ] 训练 `(70 deg, 30 cm)` expert。
- [ ] 每个 expert 均保存 checkpoint 和 demo video。

## L. 日志与验收

每个 run 至少保存：

- [ ] `config.yaml`
- [ ] `checkpoint.pt`
- [ ] `training reward curves`
- [ ] `success_rate.json`
- [ ] `termination_stats.json`
- [ ] `contact_error_curves.csv`
- [ ] `tracking_error_curves.csv`
- [ ] `demo.mp4`
- [ ] `notes.md`

第一阶段最终验收：

- [ ] 单条 reference motion 没有被每个 expert 手工改写。
- [ ] Ladder A expert 能完成完整爬升。
- [ ] 多个 `(phi,z)` expert 能训练成功。
- [ ] contact target 来自当前 ladder geometry。
- [ ] required contact 大部分时间保持。
- [ ] failure reason 可解释。
- [ ] 进入 visual policy 前，expert 行为足够稳定。

