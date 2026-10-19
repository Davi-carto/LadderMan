# LadderMan 第一阶段 Expert 策略复现指南

本文档面向 `LadderMan: Learning Humanoid Perceptive Ladder Climbing` 的第一阶段复现：训练 state-based expert climbing policies。目标不是一开始复现完整论文系统，而是先把基于参考动作的 hybrid motion tracking expert 训练出来。

## 1. 阶段目标

第一阶段要完成：

```text
单条爬梯 reference motion
→ hybrid motion tracking
→ 多个 state-based expert policy π_expert_{phi,z}
```

暂时不做：

```text
depth-based visual policy
DAgger / visual distillation
VFM depth sim-to-real
on-ladder manipulation
dual-agent teleoperation
```

第一阶段完成的最低标准：

```text
G1 在 IsaacSim/IsaacLab 仿真中，使用 privileged/state observation，
能在参考梯子 Ladder A 上完成完整上爬动作；
手脚 required contacts 不持续丢失；
再扩展到若干不同 (phi, z) 梯子配置的 expert。
```

## 2. 推荐工程目录

桌面建议目录：

```text
C:\Users\13907\Desktop\LadderMan
  ├─ whole_body_tracking/              # BeyondMimic motion tracking 参考代码
  ├─ docs/
  │   ├─ LadderMan_expert_stage_reproduction_guide.md
  │   └─ LadderMan_expert_stage_checklist.md
  ├─ ladderman_expert/                 # 之后你自己的实现目录
  ├─ assets/
  │   ├─ robots/
  │   ├─ ladders/
  │   └─ motions/
  └─ runs/
```

已经克隆的参考仓库：

```text
HybridRobotics/whole_body_tracking
https://github.com/HybridRobotics/whole_body_tracking
```

该仓库覆盖 BeyondMimic 的 motion tracking 训练，可重点参考：

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/events.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py
source/whole_body_tracking/whole_body_tracking/robots/g1.py
source/whole_body_tracking/whole_body_tracking/robots/actuator.py
```

## 3. 环境安装建议

强烈建议训练环境使用原生 Linux，而不是 Windows：

```text
Ubuntu 22.04
NVIDIA GPU
NVIDIA Driver
Isaac Sim 4.5.0
Isaac Lab 2.1.0
Python 3.10
rsl_rl / isaaclab_rl
```

`whole_body_tracking` README 明确对应：

```text
IsaacSim 4.5.0
IsaacLab 2.1.0
Python 3.10
Linux x86_64
```

建议安装顺序：

```text
1. 安装 NVIDIA driver，确认 nvidia-smi 正常。
2. 安装 Isaac Sim 4.5.0。
3. 安装 Isaac Lab 2.1.0。
4. 运行 Isaac Lab 自带 locomotion 示例，确认仿真和 PPO 能跑。
5. 安装 whole_body_tracking：
   python -m pip install -e source/whole_body_tracking
6. 按 whole_body_tracking README 下载 Unitree description assets。
7. 能跑通 Tracking-Flat-G1-v0 后，再开始 LadderMan expert task。
```

不要混用 IsaacSim/IsaacLab 大版本。Isaac 生态对版本很敏感，版本错配会导致 import、USD、物理参数或 RL wrapper 出问题。

## 4. 机器人模型

优先级：

```text
1. 使用 whole_body_tracking 中的 G1 资产和配置。
2. 如果资产缺失，按 README 从 GCS 下载 unitree_description。
3. 如果要换模型，用 Unitree 官方 G1 URDF/USD，再导入 IsaacSim。
```

重点注意：

```text
LadderMan 附录强调 collision geometry 对爬梯特别关键。
平地 locomotion 可用的简化碰撞体，不一定适合梯子接触。
```

需要重点检查和修正的碰撞体：

```text
hands / wrists
feet / ankles
shins
forearms
torso
```

最低要求：

```text
手和脚的 collision 能稳定接触横杆；
不会因为过度简化导致穿透、卡住、虚假弹飞；
不会因为碰撞体过大导致接触目标不可达。
```

## 5. 梯子模型

第一阶段不要优先找复杂 CAD。建议程序化生成梯子，原因是 expert 训练需要大量几何随机化。

梯子参数：

```text
phi: ladder inclination
z: rung spacing
l: rung length
w/r: rung width or radius
shape: cuboid or cylinder
num_rungs
friction
```

先复现论文 reference ladder：

```text
Ladder A = (65.5 deg, 24.8 cm, 60 cm, 8 cm)
```

然后扩展 representative experts：

```text
(55 deg, 20 cm)
(65.5 deg, 24.8 cm)
(70 deg, 30 cm)
```

最后扩展网格：

```text
phi ∈ {55, 60, 65, 70} deg
z ∈ {20, 22, 24, 26, 28, 30} cm
```

程序化梯子生成建议：

```text
side rails: cuboid，可选
rungs: cylinder 或 cuboid
static collision enabled
rigid body disabled or fixed
material friction randomized
```

实现时建议保留函数：

```python
generate_ladder(phi, rung_spacing, rung_length, rung_radius_or_width, num_rungs, shape)
get_rung_pose(rung_id, ladder_geometry)
```

后续 contact target generation 会依赖 `get_rung_pose`。

## 6. Reference Motion

这是第一阶段最大难点。你需要一条人形爬梯 reference motion。

理想方案：

```text
真实 mocap / OptiTrack 采集 Unitree G1 或人类爬梯动作，然后 retarget 到 G1。
```

工程可行方案：

```text
用 IK + keyframes 手工生成一条 G1 爬梯 motion。
```

最低可行方案：

```text
在仿真里设计手脚/骨盆关键帧：
四点接触 → 单肢 swing → 新横杆接触 → 骨盆上移 → 下一肢 swing。
```

reference motion 至少要保存：

```text
q_ref[t]
qdot_ref[t]
root pose[t]
root velocity[t]
body local poses[t, body]
body velocities[t, body]
phase = t / T
contact_indicator[t, limb]
contact_rung_id[t, limb]
```

四个 limb：

```text
left_hand
right_hand
left_foot
right_foot
```

`contact_indicator` 是论文里的 `c_hat`：

```text
c_hat[t, limb] = 1 表示该 limb 在当前 reference phase 应该接触梯子；
c_hat[t, limb] = 0 表示该 limb 正在 swing 或不要求接触。
```

强烈建议额外保存 `contact_rung_id`。论文没有单独强调这个字段，但复现时它非常实用，因为它把 reference 的接触时序映射到任意目标梯子的横杆。

## 7. Contact Target Generation

这是 LadderMan 第一阶段的核心。

普通 motion tracking：

```text
reference motion 同时决定什么时候接触、在哪里接触。
```

LadderMan hybrid motion tracking：

```text
reference motion 决定什么时候接触；
current ladder geometry 决定在哪里接触。
```

核心函数：

```python
P_hat = get_target_contact_position(ladder_geometry, phase, limb)
```

逻辑：

```text
1. 根据 phase 和 limb 查询 c_hat。
2. 如果 c_hat = 0，则该 limb 不需要追 contact target。
3. 如果 c_hat = 1，查询 contact_rung_id。
4. 根据当前 ladder_geometry 计算该 rung 的世界坐标。
5. 根据 limb 加固定横向 offset。
6. 得到 P_hat。
```

示例：

```text
P_hat_left_hand  = rung_center[k] + left_hand_lateral_offset
P_hat_right_hand = rung_center[k] + right_hand_lateral_offset
P_hat_left_foot  = rung_center[j] + left_foot_lateral_offset
P_hat_right_foot = rung_center[j] + right_foot_lateral_offset
```

需要调试可视化：

```text
画出 P_hat 小球；
画出当前 hand/foot position；
画出接触误差线段；
确认不同 phi/z 下 P_hat 落在正确横杆。
```

## 8. Expert MDP

### Action

建议沿用 BeyondMimic / LadderMan：

```text
policy 输出 normalized joint position action；
转换成 target joint positions；
底层 PD controller 执行。
```

参考：

```text
whole_body_tracking/robots/g1.py
whole_body_tracking/robots/actuator.py
whole_body_tracking/tasks/tracking/mdp/actions
```

控制频率：

```text
policy: 50 Hz
physics dt: 0.005 s
decimation: 4
PD / physics: 200 Hz 或更高
```

### Observation

Expert policy 可以使用 privileged observation。

建议 observation：

```text
Reference motion:
  reference body local pose
  reference joint position
  phase t/T

Proprioception:
  root angular velocity
  root orientation
  joint positions
  joint velocities
  previous actions

Privileged state:
  pelvis/root relative to ladder
  body local pose
  body velocity
  root velocity
  body pose error
  body velocity error
  joint position error
  optional: ladder parameters phi,z,l,w
```

注意：最终 visual policy 不能依赖 privileged state，但 expert 可以用。

### Reward

Expert reward 分四组。

第一组：asymmetric motion tracking。

```text
lower body tracking 更严格；
upper body tracking 更放松。
```

原因：

```text
下半身保持稳定爬梯节奏；
上半身需要适应不同梯子的横杆位置。
```

第二组：ladder contact tracking。

公式：

```text
r_contact = w * c_hat * exp(-||P - P_hat||^2 / tau) + (1 - c_hat)
```

含义：

```text
c_hat = 1：该 limb 应该接触，奖励它靠近目标横杆点；
c_hat = 0：该 limb 不考核 contact position，给常数项避免 swing phase 被惩罚。
```

第三组：climbing / root task reward。

```text
root/pelvis 接近目标梯上 pose；
沿梯子方向产生正确 progress；
避免身体远离梯面。
```

第四组：regularization。

```text
action rate
joint limits
joint velocity
torque limits
impact force
slip
air time / contact timing
undesired collision
```

### Termination

建议 termination：

```text
time out
root global pose error too large
body local pose error too large
base roll/pitch too large
required ladder contact lost > 30 simulation/control steps
severe undesired collision
joint/torque safety violation
```

`required ladder contact lost` 的判断：

```text
如果 c_hat[t, limb] = 1，但该 limb 没有和目标 rung 形成有效接触，
连续超过阈值，则终止 episode。
```

## 9. Domain Randomization

第一阶段先不要过度随机化。按递进来。

单梯子调通时关闭大部分随机化：

```text
fixed ladder
fixed friction
fixed initial state
fixed reference start phase 或少量 phase sampling
```

稳定后加入：

```text
mass scale
torso COM offset
joint position offset
static/dynamic friction
restitution
initial root pose perturbation
initial joint pose perturbation
ladder phi local perturbation
ladder z local perturbation
rung shape / size perturbation
```

LadderMan expert 阶段论文范围可作为参考：

```text
mass scale: U(0.9, 1.1)
torso COM offset: roughly ±0.02 m
joint position offset: roughly ±0.01 rad
static friction: U(0.3, 7.2)
dynamic friction: U(0.3, 3.6)
restitution: U(0.0, 0.2)
ladder inclination perturbation: ±1 deg
rung spacing perturbation: ±0.01 m
```

## 10. 训练顺序

### Milestone 0：基础环境

完成：

```text
IsaacLab 能启动；
G1 能加载；
程序化梯子能生成；
robot reset 后站在梯子前；
能 step 1000 步不崩。
```

检查：

```text
碰撞是否正常；
横杆是否有 collision；
手脚是否能接触横杆；
相机/视角/调试可视化是否可用。
```

### Milestone 1：Reference Replay

完成：

```text
能在仿真中回放 reference motion；
能可视化 hand/foot、P_hat、rung id、c_hat。
```

检查：

```text
手脚轨迹是否物理可达；
reference body 不应明显穿过梯子；
contact phase 是否和 motion 对齐。
```

### Milestone 2：单梯子 Motion Tracking

完成：

```text
只在 Ladder A 上训练 expert；
先用 BeyondMimic-style tracking reward；
能大致跟踪 reference。
```

检查：

```text
root 不倒；
手脚不乱飞；
tracking reward 上升；
termination 不全是 fall。
```

### Milestone 3：加入 Contact Tracking

完成：

```text
加入 c_hat / P_hat contact reward；
手脚能在接触阶段靠近目标横杆。
```

检查：

```text
contact distance 曲线下降；
required contact loss 减少；
P_hat 映射不应跳错 rung。
```

### Milestone 4：Hybrid Tracking

完成：

```text
下半身 tracking 更严格；
上半身 tracking 放松；
contact reward 约束上肢接触目标。
```

检查：

```text
换 phi/z 小扰动时，手的位置能适应横杆；
不是死板复现原参考手部世界坐标。
```

### Milestone 5：局部随机化 Expert

完成：

```text
围绕目标 (phi,z) 做小范围 ladder randomization；
加物理随机化；
expert 在局部邻域内成功率稳定。
```

检查：

```text
success rate
contact loss count
root progress
tracking error
termination reason
```

### Milestone 6：多个 Expert

完成：

```text
训练多个 π_expert_{phi,z}。
```

推荐先做：

```text
(55 deg, 20 cm)
(65.5 deg, 24.8 cm)
(70 deg, 30 cm)
```

检查：

```text
每个 expert 是否能在对应梯子上完成完整爬升；
是否都来自同一条 reference motion；
是否没有为每个 expert 手工改 reference。
```

## 11. PPO 初始配置

可参考 whole_body_tracking：

```text
num_envs: 4096
num_steps_per_env: 24
actor: [512, 256, 128] 或 LadderMan [512, 256, 256]
critic: [512, 256, 128]
activation: ELU
learning_rate: 1e-3
schedule: adaptive
clip_param: 0.2
entropy_coef: 0.005
value_loss_coef: 1.0
gamma: 0.99
GAE lambda: 0.95
desired_kl: 0.01
learning_epochs: 5
mini_batches: 4
```

LadderMan 论文 expert 阶段：

```text
4096 parallel environments
8000 epochs
actor: [512, 256, 256]
critic: [512, 256, 128]
```

## 12. 常见失败原因

### Reference motion 不物理可行

现象：

```text
tracking reward 上不去；
机器人一跟踪就倒；
手脚目标不可达。
```

处理：

```text
先 replay 检查；
减少速度；
用 IK 修正关键帧；
降低一开始的梯子倾角。
```

### Contact target 映射错

现象：

```text
手脚总是追错横杆；
某些 phase P_hat 突然跳跃；
reward 看似合理但动作怪。
```

处理：

```text
可视化 rung id；
可视化 P_hat；
打印 phase、limb、contact_rung_id。
```

### Collision geometry 不适合

现象：

```text
手脚穿过横杆；
脚卡住；
接触力异常；
同一动作有时弹飞。
```

处理：

```text
修手脚 collision；
加大 physics substeps；
检查横杆 collision；
简化早期 ladder shape。
```

### PD gains / action scale 不合适

现象：

```text
动作抖；
力矩爆；
接触冲击大；
跟踪滞后。
```

处理：

```text
参考 whole_body_tracking/robots/actuator.py；
降低 stiffness；
调 action scale；
记录 torque 和 action rate。
```

### Reward 权重顺序不对

现象：

```text
策略只追姿态不接触；
或只追接触导致身体姿态崩；
或完全不往上爬。
```

处理：

```text
先单独验证 tracking；
再加 contact；
最后加 climbing；
分项记录 reward。
```

## 13. 第一阶段最终验收

必须能回答：

```text
1. 是否只用一条 reference motion？
2. 是否能在 Ladder A 上稳定完成完整爬升？
3. 是否能用同一条 reference motion 训练不同 (phi,z) expert？
4. contact target 是否由当前 ladder geometry 生成，而不是照搬 reference 世界坐标？
5. required contacts 是否大部分时间保持？
6. 失败时 termination reason 是否可解释？
7. 每个 expert 是否有明确成功率统计？
```

建议输出：

```text
runs/expert_ladder_A/
runs/expert_phi55_z20/
runs/expert_phi65_z248/
runs/expert_phi70_z30/
```

每个 run 保存：

```text
config.yaml
checkpoint.pt
training curves
success_rate.json
termination_stats.json
contact_error_curves.csv
demo video
```

## 14. 进入第二阶段前的门槛

不要急着做 visual policy。只有当下面条件满足后，再进入 DAgger + RL distillation：

```text
单 expert 能稳定爬 Ladder A；
多个 expert 能覆盖代表性 (phi,z)；
contact target generation 已经充分可视化验证；
训练失败原因大多可解释；
policy 动作平滑，力矩不过分异常。
```

第二阶段才是：

```text
multiple experts
→ hybrid DAgger + PPO
→ depth-based visual policy
```

