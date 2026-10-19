# 木制人字步梯模型说明

这是参照用户提供图片生成的木制人字步梯简化模型，用于 LadderMan 第一阶段动捕、仿真和接触目标设计。

## 文件

- `wood_step_ladder.yaml`: 参数源文件。
- `wood_step_ladder.stl`: 通用三维预览文件。
- `wood_step_ladder.obj` / `wood_step_ladder.mtl`: 通用三维模型，带简单材质。
- `wood_step_ladder.usda`: IsaacSim/IsaacLab 可导入的 USD ASCII Mesh。
- `wood_step_ladder_physics.usda`: IsaacSim/IsaacLab 仿真专用 USD，使用多个 primitive box collider。
- `wood_step_ladder_sim.urdf`: Isaac Gym 仿真专用 URDF，使用多个 primitive box collider。
- `wood_step_ladder.scad`: OpenSCAD 参数化模型，方便检查和修改。
- `generate_wood_step_ladder_model.py`: 生成脚本。

## 主要参数

- 底座平台高度: 0.16 m
- 底座平台宽度: 0.64 m
- 底座平台长度: 1.04 m
- 前侧斜面直角三角形高度: 1.6 m
- 前侧斜面直角三角形水平长: 1.04 m
- 前侧斜面长度: 1.908 m
- 前侧踏板数: 7
- 踏板宽度: 0.64 m
- 踏板深度: 0.13 m
- 踏板厚度: 0.03 m
- 相邻踏板顶部高度差: 0.23 m
- 相邻踏板前缘 y 方向距离: 0.135 m
- 前侧倾角: 56.97613244420336 deg
- 后侧支撑腿展开: 1.04 m
- 第一阶踏板顶部高度: 0.190 m
- 最高踏板顶部高度: 1.570 m
- 顶部平台 y 坐标约: -1.040 m
- 后脚 y 坐标约: -1.040 m
- 整体深度约: 1.170 m
- 斜面测量三角形: 底部前点 `(0, 0, 0.16)`，后点 `(0, -1.04, 0.16)`，顶点 `(0, -1.04, 1.76)`
- `wood_step_ladder_physics.usda` 里有三条无碰撞彩色辅助线：绿色为 1.04m 水平边，蓝色为 1.6m 竖直边，红色为斜边。

## 坐标系

原点在前侧斜面底部的地面投影点。

- x 轴: 梯子左右方向。
- y 轴: 梯子前后方向；机器人从正 y 方向接近，后支撑腿朝负 y 方向展开。
- z 轴: 世界竖直向上。

第 i 级踏板中心近似为：

```text
top_z_i = base_platform_height + step_thickness + i * step_top_spacing
center_z_i = top_z_i - step_thickness / 2
front_y_i = -i * step_y_spacing
center_y_i = front_y_i - step_depth / 2
x_i = 0
```

其中 i 从 0 到 6。第一阶踏板前缘与底座平台前缘对齐在 `y=0`；相邻踏板前缘沿 y 方向相差 `0.135 m`。训练时可把踏板中心或踏板前缘作为 foot/hand contact target。

## 建模取舍

模型用规则 cuboid 组合，优先保证仿真碰撞和尺寸清晰。没有做木纹、螺栓、真实链条和倒角。后续如果用于真实加工，需要朋友根据材料强度、连接件、倒角和安全裕度重新出工程图。

## 仿真使用建议

不要把 `wood_step_ladder.stl` 当作训练碰撞体使用；单个非凸 mesh 在足端/手端接触时更容易抖动或穿透。

Isaac Gym 推荐加载 `wood_step_ladder_sim.urdf`，并固定底座：

```python
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
ladder_asset = gym.load_asset(sim, asset_root, "wood_step_ladder_sim.urdf", asset_options)
```

IsaacLab / IsaacSim 推荐加载 `wood_step_ladder_physics.usda`，该文件由多个 `Cube` primitive 组成，并给每个部件添加 `PhysicsCollisionAPI`。它作为静态碰撞障碍物使用，不添加刚体 API，因此不会被机器人撞倒。
