# 视觉导航数据集采集统一指南

本文档汇总当前仓库已经配置好的仿真数据采集环境、可用场景、启动命令、键盘控制、手动录制、数据处理、检查、可视化和训练方法。

所有命令默认从仓库根目录执行：

```bash
cd /home/czl/visualnav-mamba
```

## 1. 当前支持的采集环境

| 环境 | 当前可直接使用的场景 | 相机 | 状态 |
| --- | --- | --- | --- |
| TurtleBot3 Gazebo | 9 个官方 world | 480x480、160 度鱼眼、16Hz | 已配置，可直接采集 |
| iGibson | demo 场景 `Rs` | 480x480、160 度等距鱼眼、16Hz | 已配置，可直接采集 |
| iGibson 完整数据集 | 获得授权并下载后的交互式室内场景 | 480x480、160 度等距鱼眼、16Hz | 代码已支持，场景数据需另行下载 |

两套环境最终都会转换成相同的 NoMaD-Mamba / ViNT 数据格式：

```text
processed/
  traj_xxx/
    0.jpg
    1.jpg
    ...
    traj_data.pkl
```

处理后的数据统一按 4Hz 采样，训练图像统一 resize 到 96x96。

## 2. TurtleBot3 Gazebo 鱼眼环境

### 2.1 环境特点

- ROS1 Noetic + Gazebo 11
- TurtleBot3 Waffle Pi
- 图像 topic：`/camera/fisheye/image_raw`
- 图像编码：`rgb8`
- 原始尺寸：480x480
- 原始频率：16Hz
- 水平视场角：约 160 度
- 处理后频率：4Hz
- 每次启动或切换场景时默认随机选择一个候选出生位姿

### 2.2 一次性安装

只需在首次配置或需要修复环境时执行：

```bash
bash scripts/turtlebot3_gazebo/setup_noetic.sh
```

该脚本不会修改 `~/.bashrc`。脚本会安装或检查 ROS Noetic，创建 `~/catkin_ws`，准备官方 TurtleBot3 仓库，并将 Waffle Pi 相机配置为 Gazebo 鱼眼相机。

### 2.3 可用的全部场景

| 场景 | 参数 | 启动采集命令 |
| --- | --- | --- |
| House 室内房屋 | `house` | `bash scripts/turtlebot3_gazebo/collect.sh --world house --name house_001` |
| TurtleBot3 World | `world` | `bash scripts/turtlebot3_gazebo/collect.sh --world world --name world_001` |
| Stage 1 | `stage_1` | `bash scripts/turtlebot3_gazebo/collect.sh --world stage_1 --name stage_1_001` |
| Stage 2 | `stage_2` | `bash scripts/turtlebot3_gazebo/collect.sh --world stage_2 --name stage_2_001` |
| Stage 3 | `stage_3` | `bash scripts/turtlebot3_gazebo/collect.sh --world stage_3 --name stage_3_001` |
| Stage 4 | `stage_4` | `bash scripts/turtlebot3_gazebo/collect.sh --world stage_4 --name stage_4_001` |
| Autorace | `autorace` | `bash scripts/turtlebot3_gazebo/collect.sh --world autorace --name autorace_001` |
| Autorace 2020 | `autorace_2020` | `bash scripts/turtlebot3_gazebo/collect.sh --world autorace_2020 --name autorace_2020_001` |
| Empty World | `empty` | `bash scripts/turtlebot3_gazebo/collect.sh --world empty --name empty_001` |

`empty` 缺少视觉特征，适合检查控制和录包功能，不建议作为正式训练数据的主要来源。

`--world` 只决定会话启动时加载的第一张地图。进入采集会话后，可以在
`record` 窗口按 `m` 切换到其他地图，因此不需要退出 tmux 或重新运行命令。

如果只想让每张地图固定使用第一个候选出生点进行调试：

```bash
bash scripts/turtlebot3_gazebo/collect.sh \
  --world house \
  --name house_debug \
  --random-pose false
```

如果不需要额外保存激光雷达和 IMU：

```bash
bash scripts/turtlebot3_gazebo/collect.sh \
  --world house \
  --name house_001 \
  --record-extra false
```

### 2.4 启动后如何操作

运行 `collect.sh` 后会自动进入一个 tmux session，其中有四个窗口：

```text
0 core     ROS master
1 gazebo   Gazebo 场景
2 teleop   WASD 键盘控制
3 record   手动录制和地图切换
```

tmux 切换窗口的方法是：先按下并松开 `Ctrl-B`，再按窗口编号。

```text
Ctrl-B，然后按 1    查看 Gazebo 日志
Ctrl-B，然后按 2    进入键盘控制
Ctrl-B，然后按 3    进入录制控制
```

脚本默认停留在 `record` 窗口，并等待相机和 odom topic。看到下面的提示后才表示可以录制：

```text
[record] Ready.
[record] idle world=house state=ready. Press r to record, m for map menu, q to quit:
```

### 2.5 WASD 控制

切换到 `teleop` 窗口后使用：

```text
w / s      前进 / 后退
a / d      左转 / 右转
W / S      加速前进 / 加速后退，即 Shift+w / Shift+s
A / D      加速左转 / 加速右转，即 Shift+a / Shift+d
空格       立即停止
h          显示帮助
q          退出键盘控制
```

TurtleBot3 是差速机器人，`a/d` 是转向，不是横向平移。控制程序带有约 0.65 秒 deadman timeout；持续移动时需要按住按键或连续按键，松开后机器人会自动停止。

### 2.6 录制一条或多条轨迹

一次启动场景后，可以连续录制任意多条轨迹，不需要每条轨迹都重新运行 `collect.sh`。

推荐操作顺序：

1. 保持在 `record` 窗口，按 `r` 开始录制。
2. 按 `Ctrl-B`，再按 `2`，切换到 `teleop`。
3. 使用 WASD 驾驶机器人。
4. 按空格停车。
5. 按 `Ctrl-B`，再按 `3`，返回 `record`。
6. 按 `r` 停止并保存当前轨迹。
7. 再按 `r` 开始下一条轨迹，重复上述步骤。
8. 当前地图采集完成后保持录制停止，按 `m`，输入地图编号并回车。
9. 等待 Gazebo 自动重启以及 `Map ready` 提示，再按 `r` 采集新地图。

每条 bag 会自动编号：

```text
/home/extra/datasets/turtlebot3_gazebo/raw/<world>/<name>/<name>_<world>_0001.bag
/home/extra/datasets/turtlebot3_gazebo/raw/<world>/<name>/<name>_<world>_0002.bag
/home/extra/datasets/turtlebot3_gazebo/raw/<world>/<name>/<name>_<world>_0003.bag
```

默认必录：

```text
/camera/fisheye/image_raw
/odom
/cmd_vel
/tf
/tf_static
/clock
```

`--record-extra true` 时还会录制：

```text
/scan
/imu
```

同一地图中连续录制多条 bag 时，机器人位置会延续。按 `m` 后选择另一张
地图，或再次选择当前地图进行重载，就会重新随机选择出生点。

地图菜单默认包含 8 张正式采集地图：

```text
house, world, stage_1, stage_2, stage_3, stage_4,
autorace, autorace_2020
```

`empty` 不在菜单中，但仍可用 `--world empty` 单独启动。录制过程中按 `m`
会被拒绝，必须先按 `r` 完整停止并保存当前 bag。

Gazebo Classic 不能在同一个进程中热加载另一份 world。按 `m` 后，脚本会只
重启 Gazebo 进程，GUI 会短暂关闭或停顿；`roscore`、teleop、record 和 tmux
会话都不会退出。切换完成前按 `r` 不会开始录制。

会话日志、地图状态、切换历史和每次生成的 launch 文件位于：

```text
/home/extra/datasets/turtlebot3_gazebo/raw/_sessions/<name>_<timestamp>/
```

### 2.7 录制前检查

另开一个普通终端，执行：

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

rostopic hz /camera/fisheye/image_raw
rostopic hz /odom
rostopic echo -n1 /camera/fisheye/image_raw/encoding
```

预期结果：

- 鱼眼图像约 16Hz
- odom 持续发布
- 图像 encoding 为 `rgb8`

停止一条轨迹后可以检查 bag：

```bash
rosbag info \
  /home/extra/datasets/turtlebot3_gazebo/raw/house/collection_001/collection_001_house_0001.bag
```

### 2.8 退出仿真

先在 `record` 窗口确认当前不处于 `RECORDING` 状态。如果正在录制，先按 `r` 停止并等待 bag 保存完成。

```text
record 窗口按 q       退出录制控制
teleop 窗口按空格     停车
teleop 窗口按 q       退出键盘控制
Ctrl-B，然后按 d      暂时离开 tmux，但仿真仍在后台运行
```

彻底结束仿真：

```bash
tmux ls
tmux kill-session -t <collect.sh 启动时打印的 session 名>
```

不要在 rosbag 正在写入时直接杀掉 tmux，否则可能留下 `.bag.active` 文件。

## 3. iGibson 环境

### 3.1 当前可用范围

本机当前已确认安装的 iGibson 场景是：

```text
Rs
```

`Rs` 是官方 demo Gibson 场景，可以直接采集。完整 iGibson 交互式室内场景的代码接口已经接好，但完整场景数据受官方许可约束，需要按 iGibson 官方流程申请和下载后才能使用。

iGibson 当前使用：

- RGB topic：`/camera/fisheye/image_raw`
- odom topic：`/odom`
- 控制 topic：`/cmd_vel`
- 图像编码：`rgb8`
- 原始图像尺寸：480x480
- 原始频率：16Hz，帧间隔 0.0625s
- 投影：equidistant，水平 FOV 160 度，完整方形鱼眼
- 处理后频率：4Hz

iGibson 当前版本的原生 fisheye renderer 不可用，因此仓库从机器人原有
相机位姿渲染六面 cubemap，再重投影成与 Gazebo 相同的 equidistant 鱼眼。
两边统一 RGB 投影、尺寸、频率和 topic，但保留各自机器人及相机的物理
安装位置。

### 3.2 一次性安装

```bash
bash scripts/igibson/setup_igibson.sh
```

该脚本会创建或复用名为 `igibson` 的 Python 3.8 conda 环境，并下载默认 assets 和 demo 场景 `Rs`。

手动进入 iGibson 环境：

```bash
source scripts/igibson/env.sh
```

`collect.sh` 内部也会自动加载该环境，因此正常采集时可以直接运行采集命令。

### 3.3 启动 `Rs` 场景

```bash
bash scripts/igibson/collect.sh \
  --scene Rs \
  --name rs_001
```

GUI 是手动采集的推荐模式，也是默认值：

```bash
bash scripts/igibson/collect.sh \
  --scene Rs \
  --name rs_001 \
  --mode gui_interactive
```

`headless` 更适合无显示器的 smoke test，不适合依赖画面观察的手动驾驶：

```bash
bash scripts/igibson/collect.sh \
  --scene Rs \
  --name rs_headless_test \
  --mode headless
```

### 3.4 使用完整 iGibson 场景

完成官方授权、数据下载和路径配置后，使用 `--scene-type igibson`：

```bash
bash scripts/igibson/collect.sh \
  --scene-type igibson \
  --scene Benevolence_1_int \
  --name benevolence_001
```

如果数据不在默认目录，可使用 iGibson 工具修改数据路径：

```bash
source scripts/igibson/env.sh
python -m igibson.utils.assets_utils --change_data_path
```

场景名必须与已下载数据中的目录名一致。未下载的场景不能仅靠修改 `--scene` 参数启动。

### 3.5 iGibson 的控制与录制

iGibson 也会创建四个 tmux 窗口：

```text
0 core     ROS master
1 sim      iGibson 仿真
2 teleop   WASD 键盘控制
3 record   手动开始或停止 rosbag
```

控制、窗口切换和录制方法与 TurtleBot3 Gazebo 完全相同：

```text
Ctrl-B，然后按 2    进入 teleop
w/a/s/d             移动
Shift+w/a/s/d       加速移动
空格                停止
Ctrl-B，然后按 3    返回 record
r                   开始录制
r                   停止并保存
q                   退出当前控制程序
```

iGibson bag 自动保存为：

```text
/home/extra/datasets/igibson/raw/<scene>/<name>/<name>_0001.bag
/home/extra/datasets/igibson/raw/<scene>/<name>/<name>_0002.bag
```

默认必录：

```text
/camera/fisheye/image_raw
/odom
/cmd_vel
/tf
```

`--record-extra true` 时还会录制 depth、lidar 和 ground-truth odom。

另开终端检查：

```bash
source scripts/igibson/env.sh
rostopic hz /camera/fisheye/image_raw
rostopic hz /odom
rostopic echo -n1 /camera/fisheye/image_raw/encoding
rostopic echo -n1 /camera/fisheye/image_raw/width
rostopic echo -n1 /camera/fisheye/image_raw/height
rostopic echo -n1 /camera/fisheye/camera_info/distortion_model
```

## 4. 将 rosbag 转换为训练数据

### 4.1 处理 TurtleBot3 Gazebo 数据

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
bash scripts/turtlebot3_gazebo/process.sh
```

默认输入和输出：

```text
raw:       /home/extra/datasets/turtlebot3_gazebo/raw
processed: /home/extra/datasets/turtlebot3_gazebo/processed
split:     train/vint_train/data/data_splits/turtlebot3_gazebo/
```

### 4.2 处理 iGibson 数据

```bash
source scripts/igibson/env.sh
bash scripts/igibson/process.sh
```

默认输入和输出：

```text
raw:       /home/extra/datasets/igibson/raw
processed: /home/extra/datasets/igibson/processed
split:     train/vint_train/data/data_splits/igibson/
```

默认处理器只读取新的 `/camera/fisheye/image_raw`。旧 pinhole bag 如需单独
转换，应放入 `/home/extra/datasets/igibson_legacy/raw`，然后执行：

```bash
bash scripts/igibson/process.sh --legacy true
```

它会使用独立的 `igibson_legacy` 输出目录和 split，不会混入正式鱼眼训练。

两个处理脚本都会：

- 从 rosbag 提取图像和 odom
- 按 4Hz 输出训练帧
- 保留原地转向片段，不启用 backwards filter
- 生成 80% train / 20% test split
- 检查轨迹长度和步距
- 在真实步距明显偏离 0.05m 时给出提示，但不会自动修改配置

## 5. 检查处理后的数据

检查 TurtleBot3：

```bash
python3 scripts/turtlebot3_gazebo/check_processed_dataset.py \
  --data-folder /home/extra/datasets/turtlebot3_gazebo/processed
```

检查 iGibson：

```bash
python3 scripts/turtlebot3_gazebo/check_processed_dataset.py \
  --data-folder /home/extra/datasets/igibson/processed
```

检查内容包括：

- 是否存在轨迹目录
- jpg 是否从 `0.jpg` 开始连续编号
- `traj_data.pkl` 是否包含 `position` 和 `yaw`
- 图像、position、yaw 长度是否一致
- 短轨迹数量
- 平均步距和中位步距
- `median_valid_step_distance`
- near-zero step ratio

如果 near-zero ratio 很高，应减少长时间停留或纯原地旋转。当前训练配置为 `learn_angle=False`，大量只有角度变化、没有平移的样本对 action label 的帮助有限。

## 6. 可视化抽查

TurtleBot3：

```bash
python3 scripts/turtlebot3_gazebo/visualize_samples.py \
  --data-folder /home/extra/datasets/turtlebot3_gazebo/processed
```

输出目录：

```text
/home/extra/datasets/turtlebot3_gazebo/debug_vis/
```

iGibson：

```bash
python3 scripts/turtlebot3_gazebo/visualize_samples.py \
  --data-folder /home/extra/datasets/igibson/processed \
  --output-dir /home/extra/datasets/igibson/debug_vis \
  --source-topic /camera/fisheye/image_raw \
  --camera-description "480x480 rgb8 equidistant fisheye, FOV 160deg" \
  --raw-frequency "16Hz"
```

可视化图会展示 current/goal 图像、世界轨迹、local GT waypoints、local goal position 和 distance label。正式训练前应重点确认：

- 图像颜色正确
- current/goal 时间顺序正确
- 世界轨迹连续
- local waypoint 朝向正确
- goal position 与机器人 yaw 一致

## 7. 启动训练

训练 TurtleBot3 Gazebo 数据：

```bash
cd /home/czl/visualnav-mamba/train
python3 train.py -c ./config/turtlebot3_gazebo.yaml
```

训练 iGibson 数据：

```bash
cd /home/czl/visualnav-mamba/train
python3 train.py -c ./config/igibson.yaml
```

不连接在线 wandb 的 smoke test：

```bash
WANDB_MODE=offline python3 train.py -c ./config/turtlebot3_gazebo.yaml
```

或：

```bash
WANDB_MODE=offline python3 train.py -c ./config/igibson.yaml
```

## 8. 推荐的实际采集策略

为了降低场景过拟合，不要只在 House 中反复走同一条路线。建议：

1. TurtleBot3 Gazebo 的 House、World、Stage 1-4 和两个 Autorace 场景都采集。
2. 使用 `m` 选择地图；需要新出生点时重新选择当前地图即可。
3. 每次启动连续录制多条 20-60 秒轨迹。
4. 路线包含直行、缓慢转弯、绕障、路口选择和不同朝向。
5. 避免长时间静止、连续撞墙和长时间纯原地旋转。
6. `empty` 只用于 smoke test。
7. 加入 iGibson `Rs`，并在取得完整数据集后增加更多室内场景。
8. train/test 最好按采集 session 或场景隔离，而不仅仅随机拆分相似轨迹，以更真实地检查泛化能力。

## 9. 常见问题与日志

### 场景启动后 record 一直等待

查看对应 run 目录中的日志：

```text
core.log
gazebo.log 或 sim.log
teleop.log
record.log
```

TurtleBot3 示例：

```bash
tail -n 100 \
  /home/extra/datasets/turtlebot3_gazebo/raw/_sessions/<会话目录>/gazebo.log
```

iGibson 示例：

```bash
tail -n 100 \
  /home/extra/datasets/igibson/raw/Rs/rs_001/sim.log
```

### 找不到当前 tmux session

```bash
tmux ls
```

重新进入：

```bash
tmux attach-session -t <session 名>
```

### 出现 `.bag.active`

这通常表示录制进程被强制中断。先确认没有 rosbag 进程仍在运行，再检查该文件：

```bash
ps aux | grep '[r]osbag record'
```

不要直接把 `.bag.active` 当作完整训练数据处理。正常停止录制时应按 record 窗口中的 `r`，等待出现 `Saved:`。

### 最短可执行流程

TurtleBot3 House：

```bash
cd /home/czl/visualnav-mamba
bash scripts/turtlebot3_gazebo/collect.sh --world house --name collection_001
```

进入后：

```text
r -> Ctrl-B 2 -> WASD 驾驶 -> 空格 -> Ctrl-B 3 -> r
m -> 输入地图编号 -> 等待 Map ready -> r 开始下一张地图
```

处理并训练：

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
bash scripts/turtlebot3_gazebo/process.sh

cd train
python3 train.py -c ./config/turtlebot3_gazebo.yaml
```

iGibson `Rs`：

```bash
cd /home/czl/visualnav-mamba
bash scripts/igibson/collect.sh --scene Rs --name rs_001
```

进入后同样执行：

```text
r -> Ctrl-B 2 -> WASD 驾驶 -> 空格 -> Ctrl-B 3 -> r
```

处理并训练：

```bash
source scripts/igibson/env.sh
bash scripts/igibson/process.sh

cd train
python3 train.py -c ./config/igibson.yaml
```

## CARLA 鱼眼导航与探索

CARLA 仿真器和原始数据保留在 `/home/czl/CARLA`，模型、checkpoint、topomap 与部署脚本统一由本仓库管理。

```bash
cd /home/czl/visualnav-mamba
bash scripts/carla/carla_vint_quickstart.sh check
bash scripts/carla/carla_vint_quickstart.sh start
bash scripts/carla/carla_vint_quickstart.sh navigate \
  --dir town01_route01_topomap \
  --no-preview-window
```

无目标条件探索：

```bash
bash scripts/carla/carla_vint_quickstart.sh explore \
  --map Town10HD_Opt \
  --spawn-index 0 \
  --no-preview-window
```

`navigate` 使用 topomap 目标图像；`explore` 强制关闭 topomap 并使用 goal-masked 无条件策略。
采集、topomap 导出、调试参数和日志分析说明见 `scripts/carla/README.md`。旧的
`/home/czl/CARLA/carla_vint_quickstart.sh` 保留为兼容入口。
