# CARLA navigation and exploration

CARLA 仿真器与原始数据保留在 `/home/czl/CARLA`；模型、checkpoint、topomap、运行日志和
CARLA 适配脚本由 `/home/czl/visualnav-mamba` 统一管理。所有命令均可从任意当前目录运行。

## 准备环境

```bash
cd /home/czl/visualnav-mamba
bash scripts/carla/carla_vint_quickstart.sh check
bash scripts/carla/carla_vint_quickstart.sh start
```

默认使用 `/home/czl/anaconda3/envs/carla_vint/bin/python`。如需覆盖：

```bash
CARLA_PYTHON=/path/to/python \
  bash scripts/carla/carla_vint_quickstart.sh check
```

## Navigate：topomap 目标导航

`navigate` 将当前鱼眼观测与 topomap 邻域节点匹配，选择子目标图像，通过条件/无条件
guidance 生成多条扩散轨迹，再转换为 CARLA 油门、转向和制动控制。

```bash
bash scripts/carla/carla_vint_quickstart.sh navigate \
  --dir town01_route01_topomap \
  --no-preview-window
```

`deploy` 是 `navigate` 的兼容别名：

```bash
bash scripts/carla/carla_vint_quickstart.sh deploy \
  --dir town01_route01_topomap \
  --no-preview-window
```

只检查定位和模型输出，不让车辆移动：

```bash
bash scripts/carla/carla_vint_quickstart.sh navigate \
  --dir town01_route01_topomap \
  --no-preview-window \
  --no-control \
  --control-debug
```

使用 CARLA 真值位置隔离视觉定位问题：

```bash
bash scripts/carla/carla_vint_quickstart.sh navigate \
  --dir town01_route01_topomap \
  --no-preview-window \
  --gt-topomap-localization \
  --gt-subgoal-offset 3
```

## Explore：无目标条件探索

`explore` 不加载 topomap。模型使用随机假目标和 goal mask，只根据观测上下文进行无目标
扩散采样；默认执行第一条采样轨迹，与 `deployment/src/explore.py` 的策略保持一致。

```bash
bash scripts/carla/carla_vint_quickstart.sh explore \
  --map Town10HD_Opt \
  --spawn-index 0 \
  --no-preview-window
```

安全检查探索输出但不移动：

```bash
bash scripts/carla/carla_vint_quickstart.sh explore \
  --map Town10HD_Opt \
  --spawn-index 0 \
  --no-preview-window \
  --no-control \
  --control-debug
```

Explore 模式会强制关闭 topomap，不能与 `--gt-topomap-localization` 同时使用。也可使用快捷入口：

```bash
bash scripts/carla/carla_vint_nomad_explore.sh --no-preview-window
```

## 数据、topomap 与日志

```bash
# 采集训练轨迹
bash scripts/carla/carla_vint_quickstart.sh collect

# 采集并导出 topomap 路线
bash scripts/carla/carla_vint_quickstart.sh collect-topomap \
  --name town01_route01_topomap \
  --stride 4

# 分析最近一次 navigate 或 explore 日志
bash scripts/carla/carla_vint_quickstart.sh analyze
```

日志保存在 `deployment/logs/carla_runs/`。日志起始记录中的 `mode` 为 `navigate` 或 `explore`；
Explore 日志中的 `topomap` 和定位字段为空。

## 常用覆盖项

- `--checkpoint`：临时覆盖默认 checkpoint；相对路径按仓库根目录解析。
- `--model-config`：覆盖 CARLA 模型注册表。
- `--topomap-root`：覆盖 Navigate 使用的 topomap 根目录。
- `--frame-rate`：观测上下文和控制循环频率；当前 CARLA 训练数据对应 `8 Hz`。
- `--metric-waypoint-spacing`：归一化 waypoint 的米制还原尺度；当前 CARLA 数据对应 `0.5 m`。
- `--trajectory-selection first|median|random`：选择执行哪条扩散轨迹。
- `--no-control`：运行完整推理但保持车辆刹停。
- `--no-preview-window`：关闭 OpenCV 窗口，适合服务器环境。
- `CARLA_WORKSPACE`、`CARLA_ROOT`：覆盖仿真器位置。
- `CARLA_DATA_DIR`、`CARLA_TOPOMAP_TRAJ_DIR`：覆盖原始数据位置。

按 `Ctrl+C` 会写入日志结束事件并清理相机和车辆。
