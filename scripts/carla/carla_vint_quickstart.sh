#!/usr/bin/env bash
# Unified CARLA workflow for the visualnav-mamba repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VINT_ROOT="${VISUALNAV_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CARLA_WORKSPACE="${CARLA_WORKSPACE:-/home/czl/CARLA}"
CARLA_ROOT="${CARLA_ROOT:-$CARLA_WORKSPACE/CARLA_Latest}"
CARLA_DATA_DIR="${CARLA_DATA_DIR:-$CARLA_WORKSPACE/carla_fisheye_dataset}"
CARLA_TOPOMAP_TRAJ_DIR="${CARLA_TOPOMAP_TRAJ_DIR:-$CARLA_WORKSPACE/carla_topomap_trajectories}"
TOPOMAP_ROOT="${CARLA_TOPOMAP_ROOT:-$VINT_ROOT/deployment/topomaps}"
MODEL_CONFIG="${CARLA_MODEL_CONFIG:-$VINT_ROOT/deployment/config/models_carla.yaml}"
BENCHMARK_CONFIG="${CARLA_BENCHMARK_CONFIG:-$VINT_ROOT/deployment/config/benchmark_carla_nomad_mamba.yaml}"
DEFAULT_ENV_PYTHON="/home/czl/anaconda3/envs/carla_vint/bin/python"

if [[ -n "${CARLA_PYTHON:-}" ]]; then
    PYTHON_BIN="$CARLA_PYTHON"
elif [[ -x "$DEFAULT_ENV_PYTHON" ]]; then
    PYTHON_BIN="$DEFAULT_ENV_PYTHON"
else
    PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "错误: 未找到可执行 Python。请设置 CARLA_PYTHON。" >&2
    exit 1
fi

header() {
    printf '\n=== VisualNav-Mamba CARLA: %s ===\n' "$1"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "错误: 未找到文件: $1" >&2
        return 1
    fi
}

check_module() {
    "$PYTHON_BIN" -c "import $1" >/dev/null 2>&1
}

check_environment() {
    header "环境检查"
    local failed=0
    printf 'VisualNav root : %s\n' "$VINT_ROOT"
    printf 'CARLA workspace: %s\n' "$CARLA_WORKSPACE"
    printf 'CARLA root     : %s\n' "$CARLA_ROOT"
    printf 'Python         : %s\n' "$PYTHON_BIN"
    printf 'Dataset        : %s\n' "$CARLA_DATA_DIR"
    printf 'Topomap routes : %s\n' "$CARLA_TOPOMAP_TRAJ_DIR"
    printf 'Topomap root   : %s\n' "$TOPOMAP_ROOT"
    printf 'Model config   : %s\n' "$MODEL_CONFIG"
    printf 'Benchmark      : %s\n' "$BENCHMARK_CONFIG"

    for path in \
        "$CARLA_ROOT/CarlaUE4.sh" \
        "$MODEL_CONFIG" \
        "$BENCHMARK_CONFIG" \
        "$SCRIPT_DIR/carla_navigate.py"; do
        if [[ -f "$path" ]]; then
            printf '[ok] file %s\n' "$path"
        else
            printf '[missing] file %s\n' "$path"
            failed=1
        fi
    done

    local checkpoint="$VINT_ROOT/deployment/model_weights/ema_latest.pth"
    if [[ -e "$checkpoint" ]]; then
        printf '[ok] checkpoint %s -> %s\n' "$checkpoint" "$(readlink -f "$checkpoint")"
    else
        printf '[missing] checkpoint %s\n' "$checkpoint"
        failed=1
    fi

    local modules=(carla torch torchvision yaml diffusers mamba_ssm einops PIL numpy)
    for module in "${modules[@]}"; do
        if check_module "$module"; then
            printf '[ok] python module %s\n' "$module"
        else
            printf '[missing] python module %s\n' "$module"
            failed=1
        fi
    done

    if [[ -d "$TOPOMAP_ROOT/images" ]]; then
        printf 'Available topomaps:\n'
        find "$TOPOMAP_ROOT/images" -mindepth 1 -maxdepth 1 -type d -printf '  - %f\n' | sort
    else
        printf '[missing] topomap images directory %s/images\n' "$TOPOMAP_ROOT"
        failed=1
    fi
    return "$failed"
}

start_carla() {
    header "启动 CARLA"
    require_file "$CARLA_ROOT/CarlaUE4.sh"
    if pgrep -f '[C]arlaUE4' >/dev/null; then
        echo "CARLA 已经在运行。"
        return 0
    fi
    (
        cd "$CARLA_ROOT"
        ./CarlaUE4.sh -quality-level=Low -RenderOffScreen &
    )
    echo "等待 CARLA 服务启动……"
    sleep "${CARLA_START_WAIT_SECONDS:-10}"
    if ! pgrep -f '[C]arlaUE4' >/dev/null; then
        echo "错误: CARLA 进程未启动。" >&2
        return 1
    fi
    echo "CARLA 启动完成。"
}

stop_carla() {
    header "停止 CARLA"
    if pgrep -f '[C]arlaUE4' >/dev/null; then
        pkill -f '[C]arlaUE4'
        echo "已发送停止信号。"
    else
        echo "CARLA 未在运行。"
    fi
}

collect_data() {
    header "采集鱼眼训练轨迹"
    mkdir -p "$CARLA_DATA_DIR"
    "$PYTHON_BIN" "$SCRIPT_DIR/carla_simple_control.py" \
        --output_dir "$CARLA_DATA_DIR" "$@"
}

collect_topomap() {
    header "采集并导出 topomap 轨迹"
    local name="fisheye_topomap"
    local stride="4"
    local export_after=1
    local collector_args=()
    local before=""
    [[ -d "$CARLA_TOPOMAP_TRAJ_DIR" ]] && before="$(find "$CARLA_TOPOMAP_TRAJ_DIR" -maxdepth 1 -type d -name 'trajectory_*' -printf '%f\n' | sort | tail -n 1)"
    while (($#)); do
        case "$1" in
            --name) name="$2"; shift 2 ;;
            --stride) stride="$2"; shift 2 ;;
            --no-export) export_after=0; shift ;;
            *) collector_args+=("$1"); shift ;;
        esac
    done
    mkdir -p "$CARLA_TOPOMAP_TRAJ_DIR"
    "$PYTHON_BIN" "$SCRIPT_DIR/carla_simple_control.py" \
        --output_dir "$CARLA_TOPOMAP_TRAJ_DIR" \
        --record-role topomap \
        "${collector_args[@]}"
    if ((export_after)); then
        local after
        after="$(find "$CARLA_TOPOMAP_TRAJ_DIR" -maxdepth 1 -type d -name 'trajectory_*' -printf '%f\n' | sort | tail -n 1)"
        if [[ -z "$after" || "$after" == "$before" ]]; then
            echo "错误: 未检测到新保存的 topomap 轨迹。" >&2
            return 1
        fi
        "$PYTHON_BIN" "$SCRIPT_DIR/carla_export_topomap.py" \
            --dataset-dir "$CARLA_TOPOMAP_TRAJ_DIR" \
            --topomap-root "$TOPOMAP_ROOT" \
            --trajectory "$after" \
            --name "$name" \
            --stride "$stride"
    fi
}

export_topomap() {
    local dataset_dir="$1"
    shift
    "$PYTHON_BIN" "$SCRIPT_DIR/carla_export_topomap.py" \
        --dataset-dir "$dataset_dir" \
        --topomap-root "$TOPOMAP_ROOT" \
        "$@"
}

run_nomad() {
    local mode="$1"
    shift
    header "NoMaD-Mamba CARLA ${mode}"
    require_file "$MODEL_CONFIG"
    require_file "$BENCHMARK_CONFIG"
    for module in torch mamba_ssm einops carla; do
        if ! check_module "$module"; then
            echo "错误: $PYTHON_BIN 缺少模块 $module。" >&2
            return 1
        fi
    done
    export PYTHONPATH="$VINT_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
    export CARLA_ROOT CARLA_WORKSPACE VISUALNAV_ROOT="$VINT_ROOT"
    exec "$PYTHON_BIN" "$SCRIPT_DIR/carla_navigate.py" \
        --mode "$mode" \
        --benchmark-config "$BENCHMARK_CONFIG" \
        --model-config "$MODEL_CONFIG" \
        --topomap-root "$TOPOMAP_ROOT" \
        --carla-root "$CARLA_ROOT" \
        "$@"
}

analyze_logs() {
    header "分析 CARLA 部署日志"
    "$PYTHON_BIN" "$SCRIPT_DIR/carla_analyze_motion_log.py" "$@"
}

show_help() {
    cat <<EOF
用法: bash scripts/carla/carla_vint_quickstart.sh <command> [args]

命令:
  check             检查路径、checkpoint 和 Python 依赖（不自动安装）
  start             后台启动 CARLA offscreen 服务
  stop              停止 CARLA 服务
  collect           采集鱼眼训练轨迹
  collect-topomap   采集 topomap 专用轨迹并自动导出
  topomap           从训练数据目录导出 topomap
  topomap-route     从 topomap 专用轨迹目录导出 topomap
  navigate          使用 topomap 目标执行闭环导航
  explore           不使用 topomap，执行无目标条件探索
  deploy            navigate 的兼容别名
  analyze           分析最近或指定的部署日志

可覆盖环境变量:
  CARLA_WORKSPACE CARLA_ROOT CARLA_DATA_DIR CARLA_TOPOMAP_TRAJ_DIR
  CARLA_TOPOMAP_ROOT CARLA_MODEL_CONFIG CARLA_BENCHMARK_CONFIG CARLA_PYTHON
EOF
}

command="${1:-help}"
[[ $# -gt 0 ]] && shift
case "$command" in
    check) check_environment "$@" ;;
    start) start_carla "$@" ;;
    stop) stop_carla "$@" ;;
    collect) collect_data "$@" ;;
    collect-topomap) collect_topomap "$@" ;;
    topomap) export_topomap "$CARLA_DATA_DIR" "$@" ;;
    topomap-route) export_topomap "$CARLA_TOPOMAP_TRAJ_DIR" "$@" ;;
    navigate) run_nomad navigate "$@" ;;
    explore) run_nomad explore "$@" ;;
    deploy) run_nomad navigate "$@" ;;
    analyze) analyze_logs "$@" ;;
    help|-h|--help) show_help ;;
    *) echo "错误: 未知命令 $command" >&2; show_help; exit 2 ;;
esac
