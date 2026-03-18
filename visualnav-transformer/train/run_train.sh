#!/bin/bash
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate mamba
# export PYTHONPATH="/workspace/visualnav-transformer/diffusion_policy:$PYTHONPATH"

# 设置 HuggingFace 镜像解决网络超时问题
export HF_ENDPOINT="https://hf-mirror.com"

# 使用 nohup 后台运行，日志输出到 training.log
nohup python3 train.py -c ./config/nomad_mamba.yaml > training.log 2>&1 &
echo "训练已在后台启动，PID: $!"
echo "查看日志: tail -f training.log"
