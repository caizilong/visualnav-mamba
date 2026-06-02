# 使用官方 NVIDIA CUDA 12.4.1 + cuDNN 开发镜像（基于 Ubuntu 20.04）
# 该镜像预装了 NVIDIA 驱动兼容的 CUDA 和 cuDNN
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 设置非交互式安装模式，避免 apt 安装过程中弹出交互式配置界面
ENV DEBIAN_FRONTEND=noninteractive

# # 设置 Conda 安装路径（将 Miniconda 安装到 /opt/conda）
# ENV CONDA_DIR=/opt/conda

# # 将 Conda 的 bin 目录加入 PATH，使 conda、python 等命令全局可用
# ENV PATH=$CONDA_DIR/bin:$PATH

# 设置默认 SHELL 为 bash，确保后续 RUN 指令在 bash 环境中执行（支持高级 shell 特性）
SHELL ["/bin/bash", "-c"]

# 默认工作目录
WORKDIR /workspace

# 1. 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git lsb-release gnupg gnupg2 gdb vim cmake htop net-tools build-essential libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 2. 安装 ROS2 Humble
# 添加密钥（使用 keyring）
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

# 添加软件源
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" \
    > /etc/apt/sources.list.d/ros2.list

# 安装 ROS2 Humble Desktop
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool \
    && rm -rf /var/lib/apt/lists/*

# 自动 source
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# # 3. 创建 conda 环境
# # 下载并静默安装 Miniconda（Python 3 版本）
# # -b: 批处理模式（非交互）
# # -p: 指定安装路径为 $CONDA_DIR
# RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
#     bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
#     rm /tmp/miniconda.sh && \
#     conda clean --all -y

# # 初始化 Conda 以支持 bash shell（写入 ~/.bashrc）
# RUN conda init bash

# # 创建名为 mamba_train 的 Conda 环境（Python 3.9）
# RUN conda config --set always_yes yes --set changeps1 no && \
#     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
#     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
#     conda create -y -n mamba_train python=3.9 && conda clean -a -y
# SHELL ["conda", "run", "-n", "mamba_train", "/bin/bash", "-c"]
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir \
#     torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
#     --index-url https://download.pytorch.org/whl/cu124

# # 安装项目依赖
# COPY requirements.txt /workspace/requirements.txt
# RUN pip install --no-cache-dir --upgrade pip && \ 
#     pip install --no-cache-dir -r /workspace/requirements.txt --no-build-isolation && \
#     git clone https://github.com/real-stanford/diffusion_policy.git /workspace/diffusion_policy && \
#     pip install --no-cache-dir -e /workspace/diffusion_policy

# # 默认启动 bash，并激活 mamba_train
# CMD ["conda", "run", "-n", "mamba_train", "/bin/bash", "-c"]
