#!/usr/bin/env python3
"""
转换 traj_data.pkl 文件，将 np.array 转换为 Python 原生列表格式，
以解决不同 numpy 版本之间的兼容性问题（numpy._core 模块不存在）。

使用方法:
    python convert_traj_data.py /path/to/datasets/carla_dataset
"""

import os
import sys
import pickle
import io
import numpy as np
from pathlib import Path


class NumpyCompatUnpickler(pickle.Unpickler):
    """
    自定义 Unpickler，处理 numpy 2.x 和 numpy 1.x 之间的兼容性问题。
    numpy 2.x 将部分模块移到了 numpy._core，而 numpy 1.x 使用 numpy.core。
    """
    def find_class(self, module, name):
        # 将 numpy._core 重定向到 numpy.core（numpy 1.x 兼容）
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        # 将 numpy.core 重定向到 numpy._core（numpy 2.x 兼容）
        elif module.startswith("numpy.core") and not hasattr(np, 'core'):
            module = module.replace("numpy.core", "numpy._core")
        return super().find_class(module, name)


def load_pickle_compat(filepath):
    """
    兼容性加载 pickle 文件，处理不同 numpy 版本的差异。
    """
    with open(filepath, "rb") as f:
        return NumpyCompatUnpickler(f).load()


def convert_numpy_to_list(obj):
    """
    递归地将 numpy 数组转换为 Python 原生列表。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_list(item) for item in obj)
    else:
        return obj


def convert_traj_data(data_folder: str, backup: bool = True):
    """
    转换指定数据集文件夹中所有 trajectory 的 traj_data.pkl 文件。
    
    Args:
        data_folder: 数据集根目录路径（如 /workspace/datasets/carla_dataset）
        backup: 是否备份原始文件
    """
    data_path = Path(data_folder)
    
    if not data_path.exists():
        print(f"错误：路径不存在 {data_folder}")
        return False
    
    # 查找所有 trajectory 目录
    traj_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("trajectory_")])
    
    if not traj_dirs:
        print(f"警告：在 {data_folder} 中未找到 trajectory_* 目录")
        return False
    
    print(f"找到 {len(traj_dirs)} 个轨迹目录")
    
    converted_count = 0
    error_count = 0
    
    for traj_dir in traj_dirs:
        pkl_path = traj_dir / "traj_data.pkl"
        
        if not pkl_path.exists():
            print(f"  跳过 {traj_dir.name}: traj_data.pkl 不存在")
            continue
        
        try:
            # 读取原始数据（使用兼容性加载器）
            traj_data = load_pickle_compat(pkl_path)
            
            # 检查是否需要转换
            needs_conversion = False
            for key, value in traj_data.items():
                if isinstance(value, np.ndarray):
                    needs_conversion = True
                    break
            
            if not needs_conversion:
                print(f"  跳过 {traj_dir.name}: 已经是兼容格式")
                continue
            
            # 备份原始文件
            if backup:
                backup_path = traj_dir / "traj_data.pkl.bak"
                if not backup_path.exists():
                    with open(pkl_path, "rb") as src:
                        with open(backup_path, "wb") as dst:
                            dst.write(src.read())
            
            # 转换数据
            converted_data = convert_numpy_to_list(traj_data)
            
            # 保存转换后的数据
            with open(pkl_path, "wb") as f:
                pickle.dump(converted_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"  已转换 {traj_dir.name}")
            converted_count += 1
            
        except Exception as e:
            print(f"  错误 {traj_dir.name}: {e}")
            error_count += 1
    
    print(f"\n转换完成: 成功 {converted_count}, 错误 {error_count}")
    return error_count == 0


def main():
    if len(sys.argv) < 2:
        print("用法: python convert_traj_data.py <数据集路径>")
        print("示例: python convert_traj_data.py /workspace/datasets/carla_dataset")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    backup = "--no-backup" not in sys.argv
    
    print(f"开始转换数据集: {data_folder}")
    print(f"备份原始文件: {'是' if backup else '否'}")
    print("-" * 50)
    
    success = convert_traj_data(data_folder, backup=backup)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
