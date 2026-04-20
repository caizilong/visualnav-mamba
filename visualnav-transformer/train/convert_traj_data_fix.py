#!/usr/bin/env python3
"""
拯救旧数据集的终极转换脚本
功能 1：兼容读取 Numpy 2.x 产生的数据
功能 2：将旧版代码产生的 Python List 强制转换为训练所需的 np.float32 矩阵
功能 3：以 Numpy 1.x 的格式重新保存，彻底解决训练管线报错
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        elif module.startswith("numpy.core") and not hasattr(np, 'core'):
            module = module.replace("numpy.core", "numpy._core")
        return super().find_class(module, name)

def load_pickle_compat(filepath):
    with open(filepath, "rb") as f:
        return NumpyCompatUnpickler(f).load()

def convert_and_salvage_traj_data(data_folder: str, backup: bool = True):
    data_path = Path(data_folder)
    if not data_path.exists():
        print(f"错误：路径不存在 {data_folder}")
        return False
    
    traj_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("trajectory_")])
    print(f"找到 {len(traj_dirs)} 个轨迹目录，准备开始洗白数据...")
    
    for traj_dir in traj_dirs:
        pkl_path = traj_dir / "traj_data.pkl"
        if not pkl_path.exists():
            continue
        
        try:
            # 1. 兼容读取旧数据
            traj_data = load_pickle_compat(pkl_path)
            
            # 2. 备份防丢失
            if backup:
                backup_path = traj_dir / "traj_data.pkl.bak"
                if not backup_path.exists():
                    with open(pkl_path, "rb") as src:
                        with open(backup_path, "wb") as dst:
                            dst.write(src.read())
            
            # 3. 核心修复：把不管是什么格式的数据，统统强制转成 float32 类型的 numpy array
            converted_data = {
                'position': np.array(traj_data['position'], dtype=np.float32),
                'yaw': np.array(traj_data['yaw'], dtype=np.float32)
            }
            
            # 如果你旧数据里还存了别的，也一并转过去
            if 'action' in traj_data:
                converted_data['action'] = np.array(traj_data['action'], dtype=np.float32)
            if 'reward' in traj_data:
                converted_data['reward'] = np.array(traj_data['reward'], dtype=np.float32)
            
            # 4. 在当前环境 (Numpy 1.x) 下重新打包存回硬盘
            with open(pkl_path, "wb") as f:
                pickle.dump(converted_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"  [成功] 轨迹 {traj_dir.name} 已修复为标准 NumPy 矩阵格式。")
            
        except Exception as e:
            print(f"  [错误] 修复 {traj_dir.name} 失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python convert_traj_data_fix.py <数据集路径>")
        sys.exit(1)
    
    convert_and_salvage_traj_data(sys.argv[1])