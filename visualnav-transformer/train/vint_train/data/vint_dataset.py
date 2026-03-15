import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from PIL import Image

from vint_train.data.data_utils import (
    resize_and_aspect_crop,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)


class ViNT_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
    ):
        """
        ViNT 训练与评估使用的主数据集类。

        每条样本由：
        - 上下文观测序列（多帧图像, obs_image）
        - 目标图像（goal_image）
        - 轨迹动作序列（actions）
        - 观测到目标之间的离散时间距离（distance）
        - 目标在机器人坐标系下的位置（goal_pos）
        - 数据集索引（which_dataset）
        - 动作是否有效的掩码（action_mask）

        Args:
            data_folder (string): 所有轨迹与图像数据所在根目录
            data_split_folder (string): 某个 split 的目录，内含 traj_names.txt 等索引文件
            dataset_name (string): 数据集名称 [recon, go_stanford, scand, tartandrive, ...]
            image_size (Tuple[int, int]): 输出图像大小 (H, W)
            waypoint_spacing (int): 相邻 waypoint 之间的时间间隔（以帧数计）
            min_dist_cat (int): 使用的最小距离类别
            max_dist_cat (int): 使用的最大距离类别
            min_action_distance (int): 计算动作损失时允许的最小时间距离
            max_action_distance (int): 计算动作损失时允许的最大时间距离
            negative_mining (bool): 是否使用 ViNG 中提出的负样本挖掘策略
            len_traj_pred (int): 需要预测的未来 waypoint 数量
            learn_angle (bool): 是否同时预测机器人朝向（yaw）
            context_size (int): 用作上下文的历史观测帧数
            context_type (str): 上下文采样方式（目前仅支持 temporal）
            end_slack (int): 轨迹尾部丢弃的时间步数
            goals_per_obs (int): 每个观测采样多少个目标
            normalize (bool): 是否对动作和目标位置做归一化
            obs_type (str): 观测模态（目前为 "image"）
            goal_type (str): 目标模态（目前为 "image"）
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        # 1) 读取该 split 中包含的轨迹名称列表
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        # 2) 基于距离类别构建可用的标签集合
        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle

        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type

        # 3) 读取每个 dataset 的度量配置（例如 metric_waypoint_spacing）
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        # 4) 轨迹缓存：避免频繁从磁盘重复读取同一条轨迹
        self.trajectory_cache = {}
        self._load_index()
        self._build_caches()
        
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        """
        使用 LMDB 建立图像缓存，加速读取。

        - 首先确保所有轨迹的元数据已加载到内存
        - 若缓存文件不存在，则按索引逐张写入 LMDB
        - 构建缓存时直接将图像 resize 到 image_size，训练时无需重复 resize
        - 最后以只读方式重新打开 LMDB 环境
        """
        w, h = self.image_size[0], self.image_size[1]
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}_{w}x{h}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset.
        Images are resized to image_size during cache building to avoid resize at training time.
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name} ({w}x{h})"
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        img = Image.open(image_path).convert("RGB")
                        img_tensor = resize_and_aspect_crop(img, self.image_size)
                        img_np = img_tensor.numpy().astype(np.float32)
                        txn.put(image_path.encode(), pickle.dumps(img_np))

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        """
        构建样本索引：
        - samples_index: (traj_name, curr_time, max_goal_distance)
        - goals_index: (traj_name, goal_time)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        在同一条轨迹的未来随机采样一个目标，或采样负样本。

        Returns:
            (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        采样一个“负样本”目标：大概率来自不同轨迹，使得目标与当前观测不匹配。
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        从磁盘加载或重新构建样本索引：
        - index_to_data: 对应 __getitem__ 中的 f_curr, curr_time, max_goal_dist
        - goals_index: 所有可能被采样为目标的 (traj_name, time)
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                cached = txn.get(image_path.encode())
            # 缓存中存储的是预 resize 的 numpy 数组 [3, H, W]，直接反序列化返回
            img_np = pickle.loads(cached)
            return torch.from_numpy(img_np).float()
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        # 提取当前时间之后一段时间窗口内的 yaw 与 position
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        # 将全局坐标转换到以起点为原点、起始朝向为 x 轴的局部坐标系
        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        # 根据是否学习角度，拼接 (dx, dy) 或 (dx, dy, d_yaw)
        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]
        
        # 若需要，将位移归一化到 metric_waypoint_spacing 的尺度下
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        # 确保为 float32，避免 traj_data 中混合类型导致 object dtype
        return np.asarray(actions, dtype=np.float32), np.asarray(goal_pos, dtype=np.float32)
    
    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        获取第 i 条样本。

        Args:
            i (int): 样本索引

        Returns:
            Tuple[Tensor]:
                obs_image: shape [3 * (context_size+1), H, W] 的上下文观测图像拼接
                goal_image: shape [3, H, W] 的目标图像
                actions_torch: shape [len_traj_pred, num_action_params] 的局部轨迹
                distance: 观测到目标的“时间步距离”标签
                goal_pos: 目标在局部坐标系中的位置
                which_dataset: 当前样本属于哪个 dataset（整数索引）
                action_mask: 是否对这条样本计算动作损失的掩码（0 或 1）
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # Load images
        context = []
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        # 将多帧图像在通道维度拼接，得到 [3 * (context_size+1), H, W]
        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])

        # Load goal image
        # 目标帧只取一张图像
        goal_image = self._load_image(f_goal, goal_time)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        # 基于当前轨迹数据与目标时间，计算局部动作序列与目标位置
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
        
        # Compute distances
        # goal_is_negative=True 时，将距离标签置为最大类别（“非常远/不相关”）
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        
        actions_torch = torch.from_numpy(np.asarray(actions, dtype=np.float32))
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
        
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)
        )

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
        )
