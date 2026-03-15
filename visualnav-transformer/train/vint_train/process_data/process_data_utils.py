import numpy as np
import io
import os
import rosbag
from PIL import Image
import cv2
from typing import Any, Tuple, List, Dict
import torchvision.transforms.functional as TF

IMAGE_SIZE = (160, 120)          # 统一将原始图像 resize 到的尺寸 (宽, 高)
IMAGE_ASPECT_RATIO = 4 / 3       # 训练阶段所有图像都会中心裁剪成 4:3 的宽高比


def process_images(im_list: List, img_process_func) -> List:
    """
    将 ROS topic 接收到的一串图像消息，批量转换为 PIL.Image 列表。

    具体的消息 → PIL 转换逻辑由 img_process_func 决定，不同数据集可以传入不同函数。
    """
    images = []
    for img_msg in im_list:
        img = img_process_func(img_msg)
        images.append(img)
    return images


def process_tartan_img(msg) -> Image:
    """
    将 `sensor_msgs/Image` 消息转换为 tartan_drive 数据集使用的 PIL 图像：
    - 使用 ros_to_numpy 解码并归一化
    - 转为 0~255 的 uint8，再从 (C, H, W) 转回 (H, W, C)
    - 做 RGB→BGR 转换，再交给 PIL 生成图像
    """
    img = ros_to_numpy(msg, output_resolution=IMAGE_SIZE) * 255
    img = img.astype(np.uint8)
    # reverse the axis order to get the image in the right orientation
    img = np.moveaxis(img, 0, -1)
    # convert rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    return img


def process_locobot_img(msg) -> Image:
    """
    将 locobot 数据集上的 `sensor_msgs/Image` 直接 reshape 成 (H, W, C)，并转为 PIL 图像。
    """
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = Image.fromarray(img)
    return pil_image


def process_scand_img(msg) -> Image:
    """
    将 scand 数据集上的 `sensor_msgs/CompressedImage` 解码为 PIL 图像：
    - 先用 PIL 打开压缩字节流
    - 居中裁剪到 4:3 宽高比
    - 再 resize 到统一的 IMAGE_SIZE
    """
    # convert sensor_msgs/CompressedImage to PIL image
    img = Image.open(io.BytesIO(msg.data))
    # center crop image to 4:3 aspect ratio
    w, h = img.size
    img = TF.center_crop(
        img, (h, int(h * IMAGE_ASPECT_RATIO))
    )  # crop to the right ratio
    # resize image to IMAGE_SIZE
    img = img.resize(IMAGE_SIZE)
    return img


############## Add custom image processing functions here #############

def process_sacson_img(msg) -> Image:
    np_arr = np.fromstring(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_np)
    return pil_image


#######################################################################


def process_odom(
    odom_list: List,
    odom_process_func: Any,
    ang_offset: float = 0.0,
) -> Dict[np.ndarray, np.ndarray]:
    """
    将一系列里程计消息（通常是 `nav_msgs/Odometry`）统一转换为：
        {"position": (N, 2), "yaw": (N,)}

    具体的单条消息解码逻辑由 odom_process_func 决定。
    """
    xys = []
    yaws = []
    for odom_msg in odom_list:
        xy, yaw = odom_process_func(odom_msg, ang_offset)
        xys.append(xy)
        yaws.append(yaw)
    return {"position": np.array(xys), "yaw": np.array(yaws)}


def nav_to_xy_yaw(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    将单条 `nav_msgs/Odometry` 解析为 2D 位置与 yaw：
    - 从四元数中取出 yaw（绕 z 轴旋转），再加上外部提供的 ang_offset
    - 仅返回平面位置 [x, y] 与 yaw（单位弧度）
    """

    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    return [position.x, position.y], yaw


############ Add custom odometry processing functions here ############


#######################################################################


def get_images_and_odom(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    odomtopics: List[str] or str,
    img_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,
):
    """
    从 rosbag 中同步采样图像与里程计数据，并转换为统一格式。

    每隔 `1/rate` 秒，将当前缓存到的图像消息与 odom 消息配对一次。

    Args:
        bag (rosbag.Bag): bag 文件句柄
        imtopics (list[str] or str): 候选图像 topic 名（会自动选择首个非空 topic）
        odomtopics (list[str] or str): 候选里程计 topic 名
        img_process_func (Any): 将原始 ROS 图像消息转换为 PIL.Image 的函数
        odom_process_func (Any): 将原始 odom 消息转换为 (xy, yaw) 的函数
        rate (float): 下采样频率（Hz）
        ang_offset (float): 所有 yaw 上额外叠加的偏置（单位弧度）
    Returns:
        img_data (List[PIL.Image]): 同步后的图像序列
        traj_data (Dict[str, np.ndarray]): {"position": (N,2), "yaw": (N,)} 的轨迹数据
    """
    # check if bag has both topics
    odomtopic = None
    imtopic = None
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break
    if not (imtopic and odomtopic):
        # bag doesn't have both topics
        return None, None

    synced_imdata = []
    synced_odomdata = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()

    curr_imdata = None
    curr_odomdata = None

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
                currtime = t.to_sec()

    img_data = process_images(synced_imdata, img_process_func)
    traj_data = process_odom(
        synced_odomdata,
        odom_process_func,
        ang_offset=ang_offset,
    )

    return img_data, traj_data


def is_backwards(
    pos1: np.ndarray, yaw1: float, pos2: np.ndarray, eps: float = 1e-5
) -> bool:
    """
    根据两个连续位置点与前一时刻的航向，判断是否“在往后退”。

    原理：计算 pos2 - pos1 在机器人前向方向 (cos(yaw1), sin(yaw1)) 上的投影，
    若投影过小（< eps），则认为机器人速度在前向上的分量不够大，可视为倒车或停止。
    """
    dx, dy = pos2 - pos1
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps


# cut out non-positive velocity segments of the trajectory
def filter_backwards(
    img_list: List[Image.Image],
    traj_data: Dict[str, np.ndarray],
    start_slack: int = 0,
    end_slack: int = 0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    将一条完整轨迹切分成若干“正向运动”片段（过滤掉倒退或静止段）。

    Args:
        img_list: 与轨迹对齐的图像列表
        traj_data: {"position": (T,2), "yaw": (T,)} 的轨迹数据
        start_slack: 轨迹起始处忽略的若干点
        end_slack: 轨迹末尾忽略的若干点
    Returns:
        cut_trajs: List[Tuple[List[PIL.Image], Dict{"position","yaw"}]]，每个元素是一段子轨迹
    """
    traj_pos = traj_data["position"]
    traj_yaws = traj_data["yaw"]
    cut_trajs = []
    start = True

    def process_pair(traj_pair: list) -> Tuple[List, Dict]:
        new_img_list, new_traj_data = zip(*traj_pair)
        new_traj_data = np.array(new_traj_data)
        new_traj_pos = new_traj_data[:, :2]
        new_traj_yaws = new_traj_data[:, 2]
        return (new_img_list, {"position": new_traj_pos, "yaw": new_traj_yaws})

    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        pos1 = traj_pos[i - 1]
        yaw1 = traj_yaws[i - 1]
        pos2 = traj_pos[i]
        if not is_backwards(pos1, yaw1, pos2):
            if start:
                new_traj_pairs = [
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                ]
                start = False
            elif i == len(traj_pos) - end_slack - 1:
                cut_trajs.append(process_pair(new_traj_pairs))
            else:
                new_traj_pairs.append(
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                )
        elif not start:
            cut_trajs.append(process_pair(new_traj_pairs))
            start = True
    return cut_trajs


def quat_to_yaw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    将四元数 (x, y, z, w) 转换为 yaw（绕 z 轴的旋转角，单位弧度，逆时针为正）。
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw


def ros_to_numpy(
    msg, nchannels=3, empty_value=None, output_resolution=None, aggregate="none"
):
    """
    将 ROS 图像消息转换为 numpy 数组（channels-first 形式）。

    支持：
        - 8bit 编码的 RGB 图像（uint8 → [0,1] 浮点数）
        - 浮点编码的深度/其它多通道信息（float32）
    可以选择：
        - output_resolution: 输出分辨率
        - aggregate: 将多通道合成为 big/little endian 整数值
        - empty_value: 遇到无效值时用 99 分位数填充
    """
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)

    is_rgb = "8" in msg.encoding
    if is_rgb:
        data = np.frombuffer(msg.data, dtype=np.uint8).copy()
    else:
        data = np.frombuffer(msg.data, dtype=np.float32).copy()

    data = data.reshape(msg.height, msg.width, nchannels)

    if empty_value:
        mask = np.isclose(abs(data), empty_value)
        fill_value = np.percentile(data[~mask], 99)
        data[mask] = fill_value

    data = cv2.resize(
        data,
        dsize=(output_resolution[0], output_resolution[1]),
        interpolation=cv2.INTER_AREA,
    )

    if aggregate == "littleendian":
        data = sum([data[:, :, i] * (256**i) for i in range(nchannels)])
    elif aggregate == "bigendian":
        data = sum([data[:, :, -(i + 1)] * (256**i) for i in range(nchannels)])

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    else:
        data = np.moveaxis(data, 2, 0)  # Switch to channels-first

    if is_rgb:
        data = data.astype(np.float32) / (
            255.0 if aggregate == "none" else 255.0**nchannels
        )

    return data
