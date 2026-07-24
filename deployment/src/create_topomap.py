import argparse
import csv
import json
import math
import os
import shutil
import socket
import threading
import time

# ROS
import rospy
import numpy as np
from gazebo_msgs.msg import ModelStates
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy

IMAGE_TOPIC = os.environ.get("IMAGE_TOPIC", "/usb_cam/image_raw")
TOPOMAP_IMAGES_DIR = os.environ.get("TOPOMAP_IMAGES_DIR", "../topomaps/images")
obs_img = None
obs_lock = threading.Lock()
robot_pose = None
robot_pose_lock = threading.Lock()


def msg_to_pil(msg: Image) -> PILImage.Image:
    """Convert common ROS image encodings to PIL RGB without torch imports."""
    encoding = msg.encoding.lower()
    if encoding in {"rgb8", "bgr8"}:
        channels = 3
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding == "mono8":
        channels = 1
    else:
        raise ValueError(f"Unsupported topomap image encoding: {msg.encoding}")

    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    rows = rows[:, : msg.width * channels]
    img = rows.reshape(msg.height, msg.width, channels)

    if encoding == "bgr8":
        img = img[:, :, ::-1]
    elif encoding == "rgba8":
        img = img[:, :, :3]
    elif encoding == "bgra8":
        img = img[:, :, [2, 1, 0]]
    elif encoding == "mono8":
        img = np.repeat(img, 3, axis=2)

    return PILImage.fromarray(np.ascontiguousarray(img))


def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def callback_obs(msg: Image):
    global obs_img
    with obs_lock:
        obs_img = msg_to_pil(msg)


def callback_joy(msg: Joy):
    if msg.buttons[0]:
        rospy.signal_shutdown("shutdown")


def yaw_from_quaternion(quat) -> float:
    siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    return math.atan2(siny_cosp, cosy_cosp)


def callback_model_states(msg: ModelStates, robot_model_name: str):
    global robot_pose
    try:
        index = msg.name.index(robot_model_name)
    except ValueError:
        return
    pose = msg.pose[index]
    stamp = rospy.Time.now()
    with robot_pose_lock:
        robot_pose = {
            "stamp": stamp.to_sec(),
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "yaw": float(yaw_from_quaternion(pose.orientation)),
        }


def read_robot_pose():
    with robot_pose_lock:
        return dict(robot_pose) if robot_pose is not None else None


class TopomapMetadataWriter:
    def __init__(self, topomap_dir: str, topomap_name: str, image_topic: str, robot_model_name: str):
        self.topomap_dir = topomap_dir
        self.json_path = os.path.join(topomap_dir, "metadata.json")
        self.csv_path = os.path.join(topomap_dir, "metadata.csv")
        self.topomap_name = topomap_name
        self.image_topic = image_topic
        self.robot_model_name = robot_model_name
        self.nodes = []
        self.cumulative_length = 0.0
        self.last_xy = None

    def append(self, index: int, image_file: str, pose: dict, stamp: float):
        xy = np.array([pose["x"], pose["y"]], dtype=np.float64)
        step_distance = 0.0
        if self.last_xy is not None:
            step_distance = float(np.linalg.norm(xy - self.last_xy))
        self.cumulative_length += step_distance
        self.last_xy = xy

        node = {
            "index": int(index),
            "image_file": image_file,
            "stamp": float(stamp),
            "pose_stamp": float(pose["stamp"]),
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "yaw": float(pose["yaw"]),
            "step_distance": float(step_distance),
            "cumulative_length": float(self.cumulative_length),
        }
        self.nodes.append(node)
        self.write()

    def write(self):
        payload = {
            "topomap": self.topomap_name,
            "image_topic": self.image_topic,
            "robot_model_name": self.robot_model_name,
            "num_nodes": len(self.nodes),
            "reference_path_length": float(self.cumulative_length),
            "nodes": self.nodes,
        }
        tmp_json = self.json_path + ".tmp"
        with open(tmp_json, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp_json, self.json_path)

        tmp_csv = self.csv_path + ".tmp"
        fieldnames = [
            "index",
            "image_file",
            "stamp",
            "pose_stamp",
            "x",
            "y",
            "yaw",
            "step_distance",
            "cumulative_length",
        ]
        with open(tmp_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.nodes)
        os.replace(tmp_csv, self.csv_path)


class TopomapControl:
    def __init__(self, socket_path: str, output_dir: str, start_recording: bool):
        self.socket_path = socket_path
        self.output_dir = output_dir
        self.recording = start_recording
        self.saved_count = 0
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.server_socket = None
        self.thread = None

    def status(self, message: str = ""):
        with self.lock:
            return {
                "ok": True,
                "recording": self.recording,
                "bag_path": self.output_dir,
                "message": message or f"saved {self.saved_count} images",
                "saved_count": self.saved_count,
            }

    def set_saved_count(self, value: int):
        with self.lock:
            self.saved_count = value

    def is_recording(self) -> bool:
        with self.lock:
            return self.recording

    def toggle(self):
        with self.lock:
            self.recording = not self.recording
            if self.recording:
                message = f"Recording topomap: {self.output_dir}"
            else:
                message = f"Paused topomap, saved {self.saved_count} images"
        print(f"[topomap] {message}", flush=True)
        return self.status(message)

    def handle_payload(self, payload):
        command = payload.get("command")
        if command == "toggle":
            return self.toggle()
        if command == "status":
            return self.status()
        return {
            "ok": False,
            "recording": self.is_recording(),
            "bag_path": self.output_dir,
            "message": f"Unknown topomap command: {command}",
        }

    def serve_client(self, connection):
        with connection:
            try:
                reader = connection.makefile("r")
                line = reader.readline()
                payload = json.loads(line) if line else {}
                response = self.handle_payload(payload)
            except Exception as exc:  # noqa: BLE001 - return errors to UI
                response = {
                    "ok": False,
                    "recording": self.is_recording(),
                    "bag_path": self.output_dir,
                    "message": f"Topomap control error: {exc}",
                }
            connection.sendall((json.dumps(response) + "\n").encode())

    def loop(self):
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                connection, _address = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.serve_client(connection)

    def start(self):
        if not self.socket_path:
            return
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self.server_socket.listen(5)
        self.server_socket.settimeout(0.5)
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        print(f"[topomap] Control socket: {self.socket_path}", flush=True)

    def stop(self):
        self.stop_event.set()
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.socket_path:
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass


def main(args: argparse.Namespace):
    global obs_img
    rospy.init_node("CREATE_TOPOMAP", anonymous=False)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    model_states_sub = rospy.Subscriber(
        "/gazebo/model_states",
        ModelStates,
        callback_model_states,
        callback_args=args.robot_model_name,
        queue_size=1,
    )
    subgoals_pub = rospy.Publisher(
        "/subgoals", Image, queue_size=1)
    joy_sub = rospy.Subscriber("joy", Joy, callback_joy)

    topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
    if not os.path.isdir(topomap_name_dir):
        os.makedirs(topomap_name_dir)
    else:
        print(f"{topomap_name_dir} already exists. Removing previous images...")
        remove_files_in_dir(topomap_name_dir)
        

    assert args.dt > 0, "dt must be positive"
    control = TopomapControl(
        socket_path=args.control_socket_path,
        output_dir=topomap_name_dir,
        start_recording=not args.start_paused,
    )
    metadata_writer = TopomapMetadataWriter(
        topomap_name_dir,
        args.dir,
        IMAGE_TOPIC,
        args.robot_model_name,
    )
    control.start()
    print("Registered with master node. Waiting for images...")
    print(f"Waiting for Gazebo model pose: {args.robot_model_name}", flush=True)
    if args.start_paused:
        print("Topomap capture is paused. Press R/REC in teleop to start.", flush=True)
    else:
        print("Topomap capture is recording.", flush=True)
    i = 0
    start_time = float("inf")
    next_save_time = 0.0
    try:
        while not rospy.is_shutdown():
            now = time.time()
            if control.is_recording() and now >= next_save_time:
                with obs_lock:
                    image = obs_img
                    obs_img = None
                pose = read_robot_pose()
                if image is not None and pose is not None:
                    image_file = f"{i}.png"
                    image.save(os.path.join(topomap_name_dir, image_file))
                    metadata_writer.append(i, image_file, pose, now)
                    print(
                        "saved topomap image "
                        f"{i} pose=({pose['x']:.3f}, {pose['y']:.3f}, {pose['yaw']:.3f})",
                        flush=True,
                    )
                    i += 1
                    control.set_saved_count(i)
                    start_time = now
                    next_save_time = now + args.dt
                elif image is not None and pose is None:
                    print(
                        f"Waiting for /gazebo/model_states pose for {args.robot_model_name}...",
                        flush=True,
                    )
                elif start_time != float("inf") and now - start_time > 2 * args.dt:
                    print(f"Topic {IMAGE_TOPIC} not publishing anymore. Shutting down...")
                    rospy.signal_shutdown("shutdown")
            time.sleep(0.05)
    finally:
        control.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Code to generate topomaps from the {IMAGE_TOPIC} topic"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topological map images in ../topomaps/images directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=2.,
        type=float,
        help=f"time between images sampled from the {IMAGE_TOPIC} topic (default: 2.0)",
    )
    parser.add_argument(
        "--control-socket-path",
        default="",
        type=str,
        help="Unix socket path for teleop start/stop control",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="wait for teleop R/REC before saving topomap images",
    )
    parser.add_argument(
        "--robot-model-name",
        default="turtlebot3_waffle_pi",
        type=str,
        help="Gazebo model name used for topomap pose metadata",
    )
    args = parser.parse_args()

    main(args)
