#!/usr/bin/env python3
"""
CARLA 简易车辆控制器 + 数据录制 (24FPS 完美步数对齐 纯净版)
- W/S: 前进/后退
- A/D: 左转/右转
- R: 开始/停止录制
- M: 切换地图
- SPACE: 刹车
"""

import carla
import pygame
import argparse
import os
import pickle
import time
import numpy as np
import queue
from PIL import Image

from carla_camera_utils import (
    DEFAULT_CAMERA_TYPE,
    DEFAULT_FISHEYE_FOV,
    DEFAULT_FISHEYE_MODEL,
    DEFAULT_FOV_FADE_SIZE,
    DEFAULT_FOV_MASK,
    DEFAULT_RECORD_IMAGE_SIZE,
    DEFAULT_RGB_FOV,
    add_camera_arguments,
    build_camera_blueprint,
    camera_metadata,
    make_camera_transform,
    write_camera_metadata,
)

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class SimpleVehicleController:
    def __init__(
        self,
        host='localhost',
        port=2000,
        map_name=None,
        output_dir='./carla_fisheye_dataset',
        camera_type=DEFAULT_CAMERA_TYPE,
        record_image_size=DEFAULT_RECORD_IMAGE_SIZE,
        rgb_fov=DEFAULT_RGB_FOV,
        fisheye_fov=DEFAULT_FISHEYE_FOV,
        fisheye_model=DEFAULT_FISHEYE_MODEL,
        fov_mask=DEFAULT_FOV_MASK,
        fov_fade_size=DEFAULT_FOV_FADE_SIZE,
        spawn_index=-1,
        spawn_retries=3,
        spawn_retry_delay=0.5,
        vehicle_filter='vehicle.mini.cooper_s',
        record_role='training',
    ):
        self.host = host
        self.port = port
        self.map_name = map_name
        self.output_dir = output_dir
        self.camera_type = camera_type
        self.rgb_fov = rgb_fov
        self.fisheye_fov = fisheye_fov
        self.fisheye_model = fisheye_model
        self.fov_mask = fov_mask
        self.fov_fade_size = fov_fade_size
        self.spawn_index = spawn_index
        self.spawn_retries = spawn_retries
        self.spawn_retry_delay = spawn_retry_delay
        self.vehicle_filter = vehicle_filter
        self.record_role = record_role
        
        # CARLA对象
        self.client = None
        self.world = None
        self.tm = None             
        self.vehicle = None
        self.camera = None         
        self.front_camera = None   
        self.front_camera_meta = None
        
        # --- 同步模式队列 ---
        self.display_queue = queue.Queue()
        self.front_queue = queue.Queue()
        
        # Pygame
        self.display = None
        self.clock = None
        self.font = None
        
        # 控制参数
        self.target_speed = 4.0  
        self.max_throttle = 1.0  
        self.steer_amount = 0.3  

        # --- 新增：PI 控制与方向盘平滑参数 ---
        self.speed_integral = 0.0    # 速度误差积分
        self.current_steer = 0.0     # 当前方向盘真实角度
        self.steer_speed = 0.15      # 方向盘打轮平滑系数
        
        # 图像缓存
        self.current_image = None
        self.front_image = None  
        
        # --- 录制与对齐核心参数 ---
        self.is_recording = False
        self.trajectory_count = 0
        self.current_trajectory = {
            'images': [],
            'positions': [],
            'yaws': []
        }
        self.record_image_size = tuple(record_image_size)
        
        # 核心：引入 Tick 计数器
        self.tick_counter = 0
        self.record_tick_interval = 3   # 24 FPS 下，每 3 ticks 录制一次 = 8 FPS (0.125s)
        
    def connect(self):
        print(f"连接 CARLA 服务器 {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        
        if self.map_name:
            print(f"加载地图: {self.map_name}")
            self.world = self.client.load_world(self.map_name)
        else:
            self.world = self.client.get_world()
            
        print(f"当前地图: {self.world.get_map().name}")
        self._enable_synchronous_mode()

    def _enable_synchronous_mode(self):
        """开启服务器同步模式，固定帧率为 24 FPS"""
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 24.0  
        self.world.apply_settings(settings)
        print("已开启 CARLA 同步模式 (锁定 24 FPS)")
            
    def spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self.vehicle_filter)[0]
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, carla.Transform())
        else:
            base_indices = list(range(len(spawn_points)))
            if self.spawn_index >= 0:
                preferred = self.spawn_index % len(spawn_points)
                base_indices = [preferred] + [i for i in base_indices if i != preferred]
            for attempt in range(max(1, int(self.spawn_retries))):
                if attempt == 0:
                    candidate_indices = base_indices
                else:
                    candidate_indices = list(np.random.permutation(base_indices))
                for idx in candidate_indices:
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[idx])
                    if self.vehicle is not None:
                        print(f"车辆已生成: {vehicle_bp.id} (spawn index {idx}, attempt {attempt + 1})")
                        return self.vehicle
                time.sleep(max(0.0, float(self.spawn_retry_delay)))
        if self.vehicle is None:
            raise RuntimeError("车辆生成失败：所有候选 spawn point 均不可用。")
        print(f"车辆已生成: {vehicle_bp.id}")
        return self.vehicle
        
    def setup_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1080')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(
            carla.Location(x=-5.0, y=0.0, z=3.0),
            carla.Rotation(pitch=-15, yaw=0, roll=0)
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(self.display_queue.put)
        
    def setup_front_camera(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = build_camera_blueprint(
            blueprint_library,
            camera_type=self.camera_type,
            image_size=self.record_image_size,
            rgb_fov=self.rgb_fov,
            fisheye_fov=self.fisheye_fov,
            fisheye_model=self.fisheye_model,
            fov_mask=self.fov_mask,
            fov_fade_size=self.fov_fade_size,
        )
        camera_transform = make_camera_transform(carla)
        self.front_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.front_camera.listen(self.front_queue.put)
        self.front_camera_meta = camera_metadata(
            camera_bp,
            self.camera_type,
            self.record_image_size,
            camera_transform,
            extra={
                "role": "record_front",
                "sim_fps": 24,
                "record_tick_interval": int(self.record_tick_interval),
                "record_rate_hz": 24.0 / float(self.record_tick_interval),
            },
        )
        print(f"前视录制相机已设置: {camera_bp.id}, {self.record_image_size[0]}x{self.record_image_size[1]}")
        
    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.current_image = array[:, :, :3][:, :, ::-1]
        
    def _process_front_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.front_image = array[:, :, :3][:, :, ::-1]
        
    def get_vehicle_state(self):
        transform = self.vehicle.get_transform()
        return [transform.location.x, transform.location.y], float(np.radians(transform.rotation.yaw))
        
    def record_frame(self):
        if self.front_image is None:
            return
            
        img = Image.fromarray(self.front_image)
        position, yaw = self.get_vehicle_state()
        
        self.current_trajectory['images'].append(np.array(img))
        self.current_trajectory['positions'].append(position)
        self.current_trajectory['yaws'].append(yaw)
        
    def save_trajectory(self):
        if len(self.current_trajectory['images']) < 10:
            print("轨迹太短（<10帧），不保存")
            return
            
        os.makedirs(self.output_dir, exist_ok=True)
        existing = [d for d in os.listdir(self.output_dir) if d.startswith('trajectory_')]
        traj_id = max([int(d.split('_')[1]) for d in existing]) + 1 if existing else 0
        traj_dir = os.path.join(self.output_dir, f"trajectory_{traj_id:06d}")
        os.makedirs(traj_dir, exist_ok=True)
        
        for i, img in enumerate(self.current_trajectory['images']):
            Image.fromarray(img).save(os.path.join(traj_dir, f"{i}.jpg"), 'JPEG', quality=95)
            
        with open(os.path.join(traj_dir, 'traj_data.pkl'), 'wb') as f:
            pickle.dump({
                # 强制转换为 NumPy 数组，并指定深度学习常用的 float32 精度
                'position': np.array(self.current_trajectory['positions'], dtype=np.float32),
                'yaw': np.array(self.current_trajectory['yaws'], dtype=np.float32)
            }, f)
        if self.front_camera_meta is not None:
            meta = dict(self.front_camera_meta)
            meta.update({
                "map": self.world.get_map().name if self.world is not None else None,
                "record_role": self.record_role,
                "trajectory_frames": len(self.current_trajectory['images']),
                "output_format": "nomad_trajectory",
            })
            write_camera_metadata(traj_dir, meta)
            
        print(f"已保存轨迹: {traj_dir} ({len(self.current_trajectory['images'])} 帧)")
        self.current_trajectory = {'images': [], 'positions': [], 'yaws': []}
        self.trajectory_count += 1
        
    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.save_trajectory()
            print("录制停止")
        else:
            self.is_recording = True
            self.current_trajectory = {'images': [], 'positions': [], 'yaws': []}
            self.tick_counter = 0 
            print(f"录制开始... (严格 {24//self.record_tick_interval} FPS)")
        
    def setup_pygame(self):
        pygame.init()
        pygame.display.set_caption('CARLA 同步录制 (纯净对齐版)')
        self.display = pygame.display.set_mode((1080, 1080))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
    def get_speed(self):
        velocity = self.vehicle.get_velocity()
        return (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        
    def control_vehicle(self, keys):
        throttle = brake = 0.0
        reverse = False
        current_speed = self.get_speed()
        
        # ==========================================
        # 1. 纵向控制：PI 控制器 (消除稳态误差，死死咬住 4m/s)
        # ==========================================
        speed_error = self.target_speed - current_speed
        
        # 累加误差 (积分项)，乘以 dt (1/24 秒)
        self.speed_integral += speed_error * (1.0 / 24.0)
        # 积分限幅 (Anti-windup)，防止积分爆炸导致撒开油门后还往前窜
        self.speed_integral = np.clip(self.speed_integral, -2.0, 2.0)
        
        if keys[pygame.K_w]:
            if speed_error > 0:
                # 基础油门 0.3 + 比例P(0.5) + 积分I(0.15)
                throttle = np.clip(0.3 + 0.5 * speed_error + 0.15 * self.speed_integral, 0.0, 1.0)
            else:
                if speed_error < -0.5:
                    brake = np.clip(-0.5 * speed_error, 0.0, 1.0)
                    # 刹车时清空积分，防止对抗
                    self.speed_integral = 0.0 
                    
        elif keys[pygame.K_s]:
            reverse = True
            if speed_error > 0:
                throttle = np.clip(0.3 + 0.5 * speed_error + 0.15 * self.speed_integral, 0.0, 1.0)
            else:
                if speed_error < -0.5:
                    brake = np.clip(-0.5 * speed_error, 0.0, 1.0)
                    self.speed_integral = 0.0
        else:
            brake = 0.1 # 松开按键时自然减速
            self.speed_integral = 0.0 # 不踩油门时清空积分
            
        # ==========================================
        # 2. 横向控制：一阶低通滤波 (带有回正手感)
        # ==========================================
        target_steer = 0.0
        if keys[pygame.K_a]:
            target_steer = -self.steer_amount
        elif keys[pygame.K_d]:
            target_steer = self.steer_amount
            
        # 方向盘平滑逼近目标角度
        self.current_steer += (target_steer - self.current_steer) * self.steer_speed
            
        # ==========================================
        # 3. 紧急制动与执行
        # ==========================================
        if keys[pygame.K_SPACE]:
            throttle, brake = 0.0, 1.0
            self.speed_integral = 0.0
            
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=float(throttle), 
            steer=float(self.current_steer), 
            brake=float(brake), 
            reverse=reverse
        ))
        
        return throttle, self.current_steer, brake, reverse
        
    def render(self, throttle, steer, brake, reverse):
        if self.current_image is not None:
            surface = pygame.surfarray.make_surface(self.current_image.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
            
        info_lines = [
            f"Speed: {self.get_speed():.2f} m/s",
            f"Throttle: {throttle:.2f}  Brake: {brake:.2f}",
            f"Steer: {steer:.2f}  Reverse: {reverse}",
            "",
            f"Recording: {'ON (8 FPS STRICT)' if self.is_recording else 'OFF'}",
            f"Camera: {self.camera_type}",
            f"Frames: {len(self.current_trajectory['images'])}",
            f"Saved trajectories: {self.trajectory_count}",
            "",
            "W/S: Forward/Back  A/D: Turn",
            "SPACE: Brake  R: Record",
            "M: Change map  ESC: Exit"
        ]
        
        for i, line in enumerate(info_lines):
            color = RED if "Recording: ON" in line else WHITE
            self.display.blit(self.font.render(line, True, color, BLACK), (10, 10 + i * 30))
        pygame.display.flip()
        
    def run(self):
        print("\n=== 控制说明 ===")
        print("已启用方案 A: 仿真引擎锁定 24 FPS, 每 3 步采样一次 (严格 8 FPS)")
        print("================\n")
        
        running = True
        map_index = 0
        available_maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']
        # available_maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']
        
        try:
            while running:
                self.clock.tick(24) 
                
                keys = pygame.key.get_pressed()
                throttle, steer, brake, reverse = self.control_vehicle(keys)

                self.world.tick()
                
                try:
                    display_data = self.display_queue.get(True, 2.0)
                    front_data = self.front_queue.get(True, 2.0)
                    self._process_image(display_data)
                    self._process_front_image(front_data)
                except queue.Empty:
                    print("警告: 图像队列获取超时")
                    continue
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            self.toggle_recording()
                        elif event.key == pygame.K_m:
                            if self.is_recording: self.toggle_recording()
                            map_index = (map_index + 1) % len(available_maps)
                            print(f"\n切换到地图: {available_maps[map_index]}")
                            self.cleanup(switch_map=True)
                            self.world = self.client.load_world(available_maps[map_index])
                            self._enable_synchronous_mode() 
                            self.spawn_vehicle()
                            self.setup_camera()
                            self.setup_front_camera()
                
                if self.is_recording:
                    if self.tick_counter % self.record_tick_interval == 0:
                        self.record_frame()
                    self.tick_counter += 1
                
                self.render(throttle, steer, brake, reverse)
                
        except KeyboardInterrupt:
            pass
        finally:
            if self.is_recording: self.save_trajectory()
            
    def cleanup(self, switch_map=False):
        if self.front_camera: self.front_camera.destroy(); self.front_camera = None
        self.front_camera_meta = None
        if self.camera: self.camera.destroy(); self.camera = None
        if self.vehicle: self.vehicle.destroy(); self.vehicle = None
        
        with self.display_queue.mutex: self.display_queue.queue.clear()
        with self.front_queue.mutex: self.front_queue.queue.clear()
            
        if self.world and not switch_map:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            if self.tm: self.tm.set_synchronous_mode(False)
            
    def shutdown(self):
        self.cleanup()
        pygame.quit()
        print("已退出")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA 数据录制 (纯净版)')
    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'carla_fisheye_dataset')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--map', default=None)
    parser.add_argument('--speed', type=float, default=4.0)
    parser.add_argument('--output_dir', default=default_output_dir)
    parser.add_argument('--spawn-index', type=int, default=-1)
    parser.add_argument('--spawn-retries', type=int, default=3)
    parser.add_argument('--spawn-retry-delay', type=float, default=0.5)
    parser.add_argument('--vehicle-filter', default='vehicle.mini.cooper_s')
    parser.add_argument(
        '--record-role',
        choices=['training', 'topomap'],
        default='training',
        help='Metadata role for the saved trajectory.',
    )
    add_camera_arguments(parser)
    args = parser.parse_args()
    
    controller = SimpleVehicleController(
        host=args.host,
        port=args.port,
        map_name=args.map,
        output_dir=args.output_dir,
        camera_type=args.camera_type,
        record_image_size=(args.record_width, args.record_height),
        rgb_fov=args.rgb_fov,
        fisheye_fov=args.fisheye_fov,
        fisheye_model=args.fisheye_model,
        fov_mask=args.fov_mask,
        fov_fade_size=args.fov_fade_size,
        spawn_index=args.spawn_index,
        spawn_retries=args.spawn_retries,
        spawn_retry_delay=args.spawn_retry_delay,
        vehicle_filter=args.vehicle_filter,
        record_role=args.record_role,
    )
    controller.target_speed = args.speed
    
    try:
        controller.connect()
        controller.spawn_vehicle()
        controller.setup_camera()
        controller.setup_front_camera()
        controller.setup_pygame()
        controller.run()
    except Exception as e:
        print(f"错误: {e}")
        raise
    finally:
        controller.shutdown()
