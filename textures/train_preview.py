from tkinter import Image

import cv2
import numpy as np

import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import train
from train import MultiPlaneRenderer
import planes.plane as plane
from planes.plane import PlaneRectangle

import argparse


def generate_dummy_rays(width, height):
    import torch

    # 创建像素坐标网格
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    
    # 构造 rays_o (假设相机在 (0, 0, -2))
    rays_o = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)], dim=-1).view(-1, 3)
    
    # 构造 rays_rot (theta, phi)
    # 假设相机正对平面，rays_rot 设为常数 (即所有射线平行，垂直投射)
    # theta=0, phi=0 表示朝向正 z 轴方向
    rays_rot = torch.zeros((width * height, 2))
    
    return rays_o[:, :2], rays_rot


def get_camera_rays(eye, width, height, fov_deg):
    """
    生成相机视线信息
    :param eye: list/tensor [x, y, z] 相机位置
    :param width: 图像宽度
    :param height: 图像高度
    :param fov_deg: 垂直视野角度 (degrees)
    :return: origin (H, W, 2), rotate (H, W, 2)
    """
    eye = torch.tensor(eye, dtype=torch.float32)
    device = eye.device
    
    # 1. 生成像素坐标网格 (归一化到 [-1, 1])
    y, x = torch.meshgrid(torch.linspace(1, -1, height), 
                          torch.linspace(-1, 1, width), indexing='ij')
    
    # 2. 计算焦距 (假设相机位于原点前方，屏幕在 z=0 处)
    # 根据 FOV 计算: tan(fov/2) = (height/2) / focal_length
    aspect = width / height
    fov_rad = torch.deg2rad(torch.tensor(fov_deg))
    focal_length = 1.0 / torch.tan(fov_rad / 2.0)
    
    # 3. 计算射线方向 (相机朝 -z 方向)
    # 方向向量: (x * aspect / focal, y / focal, -1)
    dir_x = x * aspect / focal_length
    dir_y = y / focal_length
    dir_z = -torch.ones_like(dir_x)
    
    # 4. 计算 Origin (射线与 z=0 平面的交点)
    # 射线方程: P = Eye + t * Dir
    # z = Eye_z + t * Dir_z = 0  => t = -Eye_z / Dir_z = Eye_z
    t = eye[2]
    origin_x = eye[0] + t * dir_x
    origin_y = eye[1] + t * dir_y
    
    origin = torch.stack([origin_x, origin_y], dim=-1)
    
    # 5. 计算 Rotate (Theta: 与Z轴夹角, Phi: 在XY平面的方位角)
    # 归一化方向向量
    dirs = torch.stack([dir_x, dir_y, dir_z], dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    
    # Theta (0 到 PI): acos(dir_z)
    theta = torch.acos(dirs[..., 2])
    # Phi (-PI 到 PI): atan2(dir_y, dir_x)
    phi = torch.atan2(dirs[..., 1], dirs[..., 0])
    
    rotate = torch.stack([theta, phi], dim=-1)
    
    return origin, rotate


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="多平面纹理渲染预览")
    argparser.add_argument("-j", "--planes_json", type=str, help="包含平面信息的JSON文件路径，纹理文件应与JSON在同一目录下")
    args = argparser.parse_args()

    import torch

    # 1. 实例化渲染器
    renderer = train.load_planes(args.planes_json, noise_level=0.1)

    # 2. 设置一个测试相机视角
    #rays_o, rays_rot = generate_dummy_rays(width=800, height=600) 
    rays_o, rays_rot = get_camera_rays([0, 0, 5], width=800, height=600, fov_deg=60.0) # (H, W, 2)

    # 3. 关闭梯度计算（预览不需要记录任何反向传播的信息）
    with torch.no_grad():
        # 前向传播
        rendered_image, weights = renderer(rays_o.reshape(-1, 2), rays_rot.reshape(-1, 2)) # (N_rays, 3), (N_rays, 1)

    # 4. 将输出转换为 OpenCV 可读的格式
    # rendered_image 的形状是 (N_rays, 3)，你需要把它 reshape 回 (H, W, 3)
    output_img = rendered_image.cpu().numpy().reshape(600, 800, 3)
    output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)

    # 5. 显示结果
    cv2.imshow("Preview", output_img)
    cv2.waitKey(0)