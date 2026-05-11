import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Fv

from typing import List
import json
import numpy as np
from PIL import Image
import math

import lpips

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import planes.plane as plane
from planes.plane import PlaneRectangle

class MultiPlaneRenderer(nn.Module):
    '''
    多平面纹理渲染器
    Attributes:
        plane_normals: (num_planes, 3) 每个平面的法线
        plane_centers: (num_planes, 3) 每个平面的中心点坐标
        plane_lengths: (num_planes, 3) 每个平面的长度方向和尺寸
        plane_widths: (num_planes, 3) 每个平面的宽度方向和尺寸
        textures: (num_planes, 4, tex_width, tex_height) 可学习的纹理贴图
    '''

    def __init__(self, planes: List[PlaneRectangle[torch.Tensor]]):
        """
        初始化多平面渲染器
        Parameters:
            planes: List[PlaneRectangle]
                每个平面的几何信息和初始纹理（Tensor[4(RGBA), tex_width, tex_height]）

        """
        super().__init__()
        # num_planes = len(planes)
        # tex_width, tex_height = planes[0].texture.shape[2], planes[0].texture.shape[1] # 假设所有平面纹理尺寸相同

        # 定义平面的几何信息
        self.register_buffer('plane_normals', torch.stack([torch.from_numpy(plane.normal).float() for plane in planes])) # (N_planes, 3)
        self.register_buffer('plane_centers', torch.stack([torch.from_numpy(plane.center).float() for plane in planes])) # (N_planes, 3)
        self.register_buffer('plane_lengths', torch.stack([torch.from_numpy(plane.length_vec).float() for plane in planes])) # (N_planes, 3)
        self.register_buffer('plane_widths', torch.stack([torch.from_numpy(plane.width_vec).float() for plane in planes])) # (N_planes, 3)

        initial_textures = torch.stack([plane.texture for plane in planes], dim=0) # (N_planes, 4, tex_width, tex_height)

        self.textures = nn.Parameter(initial_textures) # nn.Parameter 使纹理成为可学习的参数
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """ 
        加载模型 checkpoint
        Parameters:
            checkpoint_path: 模型 checkpoint 的文件路径
        """
        try:
            checkpoint = torch.load(checkpoint_path)
            self.textures.data.copy_(checkpoint['textures'])
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """ 
        保存模型 checkpoint
        Parameters:
            checkpoint_path: 模型 checkpoint 的文件路径
        """
        try:
            checkpoint = {
                'textures': self.textures.data.cpu()
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint to {checkpoint_path}: {e}")

    def forward(self, rays_o: torch.Tensor, rays_rot: torch.Tensor):
        """ 
        前向渲染逻辑 
        Parameters:
            rays_o: (N_rays, 2) 射线原点（x, y）
            rays_rot: (N_rays, 2) 射线方向（theta, phi）
        """

        # 0. 准备工作：格式化要用到的张量

        # 从输入的 rays_rot 中计算单位射线方向向量 rays_d
        theta = rays_rot[:, 0]
        phi = rays_rot[:, 1]
        rays_d = torch.stack([
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta),
            - torch.cos(theta) * torch.cos(phi)
        ], dim=-1) # (N_rays, 3)

        # 为 rays_o 添加 z=0 分量
        rays_o = F.pad(rays_o, (0, 1), mode='constant', value=0) # (N_rays, 3)，z=0

        # 调整相关张量到(planes, rays, 3)的形状以便后续计算
        plane_c = self.plane_centers.unsqueeze(1) # (N_planes, 1, 3)
        plane_n = self.plane_normals.unsqueeze(1) # (N_planes, 1, 3)
        plane_l = self.plane_lengths.unsqueeze(1) # (N_planes, 1, 3)
        plane_w = self.plane_widths.unsqueeze(1)  # (N_planes, 1, 3)
        rays_o = rays_o.unsqueeze(0)              # (1, N_rays, 3)
        rays_d = rays_d.unsqueeze(0)              # (1, N_rays, 3)
        textures = self.textures                  # (N_planes, 4, tex_w, tex_h)


        # 1. 射线与所有平面求交，得到交点坐标和深度 t (N_planes, N_rays)

        # t = (P - O) · N / (D · N)
        numerator = torch.sum((plane_c - rays_o) * plane_n, dim=-1)
        denominator = torch.sum(rays_d * plane_n, dim=-1)
        t = numerator / (denominator + 1e-6) # 加 1e-6 防止除以零

            
        # 2. 根据交点计算 UV 坐标
        p = rays_o + t.unsqueeze(-1) * rays_d # (N_planes, N_rays, 3)
        u = torch.sum((p - plane_c) * plane_l, dim=-1) / torch.sum(plane_l * plane_l, dim=-1) # (N_planes, N_rays)
        v = torch.sum((p - plane_c) * plane_w, dim=-1) / torch.sum(plane_w * plane_w, dim=-1) # (N_planes, N_rays)
        # 从[-0.5, 0.5] 归一化到 [-1, 1] 并翻转v轴，适配 grid_sample 的输入要求
        uv = torch.stack([v, u], dim=-1) * 2 # (N_planes, N_rays, 2) -> [-1, 1] 范围内的 UV 坐标
        

        # 3. 采样纹理
        # 使用 F.grid_sample 来采样纹理，它是可微的双线性插值
        sampled_rgba = F.grid_sample(
            textures,               # (N_planes, 4, tex_w, tex_h)
            uv.unsqueeze(1),        # (N_planes, N_rays, 2) -> (N_planes, 1, N_rays, 2) 以适配 grid_sample 的输入要求
            mode='bilinear',        # 双线性插值
            padding_mode='zeros',   # 超出平面的部分设为透明
            align_corners=False     # 是否将 [-1, 1] 映射到像素中心
        ) # (N_planes, 4, 1, N_rays)
        sampled_rgba = sampled_rgba.squeeze(2).permute(0, 2, 1) # (N_planes, N_rays, 4)


        # 3. 深度排序
        t_sorted, indices = torch.sort(t, dim=0, descending=False) # (N_planes, N_rays)
        sampled_rgba_sorted = torch.gather(sampled_rgba, 0, indices.unsqueeze(-1).expand(-1, -1, 4)) # (N_planes, N_rays, 4)


        # 5. Alpha Blending (从前向后累加)
        alpha = sampled_rgba_sorted[..., 3:4] # (N_planes, N_rays, 1)
        rgb = sampled_rgba_sorted[..., :3]  # (N_planes, N_rays, 3)
        
        one_minus_alpha = 1.0 - alpha
        input_for_cumprod = torch.cat([torch.ones_like(alpha[:1]), one_minus_alpha[:-1]], dim=0) # (N_planes, N_rays, 1)，第一层的 transmittance 是 1

        transmittance = torch.cumprod(input_for_cumprod, dim=0) # (N_planes, N_rays, 1)
        weights = transmittance * alpha # (N_planes, N_rays, 1)
        rendered_color = torch.sum(weights * rgb, dim=0) # (N_rays, 3)


        # final. 返回渲染结果和权重（权重用于深度学习）
        return rendered_color, weights, rays_d.squeeze(0) # (N_rays, 3), (N_planes, N_rays, 1), (N_rays, 3)
    
    def compute_loss(self, rendered_color : torch.Tensor, gt_color: torch.Tensor, weights: torch.Tensor,
                 plane_normals: torch.Tensor, view_dir: torch.Tensor,
                 mse_weight: float = 1.0, lpips_weight: float = 0.1, tv_weight: float = 0.01,
                 lpips_fn=None, img_w: int = None, img_h: int = None) -> torch.Tensor:
        """
        计算损失函数
        Parameters:
            rendered_color: (N_rays, 3) 渲染得到的颜色
            gt_color: (N_rays, 3) 真实的颜色
            weights: (N_planes, N_rays, 1) 每个平面对每条射线的贡献度
            plane_normals: (N_planes, 3) 每个平面的法线
            view_dir: (N_rays, 3) 每个视线的观察方向
            mse_weight: MSE损失的权重
            lpips_weight: LPIPS损失的权重
            lpips_fn: LPIPS 模型（None 时不计算 LPIPS）
            img_w: 图像宽度，LPIPS 需要 reshape 回 2D
            img_h: 图像高度，LPIPS 需要 reshape 回 2D
        """

        # 格式化张量
        plane_normals = plane_normals.unsqueeze(1) # (N_planes, 1, 3)
        view_dir = view_dir.unsqueeze(0)           # (1, N_rays, 3)

        # 基础L2误差
        loss_l2 = (rendered_color - gt_color) ** 2 # (N_rays, 3)
        loss_l2 = loss_l2.unsqueeze(0)             # (1, N_rays, 3)

        # 平面视角加权
        cos = (plane_normals * view_dir).sum(dim=-1).abs() # (N_planes, N_rays)
        loss = loss_l2 * cos.unsqueeze(-1) # (N_planes, N_rays, 3)

        loss = loss.mean() * mse_weight

        # LPIPS 感知损失
        if lpips_fn is not None and img_w is not None and img_h is not None:
            # 从 (W*H, 3) 重塑回 (1, 3, H, W)
            rendered_img = rendered_color.reshape(img_w, img_h, 3).permute(2, 1, 0).unsqueeze(0)  # (1, 3, H, W)
            gt_img = gt_color.reshape(img_w, img_h, 3).permute(2, 1, 0).unsqueeze(0)             # (1, 3, H, W)

            # LPIPS 期望输入范围 [-1, 1]
            rendered_img = rendered_img * 2 - 1
            gt_img = gt_img * 2 - 1

            loss_lpips = lpips_fn(rendered_img, gt_img).mean()
            loss = loss + loss_lpips * lpips_weight

        # TV 正则化
        if tv_weight > 0:
            # 计算纹理的总变差（Total Variation）正则化，鼓励纹理平滑
            # textures: (N_planes, 4, tex_w, tex_h)
            # 计算水平方向的梯度差 (相邻列)
            diff_h = torch.abs(self.textures[:, :, :, 1:] - self.textures[:, :, :, :-1])
            # 计算垂直方向的梯度差 (相邻行)
            diff_w = torch.abs(self.textures[:, :, 1:, :] - self.textures[:, :, :-1, :])
            tv_loss = diff_h.mean() + diff_w.mean()
            loss = loss + tv_loss * tv_weight

        return loss


def load_planes(path: str, noise_level: float = 0.1, alpha_level: float = 0.5, n: int = -1) -> MultiPlaneRenderer:
    def texture_loader(tex_path: str) -> torch.Tensor:
        # 加载纹理图像并转换为 Tensor，范围归一化到 [0, 1]
        img = Image.open(tex_path).convert('L') # 单通道灰度图

        width, height = img.size

        alpha = Fv.to_tensor(img) # (1, H, W)，范围 [0, 1]
        # 调换w和h的维度以适配后续处理
        alpha = alpha.permute(0, 2, 1) # (1, W, H)
        alpha *= alpha_level # alpha 缩放至指定水平，增加一些透明度余量

        rgb = torch.clamp(0.5 + torch.randn(3, width, height) * noise_level, 0, 1) # (3, W, H)
        rgba = torch.cat([rgb, alpha], dim=0) # (4, W, H)
        return rgba

    planes = plane.read_from_json(path, texture_loader)

    if n < 0: n = len(planes)
    n = min(n, len(planes))

    return MultiPlaneRenderer(planes[:n])


def run():
    import argparse
    from pathlib import Path
    import re
    import random

    parser = argparse.ArgumentParser(description="训练多平面纹理渲染器")

    parser.add_argument("-j", "--planes_json", type=str, help="[必选]包含平面信息的JSON文件路径，纹理文件应与JSON在同一目录下")
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None, help="模型的checkpoint路径，用于继续训练")
    parser.add_argument("-n", "--noise_level", type=float, default=0.1, help="[仅当未使用-c时有效]初始纹理的噪声水平，范围 [0, 1]")
    parser.add_argument("-a", "--alpha_level", type=float, default=0.5, help="[仅当未使用-c时有效]初始纹理的透明度水平，范围 [0, 1]")
    parser.add_argument("-g", "--gt_colors_path", type=str, help="[必选]包含一组斜正交投影下真实光场颜色文件的目录，格式为RGB通道的png图像，文件名格式为'rgba_phi_theta.png'，其中theta和phi是对应视角的旋转参数(角度)")
    parser.add_argument("-gx", "--gt_size_x", type=float, default=1, help="gt颜色图像对应视口的x轴尺寸")
    parser.add_argument("-gy", "--gt_size_y", type=float, default=1, help="gt颜色图像对应视口的y轴尺寸")

    parser.add_argument("-o", "--output_dir", type=str, help="[必选]训练过程中保存中间结果的目录，checkpoint将被保存在这里")
    parser.add_argument("-s", "--save_interval", type=int, default=20, help="每隔多少轮保存一次checkpoint")

    parser.add_argument("-r", "--learning_rate", type=float, default=0.01, help="优化器的学习率")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="训练的总轮数")
    parser.add_argument("-wm", "--mse_weight", type=float, default=1.0, help="MSE损失的权重")
    parser.add_argument("-wl", "--lpips_weight", type=float, default=0.1, help="LPIPS损失的权重")
    parser.add_argument("-wt", "--tv_weight", type=float, default=0.01, help="TV正则化的权重")
    args = parser.parse_args()


    renderer = load_planes(args.planes_json, noise_level=args.noise_level, alpha_level=args.alpha_level)

    if args.checkpoint_path is not None:
        renderer.load_checkpoint(args.checkpoint_path)

    renderer.cuda() # 将模型移动到GPU

    # 初始化 LPIPS 感知损失模型（使用 AlexNet 主干，轻量快速）
    if args.lpips_weight > 0:
        lpips_fn = lpips.LPIPS(net='alex').cuda()
        lpips_fn.eval()
    else:
        lpips_fn = None

    optimizer = torch.optim.Adam(renderer.parameters(), lr=args.learning_rate)
    epochs = args.epochs


    # 从指定目录加载所有的 ground truth 颜色图像，并根据文件名提取对应的视角参数，构建训练数据列表
    gtfile_regex = re.compile(r"rgba_(?P<phi>-?[\d.]+)_(?P<theta>-?[\d.]+)\.png") # 从文件名中提取theta和phi的正则表达式
    ground_truths : List[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = []
    for gtpath in Path(args.gt_colors_path).rglob("*.png"):
        match = gtfile_regex.match(gtpath.name)
        if match:
            image = Image.open(gtpath).convert('RGB')
            width, height = image.size

            x_coords = (torch.linspace(-0.5, 0.5, steps=width+1)[:width] + 0.5/width) * args.gt_size_x # (W,)
            y_coords = (torch.linspace(0.5, -0.5, steps=height+1)[1:] + 0.5/height) * args.gt_size_y # (H,)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij') # (W, H)
            ray_o = torch.stack([grid_x, grid_y], dim=-1) # (W, H, 2)，范围 [-0.5, 0.5]

            theta = math.radians(float(match.group("theta")))
            phi = math.radians(float(match.group("phi")))
            ray_rot = torch.tensor([theta, phi]) # (2,)

            gt_color = Fv.to_tensor(image).permute(2, 1, 0) # (W, H, 3)
            ground_truths.append((ray_o, ray_rot, gt_color, width, height))
        else:
            print(f"警告: 文件 {gtpath} 的命名不符合 'rgba_phi_theta.png' 格式，已跳过")
    print(f"共加载了 {len(ground_truths)} 条训练数据")

    os.makedirs(args.output_dir, exist_ok=True)

    # 训练循环
    just_saved = False
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        gts : List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = random.sample(ground_truths, len(ground_truths))

        for ray_o, ray_rot, gt_color, img_w, img_h in gts:
            optimizer.zero_grad()

            # 传入cuda，并调整形状以适配模型输入要求
            ray_o = ray_o.reshape(-1, 2).cuda() # (N_rays, 2)
            ray_rot = ray_rot.unsqueeze(0).cuda() # (1, 2)
            gt_color = gt_color.reshape(-1, 3).cuda() # (N_rays, 3)

            rendered_color, weights, view_dir = renderer(ray_o, ray_rot) # (N_rays, 3), (N_planes, N_rays, 1), (N_rays, 3)

            loss = renderer.compute_loss(rendered_color, gt_color.reshape(-1, 3), weights, renderer.plane_normals, view_dir,
                                args.mse_weight, args.lpips_weight, args.tv_weight,
                                lpips_fn=lpips_fn, img_w=img_w, img_h=img_h)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                renderer.textures.data = torch.clamp(renderer.textures.data, 0.0, 1.0)

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(ground_truths):.4f}")

        just_saved = False
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, "latest.pth")
            renderer.save_checkpoint(checkpoint_path)
            just_saved = True
        if total_loss < best_loss:
            best_loss = total_loss
            checkpoint_path = os.path.join(args.output_dir, "best.pth")
            renderer.save_checkpoint(checkpoint_path)

    # todo: 可以考虑摘出来一部分作为验证集，以后效果不好就这样做试试
    if not just_saved:
        checkpoint_path = os.path.join(args.output_dir, "latest.pth")
        renderer.save_checkpoint(checkpoint_path)



if __name__ == "__main__":
    run()
