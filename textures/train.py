import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, num_planes: int, tex_width: int, tex_height: int):
        super().__init__()

        # 定义平面的几何信息
        self.plane_normals = torch.tensor([...], dtype=torch.float32) 
        self.plane_centers = torch.tensor([...], dtype=torch.float32)
        self.plane_lengths = torch.tensor([...], dtype=torch.float32)
        self.plane_widths = torch.tensor([...], dtype=torch.float32)
        
        initial_textures = torch.full((num_planes, 4, tex_width, tex_height), 0.5)
        # todo: 为平面设定空间信息、为alpha通道填充初始数据

        self.textures = nn.Parameter(initial_textures) # nn.Parameter 告诉 PyTorch 这玩意需要被训练！

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
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
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
        uv = torch.stack([u, -v], dim=-1) * 2 # (N_planes, N_rays, 2) -> [-1, 1] 范围内的 UV 坐标


        # 3. 深度排序
        t_sorted, indices = torch.sort(t, dim=0, descending=False) # (N_planes, N_rays)
        uv_sorted = uv.gather(0, indices.unsqueeze(-1).expand(-1, -1, 2)) # (N_planes, N_rays, 2)
        

        # 4. 采样纹理
        # 使用 F.grid_sample 来采样纹理，它是可微的双线性插值
        sampled_rgba_sorted = F.grid_sample(
            textures,               # (N_planes, 4, tex_w, tex_h)
            uv_sorted.unsqueeze(1), # (N_planes, N_rays, 2) -> (N_planes, 1, N_rays, 2) 以适配 grid_sample 的输入要求
            mode='bilinear',        # 双线性插值
            padding_mode='zeros',   # 超出平面的部分设为透明
            align_corners=False     # 是否将 [-1, 1] 映射到像素中心
        ) # (N_planes, 4, 1, N_rays)
        sampled_rgba_sorted = sampled_rgba_sorted.squeeze(2).permute(0, 2, 1) # (N_planes, N_rays, 4)


        # 5. Alpha Blending (从前向后累加)
        alpha = sampled_rgba_sorted[..., 3] # (N_planes, N_rays, 1)
        rgb = sampled_rgba_sorted[..., :3]  # (N_planes, N_rays, 3)
        
        one_minus_alpha = 1.0 - alpha
        input_for_cumprod = torch.cat([torch.ones_like(alpha[:1]), one_minus_alpha[:-1]], dim=0) # (N_planes, N_rays, 1)，第一层的 transmittance 是 1

        transmittance = torch.cumprod(input_for_cumprod, dim=0) # (N_planes, N_rays, 1)
        weights = transmittance * alpha # (N_planes, N_rays, 1)
        rendered_color = torch.sum(weights * rgb, dim=0) # (N_rays, 3)


        # final. 返回渲染结果和权重（权重用于深度学习）
        return rendered_color, weights