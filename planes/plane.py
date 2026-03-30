from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import open3d as o3d
import copy

import argparse
import json

@dataclass
class PlaneRectangle:
    center: np.ndarray      # 矩形中心点 (3,)
    normal: np.ndarray      # 单位法向量 (3,)
    length_vec: np.ndarray  # 长方向向量 (3,), 模长为长度
    width_vec: np.ndarray   # 宽方向向量 (3,), 模长为宽度

def fit_planes_with_rectangles(
    pcd: o3d.geometry.PointCloud,
    num_planes: int,
    dist_thresh: float = 0.02,
    cost_thresh: float = 0.08,            # 判定点是否属于平面的综合阈值
    max_em_iter: int = 10,
    init_ransac_iter: int = 1000,
    normal_radius: float = 0.1,           # 法线估计半径，建议设为体素大小的2-3倍
    normal_max_nn: int = 30,
    alpha: float = 1.0,                   # 距离代价权重
    beta: float = 0.5,                    # 法线代价权重
    plane_margin: float = 1.0/32          # 平面边界外扩
) -> List[PlaneRectangle]:
    """
    基于全局 EM 优化的多平面拟合算法

    Parameters: 
        pcd: 输入点云（仅包含位置信息）
        num_planes: 需要拟合的平面数量（最多）
        dist_thresh: RANSAC 平面拟合的距离阈值
        cost_thresh: EM 迭代中点归属阈值（超出则视为离群）
        max_em_iter: EM 最大迭代次数
        init_ransac_iter: RANSAC 初始化迭代次数
        normal_radius: 法线估计邻域半径
        normal_max_nn: 法线估计最大邻居数
        alpha: 距离代价权重
        beta: 法线代价权重
        plane_margin: 平面边界外扩尺寸
    Returns:
        List[PlaneRectangle]
            每个平面的有界矩形表示
    """
    
    # ---- 1. 内部计算法线 (输入点云仅含位置信息) ----
    # 使用深拷贝防止修改原始点云
    working_pcd = copy.deepcopy(pcd)
    working_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    # 统一法线朝向（可选，有助于计算一致性）
    working_pcd.orient_normals_towards_camera_location(working_pcd.get_center())

    points = np.asarray(working_pcd.points)
    normals = np.asarray(working_pcd.normals)

    # ---- 2. 初始种子生成 (Sequential RANSAC) ----
    planes = []
    temp_pcd = copy.deepcopy(working_pcd)
    
    for _ in range(num_planes):
        if len(temp_pcd.points) < 10:
            break
        plane_model, inliers = temp_pcd.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=init_ransac_iter
        )
        planes.append(np.array(plane_model))
        temp_pcd = temp_pcd.select_by_index(inliers, invert=True)

    if not planes:
        return []

    # ---- 3. EM 全局迭代优化 ----
    # 目的：统筹分配资源，让大结构更准，小结构归类
    for _ in range(max_em_iter):
        clusters = [[] for _ in range(len(planes))]

        # E-Step: 每个点在全局 N 个平面中找最优归属
        for i in range(len(points)):
            p = points[i]
            n = normals[i]
            
            best_cost = np.inf
            best_idx = -1
            
            for k, plane in enumerate(planes):
                a, b, c, d = plane
                plane_n = plane[:3]
                
                # 计算距离
                dist = abs(np.dot(plane_n, p) + d) / np.linalg.norm(plane_n)
                # 计算法线夹角偏差 (1 - cos_theta)
                cos_theta = abs(np.dot(n, plane_n / np.linalg.norm(plane_n)))
                angle_err = 1.0 - np.clip(cos_theta, 0.0, 1.0)
                
                # 综合 Cost
                cost = alpha * dist + beta * angle_err
                
                if cost < best_cost:
                    best_cost = cost
                    best_idx = k
            
            # 只有在阈值内的点才参与拟合，排除干扰点
            if best_cost < cost_thresh:
                clusters[best_idx].append(i)

        # M-Step: 根据分配结果更新平面方程
        new_planes = []
        for k in range(len(planes)):
            if len(clusters[k]) < 3:
                new_planes.append(planes[k])
                continue
            
            cluster_pts = points[clusters[k]]
            # 使用 PCA 拟合平面
            centroid = np.mean(cluster_pts, axis=0)
            cov = np.cov(cluster_pts.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            new_normal = eigenvectors[:, 0] # 最小特征值对应的特征向量是法线
            new_d = -np.dot(new_normal, centroid)
            new_planes.append(np.array([*new_normal, new_d]))
        
        # 检查是否收敛
        change = sum(np.linalg.norm(planes[i][:3] - new_planes[i][:3]) for i in range(len(planes)))
        planes = new_planes
        if change < 1e-4:
            break

    # ---- 4. 提取有界矩形 ----
    rectangles: List[PlaneRectangle] = []
    
    for k, cluster_indices in enumerate(clusters):
        if len(cluster_indices) < 4:
            continue
        
        cluster_pts = points[cluster_indices]
        plane_n = planes[k][:3]
        plane_n /= np.linalg.norm(plane_n)
        
        # 建立平面局部坐标系 (u, v)
        if abs(plane_n[2]) < 0.9:
            u = np.cross(plane_n, [0, 0, 1])
        else:
            u = np.cross(plane_n, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(plane_n, u)

        # 投影到 2D
        centroid = np.mean(cluster_pts, axis=0)
        pts_rel = cluster_pts - centroid
        pts_2d = np.stack([np.dot(pts_rel, u), np.dot(pts_rel, v)], axis=1)

        # 在 2D 中计算最小外接矩形的轴 (使用 2D PCA 简化)
        cov_2d = np.cov(pts_2d.T)
        _, evecs_2d = np.linalg.eigh(cov_2d) # evecs_2d[:, 1] 是主成分方向
        
        axis_long_2d = evecs_2d[:, 1]
        axis_wide_2d = evecs_2d[:, 0]
        
        # 计算 2D 投影范围
        proj_long = pts_2d @ axis_long_2d
        proj_wide = pts_2d @ axis_wide_2d
        
        l_min, l_max = proj_long.min() - plane_margin, proj_long.max() + plane_margin
        w_min, w_max = proj_wide.min() - plane_margin, proj_wide.max() + plane_margin
        
        # 转换回 3D 坐标
        # 矩形中心 (考虑 2D 投影的偏移)
        center_2d = ((l_min + l_max) / 2.0) * axis_long_2d + ((w_min + w_max) / 2.0) * axis_wide_2d
        rect_center_3d = centroid + center_2d[0] * u + center_2d[1] * v
        
        # 长度和宽度向量
        rect_l_vec = (l_max - l_min) * (axis_long_2d[0] * u + axis_long_2d[1] * v)
        rect_w_vec = (w_max - w_min) * (axis_wide_2d[0] * u + axis_wide_2d[1] * v)
        
        # 确保 length_vec 是较长的那一个
        if np.linalg.norm(rect_w_vec) > np.linalg.norm(rect_l_vec):
            rect_l_vec, rect_w_vec = rect_w_vec, rect_l_vec

        rectangles.append(PlaneRectangle(
            center=rect_center_3d,
            normal=plane_n,
            length_vec=rect_l_vec,
            width_vec=rect_w_vec
        ))
        
    return rectangles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从点云中拟合指定数量的平面")
    parser.add_argument("-i","--input_path", type=str, help="输入点云路径 (.ply / .pcd)")
    parser.add_argument("-o","--output_path", type=str, help="输出平面信息路径 (.json)")
    parser.add_argument("-n","--num_planes", type=int, default=8, help="拟合的平面数量")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    num_planes = args.num_planes

    pcd = o3d.io.read_point_cloud(input_path)
    rectangles = fit_planes_with_rectangles(pcd, num_planes=num_planes)

    with open(output_path, "w") as f:
        json.dump([{
            "center": rect.center.tolist(),
            "normal": rect.normal.tolist(),
            "length_vec": rect.length_vec.tolist(),
            "width_vec": rect.width_vec.tolist()
        } for rect in rectangles], f, indent=4)

    print(f"平面拟合完成，结果已保存到: {output_path}")