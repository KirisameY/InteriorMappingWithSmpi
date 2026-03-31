from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import open3d as o3d
import copy
import os
from PIL import Image

import argparse
import json

@dataclass
class PlaneRectangle:
    center: np.ndarray
    normal: np.ndarray
    length_vec: np.ndarray
    width_vec: np.ndarray
    texture: Optional[np.ndarray] = None  # float 0~1

def fit_planes_with_rectangles(
    pcd: o3d.geometry.PointCloud,
    num_planes: int,
    dist_thresh: float = 0.02,
    cost_thresh: float = 0.08,
    max_em_iter: int = 10,
    init_ransac_iter: int = 1000,
    normal_radius: float = 0.1,
    normal_max_nn: int = 30,
    alpha: float = 1.0,
    beta: float = 0.5,
    plane_margin: float = 1.0/32,
    texture_size: Tuple[int, int] = (512, 512),
    point_spread: float = 1.5/32
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
        point_spread: 每个点在纹理上的高斯扩散 sigma
    Returns:
        List[PlaneRectangle]
            每个平面的有界矩形表示
    """
    
    # ---- 1. 计算法线 ----
    print("⏳ 计算法线中...")
    working_pcd = copy.deepcopy(pcd)
    working_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    working_pcd.orient_normals_towards_camera_location(working_pcd.get_center())

    points = np.asarray(working_pcd.points)
    normals = np.asarray(working_pcd.normals)

    # ---- 2. 初始种子 ----
    planes = []
    temp_pcd = copy.deepcopy(working_pcd)

    for _ in range(num_planes):
        print(f"⏳ 初始平面拟合中... ({_+1}/{num_planes})")
        if len(temp_pcd.points) < 10:
            break
        plane_model, inliers = temp_pcd.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=init_ransac_iter
        )
        planes.append(np.array(plane_model))
        temp_pcd = temp_pcd.select_by_index(inliers, invert=True)
        print(f"  - 平面{len(planes)}/{num_planes}拟合完成，剩余点数: {len(temp_pcd.points)}")

    if not planes:
        print("⚠️ 警告: 未能拟合出任何平面！")
        return []

    # ---- 3. EM 优化 ----
    for _ in range(max_em_iter):
        print(f"⏳ EM 优化中... ({_:d}/{max_em_iter})")
        clusters = [[] for _ in range(len(planes))]

        for i in range(len(points)):
            p = points[i]
            n = normals[i]
            best_cost = np.inf
            best_idx = -1

            for k, plane in enumerate(planes):
                plane_n = plane[:3]

                dist = abs(np.dot(plane_n, p) + plane[3]) / np.linalg.norm(plane_n)
                cos_theta = abs(np.dot(n, plane_n / np.linalg.norm(plane_n)))
                angle_err = 1.0 - np.clip(cos_theta, 0.0, 1.0)

                cost = alpha * dist + beta * angle_err
                if cost < best_cost:
                    best_cost = cost
                    best_idx = k

            if best_cost < cost_thresh:
                clusters[best_idx].append(i)

        new_planes = []
        for k in range(len(planes)):
            if len(clusters[k]) < 3:
                new_planes.append(planes[k])
                continue

            cluster_pts = points[clusters[k]]
            centroid = np.mean(cluster_pts, axis=0)
            cov = np.cov(cluster_pts.T)
            _, eigenvectors = np.linalg.eigh(cov)
            new_normal = eigenvectors[:, 0]
            new_d = -np.dot(new_normal, centroid)
            new_planes.append(np.array([*new_normal, new_d]))

        change = sum(np.linalg.norm(planes[i][:3] - new_planes[i][:3]) for i in range(len(planes)))
        planes = new_planes
        if change < 1e-4:
            break

    # ---- 4. 提取矩形 + 纹理 ----
    rectangles: List[PlaneRectangle] = []
    tex_w, tex_h = texture_size

    for k, cluster_indices in enumerate(clusters):
        print(f"⏳ 提取平面矩形中... ({k+1}/{len(planes)})")
        if len(cluster_indices) < 4:
            continue

        cluster_pts = points[cluster_indices]
        plane_n = planes[k][:3]
        plane_n /= np.linalg.norm(plane_n)

        if abs(plane_n[2]) < 0.9:
            u = np.cross(plane_n, [0, 0, 1])
        else:
            u = np.cross(plane_n, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(plane_n, u)

        centroid = np.mean(cluster_pts, axis=0)
        pts_rel = cluster_pts - centroid
        # 仅投影，不关心法向距离
        pts_2d = np.stack([np.dot(pts_rel, u), np.dot(pts_rel, v)], axis=1)

        cov_2d = np.cov(pts_2d.T)
        _, evecs_2d = np.linalg.eigh(cov_2d)
        axis_long_2d = evecs_2d[:, 1]
        axis_wide_2d = evecs_2d[:, 0]

        proj_long = pts_2d @ axis_long_2d
        proj_wide = pts_2d @ axis_wide_2d

        l_min, l_max = proj_long.min() - plane_margin, proj_long.max() + plane_margin
        w_min, w_max = proj_wide.min() - plane_margin, proj_wide.max() + plane_margin

        center_2d = ((l_min + l_max) / 2.0) * axis_long_2d + ((w_min + w_max) / 2.0) * axis_wide_2d
        rect_center_3d = centroid + center_2d[0] * u + center_2d[1] * v

        rect_l_vec = (l_max - l_min) * (axis_long_2d[0] * u + axis_long_2d[1] * v)
        rect_w_vec = (w_max - w_min) * (axis_wide_2d[0] * u + axis_wide_2d[1] * v)
        if np.linalg.norm(rect_w_vec) > np.linalg.norm(rect_l_vec):
            rect_l_vec, rect_w_vec = rect_w_vec, rect_l_vec

        # ---- 高斯纹理生成 ----
        print(f"⏳ 生成平面纹理中... ({k+1}/{len(planes)})")
        texture = np.zeros((tex_h, tex_w), dtype=np.float32)
        sigma = point_spread 
        radius_world = 3 * sigma

        if l_max > l_min and w_max > w_min:
            xs = (proj_long - l_min) / (l_max - l_min) * (tex_w - 1)
            ys = (proj_wide - w_min) / (w_max - w_min) * (tex_h - 1)

            # 像素对应的世界长度
            dx = (l_max - l_min) / (tex_w - 1)
            dy = (w_max - w_min) / (tex_h - 1)

            radius_px_x = int(np.ceil(radius_world / dx))
            radius_px_y = int(np.ceil(radius_world / dy))

            for px, py in zip(xs, ys):
                if px < 0 or px >= tex_w or py < 0 or py >= tex_h:
                    continue
                x0 = int(np.floor(px))
                y0 = int(np.floor(py))
                for yy in range(y0 - radius_px_y, y0 + radius_px_y + 1):
                    if yy < 0 or yy >= tex_h:
                        continue
                    for xx in range(x0 - radius_px_x, x0 + radius_px_x + 1):
                        if xx < 0 or xx >= tex_w:
                            continue
                        # 转为世界距离再计算高斯
                        dx_world = (xx - px) * dx
                        dy_world = (yy - py) * dy
                        d2_world = dx_world**2 + dy_world**2
                        weight = np.exp(-d2_world / (2 * sigma * sigma))
                        texture[yy, xx] += weight

            # 限制为0~1
            texture = np.clip(texture, 0.0, 1.0)

        rectangles.append(PlaneRectangle(
            center=rect_center_3d,
            normal=plane_n,
            length_vec=rect_l_vec,
            width_vec=rect_w_vec,
            texture=texture
        ))

    return rectangles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从点云中拟合指定数量的平面")
    parser.add_argument("-i", "--input_path", type=str, help="输入点云路径")
    parser.add_argument("-o", "--output_dir", type=str, help="输出目录")
    parser.add_argument("-n", "--num_planes", type=int, default=8, help="拟合平面数量")
    parser.add_argument("--texture_size", type=int, nargs=2, default=[512, 512], help="纹理尺寸 (W H)")
    parser.add_argument("--point_spread", type=float, default=1.5/32, help="每个点在纹理上的高斯扩散 sigma (像素)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(args.input_path)
    rectangles = fit_planes_with_rectangles(
        pcd,
        num_planes=args.num_planes,
        texture_size=tuple(args.texture_size),
        point_spread=args.point_spread
    )

    # 保存纹理
    for i, rect in enumerate(rectangles):
        tex_path = os.path.join(args.output_dir, f"plane_{i}.png")
        # float->可视化
        Image.fromarray((rect.texture * 255).astype(np.uint8)).save(tex_path)

    # 保存 JSON
    json_path = os.path.join(args.output_dir, "planes.json")
    with open(json_path, "w") as f:
        json.dump([{
            "center": rect.center.tolist(),
            "normal": rect.normal.tolist(),
            "length_vec": rect.length_vec.tolist(),
            "width_vec": rect.width_vec.tolist(),
            "texture": f"plane_{i}.png"
        } for i, rect in enumerate(rectangles)], f, indent=4)

    print(f"✅️ 平面拟合完成，结果已保存到目录: {args.output_dir}")