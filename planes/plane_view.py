import json
import numpy as np
import open3d as o3d
import os
import sys

import argparse


def rectangle_to_lineset(center, length_vec, width_vec, color=[1, 0, 0]):
    """
    根据中心点和长宽向量生成矩形 LineSet
    """
    center = np.array(center)
    l = np.array(length_vec) / 2.0
    w = np.array(width_vec) / 2.0

    # 四个角点
    p0 = center - l - w
    p1 = center + l - w
    p2 = center + l + w
    p3 = center - l + w

    points = [p0, p1, p2, p3]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


def visualize_rectangles(json_path, pcd_path=None):
    """
    可视化 JSON 中的矩形平面，支持叠加点云
    """
    with open(json_path, "r") as f:
        rects = json.load(f)

    geometries = []

    # 加载点云（可选）
    if pcd_path is not None:
        pcd = o3d.io.read_point_cloud(pcd_path)
        geometries.append(pcd)

    # 添加每个矩形
    for rect in rects:
        line = rectangle_to_lineset(
            rect["center"],
            rect["length_vec"],
            rect["width_vec"],
            color=[1, 0, 0]
        )
        geometries.append(line)

    # 添加一个坐标系参照 (红=X, 绿=Y, 蓝=Z)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, 
        origin=[0, 0, 0]
    )
    geometries.append(mesh_frame)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"平面查看 - {os.path.basename(json_path)}",
        width=1280,
        height=720,
        left=50,
        top=50,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 JSON 中的矩形平面")
    parser.add_argument("-j", "--json_path", type=str, help="输入 JSON 文件路径")
    parser.add_argument("-p", "--pcd_path", type=str, default=None, help="可选的点云文件路径 (.ply / .pcd)")
    args = parser.parse_args()
    visualize_rectangles(args.json_path, args.pcd_path)