#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import open3d as o3d
import os
import sys

import argparse

def visualize_pcd(file_path):
    """
    加载并可视化 PCD 格式的点云文件。
    
    :param file_path: pcd文件的绝对或相对路径
    """
    # 1. 工程规范：前置检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 '{file_path}'")
        sys.exit(1)

    print(f"⏳ 正在加载点云文件: {file_path} ...")
    
    # 2. 读取点云数据
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"❌ 错误: 读取文件时发生异常 - {e}")
        sys.exit(1)

    # 3. 校验点云数据是否有效
    if pcd.is_empty():
        print("❌ 错误: 点云为空，或者 Open3D 无法解析该文件！")
        sys.exit(1)

    # 4. 打印点云基础信息 (有助于调试和数据分析)
    print(f"✅ 加载成功！")
    print(f"📊 点云基础信息:")
    print(f"   - 包含点数: {len(pcd.points):,} 个")
    print(f"   - 包含颜色: {'是' if pcd.has_colors() else '否'}")
    print(f"   - 包含法向量: {'是' if pcd.has_normals() else '否'}")

    # 5. 用户体验优化：添加一个坐标系参照 (红=X, 绿=Y, 蓝=Z)
    # 为了防止坐标系太大或太小，我们根据点云的边界框(Bounding Box)动态计算坐标系的尺寸
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    axis_size = max(extent) * 0.2 if max(extent) > 0 else 1.0
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size, 
        origin=[0, 0, 0]
    )

    print("\n启动可视化窗口...")

    # 6. 调用 Open3D 渲染引擎
    o3d.visualization.draw_geometries(
        geometry_list=[pcd, mesh_frame],  # 将点云和坐标系放入一个列表中渲染
        window_name=f"点云查看 - {os.path.basename(file_path)}",
        width=1280,
        height=720,
        left=50,
        top=50,
        point_show_normal=False  # 如果你的点云有法向量且你想查看，可以改为 True
    )

if __name__ == "__main__":
    # 使用 argparse 处理命令行参数，这是一个良好的 CLI 工具编写习惯
    parser = argparse.ArgumentParser(description="使用 Open3D 快速预览 PCD 格式点云文件")
    parser.add_argument(
        "-f",
        "--pcd_file",
        type=str, 
        help="目标 PCD 文件的路径 (例如: ./data/sample.pcd)"
    )
    
    args = parser.parse_args()
    
    visualize_pcd(args.pcd_file)