import os
import OpenEXR
import Imath
import numpy as np
import open3d as o3d

import argparse


def read_exr_xyz(exr_path) -> np.ndarray:
    """
    从EXR读取 Output.X/Y/Z 三个通道，返回 Nx3 numpy数组
    """
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    try:
        x = np.frombuffer(exr_file.channel('Output.X', FLOAT), dtype=np.float32)
        y = np.frombuffer(exr_file.channel('Output.Y', FLOAT), dtype=np.float32)
        z = np.frombuffer(exr_file.channel('Output.Z', FLOAT), dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"读取通道失败: {exr_path}, 错误: {e}")
    x = x.reshape((height, width))
    y = y.reshape((height, width))
    z = z.reshape((height, width))
    xyz = np.stack([x, y, z], axis=-1)  # H x W x 3
    xyz = xyz.reshape(-1, 3)            # (H*W) x 3
    return xyz


def exr_dir_to_pointcloud_streaming(
    exr_dir,
    voxel_size,
    remove_invalid=True
) -> o3d.geometry.PointCloud:
    """
    从EXR目录生成点云并保存

    Parameters: 
        exr_dir: 输入EXR目录
        output_path: 输出点云路径 (.ply / .pcd)
        voxel_size: 体素大小
        remove_invalid: 是否移除无效点
    """
    pcd = o3d.geometry.PointCloud()
    for file in os.listdir(exr_dir):
        if not file.lower().endswith(".exr"):
            continue
        path = os.path.join(exr_dir, file)
        print(f"读取: {path}")
        xyz = read_exr_xyz(path)
        # 过滤无效点
        if remove_invalid:
            mask = np.isfinite(xyz).all(axis=1)
            mask &= ~(np.abs(xyz).sum(axis=1) == 0)
            xyz = xyz[mask]
        # 当前帧点云
        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(xyz)
        # 累加
        pcd += current_pcd
        # 体素化规约
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"当前点数: {len(pcd.points)}")
    return pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从EXR目录生成点云")
    parser.add_argument("-e","--exr_dir", type=str, help="输入EXR目录")
    parser.add_argument("-o","--output_path", type=str, help="输出点云路径")
    parser.add_argument("-v","--voxel_size", type=float, default=1.0/32, help="体素大小")
    args = parser.parse_args()

    exr_dir = args.exr_dir
    output_path = args.output_path
    voxel_size = args.voxel_size

    pcd = exr_dir_to_pointcloud_streaming(exr_dir, voxel_size=voxel_size)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存: {output_path}")