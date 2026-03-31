import json
import numpy as np
import open3d as o3d
import os
import argparse
from PIL import Image
import colorsys


def rectangle_to_lineset(center, length_vec, width_vec, color=[1, 0, 0]):
    """
    根据中心点和长宽向量生成矩形 LineSet
    """
    center = np.array(center)
    l = np.array(length_vec) / 2.0
    w = np.array(width_vec) / 2.0

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


def create_texture_mesh(center, length_vec, width_vec, mask, color):
    """
    根据平面矩形+纹理mask生成三角面片，仅mask为1的像素才生成面片
    """
    center = np.array(center)
    l_vec = np.array(length_vec)
    w_vec = np.array(width_vec)

    h, w = mask.shape

    step_u = l_vec / w
    step_v = w_vec / h

    origin = center - l_vec / 2 - w_vec / 2

    vertices = []
    triangles = []
    colors = []

    idx = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                continue

            p0 = origin + x * step_u + y * step_v
            p1 = origin + (x + 1) * step_u + y * step_v
            p2 = origin + (x + 1) * step_u + (y + 1) * step_v
            p3 = origin + x * step_u + (y + 1) * step_v

            vertices.extend([p0, p1, p2, p3])
            triangles.extend([
                [idx, idx + 1, idx + 2],
                [idx, idx + 2, idx + 3]
            ])
            colors.extend([color] * 4)
            idx += 4

    if len(vertices) == 0:
        return None

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh


def generate_color(i, total):
    """
    生成不同颜色 (HSV -> RGB)
    """
    hue = i / max(total, 1)
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return list(rgb)


def visualize(output_dir=None, pcd_path=None):
    if output_dir is None and pcd_path is None:
        print("错误：必须至少输入点云路径或平面目录路径")
        return

    geometries = []

    # 加载点云
    if pcd_path:
        pcd = o3d.io.read_point_cloud(pcd_path)
        geometries.append(pcd)

    # 加载平面
    if output_dir:
        json_path = os.path.join(output_dir, "planes.json")
        if not os.path.exists(json_path):
            print(f"未找到 {json_path}")
        else:
            with open(json_path, "r") as f:
                rects = json.load(f)

            for i, rect in enumerate(rects):
                print(f"⏳ 加载平面数据中... ({i+1}/{len(rects)})")
                color = generate_color(i, len(rects))

                # 边框
                line = rectangle_to_lineset(
                    rect["center"],
                    rect["length_vec"],
                    rect["width_vec"],
                    color=color
                )
                geometries.append(line)

                # 纹理
                tex_file = rect.get("texture", f"plane_{i}.png")
                tex_path = os.path.join(output_dir, tex_file)
                if os.path.exists(tex_path):
                    img = np.array(Image.open(tex_path).convert("L"))
                    mask = (img > 0).astype(np.uint8)
                    mesh = create_texture_mesh(
                        rect["center"],
                        rect["length_vec"],
                        rect["width_vec"],
                        mask,
                        color
                    )
                    if mesh is not None:
                        geometries.append(mesh)

    # 坐标轴
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )
    geometries.append(mesh_frame)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="点云/平面查看器",
        width=1280,
        height=720,
        left=50,
        top=50,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云/平面查看器")
    parser.add_argument("-d", "--output_dir", type=str, default=None, help="平面输出目录")
    parser.add_argument("-p", "--pcd_path", type=str, default=None, help="点云路径 (.ply / .pcd)")
    args = parser.parse_args()
    visualize(args.output_dir, args.pcd_path)