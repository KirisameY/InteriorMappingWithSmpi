import bpy
import sys
import math

def main():
    # 获取命令行中 "--" 之后的所有参数
    try:
        args = sys.argv[sys.argv.index("--") + 1:]
    except (ValueError, IndexError):
        args = []

    # 解析参数 (目录、采样数)
    if len(args) < 2 : 
        print("参数不足")
        return
    output_path = args[0]
    samples = int(args[1])
    
    scene = bpy.context.scene

    print(f"输出路径: {scene.render.filepath}")
    print(f"采样数: {samples}")

    for theta, phi in fibonacci_hemisphere_angles(samples):
        # 弧度→角度
        theta = math.degrees(theta)
        phi = math.degrees(phi)

        # 设置相机旋转
        cam = bpy.data.objects['Camera']
        bpy.data.cameras["摄像机"].cycles_custom["angle_theta"] = theta
        bpy.data.cameras["摄像机"].cycles_custom["angle_phi"] = phi

        # 渲染并保存
        scene.render.filepath = f"{output_path}/rgba_{phi:.4f}_{theta:.4f}.png"
        bpy.data.node_groups["合成器节点"].nodes["文件输出"].directory = output_path
        bpy.data.node_groups["合成器节点"].nodes["文件输出"].file_name = f"xyz_{phi:.4f}_{theta:.4f}.png"

        bpy.ops.render.render(write_still=True)


def fibonacci_hemisphere_angles(n: int) -> list[tuple[float, float]]:
    """返回 n 个均匀分布在前方半球的 (theta, phi) 角度列表"""
    angles = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # 约等于 2.39996
    for i in range(n):
        # 均匀 z（前方半球）
        u = (i + 0.5) / n
        z = 1.0 - u  # z ∈ (0,1]
        
        r = math.sqrt(1.0 - z*z)
        az = golden_angle * i
        # 点坐标（x右 y上 z前）
        x = r * math.cos(az)
        y = r * math.sin(az)
        # 转换为相机旋转角
        phi = math.atan2(x, z)  # yaw 左右
        theta = math.atan2(y, math.sqrt(x*x + z*z))  # pitch 上下
        angles.append((theta, phi))
    return angles


if __name__ == "__main__":
    main()