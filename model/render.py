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
    # 获取 skip 参数，默认为 0
    skip_count = int(args[2]) if len(args) > 2 else 0
    
    # 测试模式：仅显示采样点方向，不进行渲染
    # if True:
    #     test(samples)
    #     return

    scene = bpy.context.scene

    print(f"输出路径: {scene.render.filepath}")
    print(f"采样数: {samples}")
    if skip_count > 0: print(f"跳过前: {skip_count} 次采样")

    for i, (theta, phi) in enumerate(fibonacci_hemisphere_angles(samples)):
        if i < skip_count: continue
        
        print(f"采样 {i+1}/{samples}: theta={math.degrees(theta):.2f}°, phi={math.degrees(phi):.2f}°")

        # 弧度→角度
        theta = math.degrees(theta)
        phi = math.degrees(phi)

        # 设置相机旋转
        cam = bpy.data.cameras['Camera']
        cam.cycles_custom["angle_theta"] = theta
        cam.cycles_custom["angle_phi"] = phi

        # 渲染并保存
        scene.render.filepath = f"{output_path}/rgba_{phi:.4f}_{theta:.4f}.png"
        bpy.data.node_groups["合成器节点"].nodes["文件输出"].directory = output_path
        bpy.data.node_groups["合成器节点"].nodes["文件输出"].file_name = f"xyz_{phi:.4f}_{theta:.4f}.png"

        bpy.ops.render.render(write_still=True)


def test(samples, collection_name="TestPoints"):
    """
    清理指定集合，并在指定集合中创建符合采样角度的空物体
    """
    # 1. 确保集合存在，如果不存在则创建
    if collection_name not in bpy.data.collections:
        new_col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_col)
    
    collection = bpy.data.collections[collection_name]
    
    # 2. 清理：删除该集合下所有名字以 "Point." 开头的物体
    # 注意：为了安全，建议先从集合中解除链接并移除数据块
    objs_to_remove = [obj for obj in collection.objects if obj.name.startswith("Point.")]
    for obj in objs_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # 3. 生成新点
    # 使用你现有的 fibonacci_hemisphere_angles 函数
    # 注意：这里需要传入弧度值，逻辑保持一致
    angles = fibonacci_hemisphere_angles(samples)
    
    for i, (theta, phi) in enumerate(angles):
        # 将球面坐标转换为笛卡尔坐标 (x, y, z)
        # 根据你的逻辑：z=1 为顶点，原点 (0,0,0) 到对应方向长度为 1
        # 计算逻辑需与 fibonacci 内部对应
        u = (i + 0.5) / samples
        z = 1.0 - (u ** 2.0)
        r = math.sqrt(max(0.0, 1.0 - z*z))
        az = math.pi * (3.0 - math.sqrt(5.0)) * i
        
        x = r * math.cos(az)
        y = r * math.sin(az)
        
        # 创建空物体
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(x, y, z))
        point = bpy.context.active_object
        point.name = f"Point.{i:03d}"
        
        # 将空物体移入指定集合
        if point.name in bpy.context.collection.objects:
            bpy.context.collection.objects.unlink(point)
        collection.objects.link(point)
        
        # 调整空物体大小方便观察
        point.empty_display_size = 0.1


def fibonacci_hemisphere_angles(n: int, exponent: float = 2.0) -> list[tuple[float, float]]:
    """
    返回 n 个分布在前方半球的 (theta, phi) 角度列表
    :param n: 采样总数
    :param exponent: 分布指数，1.0为均匀分布，>1.0值越大，圆心处越密集
    """
    angles = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # 约等于 2.39996

    for i in range(n):
        # 归一化索引 u ∈ [0, 1]
        u = (i + 0.5) / n
        
        # 应用二次方（或幂次）分布
        # u^exponent 让点在靠近圆心(z=1)处堆积
        u_weighted = u ** exponent
        
        # z 轴坐标：从 1 (圆心) 递减到 0 (圆周边缘)
        z = 1.0 - u_weighted
        
        # 确保 z 不会因为浮点误差小于0
        z = max(0.0, min(1.0, z))
        
        # 根据球面方程计算半径 r
        r = math.sqrt(max(0.0, 1.0 - z*z))
        
        # 黄金角分布
        az = golden_angle * i
        
        x = r * math.cos(az)
        y = r * math.sin(az)
        
        # 转换为相机旋转角
        phi = math.atan2(x, z) 
        theta = math.atan2(y, math.sqrt(x*x + z*z))
        
        angles.append((theta, phi))
    return angles


if __name__ == "__main__":
    main()