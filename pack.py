import argparse
import zipfile
import os
import re


def main():
    parser = argparse.ArgumentParser(description="将平面定义 JSON 和平面纹理图像打包为 .smp 文件")
    parser.add_argument("-j", "--json", type=str, required=True, help="平面定义 JSON 文件路径")
    parser.add_argument("-d", "--texture_dir", type=str, required=True, help="包含 plane_N.png 纹理文件的目录")
    parser.add_argument("-o", "--output", type=str, required=True, help="输出 .smp 文件路径")
    parser.add_argument("-vh", "--viewport_height", type=float, default=1.0, help="视口高度 (默认: 1.0)")
    parser.add_argument("-vw", "--viewport_width", type=float, default=1.0, help="视口宽度 (默认: 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"❌ JSON 文件不存在: {args.json}")
        return
    if not os.path.isdir(args.texture_dir):
        print(f"❌ 纹理目录不存在: {args.texture_dir}")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if not args.output.endswith('.smp'):
        output_path = args.output + '.smp'
    else:
        output_path = args.output

    pattern = re.compile(r"plane_(\d+)\.png")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(args.json, 'planes.json')
        zf.writestr('viewport.json', f'{{"height": {args.viewport_height}, "width": {args.viewport_width}}}')

        count = 0
        for fname in sorted(os.listdir(args.texture_dir), key=lambda n: int(m.group(1)) if (m := pattern.match(n)) else -1):
            m = pattern.match(fname)
            if not m:
                continue
            file_path = os.path.join(args.texture_dir, fname)
            arcname = f"planes/{m.group(1)}.png"
            zf.write(file_path, arcname)
            count += 1
            print(f"  + {fname} → {arcname}")

    print(f"✅ 已打包 {count} 个纹理 + 数据到 {output_path}")


if __name__ == "__main__":
    main()
