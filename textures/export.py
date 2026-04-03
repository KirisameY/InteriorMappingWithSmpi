import os
import torch
from torchvision.utils import save_image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="模型检查点路径")
    parser.add_argument("-o", "--output", type=str, help="输出纹理文件的目录")
    parser.add_argument("-m", "--merged", action="store_true", help="是否将所有平面纹理合并成一张大图")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    textures : torch.Tensor = checkpoint['textures']  # (num_planes, 4, tex_width, tex_height)
    textures = textures.cpu()
    os.makedirs(args.output, exist_ok=True)

    if args.merged:
        # 将所有纹理拼接成一张大图
        merged_texture = textures.permute(1, 2, 0, 3).reshape(4, textures.shape[2], textures.shape[0]*textures.shape[3])  # (4, tex_width, num_planes*tex_height)
        merged_texture = merged_texture.permute(0, 2, 1)  # (4, num_planes*tex_height, tex_width)
        output_path = os.path.join(args.output, "plane_merged.png")
        save_image(merged_texture, output_path)
        print(f"已保存合并纹理到 {output_path}")
    else:
        for i, texture in enumerate(textures):
            texture = texture.permute(0, 2, 1)  # (C, W, H) -> (C, H, W)
            output_path = os.path.join(args.output, f"plane_{i}.png")
            save_image(texture, output_path)
            print(f"已保存纹理 {i} 到 {output_path}")