import os
import torch
from torchvision.utils import save_image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="模型检查点路径")
    parser.add_argument("-o", "--output", type=str, help="输出纹理文件的目录")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    textures : torch.Tensor = checkpoint['textures']  # (num_planes, 4, tex_width, tex_height)
    textures = textures.cpu()
    os.makedirs(args.output, exist_ok=True)

    for i, texture in enumerate(textures):
        texture = texture.permute(0, 2, 1)  # (C, W, H) -> (C, H, W)
        output_path = os.path.join(args.output, f"plane_{i}.png")
        save_image(texture, output_path)
        print(f"已保存纹理 {i} 到 {output_path}")