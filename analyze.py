import cv2
import numpy as np
import torch
import lpips
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class ImageEvaluator:
    def __init__(self, use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img / 255.0

    def compute_metrics(self, img1, img2, metrics):
        results = {}
        if 'psnr' in metrics:
            results['PSNR'] = psnr(img1, img2)
        if 'ssim' in metrics:
            # multichannel=True 适用于彩色图像
            results['SSIM'] = ssim(img1, img2, channel_axis=2, data_range=1.0)
        
        if 'lpips' in metrics:
            t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(self.device) * 2 - 1
            t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(self.device) * 2 - 1
            results['LPIPS'] = self.lpips_loss(t1, t2).item()
        
        return results

    def show_heatmap(self, img1, img2):
        # 1. 计算误差图
        diff = np.abs(img1 - img2)
        # 2. 转换为 0-255 的 uint8 格式供 OpenCV 使用
        heatmap_gray = np.mean(diff, axis=2)
        # 将灰度图映射为伪彩色热力图 (cv2.COLORMAP_JET 看起来最直观)
        heatmap_colored = cv2.applyColorMap((heatmap_gray * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 3. 将原图和热力图水平拼接以便观察
        # img1 是 0-1 浮点数，需要转回 0-255 的 uint8
        # img1_uint8 = (img1 * 255).astype(np.uint8)
        # OpenCV 使用 BGR 格式，所以需要从 RGB 转 BGR
        # img1_bgr = cv2.cvtColor(img1_uint8, cv2.COLOR_RGB2BGR)
        
        # 水平拼接显示
        display_img = heatmap_colored # np.hstack((img1_bgr, heatmap_colored))
        
        # 不拼不拼

        # 4. 使用 OpenCV 显示
        cv2.imshow("Evaluation Result (Error Map)", display_img)
        print("Press any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Image Quality Evaluation Tool")
    parser.add_argument("-g", "--gt", required=True, help="Path to GT image")
    parser.add_argument("-i", "--img", required=True, help="Path to restored image")
    parser.add_argument("-m", "--metrics", nargs='+', choices=['psnr', 'ssim', 'lpips'], default=['psnr'])
    parser.add_argument("-s", "--show", action='store_true', help="Show error heatmap")
    args = parser.parse_args()

    evaluator = ImageEvaluator()
    img = evaluator.load_image(args.img)
    gt = evaluator.load_image(args.gt)

    # 简单的尺寸检查
    if img.shape != gt.shape:
        print(f"Warning: Dimensions mismatch!")
        return

    results = evaluator.compute_metrics(img, gt, args.metrics)
    
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    if args.show:
        evaluator.show_heatmap(img, gt)


if __name__ == "__main__":
    main()