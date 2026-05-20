import cv2
import numpy as np
import torch
import lpips
import argparse
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class ImageEvaluator:
    def __init__(self, use_gpu=True, use_lpips=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if use_lpips:
            self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img / 255.0

    def compute_metrics(self, img1, img2, metrics):
        results = {}
        if 'psnr' in metrics:
            results['PSNR'] = psnr(img1, img2)
        if 'ssim' in metrics:
            results['SSIM'] = ssim(img1, img2, channel_axis=2, data_range=1.0)
        if 'lpips' in metrics:
            t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(self.device) * 2 - 1
            t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(self.device) * 2 - 1
            results['LPIPS'] = self.lpips_loss(t1, t2).item()
        return results

    def show_heatmap(self, img1, img2):
        diff = np.abs(img1 - img2)
        heatmap_gray = np.mean(diff, axis=2)
        heatmap_colored = cv2.applyColorMap((heatmap_gray * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("Evaluation Result (Error Map)", heatmap_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Image Quality Evaluation Tool")
    parser.add_argument("-g", "--gt", required=True, help="Path to GT image")
    parser.add_argument("-i", "--img", required=True, help="Path to restored image or directory")
    parser.add_argument("-m", "--metrics", nargs='+', choices=['psnr', 'ssim', 'lpips'], default=['psnr'])
    parser.add_argument("-s", "--show", action='store_true', help="Show error heatmap (only for single file)")
    args = parser.parse_args()

    evaluator = ImageEvaluator(use_lpips='lpips' in args.metrics)
    gt = evaluator.load_image(args.gt)
    if gt is None:
        print(f"Error: Could not load GT image at {args.gt}")
        return

    # 判断输入是目录还是文件
    if os.path.isdir(args.img):
        print(f"Directory mode detected: {args.img}")
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        files = [f for f in os.listdir(args.img) if f.lower().endswith(valid_ext)]
        
        for file_name in files:
            file_path = os.path.join(args.img, file_name)
            img = evaluator.load_image(file_path)
            if img is not None and img.shape == gt.shape:
                results = evaluator.compute_metrics(img, gt, args.metrics)
                res_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items()])
                print(f"[{file_name}] {res_str}")
            else:
                print(f"[{file_name}] Skipped (Shape mismatch or load error)")
                
    else:
        # 原有的单文件逻辑
        img = evaluator.load_image(args.img)
        if img is not None and img.shape == gt.shape:
            results = evaluator.compute_metrics(img, gt, args.metrics)
            for k, v in results.items():
                print(f"{k}: {v:.4f}")
            if args.show:
                evaluator.show_heatmap(img, gt)
        else:
            print("Error: Image not found or dimension mismatch.")

if __name__ == "__main__":
    main()