import cv2
import numpy as np
import torch
import lpips
import argparse
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class ImageEvaluator:
    # ... (保持原有的 __init__, load_image, compute_metrics, show_heatmap 不变)
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
    parser.add_argument("-g", "--gt", required=True, help="Path to GT image or directory")
    parser.add_argument("-i", "--img", required=True, help="Path to restored image or directory")
    parser.add_argument("-m", "--metrics", nargs='+', choices=['psnr', 'ssim', 'lpips'], default=['psnr'])
    parser.add_argument("-s", "--show", action='store_true', help="Show error heatmap (only for single file)")
    args = parser.parse_args()

    evaluator = ImageEvaluator(use_lpips='lpips' in args.metrics)
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 逻辑判断：文件夹对文件夹，还是单文件对单文件
    is_img_dir = os.path.isdir(args.img)
    is_gt_dir = os.path.isdir(args.gt)

    if is_img_dir and is_gt_dir:
        print(f"Directory-to-Directory mode: {args.img} vs {args.gt}")
        files = [f for f in os.listdir(args.img) if f.lower().endswith(valid_ext)]
        for file_name in files:
            img_path = os.path.join(args.img, file_name)
            gt_path = os.path.join(args.gt, file_name)
            
            if os.path.exists(gt_path):
                img = evaluator.load_image(img_path)
                gt = evaluator.load_image(gt_path)
                if img is not None and gt is not None and img.shape == gt.shape:
                    results = evaluator.compute_metrics(img, gt, args.metrics)
                    res_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items()])
                    print(f"[{file_name}] {res_str}")
                else:
                    print(f"[{file_name}] Skipped (Shape mismatch or load error)")
            else:
                print(f"[{file_name}] Skipped (GT file not found)")

    elif is_img_dir: # 原有的单GT对应文件夹内多文件
        gt = evaluator.load_image(args.gt)
        if gt is None: raise ValueError("Invalid GT file")
        files = [f for f in os.listdir(args.img) if f.lower().endswith(valid_ext)]
        for file_name in files:
            img = evaluator.load_image(os.path.join(args.img, file_name))
            if img is not None and img.shape == gt.shape:
                results = evaluator.compute_metrics(img, gt, args.metrics)
                print(f"[{file_name}] {' | '.join([f'{k}: {v:.4f}' for k, v in results.items()])}")
    
    else: # 原有的单文件逻辑
        gt = evaluator.load_image(args.gt)
        img = evaluator.load_image(args.img)
        if img is not None and gt is not None and img.shape == gt.shape:
            results = evaluator.compute_metrics(img, gt, args.metrics)
            for k, v in results.items():
                print(f"{k}: {v:.4f}")
            if args.show:
                evaluator.show_heatmap(img, gt)
        else:
            print("Error: Files not found or dimension mismatch.")

if __name__ == "__main__":
    main()