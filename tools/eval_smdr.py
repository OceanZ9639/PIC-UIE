"""Evaluate images with the 256 x 256 SMDR-IS PSNR/SSIM protocol.

Usage:
  python tools/eval_smdr.py --enhanced results/v1 --gt UIEB/target \
      --list UIEB/test_list.txt
"""
import os
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim as ms_ssim


def torch_psnr(tar, prd):
    d = torch.clamp(prd, 0, 1) - torch.clamp(tar, 0, 1)
    rmse = (d ** 2).mean().sqrt()
    return float(20 * torch.log10(1.0 / rmse))


def per_channel_psnr(tar, prd):
    out = []
    for c in range(3):
        d = torch.clamp(prd[c], 0, 1) - torch.clamp(tar[c], 0, 1)
        rmse = (d ** 2).mean().sqrt()
        out.append(float(20 * torch.log10(1.0 / rmse)))
    return out  # cv2 BGR order -> [B, G, R]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhanced", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--list", default=None, help="optional split list (filenames)")
    args = ap.parse_args()

    if args.list:
        names = [l.strip() for l in open(args.list) if l.strip()]
    else:
        names = sorted(n for n in os.listdir(args.enhanced)
                       if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))

    ps, ss, pc = [], [], []
    for nm in names:
        ep = os.path.join(args.enhanced, nm)
        gp = os.path.join(args.gt, nm)
        if not (os.path.exists(ep) and os.path.exists(gp)):
            continue
        e = cv2.resize(cv2.imread(ep), [256, 256])
        g = cv2.resize(cv2.imread(gp), [256, 256])
        e = TF.to_tensor(e)
        g = TF.to_tensor(g)
        ps.append(torch_psnr(g, e))
        ss.append(float(ms_ssim(g.unsqueeze(0), e.unsqueeze(0), data_range=1.0, size_average=True)))
        pc.append(per_channel_psnr(g, e))
    if not ps:
        raise RuntimeError("No matching enhanced/reference image pairs were found")

    pc = np.array(pc)  # [N,3] in B,G,R (cv2)
    print(f"n={len(ps)}  PSNR {np.mean(ps):.3f} | SSIM {np.mean(ss):.4f}")
    print(f"per-channel PSNR  R {pc[:,2].mean():.2f} | G {pc[:,1].mean():.2f} | B {pc[:,0].mean():.2f}")


if __name__ == "__main__":
    main()
