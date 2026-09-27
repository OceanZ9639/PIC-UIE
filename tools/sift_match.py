"""Compare SIFT matching on raw and PIC-UIE-enhanced underwater frames.

For each FLSea scene, take a few adjacent frames, run SIFT detect + Lowe ratio match +
RANSAC homography on consecutive pairs. Report mean keypoints / good matches / inliers.

    python tools/sift_match.py --ckpt weights/pic_uie.pth --archive /path/to/FLSea
"""
import os, sys, glob, argparse
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pic_uie.model import PICUIE


def load_model(ckpt, device):
    s = torch.load(ckpt, map_location="cpu")
    a = s.get("args", {})
    kw = dict(c=a.get("c", 16), knots=a.get("knots", 16), chroma_ds=a.get("chroma_ds", 4),
              chroma=a.get("chroma", "affine"), clut_size=a.get("clut_size", 9),
              depth_cond_chroma=a.get("depth_cond_chroma", False), tone=a.get("tone", "curve"),
              grid_d=a.get("grid_d", 8), refine_y=a.get("refine_y", 1), lum=a.get("lum", "tgain"),
              ccm=a.get("ccm", False), colormatch=a.get("colormatch", False),
              in_norm=a.get("in_norm", False), global_stats=a.get("global_stats", False),
              y_histeq=a.get("y_histeq", False), y_rangenorm=a.get("y_rangenorm", False))
    net = PICUIE(**kw).to(device).eval()
    net.load_state_dict(s["model"])
    return net


@torch.no_grad()
def enhance(net, bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = TF.to_tensor(Image.fromarray(rgb)).unsqueeze(0).to(device)
    out = net(x)["out"].clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out = (out * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def sift_pair(sift, bf, g1, g2, ratio=0.75):
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return len(k1 or []), len(k2 or []), 0, 0
    good = []
    for m_n in bf.knnMatch(d1, d2, k=2):
        if len(m_n) == 2 and m_n[0].distance < ratio * m_n[1].distance:
            good.append(m_n[0])
    inl = 0
    if len(good) >= 4:
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if mask is not None:
            inl = int(mask.sum())
    return len(k1), len(k2), len(good), inl


def scene_frames(scene_dir, n=4, stride=4):
    fs = sorted(glob.glob(os.path.join(scene_dir, "imgs", "*")))
    if len(fs) < (n - 1) * stride + 1:
        stride = 1
    start = len(fs) // 2
    return [fs[start + i * stride] for i in range(n) if start + i * stride < len(fs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/pic_uie.pth")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--scenes", default="canyons/flatiron,canyons/u_canyon,"
                    "red_sea/coral_table_loop,red_sea/dice_path,red_sea/pier_path")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    net = load_model(args.ckpt, args.device)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)

    agg = {"raw": [0, 0, 0], "pic_uie": [0, 0, 0]}   # [sum_kp, sum_good, sum_inl]
    npairs = 0
    print(f"=== SIFT matching: raw vs PIC-UIE ({args.n} adjacent frames/scene, stride={args.stride}) ===")
    print(f"{'scene':28s} {'pair':>5} | {'raw kp/good/inl':>22} | {'PIC-UIE kp/good/inl':>22}")
    for sc in args.scenes.split(","):
        sc = sc.strip()
        frames = scene_frames(os.path.join(args.archive, sc), args.n, args.stride)
        if len(frames) < 2:
            print(f"  {sc}: <2 frames, skip"); continue
        bgr = [cv2.imread(f) for f in frames]
        enh = [enhance(net, b, args.device) for b in bgr]
        gr = [cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) for b in bgr]
        ge = [cv2.cvtColor(e, cv2.COLOR_BGR2GRAY) for e in enh]
        for i in range(len(frames) - 1):
            r = sift_pair(sift, bf, gr[i], gr[i + 1])
            d = sift_pair(sift, bf, ge[i], ge[i + 1])
            agg["raw"][0] += (r[0] + r[1]) / 2; agg["raw"][1] += r[2]; agg["raw"][2] += r[3]
            agg["pic_uie"][0] += (d[0] + d[1]) / 2; agg["pic_uie"][1] += d[2]; agg["pic_uie"][2] += d[3]
            npairs += 1
            print(f"{sc:28s} {i:>5} | {(r[0]+r[1])//2:6d}/{r[2]:5d}/{r[3]:5d}      | "
                  f"{(d[0]+d[1])//2:6d}/{d[2]:5d}/{d[3]:5d}")
    if npairs == 0:
        raise RuntimeError("No valid consecutive frame pairs were found")

    print(f"\n--- mean over {npairs} pairs ---")
    for k in ("raw", "pic_uie"):
        kp, gd, il = (v / npairs for v in agg[k])
        print(f"  {k:6s}: kp={kp:7.1f}  good_matches={gd:7.1f}  RANSAC_inliers={il:7.1f}")
    rg, dg = agg["raw"][1] / npairs, agg["pic_uie"][1] / npairs
    ri, di = agg["raw"][2] / npairs, agg["pic_uie"][2] / npairs
    print(f"\n  PIC-UIE vs raw: good_matches {dg/max(rg,1e-9)*100-100:+.1f}%   "
          f"inliers {di/max(ri,1e-9)*100-100:+.1f}%")


if __name__ == "__main__":
    main()
