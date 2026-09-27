"""PIC-UIE latency, throughput, and memory benchmark.

Measures end-to-end FP32 forward passes with batch size 1 and synchronized CUDA timing.

    python tools/latency_bench.py --ckpt weights/pic_uie.pth --img_dir /path/to/images
"""
import os, sys, glob, time, argparse
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
              y_histeq=a.get("y_histeq", False), histeq_bins=a.get("histeq_bins", 64),
              ccm=a.get("ccm", False), y_rangenorm=a.get("y_rangenorm", False),
              rangenorm_grid=a.get("rangenorm_grid", 16), colormatch=a.get("colormatch", False),
              in_norm=a.get("in_norm", False),
              global_stats=a.get("global_stats", False))
    net = PICUIE(**kw).to(device).eval()
    net.load_state_dict(s["model"])
    return net


def pick(img_dir, size, n=8):
    out = []
    for f in sorted(glob.glob(os.path.join(img_dir, "*"))):
        if not f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            continue
        try:
            if Image.open(f).size == size:
                out.append(f)
        except Exception:
            pass
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def bench(net, files, device, warmup=10, iters=50, resize=None):
    images = [Image.open(f).convert("RGB") for f in files]
    if resize is not None:
        images = [image.resize(resize, Image.BICUBIC) for image in images]
    xs = [TF.to_tensor(image).unsqueeze(0).to(device) for image in images]
    # warmup
    for i in range(warmup):
        _ = net(xs[i % len(xs)])["out"]
    if device.startswith("cuda"):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(iters):
        _ = net(xs[i % len(xs)])["out"]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    mem = torch.cuda.max_memory_allocated() / 1024**2 if device.startswith("cuda") else 0.0
    return ms, 1000.0 / ms, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/pic_uie.pth")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    net = load_model(args.ckpt, args.device)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"=== PIC-UIE latency bench (params={nparam}, FP32, bs=1, dev={args.device}) ===")
    for tag, size in [("256x256", (256, 256)), ("1080P", (1920, 1080)), ("4K", (3840, 2160))]:
        resize = None
        if size == (256, 256):
            files = pick(args.img_dir, (1920, 1080), n=8) or pick(args.img_dir, (3840, 2160), n=8)
            resize = (256, 256)
        else:
            files = pick(args.img_dir, size, n=8)
        if not files:
            print(f"  {tag:8s}: no images of size {size} found, skipped"); continue
        ms, fps, mem = bench(net, files, args.device, resize=resize)
        print(f"  {tag:8s} ({size[0]}x{size[1]}, n={len(files)}): {ms:7.2f} ms  {fps:7.1f} FPS  peak {mem:6.1f} MB")


if __name__ == "__main__":
    main()
