"""Run PIC-UIE RGB inference and optionally report paired-image metrics."""
import os
import sys
import glob
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pic_uie.model import PICUIE
from pic_uie.losses import psnr, ssim

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--list_file", default=None, help="split list (one filename per line)")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--size", type=int, default=None, help="resize (None = native resolution)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    state = torch.load(args.ckpt, map_location="cpu")
    margs = state.get("args", {})
    def _ycc_kwargs():
        return dict(c=margs.get("c", 16), knots=margs.get("knots", 16),
                    chroma_ds=margs.get("chroma_ds", 4), chroma=margs.get("chroma", "affine"),
                    clut_size=margs.get("clut_size", 9),
                    depth_cond_chroma=margs.get("depth_cond_chroma", False),
                    tone=margs.get("tone", "curve"), grid_d=margs.get("grid_d", 8),
                    refine_y=margs.get("refine_y", 1), lum=margs.get("lum", "tgain"),
                    y_histeq=margs.get("y_histeq", False), histeq_bins=margs.get("histeq_bins", 64),
                    ccm=margs.get("ccm", False), y_rangenorm=margs.get("y_rangenorm", False),
                    rangenorm_grid=margs.get("rangenorm_grid", 16),
                    colormatch=margs.get("colormatch", False),
                    in_norm=margs.get("in_norm", False),
                    global_stats=margs.get("global_stats", False))

    arch = margs.get("arch", "lut")
    if arch != "lut":
        raise ValueError(f"Unsupported checkpoint architecture: {arch!r}")
    net = PICUIE(**_ycc_kwargs()).to(device).eval()
    net.load_state_dict(state["model"])

    if args.list_file:
        with open(args.list_file) as f:
            paths = [os.path.join(args.in_dir, ln.strip()) for ln in f if ln.strip()]
    else:
        paths = sorted(p for p in glob.glob(os.path.join(args.in_dir, "*")) if p.lower().endswith(_EXTS))
    ps, ss, n = 0.0, 0.0, 0
    for p in paths:
        name = os.path.basename(p)
        img = Image.open(p).convert("RGB")
        if args.size:
            img = img.resize((args.size, args.size), Image.BICUBIC)
        x = TF.to_tensor(img).unsqueeze(0).to(device)
        out = net(x)["out"]
        TF.to_pil_image(out.squeeze(0).clamp(0, 1).cpu()).save(os.path.join(args.out_dir, name))

        if args.gt_dir:
            gp = os.path.join(args.gt_dir, name)
            if os.path.exists(gp):
                g = Image.open(gp).convert("RGB").resize(img.size, Image.BICUBIC)
                g = TF.to_tensor(g).unsqueeze(0).to(device)
                ps += float(psnr(out.clamp(0, 1), g))
                ss += float(ssim(out.clamp(0, 1), g))
                n += 1
    print(f"saved {len(paths)} images to {args.out_dir}")
    if n:
        print(f"PSNR: {ps / n:.3f} | SSIM: {ss / n:.4f}  (over {n} pairs)")


if __name__ == "__main__":
    main()
