"""Train PIC-UIE with depth-supervised transmission and RGB-only model input.

Example:
    python tools/train.py \
        --in_dir UIEB/input --gt_dir UIEB/target --depth_dir UIEB/depth --depth_invert \
        --list_file UIEB/train_list.txt \
        --size 256 --bs 16 --epochs 700 --lr 3e-4 \
        --c 16 --chroma lut --clut_size 9 --knots 32 --chroma_ds 1 --depth_cond_chroma \
        --tone curve --refine_y 16 --lum tgain --ccm --colormatch --ema 0.999 \
        --in_norm --global_stats --w_lutsmooth 0.05 --w_lutmono 1.0 \
        --w_l1 1.0 --w_ssim 0.6 --w_msssim 0.4 --w_red 0.3 --w_depth 0.1 \
        --out_dir runs/pic_uie
"""
import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pic_uie.model import PICUIE
from pic_uie.losses import PICUIELoss, psnr
from pic_uie.dataset import PairedUIE, collate


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--depth_dir", required=True, help="depth maps used for training supervision")
    ap.add_argument("--list_file", default=None, help="split list (one filename per line)")
    ap.add_argument("--depth_invert", action="store_true",
                    help="invert depth (d<-1-d): DAv2 disparity -> distance-like")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=700)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--knots", type=int, default=32)
    ap.add_argument("--chroma_ds", type=int, default=1)
    ap.add_argument("--chroma", choices=["affine", "lut"], default="lut")
    ap.add_argument("--clut_size", type=int, default=9)
    ap.add_argument("--depth_cond_chroma", action="store_true")
    ap.add_argument("--tone", choices=["curve", "grid"], default="curve")
    ap.add_argument("--grid_d", type=int, default=8)
    ap.add_argument("--refine_y", type=int, default=16)
    ap.add_argument("--lum", choices=["tgain", "div"], default="tgain")
    ap.add_argument("--y_histeq", action="store_true")
    ap.add_argument("--histeq_bins", type=int, default=64)
    ap.add_argument("--ccm", action="store_true")
    ap.add_argument("--y_rangenorm", action="store_true")
    ap.add_argument("--rangenorm_grid", type=int, default=16)
    ap.add_argument("--colormatch", action="store_true")
    ap.add_argument("--in_norm", action="store_true",
                    help="InstanceNorm in encoder: domain-invariant spatial features (+generalization)")
    ap.add_argument("--global_stats", action="store_true",
                    help="inject raw input mean/std into global vector: per-image self-calibration")
    ap.add_argument("--w_red", type=float, default=0.3, help="extra L1 weight on the red channel")
    ap.add_argument("--w_msssim", type=float, default=0.4, help="MS-SSIM loss weight")
    ap.add_argument("--w_grad", type=float, default=0.0, help="edge/gradient L1 loss weight")
    ap.add_argument("--w_lutsmooth", type=float, default=0.0, help="TV smoothness on chroma LUT")
    ap.add_argument("--w_lutmono", type=float, default=0.0, help="monotonicity reg on chroma LUT")
    ap.add_argument("--w_l1", type=float, default=1.0)
    ap.add_argument("--w_ssim", type=float, default=0.6)
    ap.add_argument("--w_depth", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="runs/pic_uie")
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA decay (e.g. 0.999); 0 disables. No extra inference params.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    ds = PairedUIE(args.in_dir, args.gt_dir, args.depth_dir, size=args.size, augment=True,
                   list_file=args.list_file, depth_invert=args.depth_invert)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, collate_fn=collate)
    print(f"#train pairs: {len(ds)} | depth supervision: {args.depth_dir is not None}")

    net = PICUIE(c=args.c, knots=args.knots, chroma_ds=args.chroma_ds,
                chroma=args.chroma, clut_size=args.clut_size,
                depth_cond_chroma=args.depth_cond_chroma,
                tone=args.tone, grid_d=args.grid_d, refine_y=args.refine_y,
                lum=args.lum, y_histeq=args.y_histeq, histeq_bins=args.histeq_bins,
                ccm=args.ccm, y_rangenorm=args.y_rangenorm,
                rangenorm_grid=args.rangenorm_grid, colormatch=args.colormatch,
                in_norm=args.in_norm, global_stats=args.global_stats).to(device)
    crit = PICUIELoss(args.w_l1, args.w_ssim, args.w_depth, w_red=args.w_red,
                     w_msssim=args.w_msssim, w_grad=args.w_grad,
                     w_lutsmooth=args.w_lutsmooth, w_lutmono=args.w_lutmono)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"PIC-UIE params: {n_params:,}")

    ema = None
    if args.ema and args.ema > 0:
        import copy
        ema = copy.deepcopy(net.state_dict())
        print(f"EMA enabled (decay={args.ema})")

    def _save(epoch, state_dict, suffix=""):
        ckpt = os.path.join(args.out_dir, f"pic_uie_e{epoch}{suffix}.pth")
        torch.save({"model": state_dict, "args": vars(args), "epoch": epoch}, ckpt)
        torch.save({"model": state_dict, "args": vars(args), "epoch": epoch},
                   os.path.join(args.out_dir, "last.pth"))
        return ckpt

    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        running = {}
        ps = 0.0
        for batch in dl:
            x = batch["in"].to(device, non_blocking=True)
            g = batch["gt"].to(device, non_blocking=True)
            d = batch.get("depth")
            if d is not None:
                d = d.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = net(x)
                loss, logs = crit(pred, g, d)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                with torch.no_grad():
                    msd = net.state_dict()
                    for k in ema:
                        if ema[k].dtype.is_floating_point:
                            ema[k].mul_(args.ema).add_(msd[k].detach(), alpha=1 - args.ema)
                        else:
                            ema[k].copy_(msd[k])
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            with torch.no_grad():
                ps += float(psnr(pred["out"].clamp(0, 1), g))
        sched.step()

        nb = len(dl)
        msg = " | ".join(f"{k}:{v / nb:.4f}" for k, v in running.items())
        print(f"[{epoch:03d}/{args.epochs}] {msg} | train_psnr:{ps / nb:.2f} | "
              f"{time.time() - t0:.1f}s")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            state = ema if ema is not None else net.state_dict()
            ckpt = _save(epoch, state)
            print(f"  saved {ckpt}")


if __name__ == "__main__":
    main()
