"""Paired underwater dataset (UIEB / LSUI style): input/ and GT/ with matching filenames.

Supports a split list file (one filename per line, e.g. UIEB train_list.txt / test_list.txt).

Optional `depth_dir` holds precomputed monocular depth (.png grayscale or .npy), used ONLY
for the train-time transmission supervision. The transmission loss expects DISTANCE-LIKE
depth (farther = larger). Depth-Anything-V2 maps are disparity-like (nearer = brighter),
so pass `depth_invert=True` to convert them (d <- 1 - d).
"""
import os
import glob
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class PairedUIE(torch.utils.data.Dataset):
    def __init__(self, in_dir, gt_dir, depth_dir=None, size=256, augment=True,
                 list_file=None, depth_invert=False):
        if list_file is not None:
            with open(list_file) as f:
                names = [ln.strip() for ln in f if ln.strip()]
            self.in_paths = [os.path.join(in_dir, n) for n in names]
        else:
            self.in_paths = sorted(
                p for p in glob.glob(os.path.join(in_dir, "*")) if p.lower().endswith(_EXTS)
            )
        if not self.in_paths:
            raise FileNotFoundError(f"No input images for {in_dir} (list_file={list_file})")
        self.gt_dir = gt_dir
        self.depth_dir = depth_dir
        self.size = size
        self.augment = augment
        self.depth_invert = depth_invert

    def __len__(self):
        return len(self.in_paths)

    def _load_rgb(self, path):
        img = Image.open(path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return TF.to_tensor(img)

    def _load_depth(self, name):
        base = os.path.splitext(name)[0]
        cands = [os.path.join(self.depth_dir, name),
                 os.path.join(self.depth_dir, base + ".png"),
                 os.path.join(self.depth_dir, base + ".npy")]
        dp = next((c for c in cands if os.path.exists(c)), None)
        if dp is None:
            raise FileNotFoundError(f"No depth map found for {name} in {self.depth_dir}")
        if dp.endswith(".npy"):
            d = np.load(dp).astype(np.float32)
            if d.max() > 1.5:
                d = d / 255.0
        else:
            d = np.asarray(Image.open(dp).convert("L"), dtype=np.float32) / 255.0
        if self.depth_invert:               # DAv2 disparity (near=bright) -> distance-like
            d = 1.0 - d
        d = torch.from_numpy(d)[None]
        return TF.resize(d, [self.size, self.size], antialias=True)

    def __getitem__(self, idx):
        ip = self.in_paths[idx]
        name = os.path.basename(ip)
        x = self._load_rgb(ip)
        g = self._load_rgb(os.path.join(self.gt_dir, name))

        depth = self._load_depth(name) if self.depth_dir is not None else None

        if self.augment:
            if torch.rand(1).item() < 0.5:
                x, g = TF.hflip(x), TF.hflip(g)
                if depth is not None:
                    depth = TF.hflip(depth)
            if torch.rand(1).item() < 0.5:
                x, g = TF.vflip(x), TF.vflip(g)
                if depth is not None:
                    depth = TF.vflip(depth)
            k = int(torch.randint(0, 4, (1,)).item())   # random 0/90/180/270 rotation
            if k:
                x, g = torch.rot90(x, k, [1, 2]), torch.rot90(g, k, [1, 2])
                if depth is not None:
                    depth = torch.rot90(depth, k, [1, 2])

        item = {"in": x, "gt": g, "name": name}
        if depth is not None:
            item["depth"] = depth
        return item


def collate(batch):
    out = {
        "in": torch.stack([b["in"] for b in batch]),
        "gt": torch.stack([b["gt"] for b in batch]),
        "name": [b["name"] for b in batch],
    }
    if all("depth" in b for b in batch):
        out["depth"] = torch.stack([b["depth"] for b in batch])
    return out
