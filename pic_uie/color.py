"""Differentiable RGB <-> YCbCr (BT.601, full range, inputs in [0,1]).

Cb ~ (B - Y), Cr ~ (R - Y): the chroma plane carries the wavelength-selective
(red-deficit) color cast, which drifts monotonically with scene depth underwater.
Y carries the achromatic transmission/veil term (depth enters Y multiplicatively).
"""
import torch


def rgb_to_ycbcr(x):
    """x: [B,3,H,W] in [0,1] -> (y, cb, cr), each [B,1,H,W]. cb,cr centered at 0."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y)
    cr = 0.713 * (r - y)
    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr, clamp=True):
    """Inverse of rgb_to_ycbcr. y,cb,cr: [B,1,H,W] -> rgb [B,3,H,W]."""
    r = y + 1.402 * cr
    b = y + 1.773 * cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    out = torch.cat([r, g, b], dim=1)
    return out.clamp(0.0, 1.0) if clamp else out
