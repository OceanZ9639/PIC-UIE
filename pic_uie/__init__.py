"""PIC-UIE underwater image enhancement components.

Factorizes the underwater degradation along the YCbCr axes:
  * Y  (luminance) -> depth-driven transmission/veil de-haze + monotone tone curve (+ tiny refine)
  * Cb,Cr (chroma) -> low-resolution 2D affine / LUT cast correction (chroma is low-frequency)

A tiny low-res encoder predicts the *transforms* (transmission map, veil, chroma corrector,
tone curve); they are applied cheaply at full resolution ("predict-transform, not pixels").

Monocular depth supervises luminance transmission during training. The model itself
always receives RGB, so inference is depth-free.
"""
from .model import PICUIE, TinyEncoder
from .color import rgb_to_ycbcr, ycbcr_to_rgb
from .losses import PICUIELoss, ssim, psnr, transmission_depth_loss, gradient_loss

__all__ = [
    "PICUIE", "TinyEncoder",
    "rgb_to_ycbcr", "ycbcr_to_rgb",
    "PICUIELoss", "ssim", "psnr", "transmission_depth_loss", "gradient_loss",
]
