"""Losses for PIC-UIE.

Reconstruction: (red-weighted) L1 + (1 - SSIM), where SSIM matches the evaluation
protocol (pytorch_msssim, win=11, data_range=1) so we optimize the metric we report.
Depth (train-only, optional): supervises the predicted luminance transmission `t`.
  We use a SCALE/SHIFT-INVARIANT correlation loss between -log(t) and the (relative,
  distance-like) monocular depth, so an imprecise/up-to-scale monocular depth is fine.
  This matches the design claim that depth is only a coarse global cue.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _eval_ssim, ms_ssim as _eval_ms_ssim
    _HAS_MSSSIM = True
except Exception:
    _HAS_MSSSIM = False


def _gauss_window(channels, ksize=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = (g[:, None] * g[None, :])[None, None]
    return w2d.expand(channels, 1, ksize, ksize).contiguous()


def ssim(x, y, ksize=11, sigma=1.5):
    """Mean SSIM over the batch, x,y in [0,1].

    Prefers pytorch_msssim (identical to the SMDR-IS eval protocol) so the trained
    metric matches the reported one; falls back to a local Gaussian-window SSIM.
    """
    if _HAS_MSSSIM:
        return _eval_ssim(x, y, data_range=1.0, size_average=True, win_size=ksize)
    c = x.shape[1]
    w = _gauss_window(c, ksize, sigma, x.device, x.dtype)
    pad = ksize // 2
    mux = F.conv2d(x, w, padding=pad, groups=c)
    muy = F.conv2d(y, w, padding=pad, groups=c)
    mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
    sx = F.conv2d(x * x, w, padding=pad, groups=c) - mux2
    sy = F.conv2d(y * y, w, padding=pad, groups=c) - muy2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - muxy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * muxy + c1) * (2 * sxy + c2)) / ((mux2 + muy2 + c1) * (sx + sy + c2))
    return s.mean()


def ms_ssim(x, y, ksize=11):
    """MS-SSIM (pytorch_msssim) if available, else single-scale ssim."""
    if _HAS_MSSSIM:
        return _eval_ms_ssim(x, y, data_range=1.0, size_average=True, win_size=ksize)
    return ssim(x, y, ksize)


def psnr(x, y, eps=1e-10):
    """Mean PSNR (dB) over the batch. x, y in [0,1]."""
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(1)
    return (-10.0 * torch.log10(mse + eps)).mean()


def transmission_depth_loss(t, depth):
    """Scale/shift-invariant loss tying transmission t to (distance-like) depth.

    t: [B,1,h,w] in (0,1]; depth: [B,1,H,W], LARGER = FARTHER (distance-like).
    Maximizes Pearson correlation between -log(t) (proportional to beta*distance) and depth.
    """
    d = F.interpolate(depth, t.shape[-2:], mode="bilinear", align_corners=False)
    s = -torch.log(t.clamp(1e-3, 1.0))
    s = s.flatten(1)
    d = d.flatten(1)
    s = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)
    d = (d - d.mean(1, keepdim=True)) / (d.std(1, keepdim=True) + 1e-6)
    corr = (s * d).mean(1)
    return (1.0 - corr).mean()


def lut_smoothness_loss(lut):
    """TV smoothness on the per-image chroma 2D-LUT. lut: [B,2,L,L]."""
    dh = lut[:, :, 1:, :] - lut[:, :, :-1, :]
    dw = lut[:, :, :, 1:] - lut[:, :, :, :-1]
    return (dh ** 2).mean() + (dw ** 2).mean()


def lut_monotonicity_loss(lut):
    """Encourage the chroma LUT to be order-preserving (no color inversions):
    cb-out (ch0) non-decreasing along the cb axis (width), cr-out (ch1) along the cr axis (height).
    """
    dw0 = lut[:, 0:1, :, 1:] - lut[:, 0:1, :, :-1]   # cb-out along width
    dh1 = lut[:, 1:2, 1:, :] - lut[:, 1:2, :-1, :]   # cr-out along height
    return F.relu(-dw0).mean() + F.relu(-dh1).mean()


def _gradient(x):
    """Image gradients (dx, dy) via finite differences. x: [B,C,H,W]."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def gradient_loss(pred, gt):
    """L1 on image gradients -- penalizes edge/structure mismatch (helps SSIM)."""
    pdx, pdy = _gradient(pred)
    gdx, gdy = _gradient(gt)
    return F.l1_loss(pdx, gdx) + F.l1_loss(pdy, gdy)


class PICUIELoss(nn.Module):
    """Reconstruction loss aligned with the eval protocol.

    w_red controls additional red-channel L1 supervision. w_msssim adds an MS-SSIM
    term, and w_grad adds an image-gradient consistency term.
    """

    def __init__(self, w_l1=1.0, w_ssim=0.5, w_depth=0.1, w_red=0.0, w_msssim=0.0,
                 w_grad=0.0, w_phys=0.0, w_lutsmooth=0.0, w_lutmono=0.0):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_depth = w_depth
        self.w_red = w_red
        self.w_msssim = w_msssim
        self.w_grad = w_grad
        self.w_phys = w_phys      # direct t/veil supervision weight for synthetic samples
        self.w_lutsmooth = w_lutsmooth   # TV smoothness on chroma LUT
        self.w_lutmono = w_lutmono       # monotonicity (order-preserving) on chroma LUT

    def forward(self, pred, gt, depth=None, t_gt=None, veil_gt=None, synth=None):
        out = pred["out"]
        l1 = F.l1_loss(out, gt)
        loss = self.w_l1 * l1
        logs = {"l1": float(l1.detach())}

        if self.w_red > 0:
            red = F.l1_loss(out[:, 0:1], gt[:, 0:1])   # channel 0 = R (tensors are RGB)
            loss = loss + self.w_red * red
            logs["red"] = float(red.detach())

        if self.w_ssim > 0:
            ssim_term = 1.0 - ssim(out, gt)
            loss = loss + self.w_ssim * ssim_term
            logs["ssim_loss"] = float(ssim_term.detach())

        if self.w_msssim > 0:
            ms_term = 1.0 - ms_ssim(out, gt)
            loss = loss + self.w_msssim * ms_term
            logs["msssim_loss"] = float(ms_term.detach())

        if self.w_grad > 0:
            gl = gradient_loss(out, gt)
            loss = loss + self.w_grad * gl
            logs["grad"] = float(gl.detach())

        if (self.w_lutsmooth > 0 or self.w_lutmono > 0) and "chroma" in pred:
            lut = pred["chroma"]
            if lut.dim() == 4 and lut.shape[1] == 2:   # chroma="lut": [B,2,L,L]
                if self.w_lutsmooth > 0:
                    sm = lut_smoothness_loss(lut)
                    loss = loss + self.w_lutsmooth * sm
                    logs["lut_sm"] = float(sm.detach())
                if self.w_lutmono > 0:
                    mo = lut_monotonicity_loss(lut)
                    loss = loss + self.w_lutmono * mo
                    logs["lut_mo"] = float(mo.detach())

        if synth is not None and (self.w_depth > 0 or self.w_phys > 0):
            # Mixed batch: synthetic samples get DIRECT t/veil supervision (exact targets),
            # real samples get the scale/shift-invariant transmission<->depth loss.
            t_pred, v_pred = pred["t"], pred["veil"]
            m_s = synth > 0.5
            m_r = ~m_s
            if self.w_phys > 0 and t_gt is not None and m_s.any():
                tt = F.interpolate(t_gt, t_pred.shape[-2:], mode="bilinear", align_corners=False)
                vt = F.interpolate(veil_gt, v_pred.shape[-2:], mode="bilinear", align_corners=False)
                phys = F.l1_loss(t_pred[m_s], tt[m_s]) + F.l1_loss(v_pred[m_s], vt[m_s])
                loss = loss + self.w_phys * phys
                logs["phys"] = float(phys.detach())
            if self.w_depth > 0 and depth is not None and m_r.any():
                dl = transmission_depth_loss(t_pred[m_r], depth[m_r])
                loss = loss + self.w_depth * dl
                logs["depth"] = float(dl.detach())
        elif depth is not None and self.w_depth > 0:
            dl = transmission_depth_loss(pred["t"], depth)
            loss = loss + self.w_depth * dl
            logs["depth"] = float(dl.detach())

        logs["total"] = float(loss.detach())
        return loss, logs
