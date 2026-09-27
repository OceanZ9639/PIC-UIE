"""PIC-UIE model in YCbCr space with RGB input and output.

TinyEncoder runs at a fixed resolution and predicts transforms that are applied at
the input resolution. The Y branch performs transmission-guided restoration, tone
mapping, and optional residual refinement. The Cb/Cr branch applies either an affine
transform or an image-adaptive 2D lookup table. Monocular depth supervises the
transmission estimate during training but is not a model input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import rgb_to_ycbcr, ycbcr_to_rgb

CB_RANGE = 0.5
CR_RANGE = 0.5


class TinyEncoder(nn.Module):
    def __init__(self, c=16, knots=16, in_res=256, chroma="affine", clut_size=9,
                 tone="curve", grid_d=8, ccm=False, colormatch=False,
                 in_norm=False, global_stats=False):
        super().__init__()
        self.in_res = in_res
        self.knots = knots
        self.chroma = chroma
        self.clut_size = clut_size
        self.tone = tone
        self.grid_d = grid_d
        self.ccm = ccm
        self.colormatch = colormatch
        # Optional normalization and raw-statistics conditioning.
        self.in_norm = in_norm
        self.global_stats = global_stats
        in_ch = 3
        if in_norm:
            self.body = nn.Sequential(
                nn.Conv2d(in_ch, c, 3, 2, 1), nn.InstanceNorm2d(c, affine=True), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, 2, 1), nn.InstanceNorm2d(c, affine=True), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, 2, 1), nn.InstanceNorm2d(c, affine=True), nn.ReLU(inplace=True),
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(in_ch, c, 3, 2, 1), nn.ReLU(inplace=True),   # /2
                nn.Conv2d(c, c, 3, 2, 1), nn.ReLU(inplace=True),   # /4
                nn.Conv2d(c, c, 3, 2, 1), nn.ReLU(inplace=True),   # /8
            )
        if global_stats:
            self.stat_fc = nn.Linear(6, c)   # raw RGB mean(3)+std(3) -> global feature
        self.head_t = nn.Conv2d(c, 1, 1)
        self.head_vl = nn.Conv2d(c, 1, 1)
        self.head_gain = nn.Conv2d(c, 1, 1)   # transmission-guided luminance gain coeff (a)
        if tone == "curve":
            self.head_crv = nn.Linear(c, knots)
        elif tone == "grid":
            self.head_grid = nn.Conv2d(c, 2 * grid_d, 1)
        else:
            raise ValueError(f"tone must be 'curve' or 'grid', got {tone}")
        if chroma == "affine":
            self.head_aff = nn.Conv2d(c, 6, 1)
        elif chroma == "lut":
            L = clut_size
            self.head_clut = nn.Linear(c, 2 * L * L)
            self.register_buffer("clut_id", self._identity_clut(L), persistent=False)
        else:
            raise ValueError(f"chroma must be 'affine' or 'lut', got {chroma}")
        if ccm:
            self.head_ccm = nn.Linear(c, 12)   # per-image 3x3 + bias cross-channel affine on YCbCr
        if colormatch:
            # Learned target mean, lower-triangular factor, and blend coefficient.
            self.head_cmatch = nn.Linear(c, 10)
        self._init_identity()

    @staticmethod
    def _identity_clut(L):
        ax = torch.linspace(-1.0, 1.0, L)
        cb_grid = ax.view(1, 1, L).expand(1, L, L)
        cr_grid = ax.view(1, L, 1).expand(1, L, L)
        return torch.stack([cb_grid, cr_grid], dim=1)

    def _init_identity(self):
        nn.init.zeros_(self.head_t.weight)
        nn.init.zeros_(self.head_vl.weight)
        nn.init.zeros_(self.head_gain.weight)
        nn.init.constant_(self.head_t.bias, 4.0)
        nn.init.constant_(self.head_vl.bias, -4.0)
        nn.init.constant_(self.head_gain.bias, -4.0)   # softplus(-4)~0.018 -> gain s~1 (identity)
        if self.tone == "curve":
            nn.init.zeros_(self.head_crv.weight)
            nn.init.zeros_(self.head_crv.bias)
        else:
            nn.init.zeros_(self.head_grid.weight)
            b = torch.zeros(2 * self.grid_d)
            b[:self.grid_d] = 1.0
            self.head_grid.bias.data = b
        if self.chroma == "affine":
            nn.init.zeros_(self.head_aff.weight)
            self.head_aff.bias.data = torch.tensor([1., 0., 0., 0., 1., 0.])
        else:
            nn.init.zeros_(self.head_clut.weight)
            nn.init.zeros_(self.head_clut.bias)
        if self.ccm:
            nn.init.zeros_(self.head_ccm.weight)
            self.head_ccm.bias.data = torch.tensor(
                [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.])   # identity 3x3 + zero bias
        if self.colormatch:
            nn.init.zeros_(self.head_cmatch.weight)
            # mu_t(3)=0, L_t lower-tri(6)= identity [1,0,1,0,0,1], blend logit = -4 (~0 -> identity start)
            self.head_cmatch.bias.data = torch.tensor([0., 0., 0., 1., 0., 1., 0., 0., 1., -4.])
        if self.global_stats:
            nn.init.zeros_(self.stat_fc.weight)   # start as no-op (g unchanged)
            nn.init.zeros_(self.stat_fc.bias)

    def forward(self, rgb):
        lr = F.interpolate(rgb, (self.in_res, self.in_res), mode="bilinear", align_corners=False)
        f = self.body(lr)
        t = torch.sigmoid(self.head_t(f)).clamp(0.05, 1.0)
        veil = torch.sigmoid(self.head_vl(f))
        gain = F.softplus(self.head_gain(f))             # >=0 ; transmission-guided gain coeff
        g = F.adaptive_avg_pool2d(f, 1).flatten(1)
        if self.global_stats:
            rgb = lr[:, :3]
            s = torch.cat([rgb.mean(dim=(2, 3)),
                           rgb.std(dim=(2, 3)) + 1e-5], dim=1)   # [B,6] raw color stats
            g = g + self.stat_fc(s)

        if self.tone == "curve":
            crv = F.softplus(self.head_crv(g)) + 1e-3
            crv = torch.cumsum(crv, dim=1)
            crv = crv / crv[:, -1:]
            tone_param = F.pad(crv, (1, 0), value=0.0)
        else:
            tone_param = self.head_grid(f)

        if self.chroma == "affine":
            chroma = self.head_aff(f)
        else:
            L = self.clut_size
            chroma = self.clut_id + self.head_clut(g).view(-1, 2, L, L)
        ccm = self.head_ccm(g) if self.ccm else None
        cmatch = self.head_cmatch(g) if self.colormatch else None
        return t, veil, gain, chroma, tone_param, ccm, cmatch


class PICUIE(nn.Module):
    def __init__(self, c=16, knots=16, in_res=256, chroma_ds=4,
                 chroma="affine", clut_size=9, depth_cond_chroma=False,
                 tone="curve", grid_d=8, refine_y=1, lum="tgain",
                 y_histeq=False, histeq_bins=64, ccm=False, y_rangenorm=False,
                 rangenorm_grid=16, colormatch=False,
                 in_norm=False, global_stats=False):
        super().__init__()
        self.enc = TinyEncoder(c, knots, in_res, chroma=chroma, clut_size=clut_size,
                               tone=tone, grid_d=grid_d, ccm=ccm, colormatch=colormatch,
                               in_norm=in_norm, global_stats=global_stats)
        self.ccm = ccm
        self.colormatch = colormatch
        self.chroma_ds = chroma_ds
        self.chroma = chroma
        self.tone = tone
        self.grid_d = grid_d
        self.refine_y = refine_y
        self.y_histeq = y_histeq
        self.histeq_bins = histeq_bins
        self.y_rangenorm = y_rangenorm    # adaptive luminance range normalization (path 1)
        self.rangenorm_grid = rangenorm_grid
        # luminance restoration: "tgain" = transmission-guided de-scatter + bounded gain
        # (Y1=Y-(1-t)B ; s=1+a(1-t) ; Y'=Y1*s), "div" = classic de-veil (Y-B(1-t))/t.
        self.lum = lum
        self.depth_cond_chroma = depth_cond_chroma
        if depth_cond_chroma:
            self.chroma_beta = nn.Parameter(torch.tensor(1.0))
            self.chroma_gamma = nn.Parameter(torch.tensor(0.0))
        # Y-only residual refine (stays in YCbCr; structure/detail lives in luminance).
        #   refine_y == 0 : disabled
        #   refine_y == 1 : single 3x3 conv (minimal)
        #   refine_y  > 1 : a small Y->w->w->Y block for more structural capacity (SSIM)
        if refine_y == 1:
            self.refine = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
            nn.init.zeros_(self.refine.weight)
        elif refine_y and refine_y > 1:
            w = refine_y
            self.refine = nn.Sequential(
                nn.Conv2d(1, w, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, 1, 1, groups=w), nn.ReLU(inplace=True),
                nn.Conv2d(w, 1, 3, 1, 1),
            )
            nn.init.zeros_(self.refine[-1].weight)     # start as no-op residual
            nn.init.zeros_(self.refine[-1].bias)
        else:
            self.refine = None

    @staticmethod
    def _apply_curve(y, lut):
        B, _, H, W = y.shape
        K = lut.shape[1]
        inp = lut.view(B, 1, 1, K)
        gx = (2 * y - 1).clamp(-1, 1).squeeze(1).unsqueeze(-1)
        grid = torch.cat([gx, torch.zeros_like(gx)], dim=-1)
        return F.grid_sample(inp, grid, mode="bilinear", align_corners=True)

    @staticmethod
    def _apply_grid(y, coeffs, grid_d):
        """Bilateral-grid local affine on Y. y:[B,1,H,W]; coeffs:[B,2*grid_d,gh,gw]."""
        B, _, H, W = y.shape
        gh, gw = coeffs.shape[-2:]
        G = coeffs.view(B, 2, grid_d, gh, gw)
        dev, dt = y.device, y.dtype
        ys = torch.linspace(-1, 1, H, device=dev, dtype=dt)
        xs = torch.linspace(-1, 1, W, device=dev, dtype=dt)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx = gx.expand(B, H, W)
        gy = gy.expand(B, H, W)
        gz = (2 * y - 1).clamp(-1, 1).squeeze(1)
        samp = torch.stack([gx, gy, gz], dim=-1).unsqueeze(1)
        ab = F.grid_sample(G, samp, mode="bilinear", align_corners=True, padding_mode="border")
        a = ab[:, 0:1, 0]
        b = ab[:, 1:2, 0]
        return (a * y + b).clamp(0.0, 1.0)

    @staticmethod
    def _cov_match(ycc, cmatch, eps=1e-4):
        """Whiten YCbCr statistics and blend toward a learned target covariance."""
        B, C, H, W = ycc.shape
        mu_t = cmatch[:, 0:3].view(B, 3, 1)
        l = cmatch[:, 3:9]
        L_t = ycc.new_zeros(B, 3, 3)
        L_t[:, 0, 0] = l[:, 0]; L_t[:, 1, 0] = l[:, 1]; L_t[:, 1, 1] = l[:, 2]
        L_t[:, 2, 0] = l[:, 3]; L_t[:, 2, 1] = l[:, 4]; L_t[:, 2, 2] = l[:, 5]
        alpha = torch.sigmoid(cmatch[:, 9]).view(B, 1, 1)
        with torch.cuda.amp.autocast(enabled=False):
            x = ycc.reshape(B, C, -1).float()
            mu = x.mean(dim=2, keepdim=True)
            xc = x - mu
            cov = torch.bmm(xc, xc.transpose(1, 2)) / (x.shape[2] - 1)
            cov = cov + eps * torch.eye(C, device=x.device).unsqueeze(0)
            evals, evecs = torch.linalg.eigh(cov)
            inv_sqrt = torch.bmm(evecs, torch.bmm(
                torch.diag_embed(evals.clamp_min(eps).rsqrt()), evecs.transpose(1, 2)))
            x_w = torch.bmm(inv_sqrt, xc)
            x_cm = torch.bmm(L_t.float(), x_w) + mu_t.float()
            out = (1 - alpha.float()) * x + alpha.float() * x_cm
        return out.reshape(B, C, H, W).to(ycc.dtype)

    @staticmethod
    def _range_norm(y, grid=16):
        """Apply a robust per-image min-max stretch to luminance."""
        d = F.avg_pool2d(y, grid, grid) if min(y.shape[-2:]) >= grid else y
        d = d.flatten(2)
        mi = d.min(2, keepdim=True)[0].unsqueeze(-1)
        ma = d.max(2, keepdim=True)[0].unsqueeze(-1)
        return ((y - mi) / (ma - mi + 1e-3)).clamp(0.0, 1.0)

    @staticmethod
    def _hist_eq(y, bins=64, sigma=0.05, strength=1.0):
        """Apply differentiable CDF-based histogram equalization to luminance."""
        B = y.shape[0]
        dev, dt = y.device, y.dtype
        centers = torch.linspace(0.0, 1.0, bins, device=dev, dtype=dt).view(1, bins, 1)
        ys = F.adaptive_avg_pool2d(y, 64).reshape(B, 1, -1)            # [B,1,Npix]
        # soft assignment of each pixel to bins (Gaussian kernel) -> soft histogram
        w = torch.exp(-((ys - centers) ** 2) / (2 * sigma * sigma))    # [B,bins,Npix]
        hist = w.sum(dim=2) + 1e-6                                     # [B,bins]
        cdf = torch.cumsum(hist, dim=1)
        cdf = (cdf - cdf[:, :1]) / (cdf[:, -1:] - cdf[:, :1] + 1e-6)   # [B,bins] in [0,1]
        # apply CDF as a monotone LUT to full-res Y via grid_sample
        lut = cdf.view(B, 1, 1, bins)
        gx = (2 * y - 1).clamp(-1, 1).squeeze(1).unsqueeze(-1)
        grid = torch.cat([gx, torch.zeros_like(gx)], dim=-1)
        y_eq = F.grid_sample(lut, grid, mode="bilinear", align_corners=True)
        return (strength * y_eq + (1.0 - strength) * y).clamp(0.0, 1.0)

    @staticmethod
    def _apply_chroma_lut(cb, cr, clut):
        cbn = (cb / CB_RANGE).clamp(-1, 1)
        crn = (cr / CR_RANGE).clamp(-1, 1)
        grid = torch.cat([cbn, crn], dim=1).permute(0, 2, 3, 1)
        out = F.grid_sample(clut, grid, mode="bilinear", align_corners=True, padding_mode="border")
        return out[:, 0:1] * CB_RANGE, out[:, 1:2] * CR_RANGE

    def forward(self, rgb):
        H, W = rgb.shape[-2:]
        y, cb, cr = rgb_to_ycbcr(rgb)
        t, veil, gain, chroma, tone_param, ccm, cmatch = self.enc(rgb)

        def up(m, size):
            return F.interpolate(m, size, mode="bilinear", align_corners=False)

        # ---- Y branch: luminance restoration -> tone -> Y-only residual refine ----
        tf = up(t, (H, W))
        vlf = up(veil, (H, W))
        if self.lum == "tgain":
            # transmission-guided: de-scatter (subtract veil) then bounded depth-driven gain.
            af = up(gain, (H, W))
            y1 = y - (1.0 - tf) * vlf
            s = 1.0 + af * (1.0 - tf)
            yj = (y1 * s).clamp(0.0, 1.0)
        else:  # "div": classic physical de-veil (can amplify dark/far regions)
            yj = ((y - vlf * (1.0 - tf)) / tf).clamp(0.0, 1.0)
        if self.y_histeq:
            yj = self._hist_eq(yj, bins=self.histeq_bins)
        if self.tone == "curve":
            yj = self._apply_curve(yj, tone_param)
        else:
            yj = self._apply_grid(yj, tone_param, self.grid_d)
        if self.refine is not None:
            yj = (yj + self.refine(yj)).clamp(0.0, 1.0)
        if self.y_rangenorm:
            yj = self._range_norm(yj, self.rangenorm_grid)

        # ---- Chroma branch: low-res correction, then upsample ----
        hC = max(1, H // self.chroma_ds)
        wC = max(1, W // self.chroma_ds)
        cbl = up(cb, (hC, wC))
        crl = up(cr, (hC, wC))
        if self.chroma == "affine":
            a = up(chroma, (hC, wC))
            cb2 = a[:, 0:1] * cbl + a[:, 1:2] * crl + a[:, 2:3]
            cr2 = a[:, 3:4] * cbl + a[:, 4:5] * crl + a[:, 5:6]
        else:
            cb2, cr2 = self._apply_chroma_lut(cbl, crl, chroma)
        cb2 = up(cb2, (H, W))
        cr2 = up(cr2, (H, W))

        if self.depth_cond_chroma:
            w = (self.chroma_beta + self.chroma_gamma * (1.0 - tf)).clamp(0.0, 1.0)
            cb2 = cb + w * (cb2 - cb)
            cr2 = cr + w * (cr2 - cr)

        # ---- cross-channel color-correction matrix on (Y,Cb,Cr), in YCbCr ----
        if self.ccm:
            ych = torch.cat([yj, cb2, cr2], dim=1)             # [B,3,H,W]
            M = ccm[:, :9].view(-1, 3, 3)
            b = ccm[:, 9:12].view(-1, 3, 1, 1)
            ych = torch.einsum("bij,bjhw->bihw", M, ych) + b
            yj, cb2, cr2 = ych[:, 0:1], ych[:, 1:2], ych[:, 2:3]
            yj = yj.clamp(0.0, 1.0)

        # Data-adaptive covariance matching in YCbCr.
        if self.colormatch:
            ych = torch.cat([yj, cb2, cr2], dim=1)
            ych = self._cov_match(ych, cmatch)
            yj, cb2, cr2 = ych[:, 0:1].clamp(0.0, 1.0), ych[:, 1:2], ych[:, 2:3]

        out = ycbcr_to_rgb(yj, cb2, cr2)   # RGB only at output I/O
        return {"out": out, "t": t, "veil": veil, "chroma": chroma, "tone": tone_param}
