"""
Differentiable metrology layer + backbone, implementing Section 4 (Eqs. 4-20)
and Section 4.7 of the new document.

Backbone (Eq. 4): f_theta(I) -> Z in R^{2xHxW} (two edge-logit channels: L, R).
Soft sub-pixel localization (Eqs. 6-7): temperature-softmax expectation.
Differentiable detrending (Eqs. 9-13): precomputed projection matrix P = I -
A(A^T A)^-1 A^T, applied as a fixed linear operator (matches the document's
formulation exactly, rather than an equivalent but differently-derived
closed-form regression).
LER/LWR (Eqs. 14-18), differentiable PSD (Eqs. 19-20).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H = 512          # profile length (scan direction)
IMG_WIDTH = 128
WELCH_NPERSEG = 128
WELCH_NOVERLAP = 64


# ---------------------------------------------------------------------------
# Eq. 9-13: precomputed linear-detrend projection matrix
# ---------------------------------------------------------------------------
def build_projection_matrix(h=H, dtype=torch.float32):
    y = np.arange(1, h + 1)
    A = np.stack([y, np.ones_like(y)], axis=1).astype(np.float64)  # (H,2)
    P = np.eye(h) - A @ np.linalg.inv(A.T @ A) @ A.T
    return torch.tensor(P, dtype=dtype)


_P_CACHE = {}


def differentiable_detrend(x, device=None):
    """x: (B,H). Returns detrended (B,H) via the precomputed projection matrix.
    device defaults to x's own device, so the cached projection matrix always
    lands where the data is (passing an explicit device that differs from x's
    would raise in the matmul below, so this default is also the only correct
    one -- the argument is kept for backwards compatibility)."""
    if device is None:
        device = x.device
    key = (x.shape[-1], x.dtype, str(device))
    if key not in _P_CACHE:
        _P_CACHE[key] = build_projection_matrix(x.shape[-1], dtype=x.dtype).to(device)
    P = _P_CACHE[key]
    return x @ P.T  # (B,H) @ (H,H) -> (B,H), P symmetric so .T is cosmetic


# ---------------------------------------------------------------------------
# Backbone: compact U-Net-style encoder-decoder (Section 4.7), 2 output channels
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """Compact U-Net (Ronneberger et al., cited as [11] in the new doc),
    scaled down for CPU-feasible training. Outputs 2 edge-logit channels (Eq. 4)."""

    def __init__(self, base=12):
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 2, 1)  # 2 channels: left, right

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)  # (B,2,H,W)


# ---------------------------------------------------------------------------
# Eq. 6-7: soft sub-pixel localization
# ---------------------------------------------------------------------------
def soft_localize(logits_channel, x0, window_half=15, tau=0.1):
    """logits_channel: (B,H,W). Eq. 6-7 with temperature tau (analogous to 1/beta
    in the original single-edge formulation). Returns x_hat (B,H)."""
    B, Hh, W = logits_channel.shape
    x0i = int(round(x0))
    lo = max(0, x0i - window_half)
    hi = min(W, x0i + window_half + 1)
    win_logits = logits_channel[:, :, lo:hi] / tau
    q = F.softmax(win_logits, dim=-1)
    xs = torch.arange(lo, hi, device=logits_channel.device, dtype=logits_channel.dtype)
    x_hat = (q * xs.view(1, 1, -1)).sum(dim=-1)
    return x_hat, q, lo, hi


def heatmap_target(x_gt, lo, hi, sigma_t=1.0):
    xs = torch.arange(lo, hi, device=x_gt.device, dtype=x_gt.dtype).view(1, 1, -1)
    diff = xs - x_gt.unsqueeze(-1)
    t = torch.exp(-0.5 * (diff / sigma_t) ** 2)
    return t / t.sum(dim=-1, keepdim=True).clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Eq. 14-18: LER/LWR from detrended signals
# ---------------------------------------------------------------------------
def rms(x_detrended, eps=1e-8):
    return torch.sqrt((x_detrended ** 2).mean(dim=-1) + eps)


# ---------------------------------------------------------------------------
# Eq. 19-20: differentiable Welch-style PSD
# ---------------------------------------------------------------------------
def differentiable_psd(x_detrended, nperseg=WELCH_NPERSEG, noverlap=WELCH_NOVERLAP, fs=1.0):
    step = nperseg - noverlap
    window = torch.hann_window(nperseg, periodic=False, device=x_detrended.device, dtype=x_detrended.dtype)
    segments = x_detrended.unfold(-1, nperseg, step)
    segments = segments - segments.mean(dim=-1, keepdim=True)
    windowed = segments * window
    c_h = (window ** 2).sum()
    fft_vals = torch.fft.fft(windowed, dim=-1)
    psd_seg = (fft_vals.abs() ** 2) / (fs * c_h)
    psd = psd_seg.mean(dim=-2)
    freqs = torch.fft.fftfreq(nperseg, d=1.0 / fs)
    return freqs, psd


def log_psd_loss(freqs, psd_pred, psd_gt, n=H, eps=1e-6):
    mask = (freqs.abs() >= 1.0 / n) & (freqs.abs() <= 0.25) & (freqs != 0)
    mask = mask.to(psd_pred.device)
    lp = torch.log(psd_pred[:, mask] + eps)
    lg = torch.log(psd_gt[:, mask] + eps)
    return (lp - lg).abs().mean()
