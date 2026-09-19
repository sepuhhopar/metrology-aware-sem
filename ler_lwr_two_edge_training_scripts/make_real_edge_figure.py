"""
Qualitative figure on REAL SEM data: original image crop, classical
threshold+sub-pixel edge detection, and the learned (calibrated) model's edges.

IMPORTANT SCOPE NOTE. This figure is qualitative only. The real dataset has no
line-edge-roughness ground truth, and at 14.65 nm/px its sub-pixel edge-location
noise floor (~18-33 nm) is roughly an order of magnitude above the roughness of
a 128 nm-node line (~2-3 nm). No accuracy or LER/LWR claim can be made from it.
What it does show is that a model trained purely on calibrated synthetic data
produces sensible boundaries on real metal tracks it has never seen.

Transfer procedure (documented because it matters for interpretation):
  1. Locate a long straight horizontal metal track from the SVG label bounding
     box (the polygons are too decimated to be edge ground truth, but they are
     fine for locating a track).
  2. Crop, then transpose so the track runs vertically, matching the generator's
     convention.
  3. Rescale so the track width matches the synthetic nominal width w0 = 40 px.
     This is the key step: the real PSF is 3.76 px at a 22.7 px track width
     (ratio 0.166); after rescaling it becomes ~6.6 px at width 40, which is
     exactly the ratio-matched PSF the calibrated model was trained with. The
     rescale therefore aligns the real data with the calibrated training
     conditions rather than distorting it.
  4. Normalize intensity to [0,1] by background/foreground percentiles, matching
     the generator's fg=1 / bg=0 convention.

Usage:
    python3 make_real_edge_figure.py --data-dir /path/to/dataset \
        --calibration calibration_m2.json --out real_edges.png
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

from classical_two_edge import threshold_detector
from model2 import soft_localize, differentiable_detrend, rms
from train2 import (X0, W0, build_paired_dataset, train_one_variant, DEVICE,
                     set_calibration, load_calibration)

Image.MAX_IMAGE_PIXELS = None
H_OUT, W_OUT = 512, 128
NM_PER_PX = 14.65

SIGMAS = [1.5, 3.0]
XIS = [8.0, 20.0]
ALPHAS = [0.3, 0.7]

C_CLASSICAL = "#ff7f0e"
C_LEARNED = "#d62728"


def find_tracks(svg_path, min_len=600, max_thick=25):
    with open(svg_path) as f:
        s = f.read()
    out = []
    for d in re.findall(r'<path d="([^"]+)" fill="lime"', s):
        pts = np.array([tuple(map(int, m)) for m in re.findall(r"[ML](\d+),(\d+)", d)])
        if len(pts) < 4:
            continue
        if pts[:, 0].ptp() > min_len and pts[:, 1].ptp() < max_thick:
            out.append((int(pts[:, 0].min()), int(pts[:, 0].max()),
                        int(round(pts[:, 1].mean()))))
    seen, uniq = set(), []
    for t in out:
        key = (t[0] // 32, t[2] // 8)
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


def _runs(mask):
    """Contiguous True runs as (start, stop) pairs."""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def central_run(prof, target=None):
    """The one bright band nearest `target` (default: profile centre).

    This matters because the layout is dense -- neighbouring tracks sit about
    24 px away, so a global half-max threshold merges several tracks into one
    apparent structure and every downstream position is then wrong."""
    lo, hi = np.percentile(prof, 5), np.percentile(prof, 95)
    rr = _runs(prof >= 0.5 * (lo + hi))
    if not rr:
        return None
    t = 0.5 * len(prof) if target is None else target
    return min(rr, key=lambda r: abs(0.5 * (r[0] + r[1]) - t))


def measure_width(strip, target=None):
    """Width of the central track only, not of every bright band in the crop."""
    r = central_run(strip.mean(axis=1), target)
    return float(r[1] - r[0]) if r is not None else np.nan


def prepare_crop(img, x0, x1, yc):
    """Real horizontal track -> (512,128) vertical-line image in [0,1],
    rescaled so the track width matches w0 and the track is centred at X0."""
    half = 40
    r0, r1 = max(0, yc - half), min(img.shape[0], yc + half)
    strip = img[r0:r1, x0 + 20:x1 - 20]                    # (across, along)
    w_px = measure_width(strip, target=yc - r0)
    if not np.isfinite(w_px) or w_px < 4:
        return None, None, None
    scale = W0 / w_px                                       # -> nominal 40 px

    need_along = int(np.ceil(H_OUT / scale))
    need_across = int(np.ceil(W_OUT / scale))
    if strip.shape[1] < need_along:
        return None, None, None

    r = central_run(strip.mean(axis=1), target=yc - r0)
    c = int(round(0.5 * (r[0] + r[1])))
    a0 = int(np.clip(c - need_across // 2, 0, max(0, strip.shape[0] - need_across)))
    sub = strip[a0:a0 + need_across, :need_along]
    if sub.shape[0] < need_across:
        return None, None, None

    big = zoom(sub, scale, order=1)                         # (across*, along*)
    big = big[:W_OUT, :H_OUT]
    if big.shape != (W_OUT, H_OUT):
        return None, None, None
    out = big.T                                             # (512,128), vertical line

    bg, fg = np.percentile(out, 5), np.percentile(out, 95)
    out = np.clip((out - bg) / max(fg - bg, 1e-6), 0.0, 1.0)

    # Locate the target track in the finished crop and report its actual edge
    # positions. The layout is too dense to place an isolated line at the
    # generator's nominal geometry, so instead of forcing the crop to match
    # (x0=64, w0=40) we hand the measured positions to both detectors -- which
    # is how the method would be used on a real tool, where the nominal edge
    # location comes from design intent rather than from a fixed constant.
    rr = central_run(out.mean(axis=0))
    if rr is None:
        return None, None, None
    nominal = (float(rr[0]), float(rr[1]))
    return out, w_px, nominal


@torch.no_grad()
def learned_edges(model, img, nominal):
    t = torch.from_numpy(img.astype(np.float32))[None, None].to(
        next(model.parameters()).device)
    logits = model(t)
    xl, _, _, _ = soft_localize(logits[:, 0], nominal[0], 15, 0.05)
    xr, _, _, _ = soft_localize(logits[:, 1], nominal[1], 15, 0.05)
    return xl[0].cpu().numpy(), xr[0].cpu().numpy()


def roughness(xl, xr):
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))[None, ...]
    s = lambda a: float(rms(differentiable_detrend(t(a)))[0].item())
    return s(xl), s(xr), s(np.asarray(xr) - np.asarray(xl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--layer", default="m2")
    ap.add_argument("--calibration", default="calibration_m2.json")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-per-config", type=int, default=15)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.3, 0.6, -0.3])
    ap.add_argument("--n-examples", type=int, default=2)
    ap.add_argument("--scan-images", type=int, default=12)
    ap.add_argument("--out", default="real_edges.png")
    args = ap.parse_args()

    cal = load_calibration(args.calibration)
    set_calibration(cal)
    configs = [(s, x, a, r) for s in SIGMAS for x in XIS for a in ALPHAS
               for r in args.rhos]
    print(f"Device {DEVICE}; training '{args.variant}' on calibrated synthetic data")
    items = build_paired_dataset(configs, args.train_per_config, salt=700)
    model = train_one_variant(args.variant, items, epochs=args.epochs, seed=0)
    model.eval()

    sem_dir = os.path.join(args.data_dir, "sems", args.layer)
    lab_dir = os.path.join(args.data_dir, "labels", args.layer)
    sems = sorted(f for f in os.listdir(sem_dir) if f.endswith(".png"))

    examples = []
    for fn in sems[:args.scan_images]:
        stem = re.sub(r"\D", "", fn)
        svg = os.path.join(lab_dir, f"label{stem}.svg")
        if not os.path.exists(svg):
            continue
        img = np.asarray(Image.open(os.path.join(sem_dir, fn)).convert("L")).astype(np.float64)
        for (x0, x1, yc) in find_tracks(svg):
            crop, w_px, nominal = prepare_crop(img, x0, x1, yc)
            if crop is None:
                continue
            examples.append((fn, crop, w_px, nominal))
            break
        if len(examples) >= args.n_examples:
            break

    if len(examples) < args.n_examples:
        raise SystemExit(f"only found {len(examples)} usable tracks")

    n = len(examples)
    fig, axs = plt.subplots(n, 4, figsize=(15, 5.0 * n),
                            gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    if n == 1:
        axs = axs[None, :]
    y = np.arange(H_OUT)

    for i, (fn, crop, w_px, nominal) in enumerate(examples):
        cl = threshold_detector(crop, nominal[0], subpixel=True, falling=False)
        cr = threshold_detector(crop, nominal[1], subpixel=True, falling=True)
        ll, lr = learned_edges(model, crop, nominal)
        s_cl = roughness(cl, cr)
        s_ln = roughness(ll, lr)

        ext = [0, W_OUT, 0, H_OUT]
        for k, (title, pair, col) in enumerate([
                ("original (real SEM)", None, None),
                ("classical: threshold + sub-pixel", (cl, cr), C_CLASSICAL),
                (f"learned: U-Net '{args.variant}' (calibrated)", (ll, lr), C_LEARNED)]):
            ax = axs[i, k]
            ax.imshow(crop, cmap="gray", aspect="auto", origin="lower",
                      extent=ext, vmin=0, vmax=1)
            if pair is not None:
                ax.plot(pair[0], y, color=col, lw=0.9)
                ax.plot(pair[1], y, color=col, lw=0.9)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x (px)", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"{fn}\ntrack width {w_px:.1f} px "
                              f"({w_px*NM_PER_PX:.0f} nm)\ny (px)", fontsize=8)

        ax = axs[i, 3]
        ax.plot(cl, y, color=C_CLASSICAL, lw=0.8, label="classical L")
        ax.plot(cr, y, color=C_CLASSICAL, lw=0.8, ls=":", label="classical R")
        ax.plot(ll, y, color=C_LEARNED, lw=0.9, label="learned L")
        ax.plot(lr, y, color=C_LEARNED, lw=0.9, ls=":", label="learned R")
        ax.set_title("extracted edge traces", fontsize=9)
        ax.set_xlabel("x (px)", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
        ax.text(0.02, 0.02,
                f"apparent $\\sigma_L/\\sigma_R/\\sigma_W$ (px)\n"
                f"classical {s_cl[0]:.2f}/{s_cl[1]:.2f}/{s_cl[2]:.2f}\n"
                f"learned   {s_ln[0]:.2f}/{s_ln[1]:.2f}/{s_ln[2]:.2f}",
                transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(fc="w", alpha=0.8, ec="0.7"))

        print(f"\n--- {fn} (track width {w_px:.1f} px = {w_px*NM_PER_PX:.0f} nm) ---")
        print(f"  classical apparent sigma L/R/W: {s_cl[0]:.3f} / {s_cl[1]:.3f} / {s_cl[2]:.3f} px")
        print(f"  learned   apparent sigma L/R/W: {s_ln[0]:.3f} / {s_ln[1]:.3f} / {s_ln[2]:.3f} px")

    fig.suptitle("Edge extraction on real SEM images of IC metal tracks "
                 "(qualitative: no roughness ground truth available)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
