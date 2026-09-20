"""
Measurement repeatability on REAL SEM data, using the dataset's own tile overlap.

WHY THIS IS POSSIBLE WITHOUT GROUND TRUTH
-----------------------------------------
Line-edge roughness has no ground truth on a real specimen: the true edge of a
fabricated line is unobservable. Accuracy therefore cannot be measured. But
*precision* can -- and precision is what a metrology tool is actually qualified
on, and what the consistency term of the training objective is designed to
improve.

The M1/M2 images in this dataset were captured with 10% tile overlap, so the
same physical wafer region appears in two separately-acquired images. Those two
views share a specimen but not a noise realization, a scan position, or a drift
history. Measuring the same track in both and comparing the results is a direct
estimate of measurement reproducibility on a real instrument.

This is the real-data analogue of the synthetic repeatability experiment in
run_experiment9.py, with one difference that makes it harder, not easier: there
the two views were rendered from an identical geometry at an identical position,
whereas here they differ by an arbitrary translation that must be recovered.

PIPELINE
  1. Find overlapping image pairs by normalized cross-correlation of border
     strips on 8x-downsampled images.
  2. Refine the offset to full resolution by FFT phase correlation.
  3. Locate long straight horizontal tracks inside the shared region.
  4. Extract the same physical track from both images, preprocess identically
     to make_real_edge_figure.py (rotate to vertical, rescale so the track
     width matches w0, normalize), and measure edges with BOTH the classical
     threshold detector and the learned model.
  5. Report, per detector, the discrepancy between the two acquisitions.

WHAT THE NUMBERS MEAN
  d_sigma  : |sigma(view A) - sigma(view B)| for the same physical track.
             Reproducibility of the roughness estimate. Lower is better.
  d_pos    : RMS of (x_A - x_B) after removing the mean offset. Positional
             reproducibility. Lower is better.
  d_width  : |mean width A - mean width B|. Reproducibility of the CD estimate.

A caveat that must travel with these numbers: the apparent roughness here is
dominated by detection noise, not by physical line-edge roughness, because the
14.65 nm pixel and ~55 nm point-spread width place real roughness (single-digit
nm at this node) well below the measurement floor. These are therefore
statements about measurement stability, not about the roughness of the wafer.

Usage:
    python3 real_repeatability.py --data-dir /path/to/dataset \
        --calibration calibration_m2.json --n-images 60 --out-prefix real_repeat
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter

from classical_two_edge import threshold_detector
from model2 import soft_localize, differentiable_detrend, rms
from train2 import (X0, W0, build_paired_dataset, train_one_variant, DEVICE,
                     set_calibration, load_calibration)

Image.MAX_IMAGE_PIXELS = None
H_OUT, W_OUT = 512, 128
NM_PER_PX = 14.65
DOWN = 8

SIGMAS = [1.5, 3.0]
XIS = [8.0, 20.0]
ALPHAS = [0.3, 0.7]


# ---------------------------------------------------------------------------
# Overlap discovery and registration
# ---------------------------------------------------------------------------
def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def find_pairs(small, frac=0.10, thresh=0.5):
    """Border-strip NCC on downsampled images. Returns (score, A, B, direction)."""
    names = list(small)
    H, W = small[names[0]].shape
    ow, oh = int(frac * W), int(frac * H)
    out = []
    for fa in names:
        A = small[fa]
        for fb in names:
            if fa == fb:
                continue
            out.append((ncc(A[:, -ow:], small[fb][:, :ow]), fa, fb, "right"))
            out.append((ncc(A[-oh:, :], small[fb][:oh, :]), fa, fb, "below"))
    return sorted((o for o in out if o[0] >= thresh), reverse=True)


def register(A, B, direction, frac=0.14):
    """Full-resolution offset of B relative to A via FFT phase correlation on
    the overlapping strips. Returns (dy, dx) mapping A coords -> B coords."""
    H, W = A.shape
    if direction == "right":
        a = A[:, int(W * (1 - frac)):]
        b = B[:, :a.shape[1]]
        base = (0, int(W * (1 - frac)))
    else:
        a = A[int(H * (1 - frac)):, :]
        b = B[:a.shape[0], :]
        base = (int(H * (1 - frac)), 0)
    a = a - a.mean(); b = b - b.mean()
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b, s=a.shape))
    F /= np.abs(F) + 1e-12
    c = np.fft.ifft2(F).real
    dy, dx = np.unravel_index(np.argmax(c), c.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    peak = float(c.max() / (c.std() + 1e-12))
    # A[y, x] corresponds to B[y - dy - base0, x - dx - base1]
    return (base[0] + dy, base[1] + dx, peak)


# ---------------------------------------------------------------------------
# Track handling (shared with make_real_edge_figure.py)
# ---------------------------------------------------------------------------
def _runs(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def central_run(prof, target=None):
    lo, hi = np.percentile(prof, 5), np.percentile(prof, 95)
    rr = _runs(prof >= 0.5 * (lo + hi))
    if not rr:
        return None
    t = 0.5 * len(prof) if target is None else target
    return min(rr, key=lambda r: abs(0.5 * (r[0] + r[1]) - t))


def prepare(img, x_lo, x_hi, yc, force_width=None):
    """Crop a horizontal track -> (512,128) vertical-line image in [0,1],
    rescaled so the track width matches w0. Returns (crop, nominal, width).

    force_width: use this width (in native px) to set the rescale factor
    instead of the one measured in this view. Essential when comparing two
    acquisitions of the SAME track: measuring the width separately in each
    view would give each crop its own scale factor, and the comparison would
    then partly measure that scale difference rather than the instrument."""
    half = 40
    r0, r1 = yc - half, yc + half
    if r0 < 0 or r1 > img.shape[0] or x_hi - x_lo < 120:
        return None, None, None
    strip = img[r0:r1, x_lo:x_hi]
    r = central_run(strip.mean(axis=1), target=yc - r0)
    if r is None:
        return None, None, None
    w_px = float(r[1] - r[0])
    if w_px < 4:
        return None, None, None
    scale = W0 / (force_width if force_width is not None else w_px)
    need_along, need_across = int(np.ceil(H_OUT / scale)), int(np.ceil(W_OUT / scale))
    if strip.shape[1] < need_along:
        return None, None, None
    c = int(round(0.5 * (r[0] + r[1])))
    a0 = int(np.clip(c - need_across // 2, 0, max(0, strip.shape[0] - need_across)))
    sub = strip[a0:a0 + need_across, :need_along]
    if sub.shape[0] < need_across:
        return None, None, None
    big = zoom(sub, scale, order=1)[:W_OUT, :H_OUT]
    if big.shape != (W_OUT, H_OUT):
        return None, None, None
    out = big.T
    bg, fg = np.percentile(out, 5), np.percentile(out, 95)
    out = np.clip((out - bg) / max(fg - bg, 1e-6), 0.0, 1.0)
    rr = central_run(out.mean(axis=0))
    if rr is None:
        return None, None, None
    return out, (float(rr[0]), float(rr[1])), w_px


@torch.no_grad()
def learned_edges(model, img, nominal):
    t = torch.from_numpy(img.astype(np.float32))[None, None].to(
        next(model.parameters()).device)
    logits = model(t)
    xl, _, _, _ = soft_localize(logits[:, 0], nominal[0], 15, 0.05)
    xr, _, _, _ = soft_localize(logits[:, 1], nominal[1], 15, 0.05)
    return xl[0].cpu().numpy(), xr[0].cpu().numpy()


def sigmas(xl, xr):
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))[None, ...]
    s = lambda a: float(rms(differentiable_detrend(t(a)))[0].item())
    return s(xl), s(xr), s(np.asarray(xr) - np.asarray(xl))


def compare(xlA, xrA, xlB, xrB):
    """Discrepancy between two acquisitions of the same physical track."""
    sA, sB = sigmas(xlA, xrA), sigmas(xlB, xrB)
    # remove the mean offset: absolute position differs by sub-pixel registration
    dl = (xlA - xlA.mean()) - (xlB - xlB.mean())
    dr = (xrA - xrA.mean()) - (xrB - xrB.mean())
    return dict(
        d_sigma_L=abs(sA[0] - sB[0]), d_sigma_R=abs(sA[1] - sB[1]),
        d_sigma_W=abs(sA[2] - sB[2]),
        d_pos=float(0.5 * (np.sqrt((dl ** 2).mean()) + np.sqrt((dr ** 2).mean()))),
        d_width=abs(float(np.mean(xrA - xlA) - np.mean(xrB - xlB))),
        sigma_W_A=sA[2], sigma_W_B=sB[2],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--layer", default="m2")
    ap.add_argument("--calibration", default="calibration_m2.json")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-per-config", type=int, default=15)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.3, 0.6, -0.3])
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--ncc-threshold", type=float, default=0.5)
    ap.add_argument("--out-prefix", default="real_repeat")
    args = ap.parse_args()

    cal = load_calibration(args.calibration)
    set_calibration(cal)
    cfgs = [(s, x, a, r) for s in SIGMAS for x in XIS for a in ALPHAS for r in args.rhos]
    print(f"Device {DEVICE}; training '{args.variant}' on calibrated synthetic data")
    model = train_one_variant(args.variant,
                              build_paired_dataset(cfgs, args.train_per_config, salt=700),
                              epochs=args.epochs, seed=0, verbose=False)
    model.eval()

    sem_dir = os.path.join(args.data_dir, "sems", args.layer)
    files = sorted(f for f in os.listdir(sem_dir) if f.endswith(".png"))[:args.n_images]
    print(f"\nloading {len(files)} images (downsampled {DOWN}x) to find overlaps ...")
    small, full = {}, {}
    for f in files:
        im = Image.open(os.path.join(sem_dir, f)).convert("L")
        small[f] = np.asarray(im.resize((im.width // DOWN, im.height // DOWN),
                                        Image.BOX)).astype(np.float32)
    pairs = find_pairs(small, thresh=args.ncc_threshold)[:args.max_pairs]
    print(f"found {len(pairs)} overlapping pairs (NCC >= {args.ncc_threshold})")
    for s, a, b, d in pairs:
        print(f"   {s:.3f}  {a} -> {b} ({d})")

    def load(f):
        if f not in full:
            full[f] = np.asarray(Image.open(os.path.join(sem_dir, f)).convert("L")).astype(np.float64)
        return full[f]

    rows = []
    for score, fa, fb, direction in pairs:
        A, B = load(fa), load(fb)
        oy, ox, peak = register(A, B, direction)
        print(f"\n{fa} -> {fb} ({direction}) NCC {score:.3f}; offset (dy={oy}, dx={ox}) peak {peak:.1f}")

        # shared region in A coordinates
        ys0, ys1 = max(0, oy), min(A.shape[0], B.shape[0] + oy)
        xs0, xs1 = max(0, ox), min(A.shape[1], B.shape[1] + ox)
        # a right-overlap shares about 4096-3702 = 394 px, which is enough:
        # prepare() needs H_OUT/scale ~= 285 px along the track
        if ys1 - ys0 < 200 or xs1 - xs0 < 320:
            print(f"   shared region {ys1-ys0}x{xs1-xs0} too small, skipping")
            continue

        # find horizontal tracks inside the shared region, from image content
        band = A[ys0:ys1, xs0:xs1]
        prof = band.mean(axis=1)
        thr = 0.5 * (np.percentile(prof, 20) + np.percentile(prof, 90))
        cand = [r for r in _runs(prof >= thr) if 6 <= r[1] - r[0] <= 40]
        print(f"   shared region {band.shape}; candidate tracks: {len(cand)}")

        used = 0
        for r in cand:
            yc_A = ys0 + int(round(0.5 * (r[0] + r[1])))
            yc_B = yc_A - oy
            xlo_A, xhi_A = xs0 + 10, xs1 - 10
            xlo_B, xhi_B = xlo_A - ox, xhi_A - ox
            if not (0 <= yc_B - 40 and yc_B + 40 <= B.shape[0]
                    and 0 <= xlo_B and xhi_B <= B.shape[1]):
                continue
            # measure the width in each view first, purely as a consistency
            # check that registration landed on the same structure
            _, _, wA = prepare(A, xlo_A, xhi_A, yc_A)
            _, _, wB = prepare(B, xlo_B, xhi_B, yc_B)
            if wA is None or wB is None or abs(wA - wB) > 3:
                continue
            # then rebuild BOTH crops at one common scale, so the two views
            # share a coordinate system and the comparison measures the
            # instrument rather than the preprocessing
            w_common = 0.5 * (wA + wB)
            cA, nA, _ = prepare(A, xlo_A, xhi_A, yc_A, force_width=w_common)
            cB, nB, _ = prepare(B, xlo_B, xhi_B, yc_B, force_width=w_common)
            if cA is None or cB is None:
                continue

            clA = (threshold_detector(cA, nA[0], subpixel=True, falling=False),
                   threshold_detector(cA, nA[1], subpixel=True, falling=True))
            clB = (threshold_detector(cB, nB[0], subpixel=True, falling=False),
                   threshold_detector(cB, nB[1], subpixel=True, falling=True))
            lnA = learned_edges(model, cA, nA)
            lnB = learned_edges(model, cB, nB)

            for meth, (pA, pB) in (("classical", (clA, clB)), ("learned", (lnA, lnB))):
                d = compare(pA[0], pA[1], pB[0], pB[1])
                d.update(method=meth, pair=f"{fa}->{fb}", y_A=yc_A,
                         width_px=0.5 * (wA + wB), ncc=score)
                rows.append(d)
            used += 1
        print(f"   tracks compared: {used}")

    if not rows:
        raise SystemExit("no comparable tracks found")

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_prefix}_raw.csv", index=False)
    summ = df.groupby("method").agg(
        n=("d_sigma_W", "count"),
        d_sigma_W_mean=("d_sigma_W", "mean"), d_sigma_W_median=("d_sigma_W", "median"),
        d_sigma_L_mean=("d_sigma_L", "mean"), d_sigma_R_mean=("d_sigma_R", "mean"),
        d_pos_mean=("d_pos", "mean"), d_width_mean=("d_width", "mean"),
        sigma_W_mean=("sigma_W_A", "mean"),
    ).reset_index()
    summ.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    print("\n=== Repeatability across two real acquisitions of the same region ===")
    print("(lower d_* = more reproducible; no ground truth is involved)")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nin nm (pixel = {NM_PER_PX} nm, after rescale to w0={W0:g}):")
    for _, r in summ.iterrows():
        sc = r["width_px_scale"] if "width_px_scale" in r else 1.0
        print(f"  {r['method']:10s} d_sigma_W = {r['d_sigma_W_mean']:.4f} px, "
              f"d_pos = {r['d_pos_mean']:.4f} px")
    print("\nNOTE: apparent roughness on this data is dominated by detection noise, "
          "not by physical line-edge roughness. These are measurement-stability "
          "figures, not roughness measurements.")


if __name__ == "__main__":
    main()
