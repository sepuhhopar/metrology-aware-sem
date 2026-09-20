"""
Result figure on REAL SEM data: measurement repeatability, shown on the images.

Each row is one physical metal track imaged TWICE, in two separately-acquired
tiles that overlap by 10%. The two views share a specimen but not a noise
realization, scan position or drift history, so the difference between the two
measurements is measurement error -- obtainable without any ground truth.

  column 1   acquisition A, with both detectors' edges overlaid
  column 2   acquisition B, same physical track, same overlays
  column 3   linewidth profile w(y) from both acquisitions, per detector.
             Solid = view A, dashed = view B. The gap between the solid and
             dashed curve of one colour IS that detector's reproducibility
             error; a tighter pair is a more repeatable instrument+algorithm.

Usage:
    python3 make_real_result_figure.py --data-dir /path/to/dataset \
        --calibration calibration_m2.json --out real_result.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from classical_two_edge import threshold_detector
from train2 import (X0, W0, build_paired_dataset, train_one_variant, DEVICE,
                     set_calibration, load_calibration)
from real_repeatability import (find_pairs, register, prepare, learned_edges,
                                 sigmas, _runs, DOWN, NM_PER_PX,
                                 SIGMAS, XIS, ALPHAS)

Image.MAX_IMAGE_PIXELS = None
C_CL = "#ff7f0e"
C_LN = "#d62728"


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
    ap.add_argument("--n-examples", type=int, default=2)
    ap.add_argument("--out", default="real_result.png")
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
    small = {}
    for f in files:
        im = Image.open(os.path.join(sem_dir, f)).convert("L")
        small[f] = np.asarray(im.resize((im.width // DOWN, im.height // DOWN),
                                        Image.BOX)).astype(np.float32)
    pairs = find_pairs(small, thresh=0.6)
    print(f"{len(pairs)} overlapping pairs")

    cache = {}
    def load(f):
        if f not in cache:
            cache[f] = np.asarray(Image.open(os.path.join(sem_dir, f)).convert("L")).astype(np.float64)
        return cache[f]

    examples = []
    for score, fa, fb, direction in pairs:
        if len(examples) >= args.n_examples:
            break
        A, B = load(fa), load(fb)
        oy, ox, _ = register(A, B, direction)
        ys0, ys1 = max(0, oy), min(A.shape[0], B.shape[0] + oy)
        xs0, xs1 = max(0, ox), min(A.shape[1], B.shape[1] + ox)
        if ys1 - ys0 < 200 or xs1 - xs0 < 320:
            continue
        band = A[ys0:ys1, xs0:xs1]
        prof = band.mean(axis=1)
        thr = 0.5 * (np.percentile(prof, 20) + np.percentile(prof, 90))

        def valid_track(r):
            """A mean-intensity bump is not enough: a via is bright enough to
            invent a 'track' that does not run the length of the crop. Require
            the structure to be present in most columns and free of saturated
            blobs, otherwise the detectors are measuring nothing."""
            sub = band[r[0]:r[1], :]
            col_thr = 0.5 * (np.percentile(band, 20) + np.percentile(band, 90))
            coverage = (sub.max(axis=0) >= col_thr).mean()
            saturated = (sub >= np.percentile(A, 99.9)).mean()
            return coverage >= 0.85 and saturated <= 0.02

        for r in [r for r in _runs(prof >= thr)
                  if 14 <= r[1] - r[0] <= 30 and valid_track(r)]:
            yc_A = ys0 + int(round(0.5 * (r[0] + r[1])))
            yc_B, xlo_A, xhi_A = yc_A - oy, xs0 + 10, xs1 - 10
            xlo_B, xhi_B = xlo_A - ox, xhi_A - ox
            if not (0 <= yc_B - 40 and yc_B + 40 <= B.shape[0]
                    and 0 <= xlo_B and xhi_B <= B.shape[1]):
                continue
            _, _, wA = prepare(A, xlo_A, xhi_A, yc_A)
            _, _, wB = prepare(B, xlo_B, xhi_B, yc_B)
            if wA is None or wB is None or abs(wA - wB) > 2:
                continue
            wc = 0.5 * (wA + wB)
            cA, nA, _ = prepare(A, xlo_A, xhi_A, yc_A, force_width=wc)
            cB, nB, _ = prepare(B, xlo_B, xhi_B, yc_B, force_width=wc)
            if cA is None or cB is None:
                continue
            examples.append(dict(fa=fa, fb=fb, cA=cA, cB=cB, nA=nA, nB=nB,
                                 w=wc, y=yc_A))
            break

    if not examples:
        raise SystemExit("no suitable track found")

    n = len(examples)
    fig, axs = plt.subplots(n, 3, figsize=(14, 4.9 * n),
                            gridspec_kw={"width_ratios": [1, 1, 1.45]})
    if n == 1:
        axs = axs[None, :]
    y = np.arange(512)

    for i, ex in enumerate(examples):
        edges = {}
        for tag, crop, nom in (("A", ex["cA"], ex["nA"]), ("B", ex["cB"], ex["nB"])):
            edges[("cl", tag)] = (threshold_detector(crop, nom[0], subpixel=True, falling=False),
                                  threshold_detector(crop, nom[1], subpixel=True, falling=True))
            edges[("ln", tag)] = learned_edges(model, crop, nom)

        for k, (tag, crop) in enumerate((("A", ex["cA"]), ("B", ex["cB"]))):
            ax = axs[i, k]
            ax.imshow(crop, cmap="gray", aspect="auto", origin="lower",
                      extent=[0, 128, 0, 512], vmin=0, vmax=1)
            for key, col, lbl in ((("cl", tag), C_CL, "classical"),
                                  (("ln", tag), C_LN, "learned")):
                ax.plot(edges[key][0], y, color=col, lw=0.85, label=lbl)
                ax.plot(edges[key][1], y, color=col, lw=0.85)
            ax.set_title(f"acquisition {tag}: {ex['fa'] if tag=='A' else ex['fb']}", fontsize=9)
            ax.set_xlabel("x (px)", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"same physical track\nwidth {ex['w']:.0f} px "
                              f"({ex['w']*NM_PER_PX:.0f} nm)\ny (px)", fontsize=8)
                ax.legend(fontsize=7, loc="upper right", framealpha=.85)

        # Plot the DIFFERENCE between the two acquisitions rather than two
        # overlapping noisy profiles: the spread about zero is exactly the
        # reproducibility error, and it reads at a glance.
        ax = axs[i, 2]
        txt = []
        for key, col, lbl in (("cl", C_CL, "classical"), ("ln", C_LN, "learned")):
            wA_ = edges[(key, "A")][1] - edges[(key, "A")][0]
            wB_ = edges[(key, "B")][1] - edges[(key, "B")][0]
            diff = wA_ - wB_
            ax.plot(diff, y, color=col, lw=0.85, alpha=.9,
                    label=f"{lbl}  (RMS {np.sqrt((diff**2).mean()):.2f} px)")
            txt.append(f"{lbl}: |ΔCD| = {abs(wA_.mean()-wB_.mean()):.3f} px")
        ax.axvline(0, color="k", lw=1.0)
        ax.set_title("linewidth disagreement between the two acquisitions\n"
                     "$w_A(y)-w_B(y)$ — tighter about zero is more repeatable", fontsize=9)
        ax.set_xlabel("linewidth difference (px)", fontsize=8)
        ax.grid(alpha=.3)
        ax.legend(fontsize=6.5, loc="upper right", framealpha=.85)
        ax.text(.02, .02, "\n".join(txt), transform=ax.transAxes, fontsize=7.5,
                va="bottom", bbox=dict(fc="w", alpha=.85, ec="0.7"))
        print(f"example {i+1}: {ex['fa']} vs {ex['fb']}, width {ex['w']:.1f} px -> " + "; ".join(txt))

    fig.suptitle("Repeatability on real SEM images of IC metal tracks: the same physical track measured in two "
                 "independently acquired tiles\n(10% tile overlap; no roughness ground truth is used or required)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(args.out, dpi=200)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
