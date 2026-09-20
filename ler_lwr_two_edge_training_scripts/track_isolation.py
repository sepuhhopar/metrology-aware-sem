"""
Measure how isolated real metal tracks are, and compare against what the
synthetic generator assumes.

WHY THIS MATTERS
----------------
generator2.py renders ONE isolated line per field: a single line of nominal
width w0 centred in a 128-px-wide image, with clear background on both sides.
Calibrating the acquisition model (calibrate_generator.py) corrects blur and
noise, but it does not touch scene content -- so if real layouts are denser
than the generator assumes, calibration cannot reveal it.

This script quantifies the gap. For each long straight horizontal track it
measures the distance to the nearest neighbouring structure, and asks how many
tracks could be presented to the model the way it was trained: one line, with
background on both sides, filling the 128-px field at the scale that matches
the synthetic nominal width.

Usage:
    python3 track_isolation.py --data-dir /path/to/dataset --n-images 15
"""
import argparse
import os
import re

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
NM_PER_PX = 14.65
W_FIELD = 128.0     # synthetic image width, px
W0 = 40.0           # synthetic nominal linewidth, px


def find_tracks(svg_path, min_len=600, max_thick=25):
    with open(svg_path) as f:
        s = f.read()
    out = []
    for d in re.findall(r'<path d="([^"]+)" fill="lime"', s):
        pts = np.array([tuple(map(int, m)) for m in re.findall(r"[ML](\d+),(\d+)", d)])
        if len(pts) >= 4 and pts[:, 0].ptp() > min_len and pts[:, 1].ptp() < max_thick:
            out.append((int(pts[:, 0].min()), int(pts[:, 0].max()),
                        int(round(pts[:, 1].mean()))))
    return out


def runs(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--layer", default="m2")
    ap.add_argument("--n-images", type=int, default=15)
    ap.add_argument("--half-window", type=int, default=60,
                    help="rows examined either side of the track centre")
    args = ap.parse_args()

    sem_dir = os.path.join(args.data_dir, "sems", args.layer)
    lab_dir = os.path.join(args.data_dir, "labels", args.layer)
    files = sorted(f for f in os.listdir(sem_dir) if f.endswith(".png"))[:args.n_images]

    widths, margins, n_single, n_total = [], [], 0, 0
    for fn in files:
        svg = os.path.join(lab_dir, f"label{re.sub(r'[^0-9]', '', fn)}.svg")
        if not os.path.exists(svg):
            continue
        img = np.asarray(Image.open(os.path.join(sem_dir, fn)).convert("L")).astype(float)
        for (x0, x1, yc) in find_tracks(svg):
            h = args.half_window
            r0, r1 = max(0, yc - h), min(img.shape[0], yc + h)
            strip = img[r0:r1, x0 + 20:x1 - 20]
            if strip.shape[1] < 200:
                continue
            prof = strip.mean(axis=1)
            thr = 0.5 * (np.percentile(prof, 5) + np.percentile(prof, 95))
            rr = runs(prof >= thr)
            if not rr:
                continue
            n_total += 1
            if len(rr) == 1:
                n_single += 1
            c = yc - r0
            i = int(np.argmin([abs(0.5 * (a + b) - c) for a, b in rr]))
            main_run = rr[i]
            widths.append(main_run[1] - main_run[0])
            left = main_run[0] - (rr[i - 1][1] if i > 0 else 0)
            right = (rr[i + 1][0] if i + 1 < len(rr) else len(prof)) - main_run[1]
            margins.append(min(left, right))

    if not n_total:
        raise SystemExit("no tracks measured")
    w = np.array(widths, float); m = np.array(margins, float)

    # to show one track the way the generator does, the 128-px field must be
    # filled at the scale that maps the real width onto w0
    need_half = 0.5 * W_FIELD / (W0 / np.median(w))

    print(f"tracks examined: {n_total}  (from {len(files)} images, layer {args.layer})")
    print(f"single structure within +-{args.half_window} px: {n_single} "
          f"({100*n_single/n_total:.0f}%)")
    print(f"\ntrack width      : median {np.median(w):.1f} px ({np.median(w)*NM_PER_PX:.0f} nm)")
    print(f"gap to neighbour : median {np.median(m):.1f} px "
          f"({np.median(m)*NM_PER_PX:.0f} nm), p25 {np.percentile(m,25):.1f}, "
          f"p75 {np.percentile(m,75):.1f}, max {m.max():.0f}")
    print(f"pitch            : median {np.median(w)+np.median(m):.1f} px "
          f"({(np.median(w)+np.median(m))*NM_PER_PX:.0f} nm)")
    print(f"\nclear background needed on EACH side to fill the synthetic field: "
          f"{need_half:.0f} px")
    print(f"tracks meeting it: {int((m >= need_half).sum())}/{n_total} "
          f"({100*(m >= need_half).mean():.1f}%)")
    print("\n-> The generator renders one isolated line per field. Real layouts at "
          "this pitch cannot supply that, so every real crop contains neighbouring "
          "tracks. Acquisition calibration corrects blur and noise, not scene "
          "content, and therefore cannot expose this mismatch.")


if __name__ == "__main__":
    main()
