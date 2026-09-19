"""
Measure the acquisition physics of a REAL SEM dataset and convert it into
generator2.py parameters, so the synthetic data can be rendered under imaging
conditions matched to a real instrument instead of the optimistic defaults.

Nothing is trained here. The generator is a parametric physics model; this
script estimates four constants from real images and reports the generator
settings that reproduce them:

  psf_sigma   Gaussian point-spread width, from an erf fit to the averaged
              edge-spread function of long straight metal tracks. Averaging
              ALONG the track suppresses noise while preserving the PSF.
  contrast    foreground minus background level (track vs dielectric).
  sigma_read  per-pixel noise in the background, where the Poisson term
              vanishes -> this is the additive Gaussian read noise.
  dose        from the foreground noise: after removing sigma_read in
              quadrature, the remainder is shot noise, whose normalized
              std is 1/sqrt(dose).

Two ways to transfer the PSF to the synthetic grid are reported, because the
synthetic images have no intrinsic pixel size:

  absolute    use the measured psf_sigma in pixels directly (valid if you
              declare the synthetic pixel to be the real one, 14.65 nm here).
  ratio       preserve the measured PSF-to-linewidth ratio, which is what
              actually sets how hard the edges are to localize. This is the
              honest apples-to-apples setting when the synthetic linewidth
              (w0) differs from the real track width.

Usage:
    python3 calibrate_generator.py --data-dir /path/to/dataset --layer m2 \
        --n-images 40 --out calibration.json
"""
import argparse
import json
import os
import re

import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from scipy.special import erf

Image.MAX_IMAGE_PIXELS = None

# generator2.py defaults, reproduced for the comparison print-out
GEN_PSF_SIGMA_DEFAULT = 0.8
GEN_W0_DEFAULT = 40.0


def erf_edge(r, lo, hi, centre, width):
    """Ideal step blurred by a Gaussian of std `width`."""
    return lo + (hi - lo) * 0.5 * (1.0 + erf((r - centre) / (width * np.sqrt(2.0))))


def find_horizontal_tracks(svg_path, min_len=600, max_thick=25):
    """Long, thin, horizontal label polygons -> (x0, x1, y_centre).
    Only the bounding box is used: the polygons are heavily decimated and are
    NOT accurate enough to serve as edge ground truth, but they are perfectly
    good for locating a straight track to measure the PSF on."""
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
    # de-duplicate (the dataset README warns labels may repeat)
    seen, uniq = set(), []
    for t in out:
        key = (t[0] // 32, t[2] // 8)
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


def measure_track(img, x0, x1, yc, half=18, margin=20):
    """Return per-track (psf_sigma, contrast, sigma_read, sigma_fg, width_px)
    or None if the fit is not trustworthy."""
    r0, r1 = yc - half, yc + half
    if r0 < 0 or r1 > img.shape[0] or (x1 - margin) - (x0 + margin) < 256:
        return None
    strip = img[r0:r1, x0 + margin:x1 - margin]

    # average along the track: noise averages down, the PSF does not
    prof = strip.mean(axis=1)
    rows = np.arange(len(prof))
    mid = len(prof) // 2

    bg_lo, fg_lo = prof[:3].mean(), prof[mid - 2:mid + 2].mean()
    if fg_lo - bg_lo < 10:            # no usable contrast
        return None

    widths, centres = [], []
    # upper edge (rising into the track), lower edge (falling out of it)
    try:
        p_up, _ = curve_fit(erf_edge, rows[:mid], prof[:mid],
                            p0=[bg_lo, fg_lo, mid / 2.0, 2.0], maxfev=20000)
        p_dn, _ = curve_fit(lambda r, lo, hi, c, w: erf_edge(-r, lo, hi, -c, w),
                            rows[mid:], prof[mid:],
                            p0=[bg_lo, fg_lo, mid * 1.5, 2.0], maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    for p in (p_up, p_dn):
        w = abs(p[3])
        if not (0.2 < w < half):      # reject degenerate fits
            return None
        widths.append(w)
        centres.append(p[2])
    width_px = abs(centres[1] - centres[0])
    if width_px < 4:
        return None

    contrast = 0.5 * ((p_up[1] - p_up[0]) + (p_dn[1] - p_dn[0]))
    if contrast <= 0:
        return None

    # noise: residual after removing the column-averaged profile, so real
    # structure does not leak into the noise estimate
    resid = strip - prof[:, None]
    n_bg = max(2, int(round(min(centres) - 2 * max(widths))))
    bg_rows = resid[:max(2, n_bg)]
    c0, c1 = int(round(min(centres))), int(round(max(centres)))
    fg_rows = resid[c0 + 2:c1 - 2] if c1 - c0 > 6 else resid[c0:c1]
    if bg_rows.size < 32 or fg_rows.size < 32:
        return None
    # ddof: the profile mean was removed along axis 1
    sigma_read = float(bg_rows.std())
    sigma_fg = float(fg_rows.std())

    return dict(psf_sigma=float(np.mean(widths)), contrast=float(contrast),
                sigma_read=sigma_read, sigma_fg=sigma_fg, width_px=float(width_px))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="dataset root containing sems/ and labels/")
    ap.add_argument("--layer", default="m2", help="layer with labels (m2)")
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--nm-per-px", type=float, default=14.65)
    ap.add_argument("--w0", type=float, default=GEN_W0_DEFAULT,
                    help="synthetic linewidth in px, for the ratio transfer")
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    sem_dir = os.path.join(args.data_dir, "sems", args.layer)
    lab_dir = os.path.join(args.data_dir, "labels", args.layer)
    if not (os.path.isdir(sem_dir) and os.path.isdir(lab_dir)):
        raise SystemExit(f"expected {sem_dir} and {lab_dir}")

    sems = sorted(f for f in os.listdir(sem_dir) if f.endswith(".png"))[:args.n_images]
    print(f"Measuring {len(sems)} images from {sem_dir} ...")

    rows = []
    for i, fn in enumerate(sems):
        stem = re.sub(r"\D", "", fn)
        svg = os.path.join(lab_dir, f"label{stem}.svg")
        if not os.path.exists(svg):
            continue
        img = np.asarray(Image.open(os.path.join(sem_dir, fn)).convert("L")).astype(np.float64)
        for (x0, x1, yc) in find_horizontal_tracks(svg):
            m = measure_track(img, x0, x1, yc)
            if m is not None:
                m["image"] = fn
                rows.append(m)
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(sems)} images, {len(rows)} tracks measured")

    if not rows:
        raise SystemExit("no usable tracks measured -- check --layer / label paths")

    def agg(key):
        v = np.array([r[key] for r in rows])
        return float(np.median(v)), float(v.std()), len(v)

    psf, psf_sd, n = agg("psf_sigma")
    contrast, contrast_sd, _ = agg("contrast")
    s_read, _, _ = agg("sigma_read")
    s_fg, _, _ = agg("sigma_fg")
    width, width_sd, _ = agg("width_px")

    # normalize to the generator's [0,1] contrast convention
    sigma_g = s_read / contrast
    total_fg = s_fg / contrast
    shot = float(np.sqrt(max(total_fg ** 2 - sigma_g ** 2, 1e-12)))
    dose = 1.0 / shot ** 2
    snr = contrast / s_fg

    psf_ratio = psf / width          # PSF as a fraction of linewidth
    psf_for_w0 = psf_ratio * args.w0

    cal = {
        "source": {"data_dir": args.data_dir, "layer": args.layer,
                   "n_tracks": n, "n_images": len(sems),
                   "nm_per_px": args.nm_per_px},
        "measured": {
            "psf_sigma_px": psf, "psf_sigma_px_std": psf_sd,
            "psf_sigma_nm": psf * args.nm_per_px,
            "track_width_px": width, "track_width_nm": width * args.nm_per_px,
            "contrast_DN": contrast, "sigma_read_DN": s_read,
            "sigma_fg_DN": s_fg, "snr": snr,
        },
        "generator_params": {
            "psf_sigma_absolute": psf,
            "psf_sigma_ratio_matched": psf_for_w0,
            "psf_to_linewidth_ratio": psf_ratio,
            "dose": dose, "sigma_g": sigma_g,
            "equivalent_k_n": 400.0 / dose,
        },
    }
    with open(args.out, "w") as f:
        json.dump(cal, f, indent=2)

    print(f"\n=== Measured from {n} tracks across {len(sems)} images ===")
    print(f"  PSF sigma          : {psf:.2f} px  ({psf*args.nm_per_px:.1f} nm)   [sd {psf_sd:.2f}]")
    print(f"  track width        : {width:.2f} px  ({width*args.nm_per_px:.1f} nm)")
    print(f"  PSF / linewidth    : {psf_ratio:.3f}")
    print(f"  contrast           : {contrast:.1f} DN")
    print(f"  read noise (bg)    : {s_read:.2f} DN   -> sigma_g = {sigma_g:.4f}")
    print(f"  fg noise           : {s_fg:.2f} DN   -> SNR = {snr:.2f}")
    print(f"  shot noise (norm)  : {shot:.4f}      -> dose = {dose:.1f}")

    print(f"\n=== Generator settings ===")
    print(f"  current default    : psf_sigma={GEN_PSF_SIGMA_DEFAULT}, "
          f"PSF/linewidth={GEN_PSF_SIGMA_DEFAULT/args.w0:.3f}, k_n grid 0.3-4.0")
    print(f"  calibrated (ratio) : psf_sigma={psf_for_w0:.2f}  "
          f"(preserves the measured PSF/linewidth at w0={args.w0:g})")
    print(f"  calibrated (abs)   : psf_sigma={psf:.2f}  "
          f"(if you declare the synthetic pixel to be {args.nm_per_px} nm)")
    print(f"  noise              : dose={dose:.1f}, sigma_g={sigma_g:.4f} "
          f"(equivalent k_n = {400.0/dose:.2f})")
    print(f"\n  -> the real instrument is {psf_ratio/(GEN_PSF_SIGMA_DEFAULT/args.w0):.1f}x "
          f"blurrier relative to linewidth than the generator's default.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
