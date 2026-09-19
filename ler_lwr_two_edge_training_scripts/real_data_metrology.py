"""
Real-data LER/LWR/PSD extraction, for a dataset like the "Carinthia-S"-style
structure you described: per-image ground truth as either

  (a) a binary segmentation mask image (line = foreground) alongside the SEM
      image, e.g.  image_001.tif  +  image_001_mask.png
  (b) a pre-extracted edge-coordinate JSON, e.g.
          image_001.tif  +  edges_001.json
          { "left_edge": [x1, x2, ..., xN], "right_edge": [x1, ..., xN] }
  (c) a Labelme polygon/polyline annotation JSON (shapes named e.g.
      "left_edge" / "right_edge"), which gets interpolated to x(y) per row.

This script does NOT assume you already have the data in one exact layout --
`discover_dataset()` looks for all three patterns under a root directory and
tells you what it found (and, importantly, what it could NOT match, so you
can fix file naming rather than silently dropping images). Point it at
whatever directory structure Carinthia-S actually unpacks to; if none of the
three patterns match, see "Adapting to your actual file layout" in the
module docstring / README before assuming the tool is broken.

Pipeline per image (mirrors the synthetic pipeline in generator2.py so the
numbers are directly comparable to Tables 2-4):
    edges -> linear detrend -> RMS (LER_L, LER_R, LWR) -> Welch PSD (L, R, W)
Optionally (--fit-k-correlation) also fits the K-correlation model
(xi, alpha) to each PSD, so you can report where real images fall relative
to the synthetic training grid's (sigma, xi, alpha) ranges -- a direct,
honest way to characterize the domain gap, rather than assuming it away.

This script is intentionally self-contained (only numpy/scipy/pandas, plus
Pillow for image I/O) -- no dependency on the rest of this repo -- so you
can copy just this one file next to your real dataset if that's easier than
carrying the whole v2/ package around.

Usage:
    python3 real_data_metrology.py --data-dir /path/to/carinthia-s --out-prefix real1

    # Only look at the first 50 images while you're checking the file
    # matching logic works before running the full dataset:
    python3 real_data_metrology.py --data-dir /path/to/carinthia-s --limit 50 --out-prefix real_smoke

    # Also fit K-correlation (xi, alpha) per image, for domain-gap comparison
    # against the synthetic grid (sigma in {1.5,3.0}, xi in {8,20}, alpha in {0.3,0.7}):
    python3 real_data_metrology.py --data-dir /path/to/carinthia-s --fit-k-correlation --out-prefix real1
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import signal, optimize

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
MASK_HINTS = ("mask", "seg", "label", "gt", "annotation")

# Synthetic training grid this repo's models were trained on (generator2.py),
# reproduced here (NOT imported) so this script stays copy-paste portable.
SYNTH_SIGMA_RANGE = (1.5, 3.0)
SYNTH_XI_RANGE = (8.0, 20.0)
SYNTH_ALPHA_RANGE = (0.3, 0.7)


# ---------------------------------------------------------------------------
# Metrology core (self-contained: detrend, RMS, Welch PSD, K-correlation fit)
# ---------------------------------------------------------------------------
def detrend_np(x):
    """Linear (order-1) detrend along the last axis. NaNs are not allowed --
    caller must fill/interpolate first."""
    n = len(x)
    y = np.arange(1, n + 1, dtype=np.float64)
    A = np.stack([y, np.ones_like(y)], axis=1)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    trend = A @ coef
    return x - trend


def rms_np(x):
    return float(np.sqrt(np.mean(x ** 2)))


def welch_psd_np(x_detrended, nperseg=128, noverlap=64, fs=1.0):
    nperseg = min(nperseg, len(x_detrended))
    noverlap = min(noverlap, nperseg - 1) if nperseg > 1 else 0
    freqs, psd = signal.welch(x_detrended, fs=fs, window="hann",
                               nperseg=nperseg, noverlap=noverlap,
                               detrend=False, return_onesided=True, scaling="density")
    return freqs, psd


def k_correlation_model(f, sigma, xi, alpha, eps=1e-12):
    """K-correlation (fractal) PSD model, matching the synthetic generator's
    shaping function (Palasantzas 1993 form)."""
    return (2.0 * np.sqrt(np.pi) * (sigma ** 2) * xi * (alpha + 0.5) /
            (1.0 + (2.0 * np.pi * f * xi) ** 2) ** (alpha + 1.0) + eps)


def fit_xi_alpha(freqs, psd, sigma_fixed):
    """Fit (xi, alpha) by nonlinear least squares with sigma held fixed to
    the directly-measured RMS roughness (more stable than a 3-parameter
    fit). Returns (xi, alpha) or (nan, nan) if the fit fails."""
    mask = freqs > 0
    f, p = freqs[mask], psd[mask]
    if len(f) < 4:
        return float("nan"), float("nan")

    def model(f_, xi, alpha):
        return k_correlation_model(f_, sigma_fixed, xi, alpha)

    try:
        popt, _ = optimize.curve_fit(model, f, p, p0=[12.0, 0.5],
                                      bounds=([0.5, 0.05], [200.0, 0.95]), maxfev=5000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return float("nan"), float("nan")


def _fill_nan_rows(x):
    x = x.astype(np.float64).copy()
    nan_mask = np.isnan(x)
    if nan_mask.all():
        return None
    if nan_mask.any():
        idx = np.arange(len(x))
        x[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x[~nan_mask])
    return x


# ---------------------------------------------------------------------------
# Edge extraction from the three supported ground-truth formats
# ---------------------------------------------------------------------------
def edges_from_mask(mask):
    """mask: 2D array, foreground = the line (nonzero). Returns (x_L, x_R),
    pixel-quantized, one entry per row, NaN where a row has no foreground
    (filled by neighboring-row interpolation before being returned; None if
    the whole mask is empty). Assumes ONE contiguous foreground band per
    row (a single line/space structure) -- for multi-line or blob-shaped
    defect masks, segment by connected component first and call this per
    component."""
    h = mask.shape[0]
    x_L = np.full(h, np.nan)
    x_R = np.full(h, np.nan)
    fg = mask > 0
    for y in range(h):
        cols = np.nonzero(fg[y])[0]
        if len(cols) == 0:
            continue
        x_L[y] = cols[0]
        x_R[y] = cols[-1]
    x_L = _fill_nan_rows(x_L)
    x_R = _fill_nan_rows(x_R)
    if x_L is None or x_R is None:
        return None, None
    return x_L, x_R


def edges_from_edge_json(path):
    """{"left_edge": [...], "right_edge": [...]} -> (x_L, x_R) as float arrays."""
    with open(path) as f:
        d = json.load(f)
    keys_L = ["left_edge", "x_L", "xL", "left"]
    keys_R = ["right_edge", "x_R", "xR", "right"]
    kL = next((k for k in keys_L if k in d), None)
    kR = next((k for k in keys_R if k in d), None)
    if kL is None or kR is None:
        return None, None
    x_L = np.asarray(d[kL], dtype=np.float64)
    x_R = np.asarray(d[kR], dtype=np.float64)
    if len(x_L) != len(x_R):
        n = min(len(x_L), len(x_R))
        x_L, x_R = x_L[:n], x_R[:n]
    x_L = _fill_nan_rows(x_L)
    x_R = _fill_nan_rows(x_R)
    return x_L, x_R


def edges_from_labelme_json(path, left_label="left_edge", right_label="right_edge"):
    """Labelme-style JSON: {"shapes": [{"label": ..., "points": [[x,y],...]}, ...],
    "imageHeight": H, ...}. Interpolates each labeled polyline to one x per
    integer row over its own y-range; rows outside a polyline's y-range are
    left as NaN (not extrapolated) then filled by neighbor interpolation."""
    with open(path) as f:
        d = json.load(f)
    shapes = d.get("shapes", [])
    h = d.get("imageHeight")

    def _extract(label_wanted):
        pts = None
        for s in shapes:
            lbl = str(s.get("label", "")).lower()
            if lbl == label_wanted or label_wanted in lbl:
                pts = np.asarray(s.get("points", []), dtype=np.float64)
                break
        return pts

    pL = _extract(left_label)
    pR = _extract(right_label)
    if pL is None or pR is None or len(pL) < 2 or len(pR) < 2:
        return None, None

    if h is None:
        h = int(round(max(pL[:, 1].max(), pR[:, 1].max()))) + 1

    ys = np.arange(h)
    x_L = np.full(h, np.nan)
    x_R = np.full(h, np.nan)

    def _interp_onto(pts):
        order = np.argsort(pts[:, 1])
        py, px = pts[order, 1], pts[order, 0]
        py_u, idx_u = np.unique(py, return_index=True)
        px_u = px[idx_u]
        if len(py_u) < 2:
            return np.full(h, np.nan)
        lo, hi = py_u.min(), py_u.max()
        vals = np.interp(ys, py_u, px_u)
        vals[(ys < lo) | (ys > hi)] = np.nan
        return vals

    x_L = _interp_onto(pL)
    x_R = _interp_onto(pR)
    x_L = _fill_nan_rows(x_L)
    x_R = _fill_nan_rows(x_R)
    return x_L, x_R


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------
def _load_image_array(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"))


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _digits(s):
    m = re.findall(r"\d+", s)
    return m[-1] if m else None


def _has_hint(name, hints):
    """True if any hint appears as a whole underscore/dash/dot/space-delimited
    token in name (case-insensitive). Token-based rather than raw substring
    matching so e.g. 'image_004_nogt.png' is NOT mistaken for a mask/ground-
    truth file just because it contains the letters 'gt'."""
    tokens = re.split(r"[^0-9a-zA-Z]+", name.lower())
    return any(h in tokens for h in hints)


def discover_from_manifest(root_dir):
    """Pattern (d): a CSV/TSV manifest in root_dir that pairs images to masks
    explicitly, e.g. Carinthia-S's `carinthia-s.csv` with columns
    `image_path;mask_path;filename;label`. Paths in the manifest are taken
    relative to the manifest's own directory. Returns (matched, unmatched) or
    (None, None) if no usable manifest was found -- caller then falls back to
    filename-based matching. This pattern exists because a manifest-driven
    dataset can use opaque filenames (UUIDs) that carry no 'mask' token and no
    digits to match on, which defeats every name-based heuristic below."""
    img_cols = ("image_path", "image", "img", "img_path", "file", "filepath")
    mask_cols = ("mask_path", "mask", "seg_path", "segmentation", "label_path", "gt_path")

    candidates = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith((".csv", ".tsv")):
                candidates.append(os.path.join(dirpath, fn))

    for man in sorted(candidates):
        try:
            with open(man, newline="") as f:
                sample = f.read(8192)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.DictReader(f, dialect=dialect)
                fields = [c.strip().lower() for c in (reader.fieldnames or [])]
                ic = next((c for c in img_cols if c in fields), None)
                mc = next((c for c in mask_cols if c in fields), None)
                if ic is None or mc is None:
                    continue
                base = os.path.dirname(man)
                matched, unmatched = [], []
                for rec in reader:
                    rec = {k.strip().lower(): (v or "").strip()
                           for k, v in rec.items() if k is not None}
                    img = os.path.join(base, rec.get(ic, ""))
                    gt = os.path.join(base, rec.get(mc, ""))
                    if not rec.get(ic) or not os.path.exists(img):
                        continue
                    if not rec.get(mc) or not os.path.exists(gt):
                        unmatched.append(img)
                        continue
                    matched.append({"image": img, "gt": gt, "kind": "mask",
                                    "manifest": man})
        except (OSError, UnicodeDecodeError):
            continue
        if matched:
            print(f"Using manifest {man} ({len(matched)} image/mask pairs listed).")
            return matched, unmatched

    return None, None


def discover_dataset(root_dir):
    """Walk root_dir, find image files, and try to match each to ground
    truth via (in priority order): an explicit CSV/TSV manifest, a companion
    edge-coordinate JSON, a companion Labelme JSON, or a companion raster
    mask. Returns a list of dicts:
    {"image": path_or_None, "gt": path, "kind": "edge_json"|"labelme"|"mask"}
    plus a separate list of image paths that had NO match (report these --
    do not silently drop them)."""
    man_matched, man_unmatched = discover_from_manifest(root_dir)
    if man_matched:
        return man_matched, man_unmatched

    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            all_files.append(os.path.join(dirpath, fn))

    images = [p for p in all_files if p.lower().endswith(IMAGE_EXTS)
              and not _has_hint(os.path.basename(p), MASK_HINTS)]
    masks = [p for p in all_files if p.lower().endswith(IMAGE_EXTS)
              and _has_hint(os.path.basename(p), MASK_HINTS)]
    jsons = [p for p in all_files if p.lower().endswith(".json")]

    by_stem_mask = {_stem(p).lower(): p for p in masks}
    by_digits_mask = {}
    for p in masks:
        d = _digits(_stem(p))
        if d is not None:
            by_digits_mask.setdefault(d, p)

    def json_kind(path):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            return None
        if "shapes" in d:
            return "labelme"
        if any(k in d for k in ("left_edge", "x_L", "xL", "left")):
            return "edge_json"
        return None

    json_kinds = {p: json_kind(p) for p in jsons}
    by_stem_json = {_stem(p).lower(): p for p in jsons}
    by_digits_json = {}
    for p in jsons:
        d = _digits(_stem(p))
        if d is not None:
            by_digits_json.setdefault(d, p)

    matched, unmatched = [], []
    for img in images:
        stem = _stem(img).lower()
        digit = _digits(_stem(img))

        gt_json = by_stem_json.get(stem)
        if gt_json is None and digit is not None:
            gt_json = by_digits_json.get(digit)
        if gt_json is not None and json_kinds.get(gt_json) in ("edge_json", "labelme"):
            matched.append({"image": img, "gt": gt_json, "kind": json_kinds[gt_json]})
            continue

        gt_mask = by_stem_mask.get(stem)
        if gt_mask is None:
            for cand_stem, p in by_stem_mask.items():
                if cand_stem.startswith(stem) or stem.startswith(cand_stem):
                    gt_mask = p
                    break
        if gt_mask is None and digit is not None:
            gt_mask = by_digits_mask.get(digit)
        if gt_mask is not None:
            matched.append({"image": img, "gt": gt_mask, "kind": "mask"})
            continue

        unmatched.append(img)

    return matched, unmatched


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------
def process_item(item, welch_nperseg, welch_noverlap, fit_k):
    kind = item["kind"]
    if kind == "mask":
        mask = _load_image_array(item["gt"])
        x_L, x_R = edges_from_mask(mask)
    elif kind == "edge_json":
        x_L, x_R = edges_from_edge_json(item["gt"])
    elif kind == "labelme":
        x_L, x_R = edges_from_labelme_json(item["gt"])
    else:
        return None

    if x_L is None or x_R is None or len(x_L) < 8:
        return None

    dt_L = detrend_np(x_L)
    dt_R = detrend_np(x_R)
    w = x_R - x_L
    dt_W = detrend_np(w)

    sigma_L = rms_np(dt_L)
    sigma_R = rms_np(dt_R)
    sigma_W = rms_np(dt_W)

    row = {"image": item["image"], "gt_source": item["gt"], "kind": kind,
           "n_rows": len(x_L), "sigma_L": sigma_L, "sigma_R": sigma_R, "sigma_W": sigma_W,
           "mean_linewidth": float(np.mean(w))}

    if fit_k:
        fL, pL = welch_psd_np(dt_L, welch_nperseg, welch_noverlap)
        fR, pR = welch_psd_np(dt_R, welch_nperseg, welch_noverlap)
        fW, pW = welch_psd_np(dt_W, welch_nperseg, welch_noverlap)
        xi_L, alpha_L = fit_xi_alpha(fL, pL, sigma_L)
        xi_R, alpha_R = fit_xi_alpha(fR, pR, sigma_R)
        xi_W, alpha_W = fit_xi_alpha(fW, pW, sigma_W)
        row.update({"xi_L": xi_L, "alpha_L": alpha_L, "xi_R": xi_R, "alpha_R": alpha_R,
                    "xi_W": xi_W, "alpha_W": alpha_W})

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True,
                     help="root directory to search for images + ground truth")
    ap.add_argument("--limit", type=int, default=None,
                     help="only process the first N matched images (for a quick check)")
    ap.add_argument("--welch-nperseg", type=int, default=128)
    ap.add_argument("--welch-noverlap", type=int, default=64)
    ap.add_argument("--fit-k-correlation", action="store_true",
                     help="also fit (xi, alpha) per image for domain-gap comparison")
    ap.add_argument("--out-prefix", type=str, default="real1")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: --data-dir '{args.data_dir}' does not exist or is not a directory.")
        sys.exit(1)

    print(f"Scanning {args.data_dir} ...")
    matched, unmatched = discover_dataset(args.data_dir)
    print(f"Found {len(matched)} image+ground-truth pairs "
          f"({sum(1 for m in matched if m['kind']=='mask')} mask, "
          f"{sum(1 for m in matched if m['kind']=='edge_json')} edge-json, "
          f"{sum(1 for m in matched if m['kind']=='labelme')} labelme).")
    if unmatched:
        print(f"WARNING: {len(unmatched)} image(s) had NO matching ground truth found "
              f"and will be skipped. First few:")
        for p in unmatched[:10]:
            print(f"  {p}")
        print("  -> if these DO have ground truth, your file naming doesn't match any "
              "of the three patterns this script looks for; see the module docstring's "
              "'Adapting to your actual file layout' note, or adjust MASK_HINTS / the "
              "matching logic in discover_dataset().")

    if not matched:
        print("No matched pairs found -- nothing to do. Exiting.")
        sys.exit(1)

    if args.limit:
        matched = matched[:args.limit]
        print(f"(--limit set: processing only the first {len(matched)})")

    rows = []
    n_failed = 0
    for i, item in enumerate(matched):
        row = process_item(item, args.welch_nperseg, args.welch_noverlap, args.fit_k_correlation)
        if row is None:
            n_failed += 1
            continue
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(matched)}")

    if n_failed:
        print(f"WARNING: {n_failed} matched pair(s) failed to yield usable edges "
              f"(e.g. empty mask, malformed JSON, too few rows) and were skipped.")

    if not rows:
        print("All matched pairs failed to process -- check the ground-truth format. Exiting.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_prefix}_per_image.csv", index=False)

    summary_cols = ["sigma_L", "sigma_R", "sigma_W", "mean_linewidth"]
    if args.fit_k_correlation:
        summary_cols += ["xi_L", "alpha_L", "xi_R", "alpha_R", "xi_W", "alpha_W"]
    summary = df[summary_cols].agg(["mean", "std", "min", "max", "count"]).T
    summary.to_csv(f"{args.out_prefix}_summary.csv")

    print(f"\n=== Real-data LER/LWR summary ({len(df)} images) ===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    print(f"\nSynthetic training grid for comparison: "
          f"sigma in {SYNTH_SIGMA_RANGE}, xi in {SYNTH_XI_RANGE}, alpha in {SYNTH_ALPHA_RANGE}")
    lo, hi = SYNTH_SIGMA_RANGE
    frac_in_range = float(((df["sigma_W"] >= lo) & (df["sigma_W"] <= hi)).mean())
    print(f"Fraction of real images with sigma_W (LWR) inside the synthetic sigma "
          f"range {SYNTH_SIGMA_RANGE}: {frac_in_range:.1%} "
          f"(real sigma_W range: [{df['sigma_W'].min():.3f}, {df['sigma_W'].max():.3f}])")
    if args.fit_k_correlation:
        lo_x, hi_x = SYNTH_XI_RANGE
        lo_a, hi_a = SYNTH_ALPHA_RANGE
        frac_xi = float(((df["xi_W"] >= lo_x) & (df["xi_W"] <= hi_x)).mean())
        frac_alpha = float(((df["alpha_W"] >= lo_a) & (df["alpha_W"] <= hi_a)).mean())
        print(f"Fraction with xi_W inside synthetic range: {frac_xi:.1%}; "
              f"alpha_W inside synthetic range: {frac_alpha:.1%}")
    print("\nUse this comparison honestly in the manuscript: if a large fraction of real "
          "images fall OUTSIDE the synthetic (sigma, xi, alpha) training grid, that is a "
          "real domain-gap finding to report, not something to paper over by silently "
          "widening the grid after the fact without re-running Tables 2-4.")


if __name__ == "__main__":
    main()
