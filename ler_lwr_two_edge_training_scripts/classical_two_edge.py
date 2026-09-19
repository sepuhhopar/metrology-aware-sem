"""
Classical baseline detectors + localization, reimplemented from Section III-A / IV:

  - Threshold: 50%-intensity rising crossing, +-10 px search window around the
    nominal edge position, linear interpolation for sub-pixel.
  - Canny: image scaled to 8-bit, cv2.Canny(50, 150, L2gradient=True), nearest
    detection to nominal position within the window retained pixel-quantized;
    rows without detection filled by linear interpolation across neighboring rows.
  - Gaussian denoise + threshold: 5x5 Gaussian blur (sigma=1.0, reflect) then the
    same 50%-threshold sub-pixel procedure.

Each detector is implemented with BOTH a sub-pixel and a pixel-quantized
localization mode, so Section V-A' (factorial ablation) can cross detector x
localization independently -- the comparison the original paper's Section V-A
explicitly could not make because detector and localization were varied together.
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

SEARCH_WINDOW = 10
THRESH_LEVEL = 0.5


def _rising_crossing_subpixel(profile_1d, center, window=SEARCH_WINDOW, level=THRESH_LEVEL,
                               falling=False):
    """Find the sub-pixel crossing of `level` nearest `center`. By default
    looks for a RISING (background->foreground) crossing, correct for a left
    edge. Pass falling=True for a right/trailing edge (foreground->background),
    e.g. the right boundary of a two-edge line -- NOT needed for the original
    single-edge (left-only) use of this module, only for classical_two_edge's
    two-edge callers. Returns float x, or np.nan if none found."""
    w = int(round(window))
    x0 = max(0, int(round(center)) - w)
    x1 = min(len(profile_1d) - 1, int(round(center)) + w)
    seg = profile_1d[x0:x1 + 1]
    crossings = []
    for i in range(len(seg) - 1):
        a, b = seg[i], seg[i + 1]
        if not falling and a < level <= b:
            frac = (level - a) / (b - a) if b != a else 0.0
            crossings.append(x0 + i + frac)
        elif falling and a >= level > b:
            frac = (a - level) / (a - b) if a != b else 0.0
            crossings.append(x0 + i + frac)
    if not crossings:
        return np.nan
    crossings = np.array(crossings)
    return crossings[np.argmin(np.abs(crossings - center))]


def threshold_detector(image, nominal_x0, subpixel=True, falling=False):
    n = image.shape[0]
    out = np.full(n, np.nan)
    for y in range(n):
        est = _rising_crossing_subpixel(image[y], nominal_x0, falling=falling)
        out[y] = est
    if not subpixel:
        out = np.round(out)
    return _fill_nan_rows(out)


def gaussian_denoise_threshold_detector(image, nominal_x0, subpixel=True, denoise_sigma=1.0,
                                         falling=False):
    denoised = gaussian_filter(image, sigma=(0.0, denoise_sigma), mode="reflect")
    # 5x5 kernel ~ sigma=1.0 in the row direction only (per-row 1-D profile denoise
    # matches the paper's "5x5 Gaussian blur" applied to the 2-D image; use a
    # 2-D blur to match the stated 5x5 kernel more literally):
    denoised = gaussian_filter(image, sigma=denoise_sigma, mode="reflect")
    return threshold_detector(denoised, nominal_x0, subpixel=subpixel, falling=falling)


def canny_detector(image, nominal_x0, subpixel=True, low=50, high=150, falling=False):
    img8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(img8, low, high, L2gradient=True)
    n = image.shape[0]
    out = np.full(n, np.nan)
    w = SEARCH_WINDOW
    x0w = max(0, int(round(nominal_x0)) - w)
    x1w = min(image.shape[1] - 1, int(round(nominal_x0)) + w)
    for y in range(n):
        row = edges[y, x0w:x1w + 1]
        idx = np.nonzero(row)[0]
        if len(idx) == 0:
            continue
        candidates = idx + x0w
        best = candidates[np.argmin(np.abs(candidates - nominal_x0))]
        out[y] = float(best)
    out = _fill_nan_rows(out)  # linear interpolation across neighboring rows
    if subpixel:
        # sub-pixel refinement: intensity-based 50% crossing in a small window
        # around the pixel-quantized Canny location (detector locates coarsely,
        # sub-pixel step refines) -- isolates localization from detector choice.
        refined = np.copy(out)
        for y in range(n):
            c = out[y]
            if np.isnan(c):
                continue
            est = _rising_crossing_subpixel(image[y], c, window=2, falling=falling)
            if not np.isnan(est):
                refined[y] = est
        out = refined
    return out


def _fill_nan_rows(x):
    """Linear interpolation across neighboring rows (y) for missing detections."""
    x = x.copy()
    nan_mask = np.isnan(x)
    if nan_mask.all():
        return np.zeros_like(x)
    if nan_mask.any():
        idx = np.arange(len(x))
        x[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x[~nan_mask])
    return x


DETECTORS = {
    "threshold": threshold_detector,
    "canny": canny_detector,
    "gaussian_denoise_threshold": gaussian_denoise_threshold_detector,
}
