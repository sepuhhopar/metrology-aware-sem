"""
Two-edge synthetic SEM line generator, implementing Section 5 (Eqs. 30-39,
Algorithm 1) of the new document ("Metrology-Aware Learning for Noise-Robust
Edge Segmentation and Roughness Estimation in SEM Images").

Extends the single-edge generator used for the original Table I / this
project's earlier reimplementation: both left AND right edges are now
independently-but-correlated self-affine processes, enabling genuine LWR and
left/right cross-correlation studies (out of scope in every prior version).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

N_PROFILE = 512
IMG_WIDTH = 128
PSF_SIGMA = 0.8
MASTER_SEED = 20260827


def k_correlation_psd_shape(f, xi, alpha):
    return 1.0 / (1.0 + (2 * np.pi * f * xi) ** 2) ** (alpha + 0.5)


def synthesize_unit_rms_profile(rng, xi, alpha, n=N_PROFILE):
    """u(y): zero-mean self-affine Gaussian process, target PSD shape (xi,alpha),
    unit RMS after linear detrending (Eq. 30-31's u_c, u_1, u_2)."""
    freqs = np.fft.fftfreq(n, d=1.0)
    shape = k_correlation_psd_shape(np.abs(freqs), xi, alpha)
    amp = np.sqrt(shape)
    spectrum = np.zeros(n, dtype=complex)
    half = n // 2
    phases = rng.uniform(0, 2 * np.pi, size=half - 1)
    spectrum[1:half] = amp[1:half] * np.exp(1j * phases)
    spectrum[half + 1:] = np.conj(spectrum[1:half][::-1])
    spectrum[half] = amp[half] * rng.choice([-1.0, 1.0])
    spectrum[0] = 0.0
    profile = np.fft.ifft(spectrum).real
    y = np.arange(n)
    trend = np.polyfit(y, profile, 1)
    detrended = profile - np.polyval(trend, y)
    rms = np.sqrt(np.mean(detrended ** 2))
    return detrended / rms if rms > 0 else detrended  # unit RMS


def synthesize_correlated_edges(rng, sigma, xi, alpha, rho, w0, x0, n=N_PROFILE):
    """Eqs. 30-33: correlated left/right edge coordinate series."""
    u_c = synthesize_unit_rms_profile(rng, xi, alpha, n)
    u_1 = synthesize_unit_rms_profile(rng, xi, alpha, n)
    u_2 = synthesize_unit_rms_profile(rng, xi, alpha, n)

    sqrt_rho = np.sqrt(abs(rho))
    sqrt_1mrho = np.sqrt(max(0.0, 1.0 - abs(rho)))
    r_L = sigma * (sqrt_rho * u_c + sqrt_1mrho * u_1)
    r_R = sigma * (np.sign(rho) * sqrt_rho * u_c + sqrt_1mrho * u_2) if rho != 0 else sigma * (sqrt_1mrho * u_2)

    x_L = x0 - w0 / 2.0 + r_L
    x_R = x0 + w0 / 2.0 + r_R
    return x_L, x_R


def render_clean_image(x_left, x_right, fg=1.0, bg=0.0, width=IMG_WIDTH):
    """Eq. 35-36: binary mask -> ideal contrast image."""
    n = len(x_left)
    xs = np.arange(width).reshape(1, -1)
    xl = x_left.reshape(-1, 1)
    xr = x_right.reshape(-1, 1)
    mask = ((xs >= xl) & (xs < xr)).astype(np.float64)
    return bg + (fg - bg) * mask


def apply_noise(rng, img_blurred, dose, sigma_g):
    """Eqs. 38-39: Poisson shot noise at effective dose D, additive Gaussian
    read noise, clipped to [0,1]."""
    lam = np.clip(img_blurred, 0.0, 1.0) * dose
    poisson_component = rng.poisson(lam).astype(np.float64) / dose
    gauss = rng.normal(0.0, sigma_g, size=img_blurred.shape)
    return np.clip(poisson_component + gauss, 0.0, 1.0)


def dose_and_read_noise_from_kn(k_n):
    """Convenience mapping preserving continuity with Experiment 0's k_n
    sweep (paper's original Poisson-Gaussian parameterization): D=400/k_n,
    sigma_G=0.005*sqrt(k_n)."""
    return 400.0 / k_n, 0.005 * np.sqrt(k_n)


def generate_paired(rng1, rng2, sigma, xi, alpha, rho, w0=40.0, x0=64.0,
                     fg=1.0, bg=0.0, psf_sigma=PSF_SIGMA,
                     acq1=None, acq2=None, shared_geometry_rng=None):
    """Algorithm 1: sample one physical geometry, render it under two
    independently-sampled acquisition conditions (paired-noise sample).

    acq1/acq2: optional (dose, sigma_g) tuples; if None, drawn from a default
    k_n range for convenience.
    """
    geom_rng = shared_geometry_rng if shared_geometry_rng is not None else rng1
    x_L, x_R = synthesize_correlated_edges(geom_rng, sigma, xi, alpha, rho, w0, x0)
    clean = render_clean_image(x_L, x_R, fg, bg)
    blurred = gaussian_filter(clean, sigma=psf_sigma, mode="reflect")

    if acq1 is None:
        k_n1 = rng1.choice([0.3, 1.0, 2.0, 4.0])
        acq1 = dose_and_read_noise_from_kn(k_n1)
    if acq2 is None:
        k_n2 = rng2.choice([0.3, 1.0, 2.0, 4.0])
        acq2 = dose_and_read_noise_from_kn(k_n2)

    img1 = apply_noise(rng1, blurred, *acq1)
    img2 = apply_noise(rng2, blurred, *acq2)

    return {
        "img1": img1, "img2": img2,
        "x_L_gt": x_L, "x_R_gt": x_R,
        "sigma_gt": sigma, "xi": xi, "alpha": alpha, "rho": rho,
        "w0": w0, "x0": x0,
    }


def master_child_rngs(n, master_seed=MASTER_SEED, salt=0):
    ss = np.random.SeedSequence([master_seed, salt])
    return [np.random.default_rng(c) for c in ss.spawn(n)]


def detrend_np(x):
    y = np.arange(len(x))
    coefs = np.polyfit(y, x, 1)
    return x - np.polyval(coefs, y)


def rms_np(x_detrended):
    return np.sqrt(np.mean(x_detrended ** 2))
