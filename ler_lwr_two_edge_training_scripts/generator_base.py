"""
Synthetic SEM-like line-edge generator, reimplemented from the paper's
Section IV specification:

  - profile length (scan direction, y):        N = 512 px
  - synthetic image width (x):                 W = 128 px
  - nominal line width:                        Lw = 40 px  (edges centered)
  - sampling interval:                         dx = 1.0 px
  - foreground / background intensity:         1.0 / 0.0
  - imaging blur:                              Gaussian PSF, sigma_psf = 0.8 px, reflect boundary
  - noise model:                                Poisson-Gaussian, D(k_n) = 400/k_n,
                                                 sigma_G(k_n) = 0.005*sqrt(k_n)
  - roughness profile:                          inverse-FFT-shaped white noise to the
                                                 two-sided K-correlation PSD (Eq. 3),
                                                 DC=0, Nyquist forced real, linearly
                                                 detrended then rescaled to the exact
                                                 target sigma.
  - master seed 20260827, independent child RNG stream per image (SeedSequence.spawn).

Only the left edge is independently synthesized and scored (single-edge / LER-type
scope, matching the paper). The right edge is rendered at a fixed offset (nominal
line width) from the left edge, for image rendering only.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

N_PROFILE = 512          # scan-direction length (y)
IMG_WIDTH = 128           # x
NOMINAL_LINE_WIDTH = 40.0
NOMINAL_LEFT_X = (IMG_WIDTH - NOMINAL_LINE_WIDTH) / 2.0   # 44.0
NOMINAL_RIGHT_X = NOMINAL_LEFT_X + NOMINAL_LINE_WIDTH      # 84.0
PSF_SIGMA = 0.8
MASTER_SEED = 20260827


def k_correlation_psd_shape(f, xi, alpha):
    """Relative (unnormalized) shape of Eq. 3 -- the prefactor is irrelevant here
    because the generated profile is rescaled to the exact target sigma after
    detrending, so only the spectral *shape* set by (xi, alpha) matters for
    synthesis."""
    return 1.0 / (1.0 + (2 * np.pi * f * xi) ** 2) ** (alpha + 0.5)


def synthesize_edge_profile(rng, sigma, xi, alpha, n=N_PROFILE):
    """Inverse-FFT-shaped fractal profile with target RMS `sigma`, matching
    Section IV: DC=0, Nyquist bin (n even) forced real, linear detrend + exact
    rescale to `sigma`."""
    freqs = np.fft.fftfreq(n, d=1.0)  # cycles/px, includes negative frequencies
    shape = k_correlation_psd_shape(np.abs(freqs), xi, alpha)
    amp = np.sqrt(shape)

    spectrum = np.zeros(n, dtype=complex)
    half = n // 2
    # positive frequencies k=1..half-1
    phases = rng.uniform(0, 2 * np.pi, size=half - 1)
    spectrum[1:half] = amp[1:half] * np.exp(1j * phases)
    # negative frequencies mirror (Hermitian symmetry -> real ifft)
    spectrum[half + 1:] = np.conj(spectrum[1:half][::-1])
    # Nyquist bin (index `half`) must be real for even n
    nyquist_sign = rng.choice([-1.0, 1.0])
    spectrum[half] = amp[half] * nyquist_sign
    # DC = 0
    spectrum[0] = 0.0

    profile = np.fft.ifft(spectrum).real

    y = np.arange(n)
    trend_coefs = np.polyfit(y, profile, 1)
    detrended = profile - np.polyval(trend_coefs, y)

    current_rms = np.sqrt(np.mean(detrended ** 2))
    scale = sigma / current_rms if current_rms > 0 else 1.0
    profile_final = detrended * scale
    return profile_final  # this IS the ground-truth left-edge deviation, RMS==sigma exactly


def render_clean_image(x_left):
    """Render the hard-edged (unblurred, noise-free) line image from the
    left-edge coordinate series x_left(y), right edge at fixed offset."""
    n = len(x_left)
    x_right = x_left + NOMINAL_LINE_WIDTH
    xs = np.arange(IMG_WIDTH).reshape(1, -1)  # (1, W)
    xl = x_left.reshape(-1, 1)                 # (n, 1)
    xr = x_right.reshape(-1, 1)
    img = ((xs >= xl) & (xs < xr)).astype(np.float64)
    return img  # (n, W), foreground=1 background=0


def apply_noise(rng, img, k_n):
    """Poisson-Gaussian mixed noise per Section IV."""
    dose = 400.0 / k_n
    lam = np.clip(img, 0.0, 1.0) * dose
    poisson_component = rng.poisson(lam).astype(np.float64) / dose
    sigma_g = 0.005 * np.sqrt(k_n)
    gauss = rng.normal(0.0, sigma_g, size=img.shape)
    noisy = poisson_component + gauss
    return np.clip(noisy, 0.0, 1.0)


def generate_one(rng, sigma, xi, alpha, k_n, left_x0=NOMINAL_LEFT_X):
    """Generate one full synthetic realization.

    Returns: dict with noisy image, ground-truth left-edge coordinate series,
    and the nominal left-edge x for detector search windows.
    """
    profile = synthesize_edge_profile(rng, sigma, xi, alpha)
    x_left_gt = left_x0 + profile
    clean = render_clean_image(x_left_gt)
    blurred = gaussian_filter(clean, sigma=PSF_SIGMA, mode="reflect")
    noisy = apply_noise(rng, blurred, k_n)
    return {
        "image": noisy,            # (N_PROFILE, IMG_WIDTH)
        "x_left_gt": x_left_gt,    # (N_PROFILE,)
        "nominal_x0": left_x0,
        "sigma_gt": sigma,
        "xi": xi,
        "alpha": alpha,
        "k_n": k_n,
    }


def master_child_rngs(n_images, master_seed=MASTER_SEED):
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(n_images)
    return [np.random.default_rng(c) for c in children]


def detrended_rms(x_series):
    y = np.arange(len(x_series))
    coefs = np.polyfit(y, x_series, 1)
    detrended = x_series - np.polyval(coefs, y)
    return np.sqrt(np.mean(detrended ** 2))
