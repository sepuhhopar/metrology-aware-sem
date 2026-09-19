"""
Qualitative figure: predicted vs ground-truth edges on synthetic two-edge SEM
images, at two noise levels.

Produces one PNG with two rows (one per example). Each row shows

  (a) the noisy SEM image with the ground-truth and predicted left/right edges
      overlaid,
  (b) the edge traces x(y) themselves, zoomed to a window where the roughness
      is legible at print size,
  (c) the residual x_hat - x_gt per row, which is what edge_mae averages.

Trains one model (default: the 'full' variant) on the same grid the tables use,
so the picture and the numbers come from the same setup.

    python3 make_edge_figure.py --epochs 40 --train-per-config 15 \
        --rhos 0.0 0.3 0.6 -0.3 --out edges.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generator2 import (master_child_rngs, synthesize_correlated_edges,
                         render_clean_image, apply_noise,
                         dose_and_read_noise_from_kn, PSF_SIGMA)
from scipy.ndimage import gaussian_filter

from model2 import differentiable_detrend, rms
from train2 import X0, W0, build_paired_dataset, train_one_variant, _edge_forward, DEVICE

SIGMAS = [1.5, 3.0]
XIS = [8.0, 20.0]
ALPHAS = [0.3, 0.7]

C_GT = "#2ca02c"     # ground truth
C_PRED = "#d62728"   # prediction


def make_example(seed, sigma, xi, alpha, rho, k_n):
    """One geometry rendered at one noise level."""
    geom_rng, noise_rng = master_child_rngs(2, salt=seed)
    x_L, x_R = synthesize_correlated_edges(geom_rng, sigma, xi, alpha, rho, W0, X0)
    clean = render_clean_image(x_L, x_R)
    blurred = gaussian_filter(clean, sigma=PSF_SIGMA, mode="reflect")
    dose, sigma_g = dose_and_read_noise_from_kn(k_n)
    img = apply_noise(noise_rng, blurred, dose, sigma_g)
    return img, x_L, x_R


@torch.no_grad()
def predict(model, img, x_L, x_R):
    xhL, _, _, gtL, xhR, _, _, gtR = _edge_forward(
        model, img[None, ...], x_L[None, ...], x_R[None, ...], tau=0.05)
    to_np = lambda t: t[0].detach().cpu().numpy()
    return to_np(xhL), to_np(xhR), to_np(gtL), to_np(gtR)


def metrics(xhL, xhR, gtL, gtR):
    t = lambda a: torch.from_numpy(a.astype(np.float32))[None, ...]
    d = lambda a: differentiable_detrend(t(a))
    s = lambda a: float(rms(d(a))[0].item())
    edge_mae = 0.5 * (np.abs(xhL - gtL).mean() + np.abs(xhR - gtR).mean())
    return {
        "edge_mae": float(edge_mae),
        "sigma_L_hat": s(xhL), "sigma_L_gt": s(gtL),
        "sigma_R_hat": s(xhR), "sigma_R_gt": s(gtR),
        "sigma_W_hat": s(xhR - xhL), "sigma_W_gt": s(gtR - gtL),
    }


def draw_row(axes, img, xhL, xhR, gtL, gtR, m, title, zoom):
    y = np.arange(len(gtL))
    y0, y1 = zoom

    ax = axes[0]
    ax.imshow(img, cmap="gray", aspect="auto", origin="lower",
              extent=[0, img.shape[1], 0, img.shape[0]], vmin=0, vmax=1)
    ax.plot(gtL, y, color=C_GT, lw=1.0, label="ground truth")
    ax.plot(gtR, y, color=C_GT, lw=1.0)
    ax.plot(xhL, y, color=C_PRED, lw=0.8, ls="--", label="predicted")
    ax.plot(xhR, y, color=C_PRED, lw=0.8, ls="--")
    ax.axhspan(y0, y1, color="w", alpha=0.18, lw=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (px)", fontsize=8)
    ax.set_ylabel("y (scan direction, px)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    ax = axes[1]
    ax.plot(gtL[y0:y1], y[y0:y1], color=C_GT, lw=1.4, label="GT left")
    ax.plot(xhL[y0:y1], y[y0:y1], color=C_PRED, lw=1.1, ls="--", label="pred left")
    ax.plot(gtR[y0:y1], y[y0:y1], color=C_GT, lw=1.4, alpha=0.55, label="GT right")
    ax.plot(xhR[y0:y1], y[y0:y1], color=C_PRED, lw=1.1, ls="--", alpha=0.55,
            label="pred right")
    ax.set_title(f"edge traces, rows {y0}-{y1}", fontsize=9)
    ax.set_xlabel("x (px)", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(xhL - gtL, y, lw=0.6, color="#1f77b4", label="left")
    ax.plot(xhR - gtR, y, lw=0.6, color="#ff7f0e", label="right", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"residual  (edge MAE {m['edge_mae']:.3f} px)", fontsize=9)
    ax.set_xlabel(r"$\hat{x}-x_{gt}$ (px)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, default="full")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-per-config", type=int, default=15)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.3, 0.6, -0.3])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--kns", type=float, nargs=2, default=[1.0, 4.0],
                     help="noise level for example 1 and example 2")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--xi", type=float, default=20.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="edges.png")
    args = ap.parse_args()

    configs = [(s, x, a, r) for s in SIGMAS for x in XIS for a in ALPHAS
               for r in args.rhos]
    print(f"Device: {DEVICE}; training '{args.variant}' on {len(configs)} configs")
    train_items = build_paired_dataset(configs, args.train_per_config, salt=700)
    model = train_one_variant(args.variant, train_items, epochs=args.epochs,
                               batch_size=args.batch_size, seed=args.seed)
    model.eval()

    fig, axs = plt.subplots(2, 3, figsize=(13, 9),
                            gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]})
    for i, k_n in enumerate(args.kns):
        img, x_L, x_R = make_example(1000 + i, args.sigma, args.xi, args.alpha,
                                      args.rho, k_n)
        xhL, xhR, gtL, gtR = predict(model, img, x_L, x_R)
        m = metrics(xhL, xhR, gtL, gtR)
        title = (f"Example {i+1}: $k_n$={k_n:g} "
                 f"({'low' if k_n <= 1 else 'high'} noise)\n"
                 f"$\\sigma$={args.sigma:g}, $\\xi$={args.xi:g}, "
                 f"$\\alpha$={args.alpha:g}, $\\rho$={args.rho:g}")
        draw_row(axs[i], img, xhL, xhR, gtL, gtR, m, title, zoom=(180, 300))

        print(f"\n--- Example {i+1} (k_n={k_n:g}) ---")
        print(f"  edge MAE      : {m['edge_mae']:.4f} px")
        print(f"  LER left  sigma_L : pred {m['sigma_L_hat']:.3f} vs GT {m['sigma_L_gt']:.3f} px"
              f"  (err {abs(m['sigma_L_hat']-m['sigma_L_gt']):.4f})")
        print(f"  LER right sigma_R : pred {m['sigma_R_hat']:.3f} vs GT {m['sigma_R_gt']:.3f} px"
              f"  (err {abs(m['sigma_R_hat']-m['sigma_R_gt']):.4f})")
        print(f"  LWR       sigma_W : pred {m['sigma_W_hat']:.3f} vs GT {m['sigma_W_gt']:.3f} px"
              f"  (err {abs(m['sigma_W_hat']-m['sigma_W_gt']):.4f})")

    fig.suptitle(f"Predicted vs ground-truth edges -- U-Net '{args.variant}' variant, "
                 f"{args.epochs} epochs (synthetic data, known ground truth)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
