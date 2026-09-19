"""
Experiment 9 (Sections 7.3/7.5 follow-up): direct PSD-error evaluation and
noise-robustness / repeated-acquisition (repeatability) evaluation.

WHY THIS EXISTS
---------------
Table 3 (run_experiment2.py) scores every loss variant with edge_mae, ler_mae
and lwr_mae. None of those three can see what the last two loss terms actually
optimize:

  * L_PSD (Eq. 25) shapes the *spectrum* of the predicted edge trace. ler_mae
    and lwr_mae collapse that whole spectrum into a single RMS scalar -- two
    traces with very different PSDs can have identical RMS. So a PSD term can
    be doing real work and still look like a ~0% change in Table 3.
  * L_cons (Eqs. 27-28) penalizes disagreement between two noisy acquisitions
    of the SAME physical geometry. It is a *precision* (repeatability) term.
    Every Table 3 metric is an accuracy-vs-ground-truth metric averaged over
    independent samples, which is precisely the average L_cons does not change.

This script therefore measures the two matching quantities directly:

  Part A -- log-PSD error between the predicted and ground-truth edge traces,
  over the same band used by the training loss (1/H <= |f| <= 0.25), and split
  into low/mid/high frequency bands so you can see WHERE any gain lands.

  Part B -- repeatability: one fixed geometry imaged K times under independent
  noise, reporting the spread of the estimate across repeats (precision), held
  separately from its bias vs ground truth (accuracy). Swept over the k_n noise
  grid, which doubles as the noise-robustness curve.

Both parts are evaluated on the SAME trained models, so one training pass feeds
both tables.

Pilot scale by default. Submission scale:

    python3 run_experiment9.py --epochs 40 --seeds 3 --train-per-config 15 \
        --eval-per-config 15 --rhos 0.0 0.3 0.6 -0.3
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

from generator2 import (master_child_rngs, synthesize_correlated_edges,
                         render_clean_image, apply_noise,
                         dose_and_read_noise_from_kn, PSF_SIGMA)
from model2 import (H, differentiable_detrend, rms, differentiable_psd)
from train2 import (X0, W0, KNS, build_paired_dataset, train_one_variant,
                     _edge_forward, DEVICE, set_calibration, load_calibration)
from losses2 import VARIANTS

SIGMAS = [1.5, 3.0]
XIS = [8.0, 20.0]
ALPHAS = [0.3, 0.7]

# Frequency bands for the PSD breakdown. "all" is exactly the band the training
# loss optimizes (log_psd_loss's mask), so that row is directly comparable to
# the L_PSD value seen during training; the three sub-bands partition it.
PSD_BANDS = {
    "all":  (1.0 / H, 0.25),
    "low":  (1.0 / H, 0.05),
    "mid":  (0.05, 0.15),
    "high": (0.15, 0.25),
}


# ---------------------------------------------------------------------------
# Part A: direct PSD error
# ---------------------------------------------------------------------------
def _band_log_psd_mae(freqs, psd_pred, psd_gt, lo, hi, eps=1e-6):
    """Mean |log PSD_pred - log PSD_gt| over the band lo <= |f| <= hi.
    Same log-domain comparison as log_psd_loss, restricted to one band."""
    af = freqs.abs()
    mask = (af >= lo) & (af <= hi) & (freqs != 0)
    mask = mask.to(psd_pred.device)
    if mask.sum() == 0:
        return float("nan")
    lp = torch.log(psd_pred[:, mask] + eps)
    lg = torch.log(psd_gt[:, mask] + eps)
    return float((lp - lg).abs().mean().item())


@torch.no_grad()
def psd_error_for_model(model, items, label, seed):
    """Per-image log-PSD error for the L, R and W traces, by frequency band."""
    rows = []
    for it in items:
        xhL, _, _, gtL, xhR, _, _, gtR = _edge_forward(
            model, it["img1"][None, ...], it["x_L_gt"][None, ...],
            it["x_R_gt"][None, ...], tau=0.05)

        traces = {
            "L": (differentiable_detrend(xhL), differentiable_detrend(gtL)),
            "R": (differentiable_detrend(xhR), differentiable_detrend(gtR)),
            "W": (differentiable_detrend(xhR - xhL),
                  differentiable_detrend(gtR - gtL)),
        }
        row = {"method": label, "seed": seed, "config": it["config"]}
        for tname, (dt_p, dt_g) in traces.items():
            f, psd_p = differentiable_psd(dt_p)
            _, psd_g = differentiable_psd(dt_g)
            for bname, (lo, hi) in PSD_BANDS.items():
                row[f"logpsd_{tname}_{bname}"] = _band_log_psd_mae(
                    f, psd_p, psd_g, lo, hi)
        # mean over L,R,W in the training band -- the single headline number
        row["logpsd_mean_all"] = float(np.mean(
            [row["logpsd_L_all"], row["logpsd_R_all"], row["logpsd_W_all"]]))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Part B: repeated-acquisition / noise robustness
# ---------------------------------------------------------------------------
def generate_repeat_stack(geom_rng, noise_rngs, sigma, xi, alpha, rho, k_n,
                           w0=W0, x0=X0, cal=None):
    """One physical geometry, imaged len(noise_rngs) times under independent
    noise at a fixed k_n. Reuses the generator's own rendering path so the
    images are identical in kind to the training data.

    cal: calibration dict. When given, the measured PSF and read noise are used
    and k_n drives the dose only (dose = 400/k_n, as in the default model), so
    the noise-robustness sweep is retained under realistic optics rather than
    collapsing to a single operating point."""
    x_L, x_R = synthesize_correlated_edges(geom_rng, sigma, xi, alpha, rho, w0, x0)
    clean = render_clean_image(x_L, x_R)
    psf = cal["psf_sigma"] if cal is not None else PSF_SIGMA
    blurred = gaussian_filter(clean, sigma=psf, mode="reflect")
    dose, sigma_g = dose_and_read_noise_from_kn(k_n)
    if cal is not None:
        sigma_g = cal["sigma_g"]
    imgs = np.stack([apply_noise(r, blurred, dose, sigma_g) for r in noise_rngs])
    return imgs, x_L, x_R


@torch.no_grad()
def repeatability_for_model(model, configs, n_geom, n_repeats, kns, label, seed,
                             salt, cal=None):
    """For each (config, k_n, geometry): predict on n_repeats independent noisy
    acquisitions of that one geometry and report

      precision: std across repeats of the estimate (what L_cons targets)
      accuracy : |mean-across-repeats estimate - ground truth| (what Table 3
                 targets)

    Reporting them separately matters: a variant can be more repeatable and no
    more accurate, which is exactly the claim a consistency term should make.
    """
    rows = []
    n_items = len(configs) * len(kns) * n_geom
    geom_rngs = master_child_rngs(n_items, salt=salt)
    noise_rngs = master_child_rngs(n_items * n_repeats, salt=salt + 1)
    idx = 0
    for (sigma, xi, alpha, rho) in configs:
        for k_n in kns:
            for g in range(n_geom):
                nrs = noise_rngs[idx * n_repeats:(idx + 1) * n_repeats]
                imgs, x_L, x_R = generate_repeat_stack(
                    geom_rngs[idx], nrs, sigma, xi, alpha, rho, k_n, cal=cal)
                idx += 1

                K = imgs.shape[0]
                xL_rep = np.repeat(x_L[None, :], K, axis=0)
                xR_rep = np.repeat(x_R[None, :], K, axis=0)
                xhL, _, _, gtL, xhR, _, _, gtR = _edge_forward(
                    model, imgs, xL_rep, xR_rep, tau=0.05)

                dt_hat_L = differentiable_detrend(xhL)
                dt_hat_R = differentiable_detrend(xhR)
                dt_hat_W = differentiable_detrend(xhR - xhL)
                s_hat_L, s_hat_R, s_hat_W = rms(dt_hat_L), rms(dt_hat_R), rms(dt_hat_W)

                # ground truth is identical across repeats (same geometry)
                s_gt_L = float(rms(differentiable_detrend(gtL))[0].item())
                s_gt_R = float(rms(differentiable_detrend(gtR))[0].item())
                s_gt_W = float(rms(differentiable_detrend(gtR - gtL))[0].item())

                # precision: spread across repeats (unbiased, K-1)
                prec_L = float(s_hat_L.std(unbiased=True).item())
                prec_R = float(s_hat_R.std(unbiased=True).item())
                prec_W = float(s_hat_W.std(unbiased=True).item())
                # positional jitter: per-row std across repeats, averaged
                pos_jitter = float(0.5 * (xhL.std(dim=0, unbiased=True).mean().item()
                                           + xhR.std(dim=0, unbiased=True).mean().item()))

                # accuracy: bias of the repeat-averaged estimate
                bias_L = abs(float(s_hat_L.mean().item()) - s_gt_L)
                bias_R = abs(float(s_hat_R.mean().item()) - s_gt_R)
                bias_W = abs(float(s_hat_W.mean().item()) - s_gt_W)

                # per-repeat accuracy, comparable to Table 3's ler/lwr_mae
                ler_mae = float(0.5 * ((s_hat_L - s_gt_L).abs().mean().item()
                                        + (s_hat_R - s_gt_R).abs().mean().item()))
                lwr_mae = float((s_hat_W - s_gt_W).abs().mean().item())
                edge_mae = float(0.5 * ((xhL - gtL).abs().mean().item()
                                         + (xhR - gtR).abs().mean().item()))

                rows.append({
                    "method": label, "seed": seed, "k_n": k_n,
                    "config": (sigma, xi, alpha, rho),
                    "precision_sigma_L": prec_L, "precision_sigma_R": prec_R,
                    "precision_sigma_W": prec_W, "pos_jitter": pos_jitter,
                    "bias_sigma_L": bias_L, "bias_sigma_R": bias_R,
                    "bias_sigma_W": bias_W,
                    "ler_mae": ler_mae, "lwr_mae": lwr_mae, "edge_mae": edge_mae,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--train-per-config", type=int, default=3)
    ap.add_argument("--eval-per-config", type=int, default=4)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.6])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--base-channels", type=int, default=12)
    ap.add_argument("--n-repeats", type=int, default=8,
                     help="acquisitions per geometry for the repeatability test")
    ap.add_argument("--n-geom", type=int, default=4,
                     help="distinct geometries per (config, k_n) cell")
    ap.add_argument("--kns", type=float, nargs="+", default=KNS,
                     help="noise levels for the robustness sweep")
    ap.add_argument("--out-prefix", type=str, default="exp9")
    ap.add_argument("--calibration", type=str, default=None,
                     help="JSON from calibrate_generator.py: render under imaging "
                          "conditions measured from a real SEM instead of the "
                          "generator's optimistic defaults")
    ap.add_argument("--psf-transfer", type=str, default="psf_sigma_ratio_matched",
                     choices=["psf_sigma_ratio_matched", "psf_sigma_absolute"])
    args = ap.parse_args()

    cal = None
    if args.calibration:
        cal = load_calibration(args.calibration, args.psf_transfer)
        set_calibration(cal)
        # centre the noise sweep on the measured operating point rather than on
        # the generator's default grid, unless the user asked for specific k_n
        if args.kns == KNS:
            kn_eq = 400.0 / cal["dose"]
            args.kns = [kn_eq / 2.0, kn_eq, kn_eq * 2.0, kn_eq * 4.0]
            print(f"noise sweep centred on measured k_n = {kn_eq:.2f}: "
                  f"{[round(k,2) for k in args.kns]}")

    configs = [(s, x, a, r) for s in SIGMAS for x in XIS for a in ALPHAS
               for r in args.rhos]
    print(f"Device: {DEVICE}")
    print(f"Configs: {len(configs)}  (sigma x xi x alpha x rho grid)")
    print(f"Repeatability cells: {len(configs)} configs x {len(args.kns)} k_n "
          f"x {args.n_geom} geometries x {args.n_repeats} repeats")

    train_items = build_paired_dataset(configs, args.train_per_config, salt=900)
    print(f"Training set: {len(train_items)} paired items")

    psd_rows, rep_rows = [], []
    for variant in VARIANTS:
        for seed in range(args.seeds):
            print(f"\n=== Training '{variant}' seed {seed} ===")
            t0 = time.time()
            model = train_one_variant(variant, train_items, epochs=args.epochs,
                                       batch_size=args.batch_size, seed=seed,
                                       base_channels=args.base_channels)
            model.eval()
            print(f"  trained in {time.time()-t0:.1f}s; evaluating...")

            eval_items = build_paired_dataset(configs, args.eval_per_config,
                                               salt=901 + seed)
            psd_rows += psd_error_for_model(model, eval_items, variant, seed)
            rep_rows += repeatability_for_model(
                model, configs, args.n_geom, args.n_repeats, args.kns,
                variant, seed, salt=950 + seed * 2, cal=cal)

    order = list(VARIANTS.keys())

    # ---- Part A output ----
    psd_df = pd.DataFrame(psd_rows)
    psd_df.to_csv(f"{args.out_prefix}_psd_raw.csv", index=False)
    band_cols = ([f"logpsd_{t}_{b}" for t in ("L", "R", "W") for b in PSD_BANDS]
                 + ["logpsd_mean_all"])
    psd_sum = psd_df.groupby("method")[band_cols].mean().reset_index()
    psd_sum["method"] = pd.Categorical(psd_sum["method"], categories=order, ordered=True)
    psd_sum = psd_sum.sort_values("method")
    psd_sum.to_csv(f"{args.out_prefix}_psd_summary.csv", index=False)

    print("\n=== Part A: direct log-PSD error (lower is better) ===")
    print("Headline = mean over L,R,W in the training band 1/H <= |f| <= 0.25:")
    print(psd_sum[["method", "logpsd_mean_all", "logpsd_L_all", "logpsd_R_all",
                   "logpsd_W_all"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nBand breakdown for the width (W) trace:")
    print(psd_sum[["method"] + [f"logpsd_W_{b}" for b in PSD_BANDS]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- Part B output ----
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(f"{args.out_prefix}_repeat_raw.csv", index=False)

    rep_sum = rep_df.groupby("method").agg(
        precision_sigma_W=("precision_sigma_W", "mean"),
        precision_sigma_L=("precision_sigma_L", "mean"),
        precision_sigma_R=("precision_sigma_R", "mean"),
        pos_jitter=("pos_jitter", "mean"),
        bias_sigma_W=("bias_sigma_W", "mean"),
        lwr_mae=("lwr_mae", "mean"), ler_mae=("ler_mae", "mean"),
        n=("precision_sigma_W", "count"),
    ).reset_index()
    rep_sum["method"] = pd.Categorical(rep_sum["method"], categories=order, ordered=True)
    rep_sum = rep_sum.sort_values("method")
    rep_sum.to_csv(f"{args.out_prefix}_repeat_summary.csv", index=False)

    print("\n=== Part B1: repeatability across independent acquisitions "
          "of the SAME geometry ===")
    print("precision_* = std of the estimate across repeats (what L_cons targets);")
    print("bias_sigma_W = |repeat-averaged estimate - ground truth| (accuracy).")
    print(rep_sum.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    noise_sum = rep_df.groupby(["method", "k_n"]).agg(
        precision_sigma_W=("precision_sigma_W", "mean"),
        lwr_mae=("lwr_mae", "mean"), ler_mae=("ler_mae", "mean"),
        edge_mae=("edge_mae", "mean"),
    ).reset_index()
    noise_sum["method"] = pd.Categorical(noise_sum["method"], categories=order, ordered=True)
    noise_sum = noise_sum.sort_values(["method", "k_n"])
    noise_sum.to_csv(f"{args.out_prefix}_noise_summary.csv", index=False)

    print("\n=== Part B2: noise robustness (k_n sweep) ===")
    for metric in ("lwr_mae", "precision_sigma_W"):
        piv = noise_sum.pivot(index="method", columns="k_n", values=metric)
        piv = piv.reindex(order)
        print(f"\n{metric} by k_n (higher k_n = noisier):")
        print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    print(f"\n(scale: {args.seeds} seed(s), {args.epochs} epochs, "
          f"{args.train_per_config} train realizations/config, {len(configs)} configs, "
          f"{args.n_repeats} repeats x {args.n_geom} geometries per cell)")


if __name__ == "__main__":
    main()
