"""
Experiment 1 (Section 7.1-7.2 of the new document): Table 2, the two-edge
fidelity/baseline comparison. Rows:

  1. Classical: Threshold + sub-pixel   -- the Section III-A baseline,
     extended to two-edge data (same detector used throughout this repo).
  2. Learned denoiser + threshold       -- LEFT AS [TBD]. This would require
     training a separate denoising network (e.g. DnCNN/U-Net-as-denoiser)
     ahead of the classical threshold step, which is a distinct piece of
     work from the metrology-aware *localization* pipeline this repo
     implements. We do not fabricate a number for this row; fill it in only
     after actually training and evaluating such a model.
  3. Learned (U-Net), Seg-only          -- train2.py variant "seg-only"
  4. Learned (U-Net), Full              -- train2.py variant "full"

Metrics match every other table in this repo: edge_mae (pixels), ler_mae,
lwr_mae (both in roughness-sigma units).

Usage (pilot scale, default -- a few minutes on CPU):
    python3 run_experiment1.py

Submission-quality scale (see README.md for guidance):
    python3 run_experiment1.py --epochs 40 --seeds 3 --train-per-config 15 \
        --eval-per-config 15 --classical-per-config 15 --rhos 0.0 0.3 0.6 -0.3
"""
import argparse
import time
import numpy as np
import pandas as pd

from generator2 import (master_child_rngs, generate_paired, detrend_np, rms_np,
                         dose_and_read_noise_from_kn)
from classical_two_edge import threshold_detector
from train2 import (build_paired_dataset, train_one_variant, evaluate_model,
                     set_calibration, load_calibration)

SIGMAS = [1.5, 3.0]
XIS = [8.0, 20.0]
ALPHAS = [0.3, 0.7]
KNS = [0.3, 1.0, 2.0, 4.0]
X0, W0 = 64.0, 40.0

ROW_ORDER = [
    "Classical: Threshold + sub-pixel",
    "Learned denoiser + threshold [TBD -- not implemented]",
    "Learned (U-Net), Seg-only",
    "Learned (U-Net), Full",
]


def run_classical_threshold(configs, per_config, salt=555, cal=None):
    """cal: calibration dict, or None for the generator's default k_n grid.
    The classical baseline must see the SAME imaging conditions as the learned
    rows, otherwise the comparison is between an optimistic classical run and a
    realistic learned one. Under calibration the k_n sweep collapses to the one
    measured acquisition, so per_config is scaled up to keep the row's trial
    count comparable to the uncalibrated case."""
    acq_grid = ([(cal["dose"], cal["sigma_g"])] if cal is not None
                else [dose_and_read_noise_from_kn(k) for k in KNS])
    extra_kw = {"psf_sigma": cal["psf_sigma"]} if cal is not None else {}
    if cal is not None:
        per_config = per_config * len(KNS)

    n_items = len(configs) * len(acq_grid) * per_config
    rngs = master_child_rngs(n_items, salt=salt)
    rows = []
    idx = 0
    for (sigma, xi, alpha, rho) in configs:
        for (dose, sigma_g) in acq_grid:
            for _ in range(per_config):
                rng = rngs[idx]; idx += 1
                sample = generate_paired(rng, rng, sigma, xi, alpha, rho, w0=W0, x0=X0,
                                          acq1=(dose, sigma_g), acq2=(dose, sigma_g),
                                          **extra_kw)
                img = sample["img1"]
                xL_gt, xR_gt = sample["x_L_gt"], sample["x_R_gt"]
                sigma_gt_L = rms_np(detrend_np(xL_gt))
                sigma_gt_R = rms_np(detrend_np(xR_gt))
                sigma_gt_W = rms_np(detrend_np(xR_gt - xL_gt))

                xL_est = threshold_detector(img, X0 - W0 / 2, subpixel=True, falling=False)
                xR_est = threshold_detector(img, X0 + W0 / 2, subpixel=True, falling=True)
                edge_mae = 0.5 * (np.mean(np.abs(xL_est - xL_gt)) + np.mean(np.abs(xR_est - xR_gt)))
                sigma_L = rms_np(detrend_np(xL_est))
                sigma_R = rms_np(detrend_np(xR_est))
                sigma_W = rms_np(detrend_np(xR_est - xL_est))
                ler_mae = 0.5 * (abs(sigma_L - sigma_gt_L) + abs(sigma_R - sigma_gt_R))
                lwr_mae = abs(sigma_W - sigma_gt_W)
                rows.append({"method": ROW_ORDER[0], "edge_mae": edge_mae,
                             "ler_mae": ler_mae, "lwr_mae": lwr_mae})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--train-per-config", type=int, default=3)
    ap.add_argument("--eval-per-config", type=int, default=4)
    ap.add_argument("--classical-per-config", type=int, default=5)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.6])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--base-channels", type=int, default=12)
    ap.add_argument("--out-prefix", type=str, default="exp1")
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

    configs = [(s, x, a, r) for s in SIGMAS for x in XIS for a in ALPHAS for r in args.rhos]
    print(f"Configs: {len(configs)}  (sigma x xi x alpha x rho grid)")

    print(f"\n=== Row 1/4: {ROW_ORDER[0]} ===")
    df_classical = run_classical_threshold(configs, args.classical_per_config, cal=cal)

    train_items = build_paired_dataset(configs, args.train_per_config, salt=400)
    print(f"\nTraining set: {len(train_items)} paired items")

    learned_frames = [df_classical]
    cost_rows = []
    for variant, label in [("seg-only", ROW_ORDER[2]), ("full", ROW_ORDER[3])]:
        for seed in range(args.seeds):
            print(f"\n=== Training '{variant}' (seed {seed}) for Table 2 ===")
            t0 = time.time()
            model = train_one_variant(variant, train_items, epochs=args.epochs,
                                       batch_size=args.batch_size, seed=seed,
                                       base_channels=args.base_channels)
            train_time = time.time() - t0
            n_params = sum(p.numel() for p in model.parameters())
            cost_rows.append({"variant": variant, "seed": seed,
                               "train_seconds": train_time, "n_params": n_params})

            df = evaluate_model(model, configs, args.eval_per_config, salt=500 + seed, label=label)
            learned_frames.append(df[["method", "edge_mae", "ler_mae", "lwr_mae"]])

    df_all = pd.concat(learned_frames, ignore_index=True)
    df_all.to_csv(f"{args.out_prefix}_raw.csv", index=False)

    summary = df_all.groupby("method").agg(
        edge_mae=("edge_mae", "mean"), ler_mae=("ler_mae", "mean"),
        lwr_mae=("lwr_mae", "mean"), n=("edge_mae", "count"),
    ).reset_index()
    placeholder = pd.DataFrame([{"method": ROW_ORDER[1], "edge_mae": np.nan,
                                  "ler_mae": np.nan, "lwr_mae": np.nan, "n": 0}])
    summary = pd.concat([summary, placeholder], ignore_index=True)
    summary["method"] = pd.Categorical(summary["method"], categories=ROW_ORDER, ordered=True)
    summary = summary.sort_values("method")
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(f"{args.out_prefix}_cost.csv", index=False)

    print("\n=== Table 2 (fidelity / baseline comparison, two-edge) -- this run ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n(scale: {args.seeds} seed(s), {args.epochs} epochs, "
          f"{args.train_per_config} train realizations/config, "
          f"{args.classical_per_config} classical realizations/config, "
          f"{len(configs)} configs -- see README.md to scale up)")
    print("\nNOTE: row 2 ('Learned denoiser + threshold') is intentionally left "
          "blank -- see module docstring. Do not fill it with a guessed number; "
          "either train that model and report the real result, or state in the "
          "manuscript that this comparison was left for future work.")


if __name__ == "__main__":
    main()
