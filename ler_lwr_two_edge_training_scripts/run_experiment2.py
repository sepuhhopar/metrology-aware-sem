"""
Experiment 2 (Section 7.3 of the new document): loss ablation across the five
named training variants. Produces Table 3.

Run at PILOT scale by default (fast, CPU-feasible, meant to validate the
pipeline and give preliminary numbers). For the submission-quality protocol
described in the document (>=3 seeds, full parameter grid, more epochs),
increase --seeds/--epochs/--train-per-config and ideally run on a GPU:

    python3 run_experiment2.py --epochs 40 --seeds 3 --train-per-config 15 \
        --eval-per-config 15 --rhos 0.0 0.3 0.6 -0.3

Each additional seed multiplies runtime by ~5x (one training run per variant).
"""
import argparse
import numpy as np
import pandas as pd

from train2 import (build_paired_dataset, train_one_variant, evaluate_model,
                     set_calibration, load_calibration)
from losses2 import VARIANTS

SIGMAS = [1.5, 3.0]
XIS = [8.0, 20.0]
ALPHAS = [0.3, 0.7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--train-per-config", type=int, default=3)
    ap.add_argument("--eval-per-config", type=int, default=4)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.6])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--base-channels", type=int, default=12)
    ap.add_argument("--out-prefix", type=str, default="exp2")
    ap.add_argument("--calibration", type=str, default=None,
                     help="JSON from calibrate_generator.py: render under imaging "
                          "conditions measured from a real SEM instead of the "
                          "generator's optimistic defaults")
    ap.add_argument("--psf-transfer", type=str, default="psf_sigma_ratio_matched",
                     choices=["psf_sigma_ratio_matched", "psf_sigma_absolute"])
    args = ap.parse_args()

    if args.calibration:
        set_calibration(load_calibration(args.calibration, args.psf_transfer))

    configs = [(s, x, a, r) for s in SIGMAS for x in XIS for a in ALPHAS for r in args.rhos]
    print(f"Training configs: {len(configs)}  (sigma x xi x alpha x rho grid)")

    train_items = build_paired_dataset(configs, args.train_per_config, salt=200)
    print(f"Training set: {len(train_items)} paired items")

    all_eval_rows = []
    cost_rows = []
    for variant in VARIANTS:
        for seed in range(args.seeds):
            print(f"\n=== Training variant '{variant}' seed {seed} ===")
            import time
            t0 = time.time()
            model = train_one_variant(variant, train_items, epochs=args.epochs,
                                       batch_size=args.batch_size, seed=seed,
                                       base_channels=args.base_channels)
            train_time = time.time() - t0
            n_params = sum(p.numel() for p in model.parameters())
            cost_rows.append({"variant": variant, "seed": seed,
                               "train_seconds": train_time, "n_params": n_params})

            df = evaluate_model(model, configs, args.eval_per_config,
                                 salt=300 + seed, label=variant)
            df["seed"] = seed
            all_eval_rows.append(df)

    df_all = pd.concat(all_eval_rows, ignore_index=True)
    df_all.to_csv(f"{args.out_prefix}_raw.csv", index=False)

    summary = df_all.groupby("method").agg(
        edge_mae_mean=("edge_mae", "mean"), edge_mae_std=("edge_mae", "std"),
        ler_mae_mean=("ler_mae", "mean"), ler_mae_std=("ler_mae", "std"),
        lwr_mae_mean=("lwr_mae", "mean"), lwr_mae_std=("lwr_mae", "std"),
        n=("edge_mae", "count"),
    ).reset_index()
    variant_order = list(VARIANTS.keys())
    summary["method"] = pd.Categorical(summary["method"], categories=variant_order, ordered=True)
    summary = summary.sort_values("method")
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(f"{args.out_prefix}_cost.csv", index=False)

    print("\n=== Table 3 (loss ablation) -- this pilot run ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n(scale: {args.seeds} seed(s), {args.epochs} epochs, "
          f"{args.train_per_config} train realizations/config, "
          f"{len(configs)} configs -- see docstring to scale up)")


if __name__ == "__main__":
    main()
