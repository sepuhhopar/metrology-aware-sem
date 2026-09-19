"""
Experiment 8 (Section 7.9 of the new document): computational cost report.

Reports, for each of the five loss-ablation variants (Section 6) plus the
classical baseline:
  - parameter count (learned models only)
  - wall-clock training time (learned models only; CPU seconds in this
    environment -- report your own machine's numbers when you re-run this)
  - wall-clock inference time per image (all methods, classical included)

This does not re-run the full Table 3 protocol -- it trains each variant for
a short, fixed number of epochs purely to produce a fair per-epoch/per-image
timing comparison, and reports that measured cost alongside a naive
extrapolation to the full submission-quality budget. For the *accuracy*
numbers, use run_experiment2.py's Table 3 output instead; this script's
accuracy column (if you pass --with-accuracy) is a byproduct of the same
short run and should be read as a sanity check only, not the reported result.

Usage:
    python3 run_experiment8.py
    python3 run_experiment8.py --timing-epochs 5 --n-timing-items 32
"""
import argparse
import time
import numpy as np
import pandas as pd
import torch

from generator2 import master_child_rngs, generate_paired, detrend_np, rms_np, dose_and_read_noise_from_kn
from classical_two_edge import threshold_detector, canny_detector
from train2 import build_paired_dataset, train_one_variant, evaluate_model, _edge_forward
from losses2 import VARIANTS

SIGMA, XI, ALPHA, RHO = 2.0, 12.0, 0.5, 0.3
X0, W0 = 64.0, 40.0


def time_classical(detector_fn, n_images=50, salt=9001):
    rngs = master_child_rngs(n_images, salt=salt)
    dose, sigma_g = dose_and_read_noise_from_kn(1.0)
    imgs = []
    for rng in rngs:
        sample = generate_paired(rng, rng, SIGMA, XI, ALPHA, RHO, w0=W0, x0=X0,
                                  acq1=(dose, sigma_g), acq2=(dose, sigma_g))
        imgs.append(sample["img1"])
    t0 = time.time()
    for img in imgs:
        detector_fn(img, X0 - W0 / 2, subpixel=True, falling=False)
        detector_fn(img, X0 + W0 / 2, subpixel=True, falling=True)
    dt = time.time() - t0
    return dt / n_images


@torch.no_grad()
def time_inference(model, n_images=50, salt=9002):
    rngs = master_child_rngs(n_images, salt=salt)
    dose, sigma_g = dose_and_read_noise_from_kn(1.0)
    imgs, xLs, xRs = [], [], []
    for rng in rngs:
        sample = generate_paired(rng, rng, SIGMA, XI, ALPHA, RHO, w0=W0, x0=X0,
                                  acq1=(dose, sigma_g), acq2=(dose, sigma_g))
        imgs.append(sample["img1"]); xLs.append(sample["x_L_gt"]); xRs.append(sample["x_R_gt"])
    on_cuda = next(model.parameters()).is_cuda
    if on_cuda:
        # warm up (first CUDA call pays one-off context/kernel-selection cost)
        # and make sure the queue is empty before the clock starts
        _edge_forward(model, imgs[0][None, ...], xLs[0][None, ...], xRs[0][None, ...], tau=0.05)
        torch.cuda.synchronize()
    t0 = time.time()
    for img, xL, xR in zip(imgs, xLs, xRs):
        _edge_forward(model, img[None, ...], xL[None, ...], xR[None, ...], tau=0.05)
    if on_cuda:
        # CUDA kernels are launched asynchronously -- without this the loop
        # above measures launch overhead, not execution
        torch.cuda.synchronize()
    dt = time.time() - t0
    return dt / n_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing-epochs", type=int, default=5,
                     help="epochs used ONLY to produce a fair per-epoch timing measurement")
    ap.add_argument("--n-timing-items", type=int, default=32,
                     help="training items used for the timing run")
    ap.add_argument("--n-timing-images", type=int, default=50,
                     help="images used to measure per-image inference/detector time")
    ap.add_argument("--base-channels", type=int, default=12)
    ap.add_argument("--full-epochs", type=int, default=40,
                     help="epoch count to extrapolate training cost to, for the "
                          "submission-quality budget suggested in README.md")
    ap.add_argument("--out-prefix", type=str, default="exp8")
    args = ap.parse_args()

    rows = []

    print("=== Classical detectors: per-image inference time ===")
    for name, fn in [("threshold", threshold_detector), ("canny", canny_detector)]:
        t_per_img = time_classical(fn, n_images=args.n_timing_images)
        print(f"  {name:12s}: {t_per_img*1000:.3f} ms/image")
        rows.append({"method": f"Classical: {name}", "n_params": np.nan,
                      "train_seconds_measured": np.nan, "train_epochs_measured": np.nan,
                      "train_seconds_per_epoch": np.nan,
                      "train_seconds_extrapolated_full": np.nan,
                      "inference_ms_per_image": t_per_img * 1000})

    configs = [(SIGMA, XI, ALPHA, RHO)]
    train_items = build_paired_dataset(configs, args.n_timing_items, salt=8000)

    print(f"\n=== Learned (U-Net, base={args.base_channels}): training + inference cost ===")
    for variant in VARIANTS:
        t0 = time.time()
        model = train_one_variant(variant, train_items, epochs=args.timing_epochs,
                                   batch_size=8, seed=0, base_channels=args.base_channels,
                                   verbose=False)
        train_time = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())
        per_epoch = train_time / args.timing_epochs
        extrapolated = per_epoch * args.full_epochs
        t_per_img = time_inference(model, n_images=args.n_timing_images)
        print(f"  {variant:20s}: {n_params:7d} params | "
              f"{per_epoch:6.2f} s/epoch | {t_per_img*1000:.3f} ms/image inference | "
              f"~{extrapolated:7.1f} s extrapolated to {args.full_epochs} epochs")
        rows.append({"method": f"Learned (U-Net): {variant}", "n_params": n_params,
                      "train_seconds_measured": train_time,
                      "train_epochs_measured": args.timing_epochs,
                      "train_seconds_per_epoch": per_epoch,
                      "train_seconds_extrapolated_full": extrapolated,
                      "inference_ms_per_image": t_per_img * 1000})

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_prefix}_cost.csv", index=False)

    print("\n=== Table (computational cost, Section 7.9) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nNote: all timings are CPU wall-clock in whatever environment you run this "
          f"script in ({torch.get_num_threads()} torch threads here) -- re-run on your "
          f"submission machine and report those numbers, not this sandbox's. The "
          f"'extrapolated_full' column is per_epoch_time * --full-epochs (default "
          f"{args.full_epochs}), a linear extrapolation for planning purposes only; "
          f"actual full-scale runs should still be timed directly.")


if __name__ == "__main__":
    main()
