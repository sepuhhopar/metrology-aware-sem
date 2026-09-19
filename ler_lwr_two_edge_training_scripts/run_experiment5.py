"""
Experiment 5 (Section 7.6): detector x localizer factorial comparison,
extended to two-edge data -> Table 4's classical rows (Threshold/Canny x
Pixel/Sub-pixel). The learned rows (Seg-only, Full) come from
run_experiment2.py's trained models -- this script only needs the classical
detectors, so it is fast (no training) and safe to run standalone.
"""
import argparse
import numpy as np
import pandas as pd

from generator2 import generate_paired, master_child_rngs, detrend_np, rms_np, dose_and_read_noise_from_kn
from classical_two_edge import threshold_detector, canny_detector

SIGMAS = [1.5, 3.0]
XIS = [8.0, 20.0]
ALPHAS = [0.3, 0.7]
KNS = [0.3, 1.0, 2.0, 4.0]
X0, W0 = 64.0, 40.0

DETECTORS = {"threshold": threshold_detector, "canny": canny_detector}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-config", type=int, default=5)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--out-prefix", type=str, default="exp5")
    ap.add_argument("--calibration", type=str, default=None,
                     help="JSON from calibrate_generator.py: render under imaging "
                          "conditions measured from a real SEM instead of the "
                          "generator's optimistic defaults")
    ap.add_argument("--psf-transfer", type=str, default="psf_sigma_ratio_matched",
                     choices=["psf_sigma_ratio_matched", "psf_sigma_absolute"])
    args = ap.parse_args()

    # imported lazily so this script keeps working standalone without torch
    cal = None
    extra_kw = {}
    per_config = args.per_config
    if args.calibration:
        from train2 import load_calibration
        cal = load_calibration(args.calibration, args.psf_transfer)
        extra_kw = {"psf_sigma": cal["psf_sigma"]}
        # the k_n sweep collapses to the one measured acquisition; scale the
        # per-config count so the trial count per row stays comparable
        per_config = per_config * len(KNS)
        print(f"CALIBRATED: psf_sigma={cal['psf_sigma']:.2f}, dose={cal['dose']:.1f}, "
              f"sigma_g={cal['sigma_g']:.4f}")

    # (k_n, dose, sigma_g); under calibration k_n is the equivalent noise level
    # of the measured acquisition, kept so the raw CSV column stays meaningful
    acq_grid = ([(400.0 / cal["dose"],) + (cal["dose"], cal["sigma_g"])] if cal is not None
                else [(k,) + dose_and_read_noise_from_kn(k) for k in KNS])

    configs = [(s, x, a, r) for s in SIGMAS for x in XIS for a in ALPHAS for r in args.rhos]
    n_items = len(configs) * len(acq_grid) * per_config
    print(f"Generating {n_items} single-view samples "
          f"({len(configs)} configs x {len(acq_grid)} noise level(s) x {per_config})...")

    rngs = master_child_rngs(n_items, salt=777)
    rows = []
    idx = 0
    for (sigma, xi, alpha, rho) in configs:
        for (k_n, dose, sigma_g) in acq_grid:
            for r in range(per_config):
                rng = rngs[idx]; idx += 1
                sample = generate_paired(rng, rng, sigma, xi, alpha, rho, w0=W0, x0=X0,
                                          acq1=(dose, sigma_g), acq2=(dose, sigma_g),
                                          **extra_kw)
                img = sample["img1"]
                xL_gt, xR_gt = sample["x_L_gt"], sample["x_R_gt"]
                sigma_gt_L = rms_np(detrend_np(xL_gt))
                sigma_gt_R = rms_np(detrend_np(xR_gt))
                w_gt = xR_gt - xL_gt
                sigma_gt_W = rms_np(detrend_np(w_gt))

                for det_name, fn in DETECTORS.items():
                    for subpixel in (False, True):
                        xL_est = fn(img, X0 - W0 / 2, subpixel=subpixel, falling=False)
                        xR_est = fn(img, X0 + W0 / 2, subpixel=subpixel, falling=True)
                        edge_mae = 0.5 * (np.mean(np.abs(xL_est - xL_gt)) + np.mean(np.abs(xR_est - xR_gt)))
                        sigma_L = rms_np(detrend_np(xL_est))
                        sigma_R = rms_np(detrend_np(xR_est))
                        w_est = xR_est - xL_est
                        sigma_W = rms_np(detrend_np(w_est))
                        ler_mae = 0.5 * (abs(sigma_L - sigma_gt_L) + abs(sigma_R - sigma_gt_R))
                        lwr_mae = abs(sigma_W - sigma_gt_W)
                        rows.append({
                            "detector": det_name,
                            "localization": "sub-pixel" if subpixel else "pixel",
                            "k_n": k_n, "edge_mae": edge_mae, "ler_mae": ler_mae, "lwr_mae": lwr_mae,
                        })
        if idx % 200 == 0:
            print(f"  ...{idx}/{n_items}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_prefix}_raw.csv", index=False)
    summary = df.groupby(["detector", "localization"]).agg(
        edge_mae=("edge_mae", "mean"), ler_mae=("ler_mae", "mean"),
        lwr_mae=("lwr_mae", "mean"), n=("edge_mae", "count"),
    ).reset_index()
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    print("\n=== Table 4 classical rows (detector x localization, two-edge) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nNote: the 'Learned, Seg-only' / 'Learned, Full' rows of Table 4 come "
          "from run_experiment2.py's trained models (evaluate them with soft-argmax "
          "for 'Full'/sub-pixel and a hard-argmax pass for the 'Seg-only'/pixel row).")


if __name__ == "__main__":
    main()
