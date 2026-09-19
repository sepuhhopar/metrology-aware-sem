"""
Core training/evaluation library for the two-edge metrology-aware pipeline
(Sections 4-7 of the new document). Importable by the experiment runner
scripts (run_experiment1.py, run_experiment2.py, ...).

Usage as a script runs a tiny smoke test.
"""
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from generator2 import generate_paired, master_child_rngs, detrend_np, rms_np, dose_and_read_noise_from_kn
from model2 import TinyUNet, soft_localize, heatmap_target, differentiable_detrend, rms
from losses2 import VARIANTS, edge_loss, pos_loss, rms_loss, psd_loss_LRW, consistency_loss

X0 = 64.0
W0 = 40.0
KNS = [0.3, 1.0, 2.0, 4.0]


def resolve_device(pref=None):
    """Training/eval device. Order of precedence: explicit `pref` argument,
    then the LER_DEVICE environment variable, then "auto" (CUDA if a GPU is
    visible, else CPU).

    Set LER_DEVICE=cpu to force the CPU path -- worth doing when you want
    numbers comparable to a previous CPU run, since CPU and GPU convolution
    kernels do not produce bit-identical results even at a fixed seed."""
    pref = pref or os.environ.get("LER_DEVICE", "auto")
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"LER_DEVICE={pref!r} requested but torch.cuda.is_available() is False. "
            "Check your driver/toolkit, or set LER_DEVICE=cpu.")
    return pref


DEVICE = resolve_device()

# Optional acquisition calibration measured from a real SEM by
# calibrate_generator.py. None (the default) reproduces the original optimistic
# imaging model exactly, so previously-reported numbers stay reproducible.
# When set, every dataset built here -- training AND evaluation -- is rendered
# with the measured PSF width and Poisson-Gaussian noise instead.
_CALIBRATION = None


def set_calibration(cal):
    """cal: dict with keys psf_sigma, dose, sigma_g -- or None to disable."""
    global _CALIBRATION
    _CALIBRATION = cal
    if cal is None:
        print("Acquisition model: generator defaults (uncalibrated).")
    else:
        print(f"Acquisition model: CALIBRATED psf_sigma={cal['psf_sigma']:.2f}, "
              f"dose={cal['dose']:.1f}, sigma_g={cal['sigma_g']:.4f}")


def load_calibration(path, psf_key="psf_sigma_ratio_matched"):
    """Read calibrate_generator.py's JSON. psf_key selects how the measured PSF
    is transferred to the synthetic grid: 'psf_sigma_ratio_matched' preserves
    the measured PSF-to-linewidth ratio (the right choice when the synthetic
    linewidth differs from the real track width), 'psf_sigma_absolute' takes
    the measured pixel value as-is."""
    with open(path) as f:
        c = json.load(f)
    g = c["generator_params"]
    return {"psf_sigma": float(g[psf_key]), "dose": float(g["dose"]),
            "sigma_g": float(g["sigma_g"])}


def build_paired_dataset(configs, n_per_config, salt):
    """configs: list of (sigma, xi, alpha, rho). Returns list of paired items."""
    n_items = len(configs) * n_per_config
    rngs1 = master_child_rngs(n_items, salt=salt * 2 + 1)
    rngs2 = master_child_rngs(n_items, salt=salt * 2 + 2)
    geom_rngs = master_child_rngs(n_items, salt=salt * 2 + 3)
    items = []
    kw = {}
    if _CALIBRATION is not None:
        acq = (_CALIBRATION["dose"], _CALIBRATION["sigma_g"])
        # same acquisition *settings* for both views; the noise realizations
        # still differ because rng1 and rng2 are independent, so the paired
        # consistency term remains meaningful
        kw = {"psf_sigma": _CALIBRATION["psf_sigma"], "acq1": acq, "acq2": acq}

    idx = 0
    for (sigma, xi, alpha, rho) in configs:
        for r in range(n_per_config):
            sample = generate_paired(rngs1[idx], rngs2[idx], sigma, xi, alpha, rho,
                                      w0=W0, x0=X0, shared_geometry_rng=geom_rngs[idx],
                                      **kw)
            sample["config"] = (sigma, xi, alpha, rho)
            items.append(sample)
            idx += 1
    return items


def _edge_forward(model, images_np, x_gt_L, x_gt_R, x0=X0, window_half=15, tau=0.15):
    # follow whatever device the model is already on, so callers never have to
    # keep a device argument in sync with train_one_variant/evaluate_model
    dev = next(model.parameters()).device
    imgs_t = torch.from_numpy(images_np.astype(np.float32)).unsqueeze(1).to(dev)  # (B,1,H,W)
    logits = model(imgs_t)  # (B,2,H,W)
    x_hat_L, probs_L, loL, hiL = soft_localize(logits[:, 0], x0 - W0 / 2, window_half, tau)
    x_hat_R, probs_R, loR, hiR = soft_localize(logits[:, 1], x0 + W0 / 2, window_half, tau)
    x_gt_L_t = torch.from_numpy(x_gt_L.astype(np.float32)).to(dev)
    x_gt_R_t = torch.from_numpy(x_gt_R.astype(np.float32)).to(dev)
    target_L = heatmap_target(x_gt_L_t, loL, hiL)
    target_R = heatmap_target(x_gt_R_t, loR, hiR)
    return x_hat_L, probs_L, target_L, x_gt_L_t, x_hat_R, probs_R, target_R, x_gt_R_t


def _metrology(x_hat_L, x_hat_R, x_gt_L_t, x_gt_R_t):
    dt_hat_L = differentiable_detrend(x_hat_L)
    dt_hat_R = differentiable_detrend(x_hat_R)
    dt_gt_L = differentiable_detrend(x_gt_L_t)
    dt_gt_R = differentiable_detrend(x_gt_R_t)
    w_hat = x_hat_R - x_hat_L
    w_gt = x_gt_R_t - x_gt_L_t
    dt_hat_W = differentiable_detrend(w_hat)
    dt_gt_W = differentiable_detrend(w_gt)
    sigma_hat_L, sigma_hat_R, sigma_hat_W = rms(dt_hat_L), rms(dt_hat_R), rms(dt_hat_W)
    sigma_gt_L, sigma_gt_R, sigma_gt_W = rms(dt_gt_L), rms(dt_gt_R), rms(dt_gt_W)
    return dict(dt_hat_L=dt_hat_L, dt_hat_R=dt_hat_R, dt_gt_L=dt_gt_L, dt_gt_R=dt_gt_R,
                dt_hat_W=dt_hat_W, dt_gt_W=dt_gt_W,
                sigma_hat_L=sigma_hat_L, sigma_hat_R=sigma_hat_R, sigma_hat_W=sigma_hat_W,
                sigma_gt_L=sigma_gt_L, sigma_gt_R=sigma_gt_R, sigma_gt_W=sigma_gt_W)


def train_one_variant(variant_name, train_items, epochs=10, batch_size=8, lr=1e-3,
                       seed=0, base_channels=12, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = VARIANTS[variant_name]
    model = TinyUNet(base=base_channels).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(train_items)
    t0 = time.time()
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = [train_items[i] for i in idx]
            img1 = np.stack([b["img1"] for b in batch])
            img2 = np.stack([b["img2"] for b in batch])
            xL = np.stack([b["x_L_gt"] for b in batch])
            xR = np.stack([b["x_R_gt"] for b in batch])

            total = 0.0
            per_view_metro = []
            for imgs in (img1, img2):
                xhL, probsL, tgtL, gtL, xhR, probsR, tgtR, gtR = _edge_forward(model, imgs, xL, xR)
                m = _metrology(xhL, xhR, gtL, gtR)
                l_edge = edge_loss(probsL, tgtL) + edge_loss(probsR, tgtR)
                loss = 1.0 * l_edge
                if cfg["pos"] > 0:
                    loss = loss + cfg["pos"] * (pos_loss(xhL, gtL) + pos_loss(xhR, gtR))
                if cfg["rms"] > 0:
                    loss = loss + cfg["rms"] * rms_loss(
                        m["sigma_hat_L"], m["sigma_hat_R"], m["sigma_hat_W"],
                        m["sigma_gt_L"], m["sigma_gt_R"], m["sigma_gt_W"])
                if cfg["psd"] > 0:
                    loss = loss + cfg["psd"] * psd_loss_LRW(
                        m["dt_hat_L"], m["dt_hat_R"], m["dt_hat_W"],
                        m["dt_gt_L"], m["dt_gt_R"], m["dt_gt_W"])
                total = total + loss
                per_view_metro.append((xhL, xhR, m))

            if cfg["cons"] > 0:
                (xhL1, xhR1, m1), (xhL2, xhR2, m2) = per_view_metro
                l_cons = consistency_loss(xhL1, xhR1, xhL2, xhR2,
                                           m1["sigma_hat_L"], m1["sigma_hat_R"], m1["sigma_hat_W"],
                                           m2["sigma_hat_L"], m2["sigma_hat_R"], m2["sigma_hat_W"],
                                           dt1=(m1["dt_hat_L"], m1["dt_hat_R"], m1["dt_hat_W"]),
                                           dt2=(m2["dt_hat_L"], m2["dt_hat_R"], m2["dt_hat_W"]))
                total = total + cfg["cons"] * l_cons

            opt.zero_grad()
            total.backward()
            opt.step()
            epoch_loss += total.item() * len(idx)
        if verbose:
            print(f"  [{variant_name} seed={seed}] epoch {epoch+1}/{epochs} "
                  f"loss={epoch_loss/n:.4f} ({time.time()-t0:.1f}s)")
    return model


@torch.no_grad()
def evaluate_model(model, configs, n_per_config, salt, label):
    items = build_paired_dataset(configs, n_per_config, salt=salt)
    rows = []
    for it in items:
        img1 = it["img1"][None, ...]
        xL = it["x_L_gt"][None, ...]
        xR = it["x_R_gt"][None, ...]
        xhL, probsL, tgtL, gtL, xhR, probsR, tgtR, gtR = _edge_forward(model, img1, xL, xR, tau=0.05)
        m = _metrology(xhL, xhR, gtL, gtR)

        edge_mae = 0.5 * (F.l1_loss(xhL, gtL).item() + F.l1_loss(xhR, gtR).item())
        ler_mae = 0.5 * (abs(m["sigma_hat_L"].item() - m["sigma_gt_L"].item()) +
                          abs(m["sigma_hat_R"].item() - m["sigma_gt_R"].item()))
        lwr_mae = abs(m["sigma_hat_W"].item() - m["sigma_gt_W"].item())

        rows.append({"method": label, "config": it["config"], "edge_mae": edge_mae,
                      "ler_mae": ler_mae, "lwr_mae": lwr_mae})
    return pd.DataFrame(rows)


def summarize(df):
    return df.groupby("method").agg(
        edge_mae=("edge_mae", "mean"), ler_mae=("ler_mae", "mean"),
        lwr_mae=("lwr_mae", "mean"), n=("edge_mae", "count"),
    ).reset_index()


if __name__ == "__main__":
    # smoke test
    configs = [(2.0, 12.0, 0.5, 0.3)]
    items = build_paired_dataset(configs, n_per_config=4, salt=99)
    print(f"Built {len(items)} paired items")
    m = train_one_variant("seg+pos+rms", items, epochs=2, batch_size=2, seed=0)
    df = evaluate_model(m, configs, n_per_config=3, salt=100, label="smoke-test")
    print(summarize(df))
