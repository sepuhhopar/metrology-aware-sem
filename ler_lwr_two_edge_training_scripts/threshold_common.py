"""Shared helpers for run_threshold_matched.py (no torch needed)."""
import sys, json, itertools
import numpy as np, pandas as pd
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
sys.path.insert(0, HERE)
from generator2 import master_child_rngs, generate_paired, detrend_np, rms_np, dose_and_read_noise_from_kn
from classical_two_edge import threshold_detector
X0, W0 = 64.0, 40.0
KNS = [0.3, 1.0, 2.0, 4.0]
SIGMAS, XIS, ALPHAS = [1.5, 3.0], [8.0, 20.0], [0.3, 0.7]

def configs(rhos):
    return [(s, x, a, r) for s in SIGMAS for x in XIS for a in ALPHAS for r in rhos]

def load_cal():
    c = json.load(open(os.path.join(RUNS, 'calibration_m2.json')))['generator_params']
    return {'psf_sigma': c['psf_sigma_ratio_matched'], 'dose': c['dose'], 'sigma_g': c['sigma_g']}

def build_paired_dataset(cfgs, n_per_config, salt, cal=None):
    """verbatim logic of train2.build_paired_dataset (no torch needed)"""
    n_items = len(cfgs) * n_per_config
    rngs1 = master_child_rngs(n_items, salt=salt * 2 + 1)
    rngs2 = master_child_rngs(n_items, salt=salt * 2 + 2)
    geom_rngs = master_child_rngs(n_items, salt=salt * 2 + 3)
    kw = {}
    if cal is not None:
        acq = (cal['dose'], cal['sigma_g'])
        kw = {'psf_sigma': cal['psf_sigma'], 'acq1': acq, 'acq2': acq}
    items, idx = [], 0
    for (sigma, xi, alpha, rho) in cfgs:
        for _ in range(n_per_config):
            s = generate_paired(rngs1[idx], rngs2[idx], sigma, xi, alpha, rho, w0=W0, x0=X0,
                                shared_geometry_rng=geom_rngs[idx], **kw)
            s['config'] = (sigma, xi, alpha, rho); items.append(s); idx += 1
    return items

def threshold_metrics(img, xL_gt, xR_gt):
    """verbatim metric logic of run_experiment1.run_classical_threshold"""
    sL, sR, sW = rms_np(detrend_np(xL_gt)), rms_np(detrend_np(xR_gt)), rms_np(detrend_np(xR_gt - xL_gt))
    xL = threshold_detector(img, X0 - W0 / 2, subpixel=True, falling=False)
    xR = threshold_detector(img, X0 + W0 / 2, subpixel=True, falling=True)
    edge = 0.5 * (np.mean(np.abs(xL - xL_gt)) + np.mean(np.abs(xR - xR_gt)))
    ler = 0.5 * (abs(rms_np(detrend_np(xL)) - sL) + abs(rms_np(detrend_np(xR)) - sR))
    lwr = abs(rms_np(detrend_np(xR - xL)) - sW)
    return edge, ler, lwr

def original_classical(cfgs, per_config=15, salt=555, cal=None):
    """verbatim sample generation of run_experiment1.run_classical_threshold"""
    acq_grid = [(cal['dose'], cal['sigma_g'])] if cal else [dose_and_read_noise_from_kn(k) for k in KNS]
    kw = {'psf_sigma': cal['psf_sigma']} if cal else {}
    if cal: per_config *= len(KNS)
    n = len(cfgs) * len(acq_grid) * per_config
    rngs = master_child_rngs(n, salt=salt); out = []; idx = 0
    for (sigma, xi, alpha, rho) in cfgs:
        for (dose, sg) in acq_grid:
            for _ in range(per_config):
                rng = rngs[idx]; idx += 1
                s = generate_paired(rng, rng, sigma, xi, alpha, rho, w0=W0, x0=X0,
                                    acq1=(dose, sg), acq2=(dose, sg), **kw)
                out.append(threshold_metrics(s['img1'], s['x_L_gt'], s['x_R_gt']))
    return np.array(out)
