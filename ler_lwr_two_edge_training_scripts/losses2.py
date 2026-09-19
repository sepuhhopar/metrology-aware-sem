"""Loss terms implementing Eqs. 22-29 of the new document, and the five named
training variants of Section 6 as explicit lambda configurations."""
import torch
import torch.nn.functional as F
from model2 import differentiable_detrend, rms, differentiable_psd, log_psd_loss

# Section 6's five variants, as (lambda_p, lambda_r, lambda_s, lambda_c).
# lambda_e (edge/segmentation loss weight) is always 1.0.
VARIANTS = {
    "seg-only":            dict(pos=0.0,  rms=0.0, psd=0.0, cons=0.0),
    "seg+pos":             dict(pos=0.05, rms=0.0, psd=0.0, cons=0.0),
    "seg+pos+rms":         dict(pos=0.05, rms=1.0, psd=0.0, cons=0.0),
    "seg+pos+rms+psd":     dict(pos=0.05, rms=1.0, psd=0.5, cons=0.0),
    "full":                dict(pos=0.05, rms=1.0, psd=0.5, cons=0.5),
}
LAMBDA_W = 1.0   # weight on the width (LWR) term within L_RMS / L_cons
ETA_R = 0.5      # Eq. 28 weight on per-edge RMS consistency (gamma_r)
ETA_W = 0.5      # Eq. 28 weight on width RMS consistency
GAMMA_S = 0.5    # Eq. 29 weight on SPECTRAL consistency between paired views

# The spectral consistency term was specified in the method but absent from the
# original implementation, so every result produced before this flag existed
# used coordinate + RMS consistency only. Set to False to reproduce those
# earlier numbers exactly.
CONS_INCLUDE_PSD = True


def edge_loss(probs, target):
    """L_BCE analogue: cross-entropy between the predicted windowed softmax
    distribution and a soft Gaussian target (heatmap regression). Boundary-
    aware term (Eq. 22's L_boundary, e.g. Kervadec et al.) is NOT implemented
    in this reference script -- lambda_b effectively 0 here; hook left for a
    real boundary-distance-transform term if the authors want to add it."""
    return -(target * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()


def pos_loss(x_hat, x_gt):
    """Eq. 23: smooth-L1 (Huber) sub-pixel coordinate loss."""
    return F.smooth_l1_loss(x_hat, x_gt)


def rms_loss(sigma_hat_L, sigma_hat_R, sigma_hat_W, sigma_gt_L, sigma_gt_R, sigma_gt_W, eps=1e-6):
    """Eq. 24."""
    l = (sigma_hat_L - sigma_gt_L).abs() / (sigma_gt_L + eps)
    r = (sigma_hat_R - sigma_gt_R).abs() / (sigma_gt_R + eps)
    w = (sigma_hat_W - sigma_gt_W).abs() / (sigma_gt_W + eps)
    return (l + r).mean() + LAMBDA_W * w.mean()


def psd_loss_LRW(x_hat_L_dt, x_hat_R_dt, w_hat_dt, x_gt_L_dt, x_gt_R_dt, w_gt_dt):
    """Eq. 25, summed over s in {L,R,W}."""
    total = 0.0
    for pred_dt, gt_dt in [(x_hat_L_dt, x_gt_L_dt), (x_hat_R_dt, x_gt_R_dt), (w_hat_dt, w_gt_dt)]:
        f_p, psd_p = differentiable_psd(pred_dt)
        f_g, psd_g = differentiable_psd(gt_dt)
        total = total + log_psd_loss(f_p, psd_p, psd_g)
    return total / 3.0


def psd_consistency(dt1_L, dt1_R, dt1_W, dt2_L, dt2_R, dt2_W):
    """Eq. 29: spectral agreement between two independently-degraded views of
    the SAME geometry. Identical in form to L_PSD, except both arguments are
    predictions -- there is no ground truth involved, so this term is available
    on unlabelled repeated acquisitions as well."""
    total = 0.0
    for a, b in [(dt1_L, dt2_L), (dt1_R, dt2_R), (dt1_W, dt2_W)]:
        f_a, psd_a = differentiable_psd(a)
        _, psd_b = differentiable_psd(b)
        total = total + log_psd_loss(f_a, psd_a, psd_b)
    return total / 3.0


def consistency_loss(x_hat_L1, x_hat_R1, x_hat_L2, x_hat_R2,
                      sigma_L1, sigma_R1, sigma_W1, sigma_L2, sigma_R2, sigma_W2,
                      dt1=None, dt2=None):
    """Eqs. 27-29.

    dt1/dt2: optional (detrended_L, detrended_R, detrended_W) triples for the
    two paired views. When supplied and CONS_INCLUDE_PSD is True, the spectral
    consistency term of Eq. 29 is included; when omitted the term is dropped,
    which reproduces the coordinate+RMS-only behaviour of earlier runs."""
    pos_term = 0.5 * (F.l1_loss(x_hat_L1, x_hat_L2) + F.l1_loss(x_hat_R1, x_hat_R2))
    rms_term = ETA_R * (0.5 * ((sigma_L1 - sigma_L2).abs().mean() + (sigma_R1 - sigma_R2).abs().mean()))
    width_term = ETA_W * (sigma_W1 - sigma_W2).abs().mean()
    total = pos_term + rms_term + width_term
    if CONS_INCLUDE_PSD and dt1 is not None and dt2 is not None:
        total = total + GAMMA_S * psd_consistency(*dt1, *dt2)
    return total
