"""Usage: python3 run_threshold_matched.py default calibrated

Threshold + sub-pixel baseline evaluated on EXACTLY the evaluation samples the learned
Table 2 models saw: build_paired_dataset(configs, 15, salt=500+seed), image = img1, seeds 0..2."""
import numpy as np, pandas as pd, sys
from threshold_common import *
import os
cfgs = configs([0.0, 0.3, 0.6, -0.3])
res = {}
for mode, cal in [(m, c) for m, c in (('default', None), ('calibrated', load_cal())) if m in sys.argv[1:]]:
    # 1) reproduce the original classical row (sanity check of the whole setup)
    orig = original_classical(cfgs, 15, 555, cal)
    rec = pd.read_csv(os.path.join(RUNS, f"fix_exp1{'cal' if cal else ''}_raw.csv"))
    rec = rec[rec.method.str.startswith('Classical')][['edge_mae','ler_mae','lwr_mae']].to_numpy()
    exact = np.allclose(orig, rec, atol=1e-9)
    # 2) matched evaluation
    rows = []
    for seed in range(3):
        for it in build_paired_dataset(cfgs, 15, salt=500 + seed, cal=cal):
            e, l, w = threshold_metrics(it['img1'], it['x_L_gt'], it['x_R_gt'])
            rows.append(dict(seed=seed, config=str(it['config']), edge_mae=e, ler_mae=l, lwr_mae=w))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RUNS, f'threshold_matched_{mode}_raw.csv'), index=False)
    res[mode] = dict(orig_n=len(orig), orig_reproduced=exact,
                     orig_mean=orig.mean(0).round(4).tolist(), rec_mean=rec.mean(0).round(4).tolist(),
                     matched_n=len(df), matched_mean=df[['edge_mae','ler_mae','lwr_mae']].mean().round(4).tolist())
    print(mode, res[mode], flush=True)
