#!/usr/bin/env python
"""Pivotal verification: does the val-selected tuned F (lambda=3, margin=0.5) BEAT B_cast and
C_bounded per-target (paired Wilcoxon, one-sided)? This is the positive-reframe linchpin."""
import glob, os
import pandas as pd, numpy as np
from scipy.stats import wilcoxon

PROJ = "/data/workplace/wyx/censored_affinity_20260707"


def pertarget(root, scheme, arm):
    fs = glob.glob(os.path.join(PROJ, root, scheme, "pertarget_%s_seed*.csv" % arm))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d.groupby("target")[["EF1%", "BEDROC20", "AUC"]].mean()


for scheme in ["random", "scaffold"]:
    Ftu = pertarget("results/p1/f_l3m05", scheme, "F_tobit_rankdec")
    refs = [("B_cast", pertarget("results/mainline", scheme, "B_cast")),
            ("C_bounded", pertarget("results/mainline", scheme, "C_bounded")),
            ("F_default", pertarget("results/mainline", scheme, "F_tobit_rankdec"))]
    print("=== %s : tuned-F(lam3) paired Wilcoxon (per-target, seed-averaged) ===" % scheme)
    for lo_name, lo in refs:
        com = sorted(set(Ftu.index) & set(lo.index))
        for m in ["BEDROC20", "AUC", "EF1%"]:
            x = Ftu.loc[com, m].values; y = lo.loc[com, m].values
            try:
                _, p = wilcoxon(x, y, alternative="greater")
            except Exception:
                p = float("nan")
            sig = "*" if p < 0.05 else ""
            print("  Ftuned>%-9s %-9s: %.3f vs %.3f (n=%d) wilcox_p=%.4g %s"
                  % (lo_name, m, x.mean(), y.mean(), len(com), p, sig))
