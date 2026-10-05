#!/usr/bin/env python
"""D5: DUD-E external — per-target PAIRED Wilcoxon (one-sided) for the key arm contrasts.
Upgrades the Stage-9a 'reversal within noise' statement to a proper paired test. The saved
summary only reported means + bootstrap CI (over-conservative for a paired design)."""
import os, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

R = "/data/workplace/wyx/censored_affinity_20260707/results/stage9_dude"
ARMS = ["A_drop", "B_cast", "C_bounded", "D_tobit", "F_tobit_rankdec"]
tabs = {}
for a in ARMS:
    d = pd.read_csv(os.path.join(R, "dude_%s_pertarget.csv" % a)).set_index("target")
    tabs[a] = d
common = sorted(set.intersection(*[set(t.index) for t in tabs.values()]))
print("n common targets = %d" % len(common))
metrics = ["EF1%", "BEDROC20", "RIE20", "AUC"]
print("\n%-18s %-10s %8s %8s %10s %10s" % ("contrast", "metric", "hi_mean", "lo_mean", "med_diff", "wilcox_p(1s hi>lo)"))
for hi, lo in [("F_tobit_rankdec", "C_bounded"), ("F_tobit_rankdec", "B_cast"),
               ("D_tobit", "B_cast"), ("F_tobit_rankdec", "A_drop"),
               ("F_tobit_rankdec", "D_tobit")]:
    for m in metrics:
        if m not in tabs[hi].columns:
            continue
        x = tabs[hi].loc[common, m].values.astype(float)
        y = tabs[lo].loc[common, m].values.astype(float)
        d = x - y
        try:
            _, p = wilcoxon(x, y, alternative="greater", zero_method="wilcox")
        except Exception:
            p = float("nan")
        sig = "*" if p < 0.05 else ""
        print("%-18s %-10s %8.3f %8.3f %10.4f %10.4g %s"
              % (hi.split("_")[0] + "-" + lo.split("_")[0], m, x.mean(), y.mean(), np.median(d), p, sig))
