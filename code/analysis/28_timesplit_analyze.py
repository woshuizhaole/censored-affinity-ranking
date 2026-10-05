#!/usr/bin/env python
"""improvement.md #5 analysis: arms on the ChEMBL TIME-SPLIT test (recent-year compounds). Does the
tuned-novelty AUC/BEDROC edge survive a realistic temporal distribution shift? Reports per-arm means +
paired Wilcoxon (tuned-F, tuned-G vs C/B) + effect size + bootstrap CI. Honest about power (test targets)."""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

PROJ = "/data/workplace/wyx/censored_affinity_20260707"
RNG = np.random.default_rng(20260708)


def pt(sub, arm):
    fs = glob.glob("%s/results/timesplit/%s/timesplit/pertarget_%s_seed*.csv" % (PROJ, sub, arm))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d.groupby("target")[["EF1%", "BEDROC20", "AUC"]].mean()


def bootci(d, n=5000):
    idx = RNG.integers(0, len(d), size=(n, len(d))); mu = d[idx].mean(1)
    return np.percentile(mu, 2.5), np.percentile(mu, 97.5)


arms = {"A_drop": ("base", "A_drop"), "B_cast": ("base", "B_cast"), "C_bounded": ("base", "C_bounded"),
        "D_tobit": ("base", "D_tobit"), "F_default": ("base", "F_tobit_rankdec"),
        "F_tuned": ("tuned", "F_tobit_rankdec"), "G_tuned": ("tuned", "G_bounded_rankdec")}
M = {k: pt(sub, a) for k, (sub, a) in arms.items()}
print("=== TIME-SPLIT test per-arm means (ChEMBL, train<=2019 / test>2021) ===")
print("%-12s %7s %9s %7s %7s" % ("arm", "EF1%", "BEDROC20", "AUC", "ntgt"))
for k, d in M.items():
    if d is not None:
        print("%-12s %7.3f %9.4f %7.4f %7d" % (k, d["EF1%"].mean(), d["BEDROC20"].mean(), d["AUC"].mean(), len(d)))
    else:
        print("%-12s  MISSING" % k)

print("\n=== paired Wilcoxon + effect + 95%CI (tuned vs baselines, time-split test) ===")
print("%-16s %-9s %7s %7s %8s %18s %9s" % ("contrast", "metric", "hi", "lo", "delta", "95%CI", "p1"))
rows = []
for hi_name, lo_name in [("F_tuned", "C_bounded"), ("F_tuned", "B_cast"), ("G_tuned", "C_bounded"),
                         ("G_tuned", "B_cast"), ("F_tuned", "F_default"), ("C_bounded", "A_drop")]:
    hi, lo = M.get(hi_name), M.get(lo_name)
    if hi is None or lo is None:
        continue
    com = sorted(set(hi.index) & set(lo.index))
    for m in ["BEDROC20", "AUC", "EF1%"]:
        x = hi.loc[com, m].values; y = lo.loc[com, m].values; d = x - y
        lo95, hi95 = bootci(d)
        try:
            _, p = wilcoxon(x, y, alternative="greater")
        except Exception:
            p = np.nan
        print("%-16s %-9s %7.3f %7.3f %8.4f  [%7.4f,%7.4f] %9.4g %s"
              % ("%s>%s" % (hi_name, lo_name), m, x.mean(), y.mean(), d.mean(), lo95, hi95, p, "*" if p < 0.05 else ""))
        rows.append({"contrast": "%s>%s" % (hi_name, lo_name), "metric": m, "hi": x.mean(), "lo": y.mean(),
                     "delta": d.mean(), "ci_lo": lo95, "ci_hi": hi95, "p1": p, "n": len(com)})
pd.DataFrame(rows).to_csv(PROJ + "/results/tables/timesplit_paired_effect.csv", index=False)
print("\nwrote -> results/tables/timesplit_paired_effect.csv")
