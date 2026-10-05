#!/usr/bin/env python
"""improvement.md #3: effect sizes up front. For every pivotal comparison report the point effect
(mean per-target delta) + 95% bootstrap CI (paired, target-level resample) alongside the p-value and
a Holm-corrected p within the tuned-F family — so a tiny effect amplified by large n is exposed."""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

PROJ = "/data/workplace/wyx/censored_affinity_20260707"
RNG = np.random.default_rng(20260708)


def pertarget(root, scheme, arm, mets=("EF1%", "BEDROC20", "AUC")):
    fs = glob.glob(os.path.join(PROJ, root, scheme, "pertarget_%s_seed*.csv" % arm))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d.groupby("target")[[m for m in mets if m in d.columns]].mean()


def boot_ci(diff, n=5000):
    idx = RNG.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min((m - rank) * pvals[i], 1.0)); adj[i] = run
    return adj


rows = []
# --- tuned-F (lambda=3) vs B/C, both splits (family for Holm) ---
for scheme in ["random", "scaffold"]:
    Ftu = pertarget("results/p1/f_l3m05", scheme, "F_tobit_rankdec")
    for lo_name, lo_root, lo_arm in [("B_cast", "results/mainline", "B_cast"),
                                     ("C_bounded", "results/mainline", "C_bounded")]:
        lo = pertarget(lo_root, scheme, lo_arm)
        com = sorted(set(Ftu.index) & set(lo.index))
        for m in ["AUC", "BEDROC20", "EF1%"]:
            d = (Ftu.loc[com, m] - lo.loc[com, m]).values
            lo95, hi95 = boot_ci(d)
            try:
                _, p = wilcoxon(Ftu.loc[com, m].values, lo.loc[com, m].values, alternative="greater")
            except Exception:
                p = np.nan
            rows.append({"family": "tunedF_internal", "cmp": "Ftuned-%s" % lo_name, "split": scheme,
                         "metric": m, "n": len(com), "delta": d.mean(), "ci_lo": lo95, "ci_hi": hi95, "p_1sided": p})
# --- DUD-E 5-seed F vs C (external) ---
def dude(arm):
    return pd.read_csv(PROJ + "/results/stage9_dude_5seed/dude5_%s_pertarget.csv" % arm).set_index("target")
Fd, Cd, Bd = dude("F_tobit_rankdec"), dude("C_bounded"), dude("B_cast")
com = sorted(set(Fd.index) & set(Cd.index) & set(Bd.index))
for lo_name, lo in [("C_bounded", Cd), ("B_cast", Bd)]:
    for m in ["AUC", "BEDROC20", "RIE20", "EF1%"]:
        if m not in Fd.columns:
            continue
        d = (Fd.loc[com, m] - lo.loc[com, m]).values
        lo95, hi95 = boot_ci(d)
        try:
            _, p = wilcoxon(Fd.loc[com, m].values, lo.loc[com, m].values, alternative="greater")
        except Exception:
            p = np.nan
        rows.append({"family": "dude_external", "cmp": "Fdude-%s" % lo_name, "split": "DUDE",
                     "metric": m, "n": len(com), "delta": d.mean(), "ci_lo": lo95, "ci_hi": hi95, "p_1sided": p})

df = pd.DataFrame(rows)
# Holm within each family
df["p_holm"] = np.nan
for fam, g in df.groupby("family"):
    df.loc[g.index, "p_holm"] = holm(g["p_1sided"].values)
df.to_csv(PROJ + "/results/tables/effect_sizes_pivotal.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
print(df[["family", "cmp", "split", "metric", "n", "delta", "ci_lo", "ci_hi", "p_1sided", "p_holm"]].round(5).to_string(index=False))
print("\n解读: delta=每靶差均值(效应量); [ci_lo,ci_hi]=95%%bootstrap CI(不含0=稳健); p_holm=族内Holm校正")
print("wrote -> results/tables/effect_sizes_pivotal.csv")
