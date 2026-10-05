#!/usr/bin/env python
"""Stage 10: aggregate ablation into the main results table + paired stats.

For each split scheme, per-target metrics are averaged over seeds, then:
  - main table: per arm, cross-target mean + bootstrap CI + median of EF1%/BEDROC/AUC
  - paired tests (per-target, arms aligned by target): Wilcoxon signed-rank on the
    per-target metric difference for the pre-registered comparisons, with median
    diff, relative improvement, and Holm correction across the comparison family.

Pre-registered comparisons (H1/H2/baselines):
  D-B, D-A, C-B (H1: censored likelihood/bounded beats discard/cast)
  F-D, F-E     (H2: censored-aware ranking adds; F vs E isolates censored pairs)

Usage: python 30_aggregate.py --results-root results/mainline --name bindingdb_IC50 \
   --out results/tables --schemes target_disjoint scaffold random
"""
from __future__ import annotations
import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ARMS = ["A_drop", "B_cast", "C_bounded", "D_tobit", "E_tobit_rankobs", "F_tobit_rankdec"]
SHORT = {"A_drop": "A", "B_cast": "B", "C_bounded": "C", "D_tobit": "D",
         "E_tobit_rankobs": "E", "F_tobit_rankdec": "F"}
METRICS = ["EF1%", "EF0.5%", "EF5%", "BEDROC20", "BEDROC80.5", "AUC"]
COMPARISONS = [("D_tobit", "B_cast"), ("D_tobit", "A_drop"), ("C_bounded", "B_cast"),
               ("F_tobit_rankdec", "D_tobit"), ("F_tobit_rankdec", "E_tobit_rankobs"),
               ("F_tobit_rankdec", "B_cast")]


def per_target_mean_over_seeds(pertarget_dir, arm):
    files = glob.glob(os.path.join(pertarget_dir, f"pertarget_{arm}_seed*.csv"))
    if not files:
        return None
    alld = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cols = [m for m in METRICS if m in alld.columns]
    return alld.groupby("target")[cols].mean()


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        run = max(run, val)
        adj[idx] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--schemes", nargs="+", default=["target_disjoint", "scaffold", "random"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    main_rows = []
    test_rows = []
    for scheme in args.schemes:
        d = os.path.join(args.results_root, scheme)
        if not os.path.isdir(d):
            print(f"[agg] skip missing scheme {scheme}")
            continue
        tabs = {a: per_target_mean_over_seeds(d, a) for a in ARMS}
        tabs = {a: t for a, t in tabs.items() if t is not None}
        if not tabs:
            continue
        # main table
        for a, t in tabs.items():
            row = {"scheme": scheme, "arm": SHORT.get(a, a), "n_targets": len(t)}
            for m in METRICS:
                if m in t.columns:
                    mean, lo, hi = bootstrap_ci(t[m].values)
                    row[f"{m}_mean"] = round(mean, 4)
                    row[f"{m}_ci"] = f"[{lo:.3f},{hi:.3f}]"
                    row[f"{m}_median"] = round(float(np.nanmedian(t[m].values)), 4)
            main_rows.append(row)
        # paired tests (align by target)
        raw_p = []
        pend = []
        for hi_arm, lo_arm in COMPARISONS:
            if hi_arm not in tabs or lo_arm not in tabs:
                continue
            common = sorted(set(tabs[hi_arm].index) & set(tabs[lo_arm].index))
            for m in ["EF1%", "BEDROC20"]:
                if m not in tabs[hi_arm].columns:
                    continue
                hv = tabs[hi_arm].loc[common, m].values
                lv = tabs[lo_arm].loc[common, m].values
                diff = hv - lv
                nz = np.isfinite(diff)
                diff = diff[nz]
                if len(diff) < 10 or np.allclose(diff, 0):
                    continue
                try:
                    stat, p = wilcoxon(diff, alternative="greater")
                except ValueError:
                    p = 1.0
                rel = float(np.nanmean(hv[nz])) / max(1e-9, float(np.nanmean(lv[nz]))) - 1.0
                pend.append({"scheme": scheme, "comparison": f"{SHORT[hi_arm]}-{SHORT[lo_arm]}",
                             "metric": m, "n_targets": int(len(diff)),
                             "median_diff": round(float(np.median(diff)), 4),
                             "mean_hi": round(float(np.nanmean(hv[nz])), 4),
                             "mean_lo": round(float(np.nanmean(lv[nz])), 4),
                             "rel_improve": round(rel, 4),
                             "wilcoxon_p": p})
                raw_p.append(p)
        if raw_p:
            adj = holm(np.array(raw_p))
            for r, pa in zip(pend, adj):
                r["holm_p"] = round(float(pa), 5)
                r["sig_0.05"] = bool(pa < 0.05)
            test_rows.extend(pend)

    main_df = pd.DataFrame(main_rows)
    test_df = pd.DataFrame(test_rows)
    main_df.to_csv(os.path.join(args.out, f"{args.name}_main_results.csv"), index=False)
    test_df.to_csv(os.path.join(args.out, f"{args.name}_pairwise_tests.csv"), index=False)
    print("[agg] MAIN RESULTS:\n", main_df.to_string(index=False))
    print("\n[agg] PAIRWISE TESTS:\n", test_df.to_string(index=False))
    print(f"\n[agg] wrote -> {args.out}/{args.name}_{{main_results,pairwise_tests}}.csv")


if __name__ == "__main__":
    main()
