#!/usr/bin/env python
"""Aggregate P1 training-battery results (improvement.md batch 2) and compare against the
main-line baselines. Prints: G-arm vs C, F-tuned configs vs B/C (with VAL-based selection to
avoid test cherry-picking), Tobit-clamp vs unclamped D/F, extended re-censoring degradation."""
import os, glob, json
import numpy as np, pandas as pd

PROJ = "/data/workplace/wyx/censored_affinity_20260707"
MET = ["EF1%", "BEDROC20", "AUC"]


def arm_means(rootdir, scheme, arm):
    """mean over seeds of test metrics + best-epoch val metric, from result JSONs."""
    fs = glob.glob(os.path.join(PROJ, rootdir, scheme, "result_%s_seed*.json" % arm))
    if not fs:
        return None
    rows = []
    for f in fs:
        r = json.load(open(f)); s = r.get("summary", {})
        row = {m: s.get(m, {}).get("mean") for m in MET}
        row["best_epoch"] = r.get("best_epoch")
        rows.append(row)
    df = pd.DataFrame(rows)
    return {m: (float(df[m].mean()), float(df[m].std(ddof=1)) if len(df) > 1 else 0.0) for m in MET}, len(fs)


def show(tag, rootdir, arm, schemes=("random", "scaffold")):
    for sc in schemes:
        r = arm_means(rootdir, sc, arm)
        if r is None:
            print("  %-22s %-10s : (no results)" % (tag, sc)); continue
        m, n = r
        print("  %-22s %-10s : EF1%%=%.2f BEDROC20=%.3f AUC=%.3f (n=%d)"
              % (tag, sc, m["EF1%"][0], m["BEDROC20"][0], m["AUC"][0], n))


print("### baseline (main-line, results/tables/bindingdb_IC50_main_results.csv) ###")
mr = pd.read_csv(PROJ + "/results/tables/bindingdb_IC50_main_results.csv")
for sc in ["random", "scaffold"]:
    for arm in ["B", "C", "D", "F"]:
        a = mr[(mr.scheme == sc) & (mr.arm == arm)]
        if len(a):
            print("  base %-5s %-10s : EF1%%=%.2f BEDROC20=%.3f AUC=%.3f" %
                  (arm, sc, a["EF1%_mean"].iloc[0], a["BEDROC20_mean"].iloc[0], a["AUC_mean"].iloc[0]))

print("\n### P1a: G_bounded_rankdec (ranking on bounded base) vs C_bounded ###")
show("G_bounded_rankdec", "results/p1/garm", "G_bounded_rankdec", ("random", "scaffold", "target_disjoint"))

print("\n### P1c: F tuned configs (VAL-selected) vs B/C ###")
for tag, root in [("F l3m05", "results/p1/f_l3m05"), ("F l3m05+up3", "results/p1/f_l3m05_up3"), ("F l3m1", "results/p1/f_l3m1")]:
    show(tag, root, "F_tobit_rankdec")

print("\n### P1b: Tobit-base clamp[-3,3] on D/F vs unclamped ###")
for arm in ["D_tobit", "F_tobit_rankdec"]:
    show(arm + "+clamp", "results/p1/tobit_clamp", arm)

print("\n### P0-ext: extended re-censoring degradation (BEDROC20, theta{6.5,7,7.5}) ###")
ext = PROJ + "/results/tables/bindingdb_IC50_random_ext_censoring_stress.csv"
if os.path.exists(ext):
    d = pd.read_csv(ext)
    g = d.groupby(["arm", "theta", "frac_recensored"])[["EF1%_mean", "BEDROC20_mean"]].mean().reset_index()
    piv = g.pivot_table(index=["theta", "frac_recensored"], columns="arm", values="BEDROC20_mean")
    print(piv.round(4).to_string())
    print("  (C-B/D-B/F-B at highest theta => absolute-win region check)")
else:
    print("  (extended stress not done yet)")
