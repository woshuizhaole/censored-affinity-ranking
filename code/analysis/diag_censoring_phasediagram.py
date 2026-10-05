#!/usr/bin/env python
"""LINCHPIN verification: is the pooled null a Simpson's-paradox artifact of low-censoring
targets? Stratify per-target VS gains (F-B, D-B, C-B, C-A) by each target's natural censoring
rate, across datasets x splits, with partial correlation controlling target size.
Produces canonical results/tables/censoring_phasediagram_*.csv and prints the phase diagram."""
import os, sys, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr

PROJ = "/data/workplace/wyx/censored_affinity_20260707"
ARMS = ["A_drop", "B_cast", "C_bounded", "D_tobit", "E_tobit_rankobs", "F_tobit_rankdec"]
METRICS = ["EF1%", "BEDROC20", "AUC"]
BINS = [0.0, 0.10, 0.20, 0.40, 1.01]
BINLAB = ["LOW<10%", "MID10-20%", "HIGH20-40%", "XHIGH>=40%"]

DATASETS = [
    ("bindingdb_IC50", "results/mainline", "data/processed/splits/splits_bindingdb_IC50_%s.parquet",
     "data/processed/bindingdb_IC50_censored.parquet", ["random", "scaffold"]),
    ("bindingdb_Ki", "results/stage9_ki", "data/processed/splits/splits_bindingdb_Ki_%s.parquet",
     "data/processed/bindingdb_Ki_censored.parquet", ["random", "scaffold"]),
    ("chembl_IC50", "results/stage9_chembl", "data/processed/splits/splits_chembl_IC50_%s.parquet",
     "data/processed/chembl_IC50_censored.parquet", ["random", "scaffold"]),
]


def arm_pertarget(rdir, scheme, arm):
    fs = glob.glob(os.path.join(PROJ, rdir, scheme, "pertarget_%s_seed*.csv" % arm))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    cols = [m for m in METRICS if m in d.columns] + (["n_active"] if "n_active" in d.columns else [])
    return d.groupby("target")[cols].mean()


def target_censoring(dspath, splitpath):
    df = pd.read_parquet(os.path.join(PROJ, dspath)).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    sp = pd.read_parquet(os.path.join(PROJ, splitpath))
    df = df.merge(sp, on="row_id", how="inner")
    te = df[df.split == "test"]
    g = te.groupby("target_id")
    return pd.DataFrame({
        "cens_ratio": g["censor"].apply(lambda s: float((s != 0).mean())),
        "left_ratio": g["censor"].apply(lambda s: float((s == -1).mean())),
        "n_meas": g.size(),
    })


def partial_spearman(df, y, x, ctrls):
    """Spearman(y, x) controlling for ctrls via rank-residualization."""
    from numpy.linalg import lstsq
    d = df[[y, x] + ctrls].dropna()
    if len(d) < 30:
        return np.nan, np.nan, len(d)
    R = d.rank()
    C = np.column_stack([np.ones(len(d))] + [R[c].values for c in ctrls])
    ry = R[y].values - C @ lstsq(C, R[y].values, rcond=None)[0]
    rx = R[x].values - C @ lstsq(C, R[x].values, rcond=None)[0]
    rho, p = spearmanr(ry, rx)
    return rho, p, len(d)


allrows = []
for dsname, rdir, splitfmt, dspath, schemes in DATASETS:
    for scheme in schemes:
        splitpath = splitfmt % scheme
        if not os.path.exists(os.path.join(PROJ, splitpath)):
            continue
        tabs = {a: arm_pertarget(rdir, scheme, a) for a in ARMS}
        tabs = {a: t for a, t in tabs.items() if t is not None}
        if "B_cast" not in tabs:
            continue
        common = set.intersection(*[set(t.index) for t in tabs.values()])
        cens = target_censoring(dspath, splitpath)
        common = sorted(common & set(cens.index))
        base = pd.DataFrame(index=common)
        for a in tabs:
            for m in METRICS:
                if m in tabs[a].columns:
                    base["%s__%s" % (a, m)] = tabs[a].loc[common, m].values
        base = base.join(cens.reindex(common))
        base["n_active"] = tabs["B_cast"].loc[common, "n_active"].values if "n_active" in tabs["B_cast"].columns else np.nan
        base["cens_bin"] = pd.cut(base["cens_ratio"], BINS, labels=BINLAB, right=False)
        for m in ["EF1%", "BEDROC20"]:
            for hi, lo in [("F_tobit_rankdec", "B_cast"), ("D_tobit", "B_cast"),
                           ("C_bounded", "B_cast"), ("C_bounded", "A_drop")]:
                col = "gain_%s-%s__%s" % (hi.split("_")[0], lo.split("_")[0], m)
                if "%s__%s" % (hi, m) in base and "%s__%s" % (lo, m) in base:
                    base[col] = base["%s__%s" % (hi, m)] - base["%s__%s" % (lo, m)]
        # stratified means per bin
        for lab in BINLAB:
            sub = base[base["cens_bin"] == lab]
            if len(sub) == 0:
                continue
            row = {"dataset": dsname, "split": scheme, "cens_bin": lab, "n_targets": len(sub)}
            for a in tabs:
                if "%s__EF1%%" % a in sub:
                    row["%s_EF1" % a[0]] = round(float(sub["%s__EF1%%" % a].mean()), 3)
            for gc in [c for c in base.columns if c.startswith("gain_") and "EF1%" in c]:
                row[gc.replace("__EF1%", "_EF1")] = round(float(sub[gc].mean()), 3)
            for gc in [c for c in base.columns if c.startswith("gain_") and "BEDROC20" in c]:
                row[gc.replace("__BEDROC20", "_BD")] = round(float(sub[gc].mean()), 3)
            allrows.append(row)
        # partial correlation controlling target size
        base["log_nmeas"] = np.log1p(base["n_meas"]); base["log_nact"] = np.log1p(base["n_active"])
        print("\n### %s / %s (n=%d targets) ###" % (dsname, scheme, len(base)))
        for gc, lab in [("gain_F-B__BEDROC20", "F-B BEDROC"), ("gain_D-B__BEDROC20", "D-B BEDROC"),
                        ("gain_F-B__EF1%", "F-B EF1%"), ("gain_C-A__BEDROC20", "C-A BEDROC")]:
            if gc in base:
                rho, p = spearmanr(base[gc], base["cens_ratio"], nan_policy="omit")
                pr, pp, pn = partial_spearman(base, gc, "cens_ratio", ["log_nmeas", "log_nact"])
                print("  %-14s vs cens : spearman=%.3f(p=%.1e) | partial(ctrl size)=%.3f(p=%.1e n=%d)"
                      % (lab, rho, p, pr, pp, pn))
        # D3: left-censoring specificity — is the gain driven by LEFT-censored (hidden weak binders,
        # the ">X" measurements the Tobit left-tail likelihood targets)? Report left_ratio + partial
        # controlling for right_ratio, to isolate left- from right-censoring.
        base["right_ratio"] = base["cens_ratio"] - base["left_ratio"]
        for gc, lab in [("gain_F-B__BEDROC20", "F-B BEDROC"), ("gain_D-B__BEDROC20", "D-B BEDROC")]:
            if gc in base:
                rl, pl = spearmanr(base[gc], base["left_ratio"], nan_policy="omit")
                rr, prr = spearmanr(base[gc], base["right_ratio"], nan_policy="omit")
                prt, ppt, pnt = partial_spearman(base, gc, "left_ratio", ["log_nmeas", "log_nact", "right_ratio"])
                print("  %-14s vs LEFT: spearman=%.3f(p=%.1e) [right:%.3f p=%.1e] | partial(ctrl size+right)=%.3f(p=%.1e)"
                      % (lab, rl, pl, rr, prr, prt, ppt))

out = pd.DataFrame(allrows)
os.makedirs(os.path.join(PROJ, "results/tables"), exist_ok=True)
outp = os.path.join(PROJ, "results/tables/censoring_phasediagram.csv")
out.to_csv(outp, index=False)
print("\n=== PHASE DIAGRAM (F-B / D-B / C-B gain by censoring bin) ===")
show = [c for c in out.columns if c in ("dataset", "split", "cens_bin", "n_targets")
        or c.startswith("gain_F-B") or c.startswith("gain_D-B") or c.startswith("gain_C-B")]
print(out[show].to_string(index=False))
print("\nwrote -> %s" % outp)
