#!/usr/bin/env python
"""Stage 7: generalization analysis over the target-disjoint ablation results.

Per held-out test target, aggregate each arm's VS metrics over seeds, compute
per-target GAINS (F-B, D-B, F-E, F-D), annotate with (a) per-target censoring
ratio, (b) n_actives, (c) max sequence identity to train targets (MMseqs2), then:
  - Spearman(gain, censoring_ratio)  [mechanistic: more censoring -> more gain?]
  - gain binned by identity-to-train [novelty: helps more on novel targets?]

Usage: python 07_generalization.py --pertarget-dir results/mainline/target_disjoint \
   --dataset data/processed/bindingdb_IC50_censored.parquet \
   --split data/processed/splits/splits_bindingdb_IC50_target_disjoint.parquet \
   --mmseqs env/conda/bin/mmseqs --out results/tables --name bindingdb_IC50
"""
from __future__ import annotations
import os
import sys
import glob
import argparse
import subprocess
import tempfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))

ARMS = ["A_drop", "B_cast", "C_bounded", "D_tobit", "E_tobit_rankobs", "F_tobit_rankdec"]
METRICS = ["EF1%", "EF0.5%", "BEDROC20", "AUC"]


def load_arm_pertarget(pertarget_dir, arm):
    """Mean per-target metrics over seeds for one arm. Returns df indexed by target."""
    files = glob.glob(os.path.join(pertarget_dir, f"pertarget_{arm}_seed*.csv"))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    alld = pd.concat(dfs, ignore_index=True)
    agg = alld.groupby("target")[[m for m in METRICS if m in alld.columns]].mean()
    agg["n_active"] = alld.groupby("target")["n_active"].mean()
    return agg


def censoring_ratio_per_target(dataset, split):
    df = pd.read_parquet(dataset).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    sp = pd.read_parquet(split)
    df = df.merge(sp, on="row_id", how="inner")
    test = df[df.split == "test"]
    g = test.groupby("target_id")
    out = pd.DataFrame({
        "n_meas": g.size(),
        "cens_ratio": g["censor"].apply(lambda s: float((s != 0).mean())),
        "left_ratio": g["censor"].apply(lambda s: float((s == -1).mean())),
    })
    return out, df


def max_identity_to_train(df, mmseqs_bin, workdir):
    """MMseqs2 easy-search test targets vs train targets -> max pident per test target."""
    os.makedirs(workdir, exist_ok=True)
    tr = df[df.split == "train"].drop_duplicates("target_id")[["target_id", "seq"]]
    te = df[df.split == "test"].drop_duplicates("target_id")[["target_id", "seq"]]
    fa_tr = os.path.join(workdir, "train.fasta"); fa_te = os.path.join(workdir, "test.fasta")
    for fa, d in [(fa_tr, tr), (fa_te, te)]:
        with open(fa, "w") as fh:
            for tid, seq in zip(d["target_id"], d["seq"]):
                s = "".join(ch if ch.isalpha() else "X" for ch in str(seq).upper())
                if len(s) >= 10:
                    fh.write(f">{tid}\n{s}\n")
    res = os.path.join(workdir, "res.m8"); tmp = os.path.join(workdir, "tmp")
    try:
        subprocess.run([mmseqs_bin, "easy-search", fa_te, fa_tr, res, tmp,
                        "--max-seqs", "50", "-s", "6.0", "--format-output",
                        "query,target,pident"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[gen] WARN mmseqs easy-search failed: {e}")
        return pd.DataFrame(columns=["max_ident_to_train"])
    if not os.path.exists(res):
        return pd.DataFrame(columns=["max_ident_to_train"])
    m8 = pd.read_csv(res, sep="\t", names=["query", "target", "pident"])
    maxid = m8.groupby("query")["pident"].max().rename("max_ident_to_train")
    return maxid.to_frame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pertarget-dir", required=True)
    ap.add_argument("--dataset", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--out", required=True); ap.add_argument("--name", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arm_tabs = {a: load_arm_pertarget(args.pertarget_dir, a) for a in ARMS}
    arm_tabs = {a: t for a, t in arm_tabs.items() if t is not None}
    print(f"[gen] arms with results: {list(arm_tabs)}")

    # per-target gains on common targets
    common = None
    for t in arm_tabs.values():
        common = set(t.index) if common is None else (common & set(t.index))
    common = sorted(common)
    print(f"[gen] {len(common)} common test targets across arms")

    base = pd.DataFrame(index=common)
    for m in METRICS:
        for a in arm_tabs:
            if m in arm_tabs[a].columns:
                base[f"{a}__{m}"] = arm_tabs[a].loc[common, m].values
    gains = pd.DataFrame(index=common)
    pairs = [("B_cast", "A_drop"), ("C_bounded", "A_drop"),   # keep-vs-drop (the real H1 contrast)
             ("F_tobit_rankdec", "B_cast"), ("D_tobit", "B_cast"),
             ("F_tobit_rankdec", "E_tobit_rankobs"), ("F_tobit_rankdec", "D_tobit"),
             ("C_bounded", "B_cast"), ("D_tobit", "A_drop")]
    for m in ["EF1%", "BEDROC20"]:
        for hi, lo in pairs:
            if f"{hi}__{m}" in base and f"{lo}__{m}" in base:
                gains[f"gain_{hi.split('_')[0]}-{lo.split('_')[0]}__{m}"] = base[f"{hi}__{m}"] - base[f"{lo}__{m}"]

    # annotate
    cens, df = censoring_ratio_per_target(args.dataset, args.split)
    ann = gains.join(base).join(cens.reindex(common))
    try:
        ident = max_identity_to_train(df, args.mmseqs, os.path.join(args.out, f"mmseqs_gen_{args.name}"))
        ann = ann.join(ident.reindex(common))
    except Exception as e:
        print(f"[gen] identity annotation skipped: {e}")
    ann.index.name = "target_id"
    ann.to_csv(os.path.join(args.out, f"{args.name}_pertarget_gain_annotated.csv"))

    # correlations of gain vs censoring ratio / identity
    corr_rows = []
    for gcol in [c for c in ann.columns if c.startswith("gain_")]:
        for xcol in ["cens_ratio", "left_ratio", "max_ident_to_train", "n_active"]:
            if xcol in ann.columns:
                sub = ann[[gcol, xcol]].dropna()
                if len(sub) >= 20:
                    rho, p = spearmanr(sub[gcol], sub[xcol])
                    corr_rows.append({"gain": gcol, "vs": xcol, "n": len(sub),
                                      "spearman_rho": round(rho, 4), "p": round(p, 5),
                                      "mean_gain": round(float(sub[gcol].mean()), 4)})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(os.path.join(args.out, f"{args.name}_gain_correlations.csv"), index=False)
    print("[gen] correlations:\n", corr.to_string(index=False))

    # identity-binned gain
    if "max_ident_to_train" in ann.columns:
        ann["ident_bin"] = pd.cut(ann["max_ident_to_train"], [0, 30, 50, 100],
                                  labels=["<30", "30-50", ">50"])
        binned = ann.groupby("ident_bin")[[c for c in ann.columns if c.startswith("gain_") and "EF1%" in c]].mean()
        binned.to_csv(os.path.join(args.out, f"{args.name}_gain_by_identity.csv"))
        print("[gen] gain by identity bin:\n", binned.to_string())
    print(f"[gen] wrote annotated gains + correlations -> {args.out}")


if __name__ == "__main__":
    main()
