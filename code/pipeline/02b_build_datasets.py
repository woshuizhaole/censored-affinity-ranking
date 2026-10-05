#!/usr/bin/env python
"""Stage 2b: assemble per-endpoint censored datasets (p-scale, dedup, seq join).

Input : measurements parquet (smiles,target_id,endpoint,value_nM,censor,source[,pchembl]),
        targets parquet (target_id,seq,name,...), smiles_std_map parquet.
Output: data/processed/<source>_<endpoint>_censored.parquet columns:
        canonical_smiles, inchikey, target_id, seq, y_pscale, censor, source, endpoint
        + data/processed/<source>_stats.csv (per-endpoint counts, censor dist)

Dedup rule per (canonical_smiles, target_id, endpoint):
  - if any exact -> y = median(exact pAff), censor=0
  - else all-left  -> y = min(pAff), censor=-1  (tightest 'true < y')
  - else all-right -> y = max(pAff), censor=+1  (tightest 'true > y')
  - else (left & right, no exact) -> conflict, dropped (counted)
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.censoring import nM_to_pAffinity  # noqa

KEYS = ["canonical_smiles", "target_id", "endpoint"]


def aggregate(df):
    """Vectorized dedup aggregation. df has KEYS + pAff + censor."""
    out = []
    # 1) exact
    ex = df[df["censor"] == 0]
    if len(ex):
        g = ex.groupby(KEYS, sort=False)["pAff"].median().reset_index()
        g["censor"] = 0
        out.append(g)
        exact_keys = set(map(tuple, ex[KEYS].drop_duplicates().itertuples(index=False)))
    else:
        exact_keys = set()

    # 2) censored-only keys
    cen = df[df["censor"] != 0].copy()
    if len(cen):
        cen["_k"] = list(map(tuple, cen[KEYS].itertuples(index=False)))
        cen = cen[~cen["_k"].isin(exact_keys)]
    if len(cen):
        left = cen[cen["censor"] == -1]
        right = cen[cen["censor"] == +1]
        lk = set(map(tuple, left[KEYS].drop_duplicates().itertuples(index=False)))
        rk = set(map(tuple, right[KEYS].drop_duplicates().itertuples(index=False)))
        conflict = lk & rk
        # left-only
        if len(left):
            gl = left.groupby(KEYS, sort=False)["pAff"].min().reset_index()
            gl["_k"] = list(map(tuple, gl[KEYS].itertuples(index=False)))
            gl = gl[~gl["_k"].isin(conflict)].drop(columns="_k")
            gl["censor"] = -1
            out.append(gl)
        # right-only
        if len(right):
            gr = right.groupby(KEYS, sort=False)["pAff"].max().reset_index()
            gr["_k"] = list(map(tuple, gr[KEYS].itertuples(index=False)))
            gr = gr[~gr["_k"].isin(conflict)].drop(columns="_k")
            gr["censor"] = +1
            out.append(gr)
        n_conflict = len(conflict)
    else:
        n_conflict = 0

    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=KEYS + ["pAff", "censor"])
    return res, n_conflict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--stdmap", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-pAff", type=float, default=0.0)
    ap.add_argument("--max-pAff", type=float, default=13.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meas = pd.read_parquet(args.measurements)
    tgt = pd.read_parquet(args.targets)[["target_id", "seq"]].drop_duplicates("target_id")
    smap = pd.read_parquet(args.stdmap)[["smiles", "canonical_smiles", "inchikey"]]

    n0 = len(meas)
    meas = meas.merge(smap, on="smiles", how="inner")   # drops unstandardizable
    meas["pAff"] = 9.0 - np.log10(meas["value_nM"].astype(float))   # vectorized (util nM_to_pAffinity is scalar-only)
    meas = meas[(meas["pAff"] >= args.min_pAff) & (meas["pAff"] <= args.max_pAff)]
    print(f"[{args.source}] measurements {n0} -> after std+range {len(meas)}")

    stats = []
    for endpoint, dfe in meas.groupby("endpoint"):
        agg, n_conf = aggregate(dfe[KEYS + ["pAff", "censor"]])
        agg = agg.merge(tgt, on="target_id", how="inner")   # need seq
        # attach inchikey
        ik = dfe[["canonical_smiles", "inchikey"]].drop_duplicates("canonical_smiles")
        agg = agg.merge(ik, on="canonical_smiles", how="left")
        agg["source"] = args.source
        agg = agg.rename(columns={"pAff": "y_pscale"})
        agg = agg[["canonical_smiles", "inchikey", "target_id", "seq", "y_pscale", "censor", "source", "endpoint"]]
        outp = os.path.join(args.out_dir, f"{args.source}_{endpoint}_censored.parquet")
        agg.to_parquet(outp, index=False)
        cd = agg["censor"].value_counts().to_dict()
        stats.append({"endpoint": endpoint, "n_rows": len(agg),
                      "n_compounds": agg["canonical_smiles"].nunique(),
                      "n_targets": agg["target_id"].nunique(),
                      "exact": cd.get(0, 0), "left(-1)": cd.get(-1, 0), "right(+1)": cd.get(1, 0),
                      "conflicts_dropped": n_conf,
                      "pct_censored": round(100 * (cd.get(-1, 0) + cd.get(1, 0)) / max(1, len(agg)), 2)})
        print(f"[{args.source}/{endpoint}] rows={len(agg)} targets={agg['target_id'].nunique()} "
              f"compounds={agg['canonical_smiles'].nunique()} censored%={stats[-1]['pct_censored']}")
    sdf = pd.DataFrame(stats)
    sdf.to_csv(os.path.join(args.out_dir, f"{args.source}_stats.csv"), index=False)
    print(f"[{args.source} stats]\n", sdf.to_string(index=False))


if __name__ == "__main__":
    main()
