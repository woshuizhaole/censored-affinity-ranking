#!/usr/bin/env python
"""Stage 5: generate leakage-controlled splits for a processed dataset.

For dataset D (stable parquet row order), emit per-row split assignments under
three schemes, each as splits_<name>_<scheme>.parquet with columns [row_id, split]:
  - target_disjoint : MMseqs2 cluster targets (min-seq-id 0.3, c 0.8), whole
                      clusters to one fold  -> unseen targets in test.
  - scaffold        : Bemis-Murcko scaffold split over compounds.
  - random          : per-row random 70/10/20.

Usage: python 05_make_splits.py --dataset <processed.parquet> --name <tag> \
         --out data/processed/splits [--mmseqs env/conda/bin/mmseqs] [--seed 42]
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits.split_utils import (mmseqs_cluster, assign_folds_by_group,
                                scaffold_split)  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    print(f"[splits] {args.name}: {len(df)} rows, "
          f"{df['target_id'].nunique()} targets, {df['canonical_smiles'].nunique()} compounds")

    # --- target-disjoint (MMseqs2) ---
    id2seq = df.drop_duplicates("target_id").set_index("target_id")["seq"].to_dict()
    workdir = os.path.join(args.out, f"mmseqs_{args.name}")
    try:
        mapping = mmseqs_cluster(id2seq, workdir, min_seq_id=args.min_seq_id,
                                 cov=0.8, mmseqs_bin=args.mmseqs)
        split_of_t = assign_folds_by_group(mapping, fracs=(0.7, 0.1, 0.2), seed=args.seed)
        n_clusters = len(set(mapping.values()))
        df["_td"] = df["target_id"].map(split_of_t)
        # targets MMseqs2 didn't cluster (unmapped) -> train (safe: unknown
        # similarity to train, and train-only can never leak into test)
        n_unmapped = int(df["_td"].isna().sum())
        if n_unmapped:
            print(f"[splits] target_disjoint: {n_unmapped} unmapped rows -> train")
            df["_td"] = df["_td"].fillna("train")
        td = df[["row_id", "_td"]].rename(columns={"_td": "split"})
        td.to_parquet(os.path.join(args.out, f"splits_{args.name}_target_disjoint.parquet"), index=False)
        print(f"[splits] target_disjoint: {n_clusters} clusters -> "
              f"{td['split'].value_counts().to_dict()}")
    except Exception as e:
        print(f"[splits] WARN target_disjoint failed: {e}")

    # --- scaffold ---
    uniq_smiles = df["canonical_smiles"].unique().tolist()
    split_of_c = scaffold_split(uniq_smiles, fracs=(0.8, 0.1, 0.1), seed=args.seed)
    df["_sc"] = df["canonical_smiles"].map(split_of_c)
    sc = df[["row_id", "_sc"]].rename(columns={"_sc": "split"})
    sc.to_parquet(os.path.join(args.out, f"splits_{args.name}_scaffold.parquet"), index=False)
    print(f"[splits] scaffold -> {sc['split'].value_counts().to_dict()}")

    # --- random per-row ---
    rng = np.random.default_rng(args.seed)
    r = rng.random(len(df))
    rnd = pd.DataFrame({"row_id": df["row_id"],
                        "split": np.where(r < 0.7, "train", np.where(r < 0.8, "valid", "test"))})
    rnd.to_parquet(os.path.join(args.out, f"splits_{args.name}_random.parquet"), index=False)
    print(f"[splits] random -> {rnd['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
