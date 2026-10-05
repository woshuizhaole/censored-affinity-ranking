#!/usr/bin/env python
"""Stage 2a: standardize unique SMILES (chembl_structure_pipeline) in parallel.

Input : one or more measurement parquet files (need a 'smiles' column).
Output: data/interim/smiles_std_map.parquet  (raw_smiles, canonical_smiles, inchikey)
        rows that fail parsing/standardization are recorded but excluded from map.
"""
from __future__ import annotations
import os
import sys
import argparse
import pandas as pd
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from chembl_structure_pipeline import standardizer


def clean_smiles_string(smiles: str) -> str:
    """Drop CXSMILES extension (' |...|' suffix, ~397k BindingDB rows) and any
    trailing whitespace/tokens; keep the core SMILES for RDKit parsing."""
    s = smiles.strip()
    # CXSMILES: 'SMILES |extension|' -> take core before ' |'
    if " |" in s:
        s = s.split(" |", 1)[0].strip()
    # any remaining whitespace: core SMILES has none -> take first token
    if " " in s:
        s = s.split()[0]
    return s


def standardize_one(smiles: str):
    """Returns (raw_smiles, canonical_smiles, inchikey). raw_smiles is the ORIGINAL
    (merge key); cleaning is applied only to the copy passed to RDKit."""
    try:
        s = clean_smiles_string(smiles)
        if not s:
            return (smiles, None, None)
        m = Chem.MolFromSmiles(s)
        if m is None:
            return (smiles, None, None)
        std = standardizer.standardize_mol(m)
        parent, exclude = standardizer.get_parent_mol(std)
        if parent is None or exclude:
            return (smiles, None, None)
        can = Chem.MolToSmiles(parent)
        if not can:
            return (smiles, None, None)
        ik = Chem.MolToInchiKey(parent)
        return (smiles, can, ik or None)
    except Exception:
        return (smiles, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="measurement parquet(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--njobs", type=int, default=48)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    uniq = set()
    for p in args.inputs:
        df = pd.read_parquet(p, columns=["smiles"])
        uniq.update(df["smiles"].dropna().unique().tolist())
    uniq = sorted(uniq)
    print(f"[standardize] {len(uniq)} unique SMILES across {len(args.inputs)} inputs")

    res = Parallel(n_jobs=args.njobs, batch_size=512, verbose=5)(
        delayed(standardize_one)(s) for s in uniq
    )
    out = pd.DataFrame(res, columns=["smiles", "canonical_smiles", "inchikey"])
    n_fail = int(out["canonical_smiles"].isna().sum())
    print(f"[standardize] failed/excluded: {n_fail}/{len(out)} ({100*n_fail/len(out):.2f}%)")
    ok = out.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    ok.to_parquet(args.out, index=False)
    # also record failures for the reproducibility appendix
    out[out["canonical_smiles"].isna()][["smiles"]].to_csv(
        args.out.replace(".parquet", "_failures.csv"), index=False)
    print(f"[out] {len(ok)} standardized -> {args.out}")


if __name__ == "__main__":
    main()
