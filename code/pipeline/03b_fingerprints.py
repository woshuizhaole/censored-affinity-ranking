#!/usr/bin/env python
"""Stage 3b: ECFP4 (Morgan radius=2, 1024-bit) fingerprints for unique compounds.

Output: data/features/ligand_fp.npz  (keys: 'smiles' str array, 'fp' uint8 [N,1024])
Usage: python 03b_fingerprints.py --inputs <processed parquet...> --out <npz> [--bits 1024] [--radius 2]
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdFingerprintGenerator, DataStructs

_GEN = None


def _fp(smiles, radius, bits):
    global _GEN
    if _GEN is None:
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    bv = _GEN.GetFingerprint(m)
    arr = np.zeros((bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=1024)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--njobs", type=int, default=48)
    ap.add_argument("--smiles-col", default="canonical_smiles")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    uniq = set()
    for p in args.inputs:
        df = pd.read_parquet(p, columns=[args.smiles_col])
        uniq.update(df[args.smiles_col].dropna().unique().tolist())
    uniq = sorted(uniq)
    print(f"[fp] {len(uniq)} unique compounds")

    res = Parallel(n_jobs=args.njobs, batch_size=1024, verbose=5)(
        delayed(_fp)(s, args.radius, args.bits) for s in uniq
    )
    keep_s, keep_fp = [], []
    n_fail = 0
    for s, a in zip(uniq, res):
        if a is None:
            n_fail += 1
            continue
        keep_s.append(s); keep_fp.append(a)
    fp = np.vstack(keep_fp).astype(np.uint8)
    np.savez(args.out, smiles=np.array(keep_s, dtype=object), fp=fp)
    print(f"[fp] wrote {fp.shape} (failed={n_fail}) -> {args.out}")


if __name__ == "__main__":
    main()
