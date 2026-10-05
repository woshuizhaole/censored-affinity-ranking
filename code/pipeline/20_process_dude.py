#!/usr/bin/env python
"""Stage 9a-prep: process DUD-E into per-target (sequence, actives, decoys).

From all.tar.gz, for each target: read actives_final.ism / decoys_final.ism
(SMILES lists) and extract the protein sequence from receptor.pdb (SEQRES, else
CA ATOM records). Output a long parquet:
  data/external/dude/dude_compounds.parquet : target, seq, smiles, label(1/0)
  data/external/dude/dude_targets.parquet    : target, seq, n_active, n_decoy

Usage: python 20_process_dude.py --tar data/external/dude/all.tar.gz --out data/external/dude
"""
from __future__ import annotations
import os
import argparse
import tarfile
import pandas as pd

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def pdb_to_seq(pdb_text: str) -> str:
    """Prefer SEQRES of the first chain; else CA ATOM residues of first chain."""
    seqres = {}
    order = []
    for line in pdb_text.splitlines():
        if line.startswith("SEQRES"):
            chain = line[11]
            if chain not in seqres:
                seqres[chain] = []
                order.append(chain)
            for res in line[19:].split():
                seqres[chain].append(THREE2ONE.get(res, "X"))
    if order:
        return "".join(seqres[order[0]])
    # fallback: CA atoms of first chain
    seq = []
    seen = set()
    first_chain = None
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            chain = line[21]
            if first_chain is None:
                first_chain = chain
            if chain != first_chain:
                break
            resnum = line[22:27]
            if resnum in seen:
                continue
            seen.add(resnum)
            seq.append(THREE2ONE.get(line[17:20].strip(), "X"))
    return "".join(seq)


def read_ism(text: str):
    """DUD-E .ism lines: 'SMILES id [chembl_id]'. Return SMILES list."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if parts:
            out.append(parts[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # collect per target
    targets = {}   # code -> {seq, actives, decoys}
    with tarfile.open(args.tar, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if len(parts) < 3:
                continue
            code = parts[1]
            fname = parts[2]
            if fname not in ("receptor.pdb", "actives_final.ism", "decoys_final.ism"):
                continue
            targets.setdefault(code, {})
            data = tf.extractfile(m).read().decode("utf-8", "replace")
            if fname == "receptor.pdb":
                targets[code]["seq"] = pdb_to_seq(data)
            elif fname == "actives_final.ism":
                targets[code]["actives"] = read_ism(data)
            elif fname == "decoys_final.ism":
                targets[code]["decoys"] = read_ism(data)

    comp_rows = []
    tgt_rows = []
    for code, d in sorted(targets.items()):
        seq = d.get("seq", "")
        acts = d.get("actives", [])
        decs = d.get("decoys", [])
        if not seq or len(seq) < 10 or not acts:
            print(f"[dude] skip {code}: seq_len={len(seq)} n_act={len(acts)}")
            continue
        for s in acts:
            comp_rows.append((code, seq, s, 1))
        for s in decs:
            comp_rows.append((code, seq, s, 0))
        tgt_rows.append((code, seq, len(acts), len(decs)))
    comp = pd.DataFrame(comp_rows, columns=["target", "seq", "smiles", "label"])
    tgt = pd.DataFrame(tgt_rows, columns=["target", "seq", "n_active", "n_decoy"])
    comp.to_parquet(os.path.join(args.out, "dude_compounds.parquet"), index=False)
    tgt.to_parquet(os.path.join(args.out, "dude_targets.parquet"), index=False)
    print(f"[dude] {len(tgt)} targets, {len(comp)} compounds "
          f"(active {int((comp.label==1).sum())}, decoy {int((comp.label==0).sum())}) -> {args.out}")
    print(tgt.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
