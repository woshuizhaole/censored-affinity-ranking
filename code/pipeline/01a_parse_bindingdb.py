#!/usr/bin/env python
"""Stage 1a: stream-parse BindingDB full TSV, keep censoring qualifiers.

Emits (into data/interim/):
  bindingdb_measurements.parquet : smiles, target_id, endpoint, value_nM, censor, source
  bindingdb_targets.parquet      : target_id, uniprot, seq, name, n_chains
  bindingdb_censoring_ratio.csv  : endpoint x censor counts (headline number)

Columns are resolved BY HEADER NAME (indices shift with chain count / releases).
Single-chain targets only (n_chains == 1), per proposal. Streaming, low memory.

Usage: python 01a_parse_bindingdb.py --tsv <path> --out <interim_dir>
"""
from __future__ import annotations
import os
import sys
import csv
import argparse
import hashlib
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # code/
from utils.censoring import parse_bindingdb_value, EXACT, LEFT, RIGHT  # noqa

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ENDPOINTS = ["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]
ENDPOINT_NAME = {"Ki (nM)": "Ki", "IC50 (nM)": "IC50", "Kd (nM)": "Kd", "EC50 (nM)": "EC50"}


def resolve_columns(header):
    idx = {h: i for i, h in enumerate(header)}

    def find_exact(name):
        return idx.get(name)

    def find_prefix(prefix):
        for i, h in enumerate(header):
            if h.startswith(prefix):
                return i
        return None

    cols = {
        "smiles": find_exact("Ligand SMILES"),
        "n_chains": find_prefix("Number of Protein Chains in Target"),
        "seq1": find_prefix("BindingDB Target Chain Sequence"),
        "uniprot1": find_prefix("UniProt (SwissProt) Primary ID of Target Chain"),
        "source": find_exact("Curation/DataSource"),
        "tname": find_exact("Target Name"),
    }
    for e in ENDPOINTS:
        cols[e] = find_exact(e)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.tsv, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        cols = resolve_columns(header)
        print("[resolve] header has", len(header), "columns")
        for k, v in cols.items():
            print(f"[resolve] {k:10s} -> idx {v}  ({header[v] if v is not None else 'MISSING'})")
        req = ["smiles", "seq1", "n_chains"] + ENDPOINTS
        missing = [k for k in req if cols[k] is None]
        if missing:
            print("[FATAL] missing required columns:", missing)
            sys.exit(1)

        meas_rows = []
        targets = {}          # target_id -> (uniprot, seq, name, n_chains)
        ratio = defaultdict(Counter)   # endpoint -> Counter over censor codes
        n_lines = 0
        n_multichain = 0
        n_noseq = 0
        n_nosmiles = 0
        maxcol = max(i for i in cols.values() if i is not None)

        for row in reader:
            n_lines += 1
            if len(row) <= maxcol:
                continue
            smiles = row[cols["smiles"]].strip()
            if not smiles:
                n_nosmiles += 1
                continue
            nch = row[cols["n_chains"]].strip()
            if nch != "1":
                n_multichain += 1
                continue
            seq = row[cols["seq1"]].strip().replace(" ", "").upper()
            if not seq or len(seq) < 10:
                n_noseq += 1
                continue
            uniprot = row[cols["uniprot1"]].strip() if cols["uniprot1"] is not None else ""
            name = row[cols["tname"]].strip() if cols["tname"] is not None else ""
            source = row[cols["source"]].strip() if cols["source"] is not None else ""
            tid = uniprot if uniprot else "SEQ:" + hashlib.md5(seq.encode()).hexdigest()[:12]
            if tid not in targets:
                targets[tid] = (uniprot, seq, name, nch)

            for e in ENDPOINTS:
                raw = row[cols[e]].strip()
                if not raw:
                    continue
                parsed = parse_bindingdb_value(raw)
                if parsed is None:
                    ratio[ENDPOINT_NAME[e]]["unparsed_or_approx"] += 1
                    continue
                value_nM, censor = parsed
                ratio[ENDPOINT_NAME[e]][{EXACT: "exact", LEFT: "left(>)", RIGHT: "right(<)"}[censor]] += 1
                meas_rows.append((smiles, tid, ENDPOINT_NAME[e], value_nM, censor, source))

            if n_lines % 500000 == 0:
                print(f"[progress] {n_lines} lines, {len(meas_rows)} measurements, {len(targets)} targets")

    print(f"[done] scanned {n_lines} data lines")
    print(f"[filter] dropped: no_smiles={n_nosmiles}, multichain={n_multichain}, no_seq={n_noseq}")

    meas = pd.DataFrame(meas_rows, columns=["smiles", "target_id", "endpoint", "value_nM", "censor", "source"])
    tgt = pd.DataFrame(
        [(k, v[0], v[1], v[2], v[3]) for k, v in targets.items()],
        columns=["target_id", "uniprot", "seq", "name", "n_chains"],
    )
    meas.to_parquet(os.path.join(args.out, "bindingdb_measurements.parquet"), index=False)
    tgt.to_parquet(os.path.join(args.out, "bindingdb_targets.parquet"), index=False)

    # censoring ratio table
    rr = []
    for e, c in ratio.items():
        tot = sum(c.values())
        rr.append({
            "endpoint": e, "total": tot,
            "exact": c.get("exact", 0), "left(>)": c.get("left(>)", 0),
            "right(<)": c.get("right(<)", 0), "unparsed_or_approx": c.get("unparsed_or_approx", 0),
            "pct_censored": round(100.0 * (c.get("left(>)", 0) + c.get("right(<)", 0)) /
                                  max(1, c.get("exact", 0) + c.get("left(>)", 0) + c.get("right(<)", 0)), 2),
        })
    rdf = pd.DataFrame(rr).sort_values("total", ascending=False)
    rdf.to_csv(os.path.join(args.out, "bindingdb_censoring_ratio.csv"), index=False)
    print("[censoring ratio]\n", rdf.to_string(index=False))
    print(f"[out] measurements={len(meas)} targets={len(tgt)} -> {args.out}")


if __name__ == "__main__":
    main()
