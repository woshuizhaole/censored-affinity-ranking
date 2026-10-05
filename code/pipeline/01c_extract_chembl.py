#!/usr/bin/env python
"""Stage 1c: extract censored affinity activities from ChEMBL 37 SQLite.

Emits (into data/interim/):
  chembl_measurements.parquet : smiles, target_id(uniprot), endpoint, value_nM, censor, pchembl, source
  chembl_targets.parquet      : target_id, uniprot, seq, name
  chembl_censoring_ratio.csv  : endpoint x censor counts (headline number, incl. full GROUP BY standard_relation)

Usage: python 01c_extract_chembl.py --db <chembl_37.db> --out <interim_dir>
"""
from __future__ import annotations
import os
import sys
import argparse
import sqlite3
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.censoring import relation_to_censor, EXACT, LEFT, RIGHT  # noqa

EXTRACT_SQL = """
SELECT cs.accession               AS uniprot,
       cs.sequence                AS seq,
       td.pref_name               AS tname,
       a.standard_type            AS endpoint,
       a.standard_relation        AS relation,
       a.standard_value           AS value_nM,
       a.pchembl_value            AS pchembl,
       struct.canonical_smiles    AS smiles
FROM activities a
JOIN assays ass              ON a.assay_id = ass.assay_id
JOIN target_dictionary td    ON ass.tid = td.tid
JOIN target_components tc    ON td.tid = tc.tid
JOIN component_sequences cs  ON tc.component_id = cs.component_id
JOIN compound_structures struct ON a.molregno = struct.molregno
WHERE a.standard_flag = 1
  AND a.standard_units = 'nM'
  AND a.standard_type IN ('IC50','Ki','Kd','EC50')
  AND a.standard_value IS NOT NULL
  AND td.target_type = 'SINGLE PROTEIN'
  AND ass.confidence_score >= 8
"""

# full relation distribution (headline), unrestricted to our relation whitelist
RATIO_SQL = """
SELECT standard_type, standard_relation, COUNT(*) AS n
FROM activities
WHERE standard_flag = 1 AND standard_units = 'nM'
  AND standard_type IN ('IC50','Ki','Kd','EC50') AND standard_value IS NOT NULL
GROUP BY standard_type, standard_relation
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    con = sqlite3.connect(args.db)

    # headline relation distribution
    ratio_raw = pd.read_sql_query(RATIO_SQL, con)
    ratio_raw.to_csv(os.path.join(args.out, "chembl_relation_distribution_raw.csv"), index=False)
    print("[raw relation distribution]\n", ratio_raw.to_string(index=False))

    print("[extract] running main SQL (single protein, conf>=8, nM)...")
    meas_rows = []
    targets = {}
    ratio = defaultdict(Counter)
    cur = con.execute(EXTRACT_SQL)
    cols = [d[0] for d in cur.description]
    n = 0
    while True:
        batch = cur.fetchmany(100000)
        if not batch:
            break
        for r in batch:
            d = dict(zip(cols, r))
            n += 1
            smiles = (d["smiles"] or "").strip()
            seq = (d["seq"] or "").strip().replace(" ", "").upper()
            uniprot = (d["uniprot"] or "").strip()
            if not smiles or not seq or not uniprot or len(seq) < 10:
                continue
            c = relation_to_censor(d["relation"])
            if c is None:
                ratio[d["endpoint"]]["approx_or_drop"] += 1
                continue
            try:
                val = float(d["value_nM"])
            except (TypeError, ValueError):
                continue
            if val <= 0:
                continue
            tid = uniprot
            if tid not in targets:
                targets[tid] = (uniprot, seq, d["tname"] or "")
            ratio[d["endpoint"]][{EXACT: "exact", LEFT: "left(>)", RIGHT: "right(<)"}[c]] += 1
            meas_rows.append((smiles, tid, d["endpoint"], val, c,
                              d["pchembl"], "ChEMBL"))
        if n % 1000000 == 0:
            print(f"[progress] {n} activity rows scanned, {len(meas_rows)} kept")
    con.close()

    meas = pd.DataFrame(meas_rows, columns=["smiles", "target_id", "endpoint", "value_nM", "censor", "pchembl", "source"])
    tgt = pd.DataFrame([(k, v[0], v[1], v[2]) for k, v in targets.items()],
                       columns=["target_id", "uniprot", "seq", "name"])
    meas.to_parquet(os.path.join(args.out, "chembl_measurements.parquet"), index=False)
    tgt.to_parquet(os.path.join(args.out, "chembl_targets.parquet"), index=False)

    rr = []
    for e, c in ratio.items():
        keep = c.get("exact", 0) + c.get("left(>)", 0) + c.get("right(<)", 0)
        rr.append({"endpoint": e, "kept_total": keep, "exact": c.get("exact", 0),
                   "left(>)": c.get("left(>)", 0), "right(<)": c.get("right(<)", 0),
                   "approx_or_drop": c.get("approx_or_drop", 0),
                   "pct_censored": round(100.0 * (c.get("left(>)", 0) + c.get("right(<)", 0)) / max(1, keep), 2)})
    rdf = pd.DataFrame(rr).sort_values("kept_total", ascending=False)
    rdf.to_csv(os.path.join(args.out, "chembl_censoring_ratio.csv"), index=False)
    print("[censoring ratio kept]\n", rdf.to_string(index=False))
    print(f"[out] measurements={len(meas)} targets={len(tgt)} -> {args.out}")


if __name__ == "__main__":
    main()
