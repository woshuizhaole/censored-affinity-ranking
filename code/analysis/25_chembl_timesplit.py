#!/usr/bin/env python
"""improvement.md #5: naive TIME-SPLIT on ChEMBL (doc publication year) — the only experiment that
touches the original prospective-VS motivation. Assign each (inchikey,target) pair its FIRST disclosure
year from ChEMBL docs, then split train(old)/valid/test(recent). Writes a split parquet compatible with
06_run_grid (--schemes timesplit). Prospective in the realistic sense: known targets, temporally future
compounds (does NOT create cold targets — that limit stays explicit)."""
import sqlite3, os
import numpy as np, pandas as pd

PROJ = "/data/workplace/wyx/censored_affinity_20260707"
DB = PROJ + "/data/raw/chembl/chembl_37/chembl_37_sqlite/chembl_37.db"
DS = PROJ + "/data/processed/chembl_IC50_censored.parquet"
OUT = PROJ + "/data/processed/splits/splits_chembl_IC50_timesplit.parquet"

print("[1] staged single-scan extraction (avoids random-IO 6-table join)...")
con = sqlite3.connect(DB)
# each read_sql is ONE sequential scan; joins happen in RAM (fast) instead of sqlite's random-access join
print("    scan activities (IC50)..."); act = pd.read_sql(
    "SELECT molregno, assay_id FROM activities WHERE standard_type='IC50' AND molregno IS NOT NULL", con)
print("    scan assays..."); asy = pd.read_sql("SELECT assay_id, doc_id, tid FROM assays", con)
print("    scan docs..."); doc = pd.read_sql("SELECT doc_id, year FROM docs WHERE year IS NOT NULL", con)
print("    scan compound_structures..."); cs = pd.read_sql(
    "SELECT molregno, standard_inchi_key AS inchikey FROM compound_structures", con)
print("    scan target_components + component_sequences...")
tc = pd.read_sql("SELECT tid, component_id FROM target_components", con)
seq = pd.read_sql("SELECT component_id, accession FROM component_sequences WHERE accession IS NOT NULL", con)
con.close()
print("    activities=%d assays=%d docs=%d cs=%d" % (len(act), len(asy), len(doc), len(cs)))
# join in RAM: activity -> assay -> doc(year); activity.molregno -> inchikey; assay.tid -> accession
a = act.merge(asy, on="assay_id", how="inner").merge(doc, on="doc_id", how="inner")
a = a.merge(cs, on="molregno", how="inner")
a = a.merge(tc, on="tid", how="inner").merge(seq, on="component_id", how="inner")
a = a.rename(columns={"accession": "target_id"})
yr = a.groupby(["inchikey", "target_id"], as_index=False)["year"].min().rename(columns={"year": "first_year"})
print("    year rows:", len(yr), "year range", int(yr.first_year.min()), int(yr.first_year.max()))

print("[2] joining onto processed chembl_IC50 dataset...")
ds = pd.read_parquet(DS, columns=["inchikey", "target_id", "censor"])
ds["row_id"] = np.arange(len(ds))
m = ds.merge(yr, on=["inchikey", "target_id"], how="left")
matched = m.first_year.notna()
print("    total rows %d, matched year %d (%.1f%%)" % (len(m), matched.sum(), 100 * matched.mean()))

# time cutoffs by pair-year quantiles (train oldest ~70%, valid next ~10%, test newest ~20%)
yv = m.loc[matched, "first_year"].astype(int)
c_train = int(np.quantile(yv, 0.70)); c_valid = int(np.quantile(yv, 0.80))
print("    year distribution:", yv.value_counts().sort_index().to_dict())
print("    cutoffs: train<=%d, valid=(%d,%d], test>%d" % (c_train, c_train, c_valid, c_valid))

split = np.full(len(m), "drop", dtype=object)
yint = m.first_year.fillna(-1).astype(int).values
split[(yint >= 0) & (yint <= c_train)] = "train"
split[(yint > c_train) & (yint <= c_valid)] = "valid"
split[(yint > c_valid)] = "test"
out = pd.DataFrame({"row_id": m.row_id.values, "split": split})
# report test target/active coverage (VS needs actives per target)
te = m[split == "test"]
tr = m[split == "train"]
print("[3] split sizes:", pd.Series(split).value_counts().to_dict())
print("    test targets=%d, train targets=%d, shared=%d"
      % (te.target_id.nunique(), tr.target_id.nunique(),
         len(set(te.target_id) & set(tr.target_id))))
print("    test rows censor dist:", te.censor.value_counts().to_dict())
out.to_parquet(OUT, index=False)
print("[4] wrote ->", OUT)
