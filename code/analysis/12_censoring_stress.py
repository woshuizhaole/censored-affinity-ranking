#!/usr/bin/env python
"""Stage 8: censoring stress test (graceful degradation vs cast-exact collapse).

For a grid of thresholds theta, artificially re-censor TRAIN-split exact rows with
pAff < theta as left-censored at theta (hide true value); TEST stays clean. Then
retrain B/C/D/F and measure VS-enrichment degradation vs the re-censored fraction.
H3: Tobit/rank (C/D/F) degrade gracefully, cast-exact (B) collapses.

Implementation: create one re-censored dataset variant per theta (only train rows
modified, row order preserved so the split file still aligns), then run train_arm
as parallel subprocesses (grid-style), and collate degradation curves.

Usage: python 12_censoring_stress.py --dataset <proc.parquet> --split <target_disjoint.parquet> \
   --fp <fp.npz> --emb <emb.npz> --thetas 4.0 4.5 5.0 5.5 6.0 --arms B_cast C_bounded D_tobit F_tobit_rankdec \
   --seeds 1 2 3 --out results/stress --name bindingdb_IC50 --workers 10 --threads 5
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
PYBIN = os.path.join(PROJ, "env", "conda", "bin", "python")
TRAIN = os.path.join(HERE, "train_arm.py")


def make_variant(df, split, theta, out_path):
    """Re-censor TRAIN exact rows with pAff<theta -> left-censored at theta. Save."""
    d = df.merge(split, on="row_id", how="left")
    mask = (d["split"] == "train") & (d["censor"] == 0) & (d["y_pscale"] < theta)
    n = int(mask.sum())
    d.loc[mask, "y_pscale"] = float(theta)
    d.loc[mask, "censor"] = -1
    d = d.drop(columns=["split"])
    d.to_parquet(out_path, index=False)
    return n


def run_job(spec):
    arm, seed, theta, dsvar, split_file, fp, emb, outdir, threads, max_epochs = spec
    os.makedirs(outdir, exist_ok=True)
    res = os.path.join(outdir, f"result_{arm}_seed{seed}.json")
    if os.path.exists(res):
        return (arm, seed, theta, "cached")
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    cmd = [PYBIN, "-u", TRAIN, "--dataset", dsvar, "--split", split_file, "--fp", fp,
           "--emb", emb, "--arm", arm, "--seed", str(seed), "--device", "cpu", "--out", outdir]
    if max_epochs:
        cmd += ["--max-epochs", str(max_epochs)]
    log = os.path.join(outdir, f"log_{arm}_seed{seed}.log")
    with open(log, "w") as lf:
        r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return (arm, seed, theta, "ok" if r.returncode == 0 else f"FAIL{r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--fp", required=True); ap.add_argument("--emb", required=True)
    ap.add_argument("--thetas", nargs="+", type=float, default=[4.0, 4.5, 5.0, 5.5, 6.0])
    ap.add_argument("--arms", nargs="+", default=["B_cast", "C_bounded", "D_tobit", "F_tobit_rankdec"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--out", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=None, help="cap epochs (fixed fair budget across arms/thetas)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    vardir = os.path.join(PROJ, "data", "interim", "stress"); os.makedirs(vardir, exist_ok=True)

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    split = pd.read_parquet(args.split)

    specs = []; frac = {}
    for theta in args.thetas:
        dsvar = os.path.join(vardir, f"{args.name}_theta{theta}.parquet")
        n_re = make_variant(df, split, theta, dsvar)
        n_train = int((split["split"] == "train").sum())
        frac[theta] = round(n_re / max(1, n_train), 4)
        outdir = os.path.join(args.out, f"theta{theta}")
        for arm in args.arms:
            for seed in args.seeds:
                specs.append((arm, seed, theta, dsvar, args.split, args.fp, args.emb, outdir, args.threads, args.max_epochs))
        print(f"[stress] theta={theta}: re-censored {n_re} train rows (frac {frac[theta]})", flush=True)

    print(f"[stress] {len(specs)} jobs, {args.workers} workers")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, s) for s in specs]
        for fu in as_completed(futs):
            a, s, t, st = fu.result(); done += 1
            print(f"[stress] ({done}/{len(specs)}) theta{t} {a} seed{s}: {st}", flush=True)

    # collate
    rows = []
    for theta in args.thetas:
        outdir = os.path.join(args.out, f"theta{theta}")
        for arm in args.arms:
            for seed in args.seeds:
                jf = os.path.join(outdir, f"result_{arm}_seed{seed}.json")
                if not os.path.exists(jf):
                    continue
                res = json.load(open(jf))
                row = {"theta": theta, "frac_recensored": frac[theta], "arm": arm, "seed": seed,
                       "n_test_targets": res.get("n_test_targets")}
                for m in ("EF1%", "BEDROC20", "AUC"):
                    v = (res.get("summary") or {}).get(m)
                    if isinstance(v, dict):
                        row[f"{m}_mean"] = v.get("mean")
                rows.append(row)
    out = pd.DataFrame(rows)
    tdir = os.path.join(PROJ, "results", "tables"); os.makedirs(tdir, exist_ok=True)
    outcsv = os.path.join(tdir, f"{args.name}_censoring_stress.csv")
    out.to_csv(outcsv, index=False)
    # degradation summary: mean over seeds per (theta, arm)
    if len(out):
        piv = out.groupby(["arm", "theta", "frac_recensored"])[["EF1%_mean", "BEDROC20_mean"]].mean().reset_index()
        print("[stress] degradation (mean over seeds):\n", piv.to_string(index=False))
    print(f"[stress] wrote -> {outcsv}")


if __name__ == "__main__":
    main()
