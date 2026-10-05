#!/usr/bin/env python
"""Stage 6 orchestrator: run the A-F ablation grid (arms x splits x seeds [x lambda])
by invoking train_arm.py as parallel CPU subprocesses, then aggregate results.

Each job writes result_<tag>.json + pertarget_<tag>.csv into --out. Aggregation
collates the per-target summaries into results/tables/<name>_ablation.csv.

Usage:
  python 06_run_grid.py --dataset <proc.parquet> --fp <fp.npz> --emb <emb.npz> \
     --splits-dir data/processed/splits --split-tag bindingdb_IC50 \
     --schemes target_disjoint scaffold random --arms A_drop B_cast C_bounded D_tobit E_tobit_rankobs F_tobit_rankdec \
     --seeds 1 2 3 4 5 --out results/mainline --name bindingdb_IC50 [--workers 12]
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_arm.py")
PROJ = os.path.dirname(os.path.dirname(HERE))
PYBIN = os.path.join(PROJ, "env", "conda", "bin", "python")
EXTRA_ARGS = []   # extra train_arm CLI args (e.g. --clamp-logsig, --upsample-censored), set in main()


def run_job(spec):
    arm, scheme, seed, lam, margin, dataset, fp, emb, split_file, out, threads, device, gpu = spec
    tag = f"{arm}_{scheme}_seed{seed}"
    outdir = os.path.join(out, scheme)
    os.makedirs(outdir, exist_ok=True)
    res_json = os.path.join(outdir, f"result_{arm}_seed{seed}.json")
    if os.path.exists(res_json):
        return (tag, "cached")
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads))
    if device == "cuda" and gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [PYBIN, "-u", TRAIN, "--dataset", dataset, "--split", split_file,
           "--fp", fp, "--emb", emb, "--arm", arm, "--seed", str(seed),
           "--device", device, "--out", outdir]
    if lam is not None:
        cmd += ["--lambda-rank", str(lam)]
    if margin is not None:
        cmd += ["--margin", str(margin)]
    if EXTRA_ARGS:
        cmd += EXTRA_ARGS
    logf = os.path.join(outdir, f"log_{arm}_seed{seed}.log")
    with open(logf, "w") as lf:
        r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return (tag, "ok" if r.returncode == 0 else f"FAIL rc={r.returncode}")


def aggregate(out, name):
    rows = []
    for scheme in os.listdir(out):
        d = os.path.join(out, scheme)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("result_") and f.endswith(".json"):
                with open(os.path.join(d, f)) as fh:
                    res = json.load(fh)
                row = {"scheme": scheme, "arm": res["arm"], "seed": res["seed"],
                       "lambda": res.get("lambda_rank"), "n_test_targets": res.get("n_test_targets"),
                       "best_epoch": res.get("best_epoch")}
                for m, v in (res.get("summary") or {}).items():
                    if isinstance(v, dict):
                        row[f"{m}_mean"] = v.get("mean")
                        row[f"{m}_median"] = v.get("median")
                        row[f"{m}_ci_lo"] = v.get("ci_lo")
                        row[f"{m}_ci_hi"] = v.get("ci_hi")
                rows.append(row)
    df = pd.DataFrame(rows).sort_values(["scheme", "arm", "seed"])
    tdir = os.path.join(PROJ, "results", "tables")
    os.makedirs(tdir, exist_ok=True)
    outcsv = os.path.join(tdir, f"{name}_ablation.csv")
    df.to_csv(outcsv, index=False)
    print(f"[aggregate] {len(df)} runs -> {outcsv}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--fp", required=True)
    ap.add_argument("--emb", required=True); ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--split-tag", required=True)
    ap.add_argument("--schemes", nargs="+", default=["target_disjoint", "scaffold", "random"])
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--lambda-rank", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="comma GPU ids for round-robin when --device cuda")
    ap.add_argument("--clamp-logsig", type=float, default=None, help="pass-through: clamp Tobit log-sigma")
    ap.add_argument("--upsample-censored", type=float, default=None, help="pass-through: censored-pair upsample weight")
    ap.add_argument("--max-epochs", type=int, default=None, help="pass-through: cap epochs")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global EXTRA_ARGS
    if args.clamp_logsig is not None:
        EXTRA_ARGS += ["--clamp-logsig", str(args.clamp_logsig)]
    if args.upsample_censored is not None:
        EXTRA_ARGS += ["--upsample-censored", str(args.upsample_censored)]
    if args.max_epochs is not None:
        EXTRA_ARGS += ["--max-epochs", str(args.max_epochs)]
    gpus = [g for g in args.gpus.split(",") if g != ""] if args.device == "cuda" else []

    if not args.aggregate_only:
        specs = []
        i = 0
        for scheme in args.schemes:
            split_file = os.path.join(args.splits_dir, f"splits_{args.split_tag}_{scheme}.parquet")
            if not os.path.exists(split_file):
                print(f"[grid] WARN missing split {split_file}; skip scheme {scheme}")
                continue
            for arm in args.arms:
                for seed in args.seeds:
                    gpu = gpus[i % len(gpus)] if gpus else None
                    specs.append((arm, scheme, seed, args.lambda_rank, args.margin,
                                  args.dataset, args.fp, args.emb, split_file, args.out, args.threads,
                                  args.device, gpu))
                    i += 1
        print(f"[grid] {len(specs)} jobs, {args.workers} workers x {args.threads} threads, device={args.device} gpus={gpus}")
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_job, s) for s in specs]
            for fu in as_completed(futs):
                tag, status = fu.result(); done += 1
                print(f"[grid] ({done}/{len(specs)}) {tag}: {status}", flush=True)
    aggregate(args.out, args.name)


if __name__ == "__main__":
    main()
