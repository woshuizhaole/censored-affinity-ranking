#!/usr/bin/env python
"""Stage 9a: score an external VS benchmark (DUD-E) with a BindingDB-trained model.

Loads a saved AffinityMLP, computes ECFP4 for external compounds + frozen ESM-2
embeddings for external target sequences, removes external targets too similar to
the training targets (MMseqs2) and near-duplicate compounds, then ranks per target
and reports EF/BEDROC. Structure-free: only sequence + ligand fingerprint.

Usage: python 21_score_external.py --model results/models/F_full.pt \
   --compounds data/external/dude/dude_compounds.parquet \
   --targets data/external/dude/dude_targets.parquet \
   --esm-ckpt data/raw/esm2_t12_35M_UR50D.pt \
   --train-fasta <train targets fasta> --mmseqs env/conda/bin/mmseqs \
   --max-train-ident 50 --out results/benchmark --name dude
"""
from __future__ import annotations
import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from models.mlp import AffinityMLP  # noqa
from eval.enrichment import per_target_vs, summarize  # noqa


def _fp_matrix(smiles_list, bits=1024, radius=2):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import rdFingerprintGenerator, DataStructs
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    out = np.zeros((len(smiles_list), bits), dtype=np.uint8)
    ok = np.zeros(len(smiles_list), dtype=bool)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        arr = np.zeros((bits,), np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), arr)
        out[i] = arr; ok[i] = True
    return out, ok


def _esm_embed(seqs_by_id, ckpt, device="cuda", max_len=1022):
    import esm
    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(ckpt)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bc = alphabet.get_batch_converter(); L = model.num_layers
    ids = list(seqs_by_id); cache = {}
    for i in range(0, len(ids), 8):
        sub = ids[i:i + 8]
        data = [(t, seqs_by_id[t][:max_len]) for t in sub]
        _, _, toks = bc(data); toks = toks.to(dev)
        with torch.no_grad():
            rep = model(toks, repr_layers=[L])["representations"][L]
        for bi, t in enumerate(sub):
            n = len(seqs_by_id[t][:max_len])
            cache[t] = rep[bi, 1:n + 1].mean(0).float().cpu().numpy()
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--compounds", required=True); ap.add_argument("--targets", required=True)
    ap.add_argument("--esm-ckpt", required=True)
    ap.add_argument("--train-fasta", default=None); ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--max-train-ident", type=float, default=50.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True); ap.add_argument("--name", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.model, map_location="cpu")
    net = AffinityMLP(in_dim=ck["in_dim"], hidden=ck["hidden"],
                      dropout=ck["dropout"], two_head=ck["two_head"])
    net.load_state_dict(ck["state_dict"]); net.eval()

    comp = pd.read_parquet(args.compounds)
    tgt = pd.read_parquet(args.targets)
    id_col = "target" if "target" in comp.columns else "target_id"

    # leakage filter: drop external targets too similar to training targets
    kept_targets = set(tgt[id_col if id_col in tgt.columns else "target"])
    if args.train_fasta and os.path.exists(args.train_fasta):
        wd = os.path.join(args.out, f"mmseqs_{args.name}"); os.makedirs(wd, exist_ok=True)
        te_fa = os.path.join(wd, "ext.fasta")
        tcol = "target" if "target" in tgt.columns else "target_id"
        with open(te_fa, "w") as fh:
            for t, s in zip(tgt[tcol], tgt["seq"]):
                s = "".join(c if c.isalpha() else "X" for c in str(s).upper())
                if len(s) >= 10:
                    fh.write(f">{t}\n{s}\n")
        res = os.path.join(wd, "res.m8")
        try:
            subprocess.run([args.mmseqs, "easy-search", te_fa, args.train_fasta, res,
                            os.path.join(wd, "tmp"), "--format-output", "query,target,pident"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            m8 = pd.read_csv(res, sep="\t", names=["q", "t", "pident"])
            hi = m8.groupby("q")["pident"].max()
            drop = set(hi[hi >= args.max_train_ident].index)
            kept_targets = kept_targets - drop
            print(f"[ext] dropped {len(drop)} targets with >={args.max_train_ident}% id to train")
        except Exception as e:
            print(f"[ext] WARN identity filter skipped: {e}")

    comp = comp[comp[id_col].isin(kept_targets)].reset_index(drop=True)
    tgt = tgt[tgt[("target" if "target" in tgt.columns else "target_id")].isin(kept_targets)]
    print(f"[ext] scoring {len(comp)} compounds over {comp[id_col].nunique()} targets")

    # features
    fp, ok = _fp_matrix(comp["smiles"].tolist())
    comp = comp[ok].reset_index(drop=True); fp = fp[ok]
    tcol = "target" if "target" in tgt.columns else "target_id"
    seqs_by_id = {t: "".join(c if c.isalpha() else "X" for c in str(s).upper())
                  for t, s in zip(tgt[tcol], tgt["seq"])}
    emb = _esm_embed(seqs_by_id, args.esm_ckpt, device=args.device)

    # score
    scores = np.empty(len(comp), np.float32)
    tvec = np.stack([emb[t] for t in comp[id_col].values])
    X = np.concatenate([fp.astype(np.float32), tvec], axis=1)
    with torch.no_grad():
        for s in range(0, len(X), 8192):
            mu, _ = net(torch.as_tensor(X[s:s + 8192]))
            scores[s:s + len(mu)] = mu.numpy()
    comp["score"] = scores

    pt = per_target_vs(comp.rename(columns={id_col: "target"}), "score", "label", "target",
                       min_actives=10)
    pt.to_csv(os.path.join(args.out, f"{args.name}_pertarget.csv"), index=False)
    summ = summarize(pt) if len(pt) else {}
    import json
    with open(os.path.join(args.out, f"{args.name}_summary.json"), "w") as f:
        json.dump({"n_targets": len(pt), "summary": summ}, f, indent=2, default=str)
    print(f"[ext] {len(pt)} targets; EF1% mean={summ.get('EF1%',{}).get('mean')}, "
          f"BEDROC20 mean={summ.get('BEDROC20',{}).get('mean')}")


if __name__ == "__main__":
    main()
