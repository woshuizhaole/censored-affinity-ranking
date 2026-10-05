#!/usr/bin/env python
"""Stage 3a: frozen ESM-2 t12_35M mean-pooled target embeddings (cached once).

Loads reused local checkpoint (fair-esm). Embeds each UNIQUE target sequence,
mean-pools residues (excluding BOS/EOS), saves 480-dim vectors.

Output: data/features/target_emb_esm2_35M.npz  (key=target_id -> float32[480])
Usage: python 03a_esm_embed.py --targets <parquet...> --ckpt <esm .pt> --out <npz> [--device cuda] [--max-len 1022]
"""
from __future__ import annotations
import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import torch

# ESM-2 accepts the 20 standard AAs plus ambiguity codes X,B,U,Z,O.
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXBUZO")


def clean_protein_seq(seq: str) -> str:
    """Uppercase, strip whitespace, map any non-AA char (digits, punctuation from
    BindingDB annotations) to 'X'. Returns '' if unusable."""
    if not isinstance(seq, str):
        return ""
    s = re.sub(r"\s+", "", seq).upper()
    if not s:
        return ""
    return "".join(c if c in _VALID_AA else "X" for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True, help="parquet(s) with target_id,seq")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--toks-per-batch", type=int, default=8192)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import esm
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"[esm] loading local checkpoint {args.ckpt} on {device}")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.ckpt)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bc = alphabet.get_batch_converter()
    repr_layer = model.num_layers  # 12 for t12
    print(f"[esm] num_layers={model.num_layers}, embed_dim={model.embed_dim}")

    # unique sequences -> map to target_ids (sanitize non-AA chars to X)
    seq2ids = {}
    n_sanitized = 0
    n_skipped = 0
    for p in args.targets:
        df = pd.read_parquet(p)
        for tid, seq in zip(df["target_id"], df["seq"]):
            cleaned = clean_protein_seq(seq)
            if not isinstance(seq, str) or len(cleaned) < 10:
                n_skipped += 1
                continue
            if cleaned != re.sub(r"\s+", "", seq).upper():
                n_sanitized += 1
            s = cleaned[: args.max_len]
            seq2ids.setdefault(s, set()).add(tid)
    seqs = list(seq2ids.keys())
    print(f"[esm] {len(seqs)} unique sequences (<= {args.max_len} aa); "
          f"sanitized(non-AA->X)={n_sanitized} skipped(<10aa)={n_skipped}")

    # sort by length, greedy token-budget batching
    seqs.sort(key=len)
    cache = {}
    i = 0
    done = 0
    while i < len(seqs):
        batch = []
        maxlen = 0
        while i < len(seqs):
            L = len(seqs[i]) + 2
            if batch and (max(maxlen, L) * (len(batch) + 1) > args.toks_per_batch):
                break
            batch.append(seqs[i]); maxlen = max(maxlen, L); i += 1
        data = [(str(k), s) for k, s in enumerate(batch)]
        _, _, toks = bc(data)
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer]      # [B, L, D]
        for bi, s in enumerate(batch):
            L = len(s)
            vec = reps[bi, 1 : L + 1].mean(0).float().cpu().numpy()   # exclude BOS(0),EOS(L+1)
            for tid in seq2ids[s]:
                cache[tid] = vec
        done += len(batch)
        if done % 500 < len(batch):
            print(f"[esm] embedded {done}/{len(seqs)} sequences")

    np.savez(args.out, **{k: v.astype(np.float32) for k, v in cache.items()})
    dims = {v.shape for v in cache.values()}
    print(f"[esm] wrote {len(cache)} target vectors dims={dims} -> {args.out}")


if __name__ == "__main__":
    main()
