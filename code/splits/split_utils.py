"""Leakage-preventing splits: target-disjoint (MMseqs2), scaffold (RDKit Murcko),
random, and Tanimoto near-duplicate removal.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
from collections import defaultdict

import numpy as np


# ---------- target-disjoint via MMseqs2 ----------
def write_fasta(id2seq: dict, path: str):
    with open(path, "w") as f:
        for tid, seq in id2seq.items():
            f.write(f">{tid}\n{seq}\n")
    return path


def mmseqs_cluster(id2seq: dict, workdir: str, min_seq_id=0.3, cov=0.8,
                   mmseqs_bin="mmseqs"):
    """Cluster target sequences; return dict target_id -> cluster_rep."""
    os.makedirs(workdir, exist_ok=True)
    fasta = write_fasta(id2seq, os.path.join(workdir, "targets.fasta"))
    pref = os.path.join(workdir, "clusterRes")
    tmp = os.path.join(workdir, "tmp")
    cmd = [mmseqs_bin, "easy-cluster", fasta, pref, tmp,
           "--min-seq-id", str(min_seq_id), "-c", str(cov), "--cov-mode", "0"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mapping = {}
    with open(pref + "_cluster.tsv") as f:
        for line in f:
            rep, member = line.rstrip("\n").split("\t")
            mapping[member] = rep
    return mapping


def assign_folds_by_group(group_of_item: dict, fracs=(0.7, 0.1, 0.2), seed=42):
    """Assign whole groups (clusters) to train/val/test so items in a group
    never cross splits. Returns dict item -> split_name."""
    groups = defaultdict(list)
    for item, g in group_of_item.items():
        groups[g].append(item)
    gkeys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(gkeys)
    n = len(gkeys)
    n_tr = int(fracs[0] * n)
    n_va = int((fracs[0] + fracs[1]) * n)
    split_of = {}
    for i, gk in enumerate(gkeys):
        s = "train" if i < n_tr else ("valid" if i < n_va else "test")
        for it in groups[gk]:
            split_of[it] = s
    return split_of


# ---------- scaffold split ----------
def murcko_scaffold(smiles: str):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    try:
        return MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        m = Chem.MolFromSmiles(smiles)
        return None if m is None else ""


def scaffold_split(smiles_list, fracs=(0.8, 0.1, 0.1), seed=42):
    """Return dict smiles -> split, grouping by Bemis-Murcko scaffold
    (largest scaffolds to train, DeepChem-style deterministic)."""
    scaf = defaultdict(list)
    for s in smiles_list:
        sc = murcko_scaffold(s)
        scaf[sc if sc is not None else s].append(s)
    # deterministic: big scaffold sets first to train
    sets = sorted(scaf.values(), key=lambda x: (len(x), x[0]), reverse=True)
    n = len(smiles_list)
    n_tr, n_va = fracs[0] * n, (fracs[0] + fracs[1]) * n
    split_of = {}
    c = 0
    for grp in sets:
        s = "train" if c < n_tr else ("valid" if c < n_va else "test")
        for sm in grp:
            split_of[sm] = s
        c += len(grp)
    return split_of


# ---------- Tanimoto near-duplicate removal ----------
def tanimoto_near_dup_mask(query_smiles, ref_smiles, thr=0.9, bits=1024, radius=2):
    """Return boolean array over query_smiles: True if max Tanimoto to any ref >= thr."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, DataStructs
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)

    def fp(s):
        m = Chem.MolFromSmiles(s)
        return gen.GetFingerprint(m) if m is not None else None

    ref_fps = [f for f in (fp(s) for s in ref_smiles) if f is not None]
    out = np.zeros(len(query_smiles), dtype=bool)
    for i, s in enumerate(query_smiles):
        f = fp(s)
        if f is None or not ref_fps:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(f, ref_fps)
        if sims and max(sims) >= thr:
            out[i] = True
    return out
