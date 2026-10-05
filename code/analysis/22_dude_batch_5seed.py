#!/usr/bin/env python
"""P0: DUD-E 5-seed external scoring + paired tests. Computes DUD-E ECFP4 + ESM-2 embeddings
ONCE (with 95% MMseqs2 leakage filter -> 86 targets), then scores all saved (arm x seed) models,
averages per-target metrics over seeds, and runs one-sided paired Wilcoxon for the key contrasts.
Upgrades the single-seed DUD-E result (F-B EF1% p=0.14, underpowered) to a 5-seed paired test."""
import os, sys, glob, subprocess, json
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from models.mlp import AffinityMLP  # noqa
from eval.enrichment import per_target_vs, summarize  # noqa
from scipy.stats import wilcoxon

PROJ = "/data/workplace/wyx/censored_affinity_20260707"
ARMS = ["A_drop", "B_cast", "C_bounded", "D_tobit", "F_tobit_rankdec"]
SEEDS = [1, 2, 3, 4, 5]
ESM_CKPT = PROJ + "/data/raw/esm2_t12_35M_UR50D.pt"
MMSEQS = PROJ + "/env/conda/bin/mmseqs"
OUT = PROJ + "/results/stage9_dude_5seed"; os.makedirs(OUT, exist_ok=True)

comp = pd.read_parquet(PROJ + "/data/external/dude/dude_compounds.parquet")
tgt = pd.read_parquet(PROJ + "/data/external/dude/dude_targets.parquet")
idc = "target"

# leakage filter: drop DUD-E targets >=95% identity to training targets
train_fa = PROJ + "/results/stage9_dude/train_targets.fasta"
kept = set(tgt[idc])
wd = OUT + "/mmseqs"; os.makedirs(wd, exist_ok=True)
te_fa = wd + "/ext.fasta"
with open(te_fa, "w") as fh:
    for t, s in zip(tgt[idc], tgt["seq"]):
        s = "".join(c if c.isalpha() else "X" for c in str(s).upper())
        if len(s) >= 10:
            fh.write(">%s\n%s\n" % (t, s))
res = wd + "/res.m8"
subprocess.run([MMSEQS, "easy-search", te_fa, train_fa, res, wd + "/tmp",
                "--format-output", "query,target,pident"], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
m8 = pd.read_csv(res, sep="\t", names=["q", "t", "pident"])
drop = set(m8.groupby("q")["pident"].max().pipe(lambda s: s[s >= 95.0]).index)
kept = kept - drop
comp = comp[comp[idc].isin(kept)].reset_index(drop=True)
tgt = tgt[tgt[idc].isin(kept)].reset_index(drop=True)
print("[5seed] kept %d targets (dropped %d), %d compounds" % (len(kept), len(drop), len(comp)))

# fp (once)
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, DataStructs
RDLogger.DisableLog("rdApp.*")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
fp = np.zeros((len(comp), 1024), np.float32); ok = np.zeros(len(comp), bool)
for i, s in enumerate(comp["smiles"].tolist()):
    m = Chem.MolFromSmiles(s)
    if m is None:
        continue
    a = np.zeros((1024,), np.uint8); DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), a)
    fp[i] = a; ok[i] = True
comp = comp[ok].reset_index(drop=True); fp = fp[ok]
# emb (once)
import esm
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, alphabet = esm.pretrained.load_model_and_alphabet_local(ESM_CKPT)
model = model.to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
bc = alphabet.get_batch_converter(); L = model.num_layers
seqs = {t: "".join(c if c.isalpha() else "X" for c in str(s).upper())[:1022] for t, s in zip(tgt[idc], tgt["seq"])}
emb = {}
ids = list(seqs)
for i in range(0, len(ids), 8):
    sub = ids[i:i + 8]
    _, _, toks = bc([(t, seqs[t]) for t in sub]); toks = toks.to(dev)
    with torch.no_grad():
        rep = model(toks, repr_layers=[L])["representations"][L]
    for bi, t in enumerate(sub):
        emb[t] = rep[bi, 1:len(seqs[t]) + 1].mean(0).float().cpu().numpy()
tvec = np.stack([emb[t] for t in comp[idc].values])
X = torch.as_tensor(np.concatenate([fp, tvec], axis=1))
print("[5seed] features ready: X=%s" % (tuple(X.shape),))

# score every (arm, seed) model, collect per-target metrics
pertar = {a: [] for a in ARMS}
for arm in ARMS:
    for seed in SEEDS:
        mp = PROJ + "/results/models/random/model_%s_seed%d.pt" % (arm, seed)
        if not os.path.exists(mp):
            print("  MISSING", mp); continue
        ck = torch.load(mp, map_location="cpu")
        net = AffinityMLP(in_dim=ck["in_dim"], hidden=ck["hidden"], dropout=ck["dropout"], two_head=ck["two_head"])
        net.load_state_dict(ck["state_dict"]); net.eval()
        sc = np.empty(len(X), np.float32)
        with torch.no_grad():
            for s in range(0, len(X), 16384):
                sc[s:s + 16384] = net(X[s:s + 16384])[0].numpy()
        c = comp.copy(); c["score"] = sc
        pt = per_target_vs(c.rename(columns={idc: "target"}), "score", "label", "target", min_actives=10)
        pt["seed"] = seed; pertar[arm].append(pt)
# mean over seeds per (arm, target)
agg = {}
for arm in ARMS:
    d = pd.concat(pertar[arm], ignore_index=True)
    agg[arm] = d.groupby("target")[[c for c in d.columns if c not in ("target", "seed")]].mean()
    summ = summarize(agg[arm].reset_index()) if len(agg[arm]) else {}
    json.dump({"n_targets": len(agg[arm]), "summary": summ},
              open(OUT + "/dude5_%s_summary.json" % arm, "w"), indent=2, default=str)
    agg[arm].to_csv(OUT + "/dude5_%s_pertarget.csv" % arm)

# also score TUNED models (val-best G, tuned F) if retrained-with-save present
import glob as _glob
TUNED = {"G_tuned": "G_bounded_rankdec", "F_tuned": "F_tobit_rankdec"}
for label, tarm in TUNED.items():
    mps = sorted(_glob.glob(PROJ + "/results/models/tuned/model_%s_seed*.pt" % tarm))
    if not mps:
        continue
    pts = []
    for mp in mps:
        ck = torch.load(mp, map_location="cpu")
        net = AffinityMLP(in_dim=ck["in_dim"], hidden=ck["hidden"], dropout=ck["dropout"], two_head=ck["two_head"])
        net.load_state_dict(ck["state_dict"]); net.eval()
        sc = np.empty(len(X), np.float32)
        with torch.no_grad():
            for s in range(0, len(X), 16384):
                sc[s:s + 16384] = net(X[s:s + 16384])[0].numpy()
        c = comp.copy(); c["score"] = sc
        pt = per_target_vs(c.rename(columns={idc: "target"}), "score", "label", "target", min_actives=10)
        pts.append(pt)
    d = pd.concat(pts, ignore_index=True)
    agg[label] = d.groupby("target")[[c for c in d.columns if c != "target"]].mean()
    agg[label].to_csv(OUT + "/dude5_%s_pertarget.csv" % label)
    print("[5seed] scored TUNED %s (%d seeds)" % (label, len(mps)))

_rng = np.random.default_rng(20260708)
def _bootci(dd, n=5000):
    ix = _rng.integers(0, len(dd), size=(n, len(dd))); mu = dd[ix].mean(1)
    return np.percentile(mu, 2.5), np.percentile(mu, 97.5)
common = sorted(set.intersection(*[set(agg[a].index) for a in ARMS]))
print("\n[5seed] %d common targets. PAIRED Wilcoxon + effect + 95%%CI (one-sided hi>lo, 5-seed mean):" % len(common))
print("%-12s %-9s %8s %8s %8s %18s %10s" % ("contrast", "metric", "hi", "lo", "delta", "95%CI", "wilcox_p"))
contrasts = [("F_tobit_rankdec", "C_bounded"), ("F_tobit_rankdec", "B_cast"),
             ("D_tobit", "B_cast"), ("F_tobit_rankdec", "A_drop")]
for tl in ("G_tuned", "F_tuned"):
    if tl in agg:
        contrasts += [(tl, "C_bounded"), (tl, "B_cast")]
_erows = []
for hi, lo in contrasts:
    if hi not in agg or lo not in agg:
        continue
    for m in ["EF1%", "BEDROC20", "RIE20", "AUC"]:
        if m not in agg[hi].columns:
            continue
        x = agg[hi].loc[common, m].values.astype(float); y = agg[lo].loc[common, m].values.astype(float)
        dd = x - y; clo, chi = _bootci(dd)
        try:
            _, p = wilcoxon(x, y, alternative="greater")
        except Exception:
            p = float("nan")
        tag = hi.replace("_tobit_rankdec", "").replace("_bounded", "").replace("_cast", "").replace("_drop", "") + "-" + lo.split("_")[0]
        print("%-12s %-9s %8.3f %8.3f %8.4f  [%7.4f,%7.4f] %10.4g %s" % (tag, m, x.mean(), y.mean(), dd.mean(), clo, chi, p, "*" if p < 0.05 else ""))
        _erows.append({"contrast": tag, "metric": m, "hi": x.mean(), "lo": y.mean(), "delta": dd.mean(), "ci_lo": clo, "ci_hi": chi, "p1": p, "n": len(common)})
        continue
pd.DataFrame(_erows).to_csv(OUT + "/dude5_paired_effect.csv", index=False)
print("\nwrote -> %s" % OUT)
