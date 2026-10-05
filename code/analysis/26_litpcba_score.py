#!/usr/bin/env python
"""improvement.md #4: LIT-PCBA external scoring — a NON-synthetic-decoy benchmark (experimentally
confirmed PubChem actives + real inactives) beyond DUD-E. Parses actives.smi/inactives.smi per target
(inactives subsampled, fixed seed), fetches target seqs (UniProt), 95% MMseqs2 leakage filter, scores
saved (arm x seed) models, per-target enrichment + one-sided paired Wilcoxon + effect size / bootstrap CI.
Arms scored default here (models saved); tuned-F/G scored separately after retrain-with-save."""
import os, sys, glob, subprocess, json
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from models.mlp import AffinityMLP  # noqa
from eval.enrichment import per_target_vs, summarize  # noqa
from scipy.stats import wilcoxon

PROJ = "/data/workplace/wyx/censored_affinity_20260707"
LIT = PROJ + "/data/external/litpcba"
ARMS = ["A_drop", "B_cast", "C_bounded", "D_tobit", "F_tobit_rankdec"]
SEEDS = [1, 2, 3, 4, 5]
ESM_CKPT = PROJ + "/data/raw/esm2_t12_35M_UR50D.pt"
MMSEQS = PROJ + "/env/conda/bin/mmseqs"
OUT = PROJ + "/results/litpcba"; os.makedirs(OUT, exist_ok=True)
N_INACT = 20000  # subsample inactives per target (fixed seed) — keeps all actives; same set across arms
RNG = np.random.default_rng(12345)

# --- targets: acc_map + uniprot fasta ---
acc = pd.read_csv(LIT + "/acc_map.tsv", sep="\t", names=["target", "accession"])
fa = {}
cur = None
for ln in open(LIT + "/uniprot_batch.fasta"):
    if ln.startswith(">"):
        cur = ln.split("|")[1]; fa[cur] = ""
    else:
        fa[cur] += ln.strip()
acc["seq"] = acc["accession"].map(fa)
print("[lit] targets:", len(acc), "seqs ok:", acc.seq.notna().sum())

# --- compounds: parse actives/inactives.smi per target, subsample inactives ---
rows = []
for _, r in acc.iterrows():
    tdir = LIT + "/" + r["target"]
    for lab, fn in [(1, "actives.smi"), (0, "inactives.smi")]:
        p = os.path.join(tdir, fn)
        if not os.path.exists(p):
            continue
        smis = [ln.split()[0] for ln in open(p) if ln.strip()]
        if lab == 0 and len(smis) > N_INACT:
            smis = list(np.array(smis)[RNG.choice(len(smis), N_INACT, replace=False)])
        for s in smis:
            rows.append((r["target"], s, lab))
comp = pd.DataFrame(rows, columns=["target", "smiles", "label"])
print("[lit] compounds:", len(comp), "actives:", int(comp.label.sum()),
      "per-target actives:", comp[comp.label == 1].groupby("target").size().to_dict())

# --- leakage filter: drop LIT targets >=95% identity to training targets ---
train_fa = PROJ + "/results/stage9_dude/train_targets.fasta"
wd = OUT + "/mmseqs"; os.makedirs(wd, exist_ok=True)
te_fa = wd + "/ext.fasta"
with open(te_fa, "w") as fh:
    for t, s in zip(acc["target"], acc["seq"]):
        s = "".join(c if c.isalpha() else "X" for c in str(s).upper())
        if len(s) >= 10:
            fh.write(">%s\n%s\n" % (t, s))
res = wd + "/res.m8"
subprocess.run([MMSEQS, "easy-search", te_fa, train_fa, res, wd + "/tmp",
                "--format-output", "query,target,pident"], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
m8 = pd.read_csv(res, sep="\t", names=["q", "t", "pident"])
maxid = m8.groupby("q")["pident"].max()
# ALL 15 LIT-PCBA targets are >=95% identical to a BindingDB training target (BindingDB is comprehensive).
# So — like DUD-E — there is NO novel-target subset. We KEEP all 15 and interpret LIT-PCBA as a
# COMPOUND-GENERALIZATION + REAL(non-synthetic)-INACTIVE discrimination test on known targets, NOT
# novel-target transfer. The homology per target is reported for full disclosure (honest caveat).
print("[lit] leakage (ALL homologous to train -> compound-generalization + real-inactive test, not novel-target):")
print("      max %%id to train per target:", {t: round(float(maxid.get(t, float("nan"))), 1) for t in acc["target"]})
kept = list(acc["target"])  # keep all

# --- fp ---
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
print("[lit] fp ok: %d/%d" % (ok.sum(), len(ok)))

# --- emb ---
import esm
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, alphabet = esm.pretrained.load_model_and_alphabet_local(ESM_CKPT)
model = model.to(dev).eval()
for p in model.parameters():
    p.requires_grad_(False)
bc = alphabet.get_batch_converter(); L = model.num_layers
seqs = {t: "".join(c if c.isalpha() else "X" for c in str(s).upper())[:1022] for t, s in zip(acc["target"], acc["seq"])}
emb = {}
ids = list(seqs)
for i in range(0, len(ids), 4):
    sub = ids[i:i + 4]
    _, _, toks = bc([(t, seqs[t]) for t in sub]); toks = toks.to(dev)
    with torch.no_grad():
        rep = model(toks, repr_layers=[L])["representations"][L]
    for bi, t in enumerate(sub):
        emb[t] = rep[bi, 1:len(seqs[t]) + 1].mean(0).float().cpu().numpy()
tvec = np.stack([emb[t] for t in comp["target"].values])
X = torch.as_tensor(np.concatenate([fp, tvec], axis=1))
print("[lit] features ready X=%s" % (tuple(X.shape),))

# --- score default arms ---
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
        pt = per_target_vs(c, "score", "label", "target", min_actives=10)
        pt["seed"] = seed; pertar[arm].append(pt)
agg = {}
for arm in ARMS:
    d = pd.concat(pertar[arm], ignore_index=True)
    agg[arm] = d.groupby("target")[[c for c in d.columns if c not in ("target", "seed")]].mean()
    agg[arm].to_csv(OUT + "/lit_%s_pertarget.csv" % arm)
    json.dump({"n_targets": len(agg[arm]), "summary": summarize(agg[arm].reset_index()) if len(agg[arm]) else {}},
              open(OUT + "/lit_%s_summary.json" % arm, "w"), indent=2, default=str)

# also score TUNED models (val-best G, tuned F) if retrained-with-save present
TUNED = {"G_tuned": "G_bounded_rankdec", "F_tuned": "F_tobit_rankdec"}
for label, arm in TUNED.items():
    mps = sorted(glob.glob(PROJ + "/results/models/tuned/model_%s_seed*.pt" % arm))
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
        p = per_target_vs(c, "score", "label", "target", min_actives=10)
        pts.append(p)
    d = pd.concat(pts, ignore_index=True)
    agg[label] = d.groupby("target")[[c for c in d.columns if c != "target"]].mean()
    agg[label].to_csv(OUT + "/lit_%s_pertarget.csv" % label)
    print("[lit] scored TUNED %s (%d seeds)" % (label, len(mps)))

common = sorted(set.intersection(*[set(agg[a].index) for a in ARMS]))
print("\n[lit] %d common targets (after leakage filter). PAIRED Wilcoxon + effect + bootstrap CI:" % len(common))
def bootci(d, n=5000):
    idx = RNG.integers(0, len(d), size=(n, len(d))); mu = d[idx].mean(1)
    return np.percentile(mu, 2.5), np.percentile(mu, 97.5)
print("%-12s %-9s %7s %7s %8s %18s %9s" % ("contrast", "metric", "hi", "lo", "delta", "95%CI", "wilcox_p"))
res_rows = []
contrasts = [("F_tobit_rankdec", "C_bounded"), ("F_tobit_rankdec", "B_cast"),
             ("D_tobit", "B_cast"), ("F_tobit_rankdec", "A_drop"), ("C_bounded", "A_drop")]
for tl in ("G_tuned", "F_tuned"):
    if tl in agg:
        contrasts += [(tl, "C_bounded"), (tl, "B_cast")]
for hi, lo in contrasts:
    if hi not in agg or lo not in agg:
        continue
    for m in ["EF1%", "BEDROC20", "RIE20", "AUC"]:
        if m not in agg[hi].columns:
            continue
        x = agg[hi].loc[common, m].values.astype(float); y = agg[lo].loc[common, m].values.astype(float)
        d = x - y; lo95, hi95 = bootci(d)
        try:
            _, p = wilcoxon(x, y, alternative="greater")
        except Exception:
            p = float("nan")
        tag = hi.split("_")[0] + "-" + lo.split("_")[0]
        print("%-12s %-9s %7.3f %7.3f %8.3f  [%7.3f,%7.3f] %9.4g %s"
              % (tag, m, x.mean(), y.mean(), d.mean(), lo95, hi95, p, "*" if p < 0.05 else ""))
        res_rows.append({"contrast": tag, "metric": m, "hi": x.mean(), "lo": y.mean(),
                         "delta": d.mean(), "ci_lo": lo95, "ci_hi": hi95, "p_1sided": p, "n": len(common)})
pd.DataFrame(res_rows).to_csv(OUT + "/litpcba_paired_effect.csv", index=False)
print("\nwrote ->", OUT)
