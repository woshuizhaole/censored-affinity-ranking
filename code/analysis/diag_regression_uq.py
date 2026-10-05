#!/usr/bin/env python
"""D4: regression-accuracy + uncertainty axis. Do the Tobit-family arms (D/F) win on their
THEORETICAL home-turf metrics (MSE/MAE on exact rows, censored-Gaussian NLL, calibration)
even though they lose on top-1% VS enrichment? Decides alternative narrative #2
('Tobit helps regression/calibration but that gain does not transfer to early enrichment')."""
import os, sys
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from models.mlp import AffinityMLP  # noqa

PROJ = "/data/workplace/wyx/censored_affinity_20260707"
df = pd.read_parquet(PROJ + "/data/processed/bindingdb_IC50_censored.parquet").reset_index(drop=True)
df["row_id"] = np.arange(len(df))
df = df.merge(pd.read_parquet(PROJ + "/data/processed/splits/splits_bindingdb_IC50_random.parquet"), on="row_id", how="left")
te = df[df.split == "test"].reset_index(drop=True)
fp = np.load(PROJ + "/data/features/ligand_fp.npz", allow_pickle=True)
emb = np.load(PROJ + "/data/features/target_emb_esm2_35M.npz", allow_pickle=True)
ci = {s: i for i, s in enumerate(fp["smiles"])}; fptbl = torch.as_tensor(fp["fp"].astype(np.float32))
ek = list(emb.files); ti = {k: i for i, k in enumerate(ek)}
embtbl = torch.as_tensor(np.stack([emb[k] for k in ek]).astype(np.float32))
cidx = te["canonical_smiles"].map(ci).values; tidx = te["target_id"].map(ti).values
mask = ~(pd.isna(cidx) | pd.isna(tidx)); te = te[mask].reset_index(drop=True)
cidx = cidx[mask].astype(int); tidx = tidx[mask].astype(int)
X = torch.cat([fptbl[cidx], embtbl[tidx]], dim=1)
y = te["y_pscale"].values.astype(np.float32); cen = te["censor"].values.astype(int)
exact = cen == 0

from torch.special import log_ndtr
def tobit_nll(mu, sig, yv, cv):
    z = (yv - mu) / sig
    nll = np.zeros_like(mu)
    e = cv == 0; nll[e] = 0.5 * ((z[e]) ** 2) + np.log(sig[e]) + 0.5 * np.log(2 * np.pi)
    left = cv == -1; nll[left] = -log_ndtr(torch.as_tensor(z[left])).numpy()   # P(Y<=y)
    right = cv == 1; nll[right] = -log_ndtr(torch.as_tensor(-z[right])).numpy()  # P(Y>=y)
    return nll

print("%-16s  MSE_exact  MAE_exact  NLL_all(cens-Gauss)  mean_sigma" % "arm")
for arm in ["A_drop", "B_cast", "C_bounded", "D_tobit", "F_tobit_rankdec"]:
    ck = torch.load(PROJ + "/results/models/random/model_%s_seed1.pt" % arm, map_location="cpu")
    net = AffinityMLP(in_dim=ck["in_dim"], hidden=ck["hidden"], dropout=ck["dropout"], two_head=ck["two_head"])
    net.load_state_dict(ck["state_dict"]); net.eval()
    mus = []; sigs = []
    with torch.no_grad():
        for s in range(0, len(X), 20000):
            mu, logsig = net(X[s:s + 20000])
            mus.append(mu.numpy())
            sigs.append((torch.nn.functional.softplus(logsig) + 1e-4).numpy() if logsig is not None else np.ones(len(mu)))
    mu = np.concatenate(mus); sig = np.concatenate(sigs)
    mse = float(np.mean((mu[exact] - y[exact]) ** 2)); mae = float(np.mean(np.abs(mu[exact] - y[exact])))
    nll = float(np.mean(tobit_nll(mu, sig, y, cen)))
    print("%-16s  %8.4f  %8.4f  %14.4f  %10.4f" % (arm, mse, mae, nll, float(np.mean(sig))))
print("\n解读: 若D/F的MSE_exact或NLL明显低于B/C=>Tobit在回归/似然主场有效但不传导到top-1%排序(备选叙事2成立);若不低=>进一步坐实null")
