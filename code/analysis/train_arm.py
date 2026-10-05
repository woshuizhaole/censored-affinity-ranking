#!/usr/bin/env python
"""Core training + VS-evaluation driver for one experiment arm (A-F).

Loads a processed censored dataset + precomputed features + a split assignment,
trains the shared MLP with the arm's loss (early-stop on val BEDROC), evaluates
per-target VS early enrichment on the test split. Consistent binary eval labels
across arms (derived from y_pscale + censor), so differences are attributable to
label handling only.

Feature assembly is memory-light: per-row (compound_idx, target_idx, y, censor),
with fp table (uint8) and esm table (float32) looked up per batch.

Usage (real):
  python train_arm.py --dataset <processed.parquet> --split <split.parquet> \
      --fp <ligand_fp.npz> --emb <target_emb.npz> --arm F_tobit_rankdec \
      --seed 1 --out <results_dir>
Usage (synthetic plumbing smoke): python train_arm.py --synthetic --arm F_tobit_rankdec
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # code/
from utils.seeds import set_seed, run_metadata
from models.mlp import AffinityMLP
from losses.combined import compute_loss, ARMS
from eval.enrichment import per_target_vs, summarize


# ---------------- data assembly ----------------
def binarize_eval(y, censor, active_thr=6.0, inactive_thr=5.0):
    """Consistent VS eval labels from (y_pscale, censor). Returns int label array
    with 1=active, 0=inactive, -1=exclude (grey/ambiguous)."""
    y = np.asarray(y, float); c = np.asarray(censor, int)
    lab = np.full(len(y), -1, dtype=int)
    active = ((c == 0) & (y >= active_thr)) | ((c == 1) & (y >= active_thr))
    inactive = ((c == 0) & (y <= inactive_thr)) | ((c == -1) & (y <= inactive_thr))
    lab[active] = 1
    lab[inactive] = 0
    return lab


class ArmData:
    """Holds tensors for one split; arm-specific label handling applied to TRAIN."""
    def __init__(self, df, fp_tbl, emb_tbl, comp_index, targ_index, arm):
        comp_idx = df["canonical_smiles"].map(comp_index).values
        targ_idx = df["target_id"].map(targ_index).values
        keep = (~pd.isna(comp_idx)) & (~pd.isna(targ_idx))
        df = df[keep].reset_index(drop=True)
        self.comp = torch.as_tensor(comp_idx[keep].astype(np.int64))
        self.targ = torch.as_tensor(targ_idx[keep].astype(np.int64))
        self.y = torch.as_tensor(df["y_pscale"].values.astype(np.float32))
        censor = df["censor"].values.astype(np.int64)
        tid = df["target_id"].values          # actual target_id strings (aligned)
        if arm == "A_drop":
            m = censor == 0
            self.comp, self.targ, self.y = self.comp[m], self.targ[m], self.y[m]
            censor = censor[m]; tid = tid[m]
        elif arm == "B_cast":
            censor = np.zeros_like(censor)
        self.censor = torch.as_tensor(censor)
        self.target_ids = tid                 # for per-target eval grouping by real id
        self.fp_tbl = fp_tbl
        self.emb_tbl = emb_tbl
        self.df = df

    def features(self, idx):
        fp = self.fp_tbl[self.comp[idx]].float()
        emb = self.emb_tbl[self.targ[idx]]
        return torch.cat([fp, emb], dim=1)

    def __len__(self):
        return len(self.y)


def target_grouped_batches(targ, batch_size, seed, max_per_target=256):
    """Yield index tensors, grouping rows by target so within-target pairs exist.
    Caps each target's contribution per batch at max_per_target to bound the
    O(g^2) decidable-pair cost for ranking arms (large targets are subsampled
    each epoch; different subsample per epoch via seed -> full coverage over epochs)."""
    g = torch.Generator().manual_seed(seed)
    by_t = {}
    for i, t in enumerate(targ.tolist()):
        by_t.setdefault(t, []).append(i)
    tids = list(by_t.keys())
    order = torch.randperm(len(tids), generator=g).tolist()
    batch, cur = [], 0
    for oi in order:
        rows = by_t[tids[oi]]
        if len(rows) > max_per_target:
            sel = torch.randperm(len(rows), generator=g)[:max_per_target].tolist()
            rows = [rows[k] for k in sel]
        batch.extend(rows); cur += len(rows)
        if cur >= batch_size:
            yield torch.as_tensor(batch); batch, cur = [], 0
    if batch:
        yield torch.as_tensor(batch)


# ---------------- train / eval ----------------
def evaluate(model, data, device, active_thr, inactive_thr, cfg_eval, decoy_fp=None, max_targets=None):
    """Per-target VS enrichment with a realistic library. Because BindingDB compound
    sets are binder-heavy (median ~77% active rate), a measured-only library saturates
    EF. We build a realistic library per target = capped measured actives (label 1) +
    measured inactives + a shared pool of random decoys (label 0, assumed non-binding),
    bringing the active rate to ~1-5% so EF/BEDROC discriminate. (See WORKLOG deviation.)"""
    model.eval()
    n = len(data)
    preds = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, 8192):
            idx = torch.arange(s, min(s + 8192, n))
            mu, _ = model(data.features(idx).to(device))
            preds[s:s + len(idx)] = mu.cpu().numpy()
    lab = binarize_eval(data.y.numpy(), data.censor.numpy(), active_thr, inactive_thr)
    meas = pd.DataFrame({"score": preds, "label": lab,
                         "target": data.target_ids, "embidx": data.targ.numpy()})
    meas = meas[meas["label"] >= 0]
    if max_targets and meas["target"].nunique() > max_targets:
        keep_t = set(pd.Series(meas["target"].unique()).sample(max_targets, random_state=0))
        meas = meas[meas["target"].isin(keep_t)]
    cap = int(cfg_eval.get("max_actives_eval", 300))
    parts = []
    for _, g in meas.groupby("target", sort=False):
        act = g[g["label"] == 1]
        if len(act) > cap:
            act = act.sample(cap, random_state=0)
        parts.append(pd.concat([act, g[g["label"] == 0]]))
    meas = pd.concat(parts, ignore_index=True) if parts else meas
    frames = [meas[["score", "label", "target"]]]
    if decoy_fp is not None and len(meas):
        emb_tbl = data.emb_tbl
        dfp = decoy_fp.float()
        K = dfp.shape[0]
        t2e = meas.drop_duplicates("target").set_index("target")["embidx"]
        tids = t2e.index.to_numpy()
        embs = emb_tbl[torch.as_tensor(t2e.to_numpy().astype(np.int64))]  # [T,480]
        dsc = np.empty((len(tids), K), np.float32)
        CH = 32  # targets per chunk (chunk*K rows per forward)
        with torch.no_grad():
            for i in range(0, len(tids), CH):
                te = embs[i:i + CH]; c = te.shape[0]
                x = torch.cat([dfp.unsqueeze(0).expand(c, -1, -1),
                               te.unsqueeze(1).expand(-1, K, -1)], dim=2).reshape(c * K, -1)
                mu, _ = model(x.to(device))
                dsc[i:i + c] = mu.cpu().numpy().reshape(c, K)
        frames.append(pd.DataFrame({"score": dsc.ravel(), "label": 0,
                                    "target": np.repeat(tids, K)}))
    edf = pd.concat(frames, ignore_index=True)
    pt = per_target_vs(edf, "score", "label", "target",
                       min_actives=cfg_eval.get("min_actives_per_target", 10),
                       ef_fractions=tuple(cfg_eval.get("ef_fractions", (0.005, 0.01, 0.05))),
                       bedroc_alphas=tuple(cfg_eval.get("bedroc_alphas", (20.0, 80.5, 160.9))))
    return pt


def train_and_eval(arm, train, valid, test, device, cfg, seed, decoy_fp=None):
    set_seed(seed)
    in_dim = train.fp_tbl.shape[1] + train.emb_tbl.shape[1]
    model = AffinityMLP(in_dim=in_dim, hidden=tuple(cfg["model"]["hidden"]),
                        dropout=cfg["model"]["dropout"], two_head=cfg["model"]["two_head"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"].get("weight_decay", 0.0))
    tcfg, lcfg, ecfg = cfg["train"], cfg["loss"], cfg["eval"]
    best_metric, best_state, best_epoch = -1e9, None, -1
    metric_key = f"BEDROC{tcfg.get('bedroc_alpha',20.0):g}"
    patience = tcfg["patience"]; bad = 0
    history = []
    for epoch in range(tcfg["max_epochs"]):
        model.train()
        for idx in target_grouped_batches(train.targ, tcfg["batch_size"], seed * 1000 + epoch):
            x = train.features(idx).to(device)
            mu, logsig = model(x)
            clamp = lcfg.get("clamp_logsig", None)
            if clamp is not None and logsig is not None:
                logsig = torch.clamp(logsig, -float(clamp), float(clamp))
            loss, _ = compute_loss(arm, mu, logsig, train.y[idx].to(device),
                                   train.censor[idx].to(device), train.targ[idx].to(device),
                                   lambda_rank=lcfg["lambda_rank"], margin=lcfg["margin"],
                                   epoch=epoch, rank_warmup_epochs=lcfg.get("rank_warmup_epochs", 5),
                                   upsample_censored=lcfg.get("upsample_censored", 1.0))
            opt.zero_grad(); loss.backward(); opt.step()
        pt = evaluate(model, valid, device, cfg["data"]["active_pAff"], cfg["data"]["inactive_pAff"], ecfg,
                      decoy_fp, max_targets=int(ecfg.get("valid_max_targets", 200)))
        vmetric = float(pt[metric_key].mean()) if len(pt) and metric_key in pt else -1e9
        vef = float(pt["EF1%"].mean()) if len(pt) and "EF1%" in pt else float("nan")
        vauc = float(pt["AUC"].mean()) if len(pt) and "AUC" in pt else float("nan")
        history.append({"epoch": epoch, "val_" + metric_key: vmetric, "val_targets": len(pt)})
        print(f"[ep {epoch}] val {metric_key}={vmetric:.4f} EF1%={vef:.3f} AUC={vauc:.4f} (ntgt={len(pt)})", flush=True)
        if vmetric > best_metric:
            best_metric, best_epoch, bad = vmetric, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_pt = evaluate(model, test, device, cfg["data"]["active_pAff"], cfg["data"]["inactive_pAff"], ecfg, decoy_fp)
    return model, test_pt, {"best_epoch": best_epoch, "best_val_" + metric_key: best_metric,
                            "history": history}


# ---------------- config + IO ----------------
def load_cfg(path=None):
    from omegaconf import OmegaConf
    default = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "configs", "base.yaml")
    return OmegaConf.to_container(OmegaConf.load(path or default), resolve=True)


def make_synthetic(cfg, n_targets=40, n_comp=800, seed=0):
    rng = np.random.default_rng(seed)
    fp_tbl = torch.as_tensor((rng.random((n_comp, cfg["model"]["fp_bits"])) < 0.1).astype(np.uint8))
    emb_tbl = torch.as_tensor(rng.standard_normal((n_targets, cfg["model"]["esm_dim"])).astype(np.float32))
    # latent target/compound effects define a true pAff; add censoring
    tw = rng.standard_normal((n_targets, 4)); cw = rng.standard_normal((n_comp, 4))
    rows = []
    for _ in range(20000):
        c = rng.integers(n_comp); t = rng.integers(n_targets)
        true = 6.0 + (cw[c] @ tw[t]) * 0.5 + rng.standard_normal() * 0.3
        r = rng.random()
        if true > 7 and r < 0.15:
            y, cen = 7.0, 1        # right-censored strong
        elif true < 5 and r < 0.25:
            y, cen = 5.0, -1       # left-censored weak
        else:
            y, cen = true, 0
        rows.append((f"C{c}", f"T{t}", float(np.clip(y, 0, 13)), int(cen)))
    df = pd.DataFrame(rows, columns=["canonical_smiles", "target_id", "y_pscale", "censor"])
    comp_index = {f"C{i}": i for i in range(n_comp)}
    targ_index = {f"T{i}": i for i in range(n_targets)}
    # split by target (cold-target), fractional 0.7/0.1/0.2
    tt = np.array([f"T{i}" for i in range(n_targets)])
    rng.shuffle(tt)
    n_tr = max(1, int(0.7 * n_targets)); n_va = max(n_tr + 1, int(0.8 * n_targets))
    split = {}
    for i, t in enumerate(tt):
        split[t] = "train" if i < n_tr else ("valid" if i < n_va else "test")
    df["split"] = df["target_id"].map(split)
    return df, fp_tbl, emb_tbl, comp_index, targ_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset"); ap.add_argument("--split")
    ap.add_argument("--fp"); ap.add_argument("--emb")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lambda-rank", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--save-model", default=None, help="path to save best model .pt (for external scoring)")
    ap.add_argument("--active-thr", type=float, default=None, help="override eval active pAff threshold")
    ap.add_argument("--inactive-thr", type=float, default=None, help="override eval inactive pAff threshold")
    ap.add_argument("--n-decoys", type=int, default=None, help="random decoys per target in VS eval (0=off)")
    ap.add_argument("--upsample-censored", type=float, default=None,
                    help="weight on censored-involving ranking pairs (implements upsample_censored_pairs; 1=off)")
    ap.add_argument("--clamp-logsig", type=float, default=None,
                    help="clamp Tobit log-sigma head to [-c,c] (Tobit-base-fix; None=off)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.lambda_rank is not None:
        cfg["loss"]["lambda_rank"] = args.lambda_rank
    if args.margin is not None:
        cfg["loss"]["margin"] = args.margin
    if args.upsample_censored is not None:
        cfg["loss"]["upsample_censored"] = args.upsample_censored
    if args.clamp_logsig is not None:
        cfg["loss"]["clamp_logsig"] = args.clamp_logsig
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.active_thr is not None:
        cfg["data"]["active_pAff"] = args.active_thr
    if args.inactive_thr is not None:
        cfg["data"]["inactive_pAff"] = args.inactive_thr
    device = args.device
    set_seed(args.seed)

    if args.synthetic:
        cfg["train"]["max_epochs"] = args.max_epochs or 8
        df, fp_tbl, emb_tbl, comp_index, targ_index = make_synthetic(cfg, seed=args.seed)
    else:
        df = pd.read_parquet(args.dataset).reset_index(drop=True)
        df["row_id"] = np.arange(len(df))
        sp = pd.read_parquet(args.split)          # columns: row_id, split
        df = df.merge(sp, on="row_id", how="inner")
        fp_npz = np.load(args.fp, allow_pickle=True)
        emb_npz = np.load(args.emb, allow_pickle=True)
        comp_index = {s: i for i, s in enumerate(fp_npz["smiles"])}
        fp_tbl = torch.as_tensor(fp_npz["fp"])
        emb_keys = list(emb_npz.keys())
        targ_index = {k: i for i, k in enumerate(emb_keys)}
        emb_tbl = torch.as_tensor(np.stack([emb_npz[k] for k in emb_keys]).astype(np.float32))

    tr = ArmData(df[df.split == "train"], fp_tbl, emb_tbl, comp_index, targ_index, args.arm)
    va = ArmData(df[df.split == "valid"], fp_tbl, emb_tbl, comp_index, targ_index, "eval")
    te = ArmData(df[df.split == "test"], fp_tbl, emb_tbl, comp_index, targ_index, "eval")
    print(f"[data] arm={args.arm} train={len(tr)} valid={len(va)} test={len(te)}")

    # shared random decoy pool (FIXED seed across arms/seeds -> fair VS comparison)
    n_decoys = args.n_decoys if args.n_decoys is not None else int(cfg["eval"].get("n_decoys", 2000))
    decoy_fp = None
    if n_decoys > 0:
        n_decoys = min(n_decoys, fp_tbl.shape[0])
        didx = np.random.default_rng(12345).choice(fp_tbl.shape[0], size=n_decoys, replace=False)
        decoy_fp = fp_tbl[torch.as_tensor(didx)]
        print(f"[data] VS eval with {n_decoys} shared random decoys per target")

    t0 = time.time()
    model, test_pt, info = train_and_eval(args.arm, tr, va, te, device, cfg, args.seed, decoy_fp)
    summ = summarize(test_pt) if len(test_pt) else {}
    result = {"arm": args.arm, "seed": args.seed, "lambda_rank": cfg["loss"]["lambda_rank"],
              "margin": cfg["loss"]["margin"], "n_train": len(tr), "n_test": len(te),
              "best_epoch": info["best_epoch"], "n_test_targets": len(test_pt),
              "summary": summ, "elapsed_s": round(time.time() - t0, 1),
              "meta": run_metadata(cfg)}
    print(json.dumps({k: v for k, v in result.items() if k != "meta"}, indent=2, default=str))
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "in_dim": tr.fp_tbl.shape[1] + tr.emb_tbl.shape[1],
                    "hidden": list(cfg["model"]["hidden"]),
                    "dropout": cfg["model"]["dropout"], "two_head": cfg["model"]["two_head"],
                    "arm": args.arm, "seed": args.seed}, args.save_model)
        print(f"[save] model -> {args.save_model}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        tag = f"{args.arm}_seed{args.seed}"
        test_pt.to_csv(os.path.join(args.out, f"pertarget_{tag}.csv"), index=False)
        with open(os.path.join(args.out, f"result_{tag}.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    main()
