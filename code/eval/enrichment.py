"""Virtual-screening early-enrichment metrics (per-target), via rdkit.ML.Scoring.

Convention: higher predicted score = more likely active. We sort descending
(best first, low index = better) as RDKit expects. `col=0` holds the binary label.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment, CalcAUC, CalcRIE


def vs_metrics(y_true, y_score, ef_fractions=(0.005, 0.01, 0.05),
               bedroc_alphas=(20.0, 80.5, 160.9)):
    """Compute EF@fracs, BEDROC@alphas, AUC, RIE for one target's ranked list."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")   # stable, descending
    labels = y_true[order]
    scores = [[int(v)] for v in labels]
    out = {}
    ef = CalcEnrichment(scores, 0, list(ef_fractions))
    for frac, v in zip(ef_fractions, ef):
        out[f"EF{frac*100:g}%"] = float(v)
    for a in bedroc_alphas:
        out[f"BEDROC{a:g}"] = float(CalcBEDROC(scores, 0, a))
    out["AUC"] = float(CalcAUC(scores, 0))
    out["RIE20"] = float(CalcRIE(scores, 0, 20.0))
    out["n"] = int(len(labels))
    out["n_active"] = int(labels.sum())
    return out


def per_target_vs(df, score_col, label_col, target_col,
                  min_actives=10, min_inactives=1, **kw):
    """Per-target VS metrics table. df rows: predicted score, binary label, target."""
    rows = []
    for t, g in df.groupby(target_col):
        na = int(g[label_col].sum())
        ni = int((g[label_col] == 0).sum())
        if na < min_actives or ni < min_inactives:
            continue
        m = vs_metrics(g[label_col].values, g[score_col].values, **kw)
        m[target_col] = t
        rows.append(m)
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0, agg=np.mean):
    """Bootstrap CI for an aggregate (default mean) over per-target values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([agg(rng.choice(values, size=len(values), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(agg(values)), float(lo), float(hi)


def summarize(pt_df, metric_cols=None):
    """Aggregate a per-target metrics table across targets (mean/median + CI)."""
    if metric_cols is None:
        metric_cols = [c for c in pt_df.columns
                       if c.startswith(("EF", "BEDROC", "AUC", "RIE"))]
    out = {}
    for c in metric_cols:
        mean, lo, hi = bootstrap_ci(pt_df[c].values)
        out[c] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                  "median": float(np.nanmedian(pt_df[c].values))}
    out["_n_targets"] = int(len(pt_df))
    return out
