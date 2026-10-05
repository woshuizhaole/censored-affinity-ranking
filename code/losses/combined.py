"""Arm dispatcher: total loss per experiment arm A-F.

Arms (all share the identical MLP backbone; only label handling differs):
  A_drop         : MSE on exact only         (dataset pre-filtered to exact)
  B_cast         : MSE, censored cast to =X  (dataset relabels censored->exact, censor all 0)
  C_bounded      : bounded one-sided MSE      (full censored dataset)
  D_tobit        : Tobit censored NLL         (full)
  E_tobit_rankobs: D + lambda * ranking(exact-exact pairs only)
  F_tobit_rankdec: D + lambda * ranking(decidable pairs incl. censored endpoints)

For A/B the upstream dataset already encodes the label choice (A: exact rows only;
B: censor set to 0). This function trusts the batch's censor field.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from losses.tobit import tobit_nll
from losses.bounded import bounded_mse
from losses.ranking import decidable_ranking_loss

ARMS = ["A_drop", "B_cast", "C_bounded", "D_tobit", "E_tobit_rankobs", "F_tobit_rankdec",
        # G-arms: decidable/exact ranking on the WINNING simple bases (bounded-MSE / cast) instead of
        # the weaker Tobit base — the decisive "does ranking win on a strong base" test (improvement.md #1).
        #   G_bounded_rankdec: bounded-MSE base + decidable ranking (incl. censored pairs)
        #   G_cast_rankdec   : cast-MSE base    + decidable ranking (base ignores censor; ranking uses real censor)
        #   G_bounded_rankobs: bounded-MSE base + exact-only ranking (control: isolates censored-pair value vs rankdec)
        "G_bounded_rankdec", "G_cast_rankdec", "G_bounded_rankobs"]


def compute_loss(arm, mu, log_sigma, y, censor, target_id,
                 lambda_rank: float = 0.3, margin: float = 1.0,
                 epoch: int = 999, rank_warmup_epochs: int = 5,
                 upsample_censored: float = 1.0):
    """Return (total_loss, stats_dict)."""
    stats = {}
    if arm in ("A_drop", "B_cast"):
        # standard regression; censor is all-0 for these datasets
        return F.mse_loss(mu, y), stats
    if arm == "C_bounded":
        return bounded_mse(mu, y, censor), stats
    if arm in ("G_bounded_rankdec", "G_cast_rankdec", "G_bounded_rankobs"):
        base = F.mse_loss(mu, y) if arm == "G_cast_rankdec" else bounded_mse(mu, y, censor)
        lam = 0.0 if epoch < rank_warmup_epochs else lambda_rank
        rank, rstats = decidable_ranking_loss(mu, y, censor, target_id, margin=margin,
                                              only_exact=(arm == "G_bounded_rankobs"),
                                              upsample_censored=upsample_censored)
        stats.update(rstats)
        stats["lambda_eff"] = lam; stats["base"] = float(base.detach()); stats["rank"] = float(rank.detach())
        return base + lam * rank, stats
    # D/E/F share the Tobit head
    base = tobit_nll(mu, log_sigma, y, censor)
    if arm == "D_tobit":
        return base, stats
    lam = 0.0 if epoch < rank_warmup_epochs else lambda_rank
    only_exact = (arm == "E_tobit_rankobs")
    rank, rstats = decidable_ranking_loss(mu, y, censor, target_id,
                                          margin=margin, only_exact=only_exact,
                                          upsample_censored=upsample_censored)
    stats.update(rstats)
    total = base + lam * rank
    stats["lambda_eff"] = lam
    stats["tobit"] = float(base.detach())
    stats["rank"] = float(rank.detach())
    return total, stats
