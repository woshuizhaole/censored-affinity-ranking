"""Censoring-decidable pairwise ranking hinge (within-target).

A pair (i,j) is DECIDABLE with i>j iff lo_i > hi_j, where the true-value
feasible interval is:
  exact(0):  [y, y]
  left(-1):  (-inf, y]
  right(+1): [y, +inf)
i.e. lo = y if censor>=0 else -inf ;  hi = y if censor<=0 else +inf.

This is the survival-analysis comparable-pairs rule (drops indeterminate pairs).
Loss = mean over decidable pairs of relu(margin - (mu_i - mu_j)).

`only_exact=True` restricts to exact-exact pairs (ablation arm E), isolating the
contribution of censored endpoints participating in ranking (arm F vs E).
"""
from __future__ import annotations
import torch

NEG_INF = float("-inf")
POS_INF = float("inf")


def _lo_hi(y, censor):
    lo = torch.where(censor >= 0, y, torch.full_like(y, NEG_INF))
    hi = torch.where(censor <= 0, y, torch.full_like(y, POS_INF))
    return lo, hi


def decidable_pairs_in_group(y, censor):
    """Return (i_idx, j_idx) local indices of decidable pairs with i>j (stronger)."""
    lo, hi = _lo_hi(y, censor)
    # decidable_{i>j} : lo_i > hi_j
    dec = lo.unsqueeze(1) > hi.unsqueeze(0)   # [g,g], dec[i,j] True => i>j certain
    idx = dec.nonzero(as_tuple=False)
    return idx[:, 0], idx[:, 1]


def decidable_ranking_loss(mu, y, censor, target_id, margin: float = 1.0,
                           only_exact: bool = False, upsample_censored: float = 1.0):
    """Within-target decidable-pair hinge. Returns (loss, stats dict).

    mu,y,censor : [N] tensors. target_id : [N] long tensor (group key).
    upsample_censored: weight (>1) on pairs with >=1 censored endpoint, so the
      censoring-specific ranking signal is not diluted by many trivial exact-exact
      pairs (implements the previously-dead `upsample_censored_pairs` config flag;
      =1.0 reproduces the original unweighted mean exactly).
    stats: n_pairs, frac_censored_involved (>=1 endpoint censored).
    """
    total = mu.new_zeros(())
    wsum = mu.new_zeros(())
    n_pairs = 0
    n_cens_pairs = 0
    uniq = torch.unique(target_id)
    for t in uniq:
        m = (target_id == t)
        if m.sum() < 2:
            continue
        gi = m.nonzero(as_tuple=False).squeeze(1)
        yg, cg, mug = y[gi], censor[gi], mu[gi]
        ii, jj = decidable_pairs_in_group(yg, cg)
        if ii.numel() == 0:
            continue
        if only_exact:
            keep = (cg[ii] == 0) & (cg[jj] == 0)
            ii, jj = ii[keep], jj[keep]
            if ii.numel() == 0:
                continue
        diff = mug[ii] - mug[jj]           # want > margin
        hinge = torch.relu(margin - diff)
        cens_involved = (cg[ii] != 0) | (cg[jj] != 0)
        if upsample_censored != 1.0:
            w = torch.where(cens_involved, mu.new_full((), float(upsample_censored)),
                            mu.new_ones(()))
            total = total + (w * hinge).sum()
            wsum = wsum + w.sum()
        else:
            total = total + hinge.sum()
            wsum = wsum + hinge.numel()
        n_pairs += ii.numel()
        n_cens_pairs += int(cens_involved.sum().item())
    if n_pairs == 0:
        return mu.new_zeros(()), {"n_pairs": 0, "frac_censored_involved": 0.0}
    loss = total / wsum
    return loss, {"n_pairs": n_pairs,
                  "frac_censored_involved": n_cens_pairs / n_pairs}
