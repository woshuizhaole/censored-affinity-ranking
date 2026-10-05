"""Bounded/censored MSE (arm C) — SparseChem/chemprop-style one-sided squared hinge.

No likelihood; penalize only when the prediction falls on the wrong side of a
censored bound. Censor convention (p-scale): -1 left (true<y), +1 right (true>y).
  exact(0):  (mu-y)^2
  left(-1):  relu(mu-y)^2   (penalize only if predicted too high)
  right(+1): relu(y-mu)^2   (penalize only if predicted too low)
"""
from __future__ import annotations
import torch


def bounded_mse(mu, y, censor, reduction: str = "mean"):
    d = mu - y
    loss_exact = d * d
    loss_left = torch.relu(d) ** 2      # true<y: penalize mu>y
    loss_right = torch.relu(-d) ** 2    # true>y: penalize mu<y
    loss = torch.where(censor == 0, loss_exact,
                       torch.where(censor < 0, loss_left, loss_right))
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
