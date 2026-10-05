"""Tobit (censored Gaussian) NLL — numerically stable via torch.special.log_ndtr.

Censor code convention (p-scale, see utils.censoring):
  c=0  exact        : gaussian NLL
  c=-1 left  (true value < y) : -log Phi(z)      = -log_ndtr(z)
  c=+1 right (true value > y) : -log(1 - Phi(z)) = -log_ndtr(-z)
with z = (y - mu)/sigma, sigma = softplus(log_sigma_raw) + floor.

This replaces uq4dd's unstable -log(1 - cdf + eps); log_ndtr is stable at |z|~40.
"""
from __future__ import annotations
import math
import torch
from torch.special import log_ndtr

LOG2PI = math.log(2.0 * math.pi)


def sigma_from_raw(log_sigma_raw: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    return torch.nn.functional.softplus(log_sigma_raw) + floor


def tobit_nll(mu, log_sigma_raw, y, censor, sigma_floor: float = 1e-4,
              z_clamp: float = 38.0, reduction: str = "mean"):
    """Per-sample censored Gaussian NLL. `censor` in {-1,0,+1} (long/float tensor)."""
    sigma = sigma_from_raw(log_sigma_raw, sigma_floor)
    z = ((y - mu) / sigma).clamp(-z_clamp, z_clamp)
    nll_exact = 0.5 * z * z + torch.log(sigma) + 0.5 * LOG2PI
    nll_left = -log_ndtr(z)     # P(true < y)
    nll_right = -log_ndtr(-z)   # P(true > y)
    nll = torch.where(censor == 0, nll_exact,
                      torch.where(censor < 0, nll_left, nll_right))
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll
