"""Commodity 2-layer MLP scoring head (DeepPurpose-grade), two-head (mu, log_sigma_raw).

Input = [ECFP4(1024) || frozen ESM-2 t12_35M mean-pool(480)] = 1504.
Deliberately cheap so any EF/BEDROC gain is attributable to label handling.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class AffinityMLP(nn.Module):
    def __init__(self, in_dim: int = 1504, hidden=(512, 256), dropout: float = 0.1,
                 two_head: bool = True):
        super().__init__()
        dims = [in_dim] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(dims[-1], 1)
        self.two_head = two_head
        if two_head:
            self.logsig_head = nn.Linear(dims[-1], 1)
        else:
            # global learnable log-sigma
            self.log_sigma_global = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h).squeeze(-1)
        if self.two_head:
            log_sigma = self.logsig_head(h).squeeze(-1)
        else:
            log_sigma = self.log_sigma_global.expand_as(mu)
        return mu, log_sigma
