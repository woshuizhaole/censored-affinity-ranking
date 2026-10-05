"""Stage 0 smoke test: environment, reproducibility harness, and numerics.

Validates the base is sound end-to-end at the plumbing level:
imports, determinism, a tiny MLP train step, RDKit fingerprint, and the
numerically-stable log_ndtr used by the Tobit loss. The full A-F end-to-end
smoke on real features is exercised at Stage 4/6.
"""
import math
import numpy as np


def test_core_imports():
    import torch  # noqa
    import rdkit  # noqa
    import esm  # noqa
    import transformers  # noqa
    import sklearn  # noqa
    import polars  # noqa
    import pandas  # noqa
    import scipy  # noqa


def test_log_ndtr_numerics():
    """log_ndtr must equal log(Phi) and stay finite at extremes (Tobit safety)."""
    import torch
    from torch.special import log_ndtr

    assert abs(float(log_ndtr(torch.tensor(0.0))) - math.log(0.5)) < 1e-5
    from scipy.stats import norm

    for z in [-3.0, -1.0, 1.0, 3.0]:
        assert abs(float(log_ndtr(torch.tensor(z))) - norm.logcdf(z)) < 1e-4
    for z in [-40.0, -38.0, 38.0, 40.0]:
        assert math.isfinite(float(log_ndtr(torch.tensor(z))))


def test_set_seed_determinism():
    from utils.seeds import set_seed
    import torch

    set_seed(123)
    a = torch.randn(64, 8)
    na = np.random.rand(10)
    set_seed(123)
    b = torch.randn(64, 8)
    nb = np.random.rand(10)
    assert torch.allclose(a, b)
    assert np.allclose(na, nb)


def test_tiny_mlp_train_step():
    """A minimal seeded MLP forward/backward on fake data must reduce loss
    and be bit-reproducible across two seeded runs."""
    import torch
    import torch.nn as nn
    from utils.seeds import set_seed

    def run():
        set_seed(7)
        X = torch.randn(256, 1504)
        y = torch.randn(256, 1)
        net = nn.Sequential(
            nn.Linear(1504, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1),
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            loss = ((net(X) - y) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        return losses

    l1 = run()
    l2 = run()
    assert l1[-1] < l1[0]          # learns
    assert abs(l1[-1] - l2[-1]) < 1e-6  # reproducible


def test_rdkit_morgan_fp():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    m = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    fp = gen.GetFingerprint(m)
    assert fp.GetNumBits() == 1024
    assert fp.GetNumOnBits() > 0
