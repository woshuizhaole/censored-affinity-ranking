"""Validate VS enrichment metrics against known-ranking sanity cases."""
import numpy as np
from eval.enrichment import vs_metrics, per_target_vs, bootstrap_ci
import pandas as pd


def test_perfect_ranker_ef():
    # 1000 mols, 10 actives all ranked first -> EF1% = 1/0.01 = 100
    N, na = 1000, 10
    labels = np.array([1] * na + [0] * (N - na))
    scores = np.linspace(1.0, 0.0, N)   # descending, actives first
    m = vs_metrics(labels, scores)
    assert abs(m["EF1%"] - 100.0) < 1e-6
    assert m["BEDROC20"] > 0.99
    assert abs(m["AUC"] - 1.0) < 1e-6


def test_worst_ranker():
    N, na = 1000, 10
    labels = np.array([1] * na + [0] * (N - na))
    scores = np.linspace(0.0, 1.0, N)   # actives ranked LAST
    m = vs_metrics(labels, scores)
    assert m["EF1%"] < 1e-6
    assert m["AUC"] < 0.02


def test_random_ranker_near_one():
    rng = np.random.default_rng(0)
    N, na = 5000, 250
    labels = np.zeros(N, int); labels[:na] = 1
    rng.shuffle(labels)
    scores = rng.random(N)
    efs = [vs_metrics(labels, rng.random(N))["EF1%"] for _ in range(20)]
    assert 0.3 < np.mean(efs) < 2.5   # random EF ~ 1


def test_per_target_filters_and_bootstrap():
    df = pd.DataFrame({
        "t": ["A"] * 200 + ["B"] * 5,   # B has too few actives
        "label": [1] * 20 + [0] * 180 + [1] * 2 + [0] * 3,
        "score": list(np.linspace(1, 0, 200)) + [0.5] * 5,
    })
    pt = per_target_vs(df, "score", "label", "t", min_actives=10)
    assert set(pt["t"]) == {"A"}          # B filtered out
    mean, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_boot=500)
    assert lo <= mean <= hi
