"""#1379 — univariate ``matern(x)`` / ``s(x, bs="gp")`` must fit ordinary 1-D
data instead of deterministically aborting at n=200 on >50% of datasets.

Root cause: during the REML / spatial-κ optimization the per-penalty smoothing
weight ``λ_k = exp(ρ_k)`` of the redundant Matérn *stiffness* operator is driven
to ``+∞`` (the Matérn kernel already controls the smoothness that operator also
penalizes, so on these 1-D geometries REML wants λ_stiffness → ∞). The range
penalty block is assembled as ``Σ_k λ_k S_k``; wherever a transformed ``S_k``
entry is exactly ``0.0``, ``∞ · 0 = NaN``, so the *whole* block came back
non-finite with zero finite content and the eigensolve aborted the fit with
``range penalty block contains non-finite entries (max finite magnitude
0.000e0)``. n=400/800 happened to avoid the overflowing λ; ``bs="cr"/"ps"/"tp"``
and ``duchon(x)`` were never affected.

The fix finite-ceilings the per-penalty λ before it weights any penalty block
(``stable_reparameterization_with_invariant`` in src/terms/construction.rs), so
``λ · 0`` stays ``0`` and the block remains a well-formed PSD matrix while still
pinning the over-penalized direction. Ordinary finite λ pass through untouched,
so non-degenerate fits and their recorded λ̂ are unchanged.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")

import gamfit


def _ordinary_1d_dataset(seed: int, n: int = 200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    truth = np.sin(2.0 * np.pi * x)
    y = truth + rng.normal(0.0, 0.1, n)
    df = pd.DataFrame({"x": x, "y": y})
    return df, truth


def _fit_recovery(term: str, seed: int, n: int = 200) -> float:
    df, truth = _ordinary_1d_dataset(seed, n)
    model = gamfit.fit(df, f"y ~ {term}", family="gaussian")
    pred = np.asarray(model.predict(df)).ravel()
    return float(np.corrcoef(pred, truth)[0, 1])


# The seeds the issue reported as deterministically aborting at n=200.
_FAILING_SEEDS = [3, 4, 6, 7]


def test_univariate_matern_smooth_fits_ordinary_1d_data():
    # Before the #1379 fix every one of these aborted at n=200 with the
    # "range penalty block contains non-finite entries" eigensolve error.
    corrs = [_fit_recovery("matern(x)", seed) for seed in _FAILING_SEEDS]
    assert min(corrs) > 0.9, dict(zip(_FAILING_SEEDS, corrs))


def test_univariate_gp_basis_alias_fits_ordinary_1d_data():
    # `s(x, bs="gp")` is the alias for `matern(x)` and shared the same abort.
    corrs = [_fit_recovery('s(x, bs="gp")', seed) for seed in _FAILING_SEEDS]
    assert min(corrs) > 0.9, dict(zip(_FAILING_SEEDS, corrs))


def test_cr_smooth_fits_same_data_control():
    # Control: the sibling cr smooth was never affected and proves the data is
    # an easy fit at n=200 for these seeds.
    corrs = [_fit_recovery('s(x, bs="cr")', seed) for seed in _FAILING_SEEDS]
    assert min(corrs) > 0.9, dict(zip(_FAILING_SEEDS, corrs))
