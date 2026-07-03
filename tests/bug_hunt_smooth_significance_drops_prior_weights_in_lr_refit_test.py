"""Regression for #2093: ``smooth_significance`` must honor saved prior weights.

``Model.smooth_significance`` computes a per-term likelihood-ratio test via a
constrained refit. Before the fix the FFI re-materialized the design with a
``None`` weight column, so the refit silently used unit weights and the
weighted-fit LR statistic came out bit-for-bit identical to the unit-weight
fit. The fix threads the model's stored weight column into the LR refit.

The frequency-weight oracle: integer prior weights are equivalent to replicating
each row that many times under unit weights, so the weighted LR statistic must
track the LR statistic of the frequency-replicated dataset (and must NOT equal
the unit-weight statistic, which is what the buggy FFI produced).

A Poisson response is used deliberately: the response is a fixed realization
(so the frequency-replication oracle is exact), and prior weights enter the
Poisson log-likelihood as ``sum(w_i * loglik_i)``, exactly matching replication.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


def test_smooth_significance_uses_prior_weights_and_matches_replication() -> None:
    n = 60
    rng = np.random.default_rng(2093)
    x = np.linspace(0.0, 1.0, n)
    mu = np.exp(0.6 * np.sin(4.0 * x) + 0.4 * x)
    # Fixed deterministic count realization => the frequency-replication oracle
    # below is exact (every replicate of a row carries the identical response).
    y = rng.poisson(mu).astype(float)
    # Integer prior weights => a frequency-weight replication oracle exists.
    w = rng.integers(1, 5, size=n).astype(float)

    weighted_frame = {"x": list(x), "y": list(y), "w": list(w)}
    weighted = gamfit.fit(weighted_frame, "y ~ s(x)", family="poisson", weights="w")
    lr_weighted = float(weighted.smooth_significance(weighted_frame)[0]["statistic_lr"])

    # Unit-weight fit of the same rows: the buggy FFI produced THIS value even
    # for the weighted model (it dropped the weight column), so the two must now
    # differ by roughly the mean weight.
    unit_frame = {"x": list(x), "y": list(y)}
    unit = gamfit.fit(unit_frame, "y ~ s(x)", family="poisson")
    lr_unit = float(unit.smooth_significance(unit_frame)[0]["statistic_lr"])

    assert abs(lr_weighted - lr_unit) > 0.1 * max(1.0, abs(lr_unit)), (
        f"weighted LR {lr_weighted} must differ from unit-weight LR {lr_unit}; "
        "near-identical values mean prior weights were dropped in the refit"
    )

    # Frequency-weight replication oracle: replicate each row w_i times, fit with
    # unit weights, and compare the LR statistic. Prior-weighted and frequency-
    # replicated likelihoods are identical, so the LR statistics must agree to
    # within 20%.
    x_rep: list[float] = []
    y_rep: list[float] = []
    for xi, yi, wi in zip(x, y, w, strict=True):
        for _ in range(int(wi)):
            x_rep.append(float(xi))
            y_rep.append(float(yi))
    rep_frame = {"x": x_rep, "y": y_rep}
    rep = gamfit.fit(rep_frame, "y ~ s(x)", family="poisson")
    lr_rep = float(rep.smooth_significance(rep_frame)[0]["statistic_lr"])

    rel = abs(lr_weighted - lr_rep) / max(1.0, abs(lr_rep))
    assert rel < 0.20, (
        f"weighted LR {lr_weighted} must match the frequency-replication oracle "
        f"{lr_rep} within 20%; relative gap was {rel:.3f}"
    )
