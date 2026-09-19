"""Ranking must not change estimands when its criterion is absent (#2670).

A fit whose summary carries no smoothing-corrected AIC is refused with the
reason the summary records; the ranking never falls back to the conditional
AIC or the raw REML/LAML score.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

pytest.importorskip("gamfit._rust")

import gamfit


def test_compare_models_refuses_a_fit_without_its_criterion_2670() -> None:
    rng = np.random.default_rng(2670)
    n = 250
    x = rng.uniform(0.0, 1.0, n)
    data = {"x": x, "y": np.cos(3.0 * x) + rng.normal(0.0, 0.2, n)}
    # The O(n) spline scan keeps no smoothing-parameter covariance correction.
    scan = gamfit.fit(
        data, 'y ~ s(x, bs="ps", degree=3, penalty_order=2, double_penalty=False)'
    )
    summary = scan.summary()
    assert summary.aic_corrected is None
    assert summary.aic_conditional is not None
    with pytest.raises(ValueError, match=re.escape(summary.aic_corrected_unavailable)):
        gamfit.compare_models([scan], names=["incomplete"])
