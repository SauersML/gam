"""Regression test for #3878: ``Model.evidence_ratio_vs`` past the float range.

The ratio is ``exp(½·(AIC_c(other) − AIC_c(self)))``. The Python wrapper used
to exponentiate the Rust log ratio with ``math.exp``, which raises
``OverflowError`` once the corrected-AIC gap exceeds ~1419.6 — routine for a
real signal against the null model at a few thousand rows. The ratio is now
formed in Rust and rounded by IEEE arithmetic: ``inf`` above ``f64::MAX`` and
``0.0`` below the smallest subnormal, the correctly rounded values of the true
ratio.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import gamfit


def test_evidence_ratio_vs_rounds_to_inf_and_zero_past_float_range() -> None:
    rng = np.random.default_rng(3878)
    n = 2000
    x = np.sort(rng.uniform(-3, 3, n))
    y = np.sin(2 * x) + 0.5 * x + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "y": y})
    m_sx = gamfit.fit(df, "y ~ s(x)", family="gaussian")
    m_null = gamfit.fit(df, "y ~ 1", family="gaussian")

    gap = m_null.summary().aic_corrected - m_sx.summary().aic_corrected
    # Anchor: the half-gap is past the largest finite exponent of a double.
    assert 0.5 * gap > math.log(np.finfo(np.float64).max)

    assert m_sx.evidence_ratio_vs(m_null) == math.inf
    assert m_null.evidence_ratio_vs(m_sx) == 0.0
