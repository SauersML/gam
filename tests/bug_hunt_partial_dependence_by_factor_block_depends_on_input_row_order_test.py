"""Regression test for issue #2076.

``Model.partial_dependence`` for a ``by=``-factor smooth block must return the
same curve regardless of the row order of the ``data`` frame passed. The bug:
the grid template pinned the grouping factor at the value found in the FIRST
row of ``data``, so evaluating an off-level block (whose design columns are
non-zero only on rows matching its own level) produced an identically-zero
curve with zero standard error. Partial dependence of a fixed block is a model
property and must not depend on the input frame's row order.
"""

import numpy as np
import pandas as pd

import gamfit

TERM = "s(x, by=g, k=6):by=g[a]"


def _make_frame():
    rng = np.random.default_rng(0)
    xs = np.linspace(-2, 2, 60)
    x = np.concatenate([xs, xs])
    g = ["a"] * 60 + ["b"] * 60
    gg = np.array(g)
    y = 1 + np.where(gg == "a", np.sin(2 * x), -np.sin(2 * x)) + rng.normal(0, 0.05, 120)
    return pd.DataFrame({"x": x, "g": g, "y": y})


def test_partial_dependence_by_block_independent_of_row_order():
    df = _make_frame()
    model = gamfit.fit(df, "y ~ g + s(x, by=g, k=6)")

    names = [b.name for b in model.term_blocks]
    assert TERM in names, f"expected block {TERM!r}; available: {names}"

    a_first = df.copy().reset_index(drop=True)
    b_first = df.sort_values("g", ascending=False).reset_index(drop=True)
    assert a_first.iloc[0]["g"] == "a"
    assert b_first.iloc[0]["g"] == "b"

    ra = model.partial_dependence(TERM, a_first)
    rb = model.partial_dependence(TERM, b_first)

    pa = np.asarray(ra["predicted"], dtype=float)
    pb = np.asarray(rb["predicted"], dtype=float)
    sea = np.asarray(ra["standard_error"], dtype=float)
    seb = np.asarray(rb["standard_error"], dtype=float)

    # The two evaluations must agree exactly (same model property).
    np.testing.assert_allclose(pa, pb, atol=1e-6)
    np.testing.assert_allclose(sea, seb, atol=1e-6)

    # And must be the real (non-zero) sin(2x)-shaped curve, not all zeros.
    assert np.max(np.abs(pa)) > 0.1
    assert np.max(np.abs(pb)) > 0.1

    # Standard errors must be finite and positive.
    assert np.all(np.isfinite(sea))
    assert np.all(np.isfinite(seb))
    assert np.max(sea) > 0.0
    assert np.max(seb) > 0.0
