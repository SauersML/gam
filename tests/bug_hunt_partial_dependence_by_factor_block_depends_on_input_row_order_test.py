"""Regression test for issue #2076.

``Model.partial_dependence`` for a ``by=``-factor smooth block must be that block's
own curve. The bug: the grid template pinned the grouping factor at the value in the
FIRST row of the caller's ``data`` frame. So an off-level block, whose design columns
are non-zero only on rows at its level, gave an identically-zero curve with zero
standard error, and the answer depended on that frame's row order.

The grid now comes from the saved term specification (#2899 P8):
- the block holds the level its specification records;
- there is no reference frame to reorder.

The old code parsed the level out of the block's name. A level label containing
``:by=`` broke that parse, so this file pins such a label too.
"""

import numpy as np
import pandas as pd

import gamfit


def _make_frame(levels):
    rng = np.random.default_rng(0)
    xs = np.linspace(-2, 2, 60)
    x = np.concatenate([xs, xs])
    g = [levels[0]] * 60 + [levels[1]] * 60
    gg = np.array(g)
    y = 1 + np.where(gg == levels[0], np.sin(2 * x), -np.sin(2 * x)) + rng.normal(0, 0.05, 120)
    return pd.DataFrame({"x": x, "g": g, "y": y})


def _block_oracle(model, term, grid, level):
    """``X_t beta_t`` and ``sqrt(diag(X_t V_t X_t^T))`` at ``g = level``."""
    summary = model.summary()
    beta = np.asarray([c["estimate"] for c in summary.coefficients], dtype=float)
    cov = summary.covariance
    block = next(b for b in model.term_blocks if b.name == term)
    frame = pd.DataFrame({"x": grid, "g": [level] * grid.size})
    design = np.asarray(model.design_matrix(frame).matrix, dtype=float)
    columns = design[:, block.start : block.end]
    sub = cov[block.start : block.end, block.start : block.end]
    return columns @ beta[block.start : block.end], np.sqrt(
        np.einsum("ij,jk,ik->i", columns, sub, columns)
    )


def _assert_block_curve(levels):
    model = gamfit.fit(_make_frame(levels), "y ~ g + s(x, by=g, k=6)")
    term = f"s(x, by=g, k=6):by=g[{levels[0]}]"
    names = [b.name for b in model.term_blocks]
    assert term in names, f"expected block {term!r}; available: {names}"

    result = model.partial_dependence(term, n_points=25)
    assert result.held == {"g": levels[0]}
    assert result.quantity == "term_contribution"
    assert result.scale == "linear_predictor"

    predicted = result.fit
    se = result.se
    expected, expected_se = _block_oracle(
        model, term, result.x, levels[0]
    )
    np.testing.assert_allclose(predicted, expected, atol=1e-10)
    np.testing.assert_allclose(se, expected_se, atol=1e-10)

    # The real sin(2x)-shaped curve, not the zero curve an off-level template gave.
    assert np.max(np.abs(predicted)) > 0.1
    assert np.all(np.isfinite(se))
    assert np.max(se) > 0.0


def test_partial_dependence_by_block_holds_its_recorded_level():
    _assert_block_curve(("a", "b"))


def test_partial_dependence_by_block_level_label_containing_the_by_marker():
    _assert_block_curve(("a:by=h[c", "b"))
