"""Bug hunt (#1542): a factor smooth ``s(x, g, bs="sz")`` hard-failed the whole
fit on a low-cardinality covariate because its per-level *cubic-regression*
marginal was never capped to the data support — while the B-spline-marginal
``fs(x, g)`` sibling fit the identical data.

```
gamfit._rust.InvalidConfigurationError: cubic regression spline with k=5
requires at least 5 distinct values, got 3
```

mgcv's ``bs="sz"`` builds the per-level deviation curves on a cubic regression
spline marginal (``src/terms/term_builder.rs``). That marginal called
``select_cr_knots`` with an unclamped basis size, so a 3-level ordinal / small
count aborted the fit before any coefficients were produced. The asymmetry was
the tell: ``fs(x, g)`` (B-spline marginal) recovers the per-group curves on the
same data; only the ``sz`` (cr-marginal) spelling errored.

The fix mirrors the univariate ``cr``/``cs`` cap (#1541) and the tensor-margin
cap (996f829d7): cap the cr marginal to ``min(k_requested, n_distinct)`` value-
knots, build the cr basis when that is ``>= 3``, and below 3 (a binary
covariate) degrade to the B-spline marginal ``fs`` already uses — keeping the
``sz`` deviation flavour either way.

This test fits ``y ~ s(x, g, bs="sz")`` on a ternary ``x in {0,1,2}`` with two
groups whose deviation curves are non-monotone and group-specific, and asserts
the fit succeeds and recovers the group-specific contrasts — exactly what
``fs(x, g)`` already does.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _per_group_contrasts(model: Any, group: str) -> "np.ndarray":
    """Predict the per-group curve at ``x in {0,1,2}`` and centre it at x=0."""
    cols = {"x": [0.0, 1.0, 2.0], "g": [group, group, group]}
    pred = np.asarray(model.predict(cols), dtype=float).reshape(-1)
    return pred - pred[0]


def _make_ternary_factor_smooth_data(seed: int) -> dict:
    """Ternary ``x in {0,1,2}``, two groups A/B with group-specific NON-MONOTONE
    deviation curves: A bumps +2 at x=1, B dips -2 at x=1 (both return to 0 at
    x=2). ``fs(x, g)`` recovers A=[0,~2,~0], B=[0,~-2,~0] on this data."""
    rng = np.random.default_rng(seed)
    n = 1200
    x = rng.integers(0, 3, n).astype(float)
    g = np.where(rng.integers(0, 2, n) == 0, "A", "B")
    dev = {
        "A": np.array([0.0, 2.0, 0.0]),
        "B": np.array([0.0, -2.0, 0.0]),
    }
    y = np.array([dev[gi][int(xi)] for gi, xi in zip(g, x)]) + rng.normal(0.0, 0.3, n)
    return {"x": x.tolist(), "g": g.tolist(), "y": y.tolist()}


def test_sz_factor_smooth_caps_cr_marginal_and_recovers_group_curves() -> None:
    d = _make_ternary_factor_smooth_data(1542)
    # Before the fix this raised InvalidConfigurationError at fit time (cr
    # marginal k=5 with only 3 distinct x-values).
    model = gamfit.fit(d, "y ~ s(x, g, bs='sz')")
    a = _per_group_contrasts(model, "A")
    b = _per_group_contrasts(model, "B")
    # Group-specific non-monotone deviations are recovered: A bumps up at x=1,
    # B dips down, and they are distinct (sz sums deviations to zero across
    # levels, so the two curves are mirror images here).
    assert a[1] > 1.4, f"group A bump at x=1 not recovered: {a}"
    assert b[1] < -1.4, f"group B dip at x=1 not recovered: {b}"
    assert a[1] - b[1] > 2.8, f"groups not distinguished: A={a}, B={b}"


def test_sz_factor_smooth_explicit_k_caps_cr_marginal() -> None:
    d = _make_ternary_factor_smooth_data(15422)
    # Explicit k=10 must cap too, not only the auto-k path.
    model = gamfit.fit(d, "y ~ s(x, g, bs='sz', k=10)")
    a = _per_group_contrasts(model, "A")
    b = _per_group_contrasts(model, "B")
    assert a[1] > 1.4 and b[1] < -1.4, f"A={a}, B={b}"


def test_sz_matches_fs_marginal_on_low_cardinality() -> None:
    # The whole point of the asymmetry: sz (cr marginal) and fs (B-spline
    # marginal) must both fit the low-cardinality data and recover the same
    # group-specific structure. Cross-check sz against the fs sibling.
    d = _make_ternary_factor_smooth_data(15423)
    sz = gamfit.fit(d, "y ~ s(x, g, bs='sz')")
    fs = gamfit.fit(d, "y ~ fs(x, g)")
    for grp, sign in (("A", +1.0), ("B", -1.0)):
        a_sz = _per_group_contrasts(sz, grp)[1]
        a_fs = _per_group_contrasts(fs, grp)[1]
        assert sign * a_sz > 1.4, f"sz {grp} not recovered: {a_sz}"
        assert sign * a_fs > 1.4, f"fs {grp} not recovered: {a_fs}"
        # The two marginals agree on the recovered deviation to within tolerance.
        assert abs(a_sz - a_fs) < 0.6, f"sz vs fs mismatch for {grp}: {a_sz} vs {a_fs}"
