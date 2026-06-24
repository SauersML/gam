"""Bug hunt (#1541, #1542): the *binary-covariate* arm of the cubic-regression
data-support cap.

The cr/cs/sz data-support cap (``capped_cr_marginal_knotspec``,
``src/terms/term_builder.rs``) has two regimes:

* ``n_distinct >= 3`` — build the cr basis with ``k = min(k_requested,
  n_distinct)`` value-knots. Covered by the ternary tests in
  ``bug_hunt_sz_factor_smooth_cr_marginal_low_cardinality_hard_fails_test.py``
  and the univariate regression test for #1541.
* ``n_distinct < 3`` (a **binary** covariate) — too few distinct values for ANY
  natural cubic regression spline, so the marginal *degrades* to the linear
  B-spline marginal the default ``s(x, k=..)`` basis already builds. This is the
  ``None`` branch of ``capped_cr_marginal_knotspec``.

That degradation branch only had a Rust *spec* test asserting the resulting
basis is **not** a cr basis — nothing asserted that the degraded fit still
*recovers* the signal end-to-end. A regression that, say, degraded to an empty
or constant basis (or re-introduced the original hard error for the ``< 3``
case) would pass the spec test while silently destroying the fit. This test
closes that gap from the behavioural angle: on binary ``x in {0,1}`` the
degraded cr/cs/sz fits must succeed AND recover the linear effect / per-group
contrast, matching what an explicit linear basis recovers on the same data.

A binary covariate carries exactly two distinct values, i.e. one slope's worth
of information; the degraded linear marginal is the correct, fully-identified
basis for that — so requiring recovery here is both well-posed and the mgcv
behaviour (mgcv silently reduces ``k`` to the data support rather than erroring).
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _slope(model: Any, var: str = "x") -> float:
    """Predicted change in the response as ``x`` goes 0 -> 1 (the only contrast a
    binary covariate can carry)."""
    pred = np.asarray(model.predict({var: [0.0, 1.0]}), dtype=float).reshape(-1)
    return float(pred[1] - pred[0])


def _group_contrast(model: Any, group: str) -> float:
    pred = np.asarray(
        model.predict({"x": [0.0, 1.0], "g": [group, group]}), dtype=float
    ).reshape(-1)
    return float(pred[1] - pred[0])


def _binary_linear_data(seed: int, slope: float = 2.0) -> dict:
    rng = np.random.default_rng(seed)
    n = 1000
    x = rng.integers(0, 2, n).astype(float)  # binary {0, 1}
    y = slope * x + rng.normal(0.0, 0.3, n)
    return {"x": x.tolist(), "y": y.tolist()}


@pytest.mark.parametrize("bs", ["cr", "cs"])
def test_univariate_cr_cs_binary_covariate_degrades_and_recovers_slope(bs: str) -> None:
    # Before #1541 this raised InvalidConfigurationError ("cubic regression
    # spline with k=10 requires at least 10 distinct values, got 2"). After the
    # cap, n_distinct=2 < CR_MIN_KNOTS degrades to the linear B-spline marginal,
    # which must still recover the slope of 2 on a binary covariate.
    d = _binary_linear_data(seed=1541, slope=2.0)
    model = gamfit.fit(d, f"y ~ s(x, bs='{bs}', k=10)")
    s = _slope(model)
    assert 1.7 < s < 2.3, f"bs={bs!r} binary slope not recovered: {s}"

    # And it must agree with what an honest linear basis recovers on the SAME
    # data: the degradation is a no-op on the recoverable signal, not a loss.
    linear = gamfit.fit(d, "y ~ s(x, bs='ps', k=10)")
    assert abs(s - _slope(linear)) < 0.15, (
        f"degraded bs={bs!r} disagrees with linear basis: {s} vs {_slope(linear)}"
    )


def test_sz_factor_smooth_binary_covariate_degrades_and_recovers_contrasts() -> None:
    # The factor-smooth twin of the above: the sz marginal is cr by default, so a
    # binary covariate hit the same uncapped select_cr_knots hard error. The
    # degraded marginal must still recover the per-group contrast (A: +1.5 at
    # x=1, B: -1.5).
    rng = np.random.default_rng(15421)
    n = 1400
    x = rng.integers(0, 2, n).astype(float)
    g = np.where(rng.integers(0, 2, n) == 0, "A", "B")
    y = np.where(g == "A", 1.5 * x, -1.5 * x) + rng.normal(0.0, 0.3, n)
    d = {"x": x.tolist(), "g": g.tolist(), "y": y.tolist()}

    sz = gamfit.fit(d, "y ~ s(x, g, bs='sz')")
    a = _group_contrast(sz, "A")
    b = _group_contrast(sz, "B")
    assert a > 1.2, f"sz binary group A contrast not recovered: {a}"
    assert b < -1.2, f"sz binary group B contrast not recovered: {b}"
    assert a - b > 2.4, f"sz binary groups not distinguished: A={a}, B={b}"

    # Cross-check against the B-spline-marginal fs sibling, which never used the
    # cr marginal and so always fit this data: the degraded sz must land in the
    # same neighbourhood, confirming the fallback is the same linear space.
    fs = gamfit.fit(d, "y ~ fs(x, g)")
    assert abs(a - _group_contrast(fs, "A")) < 0.5, "sz vs fs disagree (A)"
    assert abs(b - _group_contrast(fs, "B")) < 0.5, "sz vs fs disagree (B)"


def test_binary_cr_does_not_raise_for_auto_k() -> None:
    # The auto-k path (no explicit k=) must also be capped: heuristic_knots
    # could request more knots than a binary covariate supports.
    d = _binary_linear_data(seed=99, slope=-1.0)
    model = gamfit.fit(d, "y ~ s(x, bs='cr')")
    assert -1.3 < _slope(model) < -0.7, "auto-k binary cr slope not recovered"
