"""Bug hunt (#1541): a univariate cubic-regression smooth ``s(x, bs="cr")`` (and
its shrinkage sibling ``bs="cs"``) hard-failed the whole fit on a low-cardinality
covariate instead of capping the basis to the data support the way mgcv — and
gam's own *tensor* path (996f829d7) — do.

```
gamfit._rust.InvalidConfigurationError: cubic regression spline with k=10
requires at least 10 distinct values, got 3
```

A cubic regression spline places exactly one basis function per value-knot, so
``select_cr_knots`` cannot place more knots than the covariate has DISTINCT
values. The univariate ``cr``/``cs`` arm
(``src/terms/term_builder.rs``) used to hand ``select_cr_knots`` an unclamped
``k`` — so an ordinary low-cardinality predictor (a binary indicator, a 3-level
ordinal/Likert score, a small integer count) aborted before any coefficients
were produced. The asymmetry was the tell: the default P-spline basis and the
``te(...)`` cr *margin* both fit the identical data; only the univariate ``cr``
spelling errored.

The fix mirrors the tensor cap: cap ``k`` to ``min(k_requested, n_distinct)``;
build the cr basis when that is ``>= 3`` (a cr basis of exactly ``n_distinct``
knots is full-rank for the data — it represents any per-distinct-value structure
exactly, so the cap never costs recoverable signal); below 3 (a binary
covariate) degrade to the linear B-spline marginal the default ``s(x, k=..)``
basis already builds. Selecting ``bs="cr"`` changes the basis, never turns a
fittable model into a hard error.

These tests fit on a ternary / binary covariate and assert (a) the fit succeeds
and (b) the cr smooth recovers the per-level structure — exactly what the
default basis does on the same data.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _predict_at(model: Any, name: str, levels: "np.ndarray", extra: dict | None = None) -> "np.ndarray":
    """Predict the response at the given covariate ``levels`` (other covariates
    held at a constant baseline) and return the level-0-centred contrasts."""
    cols: dict[str, list] = {name: [float(v) for v in levels]}
    if extra:
        for k, v in extra.items():
            cols[k] = [float(v)] * len(levels)
    pred = np.asarray(model.predict(cols), dtype=float).reshape(-1)
    return pred - pred[0]


def _fit_ternary(formula: str) -> "np.ndarray":
    """Fit ``formula`` on a ternary ``x in {0,1,2}`` whose group means are the
    NON-MONOTONE contrasts ``{0: 0, 1: +2, 2: -1}`` and return the recovered,
    level-0-centred per-level contrasts at ``x in {0,1,2}``."""
    rng = np.random.default_rng(1541)
    n = 900
    x = rng.integers(0, 3, n).astype(float)
    means = np.array([0.0, 2.0, -1.0])
    y = means[x.astype(int)] + rng.normal(0.0, 0.3, n)
    d = {"x": x.tolist(), "y": y.tolist()}
    model = gamfit.fit(d, formula)
    return _predict_at(model, "x", np.array([0.0, 1.0, 2.0]))


def test_univariate_cr_smooth_caps_to_data_support_and_recovers_levels() -> None:
    # cr basis explicitly requested with k=10 on a 3-distinct-value covariate.
    # Before the fix this raised InvalidConfigurationError at fit time.
    contrasts = _fit_ternary("y ~ s(x, bs='cr', k=10)")
    # A cr basis capped to the 3 distinct values still carries 3 functions —
    # enough to represent 3 arbitrary group levels — so the non-monotone
    # contrasts {0, +2, -1} are recoverable (the default basis recovers
    # ~[0, 1.95, -1.02] on this data).
    assert abs(contrasts[1] - 2.0) < 0.35, f"level 1 contrast not recovered: {contrasts}"
    assert abs(contrasts[2] - (-1.0)) < 0.35, f"level 2 contrast not recovered: {contrasts}"
    # Non-monotone shape: level 1 is well above level 2.
    assert contrasts[1] - contrasts[2] > 2.0, f"shape not recovered: {contrasts}"


def test_univariate_cr_auto_k_caps_to_data_support() -> None:
    # The default (auto) k path must cap too, not just explicit k=.
    contrasts = _fit_ternary("y ~ s(x, bs='cr')")
    assert abs(contrasts[1] - 2.0) < 0.35, f"level 1 contrast not recovered: {contrasts}"
    assert abs(contrasts[2] - (-1.0)) < 0.35, f"level 2 contrast not recovered: {contrasts}"


def test_univariate_cs_shrinkage_smooth_caps_to_data_support() -> None:
    # The shrinkage sibling bs="cs" reaches the same uncapped select_cr_knots.
    contrasts = _fit_ternary("y ~ s(x, bs='cs', k=10)")
    assert abs(contrasts[1] - 2.0) < 0.40, f"level 1 contrast not recovered: {contrasts}"
    assert abs(contrasts[2] - (-1.0)) < 0.40, f"level 2 contrast not recovered: {contrasts}"


def test_univariate_cr_smooth_binary_covariate_degrades_to_bspline() -> None:
    # A BINARY covariate has too few distinct values (2) for ANY cr spline
    # (needs >= 3). bs="cr"/"cs" must degrade to the linear B-spline marginal the
    # default basis uses — a hard error here was the issue's headline repro
    # (`s(badh, bs='cr', k=10)` on a 0/1 indicator).
    rng = np.random.default_rng(15411)
    n = 600
    x = rng.integers(0, 2, n).astype(float)  # binary {0,1}
    z = rng.uniform(0.0, 1.0, n)
    y = 0.8 * x + np.sin(2.0 * np.pi * z) + rng.normal(0.0, 0.3, n)
    d = {"x": x.tolist(), "z": z.tolist(), "y": y.tolist()}
    for formula in (
        "y ~ s(x, bs='cr') + s(z)",
        "y ~ s(x, bs='cr', k=10) + s(z)",
        "y ~ s(x, bs='cs', k=10) + s(z)",
    ):
        model = gamfit.fit(d, formula)  # must not raise
        contrasts = _predict_at(model, "x", np.array([0.0, 1.0]), extra={"z": 0.5})
        # The 0->1 jump of ~0.8 is recoverable from the (now linear) basis.
        assert abs(contrasts[1] - 0.8) < 0.30, f"{formula}: binary contrast {contrasts}"
