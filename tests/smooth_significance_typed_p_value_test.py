"""Every smooth-significance row publishes a p-value, a bound, or a reason.

`Model.smooth_significance` scores the per-term likelihood-ratio statistic
against its null law by Imhof inversion, which is accurate in *absolute* terms
only (to `p_value_bound`, ~1e-13). A strong term's tail lies below that, and
the row used to publish the inversion's rounding residue as the p-value:
`p_value_corrected = 0.0` for the audit's Poisson replicate 5 (W = 90.8) and
`5.55e-16` for its Gaussian replicate 34, next to `p_value_bound ~ 1e-13`.
Neither is the tail; both are noise.

The contract pinned here:

* every row carries exactly one of `p_value`, `p_value_upper_bound`,
  `unavailable_reason`, and always the same keys;
* a tail the reference cannot resolve is published as `p < p_value_upper_bound`
  — a finite, positive number no larger than the evaluation accuracy — never
  as a point value;
* a null term's tail is resolved, and equals the evaluated corrected tail.

Replicates are seeded `default_rng(1000 + rep)` over the pyGAM audit's
inference cells (bench/pygam_audit; bench/pvalue_calibration/pv-lr-refit).
"""

from __future__ import annotations

import numpy as np

import gamfit

KEYS = ("p_value", "p_value_upper_bound", "unavailable_reason")


def _cell(family: str, rep: int) -> dict[str, np.ndarray]:
    n, b0, a1, a3 = {"gaussian": (200, 0.0, 1.0, 0.30), "poisson": (200, 0.5, 0.8, 0.25)}[family]
    rng = np.random.default_rng(1000 + rep)
    X = rng.uniform(0, 1, (n, 3))
    eta = b0 + a1 * np.sin(2 * np.pi * X[:, 0]) + a3 * np.cos(2 * np.pi * X[:, 2])
    y = eta + rng.normal(0, 1.0, n) if family == "gaussian" else rng.poisson(np.exp(eta)).astype(float)
    return dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2], y=y)


def _rows(family: str, rep: int) -> dict[str, dict]:
    data = _cell(family, rep)
    model = gamfit.fit(data, "y ~ s(x1) + s(x2) + s(x3)", family=family)
    rows = model.smooth_significance(data)
    assert [r["name"] for r in rows] == ["s(x1)", "s(x2)", "s(x3)"]
    keys = {frozenset(r) for r in rows}
    assert len(keys) == 1, f"rows disagree on their keys: {keys}"
    for r in rows:
        present = [k for k in KEYS if r[k] is not None]
        assert len(present) == 1, f"{r['name']}: expected exactly one of {KEYS}, got {present}"
    return {r["name"]: r for r in rows}


def _assert_bounded_strong_term(row: dict) -> None:
    assert row["p_value"] is None, (
        f"{row['name']}: W={row['statistic_lr']:.1f} published the point value "
        f"{row['p_value']!r}, below the evaluation accuracy {row['p_value_bound']:.3g}"
    )
    bound = row["p_value_upper_bound"]
    assert bound is not None and np.isfinite(bound)
    # The ceiling is never looser than "evaluated residue + its accuracy".
    ceiling = max(row["p_value_corrected"], 0.0) + row["p_value_bound"]
    assert 0.0 < bound <= ceiling, (bound, ceiling)


def test_poisson_tail_below_the_imhof_accuracy_is_a_bound_not_zero() -> None:
    rows = _rows("poisson", 5)
    _assert_bounded_strong_term(rows["s(x1)"])
    null = rows["s(x2)"]
    assert null["p_value"] is not None and 0.0 < null["p_value"] <= 1.0
    assert null["p_value"] == null["p_value_corrected"]


def test_gaussian_tail_below_the_imhof_accuracy_is_a_bound_not_rounding() -> None:
    rows = _rows("gaussian", 34)
    _assert_bounded_strong_term(rows["s(x1)"])
    null = rows["s(x2)"]
    assert null["p_value"] is not None and 0.0 < null["p_value"] <= 1.0
    assert null["p_value"] == null["p_value_corrected"]
