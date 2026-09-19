"""Every smooth-significance row publishes a p-value, a bound, or a reason.

`Model.smooth_significance` scores the per-term likelihood-ratio statistic
against its null law. The tail used to come from Imhof inversion, accurate in
*absolute* terms only (~1e-13), and a strong term's row published the
inversion's rounding residue as its p-value: `p_value_corrected = 0.0` for the
audit's Poisson replicate 5 and `5.55e-16` for its Gaussian replicate 34, next
to `p_value_bound ~ 1e-13`. Neither was the tail; both were noise. The tail is
now evaluated with relative accuracy, so those rows resolve to a finite, tiny
p-value that clears its own accuracy.

The contract pinned here:

* every row carries exactly one of `p_value`, `p_value_upper_bound`,
  `unavailable_reason`, and always the same keys;
* a published point value is never below its own evaluation accuracy
  `p_value_bound` (a tail that cannot be resolved is `p < p_value_upper_bound`);
* a strong term's tail is finite, positive and tiny, never zero or rounding;
* a null term's tail is resolved, and equals the evaluated corrected tail;
* a term shrunk to its null is not significant. An estimated-scale statistic
  is scored on its own support, which starts below zero, rather than clamped
  to zero; that is pinned exactly by the Rust test
  `profiled_scale_reference_tests::a_statistic_between_the_offset_and_zero_is_scored_where_it_is`.

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


def _assert_resolved_strong_term(row: dict) -> None:
    p, accuracy = row["p_value"], row["p_value_bound"]
    assert p is not None, (row["statistic_lr"], row["p_value_upper_bound"], row["unavailable_reason"])
    assert np.isfinite(p) and 0.0 < p < 1e-10, (row["statistic_lr"], p)
    # A point value is published only where its accuracy resolves it.
    assert accuracy < p, (p, accuracy)
    assert p == row["p_value_corrected"]


def test_poisson_strong_term_tail_is_finite_and_resolved_not_zero() -> None:
    rows = _rows("poisson", 5)
    _assert_resolved_strong_term(rows["s(x1)"])
    null = rows["s(x2)"]
    assert null["p_value"] is not None and 0.0 < null["p_value"] <= 1.0
    assert null["p_value"] == null["p_value_corrected"]


def test_gaussian_term_shrunk_to_its_null_is_not_significant() -> None:
    # Replicate 427's `s(x2)` is shrunk to its penalty null space (`ref_df`
    # ~3e-6). With an estimated scale `W = n·ln(1 + Q/V) + B`, `B < 0`, and
    # the statistic used to be clamped to zero and published as `9.4e-16`; on
    # its own support, against a reduced model refitted from scratch, it was
    # still `9.0e-4`. The nested null at the full fit's `λ̂` leaves the
    # constraint as the only difference between the two fits.
    row = _rows("gaussian", 427)["s(x2)"]
    assert row["p_value"] is not None and 0.5 < row["p_value"] <= 1.0, (
        row["statistic_lr"],
        row["p_value"],
    )
    assert row["p_value"] == row["p_value_corrected"]


def test_gaussian_strong_term_tail_is_finite_and_resolved_not_rounding() -> None:
    rows = _rows("gaussian", 34)
    _assert_resolved_strong_term(rows["s(x1)"])
    null = rows["s(x2)"]
    assert null["p_value"] is not None and 0.0 < null["p_value"] <= 1.0
    assert null["p_value"] == null["p_value_corrected"]
