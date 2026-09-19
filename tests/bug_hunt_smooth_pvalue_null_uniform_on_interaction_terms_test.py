"""Null smooth p-values of interaction terms are U(0, 1) over the whole range.

A valid p-value has ``P(p <= a) = a`` at EVERY level ``a``: a test whose size
sits below nominal, or whose null p-values pile up near one, is miscalibrated
exactly as a liberal test is. The fractional-rank Wald test the summary used
to report failed that on every null interaction cell of
``bench/pvalue_calibration/pv-interactions`` (Kolmogorov-Smirnov p = 0): a term
REML shrank flat has its λ at the boundary, its Wald statistic collapses to
zero, and a point mass of p-values lands near one, while terms REML left
wiggly are tested against a rank that was itself chosen from the same data.

The summary now reports a variance-component score test whose statistic does
not depend on the term's own λ at all, so neither effect is present. Each case
below refits a seeded null cell 500 times and checks the p-values two-sided:
a Kolmogorov-Smirnov test against U(0, 1), and exact binomial tests that the
share of p-values at or below ``a`` (size) and above ``1 - a`` (top of the
range) is ``a``, for ``a`` in .10/.05/.01. Every one of those checks, over
every null term of every cell, runs at ``FAMILY_ALPHA / CHECKS``, so an exactly
calibrated p-value fails the null checks with probability at most
``FAMILY_ALPHA``.

A uniform p-value is only half of a test; it must also see the effects its term
can carry. The last case keeps the varying coefficient's power: an effect
``z·cos(πx)`` lies mostly in the linear part ``z, z·x`` that the smooth's
null-space penalty shrinks, and a score test that weights its directions by the
inverse of the summed penalties all but ignores that part (it rejected 4% of
these fits at .05, its size). A factor-level curve under a binomial response
is the other. The level's null-space ridge charges the curve's mean slope, and
the covariance direction of that component was the ridge's Euclidean
pseudo-inverse, which points along the mean-slope row (a curve concentrated at
the ends of the range) rather than along the linear null function the ridge
shrinks. The level's linear part was nearly invisible, and a curve of amplitude
2 was found in 7% of fits at .05.
"""

import contextlib
import io
import warnings
import zlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import gamfit

REPS = 500
ALPHAS = (0.10, 0.05, 0.01)
FAMILY_ALPHA = 0.01
TAU = 2.0 * np.pi


def _by_factor(rng):
    n = 300
    g = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    x = rng.uniform(size=n)
    shift = np.select([g == "a", g == "b"], [0.0, 0.5], -0.5)
    eta = shift + np.where(g == "a", 1.5 * np.sin(TAU * x), 0.0)
    y = eta + 0.8 * rng.standard_normal(n)
    frame = pd.DataFrame({"y": y, "x": x, "g": g})
    return frame, "y ~ s(x, by=g)", ["s(x, by=g):by=g[b]", "s(x, by=g):by=g[c]"]


def _tensor_interaction(rng):
    n = 200
    x1, x2 = rng.uniform(size=n), rng.uniform(size=n)
    eta = np.sin(TAU * x1) + 0.8 * np.cos(TAU * x2)
    y = eta + 0.8 * rng.standard_normal(n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return frame, "y ~ s(x1) + s(x2) + ti(x1, x2)", ["ti(x1, x2)"]


def _varying_coefficient(rng):
    n = 300
    x, z = rng.uniform(size=n), rng.standard_normal(n)
    y = np.sin(TAU * x) + 0.8 * rng.standard_normal(n)
    frame = pd.DataFrame({"y": y, "x": x, "z": z})
    return frame, "y ~ s(x) + s(x, by=z)", ["s(x, by=z)"]


CELLS = {
    "by_factor": _by_factor,
    "ti": _tensor_interaction,
    "varying_coefficient": _varying_coefficient,
}
# Null terms over all cells (two levels in by_factor), each checked by one KS
# test plus a size and a top-of-range test per level.
CHECKS = 4 * (1 + 2 * len(ALPHAS))
CHECK_ALPHA = FAMILY_ALPHA / CHECKS


def _null_pvalues(cell):
    builder = CELLS[cell]
    pvalues = {}
    for rep in range(REPS):
        rng = np.random.default_rng([zlib.crc32(cell.encode()), rep])
        frame, formula, null_terms = builder(rng)
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            rows = {row["name"]: row for row in gamfit.fit(frame, formula).summary().smooth_terms}
        for term in null_terms:
            row = rows[term]
            assert row["p_value"] is not None, (term, row.get("p_value_unavailable"))
            pvalues.setdefault(term, []).append(float(row["p_value"]))
    return {term: np.asarray(p) for term, p in pvalues.items()}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_null_interaction_pvalues_are_uniform_over_the_whole_range(cell):
    for term, p in _null_pvalues(cell).items():
        assert np.all((p >= 0.0) & (p <= 1.0)), term
        ks = stats.kstest(p, "uniform")
        assert ks.pvalue > CHECK_ALPHA, f"{term}: KS D={ks.statistic:.3f} p={ks.pvalue:.2g}"
        for a in ALPHAS:
            for side, count in (("p <= a", int(np.sum(p <= a))), ("p > 1 - a", int(np.sum(p > 1.0 - a)))):
                test = stats.binomtest(count, p.size, a)
                assert test.pvalue > CHECK_ALPHA, (
                    f"{term}: P({side}) = {count / p.size:.3f} at a = {a} (binomial p = {test.pvalue:.2g})"
                )


def test_varying_coefficient_effect_in_the_penalty_null_space_is_detected():
    reps, rejected = 100, 0
    for rep in range(reps):
        rng = np.random.default_rng([zlib.crc32(b"varying_coefficient_power"), rep])
        n = 300
        x, z = rng.uniform(size=n), rng.standard_normal(n)
        y = np.sin(TAU * x) + 0.4 * z * np.cos(np.pi * x) + 0.8 * rng.standard_normal(n)
        frame = pd.DataFrame({"y": y, "x": x, "z": z})
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            rows = {row["name"]: row for row in gamfit.fit(frame, "y ~ s(x) + s(x, by=z)").summary().smooth_terms}
        p = rows["s(x, by=z)"]["p_value"]
        assert p is not None, rows["s(x, by=z)"].get("p_value_unavailable")
        rejected += p <= 0.05
    assert rejected >= 0.9 * reps, f"power {rejected}/{reps} at .05"


def test_factor_level_curve_is_detected_under_a_binomial_response():
    reps, rejected = 60, 0
    for rep in range(reps):
        rng = np.random.default_rng([zlib.crc32(b"by_factor_binomial_power"), rep])
        n = 600
        g = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
        x = rng.uniform(size=n)
        shift = np.select([g == "a", g == "b"], [0.0, 0.5], -0.5)
        eta = shift + np.where(g == "a", 2.0 * np.sin(TAU * x), 0.0)
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
        frame = pd.DataFrame({"y": y, "x": x, "g": g})
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            warnings.simplefilter("ignore")
            try:
                model = gamfit.fit(frame, "y ~ s(x, by=g)", family="binomial")
            except gamfit.errors.RemlConvergenceError:
                # The outer search refused to certify this fit, so the user gets
                # no p-value and the curve is not found: a miss, not a skip.
                continue
            rows = {row["name"]: row for row in model.summary().smooth_terms}
        row = rows["s(x, by=g):by=g[a]"]
        assert row["p_value"] is not None, row.get("p_value_unavailable")
        rejected += row["p_value"] <= 0.05
    assert rejected >= 0.9 * reps, f"power {rejected}/{reps} at .05"
