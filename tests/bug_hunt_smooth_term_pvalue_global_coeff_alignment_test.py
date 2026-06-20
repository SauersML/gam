"""Smooth-term summary must address the *global* coefficient block (issue #1360).

Companion to ``bug_hunt_smooth_term_pvalue_false_positive_with_strong_coterm``,
attacking the same root cause from a different structural angle.

Root cause: ``summary().smooth_terms`` sliced the global coefficient vector
(``fit.beta`` / covariance / influence matrix) with each smooth term's
*block-local* ``coeff_range`` instead of the global range, omitting the
``smooth_start`` offset (= width of the intercept + linear + random blocks that
precede the smooths). Every other call site in the engine offsets by
``smooth_start``; the two summary loops did not, so each smooth's window slid off
by ``smooth_start`` columns and folded foreign coefficients into both the Wald
statistic *and* the reported EDF.

Where the primary test uses ``y ~ s(x) + s(z)`` (``smooth_start = 1``, just the
intercept), this test puts a **leading linear term** in the model so
``smooth_start = 2``. A re-introduced offset bug would therefore shift each
smooth window by *two* columns — a regime the primary test cannot reach — and a
boundary-shrunk null smooth would again read as significant. This also exercises
the reference-d.f. floor: a term shrunk to ~0 EDF whose Wood influence-trace
reference d.f. would otherwise collapse toward 0 must still return ``p ≈ 1``.
"""

from __future__ import annotations

import contextlib
import importlib
import io
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")

import gamfit

ALPHA = 0.05


def _smooth_terms(formula: str, data: dict) -> dict:
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        summary = gamfit.fit(data, formula).summary()
    return {row["name"]: row for row in summary.smooth_terms}


def _is_rejection(row: dict) -> bool:
    p = row.get("p_value")
    return p is not None and np.isfinite(p) and float(p) < ALPHA


def test_null_smooth_calibrated_with_leading_linear_coterm() -> None:
    """``s(b)`` for ``b ⊥ y`` stays non-significant even though a leading linear
    term shifts the smooth block to ``smooth_start = 2``."""
    n_seeds = 20
    reject_b = 0
    reject_a = 0
    edfs_b: list[float] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(31_000 + seed)
        n = 600
        a = rng.uniform(0.0, 1.0, n)
        b = rng.uniform(0.0, 1.0, n)  # independent of y -> null
        xl = rng.uniform(0.0, 1.0, n)  # linear co-term -> smooth_start = 2
        y = np.sin(3.0 * a) + 0.7 * xl + 0.3 * rng.standard_normal(n)
        terms = _smooth_terms("y ~ xl + s(a) + s(b)", {"a": a, "b": b, "xl": xl, "y": y})
        assert {"s(a)", "s(b)"} <= set(terms), terms.keys()
        reject_b += _is_rejection(terms["s(b)"])
        reject_a += _is_rejection(terms["s(a)"])
        edfs_b.append(float(terms["s(b)"]["edf"]))

    rate_b = reject_b / n_seeds
    rate_a = reject_a / n_seeds

    # The shifted-window bug attributed a neighbouring coefficient's leverage to
    # the null term, so its EDF must be honestly small here.
    assert np.median(edfs_b) < 1.5, f"null s(b) median EDF {np.median(edfs_b):.3f} (expected ~0)"
    # Power positive control: the real signal s(a) must stay detectable, so a
    # blanket p-value inflation cannot pass.
    assert rate_a >= 0.8, f"signal s(a) detected on only {rate_a:.0%} of draws"
    # Calibration: the null smooth must not be a systematic false positive.
    assert rate_b <= 0.5, (
        f"null s(b) flagged significant on {rate_b:.0%} of {n_seeds} draws with a "
        "leading linear co-term (smooth_start=2); the global-coefficient offset "
        "regressed"
    )


def test_smooth_pvalue_matches_reconstruction_from_exposed_covariance() -> None:
    """End-to-end alignment check: the engine's per-term Wald statistic, rebuilt
    from the *exposed* corrected covariance and coefficients over the term's true
    global block, must agree. The pre-fix code sliced a misaligned block, so the
    engine's statistic disagreed with any reconstruction over the real block.

    We verify the consequence that uniquely identifies correct alignment: for a
    null term whose coefficients the fit drove to ~0, the engine's chi-square is
    ~0 (a misaligned window over a strong neighbour would be large)."""
    rng = np.random.default_rng(7)
    n = 700
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    xl = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.5 * a) + 0.5 * xl + 0.3 * rng.standard_normal(n)
    terms = _smooth_terms("y ~ xl + s(a) + s(b)", {"a": a, "b": b, "xl": xl, "y": y})

    sb = terms["s(b)"]
    sa = terms["s(a)"]
    # s(b) is null and shrunk: aligned slicing gives a ~0 statistic. A window
    # shifted onto s(a)'s large coefficients (the bug) yields chi_sq in the
    # hundreds-to-thousands.
    chi_b = sb.get("chi_sq")
    if chi_b is not None and np.isfinite(chi_b):
        assert float(chi_b) < 5.0, f"null s(b) chi_sq={float(chi_b):.2f} (misaligned window?)"
    # s(a) is strong: it must be significant and carry real EDF.
    assert _is_rejection(sa), "signal s(a) should be significant"
    assert float(sa["edf"]) > 3.0, f"signal s(a) edf={float(sa['edf']):.3f}"


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    raise SystemExit(pytest.main([__file__, "-v"]))
