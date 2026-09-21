"""Complementary regression coverage for the smooth_significance() LR
reference-d.f. collapse (#1766), from two angles the original RED test does
not directly exercise:

1. The *invariant*. The likelihood-ratio test in ``smooth_significance`` drops
   the whole smooth term (its unpenalized linear null space included), so its
   chi-square reference d.f. must be at least the term's effective d.f.:
   ``ref_df >= edf``. This is the Wood ``edf1 >= edf`` relation. The bug was a
   degenerate ``tr(F)^2 / tr(F^2)`` (with a *non-symmetric* influence ``F``)
   dropping ``ref_df`` to ~1e-12 while ``edf`` stayed ~1.0. We assert the
   invariant holds per fit, which fails the instant the collapse occurs and is
   independent of the magnitude of ``W``.

2. The *calibration rate*. Under the null (pure noise, no signal) a calibrated
   test rejects at exactly alpha. The collapse drove the false-positive rate to
   ~0.23-0.35 at alpha=0.05 because every ``edf==1.0`` fit produced p~1e-12.
   We sweep independent pure-noise fits and audit the non-rejection rate with
   the shared two-sided Wilson verdict (``tests/conftest.py``), so an undersized
   test fails just like an oversized one (#3534).

A conservative test is as wrong as an anti-conservative one, and a one-sided
FPR bar cannot see it: with 40 fits the rejection count's lower tail is at
zero. The null p-values are therefore also held to U(0, 1) through their mean
(1/2, sd sqrt(1/(12 R))). The power control (a genuinely wiggly signal must
still be flagged at p < 1e-3) does not guard against inflated p-values on its
own: a strong signal's p-value stays tiny under almost any inflation.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")

import gamfit


N = 200
ALPHA = 0.05


def _record(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    model = gamfit.fit({"x": list(x), "y": list(y)}, "y ~ s(x)")
    summary = model.summary()
    # The whole-term LR reference d.f. is a property of the SMOOTH TERM alone, so
    # the quantity it must dominate is the term's own effective d.f. — not the
    # model total, which also carries the unpenalized intercept (exactly 1 d.f.
    # under the identity link for a single-smooth model). Subtracting it gives the
    # term edf; without this correction a fully-collapsed smooth (term_edf ~ 0,
    # model total ~ 1.0000+eps) trips the ref_df >= edf invariant by ~1e-5 even
    # though ref_df (floored at 1) sits vastly ABOVE the true term edf.
    term_edf = max(float(summary.edf_total) - 1.0, 0.0)
    rec = model.smooth_significance({"x": list(x), "y": list(y)})[0]
    return {
        "edf": term_edf,
        "edf_total": float(summary.edf_total),
        "W": float(rec["statistic_lr"]),
        "ref_df": float(rec["ref_df"]),
        "p_corrected": float(rec["p_value_corrected"]),
        "p_uncorrected": float(rec["p_value_uncorrected"]),
    }


def test_ref_df_never_below_term_edf_invariant() -> None:
    # The whole-term LR reference d.f. must satisfy ref_df >= term_edf for every
    # fit, including the near-flat fits whose smooth collapses to term_edf -> 0
    # (where the degenerate tr(F)^2/tr(F^2) used to crash ref_df to ~1e-12).
    violations: list[dict[str, float]] = []
    for seed in range(16):
        rng = np.random.default_rng(seed)
        x = np.linspace(0.0, 1.0, N)
        y = 0.01 * rng.standard_normal(N)  # essentially constant -> term_edf -> 0
        rec = _record(x, y)
        # A small numerical slack: ref_df should not sit materially below the
        # term's effective d.f.
        if rec["ref_df"] < rec["edf"] - 1e-6:
            violations.append({"seed": float(seed), **rec})
    assert not violations, (
        "ref_df fell below the term EDF (edf1 >= edf invariant broken) — the "
        "degenerate Wood tr(F)^2/tr(F^2) collapse is back. Offenders: "
        + "; ".join(
            f"(seed={int(v['seed'])}, term_edf={v['edf']:.4f}, ref_df={v['ref_df']:.3g})"
            for v in violations
        )
    )


def test_null_false_positive_rate_is_calibrated(
    coverage_audit: Callable[[int, int, float], Any],
    coverage_replications: Callable[[float], int],
) -> None:
    # Pure-noise responses: there is no smooth effect, so a calibrated test
    # rejects at exactly ALPHA. The non-rejection rate is audited at nominal
    # 1 - ALPHA with the shared two-sided Wilson verdict, over the smallest seed
    # count at which a test that never rejects can fail it.
    n_seeds = coverage_replications(1.0 - ALPHA)
    rejections = 0
    ps: list[float] = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        x = np.linspace(0.0, 1.0, N)
        y = rng.standard_normal(N)  # pure noise
        rec = _record(x, y)
        ps.append(rec["p_corrected"])
        if rec["p_corrected"] < ALPHA:
            rejections += 1
    # U(0, 1) has mean 1/2 and variance 1/12, so the mean of R null p-values
    # has sd sqrt(1/(12 R)). Outside 4 sd the p-values are not uniform: above
    # means conservative, below anti-conservative. This is the distribution's
    # own shape; the Wilson verdict below is the rejection RATE at ALPHA, and a
    # test can fail either without failing the other.
    mean_p = float(np.mean(ps))
    mean_sd = (1.0 / (12.0 * n_seeds)) ** 0.5
    assert abs(mean_p - 0.5) <= 4.0 * mean_sd, (
        f"null p-values are not U(0, 1): mean p = {mean_p:.3f} over {n_seeds} "
        f"pure-noise fits, outside 1/2 +- 4 sd = +-{4.0 * mean_sd:.3f} "
        f"({'conservative' if mean_p > 0.5 else 'anti-conservative'}); "
        f"sorted p = {[round(p, 3) for p in sorted(ps)]}"
    )
    verdict = coverage_audit(n_seeds - rejections, n_seeds, 1.0 - ALPHA)
    assert verdict.passed, (
        f"null false-positive rate {rejections / n_seeds:.3f} ({rejections}/{n_seeds}) "
        f"at alpha={ALPHA} is miscalibrated; non-rejection {verdict.describe()}"
    )


def test_strong_signal_still_flagged() -> None:
    rng = np.random.default_rng(100)
    xp = rng.uniform(0.0, 1.0, 300)
    yp = np.sin(8.0 * xp) + 0.3 * rng.standard_normal(300)
    rec = _record(xp, yp)
    assert rec["p_corrected"] < 1e-3, (
        "power control: a strong wiggly smooth was not flagged "
        f"(W={rec['W']:.3g}, edf={rec['edf']:.3f}, ref_df={rec['ref_df']:.3g}, "
        f"p={rec['p_corrected']:.3g})"
    )
