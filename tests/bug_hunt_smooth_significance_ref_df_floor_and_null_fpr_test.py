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
   test rejects at ~alpha. The collapse drove the false-positive rate to
   ~0.23-0.35 at alpha=0.05 because every ``edf==1.0`` fit produced p~1e-12.
   We sweep many independent pure-noise fits and bound the empirical FPR.

A power control (a genuinely wiggly signal must still be flagged) guards
against a fix that simply inflates every p-value.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")

import gamfit


N = 200
ALPHA = 0.05


def _record(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    model = gamfit.fit({"x": list(x), "y": list(y)}, "y ~ s(x)")
    edf = float(model.summary().edf_total)
    rec = model.smooth_significance({"x": list(x), "y": list(y)})[0]
    return {
        "edf": edf,
        "W": float(rec["statistic_lr"]),
        "ref_df": float(rec["ref_df"]),
        "p_corrected": float(rec["p_value_corrected"]),
        "p_uncorrected": float(rec["p_value_uncorrected"]),
    }


def test_ref_df_never_below_term_edf_invariant() -> None:
    # The whole-term LR reference d.f. must satisfy ref_df >= edf for every fit,
    # including the near-flat fits whose smooth collapses to edf == 1.0 (where
    # the degenerate tr(F)^2/tr(F^2) used to crash ref_df to ~1e-12).
    violations: list[dict[str, float]] = []
    for seed in range(16):
        rng = np.random.default_rng(seed)
        x = np.linspace(0.0, 1.0, N)
        y = 0.01 * rng.standard_normal(N)  # essentially constant -> edf -> 1.0
        rec = _record(x, y)
        # A small numerical slack: ref_df should not sit materially below edf.
        if rec["ref_df"] < rec["edf"] - 1e-6:
            violations.append({"seed": float(seed), **rec})
    assert not violations, (
        "ref_df fell below the term EDF (edf1 >= edf invariant broken) — the "
        "degenerate Wood tr(F)^2/tr(F^2) collapse is back. Offenders: "
        + "; ".join(
            f"(seed={int(v['seed'])}, edf={v['edf']:.4f}, ref_df={v['ref_df']:.3g})"
            for v in violations
        )
    )


def test_null_false_positive_rate_is_calibrated() -> None:
    # Pure-noise responses: there is no smooth effect, so a calibrated test
    # should reject at roughly ALPHA. The collapse pushed the FPR to 0.23-0.35;
    # we require it well below that. The bound (0.15) is generous for the modest
    # seed count yet far under the buggy regime.
    n_seeds = 40
    rejections = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        x = np.linspace(0.0, 1.0, N)
        y = rng.standard_normal(N)  # pure noise
        rec = _record(x, y)
        if rec["p_corrected"] < ALPHA:
            rejections += 1
    fpr = rejections / n_seeds
    assert fpr <= 0.15, (
        f"null false-positive rate {fpr:.3f} ({rejections}/{n_seeds}) far exceeds "
        f"alpha={ALPHA}; the ref_df collapse is over-rejecting flat smooths"
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


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    test_ref_df_never_below_term_edf_invariant()
    test_null_false_positive_rate_is_calibrated()
    test_strong_signal_still_flagged()
    print("ok")
