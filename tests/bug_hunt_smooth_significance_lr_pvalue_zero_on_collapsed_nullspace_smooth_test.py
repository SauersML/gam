"""``Model.smooth_significance()`` reports p ~= 0 (overwhelming significance)
for a penalized smooth that has collapsed onto its *linear null space* and
explains essentially nothing — a flagrant Type-I error driven by a degenerate
reference degrees-of-freedom.

What ``smooth_significance`` does (#1063)
-----------------------------------------
For each penalized smooth it computes a genuine likelihood-ratio statistic
``W = 2(loglik_full - loglik_null)`` by refitting with the *whole* smooth term
dropped, then references ``W`` against ``chi^2_d`` with ``d = ref_df`` (the
Wood truncation ``tr(F)^2 / tr(F^2)`` of the term's influence block).  The
implementation is ``smooth_term_lr_inference_forspec`` in
``crates/gam-models/src/fit_orchestration/drivers/spatial_optimization.rs``
(see lines ~8602-8645): ``ref_df = wood_reference_df(influence, &coeff_range)``
and ``p = 1 - ChiSquared::new(ref_df).cdf(W)``.

The defect
----------
A default ``s(x)`` carries an *unpenalized* linear null space (one degree of
freedom).  When REML shrinks the wiggly part to zero the term collapses onto
exactly that line: its effective d.f. ``edf -> 1.0``.  At that point
``wood_reference_df`` — which measures only the *penalized* (wiggle) influence —
collapses to ~0, while the LR statistic ``W`` still compares against dropping
the *entire* term (linear part included).  Referencing a positive ``W`` against
``chi^2_{~0}`` yields ``p -> 0``: the survival function of a chi-square whose
d.f. tends to zero is ~0 at every positive argument.

So a smooth that has been judged by the fit itself to be a flat line, and whose
removal changes the log-likelihood by essentially nothing (``W`` on the order of
``1e-4``), is reported as maximally significant.  The contradiction is internal
and visible at a single fit:

    near-flat response, n=200, np.random.default_rng(seed):
      seed 0:  edf=1.111  W=0.0000  ref_df=1.20      p=0.9991   (correct)
      seed 1:  edf=1.000  W=0.0002  ref_df=1.0e-12   p=4.3e-12  (FALSE POSITIVE)
      seed 2:  edf=1.000  W=0.0001  ref_df=1.0e-12   p=4.5e-12  (FALSE POSITIVE)
      seed 5:  edf=1.000  W=0.0000  ref_df=1.0e-12   p=6.2e-12  (FALSE POSITIVE)

The same near-zero ``W`` gives p=0.999 when ``edf`` lands just above 1 (ref_df>1)
and p~=4e-12 the moment ``edf`` hits exactly 1.0 (ref_df collapses) — the verdict
is decided entirely by the degenerate reference d.f., not by the data.

The term's LR statistic tests "smooth present vs. entirely absent", so the
reference d.f. must be at least the term's own effective d.f. (>= the 1-d.f.
linear null space).  A correct fix references ``W`` against ``d ~ edf >= 1``,
giving ``p ~ chi^2_1.sf(W) ~ 1`` for ``W ~ 0``.

Distinct from #1360 / the strong-coterm Wald test
-------------------------------------------------
#1360 (closed) and ``bug_hunt_smooth_term_pvalue_false_positive_with_strong_coterm``
concern the *summary* path's Wood rank-truncated **Wald** statistic
(``summary().smooth_terms[j]["p_value"]``, ``wood_smooth_test`` in
``src/inference/smooth_test.rs``), require a *strong co-term* to fire, and are
caused by an over-confident covariance block.  This bug is in the separate
**likelihood-ratio** path introduced by #1063
(``Model.smooth_significance()`` -> ``smooth_term_lr_inference_forspec``), needs
*no* co-term (a single ``s(x)`` on a flat response), and is caused by a
degenerate ``ref_df`` rather than a covariance scale.

What this test asserts
----------------------
* Null: for near-flat responses (no trend, no wiggle) the LR statistic ``W`` is
  ~0, so the smooth is the least-significant possible.  Its corrected and
  uncorrected p-values must therefore be large (> 0.05), never ~0.  Several
  independent seeds collapse ``edf`` to exactly 1.0 and currently report
  ``p ~ 1e-12``; the assertion fails until the reference d.f. is fixed.
* Power: a strong wiggly signal must still be flagged (``p < 1e-3``), so a fix
  cannot trivially inflate every p-value.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


N = 200
N_SEEDS = 12
ALPHA = 0.05


def _lr_record(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    model = gamfit.fit({"x": list(x), "y": list(y)}, "y ~ s(x)")
    edf = float(model.summary().edf_total)
    sig = model.smooth_significance({"x": list(x), "y": list(y)})
    assert sig, "expected one penalized-smooth significance record"
    rec = sig[0]
    return {
        "edf": edf,
        "W": float(rec["statistic_lr"]),
        "ref_df": float(rec["ref_df"]),
        "p_corrected": float(rec["p_value_corrected"]),
        "p_uncorrected": float(rec["p_value_uncorrected"]),
    }


def test_collapsed_nullspace_smooth_lr_pvalue_is_not_falsely_significant() -> None:
    # ── Null calibration ────────────────────────────────────────────────
    # Near-constant responses: no trend and no wiggle, so REML collapses the
    # smooth onto (at most) its 1-d.f. linear null space and dropping the whole
    # term barely changes the likelihood (W ~ 0). A smooth with a near-zero LR
    # statistic is the LEAST significant possible; its p-value must be large.
    offenders: list[dict[str, float]] = []
    checked = 0
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        x = np.linspace(0.0, 1.0, N)
        y = 0.01 * rng.standard_normal(N)  # essentially constant
        rec = _lr_record(x, y)

        # Pre-condition: we are genuinely in the null regime for this seed —
        # the term explains essentially nothing. (Holds for every seed here.)
        if rec["W"] >= 1.0:
            continue
        checked += 1
        # A chi-square statistic below 1.0 is non-significant under any
        # reasonable reference d.f. (chi^2_1.sf(1.0) = 0.317); the corrected and
        # uncorrected p-values must both exceed ALPHA.
        if rec["p_corrected"] < ALPHA or rec["p_uncorrected"] < ALPHA:
            offenders.append({"seed": float(seed), **rec})

    assert checked >= 6, (
        "precondition: expected most near-flat fits to yield a tiny LR "
        f"statistic, but only {checked}/{N_SEEDS} did"
    )
    assert not offenders, (
        "smooth_significance() flagged a collapsed-to-null smooth as significant "
        "despite a near-zero likelihood-ratio statistic — Type-I error from a "
        "degenerate reference d.f. (ref_df -> 0 when edf hits the 1-d.f. linear "
        "null space, while W tests dropping the whole term). Offending fits "
        "(seed, edf, W, ref_df, p): "
        + "; ".join(
            f"(seed={int(o['seed'])}, edf={o['edf']:.4f}, W={o['W']:.4g}, "
            f"ref_df={o['ref_df']:.3g}, p_corr={o['p_corrected']:.3g})"
            for o in offenders
        )
        + ". See smooth_term_lr_inference_forspec in "
        "crates/gam-models/src/fit_orchestration/drivers/spatial_optimization.rs "
        "(ref_df = wood_reference_df(...); p = 1 - ChiSquared::new(ref_df).cdf(W))."
    )

    # ── Power control ───────────────────────────────────────────────────
    # A genuinely wiggly signal must still be detected, so a fix cannot simply
    # inflate every p-value to escape the calibration assertion above.
    rng = np.random.default_rng(100)
    xp = rng.uniform(0.0, 1.0, 300)
    yp = np.sin(8.0 * xp) + 0.3 * rng.standard_normal(300)
    rec = _lr_record(xp, yp)
    assert rec["p_corrected"] < 1e-3, (
        "power control: a strong wiggly smooth was not flagged significant "
        f"(W={rec['W']:.3g}, edf={rec['edf']:.3f}, p={rec['p_corrected']:.3g}); "
        "the LR significance test has lost power"
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    test_collapsed_nullspace_smooth_lr_pvalue_is_not_falsely_significant()
    print("ok")
