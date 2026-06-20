"""Per-smooth-term significance test has ~100% Type-I error for an irrelevant
smooth when a strong co-term is present.

``summary().smooth_terms[j]`` reports a Wood (2013) rank-truncated Wald test
for each smooth: a statistic ``chi_sq``, reference ``ref_df``, and a
``p_value`` testing H0 "this smooth is zero".  The Rust implementation is
``wood_smooth_test`` in ``src/inference/smooth_test.rs``; it slices the term's
coefficient block ``beta[coeff_range]`` and the matching block of the
``cov_forwald`` covariance (``src/main/model_summary.rs:23`` —
``fit.beta_covariance_corrected().or(fit.beta_covariance())``).

The defect
----------
Fit ``y ~ s(x) + s(z)`` where ``y`` depends only on ``x`` (a strong signal)
and ``z`` is drawn *independently* of ``y`` (pure noise).  The engine
**correctly** shrinks ``s(z)``: its EDF lands around 0.65 (below 1 — i.e. the
term is penalised to essentially nothing, with a fitted smoothing parameter
``lambda_z ~ 1e13`` parking the term on the penalty boundary).  Yet the
reported ``p_value`` for ``s(z)`` is ``0.0`` on essentially every draw — a
~100% false-positive rate at the 5% level, where a calibrated test of a
genuinely-null term must reject ~5% of the time.

This is internally contradictory two ways:

* against the term's own EDF — a smooth shrunk *below one* effective degree of
  freedom cannot also be "overwhelmingly different from zero";
* against the model's own exposed covariance — reconstructing the Wald
  quadratic from ``summary().coefficients_frame()`` and
  ``summary().covariance_flat`` (which is the *corrected* covariance, with
  ``std_error == sqrt(diag(cov))`` exactly) gives a statistic of ~15 spread
  over ~2 reference d.f., with every coefficient z-score ``|beta/se| < 0.6``
  — clearly non-significant.  The engine instead reports ``chi_sq ~ 84``,
  about 5-6x too large.

The bug only fires when a *strong* co-term is in the model: a single
irrelevant ``y ~ s(z)`` on pure-noise ``y``, or two weak terms, both give
well-calibrated p-values.  The strong ``s(x)`` drives the residual dispersion
small and pushes ``s(z)`` to the penalty boundary (``lambda_z`` enormous),
exactly the regime where the conditional (frozen-lambda) covariance is
unreliable and the smoothing-parameter-uncertainty correction is essential.
The statistic the test actually uses behaves as if it lacks that correction
for the boundary term, so a near-zero point estimate is judged against a
covariance block far smaller than the one the model reports to users.

This is distinct from the EDF-inflation family (e.g. #1266): there the EDF
itself was wrong; here the EDF is *correct* (s(z) is properly shrunk) and the
significance test is wrong on top of a correct fit.

What this test asserts
----------------------
Over a batch of independent seeds, with a positive power control:

* the irrelevant ``s(z)`` (pure noise) must NOT be flagged significant on
  (nearly) every draw — its rejection rate at the 5% level must stay well
  below 0.5 (a calibrated test sits near 0.05);
* the relevant ``s(x)`` must still be detected (rejection rate high), so a
  trivial "make every p-value large" change cannot satisfy the test.

Currently the s(z) rejection rate is 1.0 (100%), so the calibration assertion
fails.  When the term test is fixed it should drop to ~0.05 while the s(x)
power assertion keeps passing.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


N_SEEDS = 25
N = 600
ALPHA = 0.05


def _smooth_pvalues(seed: int) -> dict[str, dict[str, float]]:
    """Fit ``y ~ s(x) + s(z)`` where y depends only on x and z is pure noise,
    independent of y. Return {term_name: {'p_value':..., 'edf':...}}."""
    rng = np.random.default_rng(20_000 + seed)
    x = rng.uniform(0.0, 1.0, N)
    z = rng.uniform(0.0, 1.0, N)  # drawn independently of y -> genuinely null
    y = np.sin(3.0 * x) + 0.3 * rng.standard_normal(N)
    data = {"x": x, "z": z, "y": y}
    model = gamfit.fit(data, "y ~ s(x) + s(z)")
    out: dict[str, dict[str, float]] = {}
    for term in model.summary().smooth_terms:
        out[term["name"]] = {
            "p_value": float(term["p_value"]),
            "edf": float(term["edf"]),
        }
    return out


def test_irrelevant_smooth_term_pvalue_is_calibrated_with_a_strong_coterm() -> None:
    reject_z = 0
    reject_x = 0
    edfs_z: list[float] = []
    pvals_z: list[float] = []
    for seed in range(N_SEEDS):
        terms = _smooth_pvalues(seed)
        assert "s(z)" in terms and "s(x)" in terms, terms.keys()
        pz = terms["s(z)"]["p_value"]
        px = terms["s(x)"]["p_value"]
        edfs_z.append(terms["s(z)"]["edf"])
        pvals_z.append(pz)
        reject_z += int(pz < ALPHA)
        reject_x += int(px < ALPHA)

    rate_z = reject_z / N_SEEDS
    rate_x = reject_x / N_SEEDS

    # The engine genuinely shrinks the noise term: its EDF is small (~<2), i.e.
    # the *fit* correctly judges s(z) to be near-null. The pre-condition guards
    # that we are in that regime (a correct fit with an over-confident test),
    # not testing an over-fit term.
    assert np.median(edfs_z) < 3.0, (
        "precondition: the irrelevant smooth should be shrunk toward null; "
        f"median EDF was {np.median(edfs_z):.3f}"
    )

    # Power control: the genuinely-relevant s(x) must be detected on (nearly)
    # every draw. This keeps the calibration check below well-posed — a fix that
    # merely inflated every p-value would fail here.
    assert rate_x >= 0.8, (
        f"s(x) carries a strong signal but was detected on only {rate_x:.2%} "
        "of draws; the term test has lost power"
    )

    # Calibration: a smooth of a covariate drawn independently of the response
    # is null, so its p-value is ~Uniform(0,1) and the rejection rate at the 5%
    # level is ~0.05. A rate at or near 1.0 is a systematic false positive.
    assert rate_z <= 0.5, (
        f"s(z) is pure noise (independent of y) yet was flagged significant at "
        f"alpha={ALPHA} on {rate_z:.0%} of {N_SEEDS} independent draws "
        f"(median p-value {np.median(pvals_z):.3g}, median EDF "
        f"{np.median(edfs_z):.3f}). A calibrated per-term test rejects ~5%; "
        "the smooth-term Wald statistic is inflated when a strong co-term is "
        "present (see src/inference/smooth_test.rs and the cov_forwald "
        "selection in src/main/model_summary.rs)."
    )
