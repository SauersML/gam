"""Regression test: the model-comparison entry points must rank fits on ONE score.

Three public APIs all claim to expose the same REML/LAML marginal-likelihood
quantity for a fitted model:

* ``Model.evidence`` (now ``Summary.aic_corrected``) / ``Summary.reml_score``
  — the model's own reported score.
* ``Model.bayes_factor_vs`` — pairwise Bayes factor between two fits.
* ``gamfit.compare_models`` — the multi-model comparison table.

They did NOT agree. ``compare_models`` (via the since-removed ``compare_reml_fits`` in
``crates/gam-pyffi/src/lib.rs`` -> ``extract_reml_score_from_view``, which wraps
the raw score with ``with_tierney_kadane_normalizer_from_view``) ranks fits on a
*rank-aware Tierney-Kadane normalized* score, namely

    raw_reml + (-0.5 * null_dim * ln(2*pi) + 0.5 * null_space_logdet),

while ``Model.evidence`` / ``Summary.reml_score`` (``model_evidence``, now
since removed,
``crates/gam-pyffi/src/lib.rs``) and ``Model.bayes_factor_vs``
(``bayes_factor_log_diff``, now ``log_evidence_ratio``) used the *raw* minimized
``reml_score`` with no normalizer.

The normalizer term cancels in a delta only when both models share the same
penalty null-space dimension. When the null-space dimensions differ (e.g. a
penalized smooth ``s(x)`` with ``null_dim >= 1`` versus a purely parametric
polynomial with ``null_dim == 0``) the two code paths disagree, so the Bayes
factor reported by ``compare_models`` differs from the one reported by
``bayes_factor_vs`` for the very same pair of fits.

This is a sibling of #575 (which fixed only the *direction* of
``bayes_factor_vs`` and is exercised by
``test_bug_hunt_bayes_factor_vs_inverted_sign.py`` on far-apart models that do
not stress the normalizer). The fix is direction-agnostic: route all three
entry points through the same score. These assertions only require *consistency*
between the paths, so they pass whichever score the maintainer settles on.

Update (#2079, pyGAM audit): ``Model.bayes_factor_vs`` (now
``Model.evidence_ratio_vs``) and the ``compare_models`` ``ranking`` are both on the
smoothing-corrected AIC ``Summary.aic_corrected`` — the ranking's ``delta_aic``
column is each model's ``aic_corrected`` gap from the winner. The raw
``score_table['reml_score']`` headline, along with ``Summary.reml_score``, still
reports the REML/LAML score and is checked separately for its own
self-consistency.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import gamfit

# The relative agreement the score paths must reach. A normalizer that one path
# skips has to lie outside it, or these tests cannot tell the paths apart.
SCORE_REL = 1e-9


def _fit_pair() -> tuple["gamfit.Model", "gamfit.Model"]:
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(-3, 3, n)
    y = np.sin(1.5 * x) + rng.normal(scale=0.5, size=n)
    data = dict(y=y, x=x, x2=x**2, x3=x**3, x4=x**4, x5=x**5)
    # Penalized smooth.
    m_smooth = gamfit.fit(data, "y ~ s(x)", family="gaussian")
    # Parametric polynomial. Since the empty-penalty normalization (#2627,
    # 192b2c1ae1) it publishes the same null-space dimension as the smooth, 1 on
    # this fixture.
    m_poly = gamfit.fit(data, "y ~ x + x2 + x3 + x4 + x5", family="gaussian")
    return m_smooth, m_poly


def _compare_scores() -> dict[str, float]:
    m_smooth, m_poly = _fit_pair()
    comparison = gamfit.compare_models([m_smooth, m_poly], names=["smooth", "poly"])
    return {row["name"]: row["reml_score"] for row in comparison["score_table"]}


def test_compare_models_precondition_normalizer_is_resolvable() -> None:
    # Establish the precondition that keeps the consistency test below non-vacuous.
    # Each fit's comparable score carries the Tierney-Kadane normalizer
    # -0.5 * null_dim * ln(2*pi) + 0.5 * null_space_logdet, and that normalizer lies
    # outside the tolerance the paths must agree to, so a path that skipped it would
    # disagree. This used to require different null dimensions (poly 0, smooth >= 1).
    # Since 192b2c1ae1 both fits publish null_dim 1, and the normalizer itself, not a
    # null-dimension gap, is what distinguishes the paths.
    for model in _fit_pair():
        summary = model.summary()
        normalizer = (
            -0.5 * summary.null_dim * math.log(2.0 * math.pi)
            + 0.5 * summary.null_space_logdet
        )
        assert summary.reml_score - summary.raw_reml_score == pytest.approx(
            normalizer, rel=SCORE_REL
        )
        assert abs(normalizer) > SCORE_REL * abs(summary.reml_score)


def test_compare_models_score_matches_model_own_score() -> None:
    # The raw ``reml_score`` that compare_models reports for a model must equal
    # the model's own reported raw score (score_table['reml_score'] ==
    # Summary.reml_score) -- the un-penalised REML/LAML evidence headline path
    # must stay self-consistent regardless of the normalizer.
    m_smooth, m_poly = _fit_pair()
    comparison = gamfit.compare_models([m_smooth, m_poly], names=["smooth", "poly"])
    compare_score = {row["name"]: row["reml_score"] for row in comparison["score_table"]}

    # Both fits carry a resolvable normalizer (see the precondition test), so each
    # assertion catches a path that applies it in only one place. Pre-fix,
    # compare_models added the Tierney-Kadane normalizer (~1.9 nats) that Summary
    # omitted.
    assert compare_score["poly"] == pytest.approx(
        m_poly.summary().reml_score, rel=SCORE_REL
    )
    assert compare_score["smooth"] == pytest.approx(
        m_smooth.summary().reml_score, rel=SCORE_REL
    ), (
        "compare_models score "
        f"{compare_score['smooth']!r} disagrees with the model's own "
        f"reml_score {m_smooth.summary().reml_score!r} (TK normalizer applied "
        "in only one path)"
    )

    # Summary.aic_corrected is the RANKING score, which compare_models exposes
    # as the ``ranking`` ``delta_aic`` (each model's ranking score minus the
    # winner's). So corrected-AIC differences must reproduce the ranking deltas
    # exactly -- the two are on one score.
    ranking = {row["name"]: row for row in comparison["ranking"]}
    aic = {
        "smooth": m_smooth.summary().aic_corrected,
        "poly": m_poly.summary().aic_corrected,
    }
    winner_aic = aic[comparison["winner"]]
    for name in ("smooth", "poly"):
        ranking_delta = ranking[name]["delta_aic"]
        assert ranking_delta == pytest.approx(
            aic[name] - winner_aic, rel=1e-9, abs=1e-9
        ), (
            f"compare_models ranking delta {ranking_delta!r} for {name!r} "
            f"disagrees with the Summary.aic_corrected gap {aic[name] - winner_aic!r}"
        )


def test_bayes_factor_vs_agrees_with_compare_models_magnitude() -> None:
    # User-facing consequence: the log Bayes factor between two fits must be the
    # same whether read from bayes_factor_vs or implied by the compare_models
    # ranking deltas. Both are on the corrected-AIC ranking score.
    m_smooth, m_poly = _fit_pair()
    comparison = gamfit.compare_models([m_smooth, m_poly], names=["smooth", "poly"])
    ranking = {row["name"]: row for row in comparison["ranking"]}

    # The ranking ``delta_aic`` is each model's corrected-AIC gap from the winner
    # (a -2*log / deviance-scale quantity), so the log Bayes factor of poly over
    # smooth implied by compare_models is HALF the delta gap: the Akaike evidence
    # ratio for an AIC gap D is exp(-D/2) (Burnham & Anderson), not exp(-D)
    # (issue #2124). The winner term cancels in the difference. bayes_factor_vs is
    # a minimized cost (lower = better), matching that direction.
    log_bf_poly_over_smooth_compare = 0.5 * (
        ranking["smooth"]["delta_aic"] - ranking["poly"]["delta_aic"]
    )
    log_bf_poly_over_smooth_pairwise = math.log(m_poly.evidence_ratio_vs(m_smooth))

    assert log_bf_poly_over_smooth_pairwise == pytest.approx(
        log_bf_poly_over_smooth_compare, rel=1e-6, abs=1e-6
    ), (
        "bayes_factor_vs and compare_models report Bayes factors that differ by "
        f"{log_bf_poly_over_smooth_compare - log_bf_poly_over_smooth_pairwise:.4f} "
        "nats for the same pair of fits"
    )
