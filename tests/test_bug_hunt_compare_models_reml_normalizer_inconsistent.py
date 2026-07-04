"""Regression test: the model-comparison entry points must rank fits on ONE score.

Three public APIs all claim to expose the same REML/LAML marginal-likelihood
quantity for a fitted model:

* ``Model.evidence`` / ``Summary.reml_score`` — the model's own reported score.
* ``Model.bayes_factor_vs`` — pairwise Bayes factor between two fits.
* ``gamfit.compare_models`` — the multi-model comparison table.

They do NOT agree. ``compare_models`` (via ``compare_reml_fits`` in
``crates/gam-pyffi/src/lib.rs`` -> ``extract_reml_score_from_view``, which wraps
the raw score with ``with_tierney_kadane_normalizer_from_view``) ranks fits on a
*rank-aware Tierney-Kadane normalized* score, namely

    raw_reml + (-0.5 * null_dim * ln(2*pi) + 0.5 * null_space_logdet),

while ``Model.evidence`` / ``Summary.reml_score`` (``model_evidence``,
``crates/gam-pyffi/src/lib.rs``) and ``Model.bayes_factor_vs``
(``bayes_factor_log_diff`` -> ``log_bayes_factor``) use the *raw* minimized
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

Update (#2079): the maintainer settled ``Model.evidence`` and
``Model.bayes_factor_vs`` on the Occam-penalised conditional-AIC *ranking* score
(``-2*loglik + 2*edf``) that ``compare_models`` ranks its ``winner`` on — so
that the two per-model APIs can no longer contradict the declared winner when a
model is augmented with a pure-noise smooth. That ranking score is exposed by
``compare_models`` through the ``ranking`` list (its ``delta`` / Bayes-factor
columns), NOT through the raw ``score_table['reml_score']`` headline (which,
along with ``Summary.reml_score``, still reports the un-penalised REML/LAML
evidence). The consistency assertions below therefore compare ``evidence`` /
``bayes_factor_vs`` against the ``ranking`` columns; the raw ``score_table`` /
``Summary.reml_score`` path is checked separately for its own self-consistency.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import gamfit


def _fit_pair() -> tuple["gamfit.Model", "gamfit.Model"]:
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(-3, 3, n)
    y = np.sin(1.5 * x) + rng.normal(scale=0.5, size=n)
    data = dict(y=y, x=x, x2=x**2, x3=x**3, x4=x**4, x5=x**5)
    # Penalized smooth: penalty null-space dimension >= 1.
    m_smooth = gamfit.fit(data, "y ~ s(x)", family="gaussian")
    # Purely parametric: no penalty, null-space dimension 0.
    m_poly = gamfit.fit(data, "y ~ x + x2 + x3 + x4 + x5", family="gaussian")
    return m_smooth, m_poly


def _compare_scores() -> dict[str, float]:
    m_smooth, m_poly = _fit_pair()
    comparison = gamfit.compare_models([m_smooth, m_poly], names=["smooth", "poly"])
    return {row["name"]: row["reml_score"] for row in comparison["score_table"]}


def test_compare_models_precondition_null_dims_differ() -> None:
    # Establish the precondition that makes the normalizer matter: the two fits
    # have different penalty null-space dimensions. If this ever stops holding,
    # the consistency test below would be vacuous.
    m_smooth, m_poly = _fit_pair()
    assert m_poly.summary().null_dim == 0
    assert m_smooth.summary().null_dim >= 1


def test_compare_models_score_matches_model_own_score() -> None:
    # The raw ``reml_score`` that compare_models reports for a model must equal
    # the model's own reported raw score (score_table['reml_score'] ==
    # Summary.reml_score) -- the un-penalised REML/LAML evidence headline path
    # must stay self-consistent regardless of the normalizer.
    m_smooth, m_poly = _fit_pair()
    comparison = gamfit.compare_models([m_smooth, m_poly], names=["smooth", "poly"])
    compare_score = {row["name"]: row["reml_score"] for row in comparison["score_table"]}

    # The null_dim == 0 model is the control: its normalizer is zero, so the two
    # paths already agree here. This anchors that we are comparing like with like.
    assert compare_score["poly"] == pytest.approx(m_poly.summary().reml_score, rel=1e-9)

    # The real assertion: the penalized smooth (null_dim >= 1) must also report
    # the same raw score in both places. Pre-fix, compare_models added the
    # Tierney-Kadane normalizer (~1.9 nats) that Summary omitted.
    assert compare_score["smooth"] == pytest.approx(
        m_smooth.summary().reml_score, rel=1e-9
    ), (
        "compare_models score "
        f"{compare_score['smooth']!r} disagrees with the model's own "
        f"reml_score {m_smooth.summary().reml_score!r} (TK normalizer applied "
        "in only one path)"
    )

    # #2079: Model.evidence reports the conditional-AIC RANKING score, which
    # compare_models exposes as the ``ranking`` ``delta`` (each model's ranking
    # score minus the winner's). So evidence differences must reproduce the
    # ranking deltas exactly -- the two are on one score.
    ranking = {row[0]: row for row in comparison["ranking"]}
    winner = comparison["winner"]
    winner_evidence = {"smooth": m_smooth.evidence, "poly": m_poly.evidence}[winner]
    for name, model in (("smooth", m_smooth), ("poly", m_poly)):
        ranking_delta = ranking[name][2]
        assert ranking_delta == pytest.approx(
            model.evidence - winner_evidence, rel=1e-9, abs=1e-9
        ), (
            f"compare_models ranking delta {ranking_delta!r} for {name!r} "
            f"disagrees with Model.evidence gap {model.evidence - winner_evidence!r}"
        )


def test_bayes_factor_vs_agrees_with_compare_models_magnitude() -> None:
    # User-facing consequence: the log Bayes factor between two fits must be the
    # same whether read from bayes_factor_vs or implied by the compare_models
    # ranking deltas. Both are on the conditional-AIC ranking score (#2079).
    m_smooth, m_poly = _fit_pair()
    comparison = gamfit.compare_models([m_smooth, m_poly], names=["smooth", "poly"])
    ranking = {row[0]: row for row in comparison["ranking"]}

    # The ranking ``delta`` is each model's conditional-AIC gap from the winner
    # (a -2*log / deviance-scale quantity), so the log Bayes factor of poly over
    # smooth implied by compare_models is HALF the delta gap: the Akaike evidence
    # ratio for an AIC gap D is exp(-D/2) (Burnham & Anderson), not exp(-D)
    # (issue #2124). The winner term cancels in the difference. bayes_factor_vs is
    # a minimized cost (lower = better), matching that direction.
    log_bf_poly_over_smooth_compare = 0.5 * (ranking["smooth"][2] - ranking["poly"][2])
    log_bf_poly_over_smooth_pairwise = math.log(m_poly.bayes_factor_vs(m_smooth))

    assert log_bf_poly_over_smooth_pairwise == pytest.approx(
        log_bf_poly_over_smooth_compare, rel=1e-6, abs=1e-6
    ), (
        "bayes_factor_vs and compare_models report Bayes factors that differ by "
        f"{log_bf_poly_over_smooth_compare - log_bf_poly_over_smooth_pairwise:.4f} "
        "nats for the same pair of fits"
    )
