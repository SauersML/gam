"""Regression test for #575: ``Model.bayes_factor_vs`` sign must not be inverted.

The bug: ``bayes_factor_vs`` returned the *reciprocal* of the correct Bayes
factor, favouring the worse-fitting model. On data with obvious signal,
``m_sx.bayes_factor_vs(m_null)`` returned a number like ``6.8e-220`` ("no
support for the good model") while ``gamfit.compare_models`` on the same two
fits correctly declared the smooth model the winner by ``~1e+219``. The two
entry points contradicted each other.

Root cause: ``reml_score`` is a minimised cost (lower = better marginal
likelihood). The Python FFI computed ``self.reml_score - other.reml_score``,
which is negative when ``self`` is the better (lower-cost) model, so its
exponential fell below 1. The comparator's convention is the opposite
subtraction. Both now route through ``evidence::log_bayes_factor``.

This test anchors that ``s(x)`` is the better fit, then asserts the Bayes
factor direction and that ``bayes_factor_vs`` agrees with ``compare_models``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit


def _fit_pair() -> tuple["gamfit.Model", "gamfit.Model"]:
    rng = np.random.default_rng(0)
    n = 600
    x = np.sort(rng.uniform(-3, 3, n))
    y = np.sin(2 * x) + 0.5 * x + 0.4 * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "y": y})
    m_sx = gamfit.fit(df, "y ~ s(x)", family="gaussian")
    m_null = gamfit.fit(df, "y ~ 1", family="gaussian")
    return m_sx, m_null


def test_bayes_factor_vs_favours_better_model_not_worse() -> None:
    m_sx, m_null = _fit_pair()

    # Anchor: the smooth model is the better fit. `evidence` is a minimised
    # cost, so the better model has the *lower* value.
    assert m_sx.evidence < m_null.evidence

    # And `compare_models` agrees the smooth model wins.
    comparison = gamfit.compare_models([m_sx, m_null], names=["sx", "null"])
    assert comparison["winner"] == "sx"

    # The core assertion: the Bayes factor of the better model over the worse
    # one must be > 1 (pre-fix it was ~6.8e-220).
    bf_good_over_null = m_sx.bayes_factor_vs(m_null)
    assert bf_good_over_null > 1.0, (
        f"bayes_factor_vs favoured the worse model: {bf_good_over_null!r}"
    )

    # ... and the worse model's Bayes factor over the better one must be < 1.
    bf_null_over_good = m_null.bayes_factor_vs(m_sx)
    assert bf_null_over_good < 1.0, (
        f"worse model's bayes_factor_vs favoured itself: {bf_null_over_good!r}"
    )

    # Reciprocity on the log scale: BF(a over b) == 1 / BF(b over a).
    assert bf_good_over_null == pytest.approx(1.0 / bf_null_over_good, rel=1e-9)


def test_bayes_factor_vs_agrees_with_compare_models_winner() -> None:
    # A different angle on the same root cause: whichever model `compare_models`
    # declares the winner must be exactly the one whose `bayes_factor_vs(loser)`
    # exceeds 1. A sign flip in either entry point breaks this agreement.
    m_sx, m_null = _fit_pair()
    models = {"sx": m_sx, "null": m_null}
    comparison = gamfit.compare_models([m_sx, m_null], names=["sx", "null"])
    winner_name = comparison["winner"]
    loser_name = "null" if winner_name == "sx" else "sx"

    winner, loser = models[winner_name], models[loser_name]
    assert winner.bayes_factor_vs(loser) > 1.0
    assert loser.bayes_factor_vs(winner) < 1.0


def test_bayes_factor_vs_of_model_against_itself_is_one() -> None:
    # Edge case: identical fits are indistinguishable, so the Bayes factor is
    # exactly 1 (log BF = score - score = 0). Guards against an off-by-sign or
    # off-by-constant that would still pass the strictly-ordered cases above.
    m_sx, _ = _fit_pair()
    assert m_sx.bayes_factor_vs(m_sx) == pytest.approx(1.0, abs=1e-12)
