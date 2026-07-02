"""Regression test for issue #2079.

``Model.evidence`` and ``Model.bayes_factor_vs`` document AGREEMENT with
``gamfit.compare_models``, but historically they contradicted it. ``compare_models``
ranks on an Occam-penalised conditional AIC (``-2*loglik + 2*edf``), while
``bayes_factor_vs`` / ``evidence`` used the RAW REML/LAML score. On a model
augmented with a near-null (pure-noise) smooth the two methods pick OPPOSITE
winners: ``compare_models`` correctly prefers the smaller model (the noise
smooth spends effective degrees of freedom fitting nothing), yet the raw-REML
Bayes factor claimed the augmented model was better supported.

The fix routes ``evidence`` and ``bayes_factor_vs`` through the SAME conditional
AIC ranking score that ``compare_models`` uses, so they can no longer disagree
about which model wins.

This test builds the issue's DGP (a genuine ``s(x)`` signal plus a pure-noise
``s(z)`` term), confirms ``compare_models`` picks the small model, and asserts
that the Bayes factor and evidence ordering agree with that winner.
"""

import numpy as np

import gamfit


def test_bayes_factor_vs_and_evidence_agree_with_compare_models_winner():
    rng = np.random.default_rng(3006)
    n = 700
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = np.sin(3.0 * x) + 0.3 * rng.standard_normal(n)
    data = {"x": x, "z": z, "y": y}

    small = gamfit.fit(data, "y ~ s(x)")
    big = gamfit.fit(data, "y ~ s(x) + s(z)")

    comparison = gamfit.compare_models([small, big], names=["small", "big"])
    assert comparison["winner"] == "small"

    # If compare_models picks `small`, then the augmented model must NOT be
    # better supported than `small`, and `small` must have the lower (better)
    # evidence cost. Before the fix these contradicted the declared winner.
    assert big.bayes_factor_vs(small) <= 1.0
    assert small.evidence <= big.evidence
