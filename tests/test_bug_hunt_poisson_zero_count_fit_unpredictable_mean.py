"""#1515 contract: a model the public API reports as fitted must be predictable.

A Poisson (or negative-binomial) GAM fitted on identically-zero counts is accepted
by ``gamfit.fit`` — it returns a fitted ``Model`` — but the all-zero count response
makes the likelihood flat as every ``eta -> -inf``, so the penalized Hessian is
near-singular (per-coefficient ``se_eta`` in the thousands). The response-scale
posterior-mean integral ``E[exp(eta)] = exp(mu_eta + se_eta**2 / 2)`` then overflows
to ``+inf``, which serialized across the gam-pyffi boundary as JSON ``null`` and
surfaced in Python as a ``None`` ``mean`` column — so ``predict()`` crashed with
``TypeError: float() argument must be a string or a real number, not 'NoneType'``
even though ``linear_predictor`` (the floored, finite ``ln(1e-10)``) came back fine.

The fix (``predict_gam_posterior_mean_from_backendwith_bc`` in
``src/inference/predict/mod.rs``) degrades gracefully to the plug-in mean
``g^-1(eta_hat)`` whenever the posterior integral is non-finite, so a fitted model
always yields a finite response mean consistent with its linear predictor.
"""
from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _mean(out: object) -> np.ndarray:
    if isinstance(out, np.ndarray):
        return np.asarray(out, dtype=float).ravel()
    return np.asarray(out["mean"], dtype=float)


@pytest.mark.parametrize("formula", ["y ~ s(x)", "y ~ 1"])
@pytest.mark.parametrize("family", ["poisson", "negative_binomial"])
def test_all_zero_count_fit_predicts_finite_mean(formula: str, family: str) -> None:
    n = 200
    data = {"x": np.linspace(0.0, 1.0, n), "y": np.zeros(n)}

    model = gamfit.fit(data, formula, family=family)

    # Must NOT raise TypeError: float() ... not 'NoneType'.
    out = model.predict(data)
    mean = _mean(out)

    assert mean.shape[0] == n
    assert np.all(np.isfinite(mean)), (
        "all-zero-count fit produced a non-finite response mean "
        f"(formula={formula!r}, family={family!r}): {mean[:3]}"
    )
    # The MLE rate for all-zero counts is ~0, so the response mean is a small,
    # non-negative number (the plug-in floor exp(ln(1e-10)) = 1e-10), not a wild
    # overflow value.
    assert np.all(mean >= 0.0)
    assert np.all(mean < 1.0), f"expected near-zero rate, got {mean[:3]}"
