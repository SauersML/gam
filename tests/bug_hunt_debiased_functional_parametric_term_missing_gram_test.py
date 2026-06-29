"""Regression test: ``debiased_functional`` must work on Gaussian/identity models
that contain a **parametric** term (e.g. ``y ~ x`` or ``y ~ s(x) + z``).

``Model.debiased_functional`` documents its support as "restricted to
Gaussian/identity-link models". A plain ``y ~ x`` ordinary-least-squares fit and
a mixed ``y ~ s(x) + z`` fit both satisfy that precondition, yet every target
(``average_value``, ``average_derivative``, ``point``, ``contrast``) aborts with

    GamError: debiased_functional: model does not carry the weighted Gram X'WX;
    refit with a smaller basis (dense fits only)

The advice is impossible to act on — ``y ~ x`` is already the smallest possible
dense fit — so the estimator is simply unavailable for any model carrying a
parametric (non-intercept) covariate term. Pure-smooth (``y ~ s(x)``,
``y ~ s(x) + s(z)``) and intercept-only (``y ~ 1``) models work, which isolates
the trigger to the presence of a parametric term.

Root cause (read, not patched): the FFI handler reads
``saved_fit.weighted_gram()`` and errors when it is ``None``
(``crates/gam-pyffi/src/latent/reml_latent_fit_ffi.rs:3584``). The weighted Gram
``X'WX = H − S(λ)`` is only populated inside the posterior-covariance block of
the REML optimizer (``crates/gam-solve/src/estimate/optimizer.rs:2644-2646``);
the fit path taken once a parametric term is present leaves
``UnifiedFitResult::weighted_gram`` at ``None`` (its default, e.g.
``crates/gam-solve/src/model_types/result_types.rs:126``), so the instrument
can never recover ``X'WX`` even though it is trivially ``Xᵀdiag(w)X`` for a dense
Gaussian fit.

This test asserts that the plug-in ``average_value`` functional
``mean_i m(x_i)`` is returned for both a pure-parametric and a mixed model, and
that it equals the mean of the model's own training-row predictions (the exact
plug-in identity for Gaussian/identity). Both currently raise; they pass once a
parametric-containing Gaussian fit carries its weighted Gram.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    n = 600
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = 2.0 + 1.5 * x - 0.7 * z + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def _avg_value_matches_predict_mean(formula: str, df: pd.DataFrame) -> None:
    model = gamfit.fit(df, formula)
    # average_value plug-in == mean of the model's training-row predictions.
    expected = float(np.asarray(model.predict(df)).ravel().mean())

    res = model.debiased_functional(df, target="average_value")

    assert np.isfinite(res["theta_plugin"]), res
    assert abs(res["theta_plugin"] - expected) < 1e-5, (
        f"{formula}: average_value plug-in {res['theta_plugin']!r} != "
        f"mean(predict)={expected!r}"
    )
    assert np.isfinite(res["theta_debiased"]), res


def test_pure_parametric_gaussian_supports_debiased_functional() -> None:
    df = _data()
    _avg_value_matches_predict_mean("y ~ x", df)


def test_mixed_smooth_plus_parametric_supports_debiased_functional() -> None:
    df = _data()
    _avg_value_matches_predict_mean("y ~ s(x) + z", df)
