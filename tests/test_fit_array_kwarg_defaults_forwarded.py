from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def test_fit_array_documented_defaults_reach_likelihood_spec() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 1))
    # fit_array now takes `formula` as a required positional; a single-column X
    # is exposed to the formula DSL as `x0`, and the response as `y`. The
    # requested likelihood family is forwarded into the fitted model and is
    # observable via Summary.family_name (the current likelihood accessor).

    tweedie_y = np.exp(0.2 + 0.4 * x[:, 0]) + 0.1
    tweedie = gamfit.fit_array(x, tweedie_y, "y ~ x0", family="tweedie")
    assert tweedie.summary().family_name == "Tweedie Log", (
        "fit_array should forward the documented Tweedie family into the fitted model"
    )

    negbin_y = np.clip(np.round(np.exp(0.3 + 0.2 * x[:, 0])), 0, None)
    negbin = gamfit.fit_array(x, negbin_y, "y ~ x0", family="negbin")
    assert negbin.summary().family_name == "Negative-Binomial Log", (
        "fit_array should forward the Negative-Binomial family into the fitted model"
    )

    beta_y = np.clip(0.5 + 0.1 * np.tanh(x[:, 0]), 1e-6, 1 - 1e-6)
    beta = gamfit.fit_array(x, beta_y, "y ~ x0", family="beta-regression-logit")
    assert beta.summary().family_name == "Beta Regression Logit", (
        "fit_array should forward the Beta-regression family into the fitted model"
    )
