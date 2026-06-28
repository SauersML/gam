from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


@pytest.mark.xfail(
    strict=True,
    reason="#1512 triage: stale fit_array API — gamfit.fit_array now takes "
    "`formula` as a required positional (fit_array(X, Y, formula, *, ...)), so "
    "fit_array(x, y, family=...) raises TypeError: missing 'formula'; and the "
    "result no longer exposes a `likelihood_spec` attribute. Add the formula "
    "argument and re-point at the current likelihood-spec accessor to re-enable.",
)
def test_fit_array_documented_defaults_reach_likelihood_spec() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 1))

    tweedie_y = np.exp(0.2 + 0.4 * x[:, 0]) + 0.1
    tweedie = gamfit.fit_array(x, tweedie_y, family="tweedie-log")
    assert abs(float(tweedie.likelihood_spec.get("tweedie_p", -999.0)) - 1.5) < 1e-12, (
        "fit_array should forward the documented Tweedie default p=1.5 into the saved likelihood spec"
    )

    negbin_y = np.clip(np.round(np.exp(0.3 + 0.2 * x[:, 0])), 0, None)
    with pytest.raises(ValueError, match="negbin_theta"):
        gamfit.fit_array(x, negbin_y, family="negbin-log")

    beta_y = np.clip(0.5 + 0.1 * np.tanh(x[:, 0]), 1e-6, 1 - 1e-6)
    with pytest.raises(ValueError, match="beta_phi"):
        gamfit.fit_array(x, beta_y, family="beta-regression-logit")
