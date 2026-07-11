"""All-zero count responses have no finite log-rate optimum (#1515/#2255).

The historical contract forced such fits through and then changed an
overflowing posterior mean into a finite plug-in inverse link. That substituted
a MAP estimand for the required posterior mean. The family validation boundary
now refuses the unidentifiable fit before optimization, with one typed message
shared by Rust, CLI, and Python.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


@pytest.mark.parametrize("formula", ["y ~ 1", "y ~ s(x)"])
@pytest.mark.parametrize("family", ["poisson", "negative_binomial"])
def test_all_zero_count_response_is_rejected_before_fit(formula: str, family: str) -> None:
    data = {"x": np.linspace(0.0, 1.0, 200), "y": np.zeros(200)}

    with pytest.raises(gamfit.GamError) as excinfo:
        gamfit.fit(data, formula, family=family)

    message = str(excinfo.value)
    assert "all counts are 0" in message, message
    assert "no finite fitted mode or finite posterior mean/variance" in message, message
    assert "at least one positive count" in message, message


@pytest.mark.parametrize("family", ["poisson", "negative_binomial"])
def test_count_response_with_positive_event_is_not_rejected_as_degenerate(family: str) -> None:
    x = np.linspace(0.0, 1.0, 200)
    y = np.zeros(200)
    y[::20] = 1.0
    model = gamfit.fit({"x": x, "y": y}, "y ~ 1", family=family)
    prediction = np.asarray(model.predict({"x": x, "y": y}), dtype=float).ravel()
    assert prediction.shape == y.shape
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction > 0.0)
