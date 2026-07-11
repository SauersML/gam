"""The public fit surface must preserve posterior estimands (#2255).

An all-zero Poisson or negative-binomial response drives the log-rate optimum
to minus infinity. No fit is minted, so prediction never needs a plug-in-mean or
delta-interval fallback to disguise non-finite posterior moments.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


@pytest.mark.parametrize("formula", ["y ~ 1", "y ~ s(x)"])
@pytest.mark.parametrize("family", ["poisson", "negative_binomial"])
def test_all_zero_count_fit_refuses_with_family_owned_error(
    formula: str, family: str
) -> None:
    n = 200
    data = {"x": np.linspace(0.0, 1.0, n), "y": np.zeros(n)}

    with pytest.raises(gamfit.GamError) as excinfo:
        gamfit.fit(data, formula, family=family)

    message = str(excinfo.value)
    assert "all counts are 0" in message, message
    assert "IntegrationError" not in message, message
    assert "StalledAtValidMinimum" not in message, message
