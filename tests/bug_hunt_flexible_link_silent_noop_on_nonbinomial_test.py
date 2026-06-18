"""Regression for #1275: non-binomial flexible links must not be silent no-ops."""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd
import pytest

import gamfit


def _assert_rejects_nonbinomial_flexible_link(df, formula, *, family, match):
    with pytest.raises(gamfit.InvalidConfigurationError, match=match):
        gamfit.fit(df, formula, family=family)


def test_flexible_log_link_is_rejected_on_poisson():
    x = np.linspace(-1.5, 1.5, 64)
    y = np.maximum(0, np.round(np.exp(0.5 + 0.8 * x))).astype(float)
    df = pd.DataFrame({"y": y, "x": x})

    _assert_rejects_nonbinomial_flexible_link(
        df,
        "y ~ x + link(type=flexible(log))",
        family="poisson",
        match="flexible\\(\\.\\.\\.\\).*non-binomial",
    )


def test_flexible_identity_link_is_rejected_on_gaussian():
    x = np.linspace(-1.5, 1.5, 64)
    y = 1.0 + 0.8 * x + 0.7 * np.tanh(3.0 * x)
    df = pd.DataFrame({"y": y, "x": x})

    _assert_rejects_nonbinomial_flexible_link(
        df,
        "y ~ x + link(type=flexible(identity))",
        family="gaussian",
        match="flexible\\(\\.\\.\\.\\).*non-binomial",
    )
