"""gam#3014: every link spelling a fit request carries is read.

Before the fix a formula ``link(type=...)`` silently won: a ``link=`` argument
naming a different link was dropped, and ``flexible_link=True`` was ignored
whenever the formula named the link. Now ``flexible_link`` flexes the formula's
link as it flexes a ``link=`` argument, and a ``link=`` that disagrees with the
formula is refused by name on the standard and location-scale paths alike.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd
import pytest

import gamfit


def _binary_frame() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 2000
    x = rng.uniform(-2.5, 2.5, n)
    # The true link is cloglog, so a flexible probit has something to bend.
    p = np.clip(1.0 - np.exp(-np.exp(-0.2 + 1.1 * x)), 1e-4, 1.0 - 1e-4)
    y = (rng.uniform(size=n) < p).astype(float)
    return pd.DataFrame({"x": x, "y": y, "z": rng.normal(size=n)})


@pytest.mark.parametrize(
    ("formula", "link"),
    [
        ("y ~ x + link(type=probit)", "logit"),
        ("y ~ x + link(type=probit)", "flexible(logit)"),
        ("y ~ x + link(type=logit)", "cloglog"),
    ],
)
def test_a_link_argument_that_disagrees_with_the_formula_is_refused(formula, link):
    with pytest.raises(gamfit.errors.FormulaError) as excinfo:
        gamfit.fit(_binary_frame(), formula, family="binomial", link=link)
    message = str(excinfo.value)
    assert "link(type=" in message and f'link="{link}"' in message, message


def test_a_disagreeing_link_argument_is_refused_on_the_location_scale_path():
    with pytest.raises(gamfit.errors.FormulaError) as excinfo:
        gamfit.fit(
            _binary_frame(),
            "y ~ x + link(type=probit)",
            family="binomial",
            link="logit",
            noise_formula="z",
        )
    message = str(excinfo.value)
    assert "link(type=probit)" in message and 'link="logit"' in message, message


def test_a_matching_link_argument_is_the_same_fit_as_the_formula_alone():
    data = _binary_frame()
    formula_only = gamfit.fit(data, "y ~ x + link(type=probit)", family="binomial")
    both = gamfit.fit(data, "y ~ x + link(type=probit)", family="binomial", link="probit")
    np.testing.assert_array_equal(
        np.asarray(formula_only.predict(data)), np.asarray(both.predict(data))
    )


def test_flexible_link_flexes_the_link_the_formula_names():
    data = _binary_frame()
    plain = gamfit.fit(data, "y ~ x + link(type=probit)", family="binomial")
    flagged = gamfit.fit(
        data, "y ~ x + link(type=probit)", family="binomial", flexible_link=True
    )
    spelled = gamfit.fit(data, "y ~ x + link(type=flexible(probit))", family="binomial")

    plain_mean = np.asarray(plain.predict(data), dtype=float)
    flagged_mean = np.asarray(flagged.predict(data), dtype=float)
    # Before #3014 the flag was dropped and this was the plain probit fit.
    assert np.max(np.abs(flagged_mean - plain_mean)) > 1e-3, (
        "flexible_link=True with link(type=probit) in the formula fit a fixed probit"
    )
    np.testing.assert_array_equal(flagged_mean, np.asarray(spelled.predict(data)))
    # The bent probit fits the cloglog data better than the fixed one.
    assert float(flagged.summary().deviance) < float(plain.summary().deviance)
