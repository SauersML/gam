"""gam#2999: ``family="bernoulli-marginal-slope"`` read neither ``link=`` nor ``flexible_link=``.

``link="logit"`` (or cloglog, sas, ``flexible(probit)``) and ``flexible_link=True`` all fitted
and predicted bit-identically to the plain probit fit, while the same links written in the
formula were refused. The keywords are now read on the one path the formula takes:

* a non-probit ``link=`` is refused, naming the argument and its value, on survival
  marginal-slope as well;
* ``link="probit"`` is the family's own link, so the fit is the default one;
* ``flexible_link=True`` and ``link="flexible(probit)"`` fit the default link deviation, the
  model the formula's ``linkwiggle()`` gives, coefficient for coefficient.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import gamfit


def _data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 600
    x = rng.uniform(-1.0, 1.0, n)
    z = (rng.gamma(3.0, 1.0, n) - 3.0) / np.sqrt(3.0)
    eta = -0.8 + 0.6 * x + (0.5 + 0.2 * x) * z
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return {"x": x, "z": z, "y": y}


_GRID = {"x": np.linspace(-1.0, 1.0, 21).repeat(5), "z": np.tile(np.linspace(-2.0, 3.0, 5), 21)}


def _fit(formula: str, **extra):
    return gamfit.fit(
        _data(),
        formula,
        family="bernoulli-marginal-slope",
        z_column="z",
        slope_formula="1 + x",
        config={"latent_measure": "global-empirical"},
        **extra,
    )


def _predictions(model) -> np.ndarray:
    return np.asarray(model.predict(_GRID), dtype=float)


@pytest.mark.parametrize("link", ["logit", "cloglog", "sas", "cauchit", "flexible(logit)"])
def test_a_non_probit_link_keyword_is_refused_by_name(link: str) -> None:
    with pytest.raises(gamfit.InvalidConfigurationError, match=re.escape(f"the link argument names '{link}'")):
        _fit("y ~ s(x, k=5)", link=link)


def test_link_probit_is_the_default_fit() -> None:
    assert np.array_equal(_predictions(_fit("y ~ s(x, k=5)", link="probit")),
                          _predictions(_fit("y ~ s(x, k=5)")))


@pytest.mark.parametrize("extra", [{"flexible_link": True}, {"link": "flexible(probit)"}])
def test_a_flexible_link_keyword_fits_the_formula_linkwiggle(extra: dict) -> None:
    keyword = _fit("y ~ s(x, k=5)", **extra)
    formula = _fit("y ~ s(x, k=5) + linkwiggle()")
    plain = _fit("y ~ s(x, k=5)")
    assert keyword._coefficient_state() == formula._coefficient_state()
    assert np.array_equal(_predictions(keyword), _predictions(formula))
    assert not np.array_equal(_predictions(keyword), _predictions(plain)), (
        "the flexible link must change the fit, not be dropped"
    )


def test_survival_marginal_slope_refuses_a_non_probit_link_keyword() -> None:
    rng = np.random.default_rng(11)
    n = 200
    entry = rng.uniform(0.0, 1.0, n)
    data = {
        "entry": entry,
        "exit": entry + rng.exponential(2.0, n) + 0.05,
        "event": (rng.uniform(size=n) < 0.6).astype(float),
        "bmi": rng.normal(0.0, 1.0, n),
        "prs_z": rng.normal(0.0, 1.0, n),
    }
    with pytest.raises(gamfit.InvalidConfigurationError, match=re.escape("the link argument names 'logit'")):
        gamfit.fit(
            data,
            "Surv(entry, exit, event) ~ bmi",
            survival_likelihood="marginal-slope",
            z_column="prs_z",
            slope_formula="1",
            link="logit",
        )
