"""Bug hunt: ``summary().deviance`` changes DEFINITION when ``noise_formula=`` is
supplied. A location-scale fit reports ``-2 * log-likelihood`` where every
standard fit reports the classical deviance ``2*(l_sat - l)``, so the same model
with the same fitted values reports two numbers differing by 5-9x.

``y ~ x + z`` on n=600, once as an ordinary fit and once with
``noise_formula="1"`` (a constant scale -- i.e. the identical model):

    family           standard .deviance   LS .deviance     ratio
    gaussian             193.47887921     1023.66945980     5.29
    gamma                214.22303356     1917.87955312     8.95

The mean fits are the same model, not merely similar: max|mu_standard -
mu_locationscale| is 1.9e-11 (gaussian) and 1.9e-09 (gamma), and evaluating the
family deviance by hand at the LOCATION-SCALE fit's own fitted mean gives
193.47887921 (gaussian) and 214.21294681 (gamma) -- the standard numbers, not
the reported ones.

What the location-scale path reports instead is exactly ``-2 * log-likelihood``:

    LS gaussian:  -2 * (-511.8347299013195) = 1023.66945980   (reported)
    LS gamma:     -2 * (-958.9397765607144) = 1917.87955312   (reported)

i.e. the ``n*log(2*pi*sigma^2)`` / ``lgamma`` normalizing terms are folded in and
the saturated log-likelihood is never subtracted. (Binomial has no constant-scale
location-scale counterpart: its sigma is identified only relative to the
threshold, so ``noise_formula="1"`` is refused there, #3879.)

This breaks every deviance-based comparison across the boundary -- explained
deviance ``1 - D/D_null``, deviance-difference tests, and any ``compare_models``
that puts a location-scale fit next to a standard one -- because the two sides
are not the same functional.

Precedent: #2126 (Gamma reported the SCALED deviance) and #2131 (Tweedie, the
same defect left in place) were both accepted and fixed to "the unscaled
deviance ``D = 2*sum(w*d(y, mu))`` every other family reports". This is the same
class of defect on the location-scale path, with a different wrong functional.

Observed: ``summary().deviance`` on a location-scale fit is ``-2*loglik``.

Expected: it is the family deviance at the fitted mean -- the number the
identical standard fit reports.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_N = 600


def _data(family: str) -> dict[str, Any]:
    rng = np.random.default_rng(11)
    x = rng.uniform(0.0, 1.0, _N)
    z = rng.uniform(0.0, 1.0, _N)
    eta = np.sin(2.0 * np.pi * x) + 0.5 * z
    response = {
        "gaussian": lambda: eta + 0.35 * rng.standard_normal(_N),
        "gamma": lambda: rng.gamma(6.0, np.exp(0.5 + eta) / 6.0),
    }[family]()
    return {"x": x, "z": z, "y": response}


def _pair(family: str) -> tuple[Any, Any, dict[str, Any]]:
    data = _data(family)
    standard = gamfit.fit(data, "y ~ x + z", family=family)
    location_scale = gamfit.fit(data, "y ~ x + z", family=family, noise_formula="1")
    return standard, location_scale, data


def _mean(model: Any, data: dict[str, Any]) -> np.ndarray:
    return np.asarray(model.predict(data, return_type="dict")["posterior_mean"], dtype=float)


@pytest.mark.parametrize("family", ["gaussian", "gamma"])
def test_constant_scale_location_scale_is_the_same_mean_fit(family: str) -> None:
    """Premise: `noise_formula="1"` is the identical model, not a nearby one."""
    standard, location_scale, data = _pair(family)
    np.testing.assert_allclose(
        _mean(location_scale, data), _mean(standard, data), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("family", ["gaussian", "gamma"])
def test_location_scale_reports_the_family_deviance(family: str) -> None:
    standard, location_scale, _ = _pair(family)
    reported = float(location_scale.summary().deviance)
    expected = float(standard.summary().deviance)
    assert reported == pytest.approx(expected, rel=1e-3), (
        f"{family}: the location-scale fit reports deviance {reported!r} where the "
        f"identical standard fit reports {expected!r} "
        f"(ratio {reported / expected:.4f}); the reported value is -2*loglik"
    )


@pytest.mark.parametrize("family", ["gaussian", "gamma"])
def test_location_scale_deviance_is_not_minus_two_loglik(family: str) -> None:
    """The reported number is exactly -2*loglik, which is what makes it wrong."""
    _, location_scale, _ = _pair(family)
    summary = location_scale.summary()
    log_likelihood = (summary.extras or {}).get("log_likelihood")
    assert log_likelihood is not None
    assert float(summary.deviance) != pytest.approx(-2.0 * float(log_likelihood), rel=1e-9), (
        f"{family}: summary().deviance is exactly -2*log_likelihood "
        f"({summary.deviance!r}); the classical deviance subtracts the saturated "
        "log-likelihood and drops the normalizing constants"
    )
