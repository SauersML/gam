"""Non-Gaussian ``Model.sample`` integrates smoothing-parameter uncertainty
exactly as ``Model.predict`` does.

``predict`` prices its bands off the smoothing-corrected covariance, whose
second-order form is the cubature mixture over the REML smoothing-parameter
posterior. ``sample`` used to draw beta conditional on the fitted smoothing
parameters for binomial and Poisson fits, so its 95% mean intervals covered
the truth ~3.5 points less often than ``predict``'s (0.912 vs 0.947 on the
binomial cell of the pyGAM audit). The draws now mix exact beta | rho draws
over the same cubature nodes, and the two interval families must agree.

Each replicate's draws must report the same smoothing treatment as that
replicate's ``predict`` band: marginalised (or the linearised correction the
fit published instead) whenever the band is smoothing-corrected, conditional
with a typed reason only when the band is conditional too. A replicate whose
fit is refused as unconverged has no posterior and is excluded from both
interval families alike.

Every tolerance is a multiple of the Monte Carlo standard error across
replicates; the multiple is the two-sided normal quantile at the declared
family-wise false-alarm rate, Bonferroni-split across the assertions.
"""

from __future__ import annotations

import importlib
from statistics import NormalDist
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_N = 400
_REPS = 200
_TEST_ROWS = 60
_LEVEL = 0.95
# Draws per replicate: an order-statistic interval from S draws misses its
# nominal content by O(1/S), two orders of magnitude below the across-replicate
# Monte Carlo standard error at _REPS replicates.
_DRAWS = 1000
_FORMULA = "y ~ s(x1) + s(x2) + s(x3)"
_FAMILIES = ("binomial", "poisson")
# Declared family-wise false-alarm rate of this module, split over the two
# assertions (nominal coverage, paired agreement) for each family.
_FAMILY_WISE_ALPHA = 0.01
_ASSERTIONS = 2 * len(_FAMILIES)
_Z = NormalDist().inv_cdf(1.0 - _FAMILY_WISE_ALPHA / (2 * _ASSERTIONS))


def _truth(family: str, x: Any) -> Any:
    two_pi = 2.0 * np.pi
    if family == "binomial":
        eta = 1.5 * np.sin(two_pi * x[:, 0]) + 0.6 * np.cos(two_pi * x[:, 2])
        return 1.0 / (1.0 + np.exp(-eta))
    eta = 0.5 + 0.8 * np.sin(two_pi * x[:, 0]) + 0.25 * np.cos(two_pi * x[:, 2])
    return np.exp(eta)


def _response(family: str, mu: Any, rng: Any) -> Any:
    if family == "binomial":
        return (rng.uniform(size=mu.shape) < mu).astype(float)
    return rng.poisson(mu).astype(float)


def _covered(truth: Any, lower: Any, upper: Any) -> float:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return float(np.mean((truth >= lo) & (truth <= hi)))


def _replicate(family: str, rep: int, x_test: Any) -> tuple[float, float] | None:
    """Coverage of the sample() and predict() mean intervals on one replicate,
    or ``None`` when the fit itself is refused as unconverged: that replicate
    has no posterior for either interval to be priced off."""
    rng = np.random.default_rng([rep, _FAMILIES.index(family)])
    x = rng.uniform(0.0, 1.0, (_N, 3))
    y = _response(family, _truth(family, x), rng)
    data = {"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "y": y}
    test = {"x1": x_test[:, 0], "x2": x_test[:, 1], "x3": x_test[:, 2]}
    mu_test = _truth(family, x_test)

    try:
        model = gamfit.fit(data, _FORMULA, family=family)
    except gamfit.FitConvergenceError:
        return None
    band = model.predict(test, interval=_LEVEL)
    posterior = model.sample(data, samples=_DRAWS, seed=rep)
    drawn = posterior.predict(test, level=_LEVEL)

    # sample() integrates the smoothing parameters exactly when predict()'s
    # band does; when the fit carries no smoothing measure both condition on
    # rho-hat, and the draws say why.
    band_source = band["covariance_source"]
    if band_source == "conditional":
        assert posterior.covariance_source == "conditional", (
            f"{family} rep {rep}: predict() is conditional but sample() reports "
            f"{posterior.covariance_source}"
        )
        assert posterior.covariance_reason, (
            f"{family} rep {rep}: conditional draws carry no reason"
        )
    else:
        assert posterior.covariance_source in {
            "smoothing-marginalised",
            "smoothing-corrected",
        }, (
            f"{family} rep {rep}: predict() is {band_source} but sample() drew "
            f"{posterior.covariance_source}"
        )
    return (
        _covered(mu_test, drawn["posterior_mean_lower"], drawn["posterior_mean_upper"]),
        _covered(mu_test, band["posterior_mean_lower"], band["posterior_mean_upper"]),
    )


def _mean_and_se(values: Any) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


@pytest.mark.slow
@pytest.mark.parametrize("family", _FAMILIES)
def test_sample_mean_intervals_cover_like_predict(family: str) -> None:
    x_test = np.random.default_rng(_TEST_ROWS).uniform(0.02, 0.98, (_TEST_ROWS, 3))
    fitted = [row for rep in range(_REPS) if (row := _replicate(family, rep, x_test))]
    sample_cover = np.asarray([row[0] for row in fitted])
    predict_cover = np.asarray([row[1] for row in fitted])

    cover, cover_se = _mean_and_se(sample_cover)
    assert abs(cover - _LEVEL) <= _Z * cover_se, (
        f"{family}: sample() {_LEVEL:.0%} mean intervals cover {cover:.4f} "
        f"(MC SE {cover_se:.4f}, tolerance {_Z:.2f} SE)"
    )

    gap, gap_se = _mean_and_se(sample_cover - predict_cover)
    assert abs(gap) <= _Z * gap_se, (
        f"{family}: sample() coverage {cover:.4f} vs predict() "
        f"{predict_cover.mean():.4f}; paired gap {gap:+.4f} (MC SE {gap_se:.4f}, "
        f"tolerance {_Z:.2f} SE)"
    )
