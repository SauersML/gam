"""Non-Gaussian ``Model.sample`` draws cover the truth like ``Model.predict``'s
bands (gam#3398, the acceptance test of the superseded #3131).

``predict`` prices its bands off the published smoothing-corrected covariance
``V_c`` (first-order ``Vb + J V_rho J^T`` or its cubature form). For binomial
and Poisson fits ``sample`` runs an exact sampler (Polya-Gamma Gibbs or NUTS)
on ``beta | rho-hat``, whose Laplace covariance is the conditional ``Vb``, and
then maps every draw through the linear optimal-transport map
``T = Vb^{-1/2} (Vb^{1/2} V_c Vb^{1/2})^{1/2} Vb^{-1/2}``
(``recolor_to_smoothing_corrected_covariance`` in
``crates/gam-inference/src/sample.rs``), so the draws leave with covariance
``V_c``. Before that recolouring the pyGAM audit measured sample() 95% mean
intervals covering 0.912 against predict()'s 0.947 on the binomial cell
(paired gap +0.035, SE 0.009).

Each replicate's draws must carry the same covariance provenance as that
replicate's ``predict`` band. A fit that recorded a typed absence of the
corrected covariance publishes ``conditional`` on both surfaces. Such a
replicate stays in both interval families (the paired comparison is still
like for like), and the module reports how many there were, and which,
instead of dropping them. Only a fit refused as unconverged
(``FitConvergenceError``) has no posterior at all and is excluded.

Every tolerance is a multiple of the Monte Carlo standard error across
replicates. The multiple is the two-sided normal quantile at the declared
family-wise false-alarm rate, Bonferroni-split over the four assertions (two
per family). Coverage that is too HIGH fails exactly like coverage that is too
low: a conservative band is a calibration defect, not a safety margin.
"""

from __future__ import annotations

import importlib
import warnings
from statistics import NormalDist
from typing import Any, NamedTuple

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit
from gamfit.errors import FitConvergenceError

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


class _Replicate(NamedTuple):
    sample_cover: float
    predict_cover: float
    covariance_source: str


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


def _replicate(family: str, rep: int, x_test: Any) -> _Replicate | None:
    """Coverage of the sample() and predict() mean intervals on one replicate,
    or ``None`` when the fit is refused as unconverged."""
    rng = np.random.default_rng([rep, _FAMILIES.index(family)])
    x = rng.uniform(0.0, 1.0, (_N, 3))
    y = _response(family, _truth(family, x), rng)
    data = {"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "y": y}
    test = {"x1": x_test[:, 0], "x2": x_test[:, 1], "x3": x_test[:, 2]}
    mu_test = _truth(family, x_test)

    try:
        model = gamfit.fit(data, _FORMULA, family=family)
    except FitConvergenceError:
        return None
    band = model.predict(test, interval=_LEVEL)
    posterior = model.sample(data, samples=_DRAWS, seed=rep)
    assert posterior.covariance_source == band["covariance_source"], (
        f"{family} rep {rep}: predict() band is {band['covariance_source']} but "
        f"sample() drew {posterior.covariance_source}"
    )
    drawn = posterior.predict(test, level=_LEVEL)
    return _Replicate(
        _covered(mu_test, drawn["posterior_mean_lower"], drawn["posterior_mean_upper"]),
        _covered(mu_test, band["posterior_mean_lower"], band["posterior_mean_upper"]),
        posterior.covariance_source,
    )


def _mean_and_se(values: Any) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


@pytest.mark.slow
@pytest.mark.parametrize("family", _FAMILIES)
def test_sample_mean_intervals_cover_like_predict(family: str) -> None:
    x_test = np.random.default_rng(_TEST_ROWS).uniform(0.02, 0.98, (_TEST_ROWS, 3))
    outcomes = {rep: _replicate(family, rep, x_test) for rep in range(_REPS)}
    refused = [rep for rep, row in outcomes.items() if row is None]
    fitted = {rep: row for rep, row in outcomes.items() if row is not None}
    conditional = [
        rep for rep, row in fitted.items() if row.covariance_source == "conditional"
    ]
    sample_cover = np.asarray([row.sample_cover for row in fitted.values()])
    predict_cover = np.asarray([row.predict_cover for row in fitted.values()])

    cover, cover_se = _mean_and_se(sample_cover)
    predict_mean, predict_se = _mean_and_se(predict_cover)
    gap, gap_se = _mean_and_se(sample_cover - predict_cover)
    # The count of fits whose corrected covariance was a typed absence is part
    # of the result (gam#3398 item 2): on those fits both surfaces ignore the
    # smoothing-parameter uncertainty. Report it on every run, pass or fail.
    report = (
        f"{family}: {len(fitted)}/{_REPS} fitted ({len(refused)} refused as "
        f"unconverged: {refused}); {len(conditional)} with a conditional "
        f"covariance (typed absence of the correction): {conditional}; "
        f"sample() cover {cover:.4f} (SE {cover_se:.4f}), predict() cover "
        f"{predict_mean:.4f} (SE {predict_se:.4f}), paired gap {gap:+.4f} "
        f"(SE {gap_se:.4f}); tolerance {_Z:.2f} SE"
    )
    warnings.warn(report, stacklevel=1)

    assert abs(cover - _LEVEL) <= _Z * cover_se, report
    assert abs(gap) <= _Z * gap_se, report
