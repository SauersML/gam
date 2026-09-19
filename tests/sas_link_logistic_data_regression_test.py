"""Binomial SAS link on ordinary logistic data (pyGAM audit families.md F4).

The audit's ``probe_sas.py`` fit ``y ~ s(x)`` with ``link="sas"`` on
``n = 1500`` Bernoulli rows with ``eta = sin(2 pi x)`` and got
"PIRLS row geometry is not representable ... eta=-10240" as the final error,
while ``logit`` and ``beta-logistic`` fit the same data. That refusal now only
rejects an outer trial point, which the outer search recovers from.

A second defect cut the outer search short: the cost-stall guard read a run of
ARC-rejected trials whose steps were still moving as a proven replay of the
incumbent, and stopped short of stationarity. Seed 7 failed with "did not
certify a stationary optimum" for that reason.

Returning a fit is itself the convergence certificate, so this pins that the
fit returns, and that its fitted mean tracks the truth on the scale of the
correctly specified logit fit on the same rows. These rows keep the mean within
about [0.27, 0.73], so the tails that the SAS skew and tail weight describe are
never observed and ``(epsilon, delta)`` are only weakly identified. On some
draws (seed 1 among them) the REML/LAML path then runs into an inner-mode fold,
a point past which the penalized likelihood has no inner mode and the Laplace
normalizer breaks down, and the engine refuses to certify rather than return an
uncertified fit. Those draws are documented in docs/families-and-links.md and
are not pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit

_N = 1500
_GRID = np.linspace(0.05, 0.95, 50)
_TRUTH = 1.0 / (1.0 + np.exp(-np.sin(2.0 * np.pi * _GRID)))


def _probe_data(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, _N)
    eta = np.sin(2.0 * np.pi * x)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return {"x": x, "y": y}


def _mean_rmse(link: str, data: dict[str, np.ndarray]) -> float:
    model = gamfit.fit(data, "y ~ s(x)", family="binomial", link=link)
    mean = np.asarray(model.predict({"x": _GRID}), dtype=float).ravel()
    assert np.all(np.isfinite(mean)), f"{link} fitted mean is not finite"
    return float(np.sqrt(np.mean((mean - _TRUTH) ** 2)))


@pytest.mark.parametrize("seed", [0, 2, 3, 7])
def test_sas_link_fits_the_audit_probe_data(seed: int) -> None:
    data = _probe_data(seed)
    sas_rmse = _mean_rmse("sas", data)
    logit_rmse = _mean_rmse("logit", data)
    # The logit fit is the correctly specified model for these rows, so its
    # error is the sampling scale of the problem. SAS spends two extra shape
    # parameters on a link it does not need; it may pay for them in variance,
    # but not by more than that scale again.
    assert sas_rmse <= 2.0 * logit_rmse, (
        f"seed {seed}: SAS mean RMSE {sas_rmse:.4f} vs logit {logit_rmse:.4f}"
    )
