"""Bug hunt: a location-scale survival fit with any smooth in its location
formula predicts a survival curve with no time dependence.

A constant-scale location-scale fit removes the monotone time warp and carries
``-log t`` on the location channel instead (gam#892): the fit's standardized
residual is ``u = (log t - mu(x)) / sigma``. That collapse needs only a
constant scale and no time wiggle, so it also happens beside a penalized
location term such as ``s(x)`` or a Duchon smooth. The fit recorded its time
parameterization from the smoothing layout instead, as ``monotone_warp``
whenever any smoothing parameter existed, and saved replay then dropped
``-log t``: the time warp it evaluated had been removed, so the predicted curve
was flat.

Observed on gamfit 0.1.272 (n = 8000, 475 events, AoU-shaped cohort):

    Surv(followup, event) ~ sex + z                  F(1), F(3) = 0.031, 0.079
    Surv(followup, event) ~ sex + z + s(lookback)    F(1), F(3) = 0.031, 0.031
    Surv(entry, exit, event) ~ sex + z               S(50, 55, 60) = 0.97, 0.91, 0.81
    Surv(entry, exit, event) ~ sex + z + duchon(PC)  S(50, 55, 60) = 1, 1, 1, lp = -23.7

The parametric fits (every location and scale term unpenalized) take the
direct AFT MLE and were always recorded correctly; they are the control.

Expected: the smooth-location fit's conditional survival ``S(t2) / S(t1)``
tracks the lognormal truth and the parametric fit of the same data, on the
follow-up scale and on the age scale with delayed entry.
"""

from __future__ import annotations

import importlib
from math import erf, sqrt
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

_N = 2000
_normal_cdf = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))
# (base, slope, sigma) of the lognormal truth log T = base + slope * x + sigma * e.
_FOLLOWUP = (1.0, 0.4, 0.35)
_AGE = (float(np.log(68.0)), 0.1, 0.12)


def _true_ratio(x: Any, truth: tuple[float, float, float], t1: float, t2: float) -> Any:
    """Lognormal conditional survival S(t2) / S(t1) at covariate x."""
    base, slope, sigma = truth
    mu = base + slope * x
    s1 = 1.0 - _normal_cdf((np.log(t1) - mu) / sigma)
    s2 = 1.0 - _normal_cdf((np.log(t2) - mu) / sigma)
    return s2 / s1


def _followup_frame(rng: Any) -> Any:
    base, slope, sigma = _FOLLOWUP
    x = rng.uniform(-1.0, 1.0, _N)
    time = np.exp(base + slope * x + sigma * rng.standard_normal(_N))
    censor = rng.uniform(1.0, 8.0, _N)
    return pd.DataFrame(
        {"time": np.minimum(time, censor), "event": (time <= censor).astype(float), "x": x}
    )


def _age_frame(rng: Any) -> Any:
    """Delayed entry on the age scale: rows enter at 40-60 and are kept only
    if they are event-free at entry (left truncation), then followed 15 y."""
    base, slope, sigma = _AGE
    rows: list[tuple[float, float, float, float]] = []
    while len(rows) < _N:
        x = rng.uniform(-1.0, 1.0)
        entry = rng.uniform(40.0, 60.0)
        age = float(np.exp(base + slope * x + sigma * rng.standard_normal()))
        if age <= entry:
            continue
        stop = entry + 15.0
        rows.append((entry, min(age, stop), float(age <= stop), x))
    return pd.DataFrame(rows, columns=["entry", "exit", "event", "x"])


def _ratios(model: Any, frame: Any, t1: float, t2: float) -> Any:
    survival = np.asarray(model.predict(frame).survival_at(np.array([t1, t2])), dtype=float)
    assert np.all(np.isfinite(survival)) and np.all(survival > 0.0)
    return survival[:, 1] / survival[:, 0]


@pytest.mark.parametrize("scale", ["followup", "age"])
def test_smooth_location_keeps_the_time_dependence_of_its_survival(scale: str) -> None:
    rng = np.random.default_rng(20260918)
    x = np.linspace(-1.0, 1.0, 41)
    if scale == "followup":
        frame, response, truth = _followup_frame(rng), "Surv(time, event)", _FOLLOWUP
        t1, t2 = 1.5, 3.5
        query = pd.DataFrame({"time": t2, "event": 0.0, "x": x})
    else:
        frame, response, truth = _age_frame(rng), "Surv(entry, exit, event)", _AGE
        t1, t2 = 62.0, 72.0
        query = pd.DataFrame({"entry": 40.0, "exit": t2, "event": 0.0, "x": x})
    expected = _true_ratio(x, truth, t1, t2)
    # The flat curve the defect produced has ratio 1: keep the truth far from it.
    assert expected.max() < 0.8

    smooth = gamfit.fit(frame, f"{response} ~ s(x, k=6)", survival_likelihood="location-scale")
    parametric = gamfit.fit(frame, f"{response} ~ x", survival_likelihood="location-scale")
    smooth_ratio = _ratios(smooth, query, t1, t2)
    parametric_ratio = _ratios(parametric, query, t1, t2)

    # n = 2000, most rows observed, and the truth is linear in x, so both fits
    # estimate the same lognormal AFT: their conditional survival sits within a
    # few hundredths of the truth and of each other. The defect's ratio is 1.
    assert np.mean(np.abs(smooth_ratio - expected)) < 0.05, (smooth_ratio, expected)
    assert np.max(np.abs(smooth_ratio - expected)) < 0.12, (smooth_ratio, expected)
    assert np.mean(np.abs(smooth_ratio - parametric_ratio)) < 0.05, (
        smooth_ratio,
        parametric_ratio,
    )
