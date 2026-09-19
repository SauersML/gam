"""Bug hunt: a constant-scale location-scale survival fit searched its baseline
target's parameters, although its likelihood never reads them.

With a constant scale and no time wiggle the location-scale fit removes the
I-spline time warp and carries ``-log t`` on the location channel (gam#892). The
collapsed time block reads none of the time offsets a baseline target defines,
so a Weibull target's ``(scale, shape)`` do not enter the likelihood. The
workflow still ran its outer search over them. Its envelope gradient contracted
the fit's residuals with offsets the fit had discarded, so it stayed nonzero on
an objective that could not move. Each probe refit the whole model. The AoU
death fit (n = 50,000, a joint Duchon smooth over six PCs, Weibull target) was
still searching when its 15-minute cap ended it; the same fit with the default
target returns in about 30 s.

Expected: a typed refusal naming the target that has no parameter here, as a
baseline option the target does not use is refused. The default target fits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _frame() -> Any:
    rng = np.random.default_rng(20260919)
    n = 1500
    x = rng.uniform(-1.0, 1.0, n)
    time = np.exp(1.0 + 0.4 * x + 0.35 * rng.standard_normal(n))
    censor = rng.uniform(1.0, 8.0, n)
    return pd.DataFrame(
        {"time": np.minimum(time, censor), "event": (time <= censor).astype(float), "x": x}
    )


def test_collapsed_warp_refuses_a_baseline_target_it_cannot_use() -> None:
    frame = _frame()
    formula = "Surv(time, event) ~ s(x, k=6)"
    with pytest.raises(Exception, match="baseline_target='weibull' has no parameter"):
        gamfit.fit(
            frame,
            formula,
            survival_likelihood="location-scale",
            baseline_target="weibull",
            baseline_scale=3.0,
            baseline_shape=1.0,
        )
    default = gamfit.fit(frame, formula, survival_likelihood="location-scale")
    query = pd.DataFrame({"time": 3.5, "event": 0.0, "x": np.linspace(-1.0, 1.0, 21)})
    survival = np.asarray(default.predict(query).survival_at(np.array([1.5, 3.5])), dtype=float)
    assert np.all(survival[:, 1] < survival[:, 0])
