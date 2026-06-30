"""#392 / #369 regression: non-linear survival baselines (and timewiggle) must
actually fit through the public API.

Both issues had the same root cause: the transformation/weibull survival
workflow optimizes the scalar baseline whenever it is non-linear, and the outer
continuation pre-warm unconditionally invoked the *gradient-free* CompassSearch
baseline's "unreachable by construction" gradient stub, so every outer seed was
rejected with::

    no candidate seeds passed outer startup validation
    ... CompassSearch dispatch only calls eval_cost; eval(gradient) is
        unreachable by construction

- #392: ``survival_likelihood="transformation"`` with
  ``baseline_target in {weibull, gompertz, gompertz-makeham}`` — every
  non-linear baseline died at REML startup while ``"linear"`` fit fine.
- #369: ``survival_likelihood="weibull"`` with a ``timewiggle(...)`` term —
  the identical data fit without the term and failed with it (the timewiggle
  term forces the same non-linear baseline optimization path).

The CompassSearch / AuxiliaryGradientFree subsystem was later purged entirely,
so the offending pre-warm path no longer exists and the baseline always supplies
an analytic gradient. BUT the regression tests both issues were closed on were
*deleted* with that purge and never replaced (``CompassSearch`` and
``aux_compass_skips_continuation_prewarm_with_interior_seed`` have zero hits in
the tree), leaving this cluster with no fit-to-completion guard. This test
restores one: it pins that each advertised non-linear baseline, and the
timewiggle path, fits to a finite, predictable model — so a future reintroduction
of a gradient-free baseline pre-warm (or any startup-validation regression on
this path) fails loudly here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit


def _make_weibull_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(40, 75, n)
    eta = -2 + 0.04 * (age - 50)
    u = rng.uniform(1e-9, 1, n)
    t = np.exp(-eta / 1.5) * (-np.log(u)) ** (1 / 1.5) * 10
    c = np.minimum(rng.exponential(25.0, n), 25.0)
    return pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": np.minimum(t, c),
            "event": (t <= c).astype(int),
            "age": age,
        }
    )


@pytest.mark.parametrize(
    "baseline_target", ["linear", "weibull", "gompertz", "gompertz-makeham"]
)
def test_transformation_survival_nonlinear_baseline_fits(baseline_target: str) -> None:
    """#392: every advertised baseline_target fits the transformation likelihood."""
    df = _make_weibull_frame()

    # Must NOT raise "no candidate seeds passed outer startup validation".
    model = gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ s(age)",
        survival_likelihood="transformation",
        baseline_target=baseline_target,
    )

    # The fitted model must be predictable and finite (a real convergence, not a
    # degenerate "fitted" sentinel). Survival predictions expose a survival
    # surface S(t) that must be finite and a valid probability in [0, 1].
    pred = model.predict(df)
    surv = np.asarray(pred.survival_at([5.0, 10.0, 20.0]), dtype=float)
    assert surv.size > 0
    assert np.all(np.isfinite(surv)), (
        f"transformation/{baseline_target} produced non-finite survival surface"
    )
    assert np.all((surv >= -1e-9) & (surv <= 1.0 + 1e-9)), (
        f"transformation/{baseline_target} survival probabilities out of [0,1]: "
        f"{surv.ravel()[:5]}"
    )


def test_weibull_survival_with_timewiggle_fits() -> None:
    """#369: weibull survival + timewiggle(...) fits (the term forces the same
    non-linear baseline optimization that used to crash)."""
    rng = np.random.default_rng(0)
    n = 1500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = 0.5 * x1 - 0.3 * x2
    t = rng.weibull(1.5, size=n) * 3.0 * np.exp(-eta / 1.5)
    cens = rng.uniform(1.0, 8.0, size=n)
    age1 = np.minimum(t, cens)
    event = (t <= cens).astype(int)
    data = {
        "age0": np.zeros(n),
        "age1": age1,
        "event": event,
        "x1": x1,
        "x2": x2,
    }

    # Control: the same data fits without the timewiggle term.
    gamfit.fit(
        data,
        "Surv(age0, age1, event) ~ x1 + x2",
        survival_likelihood="weibull",
    )

    # The regression: adding timewiggle(...) must still fit, not raise
    # "no candidate seeds passed outer startup validation".
    model = gamfit.fit(
        data,
        "Surv(age0, age1, event) ~ x1 + x2 + timewiggle(internal_knots=4)",
        survival_likelihood="weibull",
    )
    pred = model.predict(data)
    surv = np.asarray(pred.survival_at([1.0, 3.0, 6.0]), dtype=float)
    assert surv.size > 0
    assert np.all(np.isfinite(surv)), "weibull + timewiggle produced non-finite survival surface"
    assert np.all((surv >= -1e-9) & (surv <= 1.0 + 1e-9)), (
        f"weibull + timewiggle survival probabilities out of [0,1]: {surv.ravel()[:5]}"
    )
