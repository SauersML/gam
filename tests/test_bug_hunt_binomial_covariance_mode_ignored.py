"""Bug hunt: ``predict(covariance_mode=...)`` is silently ignored for binomial
smooth models — the smoothing-parameter-uncertainty correction never reaches the
interval, so binomial credible intervals are the bare conditional ones.

``Model.predict``'s ``covariance_mode`` selects the covariance source for the
response-scale SE: ``"conditional"`` = ``H^{-1}`` only; ``"smoothing"`` /
``None`` (the default required ``SmoothingCorrected`` mode) adds the
first-order smoothing correction ``J·Var(rho_hat)·J^T``; ``"required"`` demands
that correction and errors if it cannot be formed (``gamfit/_model.py:84-95``).
For a smooth model with REML-selected ``rho``, the correction is non-trivial, so
the modes must produce *different* standard errors — and they do for Poisson,
Gamma and Gaussian.

For binomial the three modes return **bitwise-identical** standard errors: the
correction is simply not applied. As a result a binomial ``s(x)`` model's
default credible intervals omit the smoothing uncertainty that every other
family includes by default, so they are systematically too narrow, and
``covariance_mode`` is a no-op (in particular ``"required"`` does not error even
when one would expect it to mean something).

Root cause (read, no patch): same dispatch as the sibling observation-interval
defect. ``crates/gam-pyffi/src/lib.rs`` ``predict_columns`` branches on
``(interval, model.prediction_uses_posterior_mean())``.
``prediction_uses_posterior_mean()`` is ``true`` for exactly the binomial family
(every link) and the wiggle models (``src/inference/model.rs:2681-2692``). The
``(Some(level), true)`` arm (``lib.rs:25389-25415``) calls
``predictor.predict_posterior_mean(&predict_input, &fit, Some(level))`` and
**never parses or threads ``options.covariance_mode``** — the
``predict_posterior_mean`` signature has no covariance-mode parameter, and
``StandardPredictor::predict_posterior_mean`` builds its backend from the stored
conditional covariance via ``posterior_mean_backend_or_warn`` (no smoothing
correction; ``src/inference/predict/mod.rs:1133-1176``). The sibling
``(Some(level), false)`` arm (``lib.rs:25416-25445``), taken by
Poisson/Gamma/Gaussian, *does* call ``parse_covariance_mode(options.covariance_mode)``
and feeds it into ``predict_full_uncertainty`` /
``select_uncertainty_backend``. So binomial — the only standard family on the
posterior-mean arm — can never observe ``covariance_mode`` nor the smoothing
correction.

This test asserts the contract that ``covariance_mode`` is honoured for binomial:
the smoothing-corrected SE must differ from (and not be smaller than) the
conditional SE, exactly as it does for the Poisson control. It fails today
(the binomial SEs are identical across modes) and will pass once the
posterior-mean dispatch threads ``covariance_mode`` through.

Related: #811 (same dispatch arm silently drops ``observation_interval`` for
binomial). Both are the binomial/posterior-mean path ignoring options the
sibling full-uncertainty arm honours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _fit(family: str, gen) -> "gamfit.Model":
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.uniform(0.0, 1.0, n)
    y = gen(rng, x)
    return gamfit.fit(pd.DataFrame({"y": y, "x": x}), "y ~ s(x)", family=family)


def _se(model: "gamfit.Model", mode: str) -> np.ndarray:
    grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 12)})
    out = model.predict(grid, interval=0.95, covariance_mode=mode)
    return np.asarray(out["std_error"], dtype=float)


def test_binomial_smooth_se_responds_to_covariance_mode() -> None:
    model = _fit(
        "binomial",
        lambda r, x: r.binomial(1, 1.0 / (1.0 + np.exp(-(np.sin(3.0 * x) * 2.0)))),
    )
    se_cond = _se(model, "conditional")
    se_smooth = _se(model, "smoothing")

    # The smoothing correction J·Var(rho)·J^T is non-trivial for a REML-selected
    # smooth, so the smoothing-corrected SE must differ from the conditional SE.
    # Today they are bitwise identical (the correction is never applied).
    assert not np.allclose(se_cond, se_smooth, rtol=1e-6, atol=1e-12), (
        "binomial std_error is identical for covariance_mode='conditional' and "
        "'smoothing' — the smoothing-parameter-uncertainty correction is being "
        f"dropped. conditional={se_cond}, smoothing={se_smooth}"
    )
    # The correction adds variance (H^{-1} + J·Var(rho)·J^T), so the
    # smoothing-corrected SE is never smaller than the conditional SE.
    assert np.mean(se_smooth) >= np.mean(se_cond) - 1e-12, (
        "smoothing-corrected SE is smaller than the conditional SE on average: "
        f"mean conditional={np.mean(se_cond)}, mean smoothing={np.mean(se_smooth)}"
    )


def test_poisson_smooth_se_responds_to_covariance_mode_control() -> None:
    # Control: the same mechanism is honoured for a family routed through the
    # full-uncertainty arm (Poisson is `uses_posterior_mean=False`). Passes today.
    model = _fit("poisson", lambda r, x: r.poisson(np.exp(0.5 + np.sin(3.0 * x))))
    se_cond = _se(model, "conditional")
    se_smooth = _se(model, "smoothing")
    assert not np.allclose(se_cond, se_smooth, rtol=1e-6, atol=1e-12), (
        "Poisson control: covariance_mode unexpectedly has no effect; "
        f"conditional={se_cond}, smoothing={se_smooth}"
    )
    assert np.mean(se_smooth) >= np.mean(se_cond) - 1e-12
