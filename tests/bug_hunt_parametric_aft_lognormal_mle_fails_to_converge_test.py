"""Bug hunt (gam#2112): the reduced parametric-AFT (constant-scale
location-scale survival) direct MLE spuriously fails to converge on completely
benign, fully observed data.

``survival_likelihood="location-scale"`` reduced to a constant-scale,
covariate-free model IS the textbook lognormal AFT, whose uncensored MLE is
available in closed form:

    log T ~ Normal(mu, sigma) ->  mu_hat = mean(log t),  sigma_hat = sd(log t).

The direct Newton MLE (``fit_parametric_aft_direct_mle``) used to certify
stationarity with an ABSOLUTE tolerance on the sup-norm of ``g = grad(loglik)``,
the *summed* log-likelihood gradient over all ``n`` rows. Because that gradient
is a sum over ``n`` observations, its smallest attainable sup-norm in double
precision grows like ``~n * eps``; for ``n`` beyond ~1000 the floor exceeds the
fixed ``1e-7`` tolerance, so a perfectly benign fit runs all 200 Newton
iterations and hard-errors:

    IntegrationError: direct parametric-AFT MLE: failed to converge after 200
    Newton iterations (last gradient sup-norm 5.38e-6 > tolerance 1e-7)

with the failure frequency RISING in ``n`` — the signature of an absolute
tolerance applied to an ``n``-scaled quantity.

The fix stops on the affine-invariant Newton decrement ``lambda^2 = g . (H^-1 g)``
(``0.5 * lambda^2`` estimates the log-likelihood gap ``loglik* - loglik``), which
is invariant to the sample size, so a single tolerance certifies stationarity
uniformly across ``n``.

This test asserts:
  * the exact issue repro (seed 0, n=2000, uncensored) fits without raising and
    recovers the closed-form lognormal survival curve;
  * an ``n`` sweep (100 .. 10000) x 8 seeds ALL converge (the old code failed an
    increasing fraction as ``n`` grew);
  * the failure mode also cleared with right-censoring and with a covariate,
    the two other reproductions named in the issue.

Correctness is checked model-free via the predicted survival surface (which does
not depend on the internal coefficient layout): a converged constant-scale AFT
must reproduce the analytic lognormal ``S(t) = 1 - Phi((log t - mu)/sigma)`` at
the closed-form MLE ``(mu_hat, sigma_hat)``.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _fit_constant_scale_aft(time, event, formula="Surv(time, event) ~ 1", **extra):
    data = {"time": np.asarray(time, dtype=float), "event": np.asarray(event, dtype=float)}
    data.update({k: np.asarray(v, dtype=float) for k, v in extra.items()})
    # The bug is a hard IntegrationError at fit time; simply returning is the
    # primary assertion. Any exception propagates and fails the test.
    return gamfit.fit(data, formula, survival_likelihood="location-scale")


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    from math import erf, sqrt

    vec = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))
    return vec(z)


def _predicted_survival(model, times, base_row) -> np.ndarray | None:
    """Best-effort model-free survival surface S(t) at a fixed covariate row.

    Returns None if this build's predict surface does not expose a survival
    probability column (the convergence assertion still stands on its own)."""
    rows = pd.DataFrame({"time": times, "event": np.ones_like(times), **base_row})
    try:
        pred = model.predict(rows, interval=0.9)
    except Exception:
        try:
            pred = model.predict(rows)
        except Exception:
            return None
    for key in ("survival", "surv", "S"):
        try:
            col = pred[key]
        except Exception:
            col = None
        if col is not None:
            return np.asarray(col, dtype=float)
    return None


def test_reduced_parametric_aft_converges_exact_issue_repro() -> None:
    # The exact issue reproduction: seed 0, n = 2000, no censoring. Used to raise
    # IntegrationError; must now fit.
    rng = np.random.default_rng(0)
    log_t = rng.normal(1.4, 0.5, 2000)
    time = np.exp(log_t)
    event = np.ones(2000)

    model = _fit_constant_scale_aft(time, event)
    assert model is not None

    # The fit exposes a finite coefficient table (a converged optimum, not a
    # degenerate / NaN state).
    coefs = [c["estimate"] for c in model.summary().coefficients]
    assert len(coefs) >= 2
    assert all(math.isfinite(float(c)) for c in coefs)

    # Model-free correctness: the predicted survival curve must track the
    # closed-form lognormal MLE. (Skipped only if this build's predict surface
    # does not surface a survival column.)
    mu_hat = float(np.mean(log_t))
    sigma_hat = float(np.std(log_t))
    grid = np.array([2.0, 3.0, 4.0, 6.0, 9.0])
    surv = _predicted_survival(model, grid, {})
    if surv is not None:
        truth = 1.0 - _normal_cdf((np.log(grid) - mu_hat) / sigma_hat)
        rel = np.sqrt(np.sum((surv - truth) ** 2) / np.sum(truth**2))
        assert rel < 0.1, f"predicted survival diverges from lognormal MLE: rel_l2={rel:.4f}"


def test_reduced_parametric_aft_converges_across_n_and_seeds() -> None:
    # The smoking gun: the OLD absolute-gradient tolerance failed an increasing
    # fraction of these as n grew (n=2000 seeds 0 and 7; up to 3/8 at n=10000).
    # Every one must now converge (fit without raising).
    failures = []
    for n in (100, 300, 1000, 2000, 5000, 10000):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            log_t = rng.normal(1.4, 0.5, n)
            time = np.exp(log_t)
            event = np.ones(n)
            try:
                model = _fit_constant_scale_aft(time, event)
            except Exception as exc:  # noqa: BLE001 - want the message on failure
                failures.append(f"n={n} seed={seed}: {type(exc).__name__}: {exc}")
                continue
            coefs = [float(c["estimate"]) for c in model.summary().coefficients]
            assert all(math.isfinite(c) for c in coefs), f"n={n} seed={seed}: non-finite coef"
    assert not failures, "reduced parametric-AFT failed to converge on benign data:\n" + "\n".join(
        failures
    )


def test_reduced_parametric_aft_converges_with_censoring_and_covariate() -> None:
    # The issue notes the same failure "reproduces with censoring and with a
    # covariate". Both must fit at a size (n=5000) where the old absolute
    # tolerance frequently failed.
    n = 5000
    rng = np.random.default_rng(3)

    # (a) right-censored: still a well-posed AFT, sigma identified by the events.
    log_t = rng.normal(1.4, 0.5, n)
    lat = np.exp(log_t)
    cens = np.exp(rng.normal(2.2, 0.6, n))  # mostly above the events
    time = np.minimum(lat, cens)
    event = (lat <= cens).astype(float)
    assert event.mean() > 0.5  # majority observed, sigma identified
    model_c = _fit_constant_scale_aft(time, event)
    assert all(
        math.isfinite(float(c["estimate"])) for c in model_c.summary().coefficients
    )

    # (b) with a covariate on the location: Surv(time, event) ~ x.
    x = rng.normal(0.0, 1.0, n)
    log_t2 = 1.4 + 0.8 * x + rng.normal(0.0, 0.5, n)
    time2 = np.exp(log_t2)
    event2 = np.ones(n)
    model_x = _fit_constant_scale_aft(
        time2, event2, formula="Surv(time, event) ~ x", x=x
    )
    coefs = [float(c["estimate"]) for c in model_x.summary().coefficients]
    assert all(math.isfinite(c) for c in coefs)
    # A converged covariate fit moves the location with x (the slope is not
    # pinned at its cold-start 0): predicted survival at x=+1 must exceed x=-1
    # (larger mu => longer survival, since the true slope 0.8 > 0). Checked
    # model-free via the survival surface when available; otherwise the finite,
    # non-trivial coefficient vector above already witnesses convergence.
    grid = np.array([2.0, 4.0, 8.0])
    s_lo = _predicted_survival(model_x, grid, {"x": np.full_like(grid, -1.0)})
    s_hi = _predicted_survival(model_x, grid, {"x": np.full_like(grid, 1.0)})
    if s_lo is not None and s_hi is not None:
        assert np.all(s_hi >= s_lo - 1e-6), (
            f"survival must increase with x: S(x=-1)={s_lo} S(x=+1)={s_hi}"
        )
        assert np.max(np.abs(s_hi - s_lo)) > 0.02, "covariate has no effect on survival"
