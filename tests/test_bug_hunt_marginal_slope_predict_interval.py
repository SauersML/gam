"""Issue #1049 — predictive intervals for the Bernoulli marginal-slope model.

`predict(interval=...)` was a silent no-op for `bernoulli-marginal-slope`: the
Rust posterior-mean path *does* emit `std_error` and the response-scale
credible bounds (`mean_lower` / `mean_upper`) from the marginal-slope
coefficient covariance, but the Python `_predict_shape` dispatcher dropped every
column except `mean`, so the user never saw an interval.

This module pins the fix with two oracles, both on a fitted model (no fakes):

1. **Exact-construction oracle** — the emitted `mean_lower` / `mean_upper` are
   the `TransformEta` credible bounds, i.e. the inverse base-link applied to the
   η-scale endpoints `eta ± z * std_error`, clipped to `[0, 1]`. We rebuild that
   from the *also-emitted* `linear_predictor` (η) and `std_error` (η-scale SE)
   with an independent `z` from the standard-normal quantile and assert a tight
   match. This is the `z · SE from Vp` independent construction the issue asks
   for: `std_error` is the diagonal of `X Vp Xᵀ` on the η scale.

2. **Coverage oracle** — on a fixture with a *known* generative probability
   `p_true(x)` per row, the nominal-95% band covers `p_true` at approximately the
   nominal rate (and the 50% band is meaningfully tighter than the 95% band, so
   the width responds honestly to the level).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import gamfit


def _standard_normal_quantile(p: float) -> float:
    # Acklam's rational approximation, refined by one Halley step against the
    # error function — independent of the Rust quantile used inside gam, so the
    # construction oracle does not borrow gam's own z.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    # One Halley refinement.
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _probit(eta: float) -> float:
    return 0.5 * math.erfc(-eta / math.sqrt(2))


def _make_fixture(n: int = 600, seed: int = 20240613):
    """A Bernoulli marginal-slope DGP with a known per-row generative
    probability. The latent score `z` is standardized N(0,1); the binary
    outcome's probit linear predictor is `b0 + slope * z + g(bmi)`, so the
    *known* generative probability is `Phi(eta_true)`.
    """
    rng = np.random.default_rng(seed)
    raw_z = rng.normal(0.0, 1.0, size=n)
    z = (raw_z - raw_z.mean()) / raw_z.std()
    bmi = 25.0 + rng.normal(0.0, 3.0, size=n)
    # Generative probit linear predictor (the truth we will check coverage of).
    eta_true = -0.3 + 0.7 * z + 0.04 * (bmi - 25.0)
    p_true = np.array([_probit(e) for e in eta_true])
    y = (rng.random(n) < p_true).astype(int)
    data = {
        "disease": y.tolist(),
        "z": z.tolist(),
        "bmi": bmi.tolist(),
    }
    return data, p_true


def _fit_model():
    data, p_true = _make_fixture()
    model = gamfit.fit(
        data,
        "disease ~ s(bmi)",
        family="bernoulli-marginal-slope",
        link="probit",
        z_column="z",
        logslope_formula="1",
    )
    return model, data, p_true


def test_marginal_slope_predict_emits_interval_columns():
    """`predict(interval=0.95)` must now return a table carrying the credible
    band, not a bare 1-D probability vector."""
    model, data, _ = _fit_model()

    out = model.predict(data, interval=0.95, return_type="dict")
    assert isinstance(out, dict)
    for key in ("mean", "std_error", "mean_lower", "mean_upper"):
        assert key in out, f"interval predict dropped column {key!r}: got {list(out)}"

    mean = np.asarray(out["mean"], dtype=float)
    lo = np.asarray(out["mean_lower"], dtype=float)
    hi = np.asarray(out["mean_upper"], dtype=float)
    # Probability-scale bounds: clipped to [0, 1] and ordered around the mean.
    assert np.all(lo >= -1e-9) and np.all(hi <= 1.0 + 1e-9)
    assert np.all(lo <= mean + 1e-9) and np.all(hi >= mean - 1e-9)
    # Non-degenerate band (the fix is not emitting a zero-width interval).
    assert np.median(hi - lo) > 1e-3


def test_marginal_slope_interval_matches_transform_eta_construction():
    """Exact oracle: the emitted credible bounds equal the inverse base-link of
    the η endpoints `eta ± z * std_error`, clipped to [0, 1].

    `std_error` is the η-scale posterior SE `sqrt(diag(X Vp Xᵀ))` from the
    marginal-slope coefficient covariance; reconstructing the band from it with
    an *independent* z and the probit inverse link reproduces gam's bounds to
    floating-point tolerance. This is the `z · SE from Vp` check the issue asks
    for."""
    model, data, _ = _fit_model()

    out = model.predict(data, interval=0.95, return_type="dict")
    eta = np.asarray(out["linear_predictor"], dtype=float)
    se = np.asarray(out["std_error"], dtype=float)
    lo = np.asarray(out["mean_lower"], dtype=float)
    hi = np.asarray(out["mean_upper"], dtype=float)

    z = _standard_normal_quantile(0.5 + 0.95 / 2.0)
    lo_oracle = np.clip(np.array([_probit(e - z * s) for e, s in zip(eta, se)]), 0.0, 1.0)
    hi_oracle = np.clip(np.array([_probit(e + z * s) for e, s in zip(eta, se)]), 0.0, 1.0)

    np.testing.assert_allclose(lo, lo_oracle, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(hi, hi_oracle, atol=1e-7, rtol=1e-6)


def test_marginal_slope_interval_covers_truth_at_nominal_rate():
    """Coverage oracle: the 95% band covers the known generative probability at
    approximately the nominal rate, and the 50% band is materially tighter than
    the 95% band (the width responds to the requested level)."""
    model, data, p_true = _fit_model()

    out95 = model.predict(data, interval=0.95, return_type="dict")
    lo95 = np.asarray(out95["mean_lower"], dtype=float)
    hi95 = np.asarray(out95["mean_upper"], dtype=float)
    covered = (p_true >= lo95 - 1e-9) & (p_true <= hi95 + 1e-9)
    coverage = float(covered.mean())
    # The band quantifies *coefficient* (epistemic) uncertainty in the smooth +
    # slope, so on a well-specified DGP it should cover the true probability
    # surface at roughly the nominal rate. Allow a generous tolerance band
    # around 0.95 for finite-sample + smoothing-bias slack while still failing
    # a no-op (which would give zero-width bands and ~0 coverage).
    assert 0.80 <= coverage <= 1.0, f"95% band coverage of p_true was {coverage:.3f}"

    out50 = model.predict(data, interval=0.50, return_type="dict")
    lo50 = np.asarray(out50["mean_lower"], dtype=float)
    hi50 = np.asarray(out50["mean_upper"], dtype=float)
    width95 = float(np.median(hi95 - lo95))
    width50 = float(np.median(hi50 - lo50))
    assert width50 < width95, (
        f"50% band (median width {width50:.4f}) is not tighter than the 95% "
        f"band (median width {width95:.4f}); the level is being ignored"
    )


def test_marginal_slope_sample_predict_returns_posterior_bands():
    """Issue #1049 part 2: `model.sample(...).predict(new_data, level=)` must
    return posterior-predictive bands instead of raising the 'posterior_predict
    currently supports only standard GAM models' stub.

    The Laplace draws already existed (sample() works and is instant); this pins
    that they are now propagated through the marginal-slope kernel to a per-row
    η matrix and collapsed to probability-scale credible bands. The bands are
    ordered, clipped to [0, 1], non-degenerate, and the 50% band is tighter than
    the 95% band, so the posterior predictive responds to the level."""
    model, data, p_true = _fit_model()

    posterior = model.sample(data, samples=300, seed=7)
    bands95 = posterior.predict(data, level=0.95)
    assert isinstance(bands95, dict)
    for key in ("mean", "mean_lower", "mean_upper"):
        assert key in bands95, f"posterior predict dropped {key!r}: got {list(bands95)}"

    mean = np.asarray(bands95["mean"], dtype=float)
    lo95 = np.asarray(bands95["mean_lower"], dtype=float)
    hi95 = np.asarray(bands95["mean_upper"], dtype=float)
    n = len(data["disease"])
    assert mean.shape == (n,) and lo95.shape == (n,) and hi95.shape == (n,)
    # Probability-scale bands: ordered, clipped, non-degenerate.
    assert np.all(lo95 >= -1e-9) and np.all(hi95 <= 1.0 + 1e-9)
    assert np.all(lo95 <= mean + 1e-9) and np.all(hi95 >= mean - 1e-9)
    assert np.median(hi95 - lo95) > 1e-3

    # Coverage of the known generative probability at roughly the nominal rate.
    covered = (p_true >= lo95 - 1e-9) & (p_true <= hi95 + 1e-9)
    coverage = float(covered.mean())
    assert 0.80 <= coverage <= 1.0, (
        f"95% posterior-predictive band coverage of p_true was {coverage:.3f}"
    )

    # The band responds to the requested level.
    bands50 = posterior.predict(data, level=0.50)
    lo50 = np.asarray(bands50["mean_lower"], dtype=float)
    hi50 = np.asarray(bands50["mean_upper"], dtype=float)
    assert float(np.median(hi50 - lo50)) < float(np.median(hi95 - lo95)), (
        "50% posterior-predictive band is not tighter than the 95% band"
    )


def test_marginal_slope_posterior_predict_draws_matrix_shape():
    """`sample().predict_draws(new_data)` must materialize a (n_draws, n_rows)
    η matrix for the marginal-slope model — the per-draw kernel evaluation, not
    the X·β fast path that only applies to standard GAMs."""
    model, data, _ = _fit_model()
    posterior = model.sample(data, samples=128, seed=11)
    draws = posterior.predict_draws(data)
    eta = np.asarray(draws.eta, dtype=float)
    n = len(data["disease"])
    assert eta.shape == (draws.n_draws, n), (
        f"unexpected posterior-predictive eta shape {eta.shape}"
    )
    # Probit response scale: every draw maps to a valid probability.
    mean = 0.5 * np.vectorize(math.erfc)(-eta / math.sqrt(2.0))
    assert np.all(mean >= -1e-9) and np.all(mean <= 1.0 + 1e-9)
    # Posterior variability is real (draws are not collapsed to the mode).
    assert np.any(np.std(eta, axis=0) > 1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
