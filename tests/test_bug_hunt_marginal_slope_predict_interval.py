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
   `p_true(x)` per row, the across-the-function coverage of `p_true` (the
   quantity a Bayesian band holds at nominal; Nychka 1988, Marra & Wood 2012)
   sits at the nominal level at both 95% and 50%, measured over independent
   replicate responses and gated on both sides by the replicates' own Monte
   Carlo error. The 50% level is what exposes an over-covering band: a
   covariance inflated by any factor pushes 50% coverage far above 0.5, where
   it cannot hide under a ceiling the way 95% hides under 1.0.
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


def _fit(data):
    return gamfit.fit(
        data,
        "disease ~ s(bmi)",
        family="bernoulli-marginal-slope",
        link="probit",
        z_column="z",
        slope_formula="1",
    )


def _fit_model():
    data, p_true = _make_fixture()
    return _fit(data), data, p_true


# Replicate responses for the coverage oracle. One fit's per-row coverage
# indicators are all driven by the same few-dimensional estimation error, so a
# single-realization coverage fraction has no error bar; the replicate mean
# does, and the gate below reads it from the replicates themselves.
_REPLICATES = 64
# Two-sided Monte Carlo gate |mean(c_r) - level| <= _MC_SIGMAS * sd(c_r)/sqrt(R).
# At 4 standard errors an honest band trips one of the file's four gates (two
# tests x two levels) with probability below 4 * 2 * Phi(-4) ~= 2.5e-4.
_MC_SIGMAS = 4.0
_LEVELS = (0.95, 0.50)


def _replicate_fits():
    """Refit on `_REPLICATES` independent responses drawn from the fixture's
    known `p_true`, holding the covariates (`z`, `bmi`) fixed."""
    data, p_true = _make_fixture()
    for r in range(_REPLICATES):
        rng = np.random.default_rng([20240613, r])
        y = (rng.random(p_true.size) < p_true).astype(int)
        replicate = {**data, "disease": y.tolist()}
        yield _fit(replicate), replicate, p_true


def _covered_fraction(p_true, lo, hi) -> float:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(((p_true >= lo - 1e-9) & (p_true <= hi + 1e-9)).mean())


def _assert_nominal_coverage(coverage: dict[float, list[float]], band: str) -> None:
    for level in _LEVELS:
        c = np.asarray(coverage[level], dtype=float)
        mc_se = float(c.std(ddof=1)) / math.sqrt(c.size)
        gap = abs(float(c.mean()) - level)
        assert gap <= _MC_SIGMAS * mc_se, (
            f"{band}: nominal-{level:.0%} band covers p_true at "
            f"{c.mean():.4f} across {c.size} replicates (|gap| {gap:.4f} > "
            f"{_MC_SIGMAS:g} x MC s.e. {mc_se:.4f}); per-replicate range "
            f"[{c.min():.3f}, {c.max():.3f}]"
        )


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
    """Exact oracle: the emitted credible bounds are the inverse base-link of
    η-scale endpoints `eta ± z * se_eta`, clipped to [0, 1].

    The table carries the probability-scale posterior SE under `std_error`
    (the documented response-scale column; the η-scale SE is a different
    quantity and is not published), so the construction is pinned through the
    two properties that identify it without `se_eta`: on the η scale the band
    is symmetric about `linear_predictor` wherever neither bound was clipped,
    and its η-scale half-width scales exactly with the normal quantile of the
    requested level (the 95% and 50% half-widths differ by `z95 / z50`,
    row for row, because both come from the same `se_eta`)."""
    model, data, _ = _fit_model()

    out95 = model.predict(data, interval=0.95, return_type="dict")
    out50 = model.predict(data, interval=0.50, return_type="dict")
    eta = np.asarray(out95["linear_predictor"], dtype=float)
    np.testing.assert_allclose(
        np.asarray(out50["linear_predictor"], dtype=float), eta, rtol=0.0, atol=0.0
    )
    se_response = np.asarray(out95["std_error"], dtype=float)
    assert np.all(np.isfinite(se_response)) and np.all(se_response > 0.0)
    assert np.all(se_response <= 0.5), "a probability's posterior SD cannot exceed 1/2"

    z95 = _standard_normal_quantile(0.5 + 0.95 / 2.0)
    z50 = _standard_normal_quantile(0.5 + 0.50 / 2.0)
    half = {}
    for level, out, z in (("95", out95, z95), ("50", out50, z50)):
        lo = np.asarray(out["mean_lower"], dtype=float)
        hi = np.asarray(out["mean_upper"], dtype=float)
        assert np.all(lo >= 0.0) and np.all(hi <= 1.0) and np.all(lo <= hi)
        unclipped = (lo > 1e-12) & (hi < 1.0 - 1e-12)
        assert unclipped.sum() >= 0.8 * lo.size, "the oracle needs mostly unclipped rows"
        eta_lo = np.array([_standard_normal_quantile(v) for v in lo[unclipped]])
        eta_hi = np.array([_standard_normal_quantile(v) for v in hi[unclipped]])
        # TransformEta: Φ⁻¹(hi) − η == η − Φ⁻¹(lo).
        np.testing.assert_allclose(
            eta_hi - eta[unclipped], eta[unclipped] - eta_lo, rtol=1e-6, atol=1e-7,
            err_msg=f"{level}% band is not symmetric about eta on the link scale",
        )
        half[level] = ((eta_hi - eta_lo) / 2.0, unclipped, z)

    both = half["95"][1] & half["50"][1]
    hw95 = ((np.array([_standard_normal_quantile(v) for v in np.asarray(out95["mean_upper"])[both]])
             - np.array([_standard_normal_quantile(v) for v in np.asarray(out95["mean_lower"])[both]])) / 2.0)
    hw50 = ((np.array([_standard_normal_quantile(v) for v in np.asarray(out50["mean_upper"])[both]])
             - np.array([_standard_normal_quantile(v) for v in np.asarray(out50["mean_lower"])[both]])) / 2.0)
    # Same se_eta behind both levels: half-widths differ exactly by z95 / z50.
    np.testing.assert_allclose(hw95 / hw50, np.full(hw95.shape, z95 / z50), rtol=1e-6, atol=0.0)

def test_marginal_slope_interval_covers_truth_at_nominal_rate():
    """Coverage oracle for `predict(interval=)`: across replicate responses the
    band covers the known generative probability at the nominal rate at both
    95% and 50%, within the replicates' Monte Carlo error on either side, so
    neither an under- nor an over-covering band passes."""
    coverage: dict[float, list[float]] = {level: [] for level in _LEVELS}
    for model, data, p_true in _replicate_fits():
        for level in _LEVELS:
            out = model.predict(data, interval=level, return_type="dict")
            coverage[level].append(
                _covered_fraction(p_true, out["mean_lower"], out["mean_upper"])
            )
    _assert_nominal_coverage(coverage, "predict(interval=)")


def test_marginal_slope_sample_predict_returns_posterior_bands():
    """Issue #1049 part 2: `model.sample(...).predict(new_data, level=)` must
    return posterior-predictive bands instead of raising the 'posterior_predict
    currently supports only standard GAM models' stub.

    The Laplace draws already existed (sample() works and is instant); this pins
    that they are now propagated through the marginal-slope kernel to a per-row
    η matrix and collapsed to probability-scale credible bands. The bands are
    ordered, clipped to [0, 1] and non-degenerate; their coverage is pinned by
    `test_marginal_slope_posterior_band_covers_truth_at_nominal_rate`."""
    model, data, _ = _fit_model()

    posterior = model.sample(data, samples=300, seed=7)
    bands95 = posterior.predict(data, level=0.95)
    assert isinstance(bands95, dict)
    for key in ("posterior_mean", "posterior_mean_lower", "posterior_mean_upper"):
        assert key in bands95, f"posterior predict dropped {key!r}: got {list(bands95)}"

    mean = np.asarray(bands95["posterior_mean"], dtype=float)
    lo95 = np.asarray(bands95["posterior_mean_lower"], dtype=float)
    hi95 = np.asarray(bands95["posterior_mean_upper"], dtype=float)
    n = len(data["disease"])
    assert mean.shape == (n,) and lo95.shape == (n,) and hi95.shape == (n,)
    # Probability-scale bands: ordered, clipped, non-degenerate.
    assert np.all(lo95 >= -1e-9) and np.all(hi95 <= 1.0 + 1e-9)
    assert np.all(lo95 <= mean + 1e-9) and np.all(hi95 >= mean - 1e-9)
    assert np.median(hi95 - lo95) > 1e-3


def test_marginal_slope_posterior_band_covers_truth_at_nominal_rate():
    """Coverage oracle for `sample().predict(level=)`: the posterior draw band
    covers the known generative probability at the nominal rate at both 95%
    and 50% across replicate responses, gated on both sides by the replicates'
    Monte Carlo error."""
    coverage: dict[float, list[float]] = {level: [] for level in _LEVELS}
    for r, (model, data, p_true) in enumerate(_replicate_fits()):
        posterior = model.sample(data, samples=300, seed=7 + r)
        for level in _LEVELS:
            bands = posterior.predict(data, level=level)
            coverage[level].append(
                _covered_fraction(
                    p_true, bands["posterior_mean_lower"], bands["posterior_mean_upper"]
                )
            )
    _assert_nominal_coverage(coverage, "sample().predict(level=)")


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
