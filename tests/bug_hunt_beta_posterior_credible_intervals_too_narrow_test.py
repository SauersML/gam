"""Bug hunt: Beta-regression posterior credible intervals are ~4-5x too narrow.

For a well-identified GLM, the posterior coefficient spread from ``model.sample``
must match the Wald/Laplace standard errors reported by ``summary()`` — both are
the diagonal of the same coefficient covariance ``Vb`` (the inverse penalized
Hessian, scaled by the coefficient-covariance scale the fit used for ``Vb``).
A draw cloud that is a constant factor narrower than the Wald SE is a posterior
*scale* bug, and it collapses credible-interval coverage.

Beta regression is the one ``Standard`` family explicitly routed to
``laplace_gaussian_fallback`` (NUTS cannot sample it) —
``crates/gam-inference/src/sample.rs:274``. That fallback rescales its draws by

    let dispersion = fit.dispersion().unwrap_or_default();   // sample.rs:350
    let sqrt_phi = dispersion.sqrt_phi();                    // sample.rs:351
    ...
    samples[(k, i)] = mode[i] + sqrt_phi * delta[i];         // sample.rs:389

i.e. it draws ``N(mode, phi * H^-1)``. For Beta, ``dispersion()`` is
``Dispersion::Known(1 / (1 + phi))`` (the Beta IRLS working weight already folds
``phi`` into the Hessian, so the stored ``H`` is the true penalized precision and
``Vb = H^-1`` with NO extra dispersion factor). The *correct* scale is the
``coefficient_covariance_scale()`` the sibling bounded/NUTS paths use — see
``sample.rs:671`` ("keeps the draw spread identical to the reported
summary().std_error", gam#1514) — which is ``1.0`` for Beta. So the fallback
shrinks every posterior SD by ``sqrt(1 / (1 + phi)) ≈ 0.22`` for ``phi ≈ 19``.

This was verified against the ground truth: across 60 independent datasets the
empirical SD of the slope MLE is ~0.0111; ``summary().std_error`` reports ~0.0126
(correct, ~14% high); the posterior SD is ~0.0028 — about 4x too small. The
draws still self-report ``method='nuts'``, ``is_exact=True``, ``rhat=1.0``,
``converged=True``, so the under-dispersion is silent.

Distinct from the NB-at-seed-theta (#1463) and logistic-normal-oracle (#1459)
siblings: different family, different mechanism (a dispersion double-count in the
Laplace fallback, not a wrong fitted hyperparameter).

This test fits a clean Beta model and asserts both the location and the spread
of the draw cloud against the fit's own published covariance, each at the
sampler's own Monte-Carlo resolution (#4532) rather than at a factor-of-two
band: the defect the file was written for moves the SD ratio by 0.79, and the
bar below is 0.051, so every smaller dispersion mistake of the same kind — a
stray ``sqrt(phi)``, a ``1/phi``, any factor outside ``[0.95, 1.05]`` — now
fails too.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

#: Rows in the fitted frame. Fixes the O(1/n) Laplace-vs-exact posterior gap.
N_ROWS = 4000
#: Draws requested per chain from ``model.sample``.
N_DRAWS = 4000
#: Two-sided normal deviate the bars below are priced at: a 6.8e-6 per-check
#: false-failure rate, small enough that the 4 checks here (2 coefficients x
#: {location, spread}) do not trip on Monte-Carlo noise alone.
SIGMA = 4.5


def _beta_frame(seed: int = 0, n: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    eta = 0.2 + 0.9 * x
    mu = 1.0 / (1.0 + np.exp(-eta))
    phi = 20.0
    y = rng.beta(mu * phi, (1.0 - mu) * phi)
    return pd.DataFrame({"x": x, "y": y})


def test_beta_posterior_sd_matches_wald_se() -> None:
    df = _beta_frame()
    model = gamfit.fit(df, "y ~ x", family="beta")

    summary = model.summary()
    wald_se = np.array([c["std_error"] for c in summary.coefficients], dtype=float)
    estimate = np.array([c["estimate"] for c in summary.coefficients], dtype=float)

    posterior = model.sample(df, samples=N_DRAWS, seed=1)
    draws = np.asarray(posterior.samples, dtype=float)
    post_mean = draws.mean(axis=0)
    # ddof = 1, the same estimator `posterior.std` reports (sample.rs uses
    # `std_axis(Axis(0), 1.0)`), so no estimator mismatch enters the bar.
    post_sd = draws.std(axis=0, ddof=1)

    # The sampler's own effective sample size is the denominator of every
    # Monte-Carlo bar below. `laplace_gaussian_fallback` marks `ess = n_total`
    # because its draws are iid by construction (sample.rs), and `n_total` is
    # exactly the row count of the matrix returned here — so these bars also
    # price that claim: an `ess` reported above the resolution the draws
    # actually carry fails them.
    ess = float(posterior.ess)
    assert ess > 0.0 and ess <= float(draws.shape[0]), (
        f"reported ess={ess} must be positive and at most the {draws.shape[0]} "
        "draws returned"
    )

    # LOCATION. The fallback centres every draw on the fitted mode, which is
    # the `estimate` column of the summary, so the only gap is the Monte-Carlo
    # error of a mean over `ess` draws of SD `wald_se`: `wald_se / sqrt(ess)`.
    # (The old bar here was atol=5e-3, about 0.4 Wald SE and 25 Monte-Carlo SE.)
    mean_band = SIGMA * wald_se / np.sqrt(ess)
    assert np.all(np.abs(post_mean - estimate) <= mean_band), (
        f"posterior mean {post_mean} must reproduce the point estimate "
        f"{estimate} to the draws' own Monte-Carlo resolution "
        f"{mean_band} (|diff| = {np.abs(post_mean - estimate)})"
    )

    # SPREAD. The contract is that the posterior SD and the Wald SE are the
    # same covariance diagonal, so the ratio is 1 up to two derived terms:
    #
    #  * the Monte-Carlo error of a sample SD. For `ess` effectively
    #    independent draws the sample variance is chi-square with `ess - 1`
    #    degrees of freedom, so the sample SD has relative standard error
    #    `1 / sqrt(2 * ess)` — here 0.0112.
    #  * the O(1/n) gap between the exact posterior SD and the Laplace SD the
    #    fit publishes. `y ~ x` carries no smooth, so there is no
    #    smoothing-parameter correction and no `Vp - Vb` term; at n = 4000 the
    #    remaining Laplace error is 2.5e-4 relative.
    #
    # Together: |ratio - 1| <= 4.5 * 1/sqrt(2*ess) + 1/n, about 0.051. The
    # `sqrt(1/(1+phi))` shrink this file was written for gives ratio 0.21, and
    # so does every dispersion factor outside [0.95, 1.05] — a stray `sqrt(phi)`
    # (ratio 4.5), a `phi` (20), a `1/phi` (0.05) or a double-counted 2.
    mc = 1.0 / np.sqrt(2.0 * ess)
    sd_band = SIGMA * mc + 1.0 / float(N_ROWS)
    ratio = post_sd / wald_se
    assert np.all(np.abs(ratio - 1.0) <= sd_band), (
        "Beta posterior SD and Wald SE are the same covariance diagonal, so "
        f"their ratio must be 1 within {sd_band:.4g} (4.5 Monte-Carlo SE at "
        f"ess={ess:.0f}, plus the 1/n Laplace gap); got post_sd={post_sd}, "
        f"wald_se={wald_se}, ratio={ratio} "
        f"(~sqrt(1/(1+phi)) under-dispersion gives 0.21)"
    )
