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

This test fits a clean Beta model, confirms the posterior MEAN reproduces the
point estimate (so only the spread is wrong), then asserts each coefficient's
posterior SD is within a factor of 2 of the Wald SE. It currently fails (ratio
~0.21); when the fallback uses the coefficient-covariance scale it passes.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _beta_frame(seed: int = 0, n: int = 4000) -> pd.DataFrame:
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

    draws = np.asarray(model.sample(df, samples=4000, seed=1).samples, dtype=float)
    post_mean = draws.mean(axis=0)
    post_sd = draws.std(axis=0)

    # The posterior MEAN must reproduce the point estimate: this isolates the
    # defect to the spread (it is purely a scale bug, not a location bug).
    assert np.allclose(post_mean, estimate, atol=5e-3), (
        f"posterior mean {post_mean} should match the point estimate {estimate}"
    )

    # The contract: posterior SD and Wald SE are the same covariance diagonal.
    # They must agree within a factor of 2. Currently post_sd ≈ 0.21 * wald_se.
    ratio = post_sd / wald_se
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0), (
        "Beta posterior SD must match the Wald SE within a factor of 2; got "
        f"post_sd={post_sd}, wald_se={wald_se}, ratio={ratio} "
        f"(~sqrt(1/(1+phi)) under-dispersion)"
    )
