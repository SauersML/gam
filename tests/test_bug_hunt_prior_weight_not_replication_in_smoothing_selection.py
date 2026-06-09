"""Regression: for a FIXED-dispersion family, a prior weight ``w == c`` must be
exactly equivalent to ``c``-fold row replication — in the *smoothing-parameter
selection*, not only in the point estimate.

For a Poisson (or any fixed-dispersion, scale==1) GAM, encoding ``c`` identical
copies of every observation as a single row carrying prior weight ``c`` is
mathematically indistinguishable from literally stacking the row ``c`` times:

* The penalised log-likelihood / deviance is identical: ``D_p(beta)`` is a sum
  of per-observation terms and ``c * sum_i d_i == sum over the c*n replicated
  rows``.
* The penalised Fisher information / Hessian is identical: the weighted
  cross-product ``X^T W X`` with ``W = c * diag(mu)`` equals the cross-product
  accumulated over the ``c``-fold replicated design.
* The LAML smoothing-selection objective the engine minimises for a fixed
  dispersion is ``D_p/2 + 0.5*log|X^T W X + S_lambda| - 0.5*log|S_lambda|_+``
  (``src/solver/reml/unified.rs``, the ``DispersionHandling::Fixed`` arm). Every
  term in it is a function of ``D_p`` and ``X^T W X``, both encoding-invariant.
  It carries NO bare row-count ``n`` term (unlike the profiled-Gaussian arm), so
  the minimiser ``lambda_hat`` is provably identical for the two encodings.

The library already honours this for the POINT ESTIMATE: a purely parametric
Poisson fit ``y ~ x`` returns byte-identical coefficients for ``w == c`` and for
``c``-fold replication (the weighted cross-products match exactly). But once a
penalised smooth ``s(x)`` is present, the OUTER smoothing-parameter search does
NOT: the weighted encoding selects a systematically LARGER ``lambda`` (more
smoothing) and a SMALLER effective dof than the replicated encoding. Across 24
independent seeds the weighted fit over-smooths in 24/24 (mean lambda ratio
~2x, up to ~22x), so this is a systematic objective defect, not optimiser noise.

The point estimate and the smoothing selection therefore disagree about what a
prior weight *means*: the inner penalised fit treats ``w == c`` as exact
replication, the outer REML/LAML selection does not.

This test asserts only the encoding-invariance the two paths must agree on; it
is agnostic to whichever weight convention the maintainer settles on, because
both paths must encode the SAME convention.
"""
from __future__ import annotations

import numpy as np

import gamfit


def _edf_and_grid(data, weights=None):
    kwargs = {"family": "poisson"}
    if weights is not None:
        kwargs["weights"] = weights
    model = gamfit.fit(data, "y ~ s(x, k=12)", **kwargs)
    summary = model.summary()
    grid = {"x": np.linspace(-2.5, 2.5, 64)}
    fitted = np.asarray(model.predict(grid))
    return float(summary.edf_total), fitted


def _datasets(seed: int, n: int, c: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    mu = np.exp(0.3 + 0.9 * np.sin(1.4 * x))
    y = rng.poisson(mu).astype(float)
    weighted = {"x": x, "y": y, "w": np.full(n, float(c))}
    rep = {"x": np.tile(x, c), "y": np.tile(y, c)}
    return weighted, rep


def test_uniform_prior_weight_equals_replication_in_smoothing_selection():
    c = 3
    weighted, rep = _datasets(seed=1, n=600, c=c)

    edf_weighted, fit_weighted = _edf_and_grid(weighted, weights="w")
    edf_rep, fit_rep = _edf_and_grid(rep)

    # The two encodings are the same Poisson likelihood, so the selected
    # effective degrees of freedom must coincide. The bug makes the weighted
    # encoding over-smooth (lower EDF) by ~0.05-0.35; a correct implementation
    # matches to optimiser precision.
    assert abs(edf_weighted - edf_rep) < 1.0e-2, (
        f"prior weight w={c} selects a different smoothness than {c}-fold "
        f"replication: edf(weighted)={edf_weighted:.4f} vs "
        f"edf(replicated)={edf_rep:.4f} (delta={edf_weighted - edf_rep:+.4f})"
    )

    # The fitted partial-effect curve must coincide as well.
    max_fit_gap = float(np.max(np.abs(fit_weighted - fit_rep)))
    assert max_fit_gap < 5.0e-3, (
        f"prior weight w={c} and {c}-fold replication produce different fitted "
        f"Poisson means: max|Delta| over the grid = {max_fit_gap:.4e}"
    )


def test_weighted_does_not_systematically_oversmooth_vs_replication():
    """Direction check across seeds: a correct implementation has no systematic
    sign to edf(weighted) - edf(replicated). The bug makes it negative every
    time (weighted always smooths more)."""
    c = 3
    deltas = []
    for seed in range(8):
        weighted, rep = _datasets(seed=seed, n=400, c=c)
        edf_weighted, _ = _edf_and_grid(weighted, weights="w")
        edf_rep, _ = _edf_and_grid(rep)
        deltas.append(edf_weighted - edf_rep)
    deltas = np.asarray(deltas)
    # Under correct behaviour every delta is ~0; the bug makes all of them
    # negative. Reject a unanimous negative sign with non-negligible magnitude.
    n_oversmooth = int(np.sum(deltas < -1.0e-3))
    assert n_oversmooth < len(deltas), (
        "prior-weight smoothing selection over-smooths relative to replication "
        f"in {n_oversmooth}/{len(deltas)} seeds (edf deltas={np.round(deltas, 4)})"
    )
