"""The default double penalty must not over-shrink low-count Poisson (ACC-4).

The pyGAM audit (bench/pygam_audit, accuracy.md ACC-4) measured
``pois_lowcount_n500`` -- ``y ~ Poisson(exp(-1.5 + 1.5 sin(4 pi x)))``, mean
count about 0.3 -- on the 0.1.267 wheel: the default ``y ~ s(x)`` scored a
paired 5-fold truth MSE of 0.00690, against 0.00519 for pyGAM's default and
0.00553 for the same smooth with ``double_penalty=false``. The double penalty
is still the right default: dropping it costs the null cases
(``null3_n200`` 0.045 -> 0.062).

The cause was the functional the null ridge charges, not the Laplace
approximation and not the posterior mean (fold-averaged, plug-in and posterior
mean differ by 1-2%). The 0.1.267 ridge ``m n n^T``, with ``n`` the unit null
vector of the centred roughness, charges ``n^T beta``. ``sin(4 pi x)`` has a
nonzero component along the centred line (``<sin(4 pi x), x - 1/2> =
-1/(4 pi)``), so that ridge charges the truth itself and LAML settles on a
finite null lambda that shrinks it. Since #1561
the ridge charges the mean slope ``f(1) - f(0)``, which is exactly zero on
``sin(4 pi x)``: LAML drives the null lambda to its deletion face and the
smooth keeps its wiggle. Measured on this data: truth MSE 0.00491 (edf 7.7)
against 0.00690 (edf 8.2) on 0.1.267. A NumPy replica of the basis and LAML
reproduces both: 0.00698 with the null-vector ridge, 0.00499 with the
mean-slope ridge.

Both contracts below are the audit's own paired comparisons on its own data
and folds, so they are reference-free apart from pyGAM's stored score. Against
0.1.267 the low-count contract fails (0.00690 > 0.00553 and > 0.00519); the
null contract holds throughout and pins the reason the double penalty stays.
"""

from __future__ import annotations

import numpy as np

import gamfit

# pyGAM 0.12.0 ``LinearGAM``/``PoissonGAM`` default fit on the identical folds,
# as stored by the audit (bench/pygam_audit/accuracy/results).
PYGAM_DEFAULT_POIS_LOWCOUNT_N500_TRUTH_MSE = 0.0051932


def _kfold_test_indices(n: int, folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """``sklearn.model_selection.KFold(folds, shuffle=True, random_state=seed)``."""
    order = np.arange(n)
    np.random.RandomState(seed).shuffle(order)
    sizes = np.full(folds, n // folds)
    sizes[: n % folds] += 1
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    return [order[bounds[k] : bounds[k + 1]] for k in range(folds)]


def _paired_truth_mse(data, mu, formula, family, n):
    test_folds = _kfold_test_indices(n)
    scores = []
    for te in test_folds:
        tr = np.setdiff1d(np.arange(n), te)
        train = {key: col[tr] for key, col in data.items()}
        model = gamfit.fit(train, formula, family=family)
        pred = model.predict(
            {key: col[te] for key, col in data.items() if key != "y"},
            return_type="dict",
        )
        scores.append(np.mean((np.asarray(pred["posterior_mean"]) - mu[te]) ** 2))
    return float(np.mean(scores))


def test_default_double_penalty_keeps_low_count_poisson_wiggle() -> None:
    n = 500
    rng = np.random.default_rng(n + 9)
    x = rng.uniform(0.0, 1.0, n)
    mu = np.exp(-1.5 + 1.5 * np.sin(2.0 * np.pi * 2.0 * x))
    y = rng.poisson(mu).astype(float)
    data = {"x": x, "y": y}

    default = _paired_truth_mse(data, mu, "y ~ s(x)", "poisson", n)
    single = _paired_truth_mse(data, mu, "y ~ s(x, double_penalty=false)", "poisson", n)

    assert default < single, (
        f"default double penalty truth MSE {default:.5f} is above the single "
        f"penalty's {single:.5f}: the null ridge is shrinking a truth whose "
        "ends agree"
    )
    assert default < PYGAM_DEFAULT_POIS_LOWCOUNT_N500_TRUTH_MSE, (
        f"default truth MSE {default:.5f} is above pyGAM's default "
        f"{PYGAM_DEFAULT_POIS_LOWCOUNT_N500_TRUTH_MSE:.5f} on the same folds"
    )


def test_default_double_penalty_still_shrinks_the_null_truth() -> None:
    n = 200
    rng = np.random.default_rng(n + 7)
    X = rng.uniform(0.0, 1.0, (n, 3))
    y = rng.normal(0.0, 1.0, n)
    data = {"a": X[:, 0], "b": X[:, 1], "c": X[:, 2], "y": y}
    mu = np.zeros(n)

    default = _paired_truth_mse(data, mu, "y ~ s(a) + s(b) + s(c)", "gaussian", n)
    single = _paired_truth_mse(
        data,
        mu,
        "y ~ s(a, double_penalty=false) + s(b, double_penalty=false)"
        " + s(c, double_penalty=false)",
        "gaussian",
        n,
    )

    assert default < single, (
        f"default double penalty truth MSE {default:.5f} on a null truth is not "
        f"below the single penalty's {single:.5f}"
    )
