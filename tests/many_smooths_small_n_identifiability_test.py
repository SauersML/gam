"""Many smooths at small n: identifiability is ``n > M_p``, not ``n > p``.

A penalized GAM with ``p`` coefficients is identified by its sample size when
``n`` exceeds ``M_p = p − rank(Σ_k S_k)``, the dimension of the unpenalized
space (intercept, parametric terms, and the penalty null spaces — zero for the
default double-penalized ``s()``). REML/LAML integrate those ``M_p`` directions
out and estimate the smoothing parameters from the ``n − M_p`` residual
contrasts they cannot absorb; every penalized direction is pinned by its
penalty, so ``p > n`` is fine.

The engine used to gate formulas on a heuristic per-smooth row floor summed
over terms, which rejected ``30 × s(x)`` at ``n = 60`` although ``M_p = 1``.
"""

import numpy as np
import pytest

import gamfit


def _many_smooths_data(n, n_terms, *, signal, family=None, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, (n, n_terms))
    eta = np.sin(6.0 * x[:, 0]) if signal else np.zeros(n)
    if family == "binomial":
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-2.0 * eta))).astype(float)
    else:
        y = eta + rng.normal(0.0, 0.3, n)
    data = {f"x{j}": x[:, j] for j in range(n_terms)}
    data["y"] = y
    return data


def _formula(n_terms, k=None):
    kk = f", k={k}" if k is not None else ""
    return "y ~ " + " + ".join(f"s(x{j}{kk})" for j in range(n_terms))


def _term_edfs(model):
    return np.array([term["edf"] for term in model.summary().smooth_terms])


@pytest.mark.parametrize("family", [None, "binomial"])
def test_twenty_smooths_k10_at_n100_converge(family):
    # p ≈ 20 × 9 + 1 = 181 columns against n = 100 rows; M_p = 1.
    n, n_terms = 100, 20
    data = _many_smooths_data(n, n_terms, signal=True, family=family)
    kwargs = {"family": family} if family else {}
    model = gamfit.fit(data, _formula(n_terms, k=10), **kwargs)
    summary = model.summary()
    assert len(model.summary().coefficients) > n
    assert summary.convergence["certified"], summary.convergence
    edfs = _term_edfs(model)
    assert edfs.shape == (n_terms,)
    assert np.all(np.isfinite(edfs))
    # The signal lives in s(x0); total model complexity stays well inside n.
    assert summary.edf_total < n
    assert edfs[0] == edfs.max()


def test_thirty_smooths_at_n60_are_identified():
    # The removed heuristic demanded 2 + 2·30 = 62 rows here; M_p = 1.
    n, n_terms = 60, 30
    data = _many_smooths_data(n, n_terms, signal=True)
    model = gamfit.fit(data, _formula(n_terms))
    assert len(model.summary().coefficients) > n
    assert model.summary().convergence["certified"]


def test_pure_noise_at_p_greater_than_n_shrinks_every_term():
    n, n_terms = 100, 20
    data = _many_smooths_data(n, n_terms, signal=False)
    model = gamfit.fit(data, _formula(n_terms, k=10))
    assert len(model.summary().coefficients) > n
    assert model.summary().convergence["certified"]
    edfs = _term_edfs(model)
    assert np.all(edfs < 0.5), edfs


def test_unpenalized_space_at_least_n_is_a_clear_error():
    # 12 parametric slopes + intercept: M_p = 13 unpenalized directions against
    # n = 13 rows, plus a double-penalized smooth that adds none. n ≤ M_p leaves
    # REML no residual contrast, whatever the smooth does.
    n, n_linear = 13, 12
    rng = np.random.default_rng(3)
    data = {f"z{j}": rng.normal(size=n) for j in range(n_linear)}
    data["x"] = rng.uniform(size=n)
    data["y"] = rng.normal(size=n)
    formula = "y ~ " + " + ".join(f"z{j}" for j in range(n_linear)) + " + s(x)"
    with pytest.raises(gamfit.ModelOverparameterizedError) as info:
        gamfit.fit(data, formula)
    message = str(info.value)
    assert "13 positive-weight rows" in message
    assert "13 unpenalized coefficient directions" in message

    # One more row identifies the same model.
    data = {name: np.append(col, rng.normal()) for name, col in data.items()}
    data["x"][-1] = 0.5
    gamfit.fit(data, formula)


def test_tiny_n_single_smooth_is_a_clear_error_or_fit():
    # #309: n=4 against y ~ s(x) used to surface as an opaque inner-state
    # message. With the double penalty M_p = 1 < 4, so it is identified.
    data = {"x": np.array([0.1, 0.4, 0.6, 0.9]), "y": np.array([0.2, 1.1, 0.8, 0.3])}
    model = gamfit.fit(data, "y ~ s(x)")
    assert model.summary().convergence["certified"]
