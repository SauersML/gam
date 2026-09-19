"""Seeded calibration of the two model-comparison p-value surfaces (pyGAM audit, lane
pv-model-comparison).

The full Monte Carlo study (1000 replicates per null cell, 500 per power
cell) lives in ``bench/pvalue_calibration/pv-model-comparison/``. This file
is its small, seeded regression.

``basis_check``
    The penalized score (Rao) lack-of-fit p-value in ``Summary.basis_checks``
    and ``Model.basis_check``. With an adequate basis (a smooth truth and the
    default ``s(x)``) it must not reject more often than its level. With too
    small a basis (``k=4`` against ``sin(6x)``, n=2000) it must reject.

    Each null test counts rejections at level ``a`` and asserts the count is
    at most the 99.9% quantile of Binomial(tested, a). A correctly sized test
    therefore fails this assertion once in a thousand seeds, and the fixed
    seeds used here were measured below that bound. A reference that rejects
    at twice its level fails it.

``compare_models``
    It ranks nested penalized fits by an information criterion and returns
    no p-value. The naive likelihood-ratio chi^2 on the EDF difference ignores
    both penalization and smoothing-parameter selection, and is
    anti-conservative (Wood, Pya & Saefken 2016). No reference
    that corrects for both is implemented, so the comparison offers no
    p-value, and the last test pins that.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from scipy import stats

gamfit = pytest.importorskip("gamfit")

SEED = 20260919


def _draw(family: str, eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if family == "gaussian":
        return eta + rng.normal(0.0, 0.5, eta.size)
    if family == "binomial":
        return rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return rng.poisson(np.exp(eta)).astype(float)


def _basis_check_p_values(family, n, formula, truth, reps, seed):
    """Return the basis_check p-values for ``reps`` seeded replicates.

    A replicate whose fit the engine refuses (``ConvergenceError``) is counted rather
    than tested: a refused fit publishes no basis check, and fit robustness is
    not what this file calibrates.
    """
    rng = np.random.default_rng(seed)
    p_values, refused = [], 0
    for _ in range(reps):
        x = rng.uniform(0.0, 1.0, n)
        y = _draw(family, truth(x), rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = gamfit.fit({"x": x, "y": y}, formula, family=family)
            except gamfit.errors.ConvergenceError:
                refused += 1
                continue
        row = model.summary().basis_checks[0]
        assert row["provenance"] == "radial_enrichment", row
        p_values.append(float(row["p_value"]))
    return np.asarray(p_values), refused


def _assert_sized(p_values: np.ndarray, level: float) -> None:
    rejections = int(np.sum(p_values <= level))
    bound = int(stats.binom.ppf(0.999, p_values.size, level))
    assert rejections <= bound, (
        f"{rejections} of {p_values.size} null replicates rejected at {level}; "
        f"a test of size {level} exceeds {bound} with probability 1e-3"
    )


def _sin2pi(x):
    return np.sin(2.0 * np.pi * x)


@pytest.mark.parametrize(
    ("family", "reps"),
    [
        # Measured at this seed, rejections at 0.10/0.05/0.01: 19/6/3 of 200
        # (gaussian), 7/6/1 of 100 (binomial). The 99.9% bounds are 34/21/8 and
        # 20/13/5.
        ("gaussian", 200),
        ("binomial", 100),
    ],
)
def test_basis_check_is_sized_under_an_adequate_basis(family, reps):
    p_values, refused = _basis_check_p_values(family, 200, "y ~ s(x)", _sin2pi, reps, SEED)
    assert refused <= 5, f"{refused} of {reps} fits refused"
    for level in (0.10, 0.05, 0.01):
        _assert_sized(p_values, level)


def test_basis_check_rejects_a_basis_too_small_for_the_truth():
    # A k=4 fit of sin(6x) at n=2000. The 500-replicate bench measures power
    # of 0.84 at 0.05. Measured at this seed: 13 of 20 rejections.
    p_values, refused = _basis_check_p_values(
        "gaussian", 2000, "y ~ s(x, k=4)", lambda x: np.sin(6.0 * x), 20, SEED + 1
    )
    assert refused <= 2, f"{refused} of 20 fits refused"
    rejections = int(np.sum(p_values <= 0.05))
    # The count must exceed what a size-0.05 test with no power reaches with
    # probability 1e-3.
    bound = int(stats.binom.ppf(0.999, p_values.size, 0.05))
    assert rejections > bound, f"only {rejections} of {p_values.size} rejected at 0.05"


def test_basis_check_is_deterministic_and_matches_the_summary():
    rng = np.random.default_rng(SEED + 2)
    x = rng.uniform(0.0, 1.0, 2000)
    data = {"x": x, "y": _draw("binomial", _sin2pi(x), rng)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = gamfit.fit(data, "y ~ s(x)", family="binomial")
        second = gamfit.fit(data, "y ~ s(x)", family="binomial")
    persisted = first.summary().basis_checks
    # A refit of the same rows publishes bit-identical rows: nothing in the
    # enrichment, the row selection or the reference draws on a random state.
    assert persisted == second.summary().basis_checks
    recomputed = first.basis_check(data)
    assert recomputed == first.basis_check(data)
    # Recomputing from the training rows re-solves the inner problem at the
    # stored smoothing parameters, so it agrees to solver tolerance, not bits.
    (row,), (saved,) = recomputed, persisted
    for key in ("name", "basis_dim", "nullspace_dim", "enrichment_dim", "enrichment_rank", "provenance"):
        assert row[key] == saved[key], key
    for key in ("edf", "statistic", "p_value"):
        assert row[key] == pytest.approx(saved[key], rel=1e-6), key


def _keys(document):
    if isinstance(document, dict):
        for key, value in document.items():
            yield str(key)
            yield from _keys(value)
    elif isinstance(document, (list, tuple)):
        for value in document:
            yield from _keys(value)


def test_compare_models_offers_no_nested_p_value():
    rng = np.random.default_rng(SEED + 3)
    n = 400
    x, z = rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)
    data = {"x": x, "z": z, "y": _sin2pi(x) + rng.normal(0.0, 0.5, n)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        small = gamfit.fit(data, "y ~ s(x)")
        large = gamfit.fit(data, "y ~ s(x) + s(z)")
    document = gamfit.compare_models([small, large], names=["small", "large"])
    keys = {key.lower().replace("-", "_") for key in _keys(document)}
    assert not {k for k in keys if "p_value" in k or k in {"pvalue", "p"}}, sorted(keys)
    assert "p_value" not in json.dumps(document).lower().replace("-", "_")
