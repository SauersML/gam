"""Seeded calibration of the two model-comparison p-value surfaces (pyGAM audit, lane
pv-model-comparison).

The full Monte Carlo study lives in ``bench/pvalue_calibration/pv-model-comparison/``.
This file is its small, seeded regression.

``basis_check``
    The penalized score (Rao) lack-of-fit p-value in ``Summary.basis_checks``
    and ``Model.basis_check``. With an adequate basis (a smooth truth and the
    default ``s(x)``) it must be U(0, 1). A conservative p-value is as
    miscalibrated as an anti-conservative one, so every null cell checks both
    tails:

    * at each level ``a`` in 0.10 / 0.05 / 0.01 the rejection count must lie
      inside the central 99.9% interval of Binomial(tested, a), and
    * a two-sided KS test against U(0, 1) must not reject at 1e-3.

    A calibrated test fails a cell with probability about 4e-3 over a random
    seed; the fixed seeds here were measured inside every bound. With too small
    a basis (``k=4`` against ``sin(6x)``, n=2000) it must reject.

    For a canonical binomial or Poisson fit the reference is the score's law
    conditional on the sufficient statistic. Where its expansion leaves its
    range of validity the row reports ``conditional_reference_unavailable``
    and no p-value; those rows are counted, not tested, and their rate is
    bounded by the rate the bench measured.

``compare_models``
    It ranks nested penalized fits by an information criterion and returns
    no p-value. The naive likelihood-ratio chi^2 on the EDF difference ignores
    both penalization and smoothing-parameter selection, and is
    anti-conservative (Wood, Pya & Saefken 2016). No reference
    that corrects for both is implemented, so the comparison offers no
    p-value, and the last test pins that.
"""

from __future__ import annotations

import collections
import json
import warnings

import numpy as np
import pytest
from scipy import stats

gamfit = pytest.importorskip("gamfit")

SEED = 20260919
LEVELS = (0.10, 0.05, 0.01)
# Two-sided tail mass of every calibration bound in this file.
TAIL = 1e-3
TESTED = "radial_enrichment"
NOT_MEASURED = "conditional_reference_unavailable"


def _draw(family: str, eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if family == "gaussian":
        return eta + rng.normal(0.0, 0.5, eta.size)
    if family == "binomial":
        return rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return rng.poisson(np.exp(eta)).astype(float)


def _basis_check_p_values(family, n, formula, truth, reps, seed):
    """Return the basis_check p-values and provenance counts for ``reps`` replicates.

    A replicate whose fit the engine refuses (``FitError``) is counted under
    ``"fit_refused"``: a refused fit publishes no basis check, and fit
    robustness is not what this file calibrates.
    """
    rng = np.random.default_rng(seed)
    p_values, provenance = [], collections.Counter()
    for _ in range(reps):
        x = rng.uniform(0.0, 1.0, n)
        y = _draw(family, truth(x), rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = gamfit.fit({"x": x, "y": y}, formula, family=family)
            except gamfit.errors.FitError:
                provenance["fit_refused"] += 1
                continue
        row = model.summary().basis_checks[0]
        provenance[row["provenance"]] += 1
        # A row that is not measured omits its p-value.
        p_value = row.get("p_value")
        assert (p_value is not None) == (row["provenance"] == TESTED), row
        if p_value is not None:
            p_values.append(float(p_value))
    return np.asarray(p_values), provenance


def _assert_calibrated(p_values: np.ndarray) -> None:
    tested = p_values.size
    for level in LEVELS:
        rejections = int(np.sum(p_values <= level))
        low = int(stats.binom.ppf(TAIL / 2, tested, level))
        high = int(stats.binom.isf(TAIL / 2, tested, level))
        assert low <= rejections <= high, (
            f"{rejections} of {tested} null replicates rejected at {level}; a test of "
            f"size {level} lands outside [{low}, {high}] with probability {TAIL}"
        )
    ks = stats.kstest(p_values, "uniform")
    assert ks.pvalue >= TAIL, f"KS against U(0, 1): D = {ks.statistic:.4f}, p = {ks.pvalue:.2e}"


def _sin2pi(x):
    return np.sin(2.0 * np.pi * x)


def _low_rate(x):
    # Poisson means 0.14 to 1.0, or success probabilities 0.12 to 0.5: the
    # regime in which the unconditional chi^2 reference is visibly off.
    return np.sin(2.0 * np.pi * x) - 1.0


@pytest.mark.parametrize(
    ("family", "truth", "reps", "not_measured_rate"),
    [
        pytest.param("gaussian", _sin2pi, 300, 0.0, id="gaussian"),
        # The not-measured rates were measured on independent seeds: 346 of
        # 5000 binomial replicates, 286 of 2000 low-count Poisson and 433 of
        # 2000 low-rate binomial ones. The bench's 1000-replicate cells read
        # 66, 149 and 190 per 1000.
        pytest.param("binomial", _sin2pi, 300, 346 / 5000, id="binomial"),
        # The unconditional chi^2 reference this replaced fails the Poisson cell
        # at this seed (KS p = 6.5e-6). Its binomial deficit (75 of 1000 at
        # 0.10, against a floor of 70) is inside this bound; the bench's pooled
        # runs resolve it. Fewer replicates see neither, hence the cost.
        pytest.param(
            "poisson", _low_rate, 1000, 286 / 2000, id="poisson-low-count", marks=pytest.mark.slow
        ),
        pytest.param(
            "binomial", _low_rate, 1000, 433 / 2000, id="binomial-low-rate", marks=pytest.mark.slow
        ),
    ],
)
def test_basis_check_is_uniform_under_an_adequate_basis(family, truth, reps, not_measured_rate):
    p_values, provenance = _basis_check_p_values(family, 200, "y ~ s(x)", truth, reps, SEED)
    assert set(provenance) <= {TESTED, NOT_MEASURED, "fit_refused"}, provenance
    measured = reps - provenance["fit_refused"]
    # The rate at which the conditional reference declines is a property of the
    # regime; it is bounded by the rate the bench measured, not assumed zero.
    ceiling = int(stats.binom.isf(TAIL, measured, not_measured_rate)) if not_measured_rate else 0
    assert provenance[NOT_MEASURED] <= ceiling, provenance
    _assert_calibrated(p_values)


def test_basis_check_rejects_a_basis_too_small_for_the_truth():
    # A k=4 fit of sin(6x) at n=2000. The 500-replicate bench measures power
    # of 0.86 at 0.05. Measured at this seed: 13 of 20 rejections.
    p_values, provenance = _basis_check_p_values(
        "gaussian", 2000, "y ~ s(x, k=4)", lambda x: np.sin(6.0 * x), 20, SEED + 1
    )
    assert provenance[TESTED] == 20, provenance
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
