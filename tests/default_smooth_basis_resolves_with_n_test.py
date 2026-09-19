"""The default ``s(x)`` resolves its basis from the data (slop.md G1).

The formula-default B-spline used to be capped at ``(unique/4).clamp(4, 8)``
internal knots, so ``y ~ s(x)`` on ``sin(8*pi*x) + N(0, 0.3**2)`` realized 11
columns at every ``n``. Its truth RMSE stayed at ~0.134 from ``n = 1_000`` to
``n = 100_000``, and ``basis_check`` rejected the basis at ``p = 0``. Every fit
also carried a note announcing the knot count.

The default now starts at that pilot resolution and grows through the adaptive
resolution loop until the basis passes its own adequacy test. Growth is bounded
only by the covariate's distinct values and the design rank. These tests pin the
user-visible contract:

* the truth RMSE falls with ``n``, and ``basis_check`` passes;
* a pure-noise response still shrinks to ~0 EDF, and a line to ~1 EDF;
* no per-fit knot-placement note is emitted.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

#: The engine's family-wise basis-adequacy level (``BASIS_ADEQUACY_NOTE_LEVEL``).
ADEQUACY_LEVEL = 1.0e-3


def _fit(n: int, seed: int, truth):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    mean = truth(x)
    data = {"x": x, "y": mean + 0.3 * rng.standard_normal(n)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = gamfit.fit(data, "y ~ s(x)")
    fitted = np.asarray(model.predict(data)).ravel()
    rmse = float(np.sqrt(np.mean((fitted - mean) ** 2)))
    (row,) = model.basis_check(data)
    return rmse, row, [str(w.message) for w in caught] + list(model.notes)


def _sin_8_pi(x):
    return np.sin(8.0 * np.pi * x)


def test_default_smooth_truth_error_falls_with_n_and_basis_check_passes() -> None:
    rmses = []
    for n in (1_000, 10_000, 100_000):
        rmse, row, messages = _fit(n, 2, _sin_8_pi)
        rmses.append(rmse)
        assert row["p_value"] is not None and row["p_value"] > ADEQUACY_LEVEL, (
            f"n={n}: basis_check must pass on the default basis, got {row}"
        )
        assert row["basis_dim"] > 11, f"n={n}: the basis never grew past the old cap: {row}"
        assert not any("internal knots" in m for m in messages), (
            f"n={n}: no per-fit knot-placement note may be emitted: {messages}"
        )
    assert rmses[0] > rmses[1] > rmses[2], f"truth RMSE must fall with n: {rmses}"
    # The capped default sat at ~0.134 for every n.
    assert rmses[-1] < 0.134 / 4.0, f"n=100000 still carries capped-basis bias: {rmses}"


def _median_edf(truth) -> float:
    # REML puts a smoothing parameter on its rail only with some probability,
    # so one replicate of a null or linear truth can keep a little wiggliness;
    # the contract is about the typical fit.
    edfs = sorted(_fit(5_000, seed, truth)[1]["edf"] for seed in range(5))
    return edfs[len(edfs) // 2]


def test_default_smooth_of_pure_noise_shrinks_to_zero_edf() -> None:
    # The REML null's EDF has an atom at 0 and a spread below one degree of
    # freedom; a smooth that manufactured signal would sit at or above 1.
    median = _median_edf(lambda x: np.zeros_like(x))
    assert median < 1.0, f"pure noise must shrink to ~0 EDF, median {median}"


def test_default_smooth_of_a_line_shrinks_to_one_edf() -> None:
    median = _median_edf(lambda x: 2.0 * x - 1.0)
    assert abs(median - 1.0) < 0.5, f"a linear truth must shrink to ~1 EDF, median {median}"
