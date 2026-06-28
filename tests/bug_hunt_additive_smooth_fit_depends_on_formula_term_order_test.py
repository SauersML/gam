"""Gauge-invariance audit (#1593): an additive GAM fit must not depend on the
order the smooth terms are typed in the formula.

``y ~ s(x1) + s(x2)`` and ``y ~ s(x2) + s(x1)`` specify the *same* additive model
— the same two centred smooth bases, the same per-term penalty family, and REML
selects one smoothing parameter per term. The two formulas span the identical
column space and define the identical penalized objective; the typed order of the
terms is a pure relabeling of the blocks. The fitted values are an invariant of
the model and cannot depend on which smooth the user wrote first.

This is the additive-term sibling of the gauge-invariance family: the ``te``
tensor margin-order bug (``te(x,z)`` vs ``te(z,x)``), multinomial reference class
(#1587), simplex ALR reference (#1549). The identifiability machinery resolves the
shared nullspace direction (both centred smooths compete for the global intercept)
by a *priority* ordering of the blocks (``gauge_priority`` in
``crates/gam-identifiability/src/canonical.rs``); if that priority is anchored to
the typed term order rather than to a symmetric quantity, swapping the terms moves
the constraint/penalty and drifts the fit — a real #1593-class bug. If the fit is
genuinely term-order invariant, this banks that as a regression guard.

The test fits both orders on the same data, predicts on the shared training frame,
and asserts the fitted values agree to a tight fraction of the signal range, with
a same-order refit deterministic to optimizer precision.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _frame(seed: int, n: int = 1400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3.0, 3.0, n)
    x2 = rng.uniform(-3.0, 3.0, n)
    # Two genuinely different additive shapes, so the per-term λ̂ differ and a
    # term swap is a non-trivial permutation of the converged smoothing params.
    y = 1.7 * np.sin(1.3 * x1) + 0.8 * x2 * np.cos(0.7 * x2) + rng.normal(0.0, 0.2, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_additive_fit_invariant_to_term_order(seed: int) -> None:
    df = _frame(seed)

    def fit_pred(formula: str) -> np.ndarray:
        model = gamfit.fit(df, formula)
        return np.asarray(model.predict(df), dtype=float).reshape(-1)

    f_ab = fit_pred("y ~ s(x1) + s(x2)")
    f_ba = fit_pred("y ~ s(x2) + s(x1)")

    signal_range = float(np.max(f_ab) - np.min(f_ab))
    assert signal_range > 1.0, (
        f"seed {seed}: degenerate fit (range {signal_range:.4f}); invariance "
        "assertion would be vacuous"
    )

    refit_noise = float(np.max(np.abs(f_ab - fit_pred("y ~ s(x1) + s(x2)"))))
    assert refit_noise < 1e-6, (
        f"seed {seed}: same-order refit is non-deterministic (drift {refit_noise:.3e})"
    )

    drift = float(np.max(np.abs(f_ab - f_ba)))
    assert drift < 1e-3 * max(signal_range, 1.0), (
        f"seed {seed}: additive fit depends on the typed term order "
        f"(max |Δfitted| {drift:.3e} over signal range {signal_range:.3f}, refit "
        f"noise {refit_noise:.3e}). 'y ~ s(x1)+s(x2)' and 'y ~ s(x2)+s(x1)' span "
        "the identical space under the identical penalty family; the fit must be "
        "invariant to term order (#1593 class — cf the te margin-order bug)."
    )
