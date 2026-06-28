"""Gauge-invariance audit (#1593): a tensor-product GAM fit must not depend on
the typed order of the smooth's covariates.

``y ~ te(x, z)`` and ``y ~ te(z, x)`` specify the *same* tensor-product model —
the identical tensor-product space spanned by the same per-margin B-spline bases
under the identical per-margin penalty family, with one smoothing parameter per
margin selected by REML. The tensor design is the Khatri–Rao product
``B_first ⊙ B_second``, so the typed order is a pure column/penalty-block
permutation of the identical penalized objective. The fitted values are an
invariant of the model and cannot depend on which covariate the user wrote first.

This is the tensor-margin sibling of the gauge-invariance family — the additive
term-order bug (``s(x1)+s(x2)`` vs ``s(x2)+s(x1)``), the categorical reference
level, the by-factor smooth labeling, the multinomial reference class (#1587), the
simplex ALR reference (#1549). It is the *exact* failure #1593 reported: the margin
permutation reorders the penalized normal-equation / REML linear algebra, and the
sub-ULP differences route the outer λ optimizer to a different terminal point in
te's flat REML valley (one margin rails to the ρ bound while the other lands on a
materially different λ̂), drifting the shipped surface ~2–6 % of range with a
cosmetic covariate swap. The fix (``crates/gam-terms/src/term_builder.rs``)
canonicalizes the margin order by source feature-column index at construction, so
the same physical model always builds the identical problem; ``te``/``ti``/``t2``
become genuinely order-invariant.

The test fits both covariate orders on the same data, predicts on the shared
training frame, and asserts the fitted values agree to a tight fraction of the
signal range, with a same-order refit deterministic to optimizer precision.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _frame(seed: int, n: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, n)
    z = rng.uniform(-3.0, 3.0, n)
    # A genuinely anisotropic interaction surface: the two margins have very
    # different roughness, so their λ̂ differ and a covariate swap is a
    # non-trivial permutation of the converged smoothing parameters (the
    # regime where the #1593 flat-valley drift actually bit).
    y = (
        1.6 * np.sin(1.4 * x) * np.cos(0.5 * z)
        + 0.7 * z
        + rng.normal(0.0, 0.2, n)
    )
    return pd.DataFrame({"x": x, "z": z, "y": y})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tensor_fit_invariant_to_covariate_order(seed: int) -> None:
    df = _frame(seed)

    def fit_pred(formula: str) -> np.ndarray:
        model = gamfit.fit(df, formula)
        return np.asarray(model.predict(df), dtype=float).reshape(-1)

    f_xz = fit_pred("y ~ te(x, z)")
    f_zx = fit_pred("y ~ te(z, x)")

    signal_range = float(np.max(f_xz) - np.min(f_xz))
    assert signal_range > 1.0, (
        f"seed {seed}: degenerate fit (range {signal_range:.4f}); invariance "
        "assertion would be vacuous"
    )

    refit_noise = float(np.max(np.abs(f_xz - fit_pred("y ~ te(x, z)"))))
    assert refit_noise < 1e-6, (
        f"seed {seed}: same-order refit is non-deterministic (drift {refit_noise:.3e})"
    )

    drift = float(np.max(np.abs(f_xz - f_zx)))
    assert drift < 1e-3 * max(signal_range, 1.0), (
        f"seed {seed}: tensor fit depends on the typed covariate order "
        f"(max |Δfitted| {drift:.3e} over signal range {signal_range:.3f}, refit "
        f"noise {refit_noise:.3e}). 'y ~ te(x, z)' and 'y ~ te(z, x)' span the "
        "identical tensor-product space under the identical per-margin penalty "
        "family; the fit must be invariant to covariate order (#1593 — the te "
        "margin-order bug the canonicalization fix targets)."
    )
