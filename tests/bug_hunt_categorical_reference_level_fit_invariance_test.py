"""Gauge-invariance audit (#1593): a penalized GAM with a categorical main effect
must not depend on the arbitrary treatment-contrast reference level.

``y ~ g + s(x)`` fits one shared penalized smooth ``s(x)`` plus a categorical main
effect for ``g``. ``g`` is treatment-coded: one level is dropped as the reference
(its effect folded into the global intercept) and the others enter as unpenalized
dummy contrasts. Which level is the reference is an *arbitrary* gauge choice
(determined in practice by the level sort order). The factor main effect is
**unpenalized**, so its coefficient gauge carries no smoothing-parameter
dependence; the per-group fitted means span the identical column space regardless
of which level is dropped. The predicted value for a given physical group is a
property of the fitted model and must be invariant to the reference-level choice.

This is the categorical-predictor sibling of the gauge-invariance family
(multinomial reference class #1587, simplex ALR reference #1549). Because the
contrast block is unpenalized — unlike the multinomial penalty, which lives on
the reference-anchored contrasts — this fit *should* be reference-invariant. The
test banks that as a regression guard (and would surface a real #1593-class bug
were the reference choice to leak into the smoothing-parameter selection, e.g.
via a reference-anchored identifiability constraint shared with ``s(x)``).

It fits three cyclic relabelings that each make a different physical group the
dropped reference, predicts each physical group on a shared ``x`` grid, realigns
to the physical group identity, and asserts the cross-reference drift is tiny.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

_N_GROUPS = 3
# Distinct per-group additive offsets on top of a single shared smooth of x.
_OFFSETS = np.array([-1.2, 0.4, 1.5])


def _sample(seed: int, n: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, n)
    gi = rng.integers(0, _N_GROUPS, n)
    y = 1.5 * np.sin(1.2 * x) + _OFFSETS[gi] + rng.normal(0.0, 0.20, n)
    return x, gi, y


def _fit_predict_aligned(x, gi, y, name_map, grid):
    labels = np.asarray(name_map)[gi]
    model = gamfit.fit(pd.DataFrame({"x": x, "g": labels, "y": y}), "y ~ g + s(x)")
    out = np.empty((_N_GROUPS, grid.size), dtype=float)
    for c in range(_N_GROUPS):
        q = pd.DataFrame({"x": grid, "g": np.full(grid.size, name_map[c])})
        out[c] = np.asarray(model.predict(q), dtype=float).reshape(-1)
    return out


_LABELINGS = (
    ["A", "B", "C"],  # reference (sorts first) = group0
    ["B", "C", "A"],  # reference = group2
    ["C", "A", "B"],  # reference = group1
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_categorical_main_effect_invariant_to_reference_level(seed: int) -> None:
    x, gi, y = _sample(seed, n=1500)
    grid = np.linspace(-2.5, 2.5, 9)

    preds = [_fit_predict_aligned(x, gi, y, lab, grid) for lab in _LABELINGS]

    base = preds[0]
    signal_range = float(np.max(base) - np.min(base))
    assert signal_range > 1.0, (
        f"seed {seed}: degenerate fit (range {signal_range:.4f}); invariance "
        "assertion would be vacuous"
    )

    refit = _fit_predict_aligned(x, gi, y, _LABELINGS[0], grid)
    refit_noise = float(np.max(np.abs(base - refit)))
    assert refit_noise < 1e-6, (
        f"seed {seed}: same-labeling refit is non-deterministic "
        f"(drift {refit_noise:.3e})"
    )

    drift = max(
        float(np.max(np.abs(preds[i] - preds[j])))
        for i in range(len(preds))
        for j in range(i + 1, len(preds))
    )
    assert drift < 1e-3, (
        f"seed {seed}: categorical main-effect fit depends on the arbitrary "
        f"treatment reference level (max cross-reference drift {drift:.3e} over "
        f"signal range {signal_range:.3f}, refit noise {refit_noise:.3e}). The "
        "contrast block is unpenalized, so the reference choice is a pure "
        "reparameterization and the per-group fit must be invariant (#1593 class)."
    )
