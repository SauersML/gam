"""Gauge-invariance audit (#1593): a by-factor smooth fit must not depend on the
arbitrary labeling / sort order of the factor levels.

``y ~ g + s(x, by=g)`` fits an **independent** centred smooth for every level of
the factor ``g`` (plus a treatment-coded factor main effect). REML selects one
smoothing parameter ``λ_a`` per level ``a``. Relabeling the levels — e.g. so that
a different level sorts first and becomes the treatment-coded reference — is a
pure permutation of the levels: it permutes which data rows carry which label,
and so it must permute the per-level smooths (and their ``λ_a``) the same way,
leaving the fitted curve **of each original group** unchanged. A sum over levels
``Σ_a λ_a β_aᵀ S β_a`` is symmetric under permutation of the levels, so unlike a
*reference-anchored* penalty (multinomial #1587, ALR #1549) the by-factor penalty
*should* be invariant. The predicted curve for a given physical group is a
property of the fitted model, independent of what string we happened to name that
level or which level the engine treats as the contrast baseline.

This is the by-factor sibling of the gauge-invariance family: multinomial
reference class (#1587), simplex ALR reference (#1549), tensor margin order
(``te`` order bug). If the engine's REML warm-start / λ-selection is sensitive to
the level ordering (as it is for ``te`` margin order), or if the identifiability
absorption anchors to the first level, the per-group fit will drift under
relabeling — a real #1593-class bug. If it is genuinely permutation-invariant,
this test banks that as a regression guard.

The test fits three cyclic relabelings that each make a *different* physical group
the treatment-coded reference, realigns every prediction back to the physical
group identity, and asserts the cross-labeling drift is tiny (≪ the per-group
signal range) while a refit under the *same* labeling agrees to optimizer
precision.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

# Three physical groups with *distinct* true smooth shapes and amplitudes, so the
# per-level smoothing parameters genuinely differ and a relabeling is a
# non-trivial permutation of the converged λ-vector (not a vacuous symmetry).
_N_GROUPS = 3


def _sample(seed: int, n: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, n)
    gi = rng.integers(0, _N_GROUPS, n)
    shape = np.where(
        gi == 0,
        1.8 * np.sin(1.5 * x),
        np.where(gi == 1, 0.9 * x, 1.3 * np.cos(2.0 * x)),
    )
    y = shape + rng.normal(0.0, 0.15, n)
    return x, gi, y


def _fit_predict_aligned(x, gi, y, name_map, grid):
    """Fit ``y ~ g + s(x, by=g)`` with the physical levels relabeled by
    ``name_map`` (``name_map[c]`` is the string assigned to physical group ``c``)
    and return predictions for each physical group on ``grid``, realigned to the
    physical group identity.

    Because the relabeling is a pure permutation of the levels, predicting
    physical group ``c`` means querying the model at label ``name_map[c]``. The
    returned array has shape ``(_N_GROUPS, len(grid))`` indexed by physical group,
    so a relabeling that leaves the model unchanged is a no-op here.
    """
    labels = np.asarray(name_map)[gi]
    model = gamfit.fit(
        pd.DataFrame({"x": x, "g": labels, "y": y}), "y ~ g + s(x, by=g)"
    )
    out = np.empty((_N_GROUPS, grid.size), dtype=float)
    for c in range(_N_GROUPS):
        q = pd.DataFrame({"x": grid, "g": np.full(grid.size, name_map[c])})
        out[c] = np.asarray(model.predict(q), dtype=float).reshape(-1)
    return out


# Each labeling makes a DIFFERENT physical group the treatment-coded reference
# (the level whose label sorts first):
#   L0 -> "A"=group0 is reference; L1 -> "A"=group2; L2 -> "A"=group1.
_LABELINGS = (
    ["A", "B", "C"],
    ["B", "C", "A"],
    ["C", "A", "B"],
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_by_factor_smooth_invariant_to_level_labeling(seed: int) -> None:
    x, gi, y = _sample(seed, n=1500)
    grid = np.linspace(-2.5, 2.5, 9)

    preds = [_fit_predict_aligned(x, gi, y, lab, grid) for lab in _LABELINGS]

    # The fit must be non-degenerate, else the invariant is vacuous: each physical
    # group's predicted curve has real signal range.
    base = preds[0]
    group_range = float(np.max(base) - np.min(base))
    assert group_range > 1.0, (
        f"seed {seed}: degenerate by-factor fit (range {group_range:.4f}); "
        "the invariance assertion would be vacuous"
    )

    # Refitting the SAME labeling agrees to optimizer precision (this is not
    # finite-sample / RNG noise — same data, same labeling).
    refit = _fit_predict_aligned(x, gi, y, _LABELINGS[0], grid)
    refit_noise = float(np.max(np.abs(base - refit)))
    assert refit_noise < 1e-6, (
        f"seed {seed}: refit under the same labeling is non-deterministic "
        f"(drift {refit_noise:.3e}); cannot attribute cross-labeling drift"
    )

    # Cross-labeling drift: predictions for each physical group, realigned, must
    # agree across the three references. A reference-anchored / order-sensitive
    # penalty would drift here (cf #1587 multinomial ~1e-2, te margin order ~2-6%).
    drift = max(
        float(np.max(np.abs(preds[i] - preds[j])))
        for i in range(len(preds))
        for j in range(i + 1, len(preds))
    )
    assert drift < 1e-3, (
        f"seed {seed}: by-factor smooth fit depends on the factor level labeling "
        f"(max cross-reference drift {drift:.3e} over signal range {group_range:.3f}, "
        f"refit noise {refit_noise:.3e}). Relabeling the levels is a pure "
        "permutation and must leave each physical group's fitted curve unchanged; "
        "a drift indicates an order-anchored penalty / λ-selection (#1593 class)."
    )
