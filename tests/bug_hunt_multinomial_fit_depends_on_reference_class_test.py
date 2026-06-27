"""Bug hunt: a penalized multinomial-logit GAM fit must not depend on the
(arbitrary) choice of reference class.

``gamfit.fit(..., family='multinomial')`` fits a softmax GAM with ``K`` classes
by reference coding: one class is pinned as the baseline (``η_ref ≡ 0``) and the
other ``K−1`` classes each get a penalized smooth of their log-odds *relative to
that baseline*. The penalized inner objective is (see
``crates/gam-models/src/multinomial.rs`` module docs)

    β̂ = argmin_β { − log L(β) + ½ Σ_{a=0}^{K-2} λ_a · β_aᵀ S β_a },

with class ``K−1`` as the reference (``η_{K-1} ≡ 0``).

The predicted class probabilities of a multinomial-logit model are a property of
the fitted distribution and are mathematically **invariant to which class is the
baseline** — the reference choice is a pure reparameterization of the same
softmax. The *unpenalized* MLE is reference-free. But the smoothing penalty above
penalizes the wiggliness of the ``K−1`` contrasts ``η_a = log(p_a / p_ref)``,
which are *defined relative to the reference class*. Relabeling the classes makes
a different class the baseline, so a different set of contrasts is penalized, and
the penalized fit — and the smoothing parameters the outer REML/LAML loop selects
for it — move with the labeling.

This is the multinomial-family analogue of the simplex ALR-reference bug #1549:
a fit whose penalty lives in a non-symmetric, reference-anchored frame is not
invariant to the arbitrary frame choice. A symmetric (sum-to-zero / class-
balanced) penalty would restore invariance.

Observed: cycling which class is the reference (by relabeling the response so a
different label sorts last) drifts the predicted probabilities by ~0.8–1.7 %
absolute on n=900 — and it does *not* shrink with n (it grows to ~1.4 % at
n=3000), confirming it is a structural penalty asymmetry, not finite-sample
noise. Refitting the *same* labeling twice agrees to ~1e-12, so the drift is not
optimizer noise.

Expected: the predicted probabilities are independent of the reference class to
near machine precision.

The test below is deterministic, seeded, and aligns every fit back to the
original class identities before comparing, so a relabeling that leaves the
softmax unchanged is detected as such.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")
pd: Any = pytest.importorskip("pandas")

import gamfit


def _sample_classes(seed: int, n: int):
    """Draw a clean 3-class softmax regression sample (integer classes 0/1/2)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    eta = np.stack([0.5 + 0.8 * x, -0.3 - 0.5 * x, np.zeros(n)], axis=1)
    p = np.exp(eta)
    p /= p.sum(axis=1, keepdims=True)
    u = rng.uniform(size=n)
    cls = (u[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
    return x, cls


def _fit_predict_aligned(x, cls, name_map, grid):
    """Fit with ``cls`` relabeled by ``name_map`` and return predicted
    probabilities re-aligned to the *original* class order 0/1/2.

    ``name_map[c]`` is the string label assigned to original class ``c``. The
    model's ``predict`` returns probability columns in sorted-label order, so the
    column for original class ``c`` is at position ``rank of name_map[c]`` among
    the sorted labels. Re-aligning makes a pure relabeling a no-op.
    """
    labels = np.asarray(name_map)[cls]
    model = gamfit.fit(
        pd.DataFrame({"x": x, "y": labels}), "y ~ s(x)", family="multinomial"
    )
    pred = np.asarray(model.predict(pd.DataFrame({"x": grid})))
    sorted_labels = sorted(name_map)
    col_of_class = {c: sorted_labels.index(name_map[c]) for c in range(len(name_map))}
    return np.stack([pred[:, col_of_class[c]] for c in range(len(name_map))], axis=1)


# Three labelings that each make a *different* original class the reference (the
# class whose label sorts last is the baseline ``K−1``):
#   ['A','B','C'] -> 'C' last  -> reference = original class 2
#   ['B','C','A'] -> 'C' last  -> reference = original class 1
#   ['C','A','B'] -> 'C' last  -> reference = original class 0
_LABELINGS = (["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_multinomial_fit_invariant_to_reference_class(seed: int) -> None:
    x, cls = _sample_classes(seed, n=900)
    grid = np.linspace(-1.5, 1.5, 7)

    preds = [_fit_predict_aligned(x, cls, nm, grid) for nm in _LABELINGS]

    for nm, p in zip(_LABELINGS, preds):
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9), (
            f"labeling {nm}: predicted probabilities do not sum to 1"
        )
        assert (p >= 0.0).all() and (p <= 1.0).all(), (
            f"labeling {nm}: predicted probabilities fall outside [0, 1]"
        )

    drift = max(
        float(np.max(np.abs(preds[i] - preds[j])))
        for i in range(len(preds))
        for j in range(i + 1, len(preds))
    )
    # Determinism (same labeling refit twice) is ~1e-12; the defect is ~8e-3..1.7e-2.
    # The 1e-3 bar sits far below the defect and far above the fit-noise floor.
    assert drift < 1e-3, (
        f"seed {seed}: multinomial fit depends on the reference class "
        f"(max cross-labeling probability drift {drift:.2e} >= 1e-3)"
    )


def test_multinomial_reference_invariance_does_not_shrink_with_n() -> None:
    """The drift is structural: it must not vanish as n grows (rules out a
    finite-sample explanation for the failure)."""
    grid = np.linspace(-1.5, 1.5, 7)
    x, cls = _sample_classes(seed=0, n=3000)
    a = _fit_predict_aligned(x, cls, _LABELINGS[0], grid)
    b = _fit_predict_aligned(x, cls, _LABELINGS[2], grid)
    drift = float(np.max(np.abs(a - b)))
    assert drift < 1e-3, (
        f"n=3000: multinomial fit still depends on the reference class "
        f"(max probability drift {drift:.2e} >= 1e-3)"
    )
