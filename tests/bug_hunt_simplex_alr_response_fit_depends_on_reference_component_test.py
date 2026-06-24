"""Bug hunt #1549: an ALR simplex-response fit must not depend on the
(arbitrary) ALR reference component.

``gamfit.fit(..., response_geometry="alr", response_columns=[...])`` maps each
simplex-valued response row to additive-log-ratio (ALR) tangent coordinates
relative to a chosen reference part, runs a shared-smoothing Gaussian REML fit
on those ``D−1`` coordinates, and inverse-ALR maps the predictions back. The
Aitchison geometry of the simplex is intrinsic and reference-free — which part
is the ALR denominator is a pure coordinate relabeling — so cycling
``response_reference`` must NOT move the predicted compositions.

Root cause of the original failure: the ALR chart is not isometric to Aitchison
geometry — in ALR coordinates the inner product is ``uᵀ G v`` with the Gram
``G = I_{D−1} − (1/D)·11ᵀ``. The fit installed ``G`` only as the residual
precision and left the smoothing penalty ``λ·tr(BᵀSB)`` in the raw ALR frame.
The change-of-reference map ``M`` between two ALR references is linear but NOT
orthogonal, so it preserves the residual ``yᵀG y`` but changes
``tr(BᵀSB) → tr(MᵀBᵀSB M)``; the penalized objective (and the shared smoothing
selection riding on it) was therefore reference-dependent, drifting the
predicted compositions by 0.4–1.3 % of range.

Fix: whiten the tangent coordinates by the symmetric ``W = G^{1/2}`` and fit the
isotropic Gaussian REML in the whitened frame, so BOTH the residual and the
penalty are Aitchison-isometric. Two ALR references then differ by an
*orthogonal* rotation of the whitened coordinates, under which the whole
shared-smoothing objective is invariant — the fit is reference-free and equal to
the CLR fit.

Angles covered here:

* **Reference invariance** (the reported failure) — cycling ``response_reference``
  over ``{0,1,2}`` on legacy seeds ``{0..4}`` must leave the predicted
  compositions identical to ~machine precision (≪ the 1e-3 bar, ≫ fit noise),
  and every prediction must stay a valid composition.
* **Equality to the reference-free CLR fit** — the invariant ALR fit must agree
  with the ``response_geometry="simplex"`` (CLR) fit, which is reference-free by
  construction. This pins the invariant down to the *correct* point, not merely
  a self-consistent wrong one.
* **D = 4 generality** — the same invariance must hold for a 4-part composition
  (Gram is ``3×3``, four reference choices), guarding against any fix that only
  handles the ``D = 3`` / 2-dimensional case.
* **Whitener algebra** — a direct unit check that the symmetric whitener ``W``
  satisfies ``Wᵀ W = G`` and ``W = Wᵀ``, isolating the geometric core from the
  fit pipeline.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")
pd: Any = pytest.importorskip("pandas")

import gamfit
from gamfit._response_geometry import _aitchison_whitener


def _make_frame(seed: int, n_parts: int = 3, n: int = 400):
    """A clean compositional regression frame: smooth log-ratio signals + noise."""
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(0, 1, n))
    # n_parts−1 smooth signals against a zero reference, then closed to the simplex.
    signals = [1.5 * np.sin(2 * np.pi * x), 0.8 * x, 0.6 * np.cos(3 * x)]
    cols = [signals[i] for i in range(n_parts - 1)] + [np.zeros(n)]
    logits = np.vstack(cols).T + 0.1 * rng.standard_normal((n, n_parts))
    comp = np.exp(logits)
    comp /= comp.sum(1, keepdims=True)
    names = [chr(ord("a") + i) for i in range(n_parts)]
    df = pd.DataFrame({"x": x})
    for i, name in enumerate(names):
        df[name] = comp[:, i]
    grid = pd.DataFrame({"x": np.linspace(0, 1, 40)})
    return df, names, grid


def _alr_pred(df, names, grid, reference):
    m = gamfit.fit(
        df,
        f"{names[0]} ~ s(x)",
        response_columns=names,
        response_geometry="alr",
        response_reference=reference,
    )
    return np.asarray(m.predict(grid))


def _is_composition(pred) -> bool:
    return bool(np.allclose(pred.sum(1), 1.0, atol=1e-9) and (pred > 0).all())


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_alr_fit_invariant_to_reference_component(seed: int) -> None:
    """Cycling response_reference must not move the predicted compositions."""
    df, names, grid = _make_frame(seed, n_parts=3)
    preds = [_alr_pred(df, names, grid, r) for r in range(3)]

    for r, p in enumerate(preds):
        assert _is_composition(p), f"ALR prediction (ref={r}) is not a valid composition"

    drift = max(
        np.max(np.abs(preds[i] - preds[j]))
        for i in range(3)
        for j in range(i + 1, 3)
    )
    # The defect was 2.3e-3..7.9e-3; real reordering/fit noise is ~1e-14. The
    # 1e-3 bar sits well below the defect and far above the noise floor.
    assert drift < 1e-3, (
        f"seed {seed}: ALR fit depends on the reference component "
        f"(max cross-reference drift {drift:.2e} ≥ 1e-3)"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_alr_fit_equals_reference_free_clr_fit(seed: int) -> None:
    """The invariant ALR fit must agree with the reference-free CLR (simplex) fit."""
    df, names, grid = _make_frame(seed, n_parts=3)
    alr = _alr_pred(df, names, grid, 0)
    clr_model = gamfit.fit(
        df, f"{names[0]} ~ s(x)", response_columns=names, response_geometry="simplex"
    )
    clr = np.asarray(clr_model.predict(grid))
    drift = np.max(np.abs(alr - clr))
    assert drift < 1e-3, (
        f"seed {seed}: ALR fit disagrees with the reference-free CLR fit "
        f"(max drift {drift:.2e} ≥ 1e-3) — the ALR penalty is not Aitchison-isometric"
    )


def test_alr_fit_reference_invariant_for_four_part_composition() -> None:
    """The invariance must hold for D = 4 (3×3 Gram, four reference choices)."""
    df, names, grid = _make_frame(seed=2, n_parts=4)
    preds = [_alr_pred(df, names, grid, r) for r in range(4)]
    for r, p in enumerate(preds):
        assert _is_composition(p), f"D=4 ALR prediction (ref={r}) is not a valid composition"
    drift = max(
        np.max(np.abs(preds[i] - preds[j]))
        for i in range(4)
        for j in range(i + 1, 4)
    )
    assert drift < 1e-3, (
        f"D=4 ALR fit depends on the reference component (max drift {drift:.2e} ≥ 1e-3)"
    )


@pytest.mark.parametrize("n_parts", [2, 3, 4, 6])
def test_aitchison_whitener_is_symmetric_square_root_of_gram(n_parts: int) -> None:
    """W is symmetric and Wᵀ W = G = I_{D−1} − (1/D)·11ᵀ (the Aitchison Gram)."""
    sqrt, inv_sqrt = _aitchison_whitener(np, n_parts)
    dim = n_parts - 1
    gram = np.eye(dim) - 1.0 / float(n_parts)
    assert np.allclose(sqrt, sqrt.T, atol=1e-12), "whitener must be symmetric"
    assert np.allclose(sqrt @ sqrt, gram, atol=1e-12), "WᵀW must equal the Aitchison Gram"
    assert np.allclose(sqrt @ inv_sqrt, np.eye(dim), atol=1e-12), "W·W⁻¹ must be I"
