"""Bug hunt #2031: zero-``by`` rows must be inert in ``gaussian_reml_fit``.

``gaussian_reml_fit(x, y, penalty, by=...)`` documents that ``by`` weights the
whole REML fit: "When ``by_i = 0`` the corresponding ``fitted_i`` is exactly
zero; the REML fit is also weighted by ``by`` so zero-``by`` rows do not
influence ``B``." But ``by`` was applied only as a design-column scaling
(``apply_by_gate``): a ``by=0`` row's modulated design became the zero vector
(so it could not move ``B`` at a fixed λ), yet its response still entered the
weighted response energy ``ywy = Σ w·y²`` and it still counted toward the
residual degrees of freedom ``ν``. That leaked the gated-off rows' response
into ``σ²``, ``λ``, and — through λ — the coefficients.

The fix folds the ``by`` gate into the REML row weights (``w_eff = w·[by≠0]``),
so a ``by=0`` row is a complete no-op.

Angles covered here:

* **Exact omission equivalence** — appending ``by=0`` junk rows must not move
  the coefficients, ``lambda``, ``edf``, or ``sigma2``.
* **Response-independence** — two fits differing only in the *response* of their
  ``by=0`` rows must agree exactly (convention-free isolation of the leak).
* **Nonzero-``by`` invariance** — an all-nonzero ``by`` fit must stay identical
  to manually gating the design and passing raw weights (the pre-existing
  contract), i.e. the fix only changes behavior at exactly ``by==0``.
* **Prior-weight composition** — the ``by`` mask must compose with prior
  ``weights`` (a ``by=0`` row is dropped regardless of its prior weight).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import numpy as np

pytest = cast(Any, import_module("pytest"))
gamfit = cast(Any, import_module("gamfit"))


def _pspline_penalty(n_basis: int) -> np.ndarray:
    d = np.diff(np.eye(n_basis), 2, axis=0)
    return d.T @ d


def _scalar(out: dict[str, Any], key: str) -> float:
    return float(np.asarray(out[key]).ravel()[0])


def _base_problem(n: int = 120, knots: int = 8, seed: int = 0):
    x = np.linspace(0.0, 1.0, n)
    design = np.asarray(gamfit.bspline_basis(x, knots=knots))
    penalty = _pspline_penalty(design.shape[1])
    y = (
        np.sin(2 * np.pi * x)
        + np.random.default_rng(seed).normal(0.0, 0.2, n)
    ).reshape(-1, 1)
    return design, penalty, y


def test_zero_by_padding_matches_baseline() -> None:
    """Appending ``by=0`` junk rows must not change the fit."""
    design, penalty, y = _base_problem()
    n = design.shape[0]

    base = gamfit.gaussian_reml_fit(design, y, penalty, by=np.ones(n))

    g = 30
    rng = np.random.default_rng(1)
    design_aug = np.vstack([design, rng.normal(0.0, 5.0, (g, design.shape[1]))])
    y_aug = np.vstack([y, rng.normal(0.0, 1e3, (g, 1))])
    by_aug = np.concatenate([np.ones(n), np.zeros(g)])
    aug = gamfit.gaussian_reml_fit(design_aug, y_aug, penalty, by=by_aug)

    np.testing.assert_allclose(
        np.asarray(aug["coefficients"]).ravel(),
        np.asarray(base["coefficients"]).ravel(),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        _scalar(aug, "lambda"), _scalar(base, "lambda"), rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        _scalar(aug, "sigma2"), _scalar(base, "sigma2"), rtol=1e-9, atol=0
    )
    np.testing.assert_allclose(
        _scalar(aug, "edf"), _scalar(base, "edf"), rtol=1e-9, atol=0
    )


def test_zero_by_rows_response_is_irrelevant() -> None:
    """Two fits differing only in the *response* of ``by=0`` rows must agree."""
    design, penalty, y = _base_problem(seed=4)
    n = design.shape[0]

    rng = np.random.default_rng(9)
    g = 40
    pad_design = rng.normal(0.0, 4.0, (g, design.shape[1]))
    design_aug = np.vstack([design, pad_design])
    by_aug = np.concatenate([np.ones(n), np.zeros(g)])

    y_a = np.vstack([y, rng.normal(0.0, 1.0, (g, 1))])
    y_b = np.vstack([y, rng.normal(0.0, 1e5, (g, 1))])

    fit_a = gamfit.gaussian_reml_fit(design_aug, y_a, penalty, by=by_aug)
    fit_b = gamfit.gaussian_reml_fit(design_aug, y_b, penalty, by=by_aug)

    np.testing.assert_allclose(
        np.asarray(fit_a["coefficients"]).ravel(),
        np.asarray(fit_b["coefficients"]).ravel(),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        _scalar(fit_a, "sigma2"), _scalar(fit_b, "sigma2"), rtol=1e-9, atol=0
    )


def test_nonzero_by_matches_manual_design_gate() -> None:
    """An all-nonzero ``by`` fit is byte-identical to manually gating the design.

    The fix must only change behavior at exactly ``by==0``; for a strictly
    positive amplitude the ``by`` path stays a pure design-column scaling.
    """
    design, penalty, y = _base_problem(seed=6)
    n = design.shape[0]
    rng = np.random.default_rng(21)
    by = rng.uniform(0.6, 1.4, n)

    manual = design.copy()
    manual *= by[:, None]  # by_start_col=0: gate all columns
    reference = gamfit.gaussian_reml_fit(manual, y, penalty)
    gated = gamfit.gaussian_reml_fit(design, y, penalty, by=by)

    np.testing.assert_allclose(
        np.asarray(gated["coefficients"]),
        np.asarray(reference["coefficients"]),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        _scalar(gated, "lambda"), _scalar(reference, "lambda"), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        _scalar(gated, "sigma2"), _scalar(reference, "sigma2"), rtol=0, atol=0
    )


def test_by_mask_composes_with_prior_weights() -> None:
    """A ``by=0`` row is dropped regardless of its (positive) prior weight."""
    design, penalty, y = _base_problem(seed=8)
    n = design.shape[0]
    rng = np.random.default_rng(31)
    weights = rng.uniform(0.5, 2.0, n)

    base = gamfit.gaussian_reml_fit(
        design, y, penalty, weights=weights, by=np.ones(n)
    )

    g = 50
    design_aug = np.vstack([design, rng.normal(0.0, 3.0, (g, design.shape[1]))])
    y_aug = np.vstack([y, rng.normal(0.0, 500.0, (g, 1))])
    by_aug = np.concatenate([np.ones(n), np.zeros(g)])
    # The gated-off rows carry large *positive* prior weights: they must still
    # be dropped, because a zero-`by` row is inert regardless of prior weight.
    weights_aug = np.concatenate([weights, rng.uniform(5.0, 20.0, g)])
    aug = gamfit.gaussian_reml_fit(
        design_aug, y_aug, penalty, weights=weights_aug, by=by_aug
    )

    np.testing.assert_allclose(
        np.asarray(aug["coefficients"]).ravel(),
        np.asarray(base["coefficients"]).ravel(),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        _scalar(aug, "sigma2"), _scalar(base, "sigma2"), rtol=1e-9, atol=0
    )
