"""torch-path Pca prices the fitted function, not its coefficient chart.

The Rust Pca builder (`pca_function_mass_penalty`) penalizes the empirical
function mass ``βᵀSβ = mean_i((Zβ)_i²)``, ``S = ZᵀZ / N``. The torch fit path
must build the same functional (SPEC: penalties on the final function, never on
the model coefficients; CLI/Python/Rust behavior parity).

A direct consequence is chart invariance: rescaling the basis columns,
``B → B·D``, maps ``Z → Z·D`` and ``S → D·S·D``. The REML criterion changes
only by the λ-free constant ``2·log|D|`` in both ``log|H|`` and ``log|S|₊``,
so λ̂ is unchanged and the fitted function is identical. An identity
coefficient ridge breaks this: it shrinks a column scaled by 0.1 far more
than one scaled by 10.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")

from gamfit.smooth import Pca  # noqa: E402
from gamfit.torch.fit import _build_design_penalty  # noqa: E402


def _data(n: int = 200, d: int = 5, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    basis = np.linalg.qr(rng.standard_normal((d, k)))[0]
    xc = x - x.mean(axis=0)
    y = xc @ basis @ np.array([1.0, -0.5, 0.25]) + 0.5 * rng.standard_normal(n)
    return (
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        basis,
    )


def test_pca_penalty_is_function_mass_gram():
    x, _y, basis = _data()
    design, penalty = _build_design_penalty(Pca(basis=basis), x)
    expected = design.transpose(0, 1) @ design / float(design.shape[0])
    assert torch.equal(penalty, expected)


def test_pca_fit_is_invariant_to_basis_column_scaling():
    x, y, basis = _data()
    scale = np.diag([10.0, 1.0, 0.1])
    reference = gt.fit(x, y, Pca(basis=basis))
    rescaled = gt.fit(x, y, Pca(basis=basis @ scale))
    ref = reference.fitted.detach()
    # The two problems are the same REML problem up to a λ-free constant, so
    # the fitted functions agree to the outer optimizer's convergence.
    tol = 1e-6 * float(ref.abs().max())
    assert float((rescaled.fitted.detach() - ref).abs().max()) <= tol


def test_pca_fit_requires_the_precomputed_basis():
    x, y, _basis = _data()
    with pytest.raises(ValueError, match="basis matrix must be provided"):
        gt.fit(x, y, Pca(K=2))
