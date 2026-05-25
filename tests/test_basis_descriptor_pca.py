"""Callable-basis contract for :class:`gamfit.Pca`.

PCA is a fixed linear projection so the Jacobian is the embedding matrix
and the Hessian is identically zero.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit


def test_evaluate_jacobian_hessian_shapes() -> None:
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((7, 4))
    spec = gamfit.Pca(basis=basis)
    B = 5
    x = torch.as_tensor(rng.standard_normal((B, 7)), dtype=torch.float64)
    # Pca uses 7 intrinsic coords -> 7 positional args.
    coords = tuple(x[:, j] for j in range(7))
    phi = spec.evaluate(*coords)
    assert phi.shape == (B, 4)
    jac = spec.jacobian(*coords)
    assert jac.shape == (B, 4, 7)
    # Each row's jacobian equals the basis matrix transposed (M, D) -> (M, d=D).
    for b in range(B):
        assert torch.allclose(
            jac[b], torch.as_tensor(basis.T, dtype=torch.float64), atol=1e-9
        )
    H = spec.hessian(*coords)
    assert H.shape == (B, 4, 7, 7)
    assert torch.allclose(H, torch.zeros_like(H), atol=0.0)
