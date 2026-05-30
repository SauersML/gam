"""Shared-scale block-orthogonal additive REML regressions."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")


def _require_ffi(name: str) -> None:
    from gamfit._binding import rust_module

    if not hasattr(rust_module(), name):
        pytest.skip(f"engine missing FFI export `{name}`")


def _orthogonal_designs(
    n: int = 48,
    p1: int = 3,
    p2: int = 4,
    d: int = 1,
    seed: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    rng = np.random.default_rng(seed)
    split = n // 2
    x1 = np.zeros((n, p1), dtype=np.float64)
    x2 = np.zeros((n, p2), dtype=np.float64)
    x1[:split] = rng.standard_normal((split, p1))
    x2[split:] = rng.standard_normal((n - split, p2))
    b1 = rng.standard_normal((p1, d))
    b2 = rng.standard_normal((p2, d))
    y = x1 @ b1 + x2 @ b2 + 0.03 * rng.standard_normal((n, d))
    designs = [
        torch.tensor(x1, dtype=torch.float64),
        torch.tensor(x2, dtype=torch.float64),
    ]
    penalties = [
        torch.eye(p1, dtype=torch.float64),
        torch.eye(p2, dtype=torch.float64),
    ]
    return designs, penalties, torch.tensor(y, dtype=torch.float64)


def test_shared_scale_orthogonal_matches_dense_joint_when_cross_gram_zero() -> None:
    _require_ffi("gaussian_reml_fit_blocks_forward")
    from gamfit.torch._reml import _gaussian_reml_fit_blocks_orthogonal

    designs, penalties, y = _orthogonal_designs(seed=1)
    dense = gt.gaussian_reml_fit_blocks(designs, penalties, y)
    orthogonal = _gaussian_reml_fit_blocks_orthogonal(designs, penalties, y)

    torch.testing.assert_close(orthogonal.fitted, dense.fitted, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(orthogonal.lambdas, dense.lambdas, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(orthogonal.edf, dense.edf, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(
        orthogonal.reml_score, dense.reml_score, rtol=2e-5, atol=2e-6
    )


def test_blocks_multi_output_keeps_per_block_lambdas() -> None:
    designs, penalties, y = _orthogonal_designs(d=3, seed=2)
    out = gt.gaussian_reml_fit_blocks(designs, penalties, y)

    assert out.fitted.shape == y.shape
    assert len(out.coefficients) == 2
    assert out.coefficients[0].shape == (designs[0].shape[1], y.shape[1])
    assert out.coefficients[1].shape == (designs[1].shape[1], y.shape[1])
    assert out.lambdas.shape == (2,)
    assert out.log_lambdas.shape == (2,)
    assert out.edf.shape == (2,)


def test_shared_scale_differs_from_private_scale_independent_loop() -> None:
    from gamfit.torch._reml import _gaussian_reml_fit_blocks_orthogonal

    designs, penalties, y = _orthogonal_designs(seed=3)
    shared = _gaussian_reml_fit_blocks_orthogonal(designs, penalties, y)
    private = [
        gt.gaussian_reml_fit(design, y, penalty)
        for design, penalty in zip(designs, penalties)
    ]
    private_score = sum(out.reml_score.reshape(()) for out in private)

    assert not torch.allclose(shared.reml_score, private_score, rtol=1e-4, atol=1e-6)
    assert shared.lambdas.shape == torch.stack(
        [out.lam.reshape(()) for out in private]
    ).shape
