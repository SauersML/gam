"""Safety tests for the gamfit.torch autograd Functions.

These tests confirm PyTorch's version-counter mechanism catches in-place
modifications between forward and backward, so silently wrong gradients
cannot slip through.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import gamfit.torch as gt  # noqa: E402


def _require_ffi(name: str) -> None:
    from gamfit._binding import rust_module

    if not hasattr(rust_module(), name):
        pytest.skip(f"engine missing FFI export `{name}`")


def test_inplace_mutation_caught_for_reml() -> None:
    _require_ffi("gaussian_reml_fit")
    x = torch.randn(20, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(20, 1, dtype=torch.float64)
    penalty = torch.eye(3, dtype=torch.float64)
    out = gt.gaussian_reml_fit(x, y, penalty)
    loss = out.fitted.sum()
    with torch.no_grad():
        x.add_(1.0)
    with pytest.raises(RuntimeError, match=r"modified by an inplace operation"):
        loss.backward()


def test_inplace_mutation_caught_for_bspline_basis() -> None:
    _require_ffi("bspline_basis")
    t = torch.linspace(0.1, 0.9, 12, dtype=torch.float64, requires_grad=True)
    knots = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    B = gt.bspline_basis(t, knots, degree=3)
    loss = B.sum()
    with torch.no_grad():
        t.add_(0.01)
    with pytest.raises(RuntimeError, match=r"modified by an inplace operation"):
        loss.backward()
