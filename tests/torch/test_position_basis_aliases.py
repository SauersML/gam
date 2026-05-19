"""Smoke tests for the user-facing position-REML API surface.

Exercises the basis-aliasing layer (`basis="duchon" | "duchon_multipenalty" |
"thinplate" | "bspline"`), the canonical default penalty per basis, the
`smoothing` / `coefficients` flags, and the `freeze()` inference helper.

These are intentionally fast smoke tests — they assert finite outputs, the
correct shape of the returned predictor, and that two routes that should
mathematically agree do (e.g. thin-plate on 1D positions ≡ Duchon m=2).
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import gamfit.torch as gt
    from gamfit.torch._reml import FrozenPositionPredictor
except ImportError:
    pytest.skip("torch dependency unavailable", allow_module_level=True)


def _require_ffi(name: str) -> None:
    from gamfit._binding import rust_module

    if not hasattr(rust_module(), name):
        pytest.skip(f"engine missing FFI export `{name}`")


def _sample(n: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = np.sin(2.0 * np.pi * t) + 0.1 * rng.standard_normal(n)
    return (
        torch.tensor(t, dtype=torch.float64),
        torch.tensor(y.reshape(-1, 1), dtype=torch.float64),
    )


def test_default_basis_is_bspline() -> None:
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y)
    assert fit.basis_kind == "bspline"
    assert torch.isfinite(fit.reml_score)


def test_duchon_default_function_norm() -> None:
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y, basis="duchon")
    assert fit.basis_kind == "duchon"
    # Function-norm penalty is a (K, K) SPSD matrix.
    pen = fit.penalty.detach().numpy()
    assert pen.shape[0] == pen.shape[1]
    assert np.allclose(pen, pen.T, atol=1e-10)


def test_thinplate_alias_to_duchon_m2() -> None:
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    fit_tps = gt.gaussian_reml_fit_positions(t, y, basis="thinplate")
    fit_duchon = gt.gaussian_reml_fit_positions(t, y, basis="duchon", basis_order=2)
    # 1D thin-plate spline = Duchon m=2 cubic smoothing spline. The display
    # name differs but the engine sees the same Duchon basis at the same
    # quantile centers, so the resolved penalty must agree.
    pen_a = fit_tps.penalty.detach().numpy()
    pen_b = fit_duchon.penalty.detach().numpy()
    assert pen_a.shape == pen_b.shape
    assert np.allclose(pen_a, pen_b, atol=1e-10)
    # User-facing display name is preserved.
    assert fit_tps.basis_kind == "thinplate"
    assert fit_duchon.basis_kind == "duchon"


def test_bspline_default_difference_penalty() -> None:
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y, basis="bspline")
    pen = fit.penalty.detach().numpy()
    # The 2nd-difference penalty has a 2D nullspace (constants + linears).
    eigs = np.linalg.eigvalsh(0.5 * (pen + pen.T))
    near_zero = (np.abs(eigs) < 1e-8).sum()
    assert near_zero >= 2


def test_duchon_multipenalty_alias_to_triple_operator() -> None:
    _require_ffi("duchon_operator_penalties")
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y, basis="duchon_multipenalty")
    assert fit.basis_kind == "duchon_multipenalty"
    # Triple-operator default = sum of mass + tension + stiffness.
    assert torch.isfinite(fit.reml_score)


def test_freeze_evaluate_roundtrip_duchon() -> None:
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y, basis="duchon")
    frozen = fit.freeze()
    assert isinstance(frozen, FrozenPositionPredictor)
    # Evaluating at the training positions reproduces the fitted curve.
    pred = frozen.evaluate(t)
    fitted = fit.fitted.detach()
    np.testing.assert_allclose(
        pred.detach().numpy(), fitted.numpy(), atol=1e-8
    )


def test_freeze_evaluate_roundtrip_thinplate() -> None:
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y, basis="thinplate")
    frozen = fit.freeze()
    pred = frozen.evaluate(t)
    fitted = fit.fitted.detach()
    np.testing.assert_allclose(
        pred.detach().numpy(), fitted.numpy(), atol=1e-8
    )


def test_explicit_penalty_override_accepted() -> None:
    t, y = _sample()
    # User supplies an explicit identity ridge instead of the default
    # canonical penalty — should override and still produce a finite fit.
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    custom_pen = torch.eye(10, dtype=torch.float64)
    fit = gt.gaussian_reml_fit_positions(
        t, y, basis="duchon", knots_or_centers=knots, penalty=custom_pen
    )
    np.testing.assert_allclose(
        fit.penalty.detach().numpy(), custom_pen.numpy(), atol=1e-12
    )
    assert torch.isfinite(fit.reml_score)


def test_string_penalty_override_function_norm_vs_triple_operator() -> None:
    _require_ffi("duchon_function_norm_penalty")
    _require_ffi("duchon_operator_penalties")
    t, y = _sample()
    fit_fn = gt.gaussian_reml_fit_positions(
        t, y, basis="duchon", penalty="function_norm"
    )
    fit_op = gt.gaussian_reml_fit_positions(
        t, y, basis="duchon", penalty="triple_operator"
    )
    pen_fn = fit_fn.penalty.detach().numpy()
    pen_op = fit_op.penalty.detach().numpy()
    # The two penalties are mathematically different objects on the same
    # basis — they must produce distinct matrices.
    assert not np.allclose(pen_fn, pen_op, atol=1e-6)
