"""Numerical parity between gamfit.torch wrappers and the gamfit numpy API.

Every wrapper in :mod:`gamfit.torch` is a thin shim over an existing numpy
entry point. The values it returns must match the numpy call on the same
inputs to within fp64 rounding, since the underlying Rust call is identical.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    from gamfit import _api as _np_api
    import gamfit.torch as gt
except ImportError:
    pytest.skip("torch dependency unavailable", allow_module_level=True)


def _require_ffi(name: str) -> None:
    from gamfit._binding import rust_module

    if not hasattr(rust_module(), name):
        pytest.skip(f"engine missing FFI export `{name}`")


def _tensor(arr, **kw):
    return torch.as_tensor(np.asarray(arr, dtype=np.float64), dtype=torch.float64, **kw)


# ----------------------------- basis primitives ----------------------------- #


def test_bspline_basis_parity():
    _require_ffi("bspline_basis")
    rng = np.random.default_rng(0)
    t = rng.uniform(0.0, 1.0, size=20)
    knots = np.linspace(0.0, 1.0, 8)
    expected = _np_api.bspline_basis(t, knots, degree=3, periodic=False)
    got = gt.bspline_basis(_tensor(t), _tensor(knots), degree=3, periodic=False)
    np.testing.assert_allclose(got.detach().numpy(), expected, rtol=0, atol=0)


def test_bspline_basis_derivative_parity():
    _require_ffi("bspline_basis_derivative")
    rng = np.random.default_rng(1)
    t = rng.uniform(0.0, 1.0, size=20)
    knots = np.linspace(0.0, 1.0, 8)
    expected = _np_api.bspline_basis_derivative(t, knots, degree=3, order=1, periodic=False)
    got = gt.bspline_basis_derivative(_tensor(t), _tensor(knots), degree=3, order=1, periodic=False)
    np.testing.assert_allclose(got.detach().numpy(), expected, rtol=0, atol=0)


def test_duchon_basis_1d_parity():
    _require_ffi("duchon_basis_1d")
    rng = np.random.default_rng(2)
    t = rng.uniform(0.0, 1.0, size=20)
    centers = np.linspace(0.0, 1.0, 6)
    expected = _np_api.duchon_basis_1d(t, centers, m=2, periodic=False)
    got = gt.duchon_basis_1d(_tensor(t), _tensor(centers), m=2, periodic=False)
    np.testing.assert_allclose(got.detach().numpy(), expected, rtol=0, atol=0)


def test_duchon_basis_1d_derivative_parity():
    _require_ffi("duchon_basis_1d_derivative")
    rng = np.random.default_rng(3)
    t = rng.uniform(0.0, 1.0, size=20)
    centers = np.linspace(0.0, 1.0, 6)
    expected = _np_api.duchon_basis_1d_derivative(t, centers, m=2, order=1, periodic=False)
    got = gt.duchon_basis_1d_derivative(
        _tensor(t), _tensor(centers), m=2, order=1, periodic=False
    )
    np.testing.assert_allclose(got.detach().numpy(), expected, rtol=0, atol=0)


def test_smoothness_penalty_parity():
    _require_ffi("smoothness_penalty")
    knots = np.linspace(0.0, 1.0, 8)
    s_expected, n_expected = _np_api.smoothness_penalty(knots, degree=3, order=2)
    s_got, n_got = gt.smoothness_penalty(_tensor(knots), degree=3, order=2)
    np.testing.assert_allclose(s_got.detach().numpy(), s_expected, rtol=0, atol=0)
    np.testing.assert_allclose(n_got.detach().numpy(), n_expected, rtol=0, atol=0)


# ------------------------------ ridge solver ------------------------------- #


def test_gaussian_weighted_ridge_parity():
    _require_ffi("gaussian_weighted_ridge_array")
    rng = np.random.default_rng(4)
    n, m, d = 30, 5, 2
    X = rng.standard_normal((n, m))
    Y = rng.standard_normal((n, d))
    penalty = np.eye(m)
    weights = rng.uniform(0.5, 1.5, size=n)
    coef_e, fitted_e = _np_api.gaussian_weighted_ridge(X, Y, penalty, weights, ridge_lambda=0.7)
    coef_g, fitted_g = gt.gaussian_weighted_ridge(
        _tensor(X), _tensor(Y), _tensor(penalty), _tensor(weights), ridge_lambda=0.7
    )
    np.testing.assert_allclose(coef_g.detach().numpy(), coef_e, rtol=0, atol=0)
    np.testing.assert_allclose(fitted_g.detach().numpy(), fitted_e, rtol=0, atol=0)


# ------------------------------ REML primitives ----------------------------- #


def _reml_setup(rng, n=25, m=4, d=1):
    X = rng.standard_normal((n, m))
    beta = rng.standard_normal((m, d))
    Y = X @ beta + 0.05 * rng.standard_normal((n, d))
    penalty = np.eye(m)
    return X, Y, penalty


def test_gaussian_reml_fit_parity():
    _require_ffi("gaussian_reml_fit")
    rng = np.random.default_rng(5)
    X, Y, penalty = _reml_setup(rng)
    e = _np_api.gaussian_reml_fit(X, Y, penalty)
    g = gt.gaussian_reml_fit(_tensor(X), _tensor(Y), _tensor(penalty))
    np.testing.assert_allclose(g.coefficients.detach().numpy(), e["coefficients"], rtol=0, atol=0)
    np.testing.assert_allclose(g.fitted.detach().numpy(), e["fitted"], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(g.lam.detach()), e["lambda"], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(g.reml_score.detach()), e["reml_score"], rtol=0, atol=0)


def test_gaussian_reml_fit_batched_parity():
    _require_ffi("gaussian_reml_fit_batched")
    rng = np.random.default_rng(6)
    counts = [12, 18, 15]
    offsets = np.cumsum([0] + counts).astype(np.uintp)
    m, d = 4, 1
    n_total = int(offsets[-1])
    X = rng.standard_normal((n_total, m))
    Y = rng.standard_normal((n_total, d))
    penalty = np.eye(m)
    e = _np_api.gaussian_reml_fit_batched(X, Y, offsets, penalty)
    g = gt.gaussian_reml_fit_batched(
        _tensor(X), _tensor(Y), torch.as_tensor(offsets), _tensor(penalty)
    )
    np.testing.assert_allclose(g.coefficients.detach().numpy(), e["coefficients"], rtol=0, atol=0)
    np.testing.assert_allclose(g.fitted.detach().numpy(), e["fitted"], rtol=0, atol=0)
    np.testing.assert_allclose(g.lam.detach().numpy(), e["lambda"], rtol=0, atol=0)
    np.testing.assert_allclose(g.reml_score.detach().numpy(), e["reml_score"], rtol=0, atol=0)


def test_gaussian_reml_fit_batched_by_matches_manual_gate():
    _require_ffi("gaussian_reml_fit_batched")
    rng = np.random.default_rng(61)
    counts = [12, 14]
    offsets = np.cumsum([0] + counts).astype(np.uintp)
    n_total = int(offsets[-1])
    m, d = 4, 1
    X = rng.standard_normal((n_total, m))
    Y = rng.standard_normal((n_total, d))
    penalty = np.eye(m)
    by = rng.uniform(0.6, 1.4, size=n_total)
    gated = X.copy()
    gated[:, 1:] *= by[:, None]
    e = _np_api.gaussian_reml_fit_batched(gated, Y, offsets, penalty)
    g = gt.gaussian_reml_fit_batched(
        _tensor(X),
        _tensor(Y),
        torch.as_tensor(offsets),
        _tensor(penalty),
        by=_tensor(by),
        by_start_col=1,
    )
    np.testing.assert_allclose(g.coefficients.detach().numpy(), e["coefficients"], rtol=0, atol=0)
    np.testing.assert_allclose(g.fitted.detach().numpy(), e["fitted"], rtol=0, atol=0)


def test_gaussian_reml_fit_positions_parity():
    _require_ffi("gaussian_reml_fit_positions")
    rng = np.random.default_rng(7)
    n = 25
    t = np.sort(rng.uniform(0.0, 1.0, size=n))
    Y = np.sin(2 * np.pi * t).reshape(-1, 1) + 0.05 * rng.standard_normal((n, 1))
    knots = np.linspace(0.0, 1.0, 8)
    M = knots.size - 3 - 1  # degree=3 default in _position_basis_order
    penalty = np.eye(M)
    e = _np_api.gaussian_reml_fit_positions(t, Y, "bspline", knots, penalty)
    g = gt.gaussian_reml_fit_positions(
        _tensor(t), _tensor(Y), "bspline", _tensor(knots), _tensor(penalty)
    )
    np.testing.assert_allclose(g.coefficients.detach().numpy(), e["coefficients"], rtol=0, atol=0)
    np.testing.assert_allclose(g.fitted.detach().numpy(), e["fitted"], rtol=0, atol=0)


def test_gaussian_reml_fit_positions_batched_parity():
    _require_ffi("gaussian_reml_fit_positions_batched")
    rng = np.random.default_rng(8)
    counts = [15, 18]
    offsets = np.cumsum([0] + counts).astype(np.uintp)
    n_total = int(offsets[-1])
    t = np.concatenate(
        [np.sort(rng.uniform(0.0, 1.0, size=c)) for c in counts]
    )
    Y = (np.sin(2 * np.pi * t)).reshape(-1, 1) + 0.05 * rng.standard_normal((n_total, 1))
    knots = np.linspace(0.0, 1.0, 8)
    M = knots.size - 3 - 1
    penalty = np.eye(M)
    e = _np_api.gaussian_reml_fit_positions_batched(t, Y, offsets, "bspline", knots, penalty)
    g = gt.gaussian_reml_fit_positions_batched(
        _tensor(t),
        _tensor(Y),
        torch.as_tensor(offsets),
        "bspline",
        _tensor(knots),
        _tensor(penalty),
    )
    np.testing.assert_allclose(g.coefficients.detach().numpy(), e["coefficients"], rtol=0, atol=0)
    np.testing.assert_allclose(g.fitted.detach().numpy(), e["fitted"], rtol=0, atol=0)


# ------------------------ response-geometry transforms ----------------------- #


def test_geometry_parity_smoke():
    from gamfit import _response_geometry as rg

    rng = np.random.default_rng(9)
    x = rng.uniform(0.1, 1.0, size=(8, 4))
    np.testing.assert_allclose(
        gt.closure(_tensor(x)).detach().numpy(), rg.closure(x), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        gt.clr(_tensor(x)).detach().numpy(), rg.clr(x), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        gt.alr(_tensor(x)).detach().numpy(), rg.alr(x), rtol=1e-12, atol=1e-12
    )
    z = rg.alr(x)
    np.testing.assert_allclose(
        gt.inverse_alr(_tensor(z)).detach().numpy(),
        rg.inverse_alr(z),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        gt.simplex_frechet_mean(_tensor(x)).detach().numpy(),
        rg.simplex_frechet_mean(x),
        rtol=1e-12,
        atol=1e-12,
    )
    # Sphere: normalize rows.
    y = rng.standard_normal((6, 3))
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    np.testing.assert_allclose(
        gt.sphere_frechet_mean(_tensor(y)).detach().numpy(),
        rg.sphere_frechet_mean(y),
        rtol=1e-12,
        atol=1e-12,
    )


def test_geometry_torch_inputs_keep_autograd():
    x = (torch.rand((6, 4), dtype=torch.float64) + 0.25).requires_grad_()
    base = torch.full((4,), 0.25, dtype=torch.float64)
    simplex = gt.simplex_exp_map(gt.simplex_log_map(x, base), base)
    loss = simplex.square().sum() + gt.clr(x).square().sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    tangent = (torch.randn((5, 3), dtype=torch.float64) * 0.05).requires_grad_()
    sphere = gt.sphere_exp_map(tangent, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    sphere.sum().backward()
    assert tangent.grad is not None
    assert torch.isfinite(tangent.grad).all()
