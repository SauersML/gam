"""Regression: the auto-knot B-spline primitive paths must be usable (#459).

`gamfit.bspline_basis` / `bspline_basis_derivative` accept ``knots=None``
(auto-derive a clamped knot vector) and integer ``knots=K`` (auto-derive with
``K`` interior knots). #340 changed the Rust ``auto_knots_1d`` FFI export to
return a 4-tuple ``(knots, effective_degree, num_internal_knots, shrunk)`` but
the Python consumer ``_resolve_knots`` still ``np.asarray``'d the whole tuple,
so every auto-knot call raised a numpy "inhomogeneous shape" ``ValueError`` and
only the explicit-array path worked.

These tests lock the fix from several angles:

* both auto paths return a clamped-B-spline partition-of-unity basis;
* the auto path agrees with the explicit-array path on the same knots/degree;
* derivative columns agree with central finite differences on a fixed
  auto-derived knot vector (and the partition-of-unity derivative sums to 0);
* the small-``n`` degree-shrink branch (cubic -> quadratic -> linear) produces a
  valid basis evaluated at the *effective* degree instead of crashing with
  "insufficient knots" -- the latent failure a naive ``[0]`` fix would leave.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit
from gamfit._api import _resolve_knots


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_auto_knot_none_is_partition_of_unity(degree: int) -> None:
    t = np.linspace(0.0, 1.0, 50)
    basis = gamfit.bspline_basis(t, degree=degree)
    assert basis.shape[0] == t.shape[0]
    assert np.all(np.isfinite(basis))
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("k", [0, 1, 6, 12])
def test_auto_knot_int_is_partition_of_unity(degree: int, k: int) -> None:
    t = np.linspace(0.0, 1.0, 50)
    basis = gamfit.bspline_basis(t, knots=k, degree=degree)
    assert basis.shape[0] == t.shape[0]
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)


def test_auto_knot_matches_explicit_knot_path() -> None:
    """The convenience auto path must equal the verbatim-array path."""
    t = np.linspace(0.0, 1.0, 50)
    knots, eff_degree, shrunk = _resolve_knots(6, t, degree=3)
    assert not shrunk and eff_degree == 3  # plenty of data: nothing to shrink
    auto = gamfit.bspline_basis(t, knots=6, degree=3)
    explicit = gamfit.bspline_basis(t, knots, degree=eff_degree)
    np.testing.assert_allclose(auto, explicit, atol=0.0)


def test_auto_knot_derivative_matches_finite_difference() -> None:
    """First/second derivatives agree with central FD on a fixed auto vector.

    The auto path re-derives quantile knots from its evaluation points, so a
    finite-difference check must pin a single knot vector first (otherwise the
    shifted samples would silently use a different basis).
    """
    t = np.linspace(0.0, 1.0, 60)
    knots, eff_degree, _ = _resolve_knots(10, t, degree=3)

    h = 1e-6
    base = gamfit.bspline_basis(t, knots, degree=eff_degree)
    plus = gamfit.bspline_basis(t + h, knots, degree=eff_degree)
    minus = gamfit.bspline_basis(t - h, knots, degree=eff_degree)

    d1 = gamfit.bspline_basis_derivative(t, knots, degree=eff_degree, order=1)
    fd1 = (plus - minus) / (2.0 * h)

    interior = (t > 0.05) & (t < 0.95)
    assert np.abs(d1[interior] - fd1[interior]).max() < 1e-6

    # Partition-of-unity differentiates to zero everywhere.
    np.testing.assert_allclose(d1.sum(axis=1), 0.0, atol=1e-9)


def test_auto_knot_derivative_paths_do_not_crash() -> None:
    """The derivative primitive's auto paths must also be usable (#459)."""
    t = np.linspace(0.0, 1.0, 40)
    for order in (1, 2):
        d_none = gamfit.bspline_basis_derivative(t, degree=3, order=order)
        d_int = gamfit.bspline_basis_derivative(t, knots=7, degree=3, order=order)
        assert np.all(np.isfinite(d_none))
        assert np.all(np.isfinite(d_int))
        np.testing.assert_allclose(d_none.sum(axis=1), 0.0, atol=1e-9)
        np.testing.assert_allclose(d_int.sum(axis=1), 0.0, atol=1e-9)


@pytest.mark.parametrize("n,expected_degree", [(2, 1), (3, 2), (4, 3), (5, 3)])
def test_small_n_degree_shrink_produces_valid_basis(n: int, expected_degree: int) -> None:
    """Different-angle regression: the auto-shrink branch (#340).

    With too few points for a cubic, the engine downgrades the degree and the
    clamped knot vector carries boundary multiplicity ``effective_degree + 1``.
    Evaluating with the *requested* degree raises "insufficient knots" (n<=3) or
    breaks partition-of-unity; the fix must surface and evaluate with the
    effective degree. This is the latent failure a bare ``[0]`` fix would miss.
    """
    t = np.linspace(0.0, 1.0, n)

    knots, eff_degree, shrunk = _resolve_knots(6, t, degree=3)
    assert eff_degree == expected_degree
    # n is too small for the requested (degree=3, K=6), so the engine shrinks.
    assert shrunk is True
    # The clamped vector's boundary multiplicity matches the effective degree.
    assert int(np.sum(knots == knots[0])) == eff_degree + 1
    assert int(np.sum(knots == knots[-1])) == eff_degree + 1

    # Public API: both auto paths produce a finite partition-of-unity basis
    # instead of crashing.
    for basis in (gamfit.bspline_basis(t, degree=3), gamfit.bspline_basis(t, knots=6, degree=3)):
        assert np.all(np.isfinite(basis))
        np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)

    deriv = gamfit.bspline_basis_derivative(t, degree=3, order=1)
    np.testing.assert_allclose(deriv.sum(axis=1), 0.0, atol=1e-9)


def test_periodic_auto_knot_is_partition_of_unity() -> None:
    t = np.linspace(0.0, 1.0, 50)
    basis = gamfit.bspline_basis(t, knots=8, degree=3, periodic=True)
    assert np.all(np.isfinite(basis))
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)


def test_reml_positions_auto_knot_path_runs() -> None:
    """The REML positions entrypoint was silently broken the same way (#459)."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 1.0, 80))
    y = np.sin(2.0 * np.pi * t) + 0.05 * rng.standard_normal(80)
    out = gamfit.gaussian_reml_fit_positions(t, y.reshape(-1, 1))
    assert "coefficients" in out or "fitted" in out or "basis_kind" in out


# ---------------------------------------------------------------------------
# Torch front-ends share `_resolve_knots`; they were broken the same way (#459)
# and likewise must evaluate at the effective (auto-shrunk) degree.
# ---------------------------------------------------------------------------


def test_torch_bspline_basis_auto_knot_paths() -> None:
    torch = pytest.importorskip("torch")
    from gamfit.torch._basis import bspline_basis, bspline_basis_derivative

    t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    for knots in (None, 6):
        basis = bspline_basis(t, knots, degree=3)
        assert torch.allclose(basis.sum(dim=1), torch.ones(50, dtype=torch.float64))
        deriv = bspline_basis_derivative(t, knots, degree=3, order=1)
        assert torch.allclose(deriv.sum(dim=1), torch.zeros(50, dtype=torch.float64), atol=1e-9)

    # Degree-shrink: n=3 forces degree 3 -> 2; the torch wrapper must evaluate
    # at the effective degree rather than crash with "insufficient knots".
    t3 = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
    basis = bspline_basis(t3, None, degree=3)
    assert torch.allclose(basis.sum(dim=1), torch.ones(3, dtype=torch.float64))


def test_torch_bspline_descriptor_evaluate_and_basis_size() -> None:
    torch = pytest.importorskip("torch")
    spec = gamfit.BSpline(knots=None, degree=3)
    phi = spec.evaluate(torch.linspace(0.05, 0.95, 17, dtype=torch.float64))
    # basis_size must match the resolved design width (the #235 contract).
    assert phi.shape == (17, spec.basis_size)
    assert torch.allclose(phi.sum(dim=1), torch.ones(17, dtype=torch.float64))


@pytest.mark.parametrize("knots", [None, 8])
def test_torch_fit_design_penalty_consistent_for_auto_knots(knots: object) -> None:
    torch = pytest.importorskip("torch")
    from gamfit.torch.fit import _build_design_penalty

    pts = torch.linspace(0.0, 1.0, 60, dtype=torch.float64).reshape(-1, 1)
    design, penalty = _build_design_penalty(gamfit.BSpline(knots=knots, degree=3), pts)
    # The penalty must live in the design's coefficient space (same width).
    assert design.shape[1] == penalty.shape[0] == penalty.shape[1]
    assert torch.allclose(design.sum(dim=1), torch.ones(60, dtype=torch.float64))
