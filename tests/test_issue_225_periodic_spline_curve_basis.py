"""RED tests for issue #225 — `periodic_spline_curve_basis` missing in pyffi.

The Python wrapper in `gamfit/_api.py:1652` calls
``rust_module().periodic_spline_curve_basis(...)``, but no such
``#[pyfunction]`` is registered in ``crates/gam-pyffi/src/lib.rs``. This
breaks the whole periodic-1D family of public APIs. These tests fail RED
today; they will go green once the pyffi binding is added.
"""

from __future__ import annotations

import importlib
import typing

import numpy as np

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


def test_periodic_spline_curve_basis_rust_pyfunction_is_registered() -> None:
    """The pyffi module must export the function the Python wrapper calls."""
    from gamfit._binding import rust_module

    assert hasattr(rust_module(), "periodic_spline_curve_basis"), (
        "crates/gam-pyffi/src/lib.rs must register a "
        "#[pyfunction] periodic_spline_curve_basis matching the wrapper at "
        "gamfit/_api.py:1652-1657"
    )


def test_standalone_cyclic_roughness_kernel_is_registered_and_identical() -> None:
    """Torch's penalty-only path must reach the same registered Rust kernel."""
    from gamfit._binding import rust_module

    rust = rust_module()
    t = np.linspace(0.0, 1.0, 17, endpoint=False, dtype=float)
    _, combined = rust.periodic_spline_curve_basis(t, 11, 3, 2)
    standalone = rust.cyclic_bspline_roughness_penalty(11, 3, 1.0, 2)
    np.testing.assert_array_equal(standalone, combined)


def test_top_level_periodic_spline_curve_basis_returns_basis_and_penalty() -> None:
    t = np.array([0.0, 0.07, 0.25, 0.5, 0.999_999, 1.0, 1.07, -0.93], dtype=float)
    basis, penalty = gamfit.periodic_spline_curve_basis(t, n_knots=12, degree=3)
    assert basis.shape == (t.size, 12)
    assert penalty.shape == (12, 12)
    # Partition of unity.
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)
    # Seam wrap: t=0 and t=1 produce the same basis row.
    np.testing.assert_allclose(basis[0], basis[5], atol=1e-12)
    # Cyclic difference penalty has the constant vector in its nullspace.
    ones = np.ones((12, 1))
    np.testing.assert_allclose(penalty @ ones, 0.0, atol=1e-10)
    # Penalty is symmetric.
    np.testing.assert_allclose(penalty, penalty.T, atol=1e-12)


def test_periodic_spline_curve_descriptor_evaluate_runs() -> None:
    """`gamfit.PeriodicSplineCurve` evaluates without raising the missing-pyfunction error."""
    spec = gamfit.PeriodicSplineCurve(n_knots=10, degree=3, output_dim=1)
    t = np.linspace(0.0, 1.0, 25, dtype=float)
    design, _ = gamfit.periodic_spline_curve_basis(t, n_knots=spec.n_knots, degree=spec.degree)
    assert design.shape == (t.size, spec.n_knots)


def test_circle_topology_smooth_builds_periodic_basis() -> None:
    """`gamfit.Circle()` must produce a working periodic smooth descriptor."""
    smooth = gamfit.Circle(n_knots=12, degree=3)
    # Construction alone should not raise; the descriptor must carry a
    # periodic 1D basis spec that the engine can lower.
    assert smooth is not None
    basis, penalty = gamfit.periodic_spline_curve_basis(
        np.linspace(0.0, 1.0, 9, dtype=float),
        n_knots=12,
        degree=3,
    )
    assert basis.shape == (9, 12)
    assert penalty.shape == (12, 12)


def test_torch_periodic_spline_curve_basis_round_trips() -> None:
    torch = pytest.importorskip("torch")
    from gamfit.torch import periodic_spline_curve_basis as torch_periodic

    t = torch.linspace(0.0, 1.0, 17, dtype=torch.float64)
    basis, penalty = torch_periodic(t, n_knots=11, degree=3, penalty_order=2)
    assert tuple(basis.shape) == (17, 11)
    assert tuple(penalty.shape) == (11, 11)
    # Partition of unity at machine precision.
    np.testing.assert_allclose(basis.sum(dim=1).cpu().numpy(), 1.0, atol=1e-12)
    # Cyclic penalty annihilates the constant vector.
    ones = torch.ones((11, 1), dtype=torch.float64)
    np.testing.assert_allclose((penalty @ ones).cpu().numpy(), 0.0, atol=1e-10)
