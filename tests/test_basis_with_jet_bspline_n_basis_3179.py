"""Issue #3179: ``basis_with_jet("bspline", n_basis=...)`` sizes both branches.

The ``n_basis`` shorthand used to hand-build one clamped open knot vector for
both branches. ``periodic=True`` then always refused it, because the cyclic
basis takes its knots as the uniform lattice ``linspace(0, 1, n_basis + 1)``.
``n_basis`` now means the number of design columns in both branches, and it
builds exactly the basis the matching explicit knot vector builds.
"""

from __future__ import annotations

import importlib
import typing

import numpy as np

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

from gamfit._binding import rust_module


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_periodic_n_basis_is_the_uniform_lattice(degree: int) -> None:
    n_basis = degree + 5
    t = np.linspace(0.0, 1.0, 29).reshape(-1, 1)
    phi, jet, _ = rust_module().basis_with_jet(
        "bspline", t, {"n_basis": n_basis, "degree": degree, "periodic": True}
    )
    ref_phi, ref_jet, _ = rust_module().basis_with_jet(
        "bspline",
        t,
        {"knots": np.linspace(0.0, 1.0, n_basis + 1), "degree": degree, "periodic": True},
    )
    assert np.asarray(phi).shape == (t.shape[0], n_basis)
    np.testing.assert_array_equal(np.asarray(phi), np.asarray(ref_phi))
    np.testing.assert_array_equal(np.asarray(jet), np.asarray(ref_jet))
    # Periodic: the seam endpoints carry the same basis row.
    np.testing.assert_allclose(np.asarray(phi)[0], np.asarray(phi)[-1], atol=1e-12)


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_open_n_basis_is_the_clamped_uniform_vector(degree: int) -> None:
    n_basis = degree + 5
    interior = n_basis - (degree + 1)
    knots = np.concatenate(
        [
            np.zeros(degree + 1),
            np.arange(1, interior + 1) / (interior + 1),
            np.ones(degree + 1),
        ]
    )
    t = np.linspace(0.0, 1.0, 29).reshape(-1, 1)
    phi, jet, _ = rust_module().basis_with_jet(
        "bspline", t, {"n_basis": n_basis, "degree": degree}
    )
    ref_phi, ref_jet, _ = rust_module().basis_with_jet(
        "bspline", t, {"knots": knots, "degree": degree}
    )
    assert np.asarray(phi).shape == (t.shape[0], n_basis)
    np.testing.assert_allclose(np.asarray(phi), np.asarray(ref_phi), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(jet), np.asarray(ref_jet), rtol=0.0, atol=1e-12)
