"""Python integration tests for ``gamfit.SheafConsistencyPenalty``.

Mirrors the Rust unit tests in ``src/terms/sheaf.rs`` and additionally
exercises the dict / list per-layer input formats.
"""

from __future__ import annotations

from importlib import import_module
from typing import cast

import numpy as np

pytest = cast("object", import_module("pytest"))

import gamfit


def test_single_edge_identity_value_matches_closed_form() -> None:
    """K=2 identity restrictions: value = ½‖s_0 − s_1‖²."""
    sheaf = gamfit.SheafConsistencyPenalty(
        edges=[(0, 1)],
        restriction_ops=[np.eye(3)],
        weight=1.0,
    )
    z = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 1.0, 0.0])}
    val = sheaf(z)
    assert val == pytest.approx(1.0, abs=1e-12)


def test_list_and_dict_input_formats_agree() -> None:
    sheaf = gamfit.SheafConsistencyPenalty(
        edges=[(0, 1), (1, 2)],
        restriction_ops=[np.eye(2), np.eye(2)],
        weight=0.5,
    )
    z_list = [np.array([1.0, 2.0]), np.array([3.0, -1.0]), np.array([0.0, 4.0])]
    z_dict = {i: arr for i, arr in enumerate(z_list)}
    assert sheaf(z_list) == pytest.approx(sheaf(z_dict), abs=1e-14)


def test_gradient_matches_finite_difference_random_restrictions() -> None:
    rng = np.random.default_rng(7)
    r_uv = rng.normal(size=(2, 3))
    r_vu = rng.normal(size=(2, 2))
    sheaf = gamfit.SheafConsistencyPenalty(
        edges=[(0, 1)],
        restriction_ops=[(r_uv, r_vu)],
        weight=0.3,
    )
    z = rng.normal(size=sheaf.total_dim)
    g = sheaf.gradient(z)
    assert g.shape == (sheaf.total_dim,)
    eps = 1e-6
    for i in range(z.size):
        zp = z.copy()
        zm = z.copy()
        zp[i] += eps
        zm[i] -= eps
        fd = (sheaf(zp) - sheaf(zm)) / (2.0 * eps)
        assert g[i] == pytest.approx(fd, abs=1e-4)


def test_hvp_equals_dense_laplacian_application() -> None:
    rng = np.random.default_rng(11)
    restrictions = [
        (rng.normal(size=(2, 2)), rng.normal(size=(2, 2))),
        (rng.normal(size=(2, 2)), rng.normal(size=(2, 2))),
    ]
    sheaf = gamfit.SheafConsistencyPenalty(
        edges=[(0, 1), (1, 2)],
        restriction_ops=restrictions,
        weight=1.0,
        stalk_dims=[2, 2, 2],
    )
    # Reconstruct L via columnwise hvp probes against zero stalk.
    dim = sheaf.total_dim
    z = np.zeros(dim)
    l_dense = np.zeros((dim, dim))
    for j in range(dim):
        ej = np.zeros(dim)
        ej[j] = 1.0
        l_dense[:, j] = sheaf.hvp(z, ej)
    v = rng.normal(size=dim)
    hv = sheaf.hvp(z, v)
    assert hv == pytest.approx(l_dense @ v, abs=1e-10)


def test_harmonic_modes_disconnected_components() -> None:
    # Two disconnected K=2 components with identity restrictions, d = 2 each.
    sheaf = gamfit.SheafConsistencyPenalty(
        edges=[(0, 1), (2, 3)],
        restriction_ops=[np.eye(2), np.eye(2)],
        weight=1.0,
    )
    h = sheaf.harmonic_modes(1e-10)
    # Each component contributes d = 2 harmonic (constant-section) modes.
    assert h == 4


def test_value_is_non_negative_psd_invariant() -> None:
    rng = np.random.default_rng(23)
    sheaf = gamfit.SheafConsistencyPenalty(
        edges=[(0, 1)],
        restriction_ops=[(rng.normal(size=(2, 2)), rng.normal(size=(2, 2)))],
        weight=0.7,
    )
    for _ in range(8):
        z = rng.normal(size=sheaf.total_dim)
        assert sheaf(z) >= -1e-15


def test_hessian_diag_matches_hvp_against_unit_vectors() -> None:
    rng = np.random.default_rng(29)
    sheaf = gamfit.SheafConsistencyPenalty(
        edges=[(0, 1), (1, 2)],
        restriction_ops=[
            (rng.normal(size=(2, 3)), rng.normal(size=(2, 2))),
            np.eye(2),
        ],
        weight=0.4,
        stalk_dims=[3, 2, 2],
    )
    z = rng.normal(size=sheaf.total_dim)
    diag = sheaf.hessian_diag(z)
    assert diag.shape == (sheaf.total_dim,)
    for i in range(sheaf.total_dim):
        e = np.zeros(sheaf.total_dim)
        e[i] = 1.0
        hv = sheaf.hvp(z, e)
        assert diag[i] == pytest.approx(hv[i], abs=1e-12)


def test_single_restriction_form_value() -> None:
    # δs = R·s_0 − s_1 form; choose s_1 = R·s_0 so value = 0.
    r = np.array([[1.0, 2.0], [3.0, 4.0]])
    sheaf = gamfit.SheafConsistencyPenalty(
        edges=[(0, 1)],
        restriction_ops=[r],
        weight=2.0,
    )
    s0 = np.array([1.0, 0.0])
    s1 = r @ s0
    assert sheaf([s0, s1]) == pytest.approx(0.0, abs=1e-12)
    assert sheaf([s0, np.zeros(2)]) == pytest.approx(0.5 * 2.0 * (1.0 + 9.0), abs=1e-12)
