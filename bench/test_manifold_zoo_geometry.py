"""Defining-equation tests for every analytic manifold-zoo object."""

from __future__ import annotations

import numpy as np

from bench.manifold_zoo_geometry import ZOO, ZOO_ORDER, validate_analytic_sample


def test_all_zoo_objects_satisfy_their_defining_equations() -> None:
    rng = np.random.default_rng(20260712)
    for kind in ZOO_ORDER:
        points, parameters = ZOO[kind].sampler(rng, 4096)
        validate_analytic_sample(kind, points, parameters)


def test_sphere_is_a_surface_not_a_loop() -> None:
    points, _ = ZOO["sphere"].sampler(np.random.default_rng(1), 20_000)
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    assert singular[2] / singular[0] > 0.9
    assert np.max(np.abs(np.linalg.norm(points, axis=1) - 1.0)) < 1.0e-12


def test_torus_mobius_swiss_and_helix_have_real_axial_extent() -> None:
    rng = np.random.default_rng(2)
    for kind in ("torus", "mobius", "swiss", "helix"):
        points, parameters = ZOO[kind].sampler(rng, 20_000)
        validate_analytic_sample(kind, points, parameters)
        assert np.ptp(points[:, 2]) > 0.7, f"{kind} collapsed to a planar loop"

