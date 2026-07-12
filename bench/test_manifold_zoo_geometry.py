"""Defining-equation tests for every analytic manifold-zoo object."""

from __future__ import annotations

import numpy as np

from bench.manifold_zoo_geometry import (
    ZOO,
    ZOO_ORDER,
    analytic_points,
    validate_analytic_sample,
)


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


def test_native_objects_satisfy_independent_implicit_invariants() -> None:
    rng = np.random.default_rng(3)

    disk, disk_parameters = ZOO["disk"].sampler(rng, 50_000)
    disk_radius = np.linalg.norm(disk, axis=1)
    assert np.max(np.abs(disk_radius - disk_parameters[:, 0])) < 1.0e-12
    assert abs(float(np.mean(disk_radius**2)) - 0.5) < 0.01

    sphere, _ = ZOO["sphere"].sampler(rng, 50_000)
    assert np.max(np.abs(np.sum(sphere * sphere, axis=1) - 1.0)) < 1.0e-12
    assert abs(float(np.mean(sphere[:, 2] ** 2)) - 1.0 / 3.0) < 0.01

    torus, _ = ZOO["torus"].sampler(rng, 50_000)
    torus_radius = np.hypot(torus[:, 0], torus[:, 1])
    torus_residual = (torus_radius - 1.0) ** 2 + torus[:, 2] ** 2 - 0.4**2
    assert np.max(np.abs(torus_residual)) < 1.0e-12

    mobius, mobius_parameters = ZOO["mobius"].sampler(rng, 50_000)
    mobius_radius = np.hypot(mobius[:, 0], mobius[:, 1])
    mobius_width = mobius_parameters[:, 1]
    mobius_residual = (mobius_radius - 1.0) ** 2 + mobius[:, 2] ** 2 - mobius_width**2
    assert np.max(np.abs(mobius_residual)) < 1.0e-12

    swiss, swiss_parameters = ZOO["swiss"].sampler(rng, 50_000)
    assert np.max(
        np.abs(np.hypot(swiss[:, 0], swiss[:, 2]) - swiss_parameters[:, 0])
    ) < 1.0e-12
    assert np.max(np.abs(swiss[:, 1] - swiss_parameters[:, 1])) < 1.0e-12

    helix, helix_parameters = ZOO["helix"].sampler(rng, 50_000)
    assert np.max(np.abs(helix[:, 0] ** 2 + helix[:, 1] ** 2 - 1.0)) < 1.0e-12
    assert np.max(np.abs(helix[:, 2] - 0.25 * helix_parameters[:, 0])) < 1.0e-12


def test_random_isometry_roundtrip_restores_raw_analytic_coordinates() -> None:
    rng = np.random.default_rng(4)
    for kind in ZOO_ORDER:
        raw, _ = ZOO[kind].sampler(rng, 10_000)
        mean = raw.mean(axis=0)
        centered = raw - mean
        scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        q, _ = np.linalg.qr(rng.standard_normal((32, ZOO[kind].span_dim)))
        frame = q.T
        assert np.max(np.abs(frame @ frame.T - np.eye(ZOO[kind].span_dim))) < 1.0e-12
        ambient = (centered / scale) @ frame
        restored = scale * (ambient @ frame.T) + mean
        assert np.max(np.abs(restored - raw)) < 1.0e-12


def test_closed_seams_and_open_helix_are_topologically_correct() -> None:
    tau = 2.0 * np.pi
    width = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])

    circle_seam = analytic_points("circle", np.array([[0.0], [tau]]))
    assert np.max(np.abs(circle_seam[0] - circle_seam[1])) < 1.0e-12

    disk_seam = analytic_points("disk", np.array([[0.7, 0.0], [0.7, tau]]))
    assert np.max(np.abs(disk_seam[0] - disk_seam[1])) < 1.0e-12

    sphere_longitude_seam = analytic_points(
        "sphere", np.array([[1.1, 0.0], [1.1, tau]])
    )
    assert np.max(np.abs(sphere_longitude_seam[0] - sphere_longitude_seam[1])) < 1.0e-12
    sphere_north_pole = analytic_points(
        "sphere", np.column_stack((np.zeros(6), np.linspace(0.0, tau, 6)))
    )
    assert np.max(np.abs(sphere_north_pole - np.array([0.0, 0.0, 1.0]))) < 1.0e-12

    torus_azimuth_seam = analytic_points(
        "torus", np.array([[0.0, 0.8], [tau, 0.8]])
    )
    torus_tube_seam = analytic_points(
        "torus", np.array([[0.8, 0.0], [0.8, tau]])
    )
    assert np.max(np.abs(torus_azimuth_seam[0] - torus_azimuth_seam[1])) < 1.0e-12
    assert np.max(np.abs(torus_tube_seam[0] - torus_tube_seam[1])) < 1.0e-12

    mobius_start = analytic_points("mobius", np.column_stack((np.zeros(5), width)))
    mobius_end = analytic_points("mobius", np.column_stack((np.full(5, tau), -width)))
    assert np.max(np.abs(mobius_start - mobius_end)) < 1.0e-12

    helix_ends = analytic_points("helix", np.array([[0.0], [4.0 * np.pi]]))
    assert np.max(np.abs(helix_ends[0, :2] - helix_ends[1, :2])) < 1.0e-12
    assert abs(float(helix_ends[1, 2] - helix_ends[0, 2]) - np.pi) < 1.0e-12
