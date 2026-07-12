"""Exact embedding, superposition, and native-coordinate handoff tests."""

from __future__ import annotations

import numpy as np

from bench.bsf_manifold_zoo import ZooData
from bench.manifold_zoo_geometry import ZOO_ORDER, validate_analytic_sample


def test_joint_zoo_superposition_and_native_roundtrip_are_exact() -> None:
    data = ZooData(
        len(ZOO_ORDER),
        32,
        3,
        20260712,
        kinds=list(ZOO_ORDER),
        dgp="toy",
    )
    x, active, contributions = data.sample(5000, 20260713, keep_contributions=True)
    assert contributions is not None
    assert np.all(active.sum(axis=1) == 3)

    reconstructed = np.zeros_like(x)
    for factor_index, (factor, contribution) in enumerate(
        zip(data.factors, contributions, strict=True)
    ):
        frame_identity_error = np.max(
            np.abs(factor.frame @ factor.frame.T - np.eye(factor.frame.shape[0]))
        )
        assert frame_identity_error < 1.0e-12

        rows = contribution["rows"]
        reconstructed[rows] += contribution["m"]
        assert np.array_equal(active[rows, factor_index], np.ones(rows.size, dtype=bool))

        native = contribution["native"]
        validate_analytic_sample(factor.kind, native, contribution["theta"])
        native_roundtrip = (
            factor.sigma * (contribution["m"] @ factor.frame.T) + factor.mu[None, :]
        )
        assert np.max(np.abs(native_roundtrip - native)) < 1.0e-12

    assert np.max(np.abs(reconstructed - x)) < 1.0e-12


def test_ambient_dimension_must_contain_each_native_span() -> None:
    try:
        ZooData(1, 2, 1, 0, kinds=["sphere"], dgp="toy")
    except ValueError as error:
        assert "cannot isometrically embed sphere" in str(error)
    else:
        raise AssertionError("sphere embedding into ambient R^2 must fail")

