from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

from gamfit._response_geometry import geometry_exp_map, geometry_log_map


def test_response_geometry_alr_reference_roundtrip_preserves_rows() -> None:
    values = np.array(
        [
            [0.15, 0.35, 0.50],
            [0.60, 0.20, 0.20],
            [0.30, 0.30, 0.40],
        ],
        dtype=float,
    )
    tangent, base_point, coordinates = geometry_log_map(
        values,
        geometry="alr",
        reference=0,
    )
    reconstructed = geometry_exp_map(
        tangent,
        geometry="alr",
        base=base_point,
        reference=0,
    )

    assert coordinates == "alr", "ALR geometry should retain ALR tangent coordinates"
    assert np.allclose(
        reconstructed,
        values,
        atol=1.0e-8,
        rtol=1.0e-8,
    ), "ALR log/exp round-trip should reconstruct each composition row"
