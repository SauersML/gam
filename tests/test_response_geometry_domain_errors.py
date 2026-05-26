"""RED tests for issue #223: response-geometry domain errors must raise ValueError.

Every FFI call routed through `gamfit._response_geometry._ffi` currently funnels
its exceptions through `gamfit._exceptions.map_exception`, which defaults to
`GamError` for anything not classified as formula/schema_mismatch/prediction.
Domain errors (antipodal sphere points, non-positive compositions, bad ALR
reference index, dimension mismatches against a base point) are Pythonic
ValueError cases — they describe a caller-supplied numerical/shape violation,
not an internal library failure.

These tests fail today and should pass once the FFI wrapper (or the classifier
in `gamfit/_exceptions.py`) routes geometry domain errors to `ValueError`.
"""

from __future__ import annotations

import numpy as np
import pytest

from gamfit._response_geometry import (
    alr,
    closure,
    clr,
    simplex_log_map,
    sphere_exp_map,
    sphere_log_map,
)


def test_sphere_log_map_antipodal_raises_value_error() -> None:
    base = np.array([1.0, 0.0, 0.0])
    antipode = np.array([[-1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="antipodal"):
        sphere_log_map(antipode, base)


def test_sphere_log_map_dimension_mismatch_raises_value_error() -> None:
    base = np.array([1.0, 0.0])
    values = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        sphere_log_map(values, base)


def test_sphere_exp_map_dimension_mismatch_raises_value_error() -> None:
    base = np.array([1.0, 0.0, 0.0])
    tangent = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError):
        sphere_exp_map(tangent, base)


def test_closure_rejects_negative_mass_with_value_error() -> None:
    bad = np.array([[1.0, -0.5, 0.5]])
    with pytest.raises(ValueError):
        closure(bad)


def test_clr_rejects_zero_mass_with_value_error() -> None:
    bad = np.array([[0.5, 0.0, 0.5]])
    with pytest.raises(ValueError):
        clr(bad)


def test_alr_invalid_reference_raises_value_error() -> None:
    y = np.array([[0.3, 0.3, 0.4]])
    with pytest.raises(ValueError):
        alr(y, reference=42)


def test_simplex_log_map_rejects_non_positive_with_value_error() -> None:
    base = np.array([0.3, 0.3, 0.4])
    bad = np.array([[1.0, -0.3, 0.3]])
    with pytest.raises(ValueError):
        simplex_log_map(bad, base)
