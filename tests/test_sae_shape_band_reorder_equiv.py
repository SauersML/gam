"""#2091 — the Rust `sae_periodic_shape_band_reorder` owner must reproduce the
Python `_periodic_shape_band` reorder bit-for-bit.

`from_payload` sorts a periodic atom's shape band ascending by its single
coordinate column (`np.argsort(..., kind="mergesort")`, a STABLE sort) and
reindexes the amplitude-correct mean / sd to match. A later increment builds the
`ManifoldSaePayload` directly from the raw fit payload and reorders through this
Rust owner, so it must match the dataclass derivation exactly, including tie
stability and the drop-the-band-without-a-mean rule."""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._sae_manifold import rust_module  # noqa: E402


def _rust_reorder(coords, mean, sd):
    fn = getattr(rust_module(), "sae_periodic_shape_band_reorder", None)
    if fn is None:
        pytest.skip("wheel predates sae_periodic_shape_band_reorder (#2091 coercion slice)")
    return fn(coords, mean, sd)


def _py_reorder(coords, mean, sd):
    """The exact `_periodic_shape_band` reorder (sort + reindex)."""
    if coords is None:
        return None, None, None
    coords = np.ascontiguousarray(np.asarray(coords, dtype=float))
    order = np.argsort(coords[:, 0], kind="mergesort")
    coords = np.ascontiguousarray(coords[order])
    if mean is None:
        return None, None, None
    mean = np.asarray(mean, dtype=float)[order]
    sd = None if sd is None else np.asarray(sd, dtype=float)[order]
    return coords, mean, sd


def _assert_triples_equal(got, expected):
    for g, e in zip(got, expected):
        if e is None:
            assert g is None
        else:
            np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


def test_shape_band_reorder_matches_python_with_ties():
    coords = np.array([[0.9], [0.1], [0.5], [0.1]], dtype=float)
    mean = np.array([9.0, 1.0, 5.0, 2.0], dtype=float)
    sd = np.array([0.9, 0.1, 0.5, 0.2], dtype=float)
    _assert_triples_equal(_rust_reorder(coords, mean, sd), _py_reorder(coords, mean, sd))


def test_shape_band_reorder_matches_python_no_sd():
    rng = np.random.default_rng(4)
    coords = rng.standard_normal((25, 1))
    mean = rng.standard_normal(25)
    _assert_triples_equal(_rust_reorder(coords, mean, None), _py_reorder(coords, mean, None))


def test_shape_band_reorder_drops_band_without_mean():
    coords = np.array([[0.1], [0.2]], dtype=float)
    _assert_triples_equal(_rust_reorder(coords, None, None), (None, None, None))


def test_shape_band_reorder_absent_coords_all_none():
    _assert_triples_equal(_rust_reorder(None, np.array([1.0]), None), (None, None, None))


def test_shape_band_reorder_rejects_multicolumn_coords():
    coords = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    with pytest.raises(ValueError):
        _rust_reorder(coords, np.array([1.0, 2.0]), None)
