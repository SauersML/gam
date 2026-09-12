"""Contract tests for issue #224: a Sphere descriptor's basis size must not
depend on the number of evaluation rows, and ``basis_size`` must be answerable
without a Rust round-trip on synthetic inputs.

See: https://github.com/SauersML/gam/issues/224

Each test asserts the descriptor contract the Python API advertises. Without
explicit ``centers=``, ``Sphere.evaluate`` resolves centers by farthest-point
sampling from the evaluation rows and refuses fewer than ``n_centers`` rows, so
the tests that evaluate fewer rows than centers fail. That is the #224 defect,
still unfixed, not an expected outcome.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def test_sphere_evaluate_more_centers_than_rows_numpy():
    """n_centers=12, eval on 6 rows: must succeed (centers are a property
    of the spec, not of the eval set)."""
    rng = np.random.default_rng(0)
    lat = rng.uniform(-60.0, 60.0, size=6)
    lon = rng.uniform(-180.0, 180.0, size=6)

    spec = gamfit.Sphere(n_centers=12)
    design = spec.evaluate(lat, lon, backend="numpy")
    arr = np.asarray(design)
    assert arr.shape[0] == 6
    assert arr.shape[1] >= 12


def test_sphere_evaluate_more_centers_than_rows_torch():
    """Same contract under the torch backend."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    lat = rng.uniform(-60.0, 60.0, size=6)
    lon = rng.uniform(-180.0, 180.0, size=6)

    spec = gamfit.Sphere(n_centers=12)
    design = spec.evaluate(lat, lon, backend="torch")
    if hasattr(design, "detach"):
        arr = design.detach().cpu().numpy()
    else:
        arr = np.asarray(design)
    assert arr.shape[0] == 6
    assert arr.shape[1] >= 12


def test_sphere_basis_size_default_no_eval():
    """``basis_size`` must be answerable for a default-constructed Sphere
    (n_centers=50) without first evaluating on >=50 rows. It must not
    probe Rust with a 2-row synthetic input.

    The Sphere kernel basis carries one identifiability (sum-to-zero)
    constraint, so basis_size is n_centers - 1 = 49 for the default."""
    spec = gamfit.Sphere()  # default n_centers=50
    size = spec.basis_size
    assert isinstance(size, int)
    assert size >= 49


def test_sphere_basis_size_custom_centers_before_evaluate():
    """basis_size accessed BEFORE any evaluate call must work and reflect
    the configured n_centers."""
    spec = gamfit.Sphere(n_centers=37)
    size = spec.basis_size
    assert isinstance(size, int)
    # n_centers - 1 (one identifiability constraint) = 36.
    assert size >= 36


def test_sphere_basis_size_then_evaluate_consistent():
    """basis_size queried first must agree with the column count of the
    eventual evaluation (even when eval has fewer rows than centers).

    The raw evaluate() design exposes one column per center (n_centers),
    while basis_size reports the identifiable dimension after the single
    sum-to-zero constraint is applied: basis_size == n_centers - 1, so the
    raw design has exactly basis_size + 1 columns. Either way the count is a
    property of the spec's centers, NOT of the eval row count (issue #224)."""
    spec = gamfit.Sphere(n_centers=20)
    size = spec.basis_size

    rng = np.random.default_rng(2)
    lat = rng.uniform(-60.0, 60.0, size=8)
    lon = rng.uniform(-180.0, 180.0, size=8)
    design = np.asarray(spec.evaluate(lat, lon, backend="numpy"))
    assert design.shape[1] == size + 1


def test_sphere_explicit_centers_round_trip_if_supported():
    """Explicit ``centers=`` are stored verbatim and decouple basis size from
    eval row count: 10 supplied centers evaluated on 3 rows give a 3-row raw
    design with one column per center (``basis_size + 1``, the sum-to-zero
    constraint removing one), with no row-count requirement."""
    rng = np.random.default_rng(3)
    centers = np.column_stack(
        [rng.uniform(-60.0, 60.0, size=10), rng.uniform(-180.0, 180.0, size=10)]
    )
    spec = gamfit.Sphere(centers=centers)
    np.testing.assert_array_equal(np.asarray(spec.centers, dtype=np.float64), centers)
    assert spec.basis_size == 9

    lat = rng.uniform(-60.0, 60.0, size=3)  # far fewer rows than centers
    lon = rng.uniform(-180.0, 180.0, size=3)
    design = np.asarray(spec.evaluate(lat, lon, backend="numpy"))
    assert design.shape == (3, spec.basis_size + 1)
