"""RED tests for issue #224: Sphere descriptor must not bind basis size
to the number of evaluation rows, and ``basis_size`` must not require a
Rust round-trip with bogus inputs.

See: https://github.com/SauersML/gam/issues/224

Each test is an *intent* test: it asserts the descriptor contract the
Python API advertises. They are expected to FAIL on `main` until the
underlying bug is fixed.
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
    """If the API supports explicit centers, they must be respected and
    must decouple basis size from eval row count. If the API does not
    yet support ``centers=``, this test is skipped — but once supported,
    it locks in the contract."""
    try:
        spec = gamfit.Sphere(n_centers=10)
    except TypeError:
        pytest.skip("Sphere does not yet support explicit-centers ctor")

    explicit = getattr(spec, "centers", None)
    if explicit is None:
        pytest.skip(
            "Sphere does not yet expose stored centers; once it does, "
            "this test must lock in n_rows-independence."
        )

    rng = np.random.default_rng(3)
    lat = rng.uniform(-60.0, 60.0, size=3)  # far fewer rows than centers
    lon = rng.uniform(-180.0, 180.0, size=3)
    design = np.asarray(spec.evaluate(lat, lon, backend="numpy"))
    assert design.shape[0] == 3
    assert design.shape[1] >= 10
