"""Contract tests for issue #237.

Public contract (``gamfit/smooth.py:124-129``):

* ``m=2, d=1`` — natural cubic smoothing spline
* ``m=2, d=2`` — thin-plate spline
* ``m=2, d ≥ 3`` — Duchon's generalized thin-plate spline

The descriptor ``gamfit.Duchon(centers, m=2).evaluate(...)`` must succeed at
every advertised ``(d, m)`` default. The PyFFI ``gamfit.duchon_basis``
(basis-only path, no penalty returned to Python) must also accept these
defaults without tripping the D2 collocation validator.

#237 is fixed; the note above the PyFFI tests records the cause. A failure
here is a regression.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _torch():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Descriptor evaluate() — every advertised (d, m) default
# ---------------------------------------------------------------------------


def test_descriptor_evaluate_d1_m2_natural_cubic() -> None:
    torch = _torch()
    centers = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    spec = gamfit.Duchon(centers=centers, m=2)
    x = torch.linspace(0.05, 0.95, 11, dtype=torch.float64)
    phi = spec.evaluate(x)
    assert phi.shape == (11, 8)
    assert torch.isfinite(phi).all()


def test_descriptor_evaluate_d2_m2_thin_plate() -> None:
    torch = _torch()
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((9, 2))
    spec = gamfit.Duchon(centers=centers, m=2)
    pts = rng.standard_normal((5, 2))
    x = torch.tensor(pts[:, 0], dtype=torch.float64)
    y = torch.tensor(pts[:, 1], dtype=torch.float64)
    phi = spec.evaluate(x, y)
    assert phi.shape == (5, 9)
    assert torch.isfinite(phi).all()


def test_descriptor_evaluate_d3_m2_generalized_tps() -> None:
    torch = _torch()
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((12, 3))
    spec = gamfit.Duchon(centers=centers, m=2)
    pts = rng.standard_normal((6, 3))
    x = torch.tensor(pts[:, 0], dtype=torch.float64)
    y = torch.tensor(pts[:, 1], dtype=torch.float64)
    z = torch.tensor(pts[:, 2], dtype=torch.float64)
    phi = spec.evaluate(x, y, z)
    assert phi.shape == (6, 12)
    assert torch.isfinite(phi).all()


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_descriptor_evaluate_d2_various_m(m: int) -> None:
    torch = _torch()
    rng = np.random.default_rng(2 + m)
    centers = rng.standard_normal((10, 2))
    spec = gamfit.Duchon(centers=centers, m=m)
    pts = rng.standard_normal((4, 2))
    x = torch.tensor(pts[:, 0], dtype=torch.float64)
    y = torch.tensor(pts[:, 1], dtype=torch.float64)
    phi = spec.evaluate(x, y)
    assert phi.shape == (4, 10)
    assert torch.isfinite(phi).all()


# ---------------------------------------------------------------------------
# PyFFI basis-only path — `gamfit.duchon_basis` must not require D2 collocation
# ---------------------------------------------------------------------------


# #237 — FIXED. The old note here said the pyffi path "returns FEWER columns than
# the number of centers for several (d, m, n_centers) configurations", listed
# "11 centers d=3 m=2 -> 9 cols", and blamed the D2 collocation validator. Two of
# those three claims were wrong, and the framing sent the reader to the wrong file.
#
# The width was never a function of (d, m, n_centers). It was a function of the
# EVALUATION ROW COUNT. Holding the spec completely fixed (the same 12 centers,
# d=2, m=2) and varying only how many points you evaluate at:
#
#       npts     1    2    3    4    5    6    8    9   10   12   16   30
#       before   4    5    6    7    8    9   11   12   12   12   12   12
#       after   12   12   12   12   12   12   12   12   12   12   12   12
#
#   i.e. `cols = min(K, n_points + d + 1)`, independent of m — verified over 63
#   configurations (d in {2,3,4} x m in {1,2,3} x K in {6..12}).
#
# So "11 centers d=3 m=2 -> 9 cols" was really 8, because that fixture passes 4
# evaluation points (4 + 3 + 1 = 8); the comment's numbers came from a different
# fixture. And the D2 collocation validator was not involved at all —
# `duchon_basis` resolves with `max_op = 0` and disables all three operator
# penalties, so the width loss happened well after that check.
#
# The real cause was the #1355 data-metric radial chart: `build_duchon_basis`
# forms `G_c = (K.Z)^T(K.Z)` from the REALIZED design and keeps only the
# eigen-directions above a numerical floor — "design columns with no realized
# data support", in the whitening step's own words. `G_c` has rank at most n, so
# the basis collapsed to the rank of whatever frame it was handed. Fits were
# never affected: a fit FREEZES that chart and replays it at predict time. Only
# the basis-only primitive, which has no fit and nothing to freeze, recomputed a
# fresh chart per call.
#
# `build_duchon_basis_spec_chart` now derives the chart from the CENTERS, so the
# width is a property of the spec: 63 of 63 configurations emit exactly K.
# The asserts below were RIGHT all along and none of them was touched.
def test_pyffi_duchon_basis_d2_m2_default() -> None:
    rng = np.random.default_rng(10)
    centers = rng.standard_normal((9, 2))
    pts = rng.standard_normal((5, 2))
    basis = gamfit.duchon_basis(pts, centers, m=2)
    assert basis.shape == (5, 9)
    assert np.all(np.isfinite(basis))


def test_pyffi_duchon_basis_d3_m2_default() -> None:
    rng = np.random.default_rng(11)
    centers = rng.standard_normal((11, 3))
    pts = rng.standard_normal((4, 3))
    basis = gamfit.duchon_basis(pts, centers, m=2)
    assert basis.shape == (4, 11)
    assert np.all(np.isfinite(basis))


@pytest.mark.parametrize("d", [2, 3, 4])
def test_pyffi_duchon_basis_higher_d_m2_default(d: int) -> None:
    rng = np.random.default_rng(20 + d)
    centers = rng.standard_normal((max(8, 2 * d), d))
    pts = rng.standard_normal((5, d))
    basis = gamfit.duchon_basis(pts, centers, m=2)
    assert basis.shape == (5, centers.shape[0])
    assert np.all(np.isfinite(basis))


@pytest.mark.parametrize("m", [1, 2, 3])
def test_pyffi_duchon_basis_d2_various_m(m: int) -> None:
    rng = np.random.default_rng(30 + m)
    centers = rng.standard_normal((10, 2))
    pts = rng.standard_normal((4, 2))
    basis = gamfit.duchon_basis(pts, centers, m=m)
    assert basis.shape == (4, 10)
    assert np.all(np.isfinite(basis))
