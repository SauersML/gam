"""ILR isometry and ALR Aitchison-metric correctness (issue #626).

The additive log-ratio (ALR) chart ``z = log(x_keep / x_ref)`` is a valid
coordinate system on the simplex but is NOT isometric to Aitchison geometry:
the Aitchison Gram in ALR coordinates is ``G = I_{D-1} − (1/D)·11ᵀ`` (for
``D = 3`` it equals ``[[2/3, -1/3], [-1/3, 2/3]] ≠ I``). Fitting an ordinary
Euclidean Gaussian model in ALR therefore minimizes the wrong residual norm.

The isometric log-ratio (ILR) chart, built from a Helmert/orthonormal contrast
basis of the centered-log-ratio (CLR) sum-zero hyperplane, IS Euclidean-isometric
to Aitchison geometry: the ordinary Euclidean distance between ILR coordinates
equals the Aitchison distance on the simplex, so plain Gaussian fitting becomes
Aitchison-correct with no extra metric weighting.

These tests pin:
  * ILR Euclidean distance == Aitchison distance (the defining isometry),
  * ILR log/exp maps round-trip through ``simplex_log_map`` / ``simplex_exp_map``,
  * ALR forward/inverse transforms round-trip,
  * the ALR Aitchison Gram has the expected non-identity structure, and that
    re-weighting the ALR coordinate difference by that Gram recovers the
    Aitchison distance (which plain Euclidean ALR fails to do).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

try:
    import torch

    from gamfit.torch.geometry import (
        aitchison_metric,
        alr,
        clr,
        ilr,
        inverse_alr,
        inverse_ilr,
        simplex_exp_map,
        simplex_log_map,
    )
except Exception:  # pragma: no cover - torch optional, conftest skips collection
    torch = None  # type: ignore[assignment]


def _aitchison_distance(x: "torch.Tensor", y: "torch.Tensor") -> float:
    """Ground-truth Aitchison distance d_A(x, y) = ||clr(x) - clr(y)||_2."""
    diff = clr(x) - clr(y)
    return float(torch.linalg.norm(diff, dim=1)[0])


def test_ilr_distance_equals_aitchison_distance() -> None:
    # Two arbitrary strictly-positive compositions on the D=4 simplex.
    x = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    y = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float64)

    ilr_dist = float(torch.linalg.norm(ilr(x) - ilr(y), dim=1)[0])
    aitchison_dist = _aitchison_distance(x, y)

    assert ilr_dist == pytest.approx(aitchison_dist, abs=1e-12), (
        "ILR Euclidean distance must equal the Aitchison distance "
        f"(isometry): {ilr_dist} vs {aitchison_dist}"
    )
    # And the distance is genuinely non-trivial (not a degenerate zero).
    assert ilr_dist > 1e-3


def test_ilr_distance_isometry_random_batch() -> None:
    rng = np.random.default_rng(626)
    for d in (3, 4, 5, 7):
        comps = rng.gamma(shape=1.0, size=(8, d)) + 1e-3
        t = torch.tensor(comps, dtype=torch.float64)
        z = ilr(t)
        c = clr(t)
        for i in range(t.shape[0]):
            for j in range(i + 1, t.shape[0]):
                ilr_d = float(torch.linalg.norm(z[i] - z[j]))
                ait_d = float(torch.linalg.norm(c[i] - c[j]))
                assert ilr_d == pytest.approx(ait_d, abs=1e-10), (
                    f"D={d} pair ({i},{j}): ILR distance {ilr_d} != "
                    f"Aitchison distance {ait_d}"
                )


def test_ilr_forward_inverse_round_trip() -> None:
    rng = np.random.default_rng(1)
    comps = rng.gamma(shape=1.0, size=(6, 5)) + 1e-3
    t = torch.tensor(comps, dtype=torch.float64)
    closed = t / t.sum(dim=1, keepdim=True)
    recovered = inverse_ilr(ilr(closed))
    assert torch.allclose(recovered, closed, atol=1e-12), (
        "inverse_ilr(ilr(x)) must recover the closed composition"
    )


def test_simplex_ilr_log_exp_round_trip() -> None:
    # Default 'simplex' coordinate must be ILR; log then exp at a base recovers x.
    rng = np.random.default_rng(2)
    comps = rng.gamma(shape=1.0, size=(5, 4)) + 1e-3
    x = torch.tensor(comps, dtype=torch.float64)
    base = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)

    for coord in ("simplex", "ilr"):
        tangent = simplex_log_map(x, base, coordinates=coord)
        # ILR tangent has D-1 columns.
        assert tangent.shape[1] == x.shape[1] - 1
        back = simplex_exp_map(tangent, base, coordinates=coord)
        closed = x / x.sum(dim=1, keepdim=True)
        assert torch.allclose(back, closed, atol=1e-10), (
            f"ILR ('{coord}') log/exp round-trip failed"
        )


def test_alr_forward_inverse_round_trip() -> None:
    rng = np.random.default_rng(3)
    comps = rng.gamma(shape=1.0, size=(7, 4)) + 1e-3
    x = torch.tensor(comps, dtype=torch.float64)
    closed = x / x.sum(dim=1, keepdim=True)
    for ref in (-1, 0, 1, 2):
        recovered = inverse_alr(alr(closed, reference=ref), reference=ref)
        assert torch.allclose(recovered, closed, atol=1e-12), (
            f"inverse_alr(alr(x)) must round-trip for reference={ref}"
        )


def test_alr_gram_is_non_identity_d3() -> None:
    # Core claim of #626: ALR is NOT isometric; its Aitchison Gram for D=3 is
    # [[2/3, -1/3], [-1/3, 2/3]] != I.
    g = aitchison_metric(3, dtype=torch.float64)
    expected = torch.tensor([[2.0 / 3.0, -1.0 / 3.0], [-1.0 / 3.0, 2.0 / 3.0]], dtype=torch.float64)
    assert torch.allclose(g, expected, atol=1e-12)
    assert not torch.allclose(g, torch.eye(2, dtype=torch.float64)), (
        "ALR Aitchison Gram must not be the identity (ALR is non-isometric)"
    )


def test_alr_euclidean_underweights_but_gram_recovers_aitchison() -> None:
    # In ALR coordinates, the plain Euclidean difference norm is NOT the
    # Aitchison distance, but re-weighting by the Aitchison Gram G recovers it.
    x = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float64)
    y = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)

    cx = x / x.sum(dim=1, keepdim=True)
    cy = y / y.sum(dim=1, keepdim=True)
    dz = (alr(cx) - alr(cy))[0]  # (D-1,)
    g = aitchison_metric(3, dtype=torch.float64)

    euclid_alr = float(torch.linalg.norm(dz))
    gram_norm = float(torch.sqrt(dz @ (g @ dz)))
    aitchison = _aitchison_distance(cx, cy)

    # Plain Euclidean ALR is the WRONG norm.
    assert not (euclid_alr == pytest.approx(aitchison, abs=1e-6)), (
        "plain Euclidean ALR distance must NOT equal the Aitchison distance"
    )
    # Gram-weighted ALR is the RIGHT norm.
    assert gram_norm == pytest.approx(aitchison, abs=1e-10), (
        f"Gram-weighted ALR distance must equal Aitchison: {gram_norm} vs {aitchison}"
    )
    # And ILR matches the Aitchison distance with no weighting at all.
    ilr_dist = float(torch.linalg.norm(ilr(cx) - ilr(cy), dim=1)[0])
    assert ilr_dist == pytest.approx(aitchison, abs=1e-10)
