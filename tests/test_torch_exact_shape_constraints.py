"""Continuum certificates for torch B-spline shape constraints."""

from __future__ import annotations

import pytest


def test_convex_constraint_is_spanwise_and_sampling_invariant() -> None:
    torch = pytest.importorskip("torch")

    from gamfit import BSpline
    from gamfit.torch.fit import _build_shape_constraint_inequality

    degree = 3
    knots = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.08, 0.37, 0.62, 1.0, 1.0, 1.0, 1.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    smooth = BSpline(knots=knots, degree=degree, shape_constraint="convex")
    width = knots.numel() - degree - 1

    a_96, b_96 = _build_shape_constraint_inequality(
        smooth,
        torch.linspace(0.0, 1.0, 96, dtype=torch.float64),
        "convex",
        width,
    )
    a_320, b_320 = _build_shape_constraint_inequality(
        smooth,
        torch.linspace(0.0, 1.0, 320, dtype=torch.float64),
        "convex",
        width,
    )

    assert a_96.shape == (width - 2, width)
    torch.testing.assert_close(a_96, a_320, rtol=0.0, atol=0.0)
    torch.testing.assert_close(b_96, b_320, rtol=0.0, atol=0.0)
    assert not a_96.requires_grad

    greville = knots.detach()[1:-1].unfold(0, degree, 1).mean(dim=1)
    inverse_spans = (greville[1:] - greville[:-1]).reciprocal()
    expected = torch.zeros_like(a_96)
    rows = torch.arange(width - 2)
    expected[rows, rows] = inverse_spans[:-1]
    expected[rows, rows + 1] = -(inverse_spans[:-1] + inverse_spans[1:])
    expected[rows, rows + 2] = inverse_spans[1:]
    expected /= torch.linalg.vector_norm(expected, dim=1, keepdim=True)
    torch.testing.assert_close(a_96, expected, rtol=1e-15, atol=1e-15)


def test_shape_constraint_rejects_periodic_coefficient_chart() -> None:
    torch = pytest.importorskip("torch")

    from gamfit import BSpline
    from gamfit.torch.fit import _build_shape_constraint_inequality

    knots = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    smooth = BSpline(
        knots=knots,
        degree=3,
        periodic=True,
        shape_constraint="monotone_increasing",
    )
    with pytest.raises(NotImplementedError, match="requires an open BSpline"):
        _build_shape_constraint_inequality(
            smooth,
            torch.linspace(0.0, 1.0, 40, dtype=torch.float64),
            "monotone_increasing",
            knots.numel() - 1,
        )
