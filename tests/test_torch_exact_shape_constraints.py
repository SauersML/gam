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

    # Rust constructs the stable derivative-control row directly from adjacent
    # knot-window widths instead of averaging Greville abscissae or taking
    # reciprocals. Positive row scaling makes the two forms the same cone.
    detached = knots.detach()
    spans = torch.stack(
        [
            (detached[i + degree + 1] - detached[i + 1]) / degree
            for i in range(width - 1)
        ]
    )
    expected = torch.zeros_like(a_96)
    rows = torch.arange(width - 2)
    row_scales = torch.maximum(spans[:-1], spans[1:])
    left = spans[:-1] / row_scales
    right = spans[1:] / row_scales
    expected[rows, rows] = right
    expected[rows, rows + 1] = -(left + right)
    expected[rows, rows + 2] = left
    expected /= torch.linalg.vector_norm(expected, dim=1, keepdim=True)
    torch.testing.assert_close(a_96, expected, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize(
    ("kind", "sign", "order"),
    [
        ("monotone_increasing", 1.0, 1),
        ("monotone_decreasing", -1.0, 1),
        ("convex", 1.0, 2),
        ("concave", -1.0, 2),
    ],
)
def test_all_shape_kinds_use_the_rust_continuum_cone(
    kind: str, sign: float, order: int,
) -> None:
    torch = pytest.importorskip("torch")

    from gamfit import BSpline
    from gamfit.torch.fit import _build_shape_constraint_inequality

    degree = 3
    knots = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.11, 0.48, 1.0, 1.0, 1.0, 1.0],
        dtype=torch.float64,
    )
    width = knots.numel() - degree - 1
    smooth = BSpline(knots=knots, degree=degree, shape_constraint=kind)
    a, b = _build_shape_constraint_inequality(
        smooth, torch.tensor([0.2, 0.8], dtype=torch.float64), kind, width,
    )

    assert a.shape == (width - order, width)
    assert b.shape == (width - order,)
    if order == 1:
        expected = torch.zeros_like(a)
        rows = torch.arange(width - 1)
        expected[rows, rows] = -sign
        expected[rows, rows + 1] = sign
        expected /= torch.linalg.vector_norm(expected, dim=1, keepdim=True)
        torch.testing.assert_close(a, expected, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(b) == 0


def test_affine_linear_spline_has_vacuous_curvature_cone() -> None:
    torch = pytest.importorskip("torch")

    from gamfit import BSpline
    from gamfit.torch.fit import _build_shape_constraint_inequality

    knots = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    for kind in ("convex", "concave"):
        smooth = BSpline(knots=knots, degree=1, shape_constraint=kind)
        a, b = _build_shape_constraint_inequality(
            smooth, torch.tensor([0.25, 0.75]), kind, 2,
        )
        assert a.shape == (0, 2)
        assert b.shape == (0,)


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
