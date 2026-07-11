"""Regression tests for metric-correct PyTorch Riemannian steps."""

from __future__ import annotations

import math
from typing import Any

import pytest

from gamfit import RustExtensionUnavailableError


torch = pytest.importorskip("torch")


def setup_module(module) -> None:  # noqa: D401 - pytest hook
    from gamfit._binding import rust_module

    try:
        rust_module()
    except RustExtensionUnavailableError as exc:  # pragma: no cover - env specific
        pytest.skip(f"gamfit._rust unavailable: {exc}")


def test_spd_step_metric_raises_differential_and_enables_closure_grad() -> None:
    from gamfit.torch import RiemannianRetraction

    point = torch.nn.Parameter(
        torch.tensor([[2.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    )
    learning_rate = 0.05
    optimizer = RiemannianRetraction(
        [point], {"kind": "spd", "n": 2}, lr=learning_rate
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        closure_calls += 1
        assert torch.is_grad_enabled()
        optimizer.zero_grad()
        loss = point[0, 0] + point[0, 3]
        loss.backward()
        return loss

    loss = optimizer.step(closure)

    assert closure_calls == 1
    assert loss is not None
    assert float(loss) == pytest.approx(3.0)
    # For P=diag(2,1) and E=I, affine-invariant metric raising gives
    # grad(P)=P E P=diag(4,1). The SPD exponential/retraction is therefore
    # diag(2 exp(-2 eta), exp(-eta)). A projection-only step would have the
    # different first coordinate 2 exp(-eta/2).
    expected = torch.tensor(
        [[
            2.0 * math.exp(-2.0 * learning_rate),
            0.0,
            0.0,
            math.exp(-learning_rate),
        ]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(point.detach(), expected, rtol=1.0e-12, atol=1.0e-12)


def test_stiefel_step_uses_canonical_riesz_representative() -> None:
    from gamfit.torch import RiemannianRetraction

    y = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
    )
    differential = torch.tensor(
        [[0.4, 1.2], [-0.3, 0.8], [0.5, -0.7]], dtype=torch.float64
    )
    point = torch.nn.Parameter(y.reshape(1, -1).clone())
    point.grad = differential.reshape(1, -1).clone()
    learning_rate = 1.0e-7

    optimizer = RiemannianRetraction(
        [point], {"kind": "stiefel", "k": 2, "n": 3}, lr=learning_rate
    )
    optimizer.step()

    canonical_gradient = differential - y @ differential.T @ y
    observed_direction = (
        point.detach().reshape(3, 2) - y
    ) / learning_rate
    torch.testing.assert_close(
        observed_direction,
        -canonical_gradient,
        rtol=2.0e-6,
        atol=2.0e-6,
    )

    embedded_projection = differential - y @ (
        0.5 * (y.T @ differential + differential.T @ y)
    )
    assert torch.linalg.vector_norm(
        observed_direction + embedded_projection
    ) > 0.1


def test_inner_steps_api_is_removed() -> None:
    from gamfit.torch import RiemannianRetraction

    point = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float64))
    constructor: Any = RiemannianRetraction
    with pytest.raises(TypeError, match="inner_steps"):
        constructor(
            [point], {"kind": "euclidean", "dim": 1}, inner_steps=2
        )
