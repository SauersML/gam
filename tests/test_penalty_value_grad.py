"""Penalty descriptors' value_grad/hvp match torch autograd to high precision.

For each penalty wrapper, assert that

    value_grad(t)[1] == autograd_grad(value(t), t)
    hvp(t, v)       == autograd_grad((g * v).sum(), t)

at random sample points. Tolerance is 1e-5 (the Rust analytic gradient is
double precision; the discrepancy is essentially round-off plus the rho=0
default the descriptors use).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import gamfit
from gamfit._penalty_descriptors import (
    ARDPenalty,
    BlockOrthogonalityDescriptor,
    OrderedBetaBernoulliPenalty,
)


def _autograd_grad(penalty: gamfit.PenaltyDescriptor, t: torch.Tensor) -> torch.Tensor:
    t = t.detach().clone().requires_grad_(True)
    v = penalty.value(t)
    (g,) = torch.autograd.grad(v, t, create_graph=False)
    return g


def _autograd_hvp(
    penalty: gamfit.PenaltyDescriptor, t: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    t = t.detach().clone().requires_grad_(True)
    val = penalty.value(t)
    (g,) = torch.autograd.grad(val, t, create_graph=True)
    inner = (g * v).sum()
    (hv,) = torch.autograd.grad(inner, t)
    return hv


@pytest.mark.parametrize(
    "penalty_factory",
    [
        lambda: ARDPenalty(weight=0.3),
        lambda: OrderedBetaBernoulliPenalty(alpha=1.0, tau=1.0),
        lambda: BlockOrthogonalityDescriptor(groups=[[0, 1], [2, 3]], weight=0.5, n_eff=8),
    ],
)
def test_value_grad_matches_autograd(penalty_factory) -> None:
    penalty = penalty_factory()
    torch.manual_seed(13)
    t = torch.randn(8, 4, dtype=torch.float64) * 0.4
    _, g_native = penalty.value_grad(t)
    g_auto = _autograd_grad(penalty, t)
    assert g_native.shape == t.shape
    assert torch.allclose(g_native, g_auto, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "penalty_factory",
    [
        lambda: ARDPenalty(weight=0.3),
        lambda: OrderedBetaBernoulliPenalty(alpha=1.0, tau=1.0),
    ],
)
def test_hvp_matches_autograd(penalty_factory) -> None:
    penalty = penalty_factory()
    torch.manual_seed(17)
    t = torch.randn(6, 3, dtype=torch.float64) * 0.4
    v = torch.randn_like(t)
    hv_native = penalty.hvp(t, v)
    hv_auto = _autograd_hvp(penalty, t, v)
    assert hv_native.shape == t.shape
    assert torch.allclose(hv_native, hv_auto, atol=1e-5, rtol=1e-5)


def test_composite_hvp_equals_sum_of_parts() -> None:
    a = ARDPenalty(weight=0.2)
    b = OrderedBetaBernoulliPenalty(alpha=1.5, tau=0.8)
    comp = a + b
    t = torch.randn(5, 3, dtype=torch.float64) * 0.3
    v = torch.randn_like(t)
    hv_sum = a.hvp(t, v) + b.hvp(t, v)
    hv_comp = comp.hvp(t, v)
    assert torch.allclose(hv_comp, hv_sum, atol=1e-12)
