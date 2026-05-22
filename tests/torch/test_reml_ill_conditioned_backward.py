"""Regression: Gaussian REML backward must NOT crash on near-singular K.

When ``λ`` saturates very large (e.g. ``1e10+``), the penalized Hessian
``K = XᵀWX + λS`` becomes effectively rank-deficient (``λS`` dominates).
The analytic VJP previously raised ``GamError: Model is ill-conditioned``
in that regime, which would crash production training at large ``F`` where
individual atoms can saturate ``λ_k`` in early batches.

The fix degrades gracefully: when ``K`` is near-singular the backward
returns zero gradients (the correct "shrink-out" limit — when ``λ`` has
saturated, the atom is effectively unused, so its contribution to the loss
is zero in the limit).

This test pins that behavior end-to-end through the torch wrapper.
"""

from __future__ import annotations

import pytest

try:
    import torch

    from gamfit.torch import gaussian_reml_fit
except ImportError:  # pragma: no cover - env-dependent
    pytest.skip("torch / gamfit unavailable", allow_module_level=True)


_F64 = torch.float64


def _saturating_problem(seed: int = 0):
    """Tiny N, comparatively large M problem with structured ``y`` so REML
    drives ``λ`` to its upper bound.

    Concretely: ``y`` is in the null space of the penalty (a constant), so
    the optimal smoother is the constant fit and REML happily sends ``λ`` to
    its cap. With ``p`` comparable to ``n``, ``K`` becomes dominated by
    ``λS`` and the backward path hits the ill-conditioned regime.
    """
    torch.manual_seed(seed)
    n, p, d = 6, 5, 1
    # Design with a constant column + smooth columns.
    x = torch.empty(n, p, dtype=_F64)
    x[:, 0] = 1.0
    t = torch.linspace(0.0, 1.0, n, dtype=_F64)
    for k in range(1, p):
        x[:, k] = torch.sin((k + 1) * t)
    # y constant => smooth contributions should shrink out (λ saturates).
    y = torch.full((n, d), 0.7, dtype=_F64)
    # Penalty: penalize everything except the intercept (full-rank on the
    # non-constant columns). With y constant, REML prefers max λ.
    s = torch.zeros(p, p, dtype=_F64)
    for k in range(1, p):
        s[k, k] = 1.0
    return x, y, s


def test_backward_does_not_raise_when_lambda_saturates():
    x, y, s = _saturating_problem(seed=0)
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    s = s.clone().requires_grad_(True)

    out = gaussian_reml_fit(x, y, s)
    # Compose a loss that touches multiple outputs so the backward must
    # populate the full ridge-profile + REML-score + edf VJP chain.
    loss = (out.coefficients ** 2).sum() + out.reml_score + out.edf
    # Must NOT raise.
    loss.backward()

    # Shapes and finiteness — the graceful-degradation path returns finite
    # values (zero on the ill-conditioned axis), never NaN/inf.
    for name, tensor in (("x", x), ("y", y), ("s", s)):
        assert tensor.grad is not None, f"{name}.grad is None"
        assert tensor.grad.shape == tensor.shape, (
            f"{name}.grad shape {tuple(tensor.grad.shape)} != input shape "
            f"{tuple(tensor.shape)}"
        )
        assert torch.isfinite(tensor.grad).all(), (
            f"{name}.grad contains non-finite values: "
            f"{tensor.grad[~torch.isfinite(tensor.grad)]}"
        )


def test_backward_well_conditioned_still_produces_nonzero_grads():
    """Sanity: the graceful-degradation branch must NOT trigger for a
    well-conditioned problem. We use a generic random fit; gradients should
    be finite and at least one of them should be nonzero (i.e. the analytic
    VJP actually ran, not the zero-grad fallback)."""
    torch.manual_seed(7)
    n, p, d = 30, 4, 2
    x = torch.randn(n, p, dtype=_F64, requires_grad=True)
    y = torch.randn(n, d, dtype=_F64, requires_grad=True)
    a = torch.randn(p, p, dtype=_F64)
    s = (a @ a.T).detach().clone().requires_grad_(True)

    out = gaussian_reml_fit(x, y, s)
    loss = (out.coefficients ** 2).sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    # At least one input grad must be nonzero — otherwise we accidentally
    # took the shrink-out path on a well-posed problem.
    assert (x.grad.abs().sum() + y.grad.abs().sum() + s.grad.abs().sum()) > 0.0
