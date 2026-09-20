"""PyTorch adapter for the Sinkhorn-barycenter kernel.

Exposes a :class:`torch.autograd.Function` whose backward pass calls
the Rust implicit-function-theorem VJP of the certified fixed point the
forward pass returns.

Importing this module raises a clear :class:`ImportError` if PyTorch
is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from . import kernels as _kernels

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gamfit.kernels_torch requires PyTorch. Install with "
        "'pip install torch'."
    ) from exc

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class _SinkhornBarycenterFn(torch.autograd.Function):
    """Differentiable Sinkhorn-barycenter ``torch.autograd.Function``.

    Forward: ``(K, M) atoms, (K,) weights -> (M,) barycenter``.
    Backward: the fixed point's implicit-function-theorem VJP from the Rust
    extension.
    """

    @staticmethod
    def forward(
        ctx: Any,
        atoms: torch.Tensor,
        weights: torch.Tensor,
        cost: torch.Tensor | ArrayLike,
        eps: float,
    ) -> torch.Tensor:
        atoms_np = atoms.detach().cpu().double().numpy()
        weights_np = weights.detach().cpu().double().numpy()
        cost_np = cost.detach().cpu().double().numpy() if isinstance(cost, torch.Tensor) else np.asarray(cost, dtype=np.float64)
        bary_np = _kernels.sinkhorn_barycenter(
            atoms_np, weights_np, cost_np, eps=float(eps)
        )
        ctx.save_for_backward(atoms.detach(), weights.detach())
        ctx._cost = cost_np
        ctx._eps = float(eps)
        ctx._device = atoms.device
        ctx._dtype = atoms.dtype
        return torch.from_numpy(bary_np).to(device=atoms.device, dtype=atoms.dtype)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        atoms, weights = ctx.saved_tensors
        atoms_np = atoms.cpu().double().numpy()
        weights_np = weights.cpu().double().numpy()
        cot_np = grad_output.detach().cpu().double().numpy()
        d_atoms_np, d_weights_np = _kernels.sinkhorn_barycenter_vjp(
            atoms_np, weights_np, ctx._cost, ctx._eps, cot_np
        )
        d_atoms = torch.from_numpy(d_atoms_np).to(device=ctx._device, dtype=ctx._dtype)
        d_weights = torch.from_numpy(d_weights_np).to(device=ctx._device, dtype=ctx._dtype)
        # cost and eps are non-differentiable.
        return d_atoms, d_weights, None, None


def sinkhorn_barycenter(
    atoms: "torch.Tensor",
    weights: Optional["torch.Tensor"] = None,
    cost: Optional["torch.Tensor"] = None,
    eps: float = 0.01,
) -> "torch.Tensor":
    """Differentiable Sinkhorn Wasserstein barycenter (PyTorch).

    Same semantics as :func:`gamfit.kernels.sinkhorn_barycenter` with
    PyTorch tensors. Backward uses the Rust implicit-function-theorem VJP
    of the certified fixed point the forward pass returns.
    """
    if not isinstance(atoms, torch.Tensor):
        raise TypeError("atoms must be a torch.Tensor")
    k, m = atoms.shape
    if weights is None:
        weights = torch.full(
            (k,), 1.0 / k, dtype=atoms.dtype, device=atoms.device
        )
    if cost is None:
        cost = torch.from_numpy(_kernels.circular_cost(m)).to(
            device=atoms.device, dtype=atoms.dtype
        )
    out: torch.Tensor = _SinkhornBarycenterFn.apply(atoms, weights, cost, eps)
    return out


__all__ = ["sinkhorn_barycenter"]
