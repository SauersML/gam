"""Uniform callable-basis protocol for gamfit descriptors.

Every basis descriptor in the public surface — :class:`gamfit.Duchon`,
:class:`gamfit.BSpline`, :class:`gamfit.Matern`, :class:`gamfit.Pca`,
:class:`gamfit.TensorBSpline`, :class:`gamfit.PeriodicSplineCurve` — gains
three methods with a single uniform contract:

* ``evaluate(*coords) -> Tensor`` of shape ``(B, M)``.
* ``jacobian(*coords) -> Tensor`` of shape ``(B, M, d)``.
* ``hessian(*coords)  -> Tensor`` of shape ``(B, M, d, d)``.

``d`` is the *intrinsic* dimensionality of the descriptor's domain.
``B`` is the common length of the input coordinate tensors (one positional
argument per intrinsic axis). ``M`` is the basis-column count
(``basis_size``).

All outputs are ``torch.Tensor`` with autograd connected to every coordinate
input. ``torch.autograd.functional.jacobian`` of :meth:`evaluate` matches
:meth:`jacobian` to numerical eps, and similarly for the hessian.

Torch is an *optional* dependency for gamfit. Constructing a descriptor and
inspecting its dataclass fields never imports torch; only the three
evaluator methods do, and they raise a clear :class:`ImportError` when torch
is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch as _torch_t

    TensorLike = _torch_t.Tensor


_TORCH_MODULE: Any = None


def _torch() -> Any:
    """Lazy-import torch with a clear error when it is missing."""
    global _TORCH_MODULE
    if _TORCH_MODULE is None:
        try:
            import torch as _t
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "gamfit basis-descriptor evaluation requires torch. "
                "Install torch (e.g. `pip install torch`) to call "
                ".evaluate / .jacobian / .hessian on a descriptor."
            ) from exc
        _TORCH_MODULE = _t
    return _TORCH_MODULE


def _as_tensor(x: Any, ref: Any | None = None) -> Any:
    """Coerce a 1D numpy-or-torch input to a float64 torch tensor."""
    torch = _torch()
    if isinstance(x, torch.Tensor):
        out = x
    else:
        out = torch.as_tensor(x)
    if not torch.is_floating_point(out):
        out = out.to(dtype=torch.float64)
    return out


def _stack_coords(coords: Sequence[Any]) -> Any:
    """Stack 1D coordinates into a (B, d) tensor with shared dtype/device."""
    torch = _torch()
    if len(coords) == 0:
        raise ValueError("evaluate() requires at least one coordinate tensor")
    tensors = [_as_tensor(c) for c in coords]
    ref = tensors[0]
    aligned = []
    for t in tensors:
        if t.dim() != 1:
            raise ValueError(
                f"each coordinate must be 1D, got shape {tuple(t.shape)}"
            )
        if t.numel() != ref.numel():
            raise ValueError(
                "coordinate tensors must share length B; got "
                f"{ref.numel()} and {t.numel()}"
            )
        aligned.append(t.to(dtype=ref.dtype, device=ref.device))
    return torch.stack(aligned, dim=1)


class BasisDescriptor:
    """Mixin endowing a smooth descriptor with callable basis methods.

    Concrete descriptors implement :meth:`_evaluate_impl` (and optionally
    :meth:`intrinsic_dim` / :meth:`basis_size` when the values are not
    inferable from a forward pass). The protocol then derives
    :meth:`jacobian` and :meth:`hessian` via ``torch.autograd.functional``,
    which guarantees they match autograd of :meth:`evaluate` by
    construction.

    Concrete implementations *may* override :meth:`jacobian` /
    :meth:`hessian` to provide an analytic closed form — the contract is
    only that autograd-of-evaluate equals the override to numerical eps.
    """

    # ------------------------------------------------------------------ shape

    @property
    def intrinsic_dim(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} must implement intrinsic_dim"
        )

    @property
    def basis_size(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} must implement basis_size"
        )

    # ------------------------------------------------------------------ core

    def _evaluate_impl(self, coords: Any) -> Any:
        """Compute ``(B, M)`` basis matrix from a stacked ``(B, d)`` input.

        Concrete subclasses override this. ``coords`` is the result of
        :func:`_stack_coords` and is a leaf tensor with ``requires_grad``
        already toggled by the caller (when a derivative is needed).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _evaluate_impl"
        )

    def evaluate(self, *coords: Any) -> Any:
        """Evaluate the basis at ``coords``. Returns ``(B, M)`` torch tensor.

        Autograd flows from each coordinate input to the output. Pass
        ``requires_grad=True`` on the inputs to differentiate.
        """
        if len(coords) != self.intrinsic_dim:
            raise ValueError(
                f"{type(self).__name__}.evaluate expected "
                f"{self.intrinsic_dim} coordinate argument(s), got {len(coords)}"
            )
        stacked = _stack_coords(coords)
        return self._evaluate_impl(stacked)

    # -------------------------------------------------------------- jacobian

    def jacobian(self, *coords: Any) -> Any:
        """Per-row Jacobian ``∂Φ/∂x``; shape ``(B, M, d)``.

        Implemented via ``torch.autograd.functional.jacobian`` on a
        decoupled per-row evaluation, which is independent of batch index.
        Concrete descriptors may override with an analytic form.
        """
        torch = _torch()
        if len(coords) != self.intrinsic_dim:
            raise ValueError(
                f"{type(self).__name__}.jacobian expected "
                f"{self.intrinsic_dim} coordinate argument(s), got {len(coords)}"
            )
        stacked = _stack_coords(coords).detach().requires_grad_(True)
        # Per-row jacobian: differentiate evaluate w.r.t. coords; rows are
        # independent across B so the cross-batch Jacobian block is zero.
        out = self._evaluate_impl(stacked)
        B, M = int(out.shape[0]), int(out.shape[1])
        d = self.intrinsic_dim
        jac = torch.zeros((B, M, d), dtype=out.dtype, device=out.device)
        for m in range(M):
            grads = torch.autograd.grad(
                out[:, m].sum(),
                stacked,
                retain_graph=True,
                create_graph=False,
            )[0]
            jac[:, m, :] = grads
        return jac

    # --------------------------------------------------------------- hessian

    def hessian(self, *coords: Any) -> Any:
        """Per-row Hessian ``∂²Φ/∂x∂xᵀ``; shape ``(B, M, d, d)``.

        Implemented by differentiating :meth:`jacobian` once more via
        autograd. Concrete descriptors may override with an analytic form.
        """
        torch = _torch()
        if len(coords) != self.intrinsic_dim:
            raise ValueError(
                f"{type(self).__name__}.hessian expected "
                f"{self.intrinsic_dim} coordinate argument(s), got {len(coords)}"
            )
        stacked = _stack_coords(coords).detach().requires_grad_(True)
        out = self._evaluate_impl(stacked)
        B, M = int(out.shape[0]), int(out.shape[1])
        d = self.intrinsic_dim
        hess = torch.zeros((B, M, d, d), dtype=out.dtype, device=out.device)
        # First derivatives, with create_graph=True so we can differentiate again.
        first_grads = []
        for m in range(M):
            g = torch.autograd.grad(
                out[:, m].sum(),
                stacked,
                retain_graph=True,
                create_graph=True,
            )[0]
            first_grads.append(g)
        for m in range(M):
            for j in range(d):
                gj = first_grads[m][:, j]
                g2 = torch.autograd.grad(
                    gj.sum(),
                    stacked,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                hess[:, m, j, :] = g2
        return hess

    # -------------------------------------------------------------- protocol

    def __call__(self, *coords: Any) -> Any:
        return self.evaluate(*coords)
