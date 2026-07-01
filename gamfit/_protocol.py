"""Uniform callable-descriptor protocols for gamfit.

Three sibling abstract base classes define the contract every descriptor must
honor so that "if X describes a thing, X.evaluate(...) and X(...) returns the
thing as a torch.Tensor with grad". No parallel `the same primitive in torch`
implementation is ever needed.

* :class:`BasisDescriptor` — anything that maps an evaluation site ``t`` to a
  design matrix ``Phi(t)``: B-splines, Duchon, Matern, Fourier, PCA, tensor
  products, etc.
* :class:`ManifoldDescriptor` — anything that exposes intrinsic Riemannian
  geometry: ``exp``, ``log``, ``metric``, ``geodesic``, ``dimension``.
* :class:`PenaltyDescriptor` — anything that contributes a scalar penalty
  on a target tensor ``t``: smoothness, ARD, isometry, sparsity, etc.
  Every concrete subclass must expose ``value / value_grad / hvp /
  hessian_diag``; the Rust trait in ``src/terms/analytic_penalties.rs``
  is the single source of truth for the math and is reached through the
  ``analytic_penalty_value_grad`` / ``analytic_penalty_hvp`` pyfunctions.

These protocols are intentionally lightweight ABCs rather than typing
``Protocol`` classes so descriptor subclasses pick up a working
``__call__`` for free (forwarding to ``evaluate``).

Torch is an optional extra: importing this module never imports torch. The
``evaluate`` / ``value`` / ``hvp`` methods that need torch defer the import
to first call and raise a clean ``ImportError`` if torch is missing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as err:
        raise ImportError(
            "this gamfit descriptor method requires torch; install with "
            "`pip install torch`"
        ) from err
    return torch


class BasisDescriptor(ABC):
    """A callable basis: ``descriptor(t)`` returns ``Phi(t)`` as a torch tensor."""

    @abstractmethod
    def evaluate(self, t: Any) -> Any:
        """Return the design matrix ``Phi(t)`` as a torch tensor with grad
        through ``t``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, K)`` for 1-D inputs, ``(N, K)`` flattened for
            multidim inputs. The exact tensor shape is descriptor-specific.
        """

    def __call__(self, t: Any) -> Any:
        return self.evaluate(t)

    @property
    def output_dim(self) -> int | None:
        """Number of basis columns ``K``. ``None`` if not statically known."""
        return None


class ManifoldDescriptor(ABC):
    """A callable manifold: ``M(p, v)`` is shorthand for ``M.exp(p, v)``.

    Subclasses implement the five core primitives. Default implementations
    route through torch ops; specializations may override for closed-form
    paths.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Intrinsic dimension of the manifold."""

    @abstractmethod
    def exp(self, p: Any, v: Any) -> Any:
        """Riemannian exponential map: return the endpoint of the geodesic
        starting at ``p`` with initial velocity ``v``."""

    @abstractmethod
    def log(self, p: Any, q: Any) -> Any:
        """Riemannian log: return the initial velocity ``v`` such that
        ``exp(p, v) == q``."""

    @abstractmethod
    def metric(self, p: Any) -> Any:
        """Riemannian metric tensor at ``p`` as a ``(d, d)`` SPD matrix."""

    def geodesic(self, p: Any, q: Any, t: Any) -> Any:
        """Point on the geodesic from ``p`` to ``q`` at parameter
        ``t ∈ [0, 1]``. Default routes through ``exp ∘ (t * log)``."""
        torch = _require_torch()
        v = self.log(p, q)
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=v.dtype if hasattr(v, "dtype") else torch.float64)
        return self.exp(p, t * v)

    def __call__(self, p: Any, v: Any) -> Any:
        return self.exp(p, v)


class PenaltyDescriptor(ABC):
    """A callable penalty: ``P(t)`` is shorthand for ``P.value(t)``.

    Subclasses route ``value / value_grad / hvp / hessian_diag`` through the
    Rust analytic-penalty registry. Composition via ``p1 + p2`` returns a
    :class:`CompositePenalty` summing both penalties' contributions.
    """

    @abstractmethod
    def value(self, t: Any) -> Any:
        """Scalar torch tensor with grad through ``t``."""

    def __call__(self, t: Any) -> Any:
        return self.value(t)

    def value_grad(self, t: Any) -> tuple[Any, Any]:
        """Return ``(value, dvalue/dt)`` as torch tensors with grad through ``t``.

        Default uses ``torch.autograd.grad`` on the result of :meth:`value`.
        Subclasses may override with analytic Rust paths.
        """
        torch = _require_torch()
        t_t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=torch.float64)
        if not t_t.requires_grad:
            t_t = t_t.detach().clone().requires_grad_(True)
        v = self.value(t_t)
        (g,) = torch.autograd.grad(v, t_t, create_graph=t_t.requires_grad)
        return v, g

    def grad(self, t: Any) -> Any:
        """First-order gradient ``dP/dt`` at ``t``. Same shape as ``t``.

        Convenience accessor: returns the second element of
        :meth:`value_grad`. Subclasses with a cheap gradient-only path
        may override.
        """
        _value, gradient = self.value_grad(t)
        return gradient

    @abstractmethod
    def hvp(self, t: Any, v: Any) -> Any:
        """Hessian-vector product ``H · v`` at ``t``. Same shape as ``t``."""

    def hessian_diag(self, t: Any) -> Any:
        """Diagonal of the Hessian at ``t``. Default extracts via
        ``hvp(t, e_i)`` per coordinate. Subclasses with cheap diagonals
        should override."""
        torch = _require_torch()
        t_t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=torch.float64)
        flat = t_t.reshape(-1)
        n = flat.numel()
        diag = torch.zeros(n, dtype=flat.dtype, device=flat.device)
        for i in range(n):
            e = torch.zeros(n, dtype=flat.dtype, device=flat.device)
            e[i] = 1.0
            hv = self.hvp(t_t, e.reshape_as(t_t)).reshape(-1)
            diag[i] = hv[i]
        return diag.reshape_as(t_t)

    def __add__(self, other: "PenaltyDescriptor") -> "PenaltyDescriptor":
        from ._composite_penalty import CompositePenalty
        if not isinstance(other, PenaltyDescriptor):
            return NotImplemented
        return CompositePenalty(self, other)

    def __radd__(self, other: "PenaltyDescriptor") -> "PenaltyDescriptor":
        if other == 0:
            return self
        return self.__add__(other)


__all__ = [
    "BasisDescriptor",
    "ManifoldDescriptor",
    "PenaltyDescriptor",
]
