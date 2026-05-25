"""Callable manifold descriptors with torch-native ``exp / log / metric / geodesic``.

Wraps the Rust manifold dataclasses (``CircleManifold``, ``SphereManifold``,
``TorusManifold``, ``EuclideanManifold``, …) — which hold only ``dim`` and
``to_json`` — with the :class:`ManifoldDescriptor` protocol so users get
fully differentiable Riemannian primitives without leaving torch.

The math here is *not* duplicated from Rust: Rust only carries manifold
descriptors as enum tags consumed by retraction logic in solver code, not as
exp/log/metric primitives. The torch routines below are the canonical
implementations, and they compose with autograd through both ``p`` and ``v``.
"""

from __future__ import annotations

import math
from typing import Any

from ._protocol import ManifoldDescriptor, _require_torch


class Euclidean(ManifoldDescriptor):
    """Flat Euclidean manifold ``R^d`` with the identity metric."""

    # Composition contract: bases ride on a Euclidean patch.
    # ``PeriodicHarmonic`` is permitted because the protocol's only torch-side
    # basis at present is the periodic harmonic family; richer Euclidean bases
    # (B-spline, RBF, ...) get added to this set as their descriptors land.
    compatible_bases = frozenset({"PeriodicHarmonic", "Fourier"})

    def __init__(self, dim: int) -> None:
        if int(dim) <= 0:
            raise ValueError("Euclidean.dim must be > 0")
        self._dim = int(dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def exp(self, p: Any, v: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        v_t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=p_t.dtype)
        return p_t + v_t

    def log(self, p: Any, q: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        q_t = q if isinstance(q, torch.Tensor) else torch.as_tensor(q, dtype=p_t.dtype)
        return q_t - p_t

    def metric(self, p: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        return torch.eye(self._dim, dtype=p_t.dtype, device=p_t.device)

    def to_json(self) -> dict[str, Any]:
        return {"kind": "euclidean", "dim": self._dim}

    def __repr__(self) -> str:
        return f"Euclidean(dim={self._dim})"


class Circle(ManifoldDescriptor):
    """The unit circle ``S^1``. Points are angles in radians; ``exp`` wraps
    modulo ``2 pi`` to stay on the canonical chart ``(-pi, pi]``."""

    # Composition contract: which BasisDescriptor classes ride on this manifold.
    # Consumed by :func:`gamfit._smooth._check_manifold_basis_compatibility`.
    compatible_bases = frozenset({"PeriodicHarmonic", "Fourier"})

    @property
    def dimension(self) -> int:
        return 1

    def _wrap(self, x: Any) -> Any:
        torch = _require_torch()
        two_pi = 2.0 * math.pi
        # Wrap to (-pi, pi]
        wrapped = torch.remainder(x + math.pi, two_pi) - math.pi
        return wrapped

    def exp(self, p: Any, v: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        v_t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=p_t.dtype)
        return self._wrap(p_t + v_t)

    def log(self, p: Any, q: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        q_t = q if isinstance(q, torch.Tensor) else torch.as_tensor(q, dtype=p_t.dtype)
        return self._wrap(q_t - p_t)

    def metric(self, p: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        return torch.ones((1, 1), dtype=p_t.dtype, device=p_t.device)

    def to_json(self) -> dict[str, Any]:
        return {"kind": "circle"}

    def __repr__(self) -> str:
        return "Circle()"


class Sphere(ManifoldDescriptor):
    """The ``n``-sphere ``S^n`` embedded in ``R^{n+1}``. Points are unit
    vectors; tangent vectors live in the orthogonal complement of ``p``.

    Default constructor takes ``intrinsic_dim`` (the ``n`` in ``S^n``).
    The ambient embedding dimension is ``intrinsic_dim + 1``.
    """

    def __init__(self, intrinsic_dim: int = 2) -> None:
        if int(intrinsic_dim) < 1:
            raise ValueError("Sphere.intrinsic_dim must be >= 1")
        self._dim = int(intrinsic_dim)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def ambient_dim(self) -> int:
        return self._dim + 1

    def _project_tangent(self, p: Any, v: Any) -> Any:
        torch = _require_torch()
        inner = (p * v).sum(dim=-1, keepdim=True)
        return v - inner * p

    def exp(self, p: Any, v: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        v_t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=p_t.dtype)
        # Project to tangent space first to be tolerant of slightly off-tangent v
        v_proj = self._project_tangent(p_t, v_t)
        norm = torch.linalg.vector_norm(v_proj, dim=-1, keepdim=True).clamp(min=1e-30)
        cos_n = torch.cos(norm)
        sin_n = torch.sin(norm)
        return cos_n * p_t + sin_n * (v_proj / norm)

    def log(self, p: Any, q: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        q_t = q if isinstance(q, torch.Tensor) else torch.as_tensor(q, dtype=p_t.dtype)
        inner = (p_t * q_t).sum(dim=-1, keepdim=True).clamp(min=-1.0 + 1e-12, max=1.0 - 1e-12)
        theta = torch.arccos(inner)
        diff = q_t - inner * p_t
        diff_norm = torch.linalg.vector_norm(diff, dim=-1, keepdim=True).clamp(min=1e-30)
        return theta * diff / diff_norm

    def metric(self, p: Any) -> Any:
        """Induced metric on the tangent space, expressed as the projector
        ``I − p p^T`` (rank ``n`` in ambient ``R^{n+1}``). This is SPD on
        the tangent space and PSD on ambient space; it is the natural matrix
        form for sphere optimization without an intrinsic chart.
        """
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        # Last dim is ambient
        d = p_t.shape[-1]
        eye = torch.eye(d, dtype=p_t.dtype, device=p_t.device)
        # Outer p p^T
        if p_t.dim() == 1:
            outer = torch.outer(p_t, p_t)
            return eye - outer
        # Broadcast batched outer
        outer = p_t.unsqueeze(-1) * p_t.unsqueeze(-2)
        return eye - outer

    def to_json(self) -> dict[str, Any]:
        return {"kind": "sphere", "intrinsic_dim": self._dim}

    def __repr__(self) -> str:
        return f"Sphere(intrinsic_dim={self._dim})"


class Torus(ManifoldDescriptor):
    """The flat ``d``-torus ``T^d = (S^1)^d``. Each axis is wrapped
    independently into ``(-pi, pi]``."""

    def __init__(self, dim: int = 2) -> None:
        if int(dim) < 1:
            raise ValueError("Torus.dim must be >= 1")
        self._dim = int(dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def _wrap(self, x: Any) -> Any:
        torch = _require_torch()
        two_pi = 2.0 * math.pi
        return torch.remainder(x + math.pi, two_pi) - math.pi

    def exp(self, p: Any, v: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        v_t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=p_t.dtype)
        return self._wrap(p_t + v_t)

    def log(self, p: Any, q: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        q_t = q if isinstance(q, torch.Tensor) else torch.as_tensor(q, dtype=p_t.dtype)
        return self._wrap(q_t - p_t)

    def metric(self, p: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        return torch.eye(self._dim, dtype=p_t.dtype, device=p_t.device)

    def to_json(self) -> dict[str, Any]:
        return {"kind": "torus", "dim": self._dim}

    def __repr__(self) -> str:
        return f"Torus(dim={self._dim})"


class CylinderManifold(ManifoldDescriptor):
    """The cylinder ``S^1 × R^k``: one periodic axis and ``k`` Euclidean
    axes. Default ``open_dim=1`` gives the standard 2-D cylinder."""

    def __init__(self, open_dim: int = 1) -> None:
        if int(open_dim) < 0:
            raise ValueError("CylinderManifold.open_dim must be >= 0")
        self._open = int(open_dim)
        self._dim = 1 + self._open

    @property
    def dimension(self) -> int:
        return self._dim

    def _wrap_periodic(self, x: Any) -> Any:
        torch = _require_torch()
        two_pi = 2.0 * math.pi
        # First column is periodic; remaining are Euclidean
        if x.dim() == 1:
            head = torch.remainder(x[0:1] + math.pi, two_pi) - math.pi
            return torch.cat([head, x[1:]], dim=0)
        head = torch.remainder(x[..., 0:1] + math.pi, two_pi) - math.pi
        tail = x[..., 1:]
        return torch.cat([head, tail], dim=-1)

    def exp(self, p: Any, v: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        v_t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=p_t.dtype)
        return self._wrap_periodic(p_t + v_t)

    def log(self, p: Any, q: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        q_t = q if isinstance(q, torch.Tensor) else torch.as_tensor(q, dtype=p_t.dtype)
        diff = q_t - p_t
        # Wrap only the periodic axis
        if diff.dim() == 1:
            head = (diff[0:1] + math.pi).remainder(2.0 * math.pi) - math.pi
            return torch.cat([head, diff[1:]], dim=0)
        head = (diff[..., 0:1] + math.pi).remainder(2.0 * math.pi) - math.pi
        tail = diff[..., 1:]
        return torch.cat([head, tail], dim=-1)

    def metric(self, p: Any) -> Any:
        torch = _require_torch()
        p_t = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.float64)
        return torch.eye(self._dim, dtype=p_t.dtype, device=p_t.device)

    def to_json(self) -> dict[str, Any]:
        return {"kind": "cylinder", "open_dim": self._open}

    def __repr__(self) -> str:
        return f"CylinderManifold(open_dim={self._open})"


__all__ = [
    "Euclidean",
    "Circle",
    "Sphere",
    "Torus",
    "CylinderManifold",
]
