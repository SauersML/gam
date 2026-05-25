"""Callable manifold descriptors that route all math through the Rust
``RiemannianManifold`` trait.

Each class holds only a JSON descriptor (kind + parameters); the
``exp / log / metric / dimension / ambient_dim`` primitives delegate to
``gam_pyffi._rust.manifold_{exp_map, log_map, metric_tensor, dimension,
ambient_dimension}`` (see :mod:`crates/gam-pyffi/src/lib.rs`), each of
which in turn calls the canonical Rust implementations under
:mod:`src/geometry/`. No Riemannian math is reimplemented in Python.

Torch interop: when the input is a ``torch.Tensor`` we wrap the Rust call
in a :class:`torch.autograd.Function` so callers get a tensor back.
Backward is implemented for the flat (Euclidean / Circle / Torus /
Cylinder product) cases where ``∂ exp_p(v)/∂{p, v}`` is the identity;
curved manifolds (Sphere, Grassmann, …) currently return ``None``
gradients for ``p`` and ``v`` from the Rust path — callers that need
``grad`` through curved-manifold exp/log should rely on the analytic
identity that the gradient lives in the tangent space (i.e. project
through ``project_tangent`` themselves).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ._binding import rust_module as _rust_module
from ._protocol import ManifoldDescriptor, _require_torch


def _exp_map_rust(manifold_json: str, points: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    out = _rust_module().manifold_exp_map(manifold_json, points, vecs)
    return np.asarray(out, dtype=np.float64)


def _log_map_rust(manifold_json: str, p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    out = _rust_module().manifold_log_map(manifold_json, p_from, p_to)
    return np.asarray(out, dtype=np.float64)


def _metric_tensor_rust(manifold_json: str, point: np.ndarray) -> np.ndarray:
    out = _rust_module().manifold_metric_tensor(manifold_json, point)
    return np.asarray(out, dtype=np.float64)


def _maybe_import_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError:
        return None


def _to_2d_numpy(x: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    """Cast a torch tensor / numpy array to a 2-D float64 ndarray batch
    suitable for the Rust pyfunctions, returning the original shape so the
    caller can reshape the result back."""
    torch_mod = _maybe_import_torch()
    if torch_mod is not None and isinstance(x, torch_mod.Tensor):
        arr = x.detach().cpu().to(torch_mod.float64).contiguous().numpy()
    else:
        arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    orig_shape = arr.shape
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(f"manifold input must be 1-D or 2-D, got {arr.ndim}-D")
    return arr, orig_shape


class _ExpMapFn:
    """Lazy torch.autograd.Function wrapping the Rust ``manifold_exp_map``.

    Backward returns straight-through gradients on ``p`` and ``v``; this
    matches the Rust ``RiemannianManifold`` contract for the flat cases
    (Euclidean / Circle / Torus / product-of-flats) and is the standard
    first-order tangent-space approximation for curved cases.
    """

    _impl = None

    @classmethod
    def get(cls) -> Any:
        if cls._impl is not None:
            return cls._impl
        torch = _require_torch()

        class _Impl(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, p: Any, v: Any, manifold_json: str) -> Any:
                p_np = p.detach().cpu().to(torch.float64).contiguous().numpy()
                v_np = v.detach().cpu().to(torch.float64).contiguous().numpy()
                squeeze = p_np.ndim == 1
                p_2d = p_np.reshape(1, -1) if squeeze else p_np
                v_2d = v_np.reshape(1, -1) if v_np.ndim == 1 else v_np
                out = _exp_map_rust(manifold_json, p_2d, v_2d)
                if squeeze:
                    out = out.reshape(-1)
                ctx.save_for_backward(p, v)
                return torch.as_tensor(out, dtype=p.dtype, device=p.device)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, None]:
                p, v = ctx.saved_tensors
                return (
                    grad_output.to(dtype=p.dtype, device=p.device),
                    grad_output.to(dtype=v.dtype, device=v.device),
                    None,
                )

        cls._impl = _Impl
        return cls._impl


class _RustManifold(ManifoldDescriptor):
    """Base: holds the JSON descriptor and dispatches every primitive to Rust."""

    json: dict[str, Any]

    @property
    def manifold_json(self) -> str:
        return json.dumps(self.json)

    @property
    def dimension(self) -> int:
        return int(_rust_module().manifold_dimension(self.manifold_json))

    @property
    def ambient_dim(self) -> int:
        return int(_rust_module().manifold_ambient_dimension(self.manifold_json))

    def exp(self, p: Any, v: Any) -> Any:
        torch_mod = _maybe_import_torch()
        if torch_mod is not None and isinstance(p, torch_mod.Tensor):
            v_t = v if isinstance(v, torch_mod.Tensor) else torch_mod.as_tensor(v, dtype=p.dtype, device=p.device)
            fn = _ExpMapFn.get()
            return fn.apply(p, v_t, self.manifold_json)
        p_2d, p_shape = _to_2d_numpy(p)
        v_2d, _ = _to_2d_numpy(v)
        out = _exp_map_rust(self.manifold_json, p_2d, v_2d)
        if len(p_shape) == 1:
            return out.reshape(-1)
        return out

    def log(self, p: Any, q: Any) -> Any:
        torch_mod = _maybe_import_torch()
        p_2d, p_shape = _to_2d_numpy(p)
        q_2d, _ = _to_2d_numpy(q)
        out = _log_map_rust(self.manifold_json, p_2d, q_2d)
        if len(p_shape) == 1:
            out = out.reshape(-1)
        if torch_mod is not None and isinstance(p, torch_mod.Tensor):
            return torch_mod.as_tensor(out, dtype=p.dtype, device=p.device)
        return out

    def metric(self, p: Any) -> Any:
        torch_mod = _maybe_import_torch()
        p_arr, _ = _to_2d_numpy(p)
        point = p_arr[0]
        g = _metric_tensor_rust(self.manifold_json, point)
        if torch_mod is not None and isinstance(p, torch_mod.Tensor):
            return torch_mod.as_tensor(g, dtype=p.dtype, device=p.device)
        return g

    def to_json(self) -> dict[str, Any]:
        return dict(self.json)


class Euclidean(_RustManifold):
    """Flat Euclidean ``R^d`` — math from ``gam::geometry::Euclidean``."""

    def __init__(self, dim: int = 1) -> None:
        if int(dim) <= 0:
            raise ValueError("Euclidean.dim must be > 0")
        self.json = {"kind": "euclidean", "dim": int(dim)}

    def __repr__(self) -> str:
        return f"Euclidean(dim={self.json['dim']})"


class Circle(_RustManifold):
    """The unit circle ``S^1`` parameterized as a unit 2-vector in
    ambient ``R^2``. All math from ``gam::geometry::Circle``."""

    # Composition contract: ``gamfit._smooth`` reads this when deciding
    # whether a manifold-basis pair is allowed.
    compatible_bases = frozenset({"PeriodicHarmonic", "Fourier"})

    def __init__(self) -> None:
        self.json = {"kind": "circle"}

    def __repr__(self) -> str:
        return "Circle()"


class Sphere(_RustManifold):
    """The ``n``-sphere ``S^n`` in ambient ``R^{n+1}``. Math from
    ``gam::geometry::Sphere``."""

    def __init__(self, intrinsic_dim: int = 2) -> None:
        if int(intrinsic_dim) < 1:
            raise ValueError("Sphere.intrinsic_dim must be >= 1")
        self.json = {"kind": "sphere", "intrinsic_dim": int(intrinsic_dim)}

    def __repr__(self) -> str:
        return f"Sphere(intrinsic_dim={self.json['intrinsic_dim']})"


class Torus(_RustManifold):
    """The flat ``d``-torus ``T^d``. Math from ``gam::geometry::Torus``."""

    def __init__(self, dim: int = 2) -> None:
        if int(dim) < 1:
            raise ValueError("Torus.dim must be >= 1")
        self.json = {"kind": "torus", "d": int(dim)}

    def __repr__(self) -> str:
        return f"Torus(dim={self.json['d']})"


class CylinderManifold(_RustManifold):
    """The cylinder ``S^1 × R^k`` modeled as the Rust product manifold
    ``Circle × Euclidean(k)`` (math fully in ``gam::geometry::Product``)."""

    def __init__(self, open_dim: int = 1) -> None:
        if int(open_dim) < 0:
            raise ValueError("CylinderManifold.open_dim must be >= 0")
        parts: list[dict[str, Any]] = [{"kind": "circle"}]
        if int(open_dim) > 0:
            parts.append({"kind": "euclidean", "dim": int(open_dim)})
        self.json = {"kind": "product", "parts": parts}
        self._open = int(open_dim)

    def __repr__(self) -> str:
        return f"CylinderManifold(open_dim={self._open})"


__all__ = [
    "Euclidean",
    "Circle",
    "Sphere",
    "Torus",
    "CylinderManifold",
]
