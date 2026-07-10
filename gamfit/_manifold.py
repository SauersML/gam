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
Backward routes through the canonical Rust analytic vector–Jacobian product
``gam_pyffi._rust.manifold_exp_map_vjp`` (dispatching
``RiemannianManifold::exp_map_vjp``): exact for flat manifolds (Euclidean /
Circle / Torus / products thereof, where the VJP collapses to the identity)
*and* for curved ones (Sphere / Grassmann / Stiefel / SPD and products thereof).
No straight-through identity is applied to a curved manifold, and no Riemannian
math is recomputed on the torch side.
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


def _exp_map_vjp_rust(
    manifold_json: str,
    points: np.ndarray,
    vecs: np.ndarray,
    grad_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic vector–Jacobian product of ``exp_p(v)``, all math in Rust.

    Returns ``(grad_points, grad_vecs)`` as float64 arrays. The canonical Rust
    implementation owns every manifold-specific pullback; Python never
    substitutes a silently-wrong identity gradient on a curved manifold."""
    grad_p, grad_v = _rust_module().manifold_exp_map_vjp(
        manifold_json, points, vecs, grad_output
    )
    return np.asarray(grad_p, dtype=np.float64), np.asarray(grad_v, dtype=np.float64)


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

    Backward is the *analytic* vector–Jacobian product of ``exp_p(v)``,
    computed entirely in Rust via ``manifold_exp_map_vjp`` (which dispatches
    ``RiemannianManifold::exp_map_vjp``). This is exact for both flat manifolds
    (Euclidean / Circle / Torus / products thereof, where the VJP reduces to
    the identity) and curved ones (Sphere / Grassmann / Stiefel / SPD and
    products thereof, where the analytic VJP is genuinely non-identity). There
    is no straight-through approximation and no torch-side recompute of the
    geometry.
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
                ctx.manifold_json = manifold_json
                ctx.squeeze = squeeze
                return torch.as_tensor(out, dtype=p.dtype, device=p.device)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, None]:
                p, v = ctx.saved_tensors
                squeeze = ctx.squeeze
                p_np = p.detach().cpu().to(torch.float64).contiguous().numpy()
                v_np = v.detach().cpu().to(torch.float64).contiguous().numpy()
                g_np = grad_output.detach().cpu().to(torch.float64).contiguous().numpy()
                p_2d = p_np.reshape(1, -1) if squeeze else p_np
                v_2d = v_np.reshape(1, -1) if v_np.ndim == 1 else v_np
                g_2d = g_np.reshape(1, -1) if squeeze else g_np
                grad_p_np, grad_v_np = _exp_map_vjp_rust(
                    ctx.manifold_json, p_2d, v_2d, g_2d
                )
                if squeeze:
                    grad_p_np = grad_p_np.reshape(-1)
                    grad_v_np = grad_v_np.reshape(-1)
                return (
                    torch.as_tensor(grad_p_np, dtype=p.dtype, device=p.device),
                    torch.as_tensor(grad_v_np, dtype=v.dtype, device=v.device),
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
        """Intrinsic dimension. Routes through Rust when the extension
        exposes ``manifold_dimension``; otherwise falls back to the
        descriptor's locally cached ``_dim`` (set by each subclass at
        construction). The Rust path is the canonical source of truth."""
        fn = getattr(_rust_module(), "manifold_dimension", None)
        if fn is not None:
            return int(fn(self.manifold_json))
        return int(self._dim)

    @property
    def ambient_dim(self) -> int:
        """Ambient embedding dimension. Same Rust-first / Python-fallback
        pattern as :attr:`dimension`."""
        fn = getattr(_rust_module(), "manifold_ambient_dimension", None)
        if fn is not None:
            return int(fn(self.manifold_json))
        return int(getattr(self, "_ambient_dim", self._dim))

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
        self._dim = int(dim)
        self._ambient_dim = int(dim)

    def __repr__(self) -> str:
        return f"Euclidean(dim={self.json['dim']})"


class Circle(_RustManifold):
    """The unit circle ``S^1`` parameterized by a single angle coordinate
    ``theta`` (a 1-vector ``[theta]``), matching the Rust
    ``gam::geometry::Circle``. Points and tangents are both 1-D; ``exp`` adds
    the tangent to the angle and wraps to ``[-pi, pi)`` (the half-open interval
    produced by ``rem_euclid``: ``+pi`` maps to ``-pi``). The ambient and
    intrinsic dimensions are therefore both ``1`` — there is no ``R^2``
    unit-vector embedding. All math from ``gam::geometry::Circle``."""

    # Composition contract: ``gamfit._smooth`` reads this when deciding
    # whether a manifold-basis pair is allowed.
    compatible_bases = frozenset({"PeriodicHarmonic", "Fourier"})

    def __init__(self) -> None:
        self.json = {"kind": "circle"}
        self._dim = 1
        self._ambient_dim = 1

    def __repr__(self) -> str:
        return "Circle()"


class Sphere(_RustManifold):
    """The ``n``-sphere ``S^n`` in ambient ``R^{n+1}``. Math from
    ``gam::geometry::Sphere``."""

    def __init__(self, intrinsic_dim: int = 2) -> None:
        if int(intrinsic_dim) < 1:
            raise ValueError("Sphere.intrinsic_dim must be >= 1")
        self.json = {"kind": "sphere", "intrinsic_dim": int(intrinsic_dim)}
        self._dim = int(intrinsic_dim)
        self._ambient_dim = int(intrinsic_dim) + 1

    def __repr__(self) -> str:
        return f"Sphere(intrinsic_dim={self.json['intrinsic_dim']})"


class Torus(_RustManifold):
    """The flat ``d``-torus ``T^d``. Math from ``gam::geometry::Torus``."""

    def __init__(self, dim: int = 2) -> None:
        if int(dim) < 1:
            raise ValueError("Torus.dim must be >= 1")
        self.json = {"kind": "torus", "d": int(dim)}
        self._dim = int(dim)
        self._ambient_dim = int(dim)

    def __repr__(self) -> str:
        return f"Torus(dim={self.json['d']})"


class CylinderManifold(_RustManifold):
    """The cylinder ``S^1 × R^k`` modeled as the Rust product manifold
    ``Circle × Euclidean(k)`` (math fully in ``gam::geometry::Product``).

    A point is the concatenation ``[theta, x_1, ..., x_k]``: a single angle
    coordinate for the circle factor (1-D, matching :class:`Circle`) followed
    by the ``k`` Euclidean coordinates. Ambient and intrinsic dimensions are
    both ``1 + open_dim`` — there is no ``R^2`` embedding of the circle."""

    def __init__(self, open_dim: int = 1) -> None:
        if int(open_dim) < 0:
            raise ValueError("CylinderManifold.open_dim must be >= 0")
        parts: list[dict[str, Any]] = [{"kind": "circle"}]
        if int(open_dim) > 0:
            parts.append({"kind": "euclidean", "dim": int(open_dim)})
        self.json = {"kind": "product", "parts": parts}
        self._open = int(open_dim)
        self._dim = 1 + int(open_dim)
        # Circle angle (1) + Euclidean(open_dim); matches Rust product ambient.
        self._ambient_dim = 1 + int(open_dim)

    def __repr__(self) -> str:
        return f"CylinderManifold(open_dim={self._open})"


class Grassmann(_RustManifold):
    """The Grassmann manifold ``Gr(k, n)`` of ``k``-dimensional linear
    subspaces of ``R^n``, represented by an ``n × k`` matrix with orthonormal
    columns (a point on the manifold is the *column span*, so any two such
    matrices related by a ``k × k`` orthogonal rotation denote the same point).

    GEOMETRY PRIMITIVES ONLY. This descriptor exposes the intrinsic Riemannian
    ``exp / log / metric / dimension / ambient_dim`` of ``gam::geometry::
    GrassmannManifold`` (routed through the generic ``manifold_*`` FFI, which
    parses the ``{"kind": "grassmann", ...}`` descriptor). It is *not* yet a
    fittable latent smooth — wiring Grassmann into the penalized
    ``Smooth(latent=..., basis=..., penalty=...)`` solver requires new Rust
    solver work and is out of scope here.

    A point / tangent is supplied to ``exp / log`` as a **flattened** length-
    ``n*k`` vector (the ``n × k`` matrix in row-major order), matching the
    Rust ``ambient_dim = n*k`` contract. The intrinsic dimension is
    ``k * (n - k)``.
    """

    def __init__(self, k: int = 1, n: int = 2) -> None:
        k_i, n_i = int(k), int(n)
        if k_i < 1 or n_i < 1 or k_i > n_i:
            raise ValueError(
                "Grassmann requires 1 <= k <= n (Gr(k, n) is the set of "
                f"k-dimensional subspaces of R^n); got k={k_i}, n={n_i}"
            )
        self.json = {"kind": "grassmann", "k": k_i, "n": n_i}
        self._k = k_i
        self._n = n_i
        self._dim = k_i * (n_i - k_i)
        self._ambient_dim = n_i * k_i

    def __repr__(self) -> str:
        return f"Grassmann(k={self._k}, n={self._n})"


class Stiefel(_RustManifold):
    """The (compact) Stiefel manifold ``St(n, k)`` of orthonormal ``k``-frames
    in ``R^n``, represented by an ``n × k`` matrix with orthonormal columns
    (here the matrix *itself* is the point — unlike Grassmann, frames differing
    by a column rotation are distinct).

    GEOMETRY PRIMITIVES ONLY. Exposes the intrinsic Riemannian
    ``exp / log / metric / dimension / ambient_dim`` of ``gam::geometry::
    StiefelManifold`` via the generic ``manifold_*`` FFI
    (``{"kind": "stiefel", ...}``). Not yet a fittable latent smooth; the
    penalized-smooth path needs new Rust solver work and is out of scope.

    A point / tangent is supplied to ``exp / log`` as a **flattened** length-
    ``n*k`` vector (the ``n × k`` matrix in row-major order), matching the Rust
    ``ambient_dim = n*k`` contract. The intrinsic dimension is
    ``n*k - k*(k+1)/2``.
    """

    def __init__(self, k: int = 1, n: int = 2) -> None:
        k_i, n_i = int(k), int(n)
        if k_i < 1 or n_i < 1 or k_i > n_i:
            raise ValueError(
                "Stiefel requires 1 <= k <= n (St(n, k) is the set of "
                f"orthonormal k-frames in R^n); got k={k_i}, n={n_i}"
            )
        self.json = {"kind": "stiefel", "k": k_i, "n": n_i}
        self._k = k_i
        self._n = n_i
        self._dim = n_i * k_i - k_i * (k_i + 1) // 2
        self._ambient_dim = n_i * k_i

    def __repr__(self) -> str:
        return f"Stiefel(k={self._k}, n={self._n})"


class Spd(_RustManifold):
    """The manifold of ``n × n`` symmetric positive-definite matrices under the
    affine-invariant Riemannian metric.

    GEOMETRY PRIMITIVES ONLY. Exposes the intrinsic Riemannian
    ``exp / log / metric / dimension / ambient_dim`` of ``gam::geometry::
    SpdManifold`` via the generic ``manifold_*`` FFI (``{"kind": "spd", ...}``).
    Not yet a fittable latent smooth; the penalized-smooth path needs new Rust
    solver work and is out of scope.

    A point / tangent is supplied to ``exp / log`` as a **flattened** length-
    ``n*n`` vector (the ``n × n`` matrix in row-major order; points must be SPD,
    tangents symmetric), matching the Rust ``ambient_dim = n*n`` contract. The
    intrinsic dimension is ``n*(n+1)/2``.
    """

    def __init__(self, n: int = 2) -> None:
        n_i = int(n)
        if n_i < 1:
            raise ValueError(f"Spd requires n >= 1; got n={n_i}")
        self.json = {"kind": "spd", "n": n_i}
        self._n = n_i
        self._dim = n_i * (n_i + 1) // 2
        self._ambient_dim = n_i * n_i

    def __repr__(self) -> str:
        return f"Spd(n={self._n})"


class Poincare(ManifoldDescriptor):
    """The Poincaré-ball model of hyperbolic space ``H^d`` (the open unit ball
    in ``R^d`` with constant negative curvature ``c < 0``).

    GEOMETRY PRIMITIVES ONLY. This descriptor exposes numpy-side hyperbolic
    primitives — ``exp / log / distance / metric / project`` — built directly on
    the registered ``poincare_*`` FFI (``gam::geometry::poincare``), so non-torch
    users can reach hyperbolic geometry without going through
    :class:`gamfit.torch.hyperbolic.PoincareAtoms`. It is *not* yet a fittable
    latent smooth: the penalized ``Smooth(...)`` solver path for hyperbolic
    latents needs new Rust solver work and is out of scope here.

    Unlike the other descriptors in this module, the Poincaré model is *not*
    routed through the generic JSON ``manifold_*`` FFI (which has no
    ``"poincare"`` kind); every primitive delegates to the canonical
    ``poincare_*`` Rust functions. The general (non-origin) exponential and
    logarithm are composed *exactly* from the registered origin primitives via
    Möbius addition, using the gyrovector identities (Ganea et al. 2018, with
    their ``c > 0`` mapped to our ``c < 0``)::

        lambda_p = 2 / (1 + c |p|^2)                       # conformal factor
        exp_p(v) = p (+) exp_origin( (lambda_p / 2) v )
        log_p(q) = (2 / lambda_p) log_origin( (-p) (+) q )

    where ``(+)`` is :func:`poincare_mobius_add`. Both reduce to the registered
    origin maps at ``p = 0`` (``lambda_0 = 2``). The Riemannian metric is
    conformal to the Euclidean one, ``g_p = lambda_p^2 I``.

    Points and tangents are length-``d`` vectors (a single point) or
    ``(N, d)`` arrays (a batch, one point per row). Ambient and intrinsic
    dimension are both ``d``.

    Parameters
    ----------
    dim:
        Ball dimension ``d`` (>= 1).
    curvature:
        Sectional curvature ``c``; must be strictly negative (default
        ``-1.0``, matching :class:`gamfit.torch.hyperbolic.PoincareAtoms`).
    """

    def __init__(self, dim: int = 2, curvature: float = -1.0) -> None:
        d = int(dim)
        if d < 1:
            raise ValueError(f"Poincare.dim must be >= 1; got {d}")
        c = float(curvature)
        if not np.isfinite(c) or c >= 0.0:
            raise ValueError(
                "Poincare requires strictly negative curvature (c < 0); "
                f"got curvature={curvature!r}"
            )
        self._dim = d
        self._ambient_dim = d
        self.curvature = c

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def ambient_dim(self) -> int:
        return self._ambient_dim

    def __repr__(self) -> str:
        return f"Poincare(dim={self._dim}, curvature={self.curvature})"

    # -- batched ManifoldDescriptor surface (one Rust call per operation) --
    def exp(self, p: Any, v: Any) -> Any:
        torch_mod = _maybe_import_torch()
        p_2d, p_shape = _to_2d_numpy(p)
        v_2d, _ = _to_2d_numpy(v)
        out = np.asarray(
            _rust_module().poincare_exp_map_batch(
                p_2d, v_2d, self._dim, self.curvature
            ),
            dtype=np.float64,
        )
        if len(p_shape) == 1:
            out = out.reshape(-1)
        if torch_mod is not None and isinstance(p, torch_mod.Tensor):
            return torch_mod.as_tensor(out, dtype=p.dtype, device=p.device)
        return out

    def log(self, p: Any, q: Any) -> Any:
        torch_mod = _maybe_import_torch()
        p_2d, p_shape = _to_2d_numpy(p)
        q_2d, _ = _to_2d_numpy(q)
        out = np.asarray(
            _rust_module().poincare_log_map_batch(
                p_2d, q_2d, self._dim, self.curvature
            ),
            dtype=np.float64,
        )
        if len(p_shape) == 1:
            out = out.reshape(-1)
        if torch_mod is not None and isinstance(p, torch_mod.Tensor):
            return torch_mod.as_tensor(out, dtype=p.dtype, device=p.device)
        return out

    def distance(self, a: Any, b: Any) -> Any:
        """Geodesic (hyperbolic) distance ``d_c(a, b)`` via
        :func:`poincare_distance`. Scalar for two single points, else a length-
        ``N`` array of per-row distances."""
        torch_mod = _maybe_import_torch()
        a_2d, a_shape = _to_2d_numpy(a)
        b_2d, _ = _to_2d_numpy(b)
        dists = np.asarray(
            _rust_module().poincare_distance_batch(
                a_2d, b_2d, self._dim, self.curvature
            ),
            dtype=np.float64,
        )
        if len(a_shape) == 1:
            scalar = float(dists[0])
            if torch_mod is not None and isinstance(a, torch_mod.Tensor):
                return torch_mod.as_tensor(scalar, dtype=a.dtype, device=a.device)
            return scalar
        if torch_mod is not None and isinstance(a, torch_mod.Tensor):
            return torch_mod.as_tensor(dists, dtype=a.dtype, device=a.device)
        return dists

    def metric(self, p: Any) -> Any:
        """Conformal Poincaré metric tensor from the batched Rust core.

        A single point returns ``(d, d)``; an ``(N, d)`` batch returns
        ``(N, d, d)``.
        """
        torch_mod = _maybe_import_torch()
        p_2d, p_shape = _to_2d_numpy(p)
        g = np.asarray(
            _rust_module().poincare_metric_tensor_batch(
                p_2d, self._dim, self.curvature
            ),
            dtype=np.float64,
        )
        if len(p_shape) == 1:
            g = g[0]
        if torch_mod is not None and isinstance(p, torch_mod.Tensor):
            return torch_mod.as_tensor(g, dtype=p.dtype, device=p.device)
        return g

    def project(self, p: Any) -> Any:
        """Clamp ``p`` strictly inside the ball via
        :func:`poincare_project_into_ball` (a no-op for already-interior
        points). Accepts a single point or an ``(N, d)`` batch."""
        torch_mod = _maybe_import_torch()
        p_2d, p_shape = _to_2d_numpy(p)
        out = np.asarray(
            _rust_module().poincare_project_into_ball_batch(
                p_2d, self._dim, self.curvature
            ),
            dtype=np.float64,
        )
        if len(p_shape) == 1:
            out = out.reshape(-1)
        if torch_mod is not None and isinstance(p, torch_mod.Tensor):
            return torch_mod.as_tensor(out, dtype=p.dtype, device=p.device)
        return out


__all__ = [
    "Euclidean",
    "Circle",
    "Sphere",
    "Torus",
    "CylinderManifold",
    "Grassmann",
    "Stiefel",
    "Spd",
    "Poincare",
]
