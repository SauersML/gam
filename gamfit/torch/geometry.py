"""Torch implementations of response-geometry transforms.

Each public function accepts torch tensors and returns torch tensors, keeping
closure, log-ratio, simplex maps, and sphere maps differentiable and
device-local. NumPy callers use :mod:`gamfit._response_geometry` directly.
"""

from __future__ import annotations

from typing import Any

import torch

from .._binding import rust_module
from . import _torch_compat as _tc
from ._coerce import from_numpy_like, to_numpy_f64


def _matrix(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.dim() != 2:
        raise ValueError(f"{label} must be a 2-D tensor")
    if value.shape[0] == 0 or value.shape[1] < 2:
        raise ValueError(f"{label} must have at least one row and at least two columns")
    if not _tc.is_floating_point(value):
        value = value.to(dtype=_tc.float64)
    if bool((~_tc.isfinite(value)).any()):
        raise ValueError(f"{label} must contain only finite values")
    return value


def _closure_tensor(values: torch.Tensor, *, label: str = "simplex values") -> torch.Tensor:
    tensor = _matrix(values, label=label)
    if bool((tensor < 0.0).any()):
        raise ValueError("simplex values must be non-negative")
    totals = tensor.sum(dim=1, keepdim=True)
    if bool((totals <= 0.0).any()):
        raise ValueError("simplex rows must have positive total mass")
    return tensor / totals


def _keep_without_reference(d: int, reference: int) -> list[int]:
    ref = reference % d
    return [j for j in range(d) if j != ref]


def _normalized_weights_tensor(
    n: int, weights: torch.Tensor | None, ref: torch.Tensor
) -> torch.Tensor:
    if weights is None:
        return _tc.full((n,), 1.0 / n, dtype=ref.dtype, device=ref.device)
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor or None")
    w = weights.to(device=ref.device, dtype=ref.dtype).reshape(-1)
    if w.shape[0] != n:
        raise ValueError("weights length must match the number of rows")
    if bool((~_tc.isfinite(w)).any()) or bool((w < 0.0).any()) or bool(w.sum() <= 0.0):
        raise ValueError("weights must be finite, non-negative, and have positive total")
    return w / w.sum()


def _normalize_sphere_tensor(values: torch.Tensor) -> torch.Tensor:
    tensor = _matrix(values, label="spherical values")
    norms = torch.linalg.norm(tensor, dim=1, keepdim=True)
    if bool((norms <= 0.0).any()):
        raise ValueError("spherical rows must have non-zero norm")
    return tensor / norms


def closure(values: torch.Tensor) -> torch.Tensor:
    """Normalize rows onto the probability simplex."""
    return _closure_tensor(values)


def clr(values: torch.Tensor) -> torch.Tensor:
    """Centered log-ratio coordinates for positive compositions."""
    comp = _closure_tensor(values)
    if bool((comp <= 0.0).any()):
        raise ValueError("CLR coordinates require strictly positive simplex values")
    logs = comp.log()
    return logs - logs.mean(dim=1, keepdim=True)


def _ilr_basis(d: int, *, dtype: Any, device: Any) -> torch.Tensor:
    """Helmert orthonormal contrast basis ``V`` of shape ``(d, d-1)``.

    Columns are an orthonormal basis of the sum-zero hyperplane (the CLR
    subspace). ILR coordinates are ``ilr(x) = clr(x) @ V`` and the inverse is
    ``clr = ilr @ Vᵀ``. Because ``V`` is orthonormal, the Euclidean metric on the
    ``(d-1)``-dim ILR coordinates is exactly the Aitchison metric pulled back from
    the simplex: ``‖ilr(x) − ilr(y)‖₂`` equals the Aitchison distance. Ordinary
    Gaussian/Euclidean fitting in ILR coordinates is therefore isometric to
    Aitchison geometry with no extra metric weighting.
    """
    if d < 2:
        raise ValueError("ILR basis requires at least two parts")
    v = _tc.zeros((d, d - 1), dtype=dtype, device=device)
    for i in range(1, d):
        scale = (i / (i + 1.0)) ** 0.5
        for k in range(i):
            v[k, i - 1] = scale / i
        v[i, i - 1] = -scale
    return v


def ilr(values: torch.Tensor, *, reference: int = -1) -> torch.Tensor:
    """Isometric log-ratio coordinates for positive compositions.

    ILR maps a ``d``-part composition to ``d-1`` Euclidean coordinates that are
    isometric to Aitchison geometry: Euclidean distance in ILR space equals
    Aitchison distance on the simplex. The ``reference`` argument is accepted for
    a uniform call signature with :func:`alr` but is unused — the Helmert basis
    is canonical and reference-free.
    """
    del reference  # Helmert ILR basis is reference-free; kept for signature parity.
    coords = clr(values)
    basis = _ilr_basis(coords.shape[1], dtype=coords.dtype, device=coords.device)
    return coords @ basis


def inverse_ilr(coords: torch.Tensor, *, reference: int = -1) -> torch.Tensor:
    """Map ILR coordinates back to the simplex."""
    del reference  # Helmert ILR basis is reference-free; kept for signature parity.
    if not isinstance(coords, torch.Tensor):
        raise TypeError("coords must be a torch.Tensor")
    if coords.dim() != 2:
        raise ValueError("ILR coordinates must be a 2-D tensor")
    if not _tc.is_floating_point(coords):
        coords = coords.to(dtype=_tc.float64)
    d = coords.shape[1] + 1
    basis = _ilr_basis(d, dtype=coords.dtype, device=coords.device)
    clr_coords = coords @ basis.transpose(0, 1)
    log_parts = clr_coords - clr_coords.max(dim=1, keepdim=True).values
    parts = log_parts.exp()
    return parts / parts.sum(dim=1, keepdim=True)


def aitchison_metric(d: int, *, dtype: Any = None, device: Any = None) -> torch.Tensor:
    """Aitchison Gram matrix ``G = I_{d-1} − (1/d)·11ᵀ`` for ALR coordinates.

    ALR is a valid chart but is NOT isometric to Aitchison geometry: in ALR
    coordinates the Aitchison inner product is ``⟨u, v⟩ = uᵀ G v`` with this
    ``(d-1)×(d-1)`` Gram matrix (for ``d = 3`` it is ``[[2/3, -1/3], [-1/3,
    2/3]]`` ≠ I). Residual norms and gradients in ALR fitting must be weighted by
    ``G`` to be Aitchison-correct; prefer ILR, which makes ``G = I``.
    """
    if d < 2:
        raise ValueError("Aitchison metric requires at least two parts")
    dt = _tc.float64 if dtype is None else dtype
    eye = _tc.eye(d - 1, dtype=dt, device=device)
    ones = _tc.full((d - 1, d - 1), 1.0 / d, dtype=dt, device=device)
    return eye - ones


def alr(values: torch.Tensor, *, reference: int = -1) -> torch.Tensor:
    """Additive log-ratio coordinates for positive compositions."""
    comp = _closure_tensor(values)
    if bool((comp <= 0.0).any()):
        raise ValueError("ALR coordinates require strictly positive simplex values")
    ref = reference % comp.shape[1]
    keep = _keep_without_reference(comp.shape[1], reference)
    return (comp[:, keep] / comp[:, [ref]]).log()


def inverse_alr(coords: torch.Tensor, *, reference: int = -1) -> torch.Tensor:
    """Map ALR coordinates back to the simplex."""
    if not isinstance(coords, torch.Tensor):
        raise TypeError("coords must be a torch.Tensor")
    if coords.dim() != 2:
        raise ValueError("ALR coordinates must be a 2-D tensor")
    if not _tc.is_floating_point(coords):
        coords = coords.to(dtype=_tc.float64)
    d = coords.shape[1] + 1
    keep = _keep_without_reference(d, reference)
    log_parts = _tc.zeros((coords.shape[0], d), dtype=coords.dtype, device=coords.device)
    log_parts[:, keep] = coords
    log_parts = log_parts - log_parts.max(dim=1, keepdim=True).values
    parts = log_parts.exp()
    return parts / parts.sum(dim=1, keepdim=True)


def simplex_frechet_mean(
    values: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Intrinsic Fréchet mean under Aitchison simplex geometry."""
    comp = _closure_tensor(values)
    if bool((comp <= 0.0).any()):
        raise ValueError("simplex Fréchet mean requires strictly positive values")
    w = _normalized_weights_tensor(comp.shape[0], weights, comp)
    mean_log = (comp.log() * w[:, None]).sum(dim=0)
    mean_log = mean_log - mean_log.max()
    out = mean_log.exp()
    return out / out.sum()


def simplex_log_map(
    values: torch.Tensor,
    base: torch.Tensor,
    *,
    coordinates: str = "ilr",
    reference: int = -1,
) -> torch.Tensor:
    """Log map at an intrinsic simplex base point in ILR, CLR, or ALR coordinates.

    ``ilr`` (default) and ``clr`` are isometric to Aitchison geometry, so the
    Euclidean norm of the returned tangent equals the Aitchison geodesic
    distance from ``base`` to each row of ``values``. ``alr`` is a valid but
    NON-isometric chart: its Euclidean tangent norm is not the Aitchison
    distance (the Aitchison Gram is :func:`aitchison_metric`, not the
    identity), so weight residuals/gradients by that Gram when fitting in ALR.
    """
    comp = _closure_tensor(values)
    if not isinstance(base, torch.Tensor):
        raise TypeError("base must be a torch.Tensor")
    base_row = base.to(device=comp.device, dtype=comp.dtype).reshape(1, -1)
    base_arr = _closure_tensor(base_row)
    if comp.shape[1] != base_arr.shape[1]:
        raise ValueError("simplex values and base point have different dimensions")
    if bool((comp <= 0.0).any()) or bool((base_arr <= 0.0).any()):
        raise ValueError("simplex log map requires strictly positive values and base point")
    coord = coordinates.lower()
    if coord in {"simplex", "ilr"}:
        return ilr(comp, reference=reference) - ilr(base_arr, reference=reference)
    if coord == "clr":
        return clr(comp) - clr(base_arr)
    if coord == "alr":
        return alr(comp, reference=reference) - alr(base_arr, reference=reference)
    raise ValueError("simplex coordinates must be 'ilr', 'clr', or 'alr'")


def simplex_exp_map(
    tangent: torch.Tensor,
    base: torch.Tensor,
    *,
    coordinates: str = "ilr",
    reference: int = -1,
) -> torch.Tensor:
    """Exponential map from simplex tangent coordinates back to compositions.

    The default ``ilr`` chart (and ``clr``) is isometric to Aitchison geometry;
    ``alr`` is the non-isometric chart. Must match the ``coordinates`` used by
    :func:`simplex_log_map`. ILR/ALR tangents have ``D-1`` columns, CLR has ``D``.
    """
    if not isinstance(tangent, torch.Tensor):
        raise TypeError("tangent must be a torch.Tensor")
    z = tangent if _tc.is_floating_point(tangent) else tangent.to(dtype=_tc.float64)
    if z.dim() == 1:
        z = z.reshape(1, -1)
    if not isinstance(base, torch.Tensor):
        raise TypeError("base must be a torch.Tensor")
    base_arr = _closure_tensor(base.to(device=z.device, dtype=z.dtype).reshape(1, -1))
    coord = coordinates.lower()
    if coord in {"simplex", "ilr"}:
        if z.shape[1] != base_arr.shape[1] - 1:
            raise ValueError("ILR tangent dimension must be simplex dimension minus one")
        base_ilr = ilr(base_arr, reference=reference)
        return inverse_ilr(base_ilr + z, reference=reference)
    if coord == "clr":
        if z.shape[1] != base_arr.shape[1]:
            raise ValueError("CLR tangent dimension must equal simplex dimension")
        log_parts = base_arr.log() + z
        log_parts = log_parts - log_parts.max(dim=1, keepdim=True).values
        parts = log_parts.exp()
        return parts / parts.sum(dim=1, keepdim=True)
    if coord == "alr":
        if z.shape[1] != base_arr.shape[1] - 1:
            raise ValueError("ALR tangent dimension must be simplex dimension minus one")
        base_alr = alr(base_arr, reference=reference)
        return inverse_alr(base_alr + z, reference=reference)
    raise ValueError("simplex coordinates must be 'ilr', 'clr', or 'alr'")


_SPHERE_ANTIPODAL_TOL = 1e-12


def sphere_frechet_mean(
    values: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 256,
) -> torch.Tensor:
    """Intrinsic Fréchet/Karcher mean on the unit sphere.

    If the minimizer is not unique, as for an exactly antipodal pair, this
    returns one deterministic minimizer rather than an endpoint surrogate.
    """
    y = _normalize_sphere_tensor(values)
    w = _normalized_weights_tensor(y.shape[0], weights, y)
    out = rust_module().sphere_frechet_mean(
        to_numpy_f64(y),
        to_numpy_f64(w),
        float(tol),
        int(max_iter),
    )
    return from_numpy_like(out, y)


def sphere_log_map(values: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Log map from the unit sphere to the tangent space at ``base``.

    The log map is non-unique at antipodal points, so those inputs are
    rejected instead of being mapped to a false zero tangent.
    """
    y = _normalize_sphere_tensor(values)
    if not isinstance(base, torch.Tensor):
        raise TypeError("base must be a torch.Tensor")
    b = _normalize_sphere_tensor(base.to(device=y.device, dtype=y.dtype).reshape(1, -1))[0]
    dots = (y @ b).clamp(-1.0, 1.0)
    theta = dots.acos()
    if bool((dots <= -1.0 + _SPHERE_ANTIPODAL_TOL).any()):
        raise ValueError("spherical log map is undefined at antipodal points")
    tangent = y - dots[:, None] * b.reshape(1, -1)
    sin_theta = theta.sin()
    scale = _tc.ones_like(theta)
    mask = sin_theta > 1e-12
    scale = _tc.where(mask, theta / sin_theta.clamp_min(1e-300), scale)
    out = tangent * scale[:, None]
    return _tc.where((theta < 1e-12)[:, None], _tc.zeros_like(out), out)


def sphere_exp_map(tangent: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Exponential map from the ambient tangent space at ``base`` to the sphere."""
    if not isinstance(tangent, torch.Tensor):
        raise TypeError("tangent must be a torch.Tensor")
    z = tangent if _tc.is_floating_point(tangent) else tangent.to(dtype=_tc.float64)
    if z.dim() == 1:
        z = z.reshape(1, -1)
    if not isinstance(base, torch.Tensor):
        raise TypeError("base must be a torch.Tensor")
    b = _normalize_sphere_tensor(base.to(device=z.device, dtype=z.dtype).reshape(1, -1))[0]
    z = z - (z @ b)[:, None] * b.reshape(1, -1)
    r = torch.linalg.norm(z, dim=1)
    small = r < 1e-12
    scaled = (r.sin() / r.clamp_min(1e-300))[:, None] * z
    curved = r.cos()[:, None] * b.reshape(1, -1) + scaled
    linear = b.reshape(1, -1) + z
    out = _tc.where(small[:, None], linear, curved)
    return out / torch.linalg.norm(out, dim=1, keepdim=True)
