"""Torch implementations of response-geometry transforms.

Each public function accepts torch tensors and returns torch tensors, keeping
closure, log-ratio, simplex maps, and sphere maps differentiable and
device-local. NumPy callers use :mod:`gamfit._response_geometry` directly.
"""

from __future__ import annotations

from typing import Any

import torch

from . import _torch_compat as _tc


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
    coordinates: str = "clr",
    reference: int = -1,
) -> torch.Tensor:
    """Log map at an intrinsic simplex base point in CLR or ALR coordinates."""
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
    if coord in {"simplex", "clr"}:
        return clr(comp) - clr(base_arr)
    if coord == "alr":
        return alr(comp, reference=reference) - alr(base_arr, reference=reference)
    raise ValueError("simplex coordinates must be 'clr' or 'alr'")


def simplex_exp_map(
    tangent: torch.Tensor,
    base: torch.Tensor,
    *,
    coordinates: str = "clr",
    reference: int = -1,
) -> torch.Tensor:
    """Exponential map from simplex tangent coordinates back to compositions."""
    if not isinstance(tangent, torch.Tensor):
        raise TypeError("tangent must be a torch.Tensor")
    z = tangent if _tc.is_floating_point(tangent) else tangent.to(dtype=_tc.float64)
    if z.dim() == 1:
        z = z.reshape(1, -1)
    if not isinstance(base, torch.Tensor):
        raise TypeError("base must be a torch.Tensor")
    base_arr = _closure_tensor(base.to(device=z.device, dtype=z.dtype).reshape(1, -1))
    coord = coordinates.lower()
    if coord in {"simplex", "clr"}:
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
    raise ValueError("simplex coordinates must be 'clr' or 'alr'")


_SPHERE_ANTIPODAL_TOL = 1e-12


def _append_sphere_candidate(
    candidates: list[torch.Tensor], candidate: torch.Tensor
) -> None:
    c = candidate.reshape(-1)
    norm = torch.linalg.norm(c)
    if bool((~_tc.isfinite(norm)).any()) or float(norm) <= 0.0:
        return
    c = c / norm
    for existing in candidates:
        if abs(float(torch.dot(existing, c))) > 1.0 - 1e-10:
            return
    candidates.append(c)


def _sphere_orthogonal_unit(vector: torch.Tensor) -> torch.Tensor:
    v = vector.reshape(-1)
    axis = _tc.zeros_like(v)
    axis[int(torch.argmin(v.abs()).item())] = 1.0
    tangent = axis - torch.dot(axis, v) * v
    norm = torch.linalg.norm(tangent)
    if float(norm) <= 0.0:
        raise ValueError("cannot construct a tangent direction for the spherical mean")
    return tangent / norm


def _sphere_frechet_objective(
    values: torch.Tensor, weights: torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    dots = (values @ base).clamp(-1.0, 1.0)
    theta = dots.acos()
    return (weights * theta * theta).sum()


def _sphere_mean_candidates(values: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
    candidates: list[torch.Tensor] = []
    extrinsic = (values * weights[:, None]).sum(dim=0)
    _append_sphere_candidate(candidates, extrinsic)

    moment = (values * weights[:, None]).transpose(0, 1) @ values
    _, eigvecs = torch.linalg.eigh(moment)
    for j in range(eigvecs.shape[1]):
        _append_sphere_candidate(candidates, eigvecs[:, j])
        _append_sphere_candidate(candidates, -eigvecs[:, j])

    for row in values[: min(values.shape[0], 16)]:
        _append_sphere_candidate(candidates, row)

    if not candidates:
        candidates.append(_sphere_orthogonal_unit(values[0]))
    return candidates


def sphere_frechet_mean(
    values: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 256,
) -> torch.Tensor:
    """Intrinsic Fréchet/Karcher mean on the unit sphere."""
    y = _normalize_sphere_tensor(values)
    w = _normalized_weights_tensor(y.shape[0], weights, y)
    best_mu = None
    best_obj = float("inf")
    for candidate in _sphere_mean_candidates(y, w):
        mu = candidate.clone()
        try:
            for _ in range(max_iter):
                logs = sphere_log_map(y, mu)
                step = (logs * w[:, None]).sum(dim=0)
                step_norm = torch.linalg.norm(step)
                if bool(step_norm < tol):
                    break
                mu = sphere_exp_map(step.reshape(1, -1), mu)[0]
        except ValueError:
            continue
        obj = float(_sphere_frechet_objective(y, w, mu))
        if obj < best_obj:
            best_obj = obj
            best_mu = mu
    if best_mu is None:
        raise ValueError("spherical Fréchet mean is not identifiable for these points")
    return best_mu


def sphere_log_map(values: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Log map from the unit sphere to the ambient tangent space at ``base``."""
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
