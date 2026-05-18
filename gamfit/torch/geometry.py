"""Pure-torch response-geometry transforms with numerical parity to numpy.

This module mirrors :mod:`gamfit._response_geometry` so the same simplex / sphere
operations can be used inside a PyTorch graph. Every function accepts torch
tensors of any device and floating dtype, returns tensors of the same device
and dtype, and is differentiable via standard torch autograd (no custom
``Function`` needed — these are compositions of standard ops).

Numerical parity with the numpy versions is checked by the test suite; the
intent here is bit-faithful behaviour up to floating-point reassociation, not
a re-derivation. Read :mod:`gamfit._response_geometry` for the mathematical
specification.
"""

from __future__ import annotations

from typing import Any

import torch


def _as_2d(x: Any, *, label: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.dim() != 2:
        raise ValueError(f"{label} must be a 2-D tensor")
    if x.shape[0] == 0 or x.shape[1] < 2:
        raise ValueError(
            f"{label} must have at least one row and at least two columns"
        )
    if not torch.isfinite(x).all():
        raise ValueError(f"{label} must contain only finite values")
    return x


def _normalized_weights(n: int, weights: Any, ref: torch.Tensor) -> torch.Tensor:
    if weights is None:
        return torch.full(
            (n,), 1.0 / n, dtype=ref.dtype, device=ref.device
        )
    w = torch.as_tensor(weights, dtype=ref.dtype, device=ref.device).reshape(-1)
    if w.shape[0] != n:
        raise ValueError("weights length must match the number of rows")
    if not torch.isfinite(w).all() or (w < 0).any() or float(w.sum()) <= 0.0:
        raise ValueError(
            "weights must be finite, non-negative, and have positive total"
        )
    return w / w.sum()


def _normalize_sphere(values: Any, *, label: str = "spherical values") -> torch.Tensor:
    arr = _as_2d(values, label=label)
    norms = torch.linalg.norm(arr, dim=1, keepdim=True)
    if (norms <= 0).any():
        raise ValueError("spherical rows must have non-zero norm")
    return arr / norms


def closure(values: Any) -> torch.Tensor:
    """Normalize rows onto the probability simplex."""
    arr = _as_2d(values, label="simplex values")
    if (arr < 0).any():
        raise ValueError("simplex values must be non-negative")
    totals = arr.sum(dim=1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("simplex rows must have positive total mass")
    return arr / totals


def clr(values: Any) -> torch.Tensor:
    """Centered log-ratio coordinates for positive compositions."""
    comp = closure(values)
    if (comp <= 0).any():
        raise ValueError("CLR coordinates require strictly positive simplex values")
    logs = torch.log(comp)
    return logs - logs.mean(dim=1, keepdim=True)


def alr(values: Any, *, reference: int = -1) -> torch.Tensor:
    """Additive log-ratio coordinates for positive compositions."""
    comp = closure(values)
    if (comp <= 0).any():
        raise ValueError("ALR coordinates require strictly positive simplex values")
    d = comp.shape[1]
    ref = reference % d
    keep = [j for j in range(d) if j != ref]
    keep_idx = torch.as_tensor(keep, dtype=torch.long, device=comp.device)
    return torch.log(comp.index_select(1, keep_idx) / comp[:, ref : ref + 1])


def inverse_alr(coords: Any, *, reference: int = -1) -> torch.Tensor:
    """Map ALR coordinates back to the simplex."""
    if not isinstance(coords, torch.Tensor):
        z = torch.as_tensor(coords)
    else:
        z = coords
    if z.dim() != 2:
        raise ValueError("ALR coordinates must be a 2-D tensor")
    d = z.shape[1] + 1
    ref = reference % d
    log_parts = torch.zeros((z.shape[0], d), dtype=z.dtype, device=z.device)
    keep = [j for j in range(d) if j != ref]
    keep_idx = torch.as_tensor(keep, dtype=torch.long, device=z.device)
    log_parts = log_parts.index_copy(1, keep_idx, z)
    # Stable softmax / closure(exp(log_parts)).
    log_parts = log_parts - log_parts.max(dim=1, keepdim=True).values
    parts = torch.exp(log_parts)
    return parts / parts.sum(dim=1, keepdim=True)


def simplex_frechet_mean(values: Any, weights: Any | None = None) -> torch.Tensor:
    """Intrinsic Fréchet mean under Aitchison simplex geometry.

    The closed-form Aitchison mean is the geometric mean of the rows followed
    by closure; matches :func:`gamfit._response_geometry.simplex_frechet_mean`.
    Returns a 1-D tensor of length ``d``.
    """
    comp = closure(values)
    if (comp <= 0).any():
        raise ValueError("simplex Fréchet mean requires strictly positive values")
    w = _normalized_weights(comp.shape[0], weights, comp)
    mean_log = (torch.log(comp) * w.unsqueeze(1)).sum(dim=0)
    mean_log = mean_log - mean_log.max()
    out = torch.exp(mean_log)
    return out / out.sum()


def simplex_log_map(
    values: Any,
    base: Any,
    *,
    coordinates: str = "clr",
    reference: int = -1,
) -> torch.Tensor:
    """Log map at an intrinsic simplex base point in CLR or ALR coordinates."""
    comp = closure(values)
    base_t = torch.as_tensor(base, dtype=comp.dtype, device=comp.device).reshape(1, -1)
    base_arr = closure(base_t)[0]
    if comp.shape[1] != base_arr.shape[0]:
        raise ValueError("simplex values and base point have different dimensions")
    if (comp <= 0).any() or (base_arr <= 0).any():
        raise ValueError(
            "simplex log map requires strictly positive values and base point"
        )
    coord = coordinates.lower()
    if coord in {"simplex", "clr"}:
        return clr(comp) - clr(base_arr.reshape(1, -1))
    if coord == "alr":
        return alr(comp, reference=reference) - alr(
            base_arr.reshape(1, -1), reference=reference
        )
    raise ValueError("simplex coordinates must be 'clr' or 'alr'")


def simplex_exp_map(
    tangent: Any,
    base: Any,
    *,
    coordinates: str = "clr",
    reference: int = -1,
) -> torch.Tensor:
    """Exponential map from simplex tangent coordinates back to compositions."""
    z = tangent if isinstance(tangent, torch.Tensor) else torch.as_tensor(tangent)
    if z.dim() == 1:
        z = z.reshape(1, -1)
    base_t = torch.as_tensor(base, dtype=z.dtype, device=z.device).reshape(1, -1)
    base_arr = closure(base_t)[0]
    coord = coordinates.lower()
    if coord in {"simplex", "clr"}:
        if z.shape[1] != base_arr.shape[0]:
            raise ValueError("CLR tangent dimension must equal simplex dimension")
        log_parts = torch.log(base_arr.reshape(1, -1)) + z
        log_parts = log_parts - log_parts.max(dim=1, keepdim=True).values
        parts = torch.exp(log_parts)
        return parts / parts.sum(dim=1, keepdim=True)
    if coord == "alr":
        if z.shape[1] != base_arr.shape[0] - 1:
            raise ValueError("ALR tangent dimension must be simplex dimension minus one")
        base_alr = alr(base_arr.reshape(1, -1), reference=reference)
        return inverse_alr(base_alr + z, reference=reference)
    raise ValueError("simplex coordinates must be 'clr' or 'alr'")


def sphere_log_map(values: Any, base: Any) -> torch.Tensor:
    """Log map from the unit sphere to the ambient tangent space at ``base``."""
    y = _normalize_sphere(values)
    base_t = torch.as_tensor(base, dtype=y.dtype, device=y.device).reshape(1, -1)
    b = _normalize_sphere(base_t)[0]
    dots = (y @ b).clamp(-1.0, 1.0)
    theta = torch.arccos(dots)
    tangent = y - dots.unsqueeze(1) * b.reshape(1, -1)
    sin_theta = torch.sin(theta)
    scale = torch.ones_like(theta)
    mask = sin_theta > 1e-12
    safe_sin = torch.where(mask, sin_theta, torch.ones_like(sin_theta))
    scale = torch.where(mask, theta / safe_sin, scale)
    out = tangent * scale.unsqueeze(1)
    zero_mask = theta < 1e-12
    out = torch.where(zero_mask.unsqueeze(1), torch.zeros_like(out), out)
    return out


def sphere_exp_map(tangent: Any, base: Any) -> torch.Tensor:
    """Exponential map from the ambient tangent space at ``base`` to the sphere."""
    z = tangent if isinstance(tangent, torch.Tensor) else torch.as_tensor(tangent)
    if z.dim() == 1:
        z = z.reshape(1, -1)
    base_t = torch.as_tensor(base, dtype=z.dtype, device=z.device).reshape(1, -1)
    b = _normalize_sphere(base_t)[0]
    # Project away tiny numerical radial components so fitted coordinates stay tangent.
    z = z - (z @ b).unsqueeze(1) * b.reshape(1, -1)
    r = torch.linalg.norm(z, dim=1)
    small = r < 1e-12
    safe_r = torch.where(small, torch.ones_like(r), r)
    big = torch.cos(r).unsqueeze(1) * b.reshape(1, -1) + (
        torch.sin(r) / safe_r
    ).unsqueeze(1) * z
    small_branch = b.reshape(1, -1) + z
    out = torch.where(small.unsqueeze(1), small_branch, big)
    norms = torch.linalg.norm(out, dim=1, keepdim=True)
    return out / norms


def sphere_frechet_mean(
    values: Any,
    weights: Any | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 256,
) -> torch.Tensor:
    """Intrinsic Fréchet/Karcher mean on the unit sphere.

    Karcher iteration in the ambient tangent chart. Backpropagation flows
    through the entire iteration; as iterations converge, the resulting
    gradient converges to the implicit-function-theorem gradient.
    """
    y = _normalize_sphere(values)
    w = _normalized_weights(y.shape[0], weights, y)
    mu = (y * w.unsqueeze(1)).sum(dim=0)
    norm = torch.linalg.norm(mu)
    if float(norm) <= 1e-14:
        mu = y[0].clone()
    else:
        mu = mu / norm
    for _ in range(max_iter):
        logs = sphere_log_map(y, mu)
        step = (logs * w.unsqueeze(1)).sum(dim=0)
        step_norm = float(torch.linalg.norm(step))
        if step_norm < tol:
            break
        mu = sphere_exp_map(step.reshape(1, -1), mu)[0]
    return mu
