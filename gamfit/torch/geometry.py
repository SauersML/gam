"""Torch implementations of response-geometry transforms.

Tensor inputs stay in torch so closure, log-ratio, simplex maps, and sphere
maps remain differentiable and device-local. Non-tensor inputs are delegated
to :mod:`gamfit._response_geometry` so NumPy callers keep the same behavior.
"""

from __future__ import annotations

from typing import Any

from .. import _response_geometry as _np_geom
from . import _torch_compat as _tc
from ._coerce import to_numpy_f64


def _is_tensor(value: Any) -> bool:
    import torch

    return isinstance(value, torch.Tensor)


def _as_float_tensor(value: Any, ref: Any | None = None) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(ref, torch.Tensor):
        tensor = _tc.as_tensor(value, device=ref.device, dtype=ref.dtype)
    else:
        tensor = _tc.as_tensor(value)
    if _tc.is_floating_point(tensor):
        return tensor
    dtype = (
        ref.dtype
        if isinstance(ref, torch.Tensor) and _tc.is_floating_point(ref)
        else _tc.float64
    )
    device = ref.device if isinstance(ref, torch.Tensor) else tensor.device
    return tensor.to(device=device, dtype=dtype)


def _matrix(value: Any, *, label: str, ref: Any | None = None) -> Any:
    tensor = _as_float_tensor(value, ref)
    if tensor.dim() != 2:
        raise ValueError(f"{label} must be a 2-D numeric array")
    if tensor.shape[0] == 0 or tensor.shape[1] < 2:
        raise ValueError(f"{label} must have at least one row and at least two columns")
    if bool((~_tc.isfinite(tensor)).any()):
        raise ValueError(f"{label} must contain only finite values")
    return tensor


def _closure_tensor(values: Any, *, label: str = "simplex values") -> Any:
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


def _normalized_weights_tensor(n: int, weights: Any | None, ref: Any) -> Any:
    if weights is None:
        return _tc.full((n,), 1.0 / n, dtype=ref.dtype, device=ref.device)
    w = _as_float_tensor(weights, ref).reshape(-1)
    if w.shape[0] != n:
        raise ValueError("weights length must match the number of rows")
    if bool((~_tc.isfinite(w)).any()) or bool((w < 0.0).any()) or bool(w.sum() <= 0.0):
        raise ValueError("weights must be finite, non-negative, and have positive total")
    return w / w.sum()


def _normalize_sphere_tensor(values: Any, *, ref: Any | None = None) -> Any:
    import torch

    tensor = _matrix(values, label="spherical values", ref=ref)
    norms = torch.linalg.norm(tensor, dim=1, keepdim=True)
    if bool((norms <= 0.0).any()):
        raise ValueError("spherical rows must have non-zero norm")
    return tensor / norms


def closure(values: Any) -> Any:
    """Normalize rows onto the probability simplex.

    Tensor inputs stay in torch and preserve autograd.
    """
    if _is_tensor(values):
        return _closure_tensor(values)
    out = _np_geom.closure(to_numpy_f64(values))
    return out


def clr(values: Any) -> Any:
    """Centered log-ratio coordinates for positive compositions.

    Tensor inputs stay in torch and preserve autograd.
    """
    if _is_tensor(values):
        comp = _closure_tensor(values)
        if bool((comp <= 0.0).any()):
            raise ValueError("CLR coordinates require strictly positive simplex values")
        logs = comp.log()
        return logs - logs.mean(dim=1, keepdim=True)
    out = _np_geom.clr(to_numpy_f64(values))
    return out


def alr(values: Any, *, reference: int = -1) -> Any:
    """Additive log-ratio coordinates for positive compositions.

    Tensor inputs stay in torch and preserve autograd.
    """
    if _is_tensor(values):
        comp = _closure_tensor(values)
        if bool((comp <= 0.0).any()):
            raise ValueError("ALR coordinates require strictly positive simplex values")
        ref = reference % comp.shape[1]
        keep = _keep_without_reference(comp.shape[1], reference)
        return (comp[:, keep] / comp[:, [ref]]).log()
    out = _np_geom.alr(to_numpy_f64(values), reference=reference)
    return out


def inverse_alr(coords: Any, *, reference: int = -1) -> Any:
    """Map ALR coordinates back to the simplex.

    Tensor inputs stay in torch and preserve autograd.
    """
    if _is_tensor(coords):
        z = _as_float_tensor(coords)
        if z.dim() != 2:
            raise ValueError("ALR coordinates must be a 2-D numeric array")
        d = z.shape[1] + 1
        keep = _keep_without_reference(d, reference)
        log_parts = _tc.zeros((z.shape[0], d), dtype=z.dtype, device=z.device)
        log_parts[:, keep] = z
        log_parts = log_parts - log_parts.max(dim=1, keepdim=True).values
        parts = log_parts.exp()
        return parts / parts.sum(dim=1, keepdim=True)
    out = _np_geom.inverse_alr(to_numpy_f64(coords), reference=reference)
    return out


def simplex_frechet_mean(values: Any, weights: Any | None = None) -> Any:
    """Intrinsic Fréchet mean under Aitchison simplex geometry.

    Tensor inputs stay in torch and preserve autograd.
    """
    if _is_tensor(values):
        comp = _closure_tensor(values)
        if bool((comp <= 0.0).any()):
            raise ValueError("simplex Fréchet mean requires strictly positive values")
        w = _normalized_weights_tensor(comp.shape[0], weights, comp)
        mean_log = (comp.log() * w[:, None]).sum(dim=0)
        mean_log = mean_log - mean_log.max()
        out = mean_log.exp()
        return out / out.sum()
    out = _np_geom.simplex_frechet_mean(
        to_numpy_f64(values),
        None if weights is None else to_numpy_f64(weights),
    )
    return out


def simplex_log_map(
    values: Any,
    base: Any,
    *,
    coordinates: str = "clr",
    reference: int = -1,
) -> Any:
    """Log map at an intrinsic simplex base point in CLR or ALR coordinates.

    Tensor inputs stay in torch and preserve autograd.
    """
    if _is_tensor(values):
        comp = _closure_tensor(values)
        base_arr = _closure_tensor(_as_float_tensor(base, values).reshape(1, -1))
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
    out = _np_geom.simplex_log_map(
        to_numpy_f64(values),
        to_numpy_f64(base),
        coordinates=coordinates,
        reference=reference,
    )
    return out


def simplex_exp_map(
    tangent: Any,
    base: Any,
    *,
    coordinates: str = "clr",
    reference: int = -1,
) -> Any:
    """Exponential map from simplex tangent coordinates back to compositions.

    Tensor inputs stay in torch and preserve autograd.
    """
    if _is_tensor(tangent):
        z = _as_float_tensor(tangent)
        if z.dim() == 1:
            z = z.reshape(1, -1)
        base_arr = _closure_tensor(_as_float_tensor(base, z).reshape(1, -1))
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
    out = _np_geom.simplex_exp_map(
        to_numpy_f64(tangent),
        to_numpy_f64(base),
        coordinates=coordinates,
        reference=reference,
    )
    return out


def sphere_frechet_mean(
    values: Any,
    weights: Any | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 256,
) -> Any:
    """Intrinsic Fréchet/Karcher mean on the unit sphere.

    Tensor inputs stay in torch and preserve autograd.
    """
    if _is_tensor(values):
        import torch

        y = _normalize_sphere_tensor(values)
        w = _normalized_weights_tensor(y.shape[0], weights, y)
        mu = (y * w[:, None]).sum(dim=0)
        norm = torch.linalg.norm(mu)
        mu = _tc.where(norm <= 1e-14, y[0], mu / norm.clamp_min(1e-300))
        for _ in range(max_iter):
            logs = sphere_log_map(y, mu)
            step = (logs * w[:, None]).sum(dim=0)
            step_norm = torch.linalg.norm(step)
            if bool(step_norm < tol):
                break
            mu = sphere_exp_map(step.reshape(1, -1), mu)[0]
        return mu
    out = _np_geom.sphere_frechet_mean(
        to_numpy_f64(values),
        None if weights is None else to_numpy_f64(weights),
        tol=tol,
        max_iter=max_iter,
    )
    return out


def sphere_log_map(values: Any, base: Any) -> Any:
    """Log map from the unit sphere to the ambient tangent space at ``base``.

    Tensor inputs stay in torch and preserve autograd.
    """
    if _is_tensor(values):
        y = _normalize_sphere_tensor(values)
        b = _normalize_sphere_tensor(_as_float_tensor(base, y).reshape(1, -1), ref=y)[0]
        dots = (y @ b).clamp(-1.0, 1.0)
        theta = dots.acos()
        tangent = y - dots[:, None] * b.reshape(1, -1)
        sin_theta = theta.sin()
        scale = _tc.ones_like(theta)
        mask = sin_theta > 1e-12
        scale = _tc.where(mask, theta / sin_theta.clamp_min(1e-300), scale)
        out = tangent * scale[:, None]
        return _tc.where((theta < 1e-12)[:, None], _tc.zeros_like(out), out)
    out = _np_geom.sphere_log_map(to_numpy_f64(values), to_numpy_f64(base))
    return out


def sphere_exp_map(tangent: Any, base: Any) -> Any:
    """Exponential map from the ambient tangent space at ``base`` to the sphere.

    Tensor inputs stay in torch and preserve autograd.
    """
    if _is_tensor(tangent):
        import torch

        z = _as_float_tensor(tangent)
        if z.dim() == 1:
            z = z.reshape(1, -1)
        b = _normalize_sphere_tensor(_as_float_tensor(base, z).reshape(1, -1), ref=z)[0]
        z = z - (z @ b)[:, None] * b.reshape(1, -1)
        r = torch.linalg.norm(z, dim=1)
        small = r < 1e-12
        scaled = (r.sin() / r.clamp_min(1e-300))[:, None] * z
        curved = r.cos()[:, None] * b.reshape(1, -1) + scaled
        linear = b.reshape(1, -1) + z
        out = _tc.where(small[:, None], linear, curved)
        return out / torch.linalg.norm(out, dim=1, keepdim=True)
    out = _np_geom.sphere_exp_map(to_numpy_f64(tangent), to_numpy_f64(base))
    return out
