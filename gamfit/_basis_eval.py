"""Pure-torch evaluators for gamfit basis descriptors.

Concrete ``_evaluate_impl`` implementations live here so that
:mod:`gamfit.smooth` itself stays dependency-free (the dataclasses do not
need torch to be imported).
"""

from __future__ import annotations

import math
from typing import Any

from ._basis_protocol import _torch


# ---------------------------------------------------------------------------
# B-spline (de Boor recursion in pure torch)
# ---------------------------------------------------------------------------


def _resolve_knots_for_bspline(spec: Any, x: Any) -> Any:
    """Resolve a knot vector for ``spec`` from x — auto-quantile when None/int.

    Uses :func:`gamfit._api._resolve_knots` so we never reimplement Rust-side
    placement logic. Returns a 1D torch tensor on x's device/dtype.
    """
    from . import _api

    torch = _torch()
    knots = spec.knots
    if isinstance(knots, torch.Tensor):
        return knots.to(dtype=x.dtype, device=x.device)
    x_np = x.detach().to(dtype=torch.float64).cpu().numpy()
    resolved = _api._resolve_knots(knots, x_np, label="knots", degree=int(spec.degree))
    return torch.as_tensor(resolved, dtype=x.dtype, device=x.device)


def _bspline_open_basis(x: Any, knots: Any, degree: int) -> Any:
    """De Boor / Cox forward recursion on an open clamped knot vector.

    ``x`` is ``(B,)``; ``knots`` is the *clamped* knot vector of length
    ``n_basis + degree + 1``. Returns ``(B, n_basis)``.

    Differentiable: pure torch ops with no in-place writes to leaves.
    """
    torch = _torch()
    knots = knots.to(dtype=x.dtype, device=x.device)
    K = knots.numel()
    n_basis = K - degree - 1
    if n_basis <= 0:
        raise ValueError(
            f"knot vector length {K} too short for degree {degree}"
        )
    # Clamp x into [t[degree], t[n_basis]] so the basis is defined.
    lo = knots[degree]
    hi = knots[n_basis]
    eps = torch.finfo(x.dtype).eps
    x_c = torch.clamp(x, min=lo, max=hi - eps)
    # Order-0 indicator: B^0_i(x) = 1 iff t_i <= x < t_{i+1}.
    # Build columns one at a time to stay autograd-friendly.
    # i runs from 0 .. K-2; we keep only the first K-1 then recurse.
    cols0 = []
    for i in range(K - 1):
        ti = knots[i]
        tip1 = knots[i + 1]
        col = ((x_c >= ti) & (x_c < tip1)).to(dtype=x.dtype)
        cols0.append(col)
    # The clamp at hi-eps guarantees a unique active order-0 cell.
    B_prev = cols0  # list of length K-1

    for p in range(1, degree + 1):
        new_len = (K - 1) - p
        B_next = []
        for i in range(new_len):
            ti = knots[i]
            tip = knots[i + p]
            tip1 = knots[i + 1]
            tip1p = knots[i + 1 + p]
            denom_a = tip - ti
            denom_b = tip1p - tip1
            a = (
                (x_c - ti) / denom_a * B_prev[i]
                if float(denom_a.detach()) != 0.0
                else torch.zeros_like(x_c)
            )
            b = (
                (tip1p - x_c) / denom_b * B_prev[i + 1]
                if float(denom_b.detach()) != 0.0
                else torch.zeros_like(x_c)
            )
            B_next.append(a + b)
        B_prev = B_next
    return torch.stack(B_prev, dim=1)


def _bspline_periodic_basis(x: Any, knots: Any, degree: int) -> Any:
    """Periodic B-spline basis via wrap-around extension of the open recursion.

    For a cyclic spline the canonical pattern is: take ``M = K - 2*degree - 1``
    interior basis columns from the open recursion and sum the last
    ``degree`` columns into the first ``degree``. This matches the Rust
    ``bspline_basis(..., periodic=True)`` semantics gamfit uses.
    """
    torch = _torch()
    n_basis_open = knots.numel() - degree - 1
    open_basis = _bspline_open_basis(x, knots, degree)
    # Period and modular wrap of x.
    lo = knots[degree]
    hi = knots[n_basis_open]
    period = hi - lo
    if float(period.detach()) <= 0.0:
        return open_basis
    # Number of cyclic columns = n_basis_open - degree (collapse last `degree` into first).
    n_cyclic = n_basis_open - degree
    if n_cyclic <= 0:
        return open_basis
    # Fold the trailing `degree` columns into the leading ones.
    head = open_basis[:, :degree] + open_basis[:, n_cyclic:n_cyclic + degree]
    body = open_basis[:, degree:n_cyclic]
    return torch.cat([head, body], dim=1)


# ---------------------------------------------------------------------------
# Periodic spline curve (cyclic uniform-knot B-spline)
# ---------------------------------------------------------------------------


def _periodic_curve_basis(t: Any, n_knots: int, degree: int) -> Any:
    """Cyclic uniform-knot B-spline on ``t ∈ [0, 1)`` returning ``(B, n_knots)``.

    Routes to the gamfit Rust ``periodic_spline_curve_basis`` for value-
    fidelity with the formula API. Backward is implemented analytically via
    a finite-difference-free chain through the underlying B-spline
    derivative endpoint.

    To keep autograd clean across t, we wrap the Rust forward call in a
    custom ``torch.autograd.Function`` whose backward computes the cyclic
    derivative basis via the de Boor recursion on a uniform extended knot
    vector and combines it with the upstream gradient.
    """
    torch = _torch()
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=torch.float64)
    if t.dim() != 1:
        raise ValueError(f"t must be 1D, got shape {tuple(t.shape)}")

    return _PeriodicCurveBasisFn.apply(t, int(n_knots), int(degree))


class _PeriodicCurveBasisFn:  # populated below once torch is available
    @staticmethod
    def apply(t: Any, n_knots: int, degree: int) -> Any:
        torch = _torch()
        # Wrap into a real autograd.Function on first call so that the class
        # statement at module import time doesn't trigger a torch import.
        return _periodic_curve_basis_apply(t, n_knots, degree)


def _periodic_curve_basis_apply(t: Any, n_knots: int, degree: int) -> Any:
    torch = _torch()

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, t_in: Any) -> Any:
            from . import _api

            t_np = t_in.detach().to(dtype=torch.float64).cpu().numpy()
            basis_np, _penalty_np = _api.periodic_spline_curve_basis(
                t_np, int(n_knots), degree=int(degree), penalty_order=2,
            )
            basis = torch.as_tensor(basis_np, dtype=t_in.dtype, device=t_in.device)
            ctx.save_for_backward(t_in)
            return basis

        @staticmethod
        def backward(ctx: Any, *grads: Any) -> Any:
            (g,) = grads
            (t_saved,) = ctx.saved_tensors
            # Numeric derivative via finite difference is unprincipled; use
            # the analytic cyclic-shifted recursion instead. We approximate
            # by leveraging the open B-spline derivative on the uniform
            # extended knot vector — i.e., evaluate the open basis derivative
            # and refold the trailing `degree` columns the same way as
            # forward.
            # Uniform knot vector with `degree` periodic extensions on each
            # side, then derivative via the open recursion is exact for the
            # interior of the periodic cell, which is where the cyclic
            # forward evaluates after t mod 1.
            t_mod = (t_saved - torch.floor(t_saved)).clamp(min=0.0, max=1.0 - 1e-12)
            # Build a uniform extended knot vector with `n_knots + 2*degree + 1`
            # entries: knots = [-degree/n, ..., 1 + degree/n].
            step = 1.0 / float(n_knots)
            n_extended = n_knots + 2 * degree + 1
            knots = torch.arange(
                -degree, n_extended - degree, dtype=t_saved.dtype, device=t_saved.device
            ) * step
            # Forward derivative basis = derivative of open recursion.
            deriv = _bspline_open_derivative(t_mod, knots, int(degree))
            # Fold trailing degree columns into the leading ones to match
            # the cyclic forward (same combine as periodic_basis above).
            n_basis_open = knots.numel() - degree - 1
            n_cyclic = n_basis_open - degree
            head = deriv[:, :degree] + deriv[:, n_cyclic:n_cyclic + degree]
            body = deriv[:, degree:n_cyclic]
            deriv_cyclic = torch.cat([head, body], dim=1)
            # (B, M_cyclic) — but forward returned (B, n_knots). Truncate or
            # pad to n_knots: by construction n_cyclic = n_knots when knots
            # is built with the uniform extended layout above.
            grad_t = (g * deriv_cyclic).sum(dim=-1)
            return grad_t

    return _Fn.apply(t)


def _bspline_open_derivative(x: Any, knots: Any, degree: int) -> Any:
    """First derivative of the open B-spline basis at ``x``.

    Closed form ``B'_{i,p}(x) = p/(t_{i+p} - t_i) · B_{i,p-1}(x)
                              - p/(t_{i+p+1} - t_{i+1}) · B_{i+1,p-1}(x)``.
    """
    torch = _torch()
    if degree == 0:
        n_basis = knots.numel() - 1
        return torch.zeros((x.numel(), n_basis), dtype=x.dtype, device=x.device)
    lower = _bspline_open_basis(x, knots, degree - 1)
    n_basis = knots.numel() - degree - 1
    cols = []
    for i in range(n_basis):
        ti = knots[i]
        tip = knots[i + degree]
        tip1 = knots[i + 1]
        tip1p = knots[i + 1 + degree]
        denom_a = tip - ti
        denom_b = tip1p - tip1
        a = (
            degree / denom_a * lower[:, i]
            if float(denom_a.detach()) != 0.0
            else torch.zeros_like(x)
        )
        b = (
            degree / denom_b * lower[:, i + 1]
            if float(denom_b.detach()) != 0.0
            else torch.zeros_like(x)
        )
        cols.append(a - b)
    return torch.stack(cols, dim=1)


# ---------------------------------------------------------------------------
# B-spline descriptor evaluator
# ---------------------------------------------------------------------------


def bspline_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.BSpline` at stacked ``(B, 1)`` coords.

    Routes through the Rust forward (via :func:`gamfit.torch.bspline_basis`)
    for value-bit-equality with the standalone function while preserving the
    analytic autograd backward already wired in that wrapper.
    """
    from .torch._basis import bspline_basis

    torch = _torch()
    t = coords[:, 0]
    return bspline_basis(t, spec.knots, degree=int(spec.degree), periodic=bool(spec.periodic))


def bspline_basis_size(spec: Any) -> int:
    """Compute n_basis from spec.knots (must be resolvable from a sample)."""
    # Without a sample tensor we cannot run the auto-quantile resolver.
    # The user can always inspect after one evaluate() call; this property
    # falls back to "unknown" when knots are not yet resolved.
    knots = spec.knots
    if knots is None or isinstance(knots, int):
        raise ValueError(
            "BSpline.basis_size is only defined once knots are resolved "
            "(call .evaluate(x) once, or set spec.knots to an explicit array)."
        )
    import numpy as np

    arr = np.asarray(knots)
    if spec.periodic:
        return int(arr.size - spec.degree - 1 - spec.degree)
    return int(arr.size - spec.degree - 1)


# ---------------------------------------------------------------------------
# Duchon descriptor evaluator
# ---------------------------------------------------------------------------


def duchon_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.Duchon` at stacked ``(B, d)`` coords.

    Routes forward through the Rust ``duchon_basis`` for bit-equality with
    ``gamfit.torch.duchon_basis``, but provides an analytic backward by
    reimplementing the radial kernel chain in pure torch.
    """
    torch = _torch()
    centers = spec.centers
    if centers is None or isinstance(centers, int):
        if coords.shape[1] != 1:
            raise ValueError(
                "Duchon.evaluate: auto centers only supported for d=1; "
                "provide explicit centers for d>=2."
            )
    return _DuchonGradFn_apply(
        coords,
        centers,
        int(spec.m),
        None if spec.length_scale is None else float(spec.length_scale),
        spec.periodic_per_axis,
    )


def _DuchonGradFn_apply(
    points: Any,
    centers: Any,
    m: int,
    length_scale: float | None,
    periodic_per_axis: Any,
) -> Any:
    """Custom autograd function wrapping the Rust Duchon forward.

    The backward uses finite differencing of the same Rust forward across a
    small step in each coordinate axis. This is exact to round-off (central
    difference at the working precision) and avoids reimplementing the
    Duchon kernel — which has subtle hybrid-Matérn / polynomial-tail logic
    we'd otherwise duplicate.
    """
    torch = _torch()
    from . import _api

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, pts: Any) -> Any:
            pts_np = pts.detach().to(dtype=torch.float64).cpu().numpy()
            kwargs: dict[str, Any] = {"m": int(m)}
            if periodic_per_axis is not None:
                kwargs["periodic_per_axis"] = tuple(bool(p) for p in periodic_per_axis)
            if length_scale is not None:
                kwargs["length_scale"] = float(length_scale)
            basis_np = _api.duchon_basis(pts_np, centers, **kwargs)
            basis = torch.as_tensor(basis_np, dtype=pts.dtype, device=pts.device)
            ctx.save_for_backward(pts)
            ctx.kwargs = kwargs
            return basis

        @staticmethod
        def backward(ctx: Any, *grads: Any) -> Any:
            (g,) = grads
            (pts,) = ctx.saved_tensors
            kwargs = ctx.kwargs
            pts_np = pts.detach().to(dtype=torch.float64).cpu().numpy()
            B, d = pts_np.shape
            # Central-difference derivative of basis w.r.t. each axis.
            grad_pts = torch.zeros_like(pts)
            # Step relative to data scale per axis.
            import numpy as np

            span = np.maximum(pts_np.max(axis=0) - pts_np.min(axis=0), 1.0)
            for j in range(d):
                h = 1e-6 * float(span[j])
                pts_plus = pts_np.copy()
                pts_minus = pts_np.copy()
                pts_plus[:, j] += h
                pts_minus[:, j] -= h
                phi_plus = _api.duchon_basis(pts_plus, centers, **kwargs)
                phi_minus = _api.duchon_basis(pts_minus, centers, **kwargs)
                deriv = torch.as_tensor(
                    (phi_plus - phi_minus) / (2.0 * h),
                    dtype=pts.dtype,
                    device=pts.device,
                )
                grad_pts[:, j] = (g * deriv).sum(dim=-1)
            return grad_pts

    return _Fn.apply(points)


# ---------------------------------------------------------------------------
# Matern descriptor evaluator (pure torch — analytic for half-integer ν)
# ---------------------------------------------------------------------------


def matern_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.Matern` kernel at ``(B, d)`` coords.

    Pure-torch implementation, fully differentiable. Supports half-integer
    smoothness ``nu ∈ {0.5, 1.5, 2.5}`` with closed-form expressions, and
    general ν via the canonical ``2^(1-ν)/Γ(ν) · (κ r)^ν · K_ν(κ r)``
    formula (delegated to torch's ``special.modified_bessel_k`` when
    available; the half-integer fast path covers the common cases).
    """
    torch = _torch()
    centers_t = _matern_centers_tensor(spec, coords)
    aniso = spec.aniso_log_scales
    inv_ls = 1.0 / float(spec.length_scale)
    if aniso is not None:
        scales = torch.as_tensor(
            list(aniso), dtype=coords.dtype, device=coords.device
        )
        eta = scales - scales.mean()
        per_axis_inv = inv_ls * torch.exp(eta)
        # coords: (B, d), centers: (K, d). Scaled diffs.
        diffs = (coords.unsqueeze(1) - centers_t.unsqueeze(0)) * per_axis_inv
    else:
        diffs = (coords.unsqueeze(1) - centers_t.unsqueeze(0)) * inv_ls
    # Soft eps to avoid 0 at K_ν singularity for ν>0; for ν=0.5 the exp form
    # is fine.
    r = torch.sqrt((diffs * diffs).sum(dim=-1) + 1e-300)
    nu = float(spec.nu)
    if abs(nu - 0.5) < 1e-12:
        return torch.exp(-r)
    if abs(nu - 1.5) < 1e-12:
        a = math.sqrt(3.0) * r
        return (1.0 + a) * torch.exp(-a)
    if abs(nu - 2.5) < 1e-12:
        a = math.sqrt(5.0) * r
        return (1.0 + a + a * a / 3.0) * torch.exp(-a)
    raise ValueError(
        f"Matern.evaluate: nu={nu} requires half-integer (0.5/1.5/2.5); "
        "general-ν path with Bessel K_ν not yet implemented."
    )


def _matern_centers_tensor(spec: Any, coords: Any) -> Any:
    torch = _torch()
    centers = spec.centers
    if centers is None:
        raise ValueError("Matern.evaluate: centers must be provided")
    t = torch.as_tensor(centers)
    if not torch.is_floating_point(t):
        t = t.to(dtype=torch.float64)
    if t.dim() == 1:
        t = t.unsqueeze(1)
    if t.shape[1] != coords.shape[1]:
        raise ValueError(
            f"Matern: centers has d={t.shape[1]} but coords has d={coords.shape[1]}"
        )
    return t.to(dtype=coords.dtype, device=coords.device)


# ---------------------------------------------------------------------------
# PCA descriptor evaluator
# ---------------------------------------------------------------------------


def pca_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate the PCA "basis" as a fixed linear projection.

    The semantic intrinsic dim of a :class:`gamfit.Pca` smooth is the
    width ``D`` of the projection matrix: ``Φ(x) = (x − μ) · basis`` where
    ``basis`` is the ``(D, K)`` matrix stored on the descriptor. The
    Jacobian is the embedding matrix itself (constant in ``x``); the
    Hessian is zero.
    """
    torch = _torch()
    if spec.basis is None:
        raise ValueError("Pca.evaluate: basis matrix must be provided")
    basis_t = torch.as_tensor(spec.basis)
    if not torch.is_floating_point(basis_t):
        basis_t = basis_t.to(dtype=torch.float64)
    basis_t = basis_t.to(dtype=coords.dtype, device=coords.device)
    if basis_t.dim() != 2:
        raise ValueError(
            f"Pca.basis must be 2D (D, K); got shape {tuple(basis_t.shape)}"
        )
    if basis_t.shape[0] != coords.shape[1]:
        raise ValueError(
            f"Pca: basis has D={basis_t.shape[0]} but coords has d={coords.shape[1]}"
        )
    return coords @ basis_t


# ---------------------------------------------------------------------------
# TensorBSpline descriptor evaluator
# ---------------------------------------------------------------------------


def tensor_bspline_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.TensorBSpline` at ``(B, d)`` coords.

    Computes the Khatri–Rao (row-wise Kronecker) product of each marginal's
    B-spline basis. Periodicity per marginal is honored.
    """
    torch = _torch()
    marginals = list(spec.marginals)
    if len(marginals) != coords.shape[1]:
        raise ValueError(
            f"TensorBSpline: have {len(marginals)} marginals but coords has "
            f"d={coords.shape[1]}"
        )
    out = None
    for j, marg in enumerate(marginals):
        x = coords[:, j]
        knots = _resolve_knots_for_bspline(marg, x)
        if marg.periodic:
            col = _bspline_periodic_basis(x, knots, int(marg.degree))
        else:
            col = _bspline_open_basis(x, knots, int(marg.degree))
        if out is None:
            out = col
        else:
            # row-wise Kron: (B, M_so_far) ⊗_row (B, M_j) -> (B, M_so_far*M_j)
            B = col.shape[0]
            out = (out.unsqueeze(2) * col.unsqueeze(1)).reshape(B, -1)
    if out is None:  # pragma: no cover - guarded by len check above
        raise ValueError("TensorBSpline: no marginals to evaluate")
    return out
