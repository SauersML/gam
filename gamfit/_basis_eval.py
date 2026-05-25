"""Rust-routed evaluators for gamfit basis descriptors.

Every concrete ``_evaluate_torch`` implementation routes through the
existing Rust kernels exposed by :mod:`gamfit._api` / :mod:`gamfit.torch._basis`,
wrapped in :class:`torch.autograd.Function` where the analytic backward is
already wired. Forward bytes are bit-equal to the standalone Rust calls.

Critical correctness rule (kept explicit here so future maintenance does
not drift): the math — B-spline de Boor recursion, Duchon kernel,
Matern half-integer reductions, etc. — is *only* allowed to live in the
Rust core. This file is a thin descriptor-to-Rust adapter; pure-torch
reimplementations of any of those kernels are forbidden.
"""

from __future__ import annotations

from typing import Any

from ._basis_protocol import _torch


# ---------------------------------------------------------------------------
# B-spline descriptor evaluator
# ---------------------------------------------------------------------------


def bspline_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.BSpline` at stacked ``(B, 1)`` coords.

    Routes forward + backward through :func:`gamfit.torch._basis.bspline_basis`,
    which wraps the Rust ``bspline_basis`` PyFFI call inside a
    :class:`torch.autograd.Function`. The analytic VJP back to ``coords``
    uses the Rust ``bspline_basis_derivative`` kernel.
    """
    from .torch._basis import bspline_basis

    if coords.shape[1] != 1:
        raise ValueError(
            f"BSpline.evaluate: 1D-only; got coords with d={coords.shape[1]}. "
            "Use TensorBSpline for multi-d with different units, "
            "or Duchon for radial."
        )
    t = coords[:, 0]
    return bspline_basis(
        t, spec.knots, degree=int(spec.degree), periodic=bool(spec.periodic),
    )


def bspline_basis_size(spec: Any) -> int:
    """Compute n_basis from spec.knots when knots are resolved."""
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
# Periodic spline curve evaluator (cyclic uniform-knot B-spline)
# ---------------------------------------------------------------------------


def _periodic_curve_basis(t: Any, n_knots: int, degree: int) -> Any:
    """Cyclic uniform-knot B-spline on ``t ∈ [0, 1)``, ``(B, n_knots)``.

    Routes through :func:`gamfit.torch._basis.periodic_spline_curve_basis`
    which calls the Rust ``periodic_spline_curve_basis`` kernel. Forward
    only — autograd through ``t`` is not exposed by the Rust binding for
    this basis. Callers needing gradients through ``t`` should compose
    with a primitive whose VJP IS exposed (e.g. a 1D BSpline upstream).
    """
    from .torch._basis import periodic_spline_curve_basis

    torch = _torch()
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=torch.float64)
    if t.dim() != 1:
        raise ValueError(f"t must be 1D, got shape {tuple(t.shape)}")
    basis, _penalty = periodic_spline_curve_basis(
        t, int(n_knots), degree=int(degree), penalty_order=2,
    )
    return basis


# ---------------------------------------------------------------------------
# Duchon descriptor evaluator
# ---------------------------------------------------------------------------


def duchon_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.Duchon` at stacked ``(B, d)`` coords.

    Forward routes through the Rust ``duchon_basis`` PyFFI call. Backward
    through ``coords`` is delivered by central-differencing the *same*
    Rust forward across each coordinate axis — this never re-derives the
    kernel in Python; it only takes a numerical derivative of the Rust
    output. For descriptors with ``length_scale`` and ``periodic_per_axis``
    the Rust kernel is the single source of math truth.
    """
    centers = spec.centers
    if (centers is None or isinstance(centers, int)) and coords.shape[1] != 1:
        raise ValueError(
            "Duchon.evaluate: auto centers only supported for d=1; "
            "provide explicit centers for d>=2."
        )
    return _DuchonRustFn_apply(
        coords,
        centers,
        int(spec.m),
        None if spec.length_scale is None else float(spec.length_scale),
        spec.periodic_per_axis,
    )


def _DuchonRustFn_apply(
    points: Any,
    centers: Any,
    m: int,
    length_scale: float | None,
    periodic_per_axis: Any,
) -> Any:
    """Autograd boundary: forward + FD backward both call the Rust kernel."""
    torch = _torch()
    from . import _api

    kwargs: dict[str, Any] = {"m": int(m)}
    if periodic_per_axis is not None:
        kwargs["periodic_per_axis"] = tuple(bool(p) for p in periodic_per_axis)
    if length_scale is not None:
        kwargs["length_scale"] = float(length_scale)

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, pts: Any) -> Any:
            pts_np = pts.detach().to(dtype=torch.float64).cpu().numpy()
            basis_np = _api.duchon_basis(pts_np, centers, **kwargs)
            basis = torch.as_tensor(basis_np, dtype=pts.dtype, device=pts.device)
            ctx.save_for_backward(pts)
            return basis

        @staticmethod
        def backward(ctx: Any, *grads: Any) -> Any:
            import numpy as np

            (g,) = grads
            (pts,) = ctx.saved_tensors
            pts_np = pts.detach().to(dtype=torch.float64).cpu().numpy()
            _B, d = pts_np.shape
            grad_pts = torch.zeros_like(pts)
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
# Matern descriptor evaluator
# ---------------------------------------------------------------------------


def matern_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.Matern` kernel at ``(B, d)`` coords.

    Currently raises :class:`NotImplementedError`: there is no
    ``matern_basis`` PyFFI binding on the torch / NumPy path that we can
    route through, and per the gamfit invariant we will not reimplement
    the Matern kernel in pure torch. Once the Rust core exposes
    ``rust_module().matern_basis(points, centers, nu, length_scale, ...)``,
    this evaluator should wrap it in a :class:`torch.autograd.Function`
    (forward through Rust; backward via the Rust kernel gradient when
    available, or a finite-difference-of-Rust fallback).
    """
    raise NotImplementedError(
        "Matern.evaluate is not yet wired: the Rust matern_basis PyFFI "
        "binding for the design matrix has not been exposed to gamfit._api. "
        "Reimplementing the Matern kernel in pure torch would violate the "
        "single-source-of-math invariant. Track the matern_basis FFI export "
        "to enable this path."
    )


# ---------------------------------------------------------------------------
# PCA descriptor evaluator (fixed linear projection)
# ---------------------------------------------------------------------------


def pca_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate the Pca smooth as a fixed linear projection.

    The PCA basis is a precomputed ``(D, K)`` projection matrix supplied
    by the user (no gamfit-specific math kernel exists for it — the
    ``Φ(x) = (x − μ) · basis`` operation is standard gemm). Returned
    tensor carries autograd back to ``coords``.
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
    if spec.centered:
        mean = coords.mean(dim=0, keepdim=True)
        return (coords - mean) @ basis_t
    return coords @ basis_t


# ---------------------------------------------------------------------------
# TensorBSpline descriptor evaluator
# ---------------------------------------------------------------------------


def tensor_bspline_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.TensorBSpline` at ``(B, d)`` coords.

    Each marginal's 1D basis is evaluated through the Rust ``bspline_basis``
    kernel (via :func:`gamfit.torch._basis.bspline_basis`, with analytic
    grad through the input). The joint design is the row-wise Khatri-Rao
    (Hadamard outer) product of the marginals, computed with standard
    torch tensor ops — that's plain tensor algebra, not a gam-specific
    math kernel.
    """
    from .torch._basis import bspline_basis

    marginals = list(spec.marginals)
    if len(marginals) != coords.shape[1]:
        raise ValueError(
            f"TensorBSpline: have {len(marginals)} marginals but coords "
            f"has d={coords.shape[1]}"
        )
    if not marginals:
        raise ValueError("TensorBSpline: no marginals to evaluate")
    out = None
    for j, marg in enumerate(marginals):
        x = coords[:, j]
        col = bspline_basis(
            x, marg.knots, degree=int(marg.degree), periodic=bool(marg.periodic),
        )
        if out is None:
            out = col
        else:
            B = col.shape[0]
            out = (out.unsqueeze(2) * col.unsqueeze(1)).reshape(B, -1)
    return out


# ---------------------------------------------------------------------------
# Sphere descriptor evaluator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NumPy backend evaluators (no autograd; route straight to the Rust FFI)
# ---------------------------------------------------------------------------


def bspline_evaluate_numpy(spec: Any, coords: Any) -> Any:
    """NumPy-backend B-spline evaluation. Routes through the Rust FFI."""
    import numpy as np

    from . import _api

    if coords.shape[1] != 1:
        raise ValueError(
            f"BSpline.evaluate: 1D-only; got coords with d={coords.shape[1]}"
        )
    return np.asarray(
        _api.bspline_basis(
            coords[:, 0],
            spec.knots,
            degree=int(spec.degree),
            periodic=bool(spec.periodic),
        ),
        dtype=np.float64,
    )


def duchon_evaluate_numpy(spec: Any, coords: Any) -> Any:
    """NumPy-backend Duchon evaluation. Routes through the Rust FFI."""
    import numpy as np

    from . import _api

    kwargs: dict[str, Any] = {"m": int(spec.m)}
    if spec.periodic_per_axis is not None:
        kwargs["periodic_per_axis"] = tuple(bool(p) for p in spec.periodic_per_axis)
    if spec.length_scale is not None:
        kwargs["length_scale"] = float(spec.length_scale)
    return np.asarray(
        _api.duchon_basis(coords, spec.centers, **kwargs), dtype=np.float64,
    )


def tensor_bspline_evaluate_numpy(spec: Any, coords: Any) -> Any:
    """NumPy-backend TensorBSpline evaluation.

    Calls the Rust ``bspline_basis`` PyFFI for each marginal and forms the
    row-wise Khatri-Rao product in NumPy. Khatri-Rao composition is
    standard tensor algebra and not a gam-specific math kernel.
    """
    import numpy as np

    from . import _api

    marginals = list(spec.marginals)
    if len(marginals) != coords.shape[1]:
        raise ValueError(
            f"TensorBSpline: have {len(marginals)} marginals but coords "
            f"has d={coords.shape[1]}"
        )
    if not marginals:
        raise ValueError("TensorBSpline: no marginals to evaluate")
    out = None
    for j, marg in enumerate(marginals):
        col = np.asarray(
            _api.bspline_basis(
                coords[:, j],
                marg.knots,
                degree=int(marg.degree),
                periodic=bool(marg.periodic),
            ),
            dtype=np.float64,
        )
        if out is None:
            out = col
        else:
            B = col.shape[0]
            out = (out[:, :, None] * col[:, None, :]).reshape(B, -1)
    return out


def pca_evaluate_numpy(spec: Any, coords: Any) -> Any:
    """NumPy-backend Pca evaluation (standard linear projection)."""
    import numpy as np

    if spec.basis is None:
        raise ValueError("Pca.evaluate: basis matrix must be provided")
    basis = np.asarray(spec.basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(
            f"Pca.basis must be 2D (D, K); got shape {tuple(basis.shape)}"
        )
    if basis.shape[0] != coords.shape[1]:
        raise ValueError(
            f"Pca: basis has D={basis.shape[0]} but coords has d={coords.shape[1]}"
        )
    if spec.centered:
        return (coords - coords.mean(axis=0, keepdims=True)) @ basis
    return coords @ basis


# ---------------------------------------------------------------------------
# Sphere descriptor evaluator
# ---------------------------------------------------------------------------


def sphere_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.Sphere` basis at ``(B, 2)`` coords.

    Routes through :func:`gamfit.torch._basis.sphere_basis`, which calls
    the Rust ``sphere_basis`` kernel. Forward only — the Rust binding does
    not expose a derivative basis through ``points``.
    """
    from .torch._basis import sphere_basis

    if coords.shape[1] != 2:
        raise ValueError(
            f"Sphere.evaluate expects (B, 2) coords [lat, lon]; "
            f"got d={coords.shape[1]}"
        )
    design, _penalty = sphere_basis(
        coords,
        n_centers=int(spec.n_centers),
        penalty_order=int(spec.penalty_order),
        kernel=str(spec.kernel),
        radians=bool(spec.radians),
    )
    return design
