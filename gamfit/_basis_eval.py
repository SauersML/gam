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
    """Autograd boundary for Duchon.

    Forward calls the Rust ``duchon_basis`` PyFFI. Backward uses the analytic
    Rust jet exposed by ``duchon_basis_with_jet`` when the descriptor is the
    plain m-spline (no ``length_scale``, no ``periodic_per_axis``). For the
    parameterized variants the closed-form jet isn't on the PyFFI surface
    today; backward then composes the Rust radial derivative kernel
    ``duchon_input_location_first_derivative`` with the closed-form
    ``(t − c) / r`` axis ratio (still all-Rust math).
    """
    torch = _torch()
    from . import _api

    kwargs: dict[str, Any] = {"m": int(m)}
    if periodic_per_axis is not None:
        kwargs["periodic_per_axis"] = tuple(bool(p) for p in periodic_per_axis)
    if length_scale is not None:
        kwargs["length_scale"] = float(length_scale)

    use_jet = periodic_per_axis is None and length_scale is None

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
            # Centers may be int / None for d=1 auto-quantile; resolve via Rust.
            d = pts_np.shape[1]
            if centers is None or isinstance(centers, int):
                if d != 1:
                    raise ValueError(
                        "Duchon backward: auto centers only supported for d=1"
                    )
                ctrs_np = _api._resolve_centers(centers, pts_np[:, 0], label="centers").reshape(-1, 1)
            else:
                ctrs_np = np.asarray(centers, dtype=float)
                if ctrs_np.ndim == 1:
                    ctrs_np = ctrs_np.reshape(-1, 1)

            if use_jet:
                # Analytic jet (B, M, d) from the Rust core in one call.
                _phi, jet_np, _pen = _api.rust_module().duchon_basis_with_jet(
                    pts_np, ctrs_np, int(m),
                )
                jet = torch.as_tensor(jet_np, dtype=pts.dtype, device=pts.device)
                # grad_pts[b, j] = sum_m g[b, m] * jet[b, m, j]
                grad_pts = torch.einsum("bm,bmj->bj", g, jet)
                return grad_pts

            # Parameterized path: combine Rust radial derivative φ'(r) with
            # the axis ratio (t − c)/r. Both pieces come from the Rust core.
            radial = _api.rust_module().duchon_input_location_first_derivative(
                pts_np, ctrs_np, length_scale, int(m),
            )  # (B, K)
            radial_np = np.asarray(radial, dtype=float)
            diffs = pts_np[:, None, :] - ctrs_np[None, :, :]  # (B, K, d)
            r = np.sqrt((diffs * diffs).sum(axis=-1) + 1e-300)  # (B, K)
            axis_ratio = diffs / r[..., None]  # (B, K, d)
            jet_np = radial_np[..., None] * axis_ratio
            jet = torch.as_tensor(jet_np, dtype=pts.dtype, device=pts.device)
            grad_pts = torch.einsum("bk,bkj->bj", g, jet)
            return grad_pts

    return _Fn.apply(points)


# ---------------------------------------------------------------------------
# Matern descriptor evaluator
# ---------------------------------------------------------------------------


def matern_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.Matern` kernel at ``(B, d)`` coords.

    Forward and backward both route through Rust:

    * forward: :func:`gamfit._api.matern_basis` (Rust ``matern_basis``);
    * backward: :func:`rust_module.matern_input_location_first_derivative`
      returns the radial derivative ``φ'(r)``, which we combine with the
      closed-form axis ratio ``(t − c) / r`` (standard tensor algebra,
      not kernel math) to assemble the per-axis jet.
    """
    if spec.centers is None:
        raise ValueError("Matern.evaluate: centers must be provided")
    return _MaternRustFn_apply(
        coords,
        spec.centers,
        float(spec.length_scale),
        _matern_nu_string(float(spec.nu)),
        spec.aniso_log_scales,
    )


def _matern_nu_string(nu: float) -> str:
    """Map float ν to the Rust-side half-integer string label."""
    if abs(nu - 0.5) < 1e-12:
        return "1/2"
    if abs(nu - 1.5) < 1e-12:
        return "3/2"
    if abs(nu - 2.5) < 1e-12:
        return "5/2"
    if abs(nu - 3.5) < 1e-12:
        return "7/2"
    if abs(nu - 4.5) < 1e-12:
        return "9/2"
    raise ValueError(
        f"Matern.nu={nu} is not a supported half-integer (1/2, 3/2, 5/2, 7/2, 9/2)"
    )


def _MaternRustFn_apply(
    points: Any,
    centers: Any,
    length_scale: float,
    nu: str,
    aniso_log_scales: Any,
) -> Any:
    """Autograd boundary for Matern. Forward + analytic backward via Rust."""
    torch = _torch()
    from . import _api

    aniso_arg = (
        None
        if aniso_log_scales is None
        else tuple(float(v) for v in aniso_log_scales)
    )

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, pts: Any) -> Any:
            import numpy as np

            pts_np = pts.detach().to(dtype=torch.float64).cpu().numpy()
            ctrs_np = np.asarray(centers, dtype=float)
            if ctrs_np.ndim == 1:
                ctrs_np = ctrs_np.reshape(-1, 1)
            basis_np = _api.matern_basis(
                pts_np,
                ctrs_np,
                length_scale=length_scale,
                nu=nu,
                aniso_log_scales=None if aniso_arg is None else list(aniso_arg),
            )
            basis = torch.as_tensor(basis_np, dtype=pts.dtype, device=pts.device)
            ctx.save_for_backward(pts)
            return basis

        @staticmethod
        def backward(ctx: Any, *grads: Any) -> Any:
            import numpy as np

            (g,) = grads
            (pts,) = ctx.saved_tensors
            pts_np = pts.detach().to(dtype=torch.float64).cpu().numpy()
            ctrs_np = np.asarray(centers, dtype=float)
            if ctrs_np.ndim == 1:
                ctrs_np = ctrs_np.reshape(-1, 1)
            radial = _api.rust_module().matern_input_location_first_derivative(
                pts_np, ctrs_np, float(length_scale), str(nu),
            )  # (B, K)
            radial_np = np.asarray(radial, dtype=float)
            diffs = pts_np[:, None, :] - ctrs_np[None, :, :]  # (B, K, d)
            r = np.sqrt((diffs * diffs).sum(axis=-1) + 1e-300)  # (B, K)
            axis_ratio = diffs / r[..., None]
            jet_np = radial_np[..., None] * axis_ratio  # (B, K, d)
            jet = torch.as_tensor(jet_np, dtype=pts.dtype, device=pts.device)
            grad_pts = torch.einsum("bk,bkj->bj", g, jet)
            return grad_pts

    return _Fn.apply(points)


def matern_evaluate_numpy(spec: Any, coords: Any) -> Any:
    """NumPy-backend Matern evaluation. Direct Rust ``matern_basis`` call."""
    import numpy as np

    from . import _api

    if spec.centers is None:
        raise ValueError("Matern.evaluate: centers must be provided")
    ctrs_np = np.asarray(spec.centers, dtype=float)
    if ctrs_np.ndim == 1:
        ctrs_np = ctrs_np.reshape(-1, 1)
    return np.asarray(
        _api.matern_basis(
            coords,
            ctrs_np,
            length_scale=float(spec.length_scale),
            nu=_matern_nu_string(float(spec.nu)),
            aniso_log_scales=spec.aniso_log_scales,
        ),
        dtype=np.float64,
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


def sphere_evaluate_numpy(spec: Any, coords: Any) -> Any:
    """NumPy-backend Sphere evaluation. Direct Rust ``sphere_basis``."""
    import numpy as np

    from . import _api

    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"Sphere.evaluate expects (B, 2) coords [lat, lon]; "
            f"got shape {tuple(pts.shape)}"
        )
    design, _ = _api.sphere_basis(
        pts,
        int(spec.n_centers),
        penalty_order=int(spec.penalty_order),
        kernel=str(spec.kernel),
        radians=bool(spec.radians),
    )
    return np.asarray(design, dtype=np.float64)


def periodic_curve_evaluate_numpy(spec: Any, coords: Any) -> Any:
    """NumPy-backend PeriodicSplineCurve evaluation. Direct Rust FFI."""
    import numpy as np

    from . import _api

    coords_np = np.asarray(coords, dtype=np.float64)
    if coords_np.ndim != 2 or coords_np.shape[1] != 1:
        raise ValueError(
            f"PeriodicSplineCurve.evaluate: 1D-only; got shape "
            f"{tuple(coords_np.shape)}"
        )
    basis_np, _ = _api.periodic_spline_curve_basis(
        coords_np[:, 0],
        int(spec.n_knots),
        degree=int(spec.degree),
        penalty_order=int(spec.penalty_order),
    )
    return np.asarray(basis_np, dtype=np.float64)
