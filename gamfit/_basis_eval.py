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


def _ensure_bspline_knots(spec: Any, t_np: Any) -> Any:
    """Resolve `spec.knots` and cache the resolved array back onto the spec.

    The descriptor advertises `knots=None` and `knots=K` as auto-derived,
    but every backend (numpy, torch, jax) needs the *same* resolved knot
    vector. Resolving on first evaluation and writing back guarantees:

      * subsequent evaluations are deterministic regardless of which `t`
        is passed (the first one wins, like a fit/transform contract).
      * `bspline_basis_size` (and therefore the JAX static-shape path)
        works after one `.evaluate(x)` call, as the docstring promises.

    Returns the resolved knot array.
    """
    import numpy as np

    from . import _api

    knots = spec.knots
    if knots is None or isinstance(knots, int):
        resolved = _api._resolve_knots(knots, t_np, label="knots", degree=int(spec.degree))
        # Cache the resolved knot vector AND the effective degree: auto-knot
        # derivation may downgrade the degree for small n (#340), and the
        # clamped vector's boundary multiplicity is `order + 1`, so the cached
        # spec must carry the matching degree for every later evaluate()/
        # basis_size call to stay consistent with the knots.
        spec.knots = np.asarray(resolved.locations, dtype=np.float64)
        spec.degree = int(resolved.order)
    return spec.knots


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
    import numpy as np

    t_np = np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t, dtype=np.float64)
    _ensure_bspline_knots(spec, t_np)
    return bspline_basis(
        t, spec.knots, degree=int(spec.degree), periodic=bool(spec.periodic),
    )


def bspline_basis_size(spec: Any) -> int:
    """Compute n_basis from spec.knots once knots are resolved.

    Knots are auto-resolved and cached by the first `evaluate()` call; that
    contract is documented on `BSpline.knots` and the JAX path depends on
    it for static output shape.
    """
    knots = spec.knots
    if knots is None or isinstance(knots, int):
        raise ValueError(
            "BSpline.basis_size: knots not resolved. Call `.evaluate(x)` "
            "once (any backend) — that resolves knots from x and caches "
            "them on the spec — or set `spec.knots` to an explicit array."
        )
    import numpy as np

    arr = np.asarray(knots)
    if spec.periodic:
        # The Rust periodic evaluator (`bspline_basis(..., periodic=True)`)
        # builds the cyclic basis on the knot-interval lattice: one basis
        # function per interval of the closed domain, i.e. len(knots) - 1
        # columns regardless of degree. This is the single periodic-basis
        # cardinality; the cyclic derivative Gram is built at the same
        # width in the torch fit path.
        return int(arr.size - 1)
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

    Thin descriptor adapter over the canonical Torch Duchon primitive
    :func:`gamfit.torch._basis.duchon_basis`, which owns the autograd engine:
    Rust ``duchon_basis`` forward plus an analytic backward through ``coords``
    (and second-order autograd) built from the input-location jets of the
    *built* design via the Rust ``duchon_basis_with_jets`` kernel. The spec's
    ``length_scale`` and ``periodic_per_axis`` are forwarded verbatim; the
    Rust kernel is the single source of math truth.
    """
    from .torch._basis import duchon_basis

    centers = spec.centers
    if (centers is None or isinstance(centers, int)) and coords.shape[1] != 1:
        raise ValueError(
            "Duchon.evaluate: auto centers only supported for d=1; "
            "provide explicit centers for d>=2."
        )
    return duchon_basis(
        coords,
        centers,
        m=int(spec.m),
        length_scale=None if spec.length_scale is None else float(spec.length_scale),
        periodic_per_axis=spec.periodic_per_axis,
    )


# ---------------------------------------------------------------------------
# Matern descriptor evaluator
# ---------------------------------------------------------------------------


def matern_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate a :class:`gamfit.Matern` kernel at ``(B, d)`` coords.

    Forward and backward both route through Rust:

    * forward: :func:`gamfit._api.matern_basis` (Rust ``matern_basis``);
    * backward: :func:`rust_module.matern_input_location_first_jet` returns the
      full per-axis jet ``∂Φ/∂t`` under the *anisotropic* metric
      ``r_A = √(Σ_a w_a (t_a − c_a)²)`` (same centred-contrast weights the
      forward applies), and :func:`rust_module.matern_input_location_hessian`
      returns the metric-aware input-location Hessian for second-order
      autograd. All kernel math (metric weights, ``φ'``/``φ''``, the
      diagonal-plus-rank-1 assembly) lives in Rust; Python only contracts the
      returned tensors with upstream gradients.
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


def _MaternJetFn_apply(
    points: Any,
    resolve_centers: Any,
    jet_fn: Any,
    hessian_fn: Any,
) -> Any:
    """Autograd-aware Matern input-location jet ``∂Φ/∂t`` (anisotropy-aware).

    Forward returns the full per-row jet ``(B, K, d)`` computed in Rust under
    the anisotropic metric ``r_A = √(Σ_a w_a (t_a − c_a)²)`` — i.e. the exact
    first derivative of the forward Matern kernel value. Backward contracts the
    full Rust-built input-location Hessian ``(B, K, d, d)`` (also metric-aware)
    against the upstream cotangent, so second-order autograd (descriptor
    ``.hessian()``) differentiates the *same* anisotropic function as the
    forward. All kernel math (metric weights, ``φ'``/``φ''``, the
    diagonal-plus-rank-1 assembly) lives in Rust; this wrapper only routes
    tensors through autograd.
    """
    torch = _torch()

    class _JetFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, pts: Any) -> Any:
            pts_np = pts.detach().cpu().to(dtype=torch.float64).numpy()
            ctrs_np = resolve_centers(pts_np)
            jet_np = jet_fn(pts_np, ctrs_np)  # (B, K, d)
            jet = torch.as_tensor(jet_np, dtype=pts.dtype, device=pts.device)
            ctx.save_for_backward(pts)
            return jet

        @staticmethod
        def backward(ctx: Any, *grads: Any) -> Any:
            (grad_jet,) = grads  # (B, K, d)
            (pts,) = ctx.saved_tensors
            pts_np = pts.detach().cpu().to(dtype=torch.float64).numpy()
            ctrs_np = resolve_centers(pts_np)
            hess_np = hessian_fn(pts_np, ctrs_np)  # (B, K, d, d)
            hess = torch.as_tensor(hess_np, dtype=pts.dtype, device=pts.device)
            grad_pts = torch.einsum(
                "bkj,bkij->bi", grad_jet.to(dtype=hess.dtype), hess
            )
            return grad_pts

    return _JetFn.apply(points)


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

    def _resolve_centers_np(_pts_np: Any) -> Any:
        import numpy as np

        ctrs_np = np.asarray(centers, dtype=float)
        if ctrs_np.ndim == 1:
            ctrs_np = ctrs_np.reshape(-1, 1)
        return ctrs_np

    def _aniso_np() -> Any:
        import numpy as np

        if aniso_arg is None:
            return None
        return np.asarray(aniso_arg, dtype=np.float64)

    def _input_jet(pts_np: Any, ctrs_np: Any) -> Any:
        import numpy as np

        out = _api.rust_module().matern_input_location_first_jet(
            pts_np, ctrs_np, float(length_scale), str(nu), _aniso_np(),
        )
        return np.asarray(out, dtype=float)

    def _input_hessian(pts_np: Any, ctrs_np: Any) -> Any:
        import numpy as np

        out = _api.rust_module().matern_input_location_hessian(
            pts_np, ctrs_np, float(length_scale), str(nu), _aniso_np(),
        )
        return np.asarray(out, dtype=float)

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, pts: Any) -> Any:
            pts_np = pts.detach().cpu().to(dtype=torch.float64).numpy()
            ctrs_np = _resolve_centers_np(pts_np)
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
            (g,) = grads
            (pts,) = ctx.saved_tensors
            jet = _MaternJetFn_apply(
                pts, _resolve_centers_np, _input_jet, _input_hessian
            )  # (B, K, d), autograd-tracked through pts
            grad_pts = torch.einsum("bk,bkj->bj", g.to(dtype=jet.dtype), jet)
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


def pca_basis_matrix(spec: Any) -> Any:
    """Validated ``(D, K_eff)`` Pca projection matrix, truncated to ``spec.K``.

    The one definition of "the matrix Pca projects through": explicit ``K``
    keeps the first ``K`` basis columns, exactly as the torch fit path does,
    so ``basis_size``, callable evaluation, and fitting all denote the same
    model.
    """
    import numpy as np

    if spec.basis is None:
        raise ValueError("Pca.evaluate: basis matrix must be provided")
    basis = np.asarray(spec.basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError(
            f"Pca.basis must be 2D (D, K); got shape {tuple(basis.shape)}"
        )
    if spec.K is not None:
        k = int(spec.K)
        if not 1 <= k <= basis.shape[1]:
            raise ValueError(
                f"Pca: K={k} must lie in 1..{basis.shape[1]} (the basis width)"
            )
        basis = basis[:, :k]
    return basis


def pca_training_mean(spec: Any, coords_np: Any) -> Any:
    """Resolve the ``centered=True`` training mean once; cache it on the spec.

    Mirrors :func:`_ensure_bspline_knots`: the first evaluation is the
    fit/transform boundary — its rows define the training mean — and every
    later call reuses the cached value, so the projection is a fixed affine
    map. Centering each evaluation batch on its own mean would make one
    row's features depend on unrelated rows in the same call and collapse
    every single-row prediction of a training row to zero.
    """
    import numpy as np

    if spec.mean is None:
        spec.mean = np.asarray(coords_np, dtype=np.float64).mean(axis=0)
    mean = np.asarray(spec.mean, dtype=np.float64).reshape(-1)
    if mean.shape[0] != np.asarray(coords_np).shape[1]:
        raise ValueError(
            f"Pca: cached mean has D={mean.shape[0]} but coords has "
            f"d={np.asarray(coords_np).shape[1]}"
        )
    return mean


def pca_evaluate(spec: Any, coords: Any) -> Any:
    """Evaluate the Pca smooth as a fixed linear projection.

    The PCA basis is a precomputed ``(D, K)`` projection matrix supplied
    by the user (no gamfit-specific math kernel exists for it — the
    ``Φ(x) = (x − μ) · basis`` operation is standard gemm). Returned
    tensor carries autograd back to ``coords``; ``μ`` is the persisted
    training mean, a constant of the map, so rows stay independent.
    """
    torch = _torch()
    basis_t = torch.as_tensor(pca_basis_matrix(spec))
    basis_t = basis_t.to(dtype=coords.dtype, device=coords.device)
    if basis_t.shape[0] != coords.shape[1]:
        raise ValueError(
            f"Pca: basis has D={basis_t.shape[0]} but coords has d={coords.shape[1]}"
        )
    if spec.centered:
        coords_np = (
            coords.detach().cpu().to(dtype=torch.float64).numpy()
            if hasattr(coords, "detach")
            else coords
        )
        mean = torch.as_tensor(
            pca_training_mean(spec, coords_np),
            dtype=coords.dtype,
            device=coords.device,
        ).reshape(1, -1)
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
    import numpy as np

    out = None
    for j, marg in enumerate(marginals):
        x = coords[:, j]
        x_np = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=np.float64)
        _ensure_bspline_knots(marg, x_np)
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
    t_np = np.asarray(coords[:, 0], dtype=np.float64)
    _ensure_bspline_knots(spec, t_np)
    return np.asarray(
        _api.bspline_basis(
            t_np,
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
        x_np = np.asarray(coords[:, j], dtype=np.float64)
        _ensure_bspline_knots(marg, x_np)
        col = np.asarray(
            _api.bspline_basis(
                x_np,
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
    """NumPy-backend Pca evaluation (standard linear projection).

    Uses the same persisted training mean and ``K``-truncated basis as the
    torch path, so both backends evaluate one fixed affine map.
    """
    basis = pca_basis_matrix(spec)
    if basis.shape[0] != coords.shape[1]:
        raise ValueError(
            f"Pca: basis has D={basis.shape[0]} but coords has d={coords.shape[1]}"
        )
    if spec.centered:
        return (coords - pca_training_mean(spec, coords).reshape(1, -1)) @ basis
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
    centers = spec._resolve_centers(coords)
    design, _penalty = sphere_basis(
        coords,
        n_centers=int(spec.n_centers),
        penalty_order=int(spec.penalty_order),
        kernel=str(spec.kernel),
        radians=bool(spec.radians),
        centers=centers,
    )
    return design


def sphere_evaluate_numpy(spec: Any, coords: Any) -> Any:
    """NumPy-backend Sphere evaluation.

    For Wahba kernels (``sobolev``/``pseudo``) this routes through
    ``sphere_basis_with_centers`` so the basis dimension is fixed by the
    descriptor's resolved center set rather than by the evaluation row
    count. For ``harmonic`` (eigen) basis, no centers are needed.
    """
    import numpy as np

    from . import _api

    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"Sphere.evaluate expects (B, 2) coords [lat, lon]; "
            f"got shape {tuple(pts.shape)}"
        )
    centers = spec._resolve_centers(pts)
    if centers is None:
        design, _ = _api.sphere_basis(
            pts,
            int(spec.n_centers),
            penalty_order=int(spec.penalty_order),
            kernel=str(spec.kernel),
            radians=bool(spec.radians),
        )
    else:
        design, _ = _api.rust_module().sphere_basis_with_centers(
            pts,
            np.ascontiguousarray(centers, dtype=np.float64),
            int(spec.penalty_order),
            str(spec.kernel),
            bool(spec.radians),
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
