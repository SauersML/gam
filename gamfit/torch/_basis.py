"""Differentiable basis, penalty, and closed-form ridge primitives for torch.

These wrappers mirror the NumPy entry points in :mod:`gamfit._api`.
``bspline_basis`` and ``duchon_basis`` carry an analytic backward with
respect to their evaluation locations through :class:`torch.autograd.Function`
subclasses (the Duchon backward contracts the upstream cotangent with the
input-location jets of the *built* design from the Rust
``duchon_basis_with_jets`` kernel, and supports second-order autograd). The
derivative, penalty, and closed-form ridge paths are forward-only and
produced via the detach-cast-call-numpy-wrap path in
:mod:`gamfit.torch._coerce`.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import torch

from .. import _api
from ._coerce import from_numpy_like, to_numpy_f64, to_numpy_uintp


def _resolve_centers_tensor(t: torch.Tensor, centers: Any) -> torch.Tensor:
    """Accept None / int / Tensor for Duchon centers; auto-derive when needed."""
    if isinstance(centers, torch.Tensor):
        return centers
    resolved = _api._resolve_centers(centers, to_numpy_f64(t), label="centers")
    return from_numpy_like(resolved, t)


def _resolve_knots_tensor(
    t: torch.Tensor, knots: Any, *, degree: int
) -> tuple[torch.Tensor, int]:
    """Accept None / int / Tensor for B-spline knots; auto-derive when needed.

    Returns ``(knots_tensor, effective_degree)``. Auto-knot derivation can
    downgrade the degree for small ``n`` (#340); the clamped knot vector is
    then built for ``effective_degree`` and must be evaluated at that degree,
    so callers must use the returned degree rather than the requested one. An
    explicit knot tensor passes the requested degree straight through.
    """
    if isinstance(knots, torch.Tensor):
        return knots, degree
    # Auto quantile knot placement delegates to Rust core, do not reimplement.
    resolved = _api._resolve_knots(knots, to_numpy_f64(t), label="knots", degree=degree)
    return from_numpy_like(resolved.locations, t), int(resolved.order)


class _BsplineBasisFn(torch.autograd.Function):
    """Autograd Function evaluating the Rust B-spline basis with grad wrt ``t``.

    Backward routes through :class:`_BsplineJetFn` (a second custom
    autograd Function) so the returned ``∂L/∂t`` is itself differentiable:
    a second autograd pass — the input-location Hessian — calls back into
    Rust via ``bspline_basis_derivative(order=2)`` for the open case.
    """

    @staticmethod
    def forward(
        ctx: Any, t: torch.Tensor, knots: torch.Tensor, degree: int, periodic: bool
    ) -> torch.Tensor:
        t_np = to_numpy_f64(t)
        knots_np = to_numpy_f64(knots)
        basis_np = _api.bspline_basis(t_np, knots_np, degree=degree, periodic=periodic)
        ctx.save_for_backward(t, knots)
        ctx.degree = degree
        ctx.periodic = periodic
        return from_numpy_like(basis_np, t)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (grad_basis,) = grad_outputs
        t, knots = ctx.saved_tensors
        jet = _BsplineJetFn.apply(t, knots, int(ctx.degree), bool(ctx.periodic))
        grad_t = (grad_basis.to(dtype=jet.dtype) * jet).sum(dim=-1)
        return grad_t, None, None, None


class _BsplineJetFn(torch.autograd.Function):
    """``∂Φ/∂t`` for the 1D B-spline basis as a tracked ``(N, K)`` tensor.

    Forward dispatches to the correct Rust derivative API:

    * ``periodic=False`` → ``bspline_basis_derivative(order=1, periodic=False)``;
    * ``periodic=True``  →
      ``periodic_bspline_input_location_first_derivative`` (the dense
      ``order=1`` API is intentionally not exposed for periodic bases —
      issue #233).

    Backward returns the input-location Hessian contraction. For the open
    case it calls ``bspline_basis_derivative(order=2)``; the periodic case
    has no Rust second-derivative API today, so the inner backward raises
    a clear ``NotImplementedError`` rather than leaking a Rust string.
    """

    @staticmethod
    def forward(
        ctx: Any, t: torch.Tensor, knots: torch.Tensor, degree: int, periodic: bool
    ) -> torch.Tensor:
        t_np = to_numpy_f64(t)
        knots_np = to_numpy_f64(knots)
        if periodic:
            from .._binding import rust_module

            left = float(knots_np[0])
            right = float(knots_np[-1])
            num_basis = int(knots_np.shape[0] - 1)
            jet_3d = rust_module().periodic_bspline_input_location_first_derivative(
                t_np.reshape(-1, 1), left, right, int(degree), num_basis,
            )
            # Rust returns (N, K, 1); reduce the trailing intrinsic-dim axis.
            jet_np = jet_3d.reshape(jet_3d.shape[0], jet_3d.shape[1])
        else:
            jet_np = _api.bspline_basis_derivative(
                t_np, knots_np, degree=int(degree), order=1, periodic=False,
            )
        ctx.save_for_backward(t, knots)
        ctx.degree = int(degree)
        ctx.periodic = bool(periodic)
        return from_numpy_like(jet_np, t)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (grad_jet,) = grad_outputs  # (N, K)
        t, knots = ctx.saved_tensors
        if ctx.periodic:
            raise NotImplementedError(
                "Second-order autograd through the periodic B-spline basis "
                "is not yet exposed; the Rust core needs a periodic "
                "input-location second-derivative kernel."
            )
        second_np = _api.bspline_basis_derivative(
            to_numpy_f64(t),
            to_numpy_f64(knots),
            degree=ctx.degree,
            order=2,
            periodic=False,
        )
        second = from_numpy_like(second_np, t)
        grad_t = (grad_jet.to(dtype=second.dtype) * second).sum(dim=-1)
        return grad_t, None, None, None


def _duchon_basis_kwargs(
    m: int,
    length_scale: float | None,
    periodic_per_axis: tuple[bool, ...] | None,
) -> dict[str, Any]:
    """Assemble the spec kwargs passed to the Rust Duchon forward / jet calls.

    Kept in one place so that the forward design and its analytic jets are
    always resolved from a bit-identical spec (nullspace order, power,
    length-scale, periodic chord embedding, amplification).
    """
    kwargs: dict[str, Any] = {"m": int(m)}
    if periodic_per_axis is not None:
        kwargs["periodic_per_axis"] = tuple(bool(p) for p in periodic_per_axis)
    if length_scale is not None:
        kwargs["length_scale"] = float(length_scale)
    return kwargs


def _duchon_design_jets(
    pts_np: Any, ctrs_np: Any, kwargs: dict[str, Any]
) -> tuple[Any, Any, Any]:
    """Forward design, first jet ``(B, M, d)``, second jet ``(B, M, d, d)``.

    A single Rust ``duchon_basis_with_jets`` call so all three are built from
    the *same* resolved spec — the jets are the exact input-location
    derivatives of the returned built design (not of the raw radial kernel).
    """
    import numpy as np

    phi_np, jet_np, hess_np = _api.rust_module().duchon_basis_with_jets(
        pts_np, ctrs_np, **kwargs
    )
    return (
        np.asarray(phi_np, dtype=float),
        np.asarray(jet_np, dtype=float),
        np.asarray(hess_np, dtype=float),
    )


class _DuchonJetFn(torch.autograd.Function):
    """Input-location first jet ``∂X/∂x`` of the built Duchon design.

    Forward returns ``(B, M, d)``. Backward contracts the upstream cotangent
    with the Rust second jet (Hessian) ``∂²X/∂x∂xᵀ`` so that a second
    autograd pass — the descriptor's ``hessian`` (autograd of ``jacobian``) —
    is exact rather than finite-differenced.
    """

    @staticmethod
    def forward(
        ctx: Any,
        points: torch.Tensor,
        centers: torch.Tensor,
        m: int,
        length_scale: float | None,
        periodic_per_axis: tuple[bool, ...] | None,
    ) -> torch.Tensor:
        pts_np = to_numpy_f64(points)
        ctrs_np = to_numpy_f64(centers)
        kwargs = _duchon_basis_kwargs(m, length_scale, periodic_per_axis)
        _phi, jet_np, _hess = _duchon_design_jets(pts_np, ctrs_np, kwargs)
        ctx.save_for_backward(points, centers)
        ctx.m = int(m)
        ctx.length_scale = length_scale
        ctx.periodic_per_axis = periodic_per_axis
        return from_numpy_like(jet_np, points)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        (grad_jet,) = grad_outputs  # (B, M, d)
        points, centers = ctx.saved_tensors
        pts_np = to_numpy_f64(points)
        ctrs_np = to_numpy_f64(centers)
        kwargs = _duchon_basis_kwargs(
            ctx.m, ctx.length_scale, ctx.periodic_per_axis
        )
        _phi, _jet, hess_np = _duchon_design_jets(pts_np, ctrs_np, kwargs)
        hess = from_numpy_like(hess_np, points)
        grad_pts = torch.einsum(
            "bmj,bmij->bi", grad_jet.to(dtype=hess.dtype), hess
        )
        return grad_pts, None, None, None, None


class _DuchonBasisFn(torch.autograd.Function):
    """Autograd Function evaluating the Rust multi-dim Duchon basis.

    The Rust Duchon *forward* design is **not** the raw centerwise radial
    kernel ``K(x, C)``; it is the built design

    ``X(x) = [ α · K(x, C) · Z ,  P(x) ]``,

    where ``Z = null(P(C)ᵀ)`` is the polynomial-constraint null space,
    ``P(x)`` the appended monomial nullspace columns, and ``α`` the kernel
    amplification factor. Backward differentiates **that** matrix,
    column-for-column, by contracting the upstream cotangent with the
    analytic first jet ``∂X/∂x`` from :func:`duchon_basis_with_jets`. The jet
    is routed through :class:`_DuchonJetFn` so that ``torch.autograd.grad`` of
    the Jacobian — the input-location Hessian — is itself well-defined via the
    Rust second jet. Gradients with respect to ``points`` are exact and
    analytic; ``centers`` and the structural spec carry no gradient.
    """

    @staticmethod
    def forward(
        ctx: Any,
        points: torch.Tensor,
        centers: torch.Tensor,
        m: int,
        length_scale: float | None,
        periodic_per_axis: tuple[bool, ...] | None,
    ) -> torch.Tensor:
        pts_np = to_numpy_f64(points)
        ctrs_np = to_numpy_f64(centers)
        kwargs = _duchon_basis_kwargs(m, length_scale, periodic_per_axis)
        basis_np = _api.duchon_basis(pts_np, ctrs_np, **kwargs)
        ctx.save_for_backward(points, centers)
        ctx.m = int(m)
        ctx.length_scale = length_scale
        ctx.periodic_per_axis = periodic_per_axis
        return from_numpy_like(basis_np, points)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        (grad_basis,) = grad_outputs  # (B, M)
        points, centers = ctx.saved_tensors
        # Jet is autograd-tracked through ``points`` so a second backward
        # (input-location Hessian) routes through ``_DuchonJetFn.backward``.
        jet = cast(Callable[..., torch.Tensor], _DuchonJetFn.apply)(
            points, centers, ctx.m, ctx.length_scale, ctx.periodic_per_axis
        )  # (B, M, d)
        grad_pts = torch.einsum(
            "bm,bmj->bj", grad_basis.to(dtype=jet.dtype), jet
        )
        return grad_pts, None, None, None, None


def bspline_basis(
    t: torch.Tensor,
    knots: Any = None,
    *,
    degree: int = 3,
    periodic: bool = False,
) -> torch.Tensor:
    """Evaluate the B-spline basis at ``t`` and route gradients back to ``t``.

    Parameters
    ----------
    t : torch.Tensor
        Evaluation locations, shape ``(n_t,)``. Differentiable input.
    knots : torch.Tensor | int | None, optional
        Knot vector. ``None`` (default) auto-derives a clamped knot
        vector with quantile-spaced interior knots from ``t``; an ``int``
        ``K`` overrides the interior-knot count. Treated as structural
        either way — no gradient is propagated.
    degree : int, optional
        Spline degree. Default ``3``.
    periodic : bool, optional
        Whether to evaluate the periodic variant. Default ``False``.

    Returns
    -------
    torch.Tensor
        Basis matrix of shape ``(n_t, n_basis)``.
    """
    knots_t, eff_degree = _resolve_knots_tensor(t, knots, degree=int(degree))
    apply = cast(Callable[..., torch.Tensor], _BsplineBasisFn.apply)
    return apply(t, knots_t, eff_degree, bool(periodic))


def bspline_basis_derivative(
    t: torch.Tensor,
    knots: Any = None,
    *,
    degree: int = 3,
    order: int = 1,
    periodic: bool = False,
) -> torch.Tensor:
    """Evaluate derivatives of the B-spline basis at ``t``.

    Forward-only: the returned tensor does not carry a backward through ``t``.
    Callers that need a differentiable basis should use ``bspline_basis`` and
    rely on autograd, since the derivative primitive has no analytic VJP in
    :mod:`gamfit._api`.

    Parameters
    ----------
    t : torch.Tensor
        Evaluation locations of shape ``(n_t,)``.
    knots : torch.Tensor
        Knot vector.
    degree : int, optional
        Spline degree. Default ``3``.
    order : int, optional
        Derivative order. Default ``1``.
    periodic : bool, optional
        Whether to evaluate the periodic variant. Default ``False``.

    Returns
    -------
    torch.Tensor
        Derivative basis matrix of shape ``(n_t, n_basis)``.
    """
    knots_t, eff_degree = _resolve_knots_tensor(t, knots, degree=int(degree))
    deriv = _api.bspline_basis_derivative(
        to_numpy_f64(t),
        to_numpy_f64(knots_t),
        degree=eff_degree,
        order=int(order),
        periodic=bool(periodic),
    )
    return from_numpy_like(deriv, t)


def duchon_basis(
    points: torch.Tensor,
    centers: Any = None,
    *,
    m: int = 2,
    length_scale: float | None = None,
    periodic_per_axis: tuple[bool, ...] | None = None,
) -> torch.Tensor:
    """Evaluate the Duchon m-spline basis at ``points``.

    Multi-dimensional: ``points`` is ``(N, d)``, ``centers`` is ``(K, d)``.
    For 1D, pass shape ``(N,)`` or ``(N, 1)`` — auto-promoted.

    This is the canonical, differentiable Torch entry point for the Duchon
    basis. Gradients with respect to ``points`` are exact and analytic: the
    backward contracts the upstream cotangent with the input-location first
    jet of the *built* design from the Rust ``duchon_basis_with_jets`` kernel,
    and a second autograd pass (the input-location Hessian) routes through the
    Rust second jet. ``centers`` and the structural spec carry no gradient.

    Parameters
    ----------
    points : torch.Tensor
        Evaluation locations, shape ``(N, d)`` or ``(N,)`` for d=1.
        Differentiable input.
    centers : torch.Tensor or int or None
        Center locations, shape ``(K, d)``. Auto-derived from ``points``
        for d=1 if None or an int. Treated as structural — no gradient is
        propagated through ``centers``.
    m : int, optional
        Duchon smoothness order. Default ``2``.
    length_scale : positive float or None, optional
        ``None`` (default) selects the scale-free pure Duchon spectrum; a
        positive value enables the hybrid (Matérn-blended) spectrum. Routed
        verbatim to the Rust forward and jet calls.
    periodic_per_axis : sequence of bool of length d, optional.
        Currently only d=1 supports periodicity.

    Returns
    -------
    torch.Tensor
        Basis matrix of shape ``(N, K)``.
    """
    if points.dim() == 1:
        points = points.unsqueeze(1)
    if points.dim() != 2:
        raise ValueError(f"points must be 1D or 2D, got {points.dim()}D")
    d = points.shape[1]
    # Resolve centers: 1D-tensor → promote; int/None → auto-quantile (d=1 only).
    if centers is None or isinstance(centers, int):
        if d != 1:
            raise ValueError(f"auto centers only supported for d=1, got d={d}")
        centers_t = _resolve_centers_tensor(points[:, 0], centers).unsqueeze(1)
    else:
        centers_t = centers if isinstance(centers, torch.Tensor) else torch.as_tensor(centers)
        if centers_t.dim() == 1:
            centers_t = centers_t.unsqueeze(1)
        if centers_t.dim() != 2 or centers_t.shape[1] != d:
            raise ValueError(
                f"centers must have shape (K, d={d}); got {tuple(centers_t.shape)}"
            )
    apply = cast(Callable[..., torch.Tensor], _DuchonBasisFn.apply)
    periodic_tuple = None if periodic_per_axis is None else tuple(bool(p) for p in periodic_per_axis)
    ls = None if length_scale is None else float(length_scale)
    return apply(points, centers_t, int(m), ls, periodic_tuple)


def sphere_basis(
    points: torch.Tensor,
    n_centers: int,
    *,
    penalty_order: int = 2,
    kernel: str = "sobolev",
    radians: bool = False,
    centers: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the spherical-spline (S²) design and penalty matrices.

    Forward-only: there is no analytic VJP through ``points``. The
    returned tensors are detached float64 copies on ``points``' device.

    When ``centers`` is supplied (the descriptor path), the basis
    dimension is fixed by ``centers.shape[0]`` and is independent of the
    number of rows in ``points``.

    Parameters
    ----------
    points : torch.Tensor of shape ``(N, 2)`` (latitude, longitude).
    n_centers : Wahba center count for ``kernel='sobolev' | 'pseudo'`` or
        truncation degree ``L`` for ``kernel='harmonic'``.
    penalty_order : roughness order ``m ∈ {1, 2, 3, 4}``. Default ``2``.
    kernel : one of ``'sobolev'``, ``'pseudo'``, ``'harmonic'``.
    radians : default ``False`` (degrees). True for radians.
    centers : optional ``(K, 2)`` array of pre-resolved centers.

    Returns
    -------
    (design (N, K), penalty (K, K)) as float64 torch tensors.
    """
    import numpy as np

    pts_np = to_numpy_f64(points)
    if centers is None:
        design_np, penalty_np = _api.sphere_basis(
            pts_np,
            int(n_centers),
            penalty_order=int(penalty_order),
            kernel=str(kernel),
            radians=bool(radians),
        )
    else:
        ctrs_np = np.ascontiguousarray(
            np.asarray(centers, dtype=np.float64)
        )
        design_np, penalty_np = _api.rust_module().sphere_basis_with_centers(
            pts_np,
            ctrs_np,
            int(penalty_order),
            str(kernel),
            bool(radians),
        )
    design = from_numpy_like(design_np, points).to(torch.float64)
    penalty = from_numpy_like(penalty_np, points).to(torch.float64)
    return design, penalty


def periodic_spline_curve_basis(
    t: torch.Tensor,
    n_knots: int,
    *,
    degree: int = 3,
    penalty_order: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the closed cyclic B-spline basis and its cyclic penalty.

    Forward-only: ``t`` is structural here (the basis is periodic uniform on
    ``[0, 1)``); no autograd path is exposed since gamfit's ``_api`` has no
    derivative primitive for this basis.

    Parameters
    ----------
    t : torch.Tensor of shape ``(N,)``. Values are reduced modulo 1.
    n_knots : number of cyclic control-point basis columns.
    degree : B-spline degree. Default 3.
    penalty_order : cyclic difference penalty order. Default 2.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``basis`` of shape ``(N, n_knots)`` and ``penalty`` of shape
        ``(n_knots, n_knots)``.
    """
    basis_np, penalty_np = _api.periodic_spline_curve_basis(
        to_numpy_f64(t),
        int(n_knots),
        degree=int(degree),
        penalty_order=int(penalty_order),
    )
    return from_numpy_like(basis_np, t).to(torch.float64), from_numpy_like(penalty_np, t).to(torch.float64)


def smoothness_penalty(
    knots: torch.Tensor,
    *,
    degree: int = 3,
    order: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the B-spline difference penalty and its null-space basis.

    Forward-only: ``knots`` is structural and the result has no autograd path.

    Parameters
    ----------
    knots : torch.Tensor
        Knot vector.
    degree : int, optional
        Spline degree. Default ``3``.
    order : int, optional
        Difference penalty order. Default ``2``.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``S`` of shape ``(M, M)`` and ``null_basis`` of shape ``(M, p)``.
    """
    s_np, null_np = _api.smoothness_penalty(
        to_numpy_f64(knots), degree=int(degree), order=int(order)
    )
    return from_numpy_like(s_np, knots), from_numpy_like(null_np, knots)


def gaussian_weighted_ridge(
    X: torch.Tensor,
    Y: torch.Tensor,
    penalty: torch.Tensor,
    weights: torch.Tensor,
    *,
    ridge_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form Gaussian row-weighted ridge solve.

    Forward-only: :mod:`gamfit._api` exposes no analytic VJP for this primitive,
    so the returned tensors carry no autograd path. ``weights`` are likelihood
    row weights, not a multiplicative gate on the design row.

    Parameters
    ----------
    X : torch.Tensor
        Design matrix of shape ``(N, M)``.
    Y : torch.Tensor
        Response matrix of shape ``(N, D)``.
    penalty : torch.Tensor
        Penalty matrix of shape ``(M, M)``.
    weights : torch.Tensor
        Row weights of shape ``(N,)``.
    ridge_lambda : float
        Ridge multiplier on ``penalty``.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``coefficients`` of shape ``(M, D)`` and ``fitted`` of shape ``(N, D)``.
    """
    coef_np, fit_np = _api.gaussian_weighted_ridge(
        to_numpy_f64(X),
        to_numpy_f64(Y),
        to_numpy_f64(penalty),
        to_numpy_f64(weights),
        ridge_lambda=float(ridge_lambda),
    )
    return from_numpy_like(coef_np, X), from_numpy_like(fit_np, X)


def gaussian_weighted_ridge_batch(
    X: torch.Tensor,
    Y: torch.Tensor,
    penalty: torch.Tensor,
    weights: torch.Tensor,
    *,
    ridge_lambda: float,
    row_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched closed-form Gaussian row-weighted ridge solve.

    Forward-only: no analytic VJP is exposed in :mod:`gamfit._api`. ``X`` has
    shape ``(K, Nmax, M)``, ``Y`` has shape ``(K, Nmax, D)``, ``weights`` has
    shape ``(K, Nmax)``, and ``row_counts`` optionally marks the active row
    prefix per problem in a padded ragged batch.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``coefficients`` of shape ``(K, M, D)`` and ``fitted`` of shape
        ``(K, Nmax, D)``.
    """
    coef_np, fit_np = _api.gaussian_weighted_ridge_batch(
        to_numpy_f64(X),
        to_numpy_f64(Y),
        to_numpy_f64(penalty),
        to_numpy_f64(weights),
        ridge_lambda=float(ridge_lambda),
        row_counts=None if row_counts is None else to_numpy_uintp(row_counts),
    )
    return from_numpy_like(coef_np, X), from_numpy_like(fit_np, X)
