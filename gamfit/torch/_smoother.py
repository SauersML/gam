"""1-D Duchon smoother with triple-operator penalty for :mod:`gamfit.torch`.

This is the open-interval companion to :class:`PeriodicSmoother`. It exposes
the same two-mode convention:

* ``mode="auto"`` (default) — gamfit selects the three smoothing weights
  internally on every forward pass; no smoothing parameter is exposed on the
  module and the caller does not have to think about ``λ``.
* ``mode="learned"`` — smoothing is part of the module's trainable state
  (one length-3 :class:`torch.nn.Parameter` named ``log_smoothing``, one entry
  per penalty operator). Embed the module inside a larger ``nn.Module`` and
  let the outer optimizer step it via the exposed ``smoothing_score`` on the
  fit output.

The basis is the 1-D Duchon-``m`` natural basis (radial kernel
:math:`r \\mapsto |r|^{2m-1}` plus polynomial null space of degree
:math:`< m`), evaluated through the analytic
:func:`gamfit.torch.duchon_basis_1d` primitive so the forward fit carries
gradients through ``t``.

The "triple operator" is the canonical shrinkage trio:

- ``S₂`` — Gram of :math:`f''`, the curvature penalty (the standard
  Duchon / thin-plate-spline term)
- ``S₁`` — Gram of :math:`f'`, the slope penalty
- ``S₀`` — Gram of :math:`f`, the value (L²) penalty

Each is a positive-semidefinite matrix obtained by Gauss–Legendre quadrature
of the analytically-known basis derivatives over the smoother's domain.

For users who just want the fixed-``λ`` ridge solve under any penalty,
:func:`penalized_ridge_solve` exposes the underlying primitive directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from ._cyclic_duchon import PeriodicFitOutput, _SmootherBase

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = ["DuchonSmoother", "penalized_ridge_solve"]


# --------------------------------------------------------------------------- #
# Low-level fixed-``λ`` penalized ridge solve. Differentiable through every
# argument via :func:`torch.linalg.solve`; useful when the smoothing weights
# are user-supplied (or part of a larger learnable module).
# --------------------------------------------------------------------------- #


def penalized_ridge_solve(
    x: torch.Tensor,
    y: torch.Tensor,
    penalty: torch.Tensor,
    lam: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable fixed-``λ`` penalized ridge solve.

    Solves ``(XᵀWX + λ·S) β = XᵀWy`` and returns ``(β, X β)``. Gradients flow
    into every argument including ``lam`` and ``penalty`` via
    :func:`torch.linalg.solve`.

    Parameters
    ----------
    x : torch.Tensor
        Design matrix of shape ``(N, M)``.
    y : torch.Tensor
        Response of shape ``(N, D)`` or ``(N,)`` (the latter is treated as
        ``(N, 1)``).
    penalty : torch.Tensor
        Symmetric penalty matrix of shape ``(M, M)``.
    lam : torch.Tensor
        Scalar smoothing multiplier on ``penalty``. Must be non-negative.
    weights : torch.Tensor or None, optional
        Optional row weights of shape ``(N,)``. ``None`` is unit weights.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``coefficients`` of shape ``(M, D)`` and ``fitted = X β`` of shape
        ``(N, D)``.
    """
    squeeze_out = y.ndim == 1
    y_mat = y.unsqueeze(1) if squeeze_out else y
    if weights is None:
        xtx = x.t() @ x
        xty = x.t() @ y_mat
    else:
        wx = weights.unsqueeze(1) * x
        xtx = wx.t() @ x
        xty = wx.t() @ y_mat
    hessian = xtx + lam * penalty
    coefficients = torch.linalg.solve(hessian, xty)
    fitted = x @ coefficients
    if squeeze_out:
        coefficients = coefficients.squeeze(1)
        fitted = fitted.squeeze(1)
    return coefficients, fitted


# --------------------------------------------------------------------------- #
# Closed-form Duchon-m=2 derivatives.
#
# Radial basis function: η(r) = |r|^3 / 12.
#   η'(r)  = r · |r| / 4
#   η''(r) = |r| / 2
# Polynomial null space: {1, t}; their first and second derivatives are
# {0, 1} and {0, 0} respectively.
# --------------------------------------------------------------------------- #


def _duchon_basis_value(t: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    radial = (t.unsqueeze(1) - centers.unsqueeze(0)).abs().pow(3) / 12.0
    ones = torch.ones_like(t).unsqueeze(1)
    return torch.cat([radial, ones, t.unsqueeze(1)], dim=1)


def _duchon_basis_first_derivative(
    t: torch.Tensor, centers: torch.Tensor
) -> torch.Tensor:
    diff = t.unsqueeze(1) - centers.unsqueeze(0)
    radial = diff * diff.abs() / 4.0
    zeros = torch.zeros_like(t).unsqueeze(1)
    ones = torch.ones_like(t).unsqueeze(1)
    return torch.cat([radial, zeros, ones], dim=1)


def _duchon_basis_second_derivative(
    t: torch.Tensor, centers: torch.Tensor
) -> torch.Tensor:
    radial = (t.unsqueeze(1) - centers.unsqueeze(0)).abs() / 2.0
    zeros = torch.zeros_like(t).unsqueeze(1)
    return torch.cat([radial, zeros, zeros], dim=1)


_BasisFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _quadrature_gram(
    basis_fn: _BasisFn,
    centers: torch.Tensor,
    lo: float,
    hi: float,
    n_quad: int,
) -> torch.Tensor:
    """Gauss–Legendre quadrature of ``∫ Bᵀ(t) B(t) dt`` on ``[lo, hi]``."""
    dtype = centers.dtype
    device = centers.device
    nodes_ref, weights_ref = _gauss_legendre(n_quad, dtype=dtype, device=device)
    # Map [-1, 1] -> [lo, hi].
    half_width = (hi - lo) / 2.0
    midpoint = (hi + lo) / 2.0
    nodes = midpoint + half_width * nodes_ref
    quad_weights = half_width * weights_ref
    basis = basis_fn(nodes, centers)
    weighted = basis * quad_weights.unsqueeze(1)
    result: torch.Tensor = weighted.t() @ basis
    return result


def _gauss_legendre(
    n: int, *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nodes and weights for Gauss–Legendre quadrature on ``[-1, 1]``.

    Uses the Golub–Welsch eigenvalue construction so the result is fully
    deterministic and works for any ``n >= 1`` without depending on numpy.
    """
    if n < 2:
        raise ValueError(f"n_quad must be >= 2, got {n}")
    k = torch.arange(1, n, dtype=dtype, device=device)
    beta = k / torch.sqrt(4.0 * k * k - 1.0)
    jacobi = torch.diag(beta, diagonal=1) + torch.diag(beta, diagonal=-1)
    eigvals, eigvecs = torch.linalg.eigh(jacobi)
    weights = 2.0 * eigvecs[0, :].pow(2)
    return eigvals, weights


# --------------------------------------------------------------------------- #
# Public smoother.
# --------------------------------------------------------------------------- #


_DEFAULT_N_CENTERS: int = 16
_DEFAULT_N_QUAD: int = 96


class DuchonSmoother(_SmootherBase):
    """1-D Duchon smoother with a triple-operator penalty.

    Two modes:

    - ``mode="auto"`` (default): each call to ``forward(t, y)`` fits the
      smoother and returns the resulting curve. The three smoothing weights
      are selected internally — no smoothing parameter is exposed on the
      module.
    - ``mode="learned"``: the three smoothing weights are a single length-3
      :class:`torch.nn.Parameter` named ``log_smoothing``. Embed the module
      inside a larger ``nn.Module`` and let an outer optimizer drive it via
      ``out.smoothing_score``.

    The penalty is a positive linear combination of three operators on the
    smoother's domain ``[domain_lo, domain_hi]``:

    - ``S₂`` — :math:`\\int (f'')^2`, the curvature penalty
    - ``S₁`` — :math:`\\int (f')^2`, the slope penalty
    - ``S₀`` — :math:`\\int f^2`, the value (L²) penalty

    Each is a positive-semidefinite Gram matrix obtained from closed-form
    basis derivatives integrated by Gauss–Legendre quadrature over the
    domain.

    Parameters
    ----------
    domain : tuple of float
        Smoother domain ``(domain_lo, domain_hi)`` (``domain_lo < domain_hi``).
        The penalty Gram matrices are integrated over this interval.
    n_centers : int, optional
        Number of equispaced radial centers placed across the domain. Ignored
        if ``centers`` is supplied. Default ``16``.
    centers : torch.Tensor, optional
        Explicit center locations. Must be 1-D with at least ``2`` entries.
    mode : str, optional
        ``"auto"`` (default) or ``"learned"``. See above.
    init_log_smoothing : tuple of three floats, optional
        Initial ``log λ`` for the ``(S₂, S₁, S₀)`` penalties, in that order.
        Default ``(0.0, 0.0, 0.0)``.
    n_quad : int, optional
        Number of Gauss–Legendre quadrature points used to build the penalty
        Gram matrices at construction time. Default ``96``.

    Examples
    --------
    Automatic smoothing — black box, no tuning required:

    >>> import torch, gamfit.torch as gt
    >>> sm = gt.DuchonSmoother(domain=(0.0, 1.0))
    >>> x = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    >>> y = torch.sin(2 * torch.pi * x)
    >>> out = sm(x, y)

    Trainable inside a learning loop — three smoothing knobs:

    >>> sm = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned")
    >>> opt = torch.optim.Adam(sm.parameters(), lr=0.1)
    >>> for _ in range(50):
    ...     opt.zero_grad()
    ...     out = sm(x, y)
    ...     torch.autograd.backward(-out.smoothing_score)
    ...     opt.step()
    """

    log_smoothing: torch.nn.Parameter | None

    def __init__(
        self,
        domain: tuple[float, float],
        *,
        n_centers: int = _DEFAULT_N_CENTERS,
        centers: torch.Tensor | None = None,
        mode: str = "auto",
        init_log_smoothing: Iterable[float] = (0.0, 0.0, 0.0),
        n_quad: int = _DEFAULT_N_QUAD,
    ) -> None:
        domain_lo, domain_hi = float(domain[0]), float(domain[1])
        if not (domain_lo < domain_hi):
            raise ValueError(
                f"domain must satisfy lo < hi, got ({domain_lo}, {domain_hi})"
            )

        dtype = torch.get_default_dtype()
        if centers is None:
            if n_centers < 2:
                raise ValueError(f"n_centers must be >= 2, got {n_centers}")
            centers_t = torch.linspace(
                domain_lo, domain_hi, n_centers, dtype=dtype
            )
        else:
            centers_t = centers.detach().to(dtype=dtype)
            if centers_t.ndim != 1 or centers_t.shape[0] < 2:
                raise ValueError(
                    f"centers must be a 1-D tensor with at least 2 entries, "
                    f"got shape {tuple(centers_t.shape)}"
                )

        init = tuple(float(v) for v in init_log_smoothing)
        if len(init) != 3:
            raise ValueError(
                f"init_log_smoothing must have exactly 3 entries (S₂, S₁, S₀), "
                f"got {len(init)}"
            )

        super().__init__(
            mode=mode,
            n_components=3,
            init_log_smoothing=init,
        )

        self._domain_lo = domain_lo
        self._domain_hi = domain_hi
        self._n_quad = int(n_quad)

        # Compute the triple penalty Gram matrices once at construction. They
        # depend only on centers and the domain, both fixed, so they are
        # buffers (no autograd, move with the module).
        s_curv = _quadrature_gram(
            _duchon_basis_second_derivative, centers_t, domain_lo, domain_hi, self._n_quad
        )
        s_slope = _quadrature_gram(
            _duchon_basis_first_derivative, centers_t, domain_lo, domain_hi, self._n_quad
        )
        s_val = _quadrature_gram(
            _duchon_basis_value, centers_t, domain_lo, domain_hi, self._n_quad
        )
        # Symmetrize numerically (quadrature gives a symmetric matrix up to
        # round-off; this enforces it exactly so downstream Cholesky stays
        # happy).
        for s in (s_curv, s_slope, s_val):
            s.copy_(0.5 * (s + s.t()))

        self.register_buffer("centers", centers_t)
        self.register_buffer("_S_curv", s_curv)
        self.register_buffer("_S_slope", s_slope)
        self.register_buffer("_S_val", s_val)

    @property
    def domain(self) -> tuple[float, float]:
        return (self._domain_lo, self._domain_hi)

    @property
    def n_quad(self) -> int:
        return self._n_quad

    def _build_basis(self, t: torch.Tensor) -> torch.Tensor:
        # In-torch evaluation of the natural Duchon-2 basis. Keeping the basis
        # evaluator in torch (instead of routing through the reduced gamfit
        # primitive ``duchon_basis_1d``) means the penalty Gram matrices
        # computed by quadrature live in the same column layout as the design
        # matrix: ``k`` radial columns followed by ``1`` and ``t`` polynomial
        # null space columns.
        centers: torch.Tensor = self.get_buffer("centers")
        return _duchon_basis_value(t, centers)

    def _penalty_components(self) -> tuple[torch.Tensor, ...]:
        return (
            self.get_buffer("_S_curv"),
            self.get_buffer("_S_slope"),
            self.get_buffer("_S_val"),
        )

    def fit(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
    ) -> PeriodicFitOutput:
        """Alias for ``self(t, y, weights=weights)``."""
        result: PeriodicFitOutput = self(t, y, weights=weights)
        return result
