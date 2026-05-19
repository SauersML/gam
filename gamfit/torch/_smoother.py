"""1-D Duchon smoother for :mod:`gamfit.torch`.

User-facing surface — two modes, set with the ``mode`` keyword:

* ``mode="auto"`` (default) — gamfit handles smoothing internally on every
  forward pass. The smoother has no learnable parameters; the caller does not
  enumerate, index, or tune anything.
* ``mode="learned"`` — gamfit holds smoothing as part of the smoother's
  trainable state. The smoother behaves like any other :class:`torch.nn.Module`
  with parameters: drop it into a larger model and pass
  ``smoother.parameters()`` to your optimizer. You never touch those
  parameters directly.

Both modes share the same forward signature ``smoother(t, y, weights=None)``
returning ``(coefficients, fitted, smoothing_score)``.

For users who just want a fixed-``λ`` penalized ridge solve as a plain
callable, :func:`penalized_ridge_solve` is exposed separately.

---

Implementation note (not part of the user contract). Internally the smoother
regularizes with three operator penalties — mass (the value/L² operator),
tension (the slope/first-derivative operator), and stiffness (the
curvature/second-derivative operator) — each with its own strength. In
``"learned"`` mode those strengths live as :class:`torch.nn.Parameter`\\s on
the module so the outer optimizer can drive them. The number, names, and
parameterisation of those internals are not part of the public surface and
may change between versions.
"""

from __future__ import annotations

from typing import Callable

import torch

from ._cyclic_duchon import PeriodicFitOutput, _SmootherBase


__all__ = ["DuchonSmoother", "penalized_ridge_solve"]


# --------------------------------------------------------------------------- #
# Low-level fixed-``λ`` penalized ridge solve. Differentiable through every
# argument via :func:`torch.linalg.solve`.
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
# Closed-form Duchon-m=2 derivatives (internal).
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

    Golub–Welsch construction — fully deterministic, works for any ``n >= 2``
    without depending on numpy.
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
    """1-D Duchon smoother on a bounded interval.

    Two modes:

    - ``mode="auto"`` (default): on every ``forward(t, y)`` gamfit picks the
      smoothing internally and returns the fitted curve. The smoother has no
      learnable parameters.
    - ``mode="learned"``: gamfit holds smoothing as part of the smoother's
      trainable state. Embed the module inside a larger ``nn.Module`` and
      pass ``smoother.parameters()`` to your optimizer; you never touch the
      smoothing parameters directly.

    Parameters
    ----------
    domain : tuple of float
        ``(domain_lo, domain_hi)`` with ``domain_lo < domain_hi``. The
        smoother is parameterised over this interval.
    n_centers : int, optional
        Number of equispaced radial centers placed across the domain. Ignored
        if ``centers`` is supplied. Default ``16``.
    centers : torch.Tensor, optional
        Explicit center locations. Must be 1-D with at least ``2`` entries.
    mode : str, optional
        ``"auto"`` (default) or ``"learned"``.
    n_quad : int, optional
        Internal quadrature resolution used to set up the smoother at
        construction time. Default ``96``; safe to leave alone.

    Examples
    --------
    Automatic smoothing — no smoothing parameter to think about:

    >>> import torch, gamfit.torch as gt
    >>> sm = gt.DuchonSmoother(domain=(0.0, 1.0))
    >>> x = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    >>> y = torch.sin(2 * torch.pi * x)
    >>> out = sm(x, y)

    Trainable inside a larger learning loop — just hand ``parameters()`` to
    your optimizer:

    >>> sm = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned")
    >>> opt = torch.optim.Adam(sm.parameters(), lr=0.1)
    >>> for _ in range(50):
    ...     opt.zero_grad()
    ...     out = sm(x, y)
    ...     torch.autograd.backward(-out.smoothing_score)
    ...     opt.step()
    """

    # Internal note: the implementation parameterises smoothing as three
    # operator strengths (mass / tension / stiffness). Both the count and the
    # interpretation are implementation details — they live on the module so
    # ``parameters()`` finds them, but the user never indexes or names them.

    def __init__(
        self,
        domain: tuple[float, float],
        *,
        n_centers: int = _DEFAULT_N_CENTERS,
        centers: torch.Tensor | None = None,
        mode: str = "auto",
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

        # Three internal operator strengths — see implementation note.
        # Initialised at log(1) = 0; the auto mode replaces this every forward,
        # the learned mode trains it.
        super().__init__(
            mode=mode,
            n_components=3,
            init_log_smoothing=(0.0, 0.0, 0.0),
        )

        self._domain_lo = domain_lo
        self._domain_hi = domain_hi
        self._n_quad = int(n_quad)

        # Triple penalty Gram matrices (mass / tension / stiffness): built
        # once at construction since they depend only on centers and domain.
        # Registered as buffers so they ride along on .to(device)/.float()/etc.
        s_stiffness = _quadrature_gram(
            _duchon_basis_second_derivative,
            centers_t,
            domain_lo,
            domain_hi,
            self._n_quad,
        )
        s_tension = _quadrature_gram(
            _duchon_basis_first_derivative,
            centers_t,
            domain_lo,
            domain_hi,
            self._n_quad,
        )
        s_mass = _quadrature_gram(
            _duchon_basis_value,
            centers_t,
            domain_lo,
            domain_hi,
            self._n_quad,
        )
        for s in (s_stiffness, s_tension, s_mass):
            s.copy_(0.5 * (s + s.t()))

        self.register_buffer("centers", centers_t)
        self.register_buffer("_S_stiffness", s_stiffness)
        self.register_buffer("_S_tension", s_tension)
        self.register_buffer("_S_mass", s_mass)

    @property
    def domain(self) -> tuple[float, float]:
        return (self._domain_lo, self._domain_hi)

    def _build_basis(self, t: torch.Tensor) -> torch.Tensor:
        centers: torch.Tensor = self.get_buffer("centers")
        return _duchon_basis_value(t, centers)

    def _penalty_components(self) -> tuple[torch.Tensor, ...]:
        return (
            self.get_buffer("_S_stiffness"),
            self.get_buffer("_S_tension"),
            self.get_buffer("_S_mass"),
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
