"""B-spline smoother for :mod:`gamfit.torch` matching the smoother family.

Same two-mode convention as :class:`gamfit.torch.PeriodicSmoother`:

* ``mode="auto"`` (default) — gamfit selects the smoothing internally on every
  forward pass; no smoothing parameter is exposed on the module and the caller
  does not have to tune anything.
* ``mode="learned"`` — smoothing is part of the module's trainable state
  (one :class:`torch.nn.Parameter` named ``log_smoothing``). Embed the module
  inside a larger ``nn.Module`` and let the outer optimizer step it via the
  exposed ``smoothing_score`` on the fit output.

Both modes share the same forward signature and output type. The B-spline
basis and its (single) difference penalty are evaluated through the same
analytic primitives that the rest of :mod:`gamfit.torch` uses.

For users who just want the underlying penalized ridge solve as a callable
(no module, no smoothing selection), :func:`penalized_ridge_solve` exposes
the fixed-``λ`` solve as a plain differentiable function.
"""

from __future__ import annotations

import torch

from ._basis import bspline_basis, smoothness_penalty
from ._cyclic_duchon import PeriodicFitOutput, _SmootherBase


__all__ = ["BSplineSmoother", "penalized_ridge_solve"]


def penalized_ridge_solve(
    x: torch.Tensor,
    y: torch.Tensor,
    penalty: torch.Tensor,
    lam: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable fixed-``λ`` penalized ridge solve.

    Solves ``(XᵀWX + λ·S) β = XᵀWy`` and returns ``(β, X β)``. All inputs are
    ordinary differentiable torch tensors — backward routes through
    :func:`torch.linalg.solve`, so gradients flow into every argument
    including ``lam`` and ``penalty``. Useful when the smoothing parameter is
    a learnable :class:`torch.nn.Parameter` on an outer module; for the
    REML-selected case, prefer :func:`gamfit.torch.gaussian_reml_fit`.

    Parameters
    ----------
    x : torch.Tensor
        Design matrix of shape ``(N, M)``.
    y : torch.Tensor
        Response of shape ``(N, D)`` (or ``(N,)``, which is treated as
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


class BSplineSmoother(_SmootherBase):
    """B-spline smoother over a fixed knot vector.

    Two modes, matching :class:`PeriodicSmoother`:

    - ``mode="auto"`` (default): each call to ``forward(t, y)`` selects the
      smoothing internally and returns the resulting fit. No
      :class:`torch.nn.Parameter` is exposed.
    - ``mode="learned"``: the smoothing is a trainable scalar
      ``log_smoothing`` (shape ``(1,)``). Place the module inside a larger
      ``nn.Module`` and let an outer optimizer update it via
      ``out.smoothing_score``.

    Parameters
    ----------
    knots : torch.Tensor
        Knot vector for the B-spline basis. Stored as a buffer in the
        module's default dtype.
    degree : int, optional
        Spline degree. Default ``3``.
    penalty_order : int, optional
        Order of the difference penalty on the coefficients. Default ``2``
        (the canonical second-difference smoothness penalty).
    periodic : bool, optional
        Whether to use the periodic B-spline variant. Default ``False``.
        For circular data prefer :class:`PeriodicSmoother`; this flag exists
        so the rare caller who wants a periodic B-spline (instead of the
        Duchon-on-a-circle that :class:`PeriodicSmoother` uses) can pick it.
    mode : str, optional
        ``"auto"`` (default) or ``"learned"``.
    init_log_smoothing : float, optional
        Initial ``log λ`` in either mode. Default ``0.0`` (i.e. ``λ = 1``).

    Examples
    --------
    Automatic smoothing — black box, no smoothing parameter exposed:

    >>> import torch, gamfit.torch as gt
    >>> knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    >>> sm = gt.BSplineSmoother(knots)
    >>> x = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    >>> y = torch.sin(2 * torch.pi * x)
    >>> out = sm(x, y)
    >>> beta, fitted = out.coefficients, out.fitted

    Trainable inside a learning loop:

    >>> sm = gt.BSplineSmoother(knots, mode="learned")
    >>> opt = torch.optim.Adam(sm.parameters(), lr=0.1)
    >>> for _ in range(50):
    ...     opt.zero_grad()
    ...     out = sm(x, y)
    ...     torch.autograd.backward(-out.smoothing_score)
    ...     opt.step()
    """

    def __init__(
        self,
        knots: torch.Tensor,
        *,
        degree: int = 3,
        penalty_order: int = 2,
        periodic: bool = False,
        mode: str = "auto",
        init_log_smoothing: float = 0.0,
    ) -> None:
        knots_t = knots.detach().to(dtype=torch.get_default_dtype())
        if knots_t.ndim != 1:
            raise ValueError(f"knots must be 1-D, got shape {tuple(knots_t.shape)}")
        n_basis = knots_t.shape[0] - degree - 1
        if n_basis <= penalty_order:
            raise ValueError(
                f"too few B-spline basis functions ({n_basis}) for a difference "
                f"penalty of order {penalty_order}; supply more knots or lower order"
            )

        super().__init__(
            mode=mode,
            n_components=1,
            init_log_smoothing=(init_log_smoothing,),
        )

        self._degree = int(degree)
        self._penalty_order = int(penalty_order)
        self._periodic = bool(periodic)
        penalty, _nullspace = smoothness_penalty(
            knots_t, degree=self._degree, order=self._penalty_order
        )
        self.register_buffer("knots", knots_t)
        self.register_buffer("_S0", penalty.to(dtype=knots_t.dtype))

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def penalty_order(self) -> int:
        return self._penalty_order

    @property
    def periodic(self) -> bool:
        return self._periodic

    def _build_basis(self, t: torch.Tensor) -> torch.Tensor:
        knots: torch.Tensor = self.get_buffer("knots")
        return bspline_basis(t, knots, degree=self._degree, periodic=self._periodic)

    def _penalty_components(self) -> tuple[torch.Tensor, ...]:
        return (self.get_buffer("_S0"),)

    def fit(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
    ) -> PeriodicFitOutput:
        """Alias for ``self(t, y, weights=weights)``."""
        return self(t, y, weights=weights)
