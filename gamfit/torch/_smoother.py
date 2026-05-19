"""User-facing smoothers for :mod:`gamfit.torch`.

Two modes, opt-in by the ``learned`` keyword:

* **Automatic** (default) — gamfit selects the smoothness internally. From the
  caller's perspective the smoother is a black box that produces a smooth fit;
  the word "REML" never appears in user code, and no smoothing parameter is
  exposed on the module. Under the hood we route through
  :func:`gamfit.torch.gaussian_reml_fit`, which solves the closed-form
  REML problem analytically and back-propagates gradients to ``x``, ``y``,
  ``weights``, and ``penalty`` via the Rust analytic backward.

* **Learned** — the smoothness is part of the module's trainable state, like
  any other ``nn.Parameter``. The module exposes a single learnable
  ``log_lambda`` that an outer optimizer steps alongside the rest of the model.
  Forward is a plain penalized ridge solve at the current ``λ``; backward is
  routed by torch autograd through :func:`torch.linalg.solve`, so gradients
  flow into ``x``, ``y``, ``penalty``, ``weights``, **and** ``log_lambda``.

Choosing a mode:

>>> sm = Smoother(knots)             # automatic — no learned smoothing
>>> sm = Smoother(knots, learned=True)  # learned smoothing; log_lambda is an nn.Parameter
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ._basis import bspline_basis, smoothness_penalty
from ._reml import gaussian_reml_fit

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = ["Smoother", "penalized_ridge_solve"]


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
    treated as ordinary differentiable torch tensors — backward routes through
    :func:`torch.linalg.solve`, so gradients flow into every argument including
    ``lam`` and ``penalty``. Use this when smoothing is part of the model's
    trainable state (the **learned** mode of :class:`Smoother`); use
    :func:`gamfit.torch.gaussian_reml_fit` instead when smoothing should be
    selected automatically by REML.

    Parameters
    ----------
    x : torch.Tensor
        Design matrix of shape ``(N, M)``.
    y : torch.Tensor
        Response of shape ``(N, D)``.
    penalty : torch.Tensor
        Symmetric penalty matrix of shape ``(M, M)``.
    lam : torch.Tensor
        Scalar smoothing multiplier on ``penalty``. Must be non-negative for the
        normal-equations to remain positive definite; pass ``log_lambda.exp()``
        to make this automatic.
    weights : torch.Tensor or None, optional
        Optional row weights of shape ``(N,)``. ``None`` is equivalent to unit
        weights.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``coefficients`` of shape ``(M, D)`` and ``fitted = X β`` of shape
        ``(N, D)``.
    """
    if weights is None:
        xtx = x.t() @ x
        xty = x.t() @ y
    else:
        wx = weights.unsqueeze(1) * x
        xtx = wx.t() @ x
        xty = wx.t() @ y
    hessian = xtx + lam * penalty
    coefficients = torch.linalg.solve(hessian, xty)
    fitted = x @ coefficients
    return coefficients, fitted


class Smoother(nn.Module):
    """1-D smoother over a B-spline basis with REML or learned smoothness.

    Holds the basis knots as a buffer and, in **learned** mode, a single scalar
    ``log_lambda`` :class:`~torch.nn.Parameter` that an outer optimizer can
    update. The forward signature is ``smoother(x, y, weights=None)`` and
    returns ``(coefficients, fitted)`` in both modes so the call site does not
    have to branch on which mode is active.

    Parameters
    ----------
    knots : array-like
        Knot vector for the B-spline basis. Stored as a float64 buffer.
    degree : int, optional
        Spline degree. Default ``3``.
    penalty_order : int, optional
        Order of the difference penalty applied to the basis coefficients.
        Default ``2`` (the usual second-difference smoothness penalty).
    periodic : bool, optional
        Whether to use the periodic B-spline variant. Default ``False``.
    learned : bool, optional
        If ``False`` (default), every forward pass selects ``λ`` internally by
        closed-form REML; no smoothing parameter is exposed. If ``True``, a
        single learnable ``log_lambda`` parameter governs smoothness and the
        forward pass becomes a fixed-``λ`` penalized ridge solve.
    init_log_lambda : float, optional
        Initial value of ``log λ`` in learned mode. Default ``0.0`` (i.e.
        ``λ = 1``). Ignored when ``learned=False``.

    Examples
    --------
    Automatic smoothing — no smoothing parameter to think about:

    >>> import torch, gamfit.torch as gt
    >>> knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    >>> sm = gt.Smoother(knots)
    >>> x = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    >>> y = torch.sin(2 * torch.pi * x).reshape(-1, 1)
    >>> beta, fitted = sm(x, y)

    Learned smoothing inside a larger training loop:

    >>> sm = gt.Smoother(knots, learned=True)
    >>> opt = torch.optim.Adam(sm.parameters(), lr=0.05)
    >>> for _ in range(50):
    ...     opt.zero_grad()
    ...     _, fitted = sm(x, y)
    ...     loss = (fitted - y).pow(2).mean()
    ...     torch.autograd.backward(loss)
    ...     opt.step()
    """

    log_lambda: nn.Parameter | None

    def __init__(
        self,
        knots: torch.Tensor | Iterable[float],
        *,
        degree: int = 3,
        penalty_order: int = 2,
        periodic: bool = False,
        learned: bool = False,
        init_log_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        knots_tensor = torch.as_tensor(knots, dtype=torch.float64)
        if knots_tensor.ndim != 1:
            raise ValueError("knots must be 1-D")
        self.register_buffer("knots", knots_tensor)
        self.degree = int(degree)
        self.penalty_order = int(penalty_order)
        self.periodic = bool(periodic)
        if learned:
            self.log_lambda = nn.Parameter(
                torch.tensor(float(init_log_lambda), dtype=torch.float64)
            )
        else:
            self.log_lambda = None

    @property
    def learned(self) -> bool:
        """``True`` when smoothness is a trainable parameter, ``False`` for automatic REML."""
        return self.log_lambda is not None

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the smoother to ``(x, y)`` and return ``(coefficients, fitted)``.

        In automatic mode this calls into the closed-form REML primitive,
        which selects ``λ`` internally. In learned mode this is a plain
        fixed-``λ`` ridge solve at ``λ = exp(log_lambda)``.

        Parameters
        ----------
        x : torch.Tensor
            Evaluation locations of shape ``(N,)``. A 2-D input of shape
            ``(N, 1)`` is also accepted.
        y : torch.Tensor
            Response of shape ``(N,)`` or ``(N, D)``.
        weights : torch.Tensor or None, optional
            Optional row weights of shape ``(N,)``.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            ``coefficients`` of shape ``(M, D)`` and ``fitted`` of shape
            ``(N, D)`` (``D = 1`` when ``y`` was 1-D).
        """
        if x.ndim == 2 and x.shape[-1] == 1:
            x_eval = x.squeeze(-1)
        else:
            x_eval = x
        if x_eval.ndim != 1:
            raise ValueError("x must be 1-D (or (N, 1))")
        y_mat = y if y.ndim == 2 else y.unsqueeze(1)

        knots: torch.Tensor = self.get_buffer("knots")
        basis = bspline_basis(x_eval, knots, degree=self.degree, periodic=self.periodic)
        penalty, _nullspace = smoothness_penalty(
            knots, degree=self.degree, order=self.penalty_order
        )

        if self.log_lambda is None:
            out = gaussian_reml_fit(basis, y_mat, penalty, weights=weights)
            return out.coefficients, out.fitted
        lam = self.log_lambda.exp()
        return penalized_ridge_solve(basis, y_mat, penalty, lam, weights=weights)
