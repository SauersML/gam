"""Cyclic (periodic) Duchon spline primitives with closed-form triple penalty.

The cyclic Green's function of ``(-d^2/dx^2)^m`` on a circle of length ``P``
admits a closed form in terms of the periodic Bernoulli polynomials:

    G_m(tau) = (-1)^{m+1} * P^{2m} / (2m)! * B_tilde_{2m}({tau / P})

where ``B_tilde_n(u) = B_n({u})`` is the standard Bernoulli polynomial pulled
back through the fractional part ``{u} = u - floor(u)``. The triple-operator
penalty entries are likewise polynomial in the periodic Bernoulli of the
``(c_i - c_j)/P`` gaps. The sign convention ``epsilon_q`` is chosen so each
kernel block is positive semi-definite (the diagonal entries are the even
Bernoulli numbers, whose signs alternate so that ``epsilon_q * B_{2(2m-q)}``
is non-negative).
"""

from __future__ import annotations

from math import comb, factorial
from typing import NamedTuple

import torch


_BERNOULLI_NUMBERS: tuple[float, ...] = (
    1.0,
    -0.5,
    1.0 / 6.0,
    0.0,
    -1.0 / 30.0,
    0.0,
    1.0 / 42.0,
    0.0,
    -1.0 / 30.0,
    0.0,
    5.0 / 66.0,
    0.0,
    -691.0 / 2730.0,
    0.0,
    7.0 / 6.0,
    0.0,
    -3617.0 / 510.0,
    0.0,
    43867.0 / 798.0,
    0.0,
    -174611.0 / 330.0,
    0.0,
    854513.0 / 138.0,
    0.0,
    -236364091.0 / 2730.0,
)


def _bernoulli_polynomial_coeffs(n: int) -> list[float]:
    if n < 0:
        raise ValueError(f"Bernoulli polynomial degree must be non-negative, got {n}")
    if n >= len(_BERNOULLI_NUMBERS):
        raise ValueError(
            f"Bernoulli number table only covers indices up to {len(_BERNOULLI_NUMBERS) - 1}"
        )
    return [comb(n, k) * _BERNOULLI_NUMBERS[n - k] for k in range(n + 1)]


def _periodic_bernoulli(
    u: torch.Tensor, coeffs: torch.Tensor
) -> torch.Tensor:
    frac = u - torch.floor(u)
    # Horner from highest-degree term down. coeffs[k] is the u^k coefficient.
    n = coeffs.shape[0] - 1
    result = torch.full_like(frac, coeffs[n].item())
    for k in range(n - 1, -1, -1):
        result = result * frac + coeffs[k]
    return result


def _coeff_tensor(n: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(_bernoulli_polynomial_coeffs(n), dtype=dtype, device=device)


def cyclic_duchon_bernoulli_basis(
    t: torch.Tensor,
    centers: torch.Tensor,
    period: float,
    m: int,
) -> torch.Tensor:
    """Cyclic Bernoulli-Green's basis: ``(N, K+1)``.

    The first ``K`` columns evaluate ``G_m(t_i - c_j)`` and the last column is
    a free unpenalized intercept of ones. Differentiable w.r.t. ``t`` since the
    floor used to reduce arguments has zero derivative almost everywhere.
    """
    if m < 1:
        raise ValueError(f"polyharmonic order m must be >= 1, got {m}")
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period}")
    if t.dim() != 1:
        raise ValueError(f"t must be a 1-D tensor, got shape {tuple(t.shape)}")
    if centers.dim() != 1:
        raise ValueError(
            f"centers must be a 1-D tensor, got shape {tuple(centers.shape)}"
        )

    dtype = t.dtype if t.is_floating_point() else torch.get_default_dtype()
    t_f = t.to(dtype=dtype)
    centers_f = centers.detach().to(dtype=dtype, device=t_f.device)
    coeffs = _coeff_tensor(2 * m, dtype=dtype, device=t_f.device)
    sign = -1.0 if (m + 1) % 2 == 1 else 1.0  # (-1)^{m+1}
    prefactor = sign * (period ** (2 * m)) / float(factorial(2 * m))

    tau = (t_f.unsqueeze(1) - centers_f.unsqueeze(0)) / period
    kernel = prefactor * _periodic_bernoulli(tau, coeffs)
    ones = torch.ones((t_f.shape[0], 1), dtype=dtype, device=t_f.device)
    return torch.cat([kernel, ones], dim=1)


def cyclic_duchon_triple_penalty(
    centers: torch.Tensor,
    period: float,
    m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(S_mass, S_tension, S_stiffness)`` of shape ``(K+1, K+1)``.

    Each matrix has the ``K x K`` kernel block in the upper-left and zero in
    the final intercept row/column. PSD by construction.
    """
    if m < 2:
        raise ValueError(
            f"triple-operator penalty requires m >= 2 (q=2 stiffness must be "
            f"well-defined), got m = {m}"
        )
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period}")
    if centers.dim() != 1:
        raise ValueError(
            f"centers must be a 1-D tensor, got shape {tuple(centers.shape)}"
        )

    dtype = centers.dtype if centers.is_floating_point() else torch.get_default_dtype()
    centers_f = centers.detach().to(dtype=dtype)
    device = centers_f.device
    k = centers_f.shape[0]
    diffs = (centers_f.unsqueeze(1) - centers_f.unsqueeze(0)) / period

    blocks: list[torch.Tensor] = []
    for q in (0, 1, 2):
        n = 2 * (2 * m - q)
        coeffs = _coeff_tensor(n, dtype=dtype, device=device)
        # epsilon_q = (-1)^{2m - q + 1}; flips the sign of B_{2k} so the
        # diagonal (equal to the even Bernoulli number) ends up non-negative.
        eps = -1.0 if (2 * m - q + 1) % 2 == 1 else 1.0
        scale = eps * (period ** (2 * (2 * m - q) + 1)) / float(factorial(n))
        kernel_block = scale * _periodic_bernoulli(diffs, coeffs)
        kernel_block = 0.5 * (kernel_block + kernel_block.t())
        full = torch.zeros((k + 1, k + 1), dtype=dtype, device=device)
        full[:k, :k] = kernel_block
        blocks.append(full)

    return blocks[0], blocks[1], blocks[2]


def _logpdet(S: torch.Tensor, *, rel_tol: float = 1e-12) -> tuple[torch.Tensor, int]:
    """Log of the pseudo-determinant and the dimension of the kernel of ``S``."""
    sym = 0.5 * (S + S.t())
    eigvals = torch.linalg.eigvalsh(sym)
    max_abs = torch.clamp(eigvals.abs().max(), min=torch.finfo(S.dtype).tiny)
    threshold = rel_tol * max_abs
    mask = eigvals > threshold
    nullity = int((~mask).sum().item())
    safe = torch.where(mask, eigvals, torch.ones_like(eigvals))
    logdet = torch.where(mask, torch.log(safe), torch.zeros_like(eigvals)).sum()
    return logdet, nullity


def cyclic_duchon_quadratic_fit(
    X: torch.Tensor,
    y: torch.Tensor,
    S: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form penalized ridge solve with analytic Gaussian REML score.

    Solves ``(X^T W X + S) beta = X^T W y`` using ``S`` as supplied (no inner
    lambda optimization). All ops flow autograd through ``S`` and ``X``.
    """
    if X.dim() != 2:
        raise ValueError(f"X must be 2-D, got shape {tuple(X.shape)}")
    if y.dim() not in (1, 2):
        raise ValueError(f"y must be 1-D or 2-D, got shape {tuple(y.shape)}")
    if S.dim() != 2 or S.shape[0] != S.shape[1] or S.shape[0] != X.shape[1]:
        raise ValueError(
            f"S must be (M, M) matching X columns, got S {tuple(S.shape)} vs "
            f"X {tuple(X.shape)}"
        )

    n_rows, n_cols = X.shape
    squeeze_out = y.dim() == 1
    y_mat = y.unsqueeze(1) if squeeze_out else y

    if weights is None:
        w = torch.ones(n_rows, dtype=X.dtype, device=X.device)
    else:
        if weights.dim() != 1 or weights.shape[0] != n_rows:
            raise ValueError(
                f"weights must be a length-N vector, got shape {tuple(weights.shape)}"
            )
        w = weights.to(dtype=X.dtype, device=X.device)

    sqrt_w = torch.sqrt(w)
    Xw = X * sqrt_w.unsqueeze(1)
    yw = y_mat * sqrt_w.unsqueeze(1)

    XtWX = Xw.t() @ Xw
    XtWy = Xw.t() @ yw
    sym_S = 0.5 * (S + S.t())
    A = XtWX + sym_S

    try:
        chol = torch.linalg.cholesky(A)
        beta = torch.cholesky_solve(XtWy, chol)
        logdet_A = 2.0 * torch.log(torch.diagonal(chol)).sum()
    except Exception:
        beta = torch.linalg.solve(A, XtWy)
        sign, logabs = torch.linalg.slogdet(A)
        if torch.any(sign <= 0):
            raise RuntimeError("X^T W X + S is not positive definite")
        logdet_A = logabs

    fitted_mat = X @ beta

    log_pdet_S, nullity = _logpdet(sym_S)
    p_null = int(nullity)
    denom = float(n_rows - p_null)
    if denom <= 0.0:
        raise ValueError(
            f"effective sample size N - dim(ker S) = {denom} is non-positive"
        )

    residual = (y_mat - fitted_mat) * sqrt_w.unsqueeze(1)
    rss = (residual * residual).sum()
    sigma2_hat = rss / denom
    sigma2_hat = torch.clamp(sigma2_hat, min=torch.finfo(X.dtype).tiny)
    reml_score = -0.5 * (logdet_A - log_pdet_S + denom * torch.log(sigma2_hat))

    if squeeze_out:
        beta = beta.squeeze(1)
        fitted = fitted_mat.squeeze(1)
    else:
        fitted = fitted_mat
    return beta, fitted, reml_score


class CyclicDuchonFitOutput(NamedTuple):
    """Output of :class:`CyclicDuchonTripleSmoother` ``forward``.

    Fields
    ------
    coefficients: (M,) or (M, D) coefficient vector / matrix.
    fitted: (N,) or (N, D) in-sample fitted values.
    reml_score: scalar REML score (higher is better).
    lambdas: (3,) tensor of the current (mass, tension, stiffness) lambdas.
    """

    coefficients: torch.Tensor
    fitted: torch.Tensor
    reml_score: torch.Tensor
    lambdas: torch.Tensor


class CyclicDuchonTripleSmoother(torch.nn.Module):
    """Periodic 1D smoother with closed-form triple-operator analytic penalty.

    Three log-lambda parameters (mass, tension, stiffness) are trained by the
    outer optimizer via backprop through the analytic REML score. The kernel
    intercept column is unpenalized.
    """

    def __init__(
        self,
        centers: torch.Tensor,
        period: float,
        m: int = 2,
        *,
        init_log_lambdas: tuple[float, float, float] = (0.0, 0.0, 0.0),
        intercept_ridge: float = 0.0,
    ) -> None:
        super().__init__()
        if intercept_ridge < 0.0:
            raise ValueError(
                f"intercept_ridge must be non-negative, got {intercept_ridge}"
            )
        if period <= 0.0:
            raise ValueError(f"period must be positive, got {period}")

        centers_t = centers.detach().to(dtype=torch.get_default_dtype())
        S0, S1, S2 = cyclic_duchon_triple_penalty(centers_t, period, m)

        self._period = float(period)
        self._m = int(m)
        self._intercept_ridge = float(intercept_ridge)

        self.register_buffer("centers", centers_t)
        self.register_buffer("S_mass", S0)
        self.register_buffer("S_tension", S1)
        self.register_buffer("S_stiffness", S2)

        self.log_lambdas = torch.nn.Parameter(
            torch.tensor(init_log_lambdas, dtype=torch.get_default_dtype())
        )

    @property
    def period(self) -> float:
        return self._period

    @property
    def m(self) -> int:
        return self._m

    def _assemble_penalty(self) -> torch.Tensor:
        lams = torch.exp(self.log_lambdas)
        S = (
            lams[0] * self.S_mass
            + lams[1] * self.S_tension
            + lams[2] * self.S_stiffness
        )
        if self._intercept_ridge > 0.0:
            ridge = torch.zeros_like(S)
            ridge[-1, -1] = self._intercept_ridge
            S = S + ridge
        return S

    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
    ) -> CyclicDuchonFitOutput:
        X = cyclic_duchon_bernoulli_basis(t, self.centers, self._period, self._m)
        S = self._assemble_penalty()
        beta, fitted, reml = cyclic_duchon_quadratic_fit(X, y, S, weights=weights)
        return CyclicDuchonFitOutput(
            coefficients=beta,
            fitted=fitted,
            reml_score=reml,
            lambdas=torch.exp(self.log_lambdas),
        )

    def predict(self, t: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """Evaluate the fitted smoother at new locations ``t``."""
        X = cyclic_duchon_bernoulli_basis(t, self.centers, self._period, self._m)
        return X @ coefficients
