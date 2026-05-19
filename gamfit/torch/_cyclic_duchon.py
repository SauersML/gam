"""Periodic 1-D smoother for ``gamfit.torch``.

Internally this module evaluates the cyclic Green's function of
``(-d^2/dx^2)^m`` on a circle of length ``P`` via the periodic Bernoulli
polynomials:

    G_m(tau) = (-1)^{m+1} * P^{2m} / (2m)! * B_tilde_{2m}({tau / P})

and assembles three closed-form penalty matrices (mass / tension / stiffness
operator blocks). The user does not see any of this; they instantiate
``PeriodicSmoother(period=...)`` and either fit (``mode="auto"``) or embed it
as an ``nn.Module`` whose smoothing is trainable (``mode="learned"``).

The shared mode-handling / quadratic-fit machinery lives in ``_SmootherBase``
so a future ``NaturalSmoother`` can drop in by overriding only the basis and
penalty construction.
"""

from __future__ import annotations

import math
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


def _periodic_bernoulli(u: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    frac = u - torch.floor(u)
    n = coeffs.shape[0] - 1
    result = torch.full_like(frac, coeffs[n].item())
    for k in range(n - 1, -1, -1):
        result = result * frac + coeffs[k]
    return result


def _coeff_tensor(n: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(_bernoulli_polynomial_coeffs(n), dtype=dtype, device=device)


def _cyclic_duchon_basis(
    t: torch.Tensor,
    centers: torch.Tensor,
    period: float,
    m: int,
) -> torch.Tensor:
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


def _cyclic_triple_penalty(
    centers: torch.Tensor,
    period: float,
    m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if m < 2:
        raise ValueError(
            f"closed-form penalty requires m >= 2, got m = {m}"
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
        # epsilon_q = (-1)^{2m - q + 1}; flips B_{2k}'s sign so each block's
        # diagonal (the even Bernoulli number) is non-negative.
        eps = -1.0 if (2 * m - q + 1) % 2 == 1 else 1.0
        scale = eps * (period ** (2 * (2 * m - q) + 1)) / float(factorial(n))
        kernel_block = scale * _periodic_bernoulli(diffs, coeffs)
        kernel_block = 0.5 * (kernel_block + kernel_block.t())
        full = torch.zeros((k + 1, k + 1), dtype=dtype, device=device)
        full[:k, :k] = kernel_block
        blocks.append(full)

    return blocks[0], blocks[1], blocks[2]


def _logpdet(S: torch.Tensor, *, rel_tol: float = 1e-12) -> tuple[torch.Tensor, int]:
    sym = 0.5 * (S + S.t())
    eigvals = torch.linalg.eigvalsh(sym)
    max_abs = torch.clamp(eigvals.abs().max(), min=torch.finfo(S.dtype).tiny)
    threshold = rel_tol * max_abs
    mask = eigvals > threshold
    nullity = int((~mask).sum().item())
    safe = torch.where(mask, eigvals, torch.ones_like(eigvals))
    logdet = torch.where(mask, torch.log(safe), torch.zeros_like(eigvals)).sum()
    return logdet, nullity


def _quadratic_fit(
    X: torch.Tensor,
    y: torch.Tensor,
    S: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if X.dim() != 2:
        raise ValueError(f"X must be 2-D, got shape {tuple(X.shape)}")
    if y.dim() not in (1, 2):
        raise ValueError(f"y must be 1-D or 2-D, got shape {tuple(y.shape)}")
    if S.dim() != 2 or S.shape[0] != S.shape[1] or S.shape[0] != X.shape[1]:
        raise ValueError(
            f"S must be (M, M) matching X columns, got S {tuple(S.shape)} vs "
            f"X {tuple(X.shape)}"
        )

    n_rows, _ = X.shape
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
    score = -0.5 * (logdet_A - log_pdet_S + denom * torch.log(sigma2_hat))

    if squeeze_out:
        beta = beta.squeeze(1)
        fitted = fitted_mat.squeeze(1)
    else:
        fitted = fitted_mat
    return beta, fitted, score


class PeriodicFitOutput(NamedTuple):
    """Output of :class:`PeriodicSmoother` forward pass.

    Fields
    ------
    coefficients:
        Fitted coefficient vector (or matrix when ``y`` is 2-D).
    fitted:
        In-sample fitted values at the supplied evaluation points.
    smoothing_score:
        Scalar score used internally to select / learn the smoothing. In
        ``mode="learned"`` this is the loss surface the outer optimizer should
        ascend (or equivalently, the negation is what backprop should descend).
    """

    coefficients: torch.Tensor
    fitted: torch.Tensor
    smoothing_score: torch.Tensor


_DEFAULT_PERIODIC_ORDER: int = 2
_DEFAULT_N_CENTERS: int = 20


class _SmootherBase(torch.nn.Module):
    """Shared skeleton for analytic 1-D smoothers.

    Subclasses provide:

    - ``_build_basis(t)`` returning the ``(N, M)`` design matrix
    - ``_penalty_components()`` returning a sequence of frozen ``(M, M)``
      penalty matrices to be linearly combined via the smoother's log-weights

    The base class handles the two user-visible modes:

    - ``"auto"``: weights are not parameters; on each forward we pick them
      internally (currently by REML profiled over each component using a short
      coordinate refinement) and return the resulting fit.
    - ``"learned"``: weights are an ``nn.Parameter`` named ``log_smoothing``,
      so the outer optimizer drives them via the exposed score.
    """

    log_smoothing: torch.nn.Parameter | None

    def __init__(
        self,
        *,
        mode: str,
        n_components: int,
        init_log_smoothing: tuple[float, ...],
    ) -> None:
        super().__init__()
        if mode not in ("auto", "learned"):
            raise ValueError(
                f"mode must be 'auto' or 'learned', got {mode!r}"
            )
        if len(init_log_smoothing) != n_components:
            raise ValueError(
                f"init_log_smoothing must have length {n_components}, got "
                f"{len(init_log_smoothing)}"
            )
        self._mode = mode
        init = torch.tensor(init_log_smoothing, dtype=torch.get_default_dtype())
        if mode == "learned":
            self.log_smoothing = torch.nn.Parameter(init)
            self.register_buffer("_log_smoothing_auto", torch.empty(0))
        else:
            self.log_smoothing = None
            self.register_buffer("_log_smoothing_auto", init.clone())

    @property
    def mode(self) -> str:
        return self._mode

    def _build_basis(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _penalty_components(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _predict_basis(self, t: torch.Tensor) -> torch.Tensor:
        return self._build_basis(t)

    def _assemble_penalty(self, log_weights: torch.Tensor) -> torch.Tensor:
        comps = self._penalty_components()
        weights = torch.exp(log_weights)
        S = weights[0] * comps[0]
        for i in range(1, len(comps)):
            S = S + weights[i] * comps[i]
        return S

    def _fit_with(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        log_weights: torch.Tensor,
        *,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S = self._assemble_penalty(log_weights)
        return _quadratic_fit(X, y, S, weights=weights)

    def _auto_select(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        # Coordinate refinement on log_smoothing using the analytic REML score.
        # Cheap, no nn.Parameter exposure to the user, and behaves like a
        # one-shot smoothing selector. Implementation can change freely.
        log_weights = self._log_smoothing_auto.clone()
        grid = torch.tensor(
            [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
            dtype=log_weights.dtype,
            device=log_weights.device,
        )
        for _ in range(3):
            for j in range(log_weights.shape[0]):
                best_val = None
                best_score = None
                for delta in grid.tolist():
                    trial = log_weights.clone()
                    trial[j] = log_weights[j] + delta
                    try:
                        _, _, score = self._fit_with(X, y, trial, weights=weights)
                    except RuntimeError:
                        continue
                    s = float(score.detach().item())
                    if not math.isfinite(s):
                        continue
                    if best_score is None or s > best_score:
                        best_score = s
                        best_val = float(trial[j].item())
                if best_val is not None:
                    log_weights[j] = best_val
        self._log_smoothing_auto = log_weights.detach().clone()
        return log_weights

    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
    ) -> PeriodicFitOutput:
        X = self._build_basis(t)
        if self._mode == "learned":
            assert self.log_smoothing is not None
            log_weights = self.log_smoothing
        else:
            with torch.no_grad():
                log_weights = self._auto_select(X, y, weights=weights)
        beta, fitted, score = self._fit_with(X, y, log_weights, weights=weights)
        return PeriodicFitOutput(
            coefficients=beta, fitted=fitted, smoothing_score=score
        )

    def predict(self, t: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """Evaluate the smoother at new locations using fitted coefficients."""
        X = self._predict_basis(t)
        return X @ coefficients


class PeriodicSmoother(_SmootherBase):
    """Smoother for closed (periodic) curves.

    Two modes:

    - ``mode="auto"`` (default): each call to ``forward(t, y)`` fits the
      smoother and returns the resulting curve. Smoothing is selected
      internally; you do not need to tune anything.
    - ``mode="learned"``: smoothing is part of the module's trainable state
      (one ``nn.Parameter``). Embed the module in a larger ``nn.Module`` and
      let your outer optimizer (Adam, etc.) update it via the exposed
      ``smoothing_score`` on the fit output.

    Parameters
    ----------
    period:
        Length of the period (positive). The data are treated as living on a
        circle of this circumference.
    n_centers:
        Number of equispaced control centers along ``[0, period)``. Ignored if
        ``centers`` is supplied.
    mode:
        ``"auto"`` (default) or ``"learned"``. See above.
    centers:
        Optional explicit centers; defaults to ``n_centers`` equispaced points.

    Examples
    --------
    Automatic fit::

        smoother = PeriodicSmoother(period=2 * math.pi)
        out = smoother(t, y)
        curve = out.fitted

    Trainable inside a learning loop::

        smoother = PeriodicSmoother(period=2 * math.pi, mode="learned")
        opt = torch.optim.Adam(smoother.parameters(), lr=0.1)
        for _ in range(steps):
            opt.zero_grad()
            out = smoother(t, y)
            (-out.smoothing_score).backward()
            opt.step()
    """

    def __init__(
        self,
        period: float,
        *,
        n_centers: int = _DEFAULT_N_CENTERS,
        mode: str = "auto",
        centers: torch.Tensor | None = None,
    ) -> None:
        if period <= 0.0:
            raise ValueError(f"period must be positive, got {period}")
        if centers is None:
            if n_centers < 4:
                raise ValueError(f"n_centers must be >= 4, got {n_centers}")
            centers_t = torch.linspace(
                0.0, period, n_centers + 1, dtype=torch.get_default_dtype()
            )[:-1]
        else:
            centers_t = centers.detach().to(dtype=torch.get_default_dtype())
            if centers_t.dim() != 1 or centers_t.shape[0] < 4:
                raise ValueError(
                    f"centers must be a 1-D tensor with at least 4 entries, "
                    f"got shape {tuple(centers_t.shape)}"
                )

        super().__init__(
            mode=mode,
            n_components=3,
            init_log_smoothing=(0.0, 0.0, 0.0),
        )

        self._period = float(period)
        self._m = _DEFAULT_PERIODIC_ORDER
        S0, S1, S2 = _cyclic_triple_penalty(centers_t, self._period, self._m)
        self.register_buffer("centers", centers_t)
        self.register_buffer("_S0", S0)
        self.register_buffer("_S1", S1)
        self.register_buffer("_S2", S2)

    @property
    def period(self) -> float:
        return self._period

    def _build_basis(self, t: torch.Tensor) -> torch.Tensor:
        return _cyclic_duchon_basis(t, self.centers, self._period, self._m)

    def _penalty_components(self) -> tuple[torch.Tensor, ...]:
        return (self._S0, self._S1, self._S2)

    def fit(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
    ) -> PeriodicFitOutput:
        """Fit the smoother to ``(t, y)`` and return the resulting curve.

        Convenience alias for ``self(t, y, weights=weights)``. Designed for the
        common happy path: ``PeriodicSmoother(period=...).fit(t, y).fitted``
        gives a sensible regularized curve with no further configuration.
        """
        return self(t, y, weights=weights)
