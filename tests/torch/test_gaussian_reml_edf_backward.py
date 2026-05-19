"""Analytic backward correctness for the Gaussian REML effective DOF (`edf`).

These tests exercise the `grad_edf` route through the closed-form Gaussian REML
analytic VJP pipeline: forward via the torch wrapper, then check that the
gradient of `out.edf` (and composite linear combinations involving it) matches
finite-difference references on every differentiable input.

Why rank-deficient ``S``
------------------------
Real penalty matrices in GAMs are rank-deficient by construction: the null
space holds the constant function (B-spline difference penalty), or constants
plus linears (1-D Duchon m=2, classical smoothing splines), or a polynomial
basis (thin-plate). That null space is what makes the smoother insensitive to
the function's mean / slope while still penalizing curvature. The code paths
that handle it — Moore-Penrose ``S⁺``, the rank-detection threshold on
eigenvalues, ``log|S|₊`` (generalized log-determinant) — are precisely what
``grad_penalty`` must differentiate correctly. So these tests use
``S = a aᵀ`` with ``a`` of shape ``(p, p-1)``, giving a one-dimensional null
space.

FD on rank-deficient ``S`` — directional, in the range
-------------------------------------------------------
A naive entry-wise FD perturbation can push the smallest eigenvalue of ``S``
slightly negative (numerical FP noise + rank-deficient base), at which point
the REML core's strict PSD validation rejects the perturbed input and FD can't
even be computed. The mathematically correct fix is to test the gradient
along directions that respect the rank structure: random symmetric directions
projected onto the *range* of ``S``. The perturbed ``S ± h·D`` then stays
rank-deficient (same null space) and stays PSD up to FP, so the forward
accepts it and FD is well-defined. Entry-wise FD is kept for the
unconstrained inputs (``x``, ``y``, ``w``).
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    from gamfit.torch import gaussian_reml_fit, gaussian_reml_fit_batched
except ImportError:
    pytest.skip("torch dependency unavailable", allow_module_level=True)


_F64 = torch.float64
_RTOL = 5e-5
_ATOL = 1e-9
_FD_STEP = 1e-5
_RANGE_REL_TOL = 1e-10


def _make_penalty(p: int, rng: np.random.Generator) -> np.ndarray:
    """Rank-deficient ``(p, p)`` PSD penalty with a one-dimensional null space.

    Built as ``a aᵀ`` with ``a`` of shape ``(p, p-1)``. This matches the
    structural property of every penalty in production: a non-trivial null
    space (constants / polynomials) that ``log|S|₊`` and ``S⁺`` must handle.
    """
    a = rng.standard_normal((p, p - 1))
    return (a @ a.T).astype(np.float64)


def _problem(seed: int = 0, n: int = 15, p: int = 4, d: int = 2):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    y = rng.standard_normal((n, d))
    s = _make_penalty(p, rng)
    w = rng.uniform(0.5, 1.5, size=n)
    return x, y, s, w


def _to_tensor(arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    t = torch.as_tensor(arr, dtype=_F64).clone()
    t.requires_grad_(requires_grad)
    return t


def _fit_edf_value(
    x: np.ndarray, y: np.ndarray, s: np.ndarray, w: np.ndarray
) -> float:
    """Forward-only edf evaluation (used for finite differences)."""
    out = gaussian_reml_fit(
        _to_tensor(x),
        _to_tensor(y),
        _to_tensor(s),
        weights=_to_tensor(w),
    )
    return float(out.edf.detach())


def _project_to_range(
    s: np.ndarray, rel_tol: float = _RANGE_REL_TOL
) -> np.ndarray:
    """Orthogonal projector onto the range of the symmetric part of ``s``.

    Returns ``U Uᵀ`` where ``U`` columns are eigenvectors of strictly positive
    eigenvalues. The threshold ``rel_tol·max|eig|`` matches the convention used
    inside the REML eigenpair cache so the "range" tested here is the same one
    the analytic helpers treat as the rank of ``S``.
    """
    sym = 0.5 * (s + s.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    scale = abs(eigvals).max() if eigvals.size else 0.0
    threshold = scale * rel_tol
    mask = eigvals > threshold
    return eigvecs[:, mask] @ eigvecs[:, mask].T


def _sample_range_directions(
    s: np.ndarray, rng: np.random.Generator, k: int = 5
) -> list[np.ndarray]:
    """``k`` random unit-norm symmetric directions living in range(``s``)."""
    projector = _project_to_range(s)
    directions: list[np.ndarray] = []
    for _ in range(k):
        raw = rng.standard_normal(s.shape)
        sym = 0.5 * (raw + raw.T)
        projected = projector @ sym @ projector
        # Re-symmetrize explicitly to absorb FP noise from the conjugation.
        projected = 0.5 * (projected + projected.T)
        norm = np.linalg.norm(projected)
        if norm > 0.0:
            projected = projected / norm
        directions.append(projected)
    return directions


def _fd_grad(
    base: np.ndarray,
    indices: list[tuple[int, ...]],
    objective,
    h: float = _FD_STEP,
) -> dict[tuple[int, ...], float]:
    """Centered entry-wise FD for unconstrained inputs (x, y, w)."""
    grads: dict[tuple[int, ...], float] = {}
    for idx in indices:
        plus = base.copy()
        minus = base.copy()
        plus[idx] += h
        minus[idx] -= h
        grads[idx] = (objective(plus) - objective(minus)) / (2.0 * h)
    return grads


def _fd_directional(
    base: np.ndarray, direction: np.ndarray, objective, h: float = _FD_STEP
) -> float:
    """Centered FD along a matrix-valued direction. Used for ``S``."""
    plus = objective(base + h * direction)
    minus = objective(base - h * direction)
    return (plus - minus) / (2.0 * h)


def _analytic_directional(grad: np.ndarray, direction: np.ndarray) -> float:
    """Frobenius inner product ``⟨grad, direction⟩``."""
    return float(np.sum(grad * direction))


def _matches(analytic: float, fd: float, rtol: float = _RTOL, atol: float = _ATOL) -> bool:
    """``np.allclose``-style scalar comparison with absolute + relative slack."""
    return abs(analytic - fd) <= atol + rtol * max(abs(analytic), abs(fd))


def _sample_indices(
    shape: tuple[int, ...], rng: np.random.Generator, k: int = 4
) -> list[tuple[int, ...]]:
    size = int(np.prod(shape))
    chosen = rng.choice(size, size=min(k, size), replace=False)
    return [tuple(int(c) for c in np.unravel_index(int(flat), shape)) for flat in chosen]


def test_edf_backward_matches_finite_difference() -> None:
    x_np, y_np, s_np, w_np = _problem(seed=2024)

    x = _to_tensor(x_np, requires_grad=True)
    y = _to_tensor(y_np, requires_grad=True)
    s = _to_tensor(s_np, requires_grad=True)
    w = _to_tensor(w_np, requires_grad=True)

    out = gaussian_reml_fit(x, y, s, weights=w)
    out.edf.backward()

    assert x.grad is not None
    assert y.grad is not None
    assert s.grad is not None
    assert w.grad is not None

    rng = np.random.default_rng(11)
    # Unconstrained inputs: entry-wise FD.
    unconstrained = {
        "x": (x_np, x.grad.detach().numpy(), _sample_indices(x_np.shape, rng, 5)),
        "y": (y_np, y.grad.detach().numpy(), _sample_indices(y_np.shape, rng, 5)),
        "w": (w_np, w.grad.detach().numpy(), _sample_indices(w_np.shape, rng, 4)),
    }
    for name, (base, analytic, idxs) in unconstrained.items():
        def obj(arr: np.ndarray, name: str = name) -> float:
            if name == "x":
                return _fit_edf_value(arr, y_np, s_np, w_np)
            if name == "y":
                return _fit_edf_value(x_np, arr, s_np, w_np)
            return _fit_edf_value(x_np, y_np, s_np, arr)

        fd = _fd_grad(base, idxs, obj)
        for idx, fd_val in fd.items():
            an_val = float(analytic[idx])
            assert _matches(an_val, fd_val), (
                f"{name}{idx}: analytic={an_val:.6e} fd={fd_val:.6e}"
            )

    # ``S`` is constrained to be symmetric PSD and is rank-deficient by
    # construction; test the gradient along random symmetric directions in
    # range(S) so the perturbed S stays inside the PSD cone.
    analytic_s = s.grad.detach().numpy()

    def obj_s(arr: np.ndarray) -> float:
        return _fit_edf_value(x_np, y_np, arr, w_np)

    for direction in _sample_range_directions(s_np, rng, k=5):
        fd_val = _fd_directional(s_np, direction, obj_s)
        an_val = _analytic_directional(analytic_s, direction)
        assert _matches(an_val, fd_val), (
            f"s directional: analytic={an_val:.6e} fd={fd_val:.6e}"
        )


def test_edf_composite_backward_matches_finite_difference() -> None:
    """grad_edf must compose correctly with other upstream gradients."""
    x_np, y_np, s_np, w_np = _problem(seed=7)

    coef_alpha = 1.0
    edf_alpha = 3.7
    reml_alpha = -0.2

    def scalar_obj(x_arr: np.ndarray) -> float:
        out = gaussian_reml_fit(
            _to_tensor(x_arr),
            _to_tensor(y_np),
            _to_tensor(s_np),
            weights=_to_tensor(w_np),
        )
        coef_sum = float(out.coefficients.detach().sum())
        edf = float(out.edf.detach())
        reml = float(out.reml_score.detach())
        return coef_alpha * coef_sum + edf_alpha * edf + reml_alpha * reml

    x = _to_tensor(x_np, requires_grad=True)
    y = _to_tensor(y_np, requires_grad=True)
    s = _to_tensor(s_np, requires_grad=True)
    w = _to_tensor(w_np, requires_grad=True)

    out = gaussian_reml_fit(x, y, s, weights=w)
    scalar = (
        coef_alpha * out.coefficients.sum()
        + edf_alpha * out.edf
        - 0.2 * out.reml_score
    )
    scalar.backward()

    assert x.grad is not None
    analytic = x.grad.detach().numpy()

    rng = np.random.default_rng(99)
    idxs = _sample_indices(x_np.shape, rng, 5)
    fd = _fd_grad(x_np, idxs, scalar_obj)
    for idx, fd_val in fd.items():
        an_val = float(analytic[idx])
        assert _matches(an_val, fd_val), (
            f"composite x{idx}: analytic={an_val:.6e} fd={fd_val:.6e}"
        )


def test_edf_batched_backward_matches_finite_difference() -> None:
    """Batched grad_edf is a length-K vector; one batch component at a time."""
    rng = np.random.default_rng(42)
    sizes = [12, 14, 13]
    p, d = 4, 2

    x_blocks = [rng.standard_normal((n, p)) for n in sizes]
    y_blocks = [rng.standard_normal((n, d)) for n in sizes]
    w_blocks = [rng.uniform(0.5, 1.5, size=n) for n in sizes]
    s_np = _make_penalty(p, rng)

    x_np = np.concatenate(x_blocks, axis=0)
    y_np = np.concatenate(y_blocks, axis=0)
    w_np = np.concatenate(w_blocks, axis=0)
    offsets_np = np.concatenate([[0], np.cumsum(sizes)]).astype(np.uintp)

    def fit_batch(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        s_arr: np.ndarray,
        w_arr: np.ndarray,
    ):
        return gaussian_reml_fit_batched(
            torch.as_tensor(x_arr, dtype=_F64),
            torch.as_tensor(y_arr, dtype=_F64),
            torch.as_tensor(offsets_np),
            torch.as_tensor(s_arr, dtype=_F64),
            weights=torch.as_tensor(w_arr, dtype=_F64),
        )

    target_batch = 1
    x_t = _to_tensor(x_np, requires_grad=True)
    y_t = _to_tensor(y_np, requires_grad=True)
    s_t = _to_tensor(s_np, requires_grad=True)
    w_t = _to_tensor(w_np, requires_grad=True)

    out = gaussian_reml_fit_batched(
        x_t, y_t, torch.as_tensor(offsets_np), s_t, weights=w_t
    )
    out.edf[target_batch].backward()

    assert x_t.grad is not None
    assert y_t.grad is not None
    assert s_t.grad is not None
    assert w_t.grad is not None

    def edf_scalar(x_arr: np.ndarray, y_arr: np.ndarray, s_arr: np.ndarray, w_arr: np.ndarray) -> float:
        o = fit_batch(x_arr, y_arr, s_arr, w_arr)
        return float(o.edf.detach()[target_batch])

    block_start = int(offsets_np[target_batch])
    block_stop = int(offsets_np[target_batch + 1])
    rng2 = np.random.default_rng(55)
    x_idxs = [
        (int(rng2.integers(block_start, block_stop)), int(rng2.integers(0, p)))
        for _ in range(4)
    ]
    y_idxs = [
        (int(rng2.integers(block_start, block_stop)), int(rng2.integers(0, d)))
        for _ in range(4)
    ]
    w_idxs = [(int(rng2.integers(block_start, block_stop)),) for _ in range(3)]

    # Entry-wise FD for unconstrained inputs.
    unconstrained = (
        ("x", x_np, x_t.grad.detach().numpy(), x_idxs, lambda arr: edf_scalar(arr, y_np, s_np, w_np)),
        ("y", y_np, y_t.grad.detach().numpy(), y_idxs, lambda arr: edf_scalar(x_np, arr, s_np, w_np)),
        ("w", w_np, w_t.grad.detach().numpy(), w_idxs, lambda arr: edf_scalar(x_np, y_np, s_np, arr)),
    )
    for name, base, analytic, idxs, obj in unconstrained:
        fd = _fd_grad(base, idxs, obj)
        for idx, fd_val in fd.items():
            an_val = float(analytic[idx])
            assert _matches(an_val, fd_val), (
                f"batched {name}{idx}: analytic={an_val:.6e} fd={fd_val:.6e}"
            )

    # ``S`` is shared across all batch problems; project FD perturbations into
    # range(S) for the same reasons as the single-fit test.
    analytic_s = s_t.grad.detach().numpy()
    for direction in _sample_range_directions(s_np, rng2, k=5):
        fd_val = _fd_directional(s_np, direction, lambda arr: edf_scalar(x_np, y_np, arr, w_np))
        an_val = _analytic_directional(analytic_s, direction)
        assert _matches(an_val, fd_val), (
            f"batched s directional: analytic={an_val:.6e} fd={fd_val:.6e}"
        )
