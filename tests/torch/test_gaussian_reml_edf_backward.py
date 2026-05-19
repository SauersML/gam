"""Analytic backward correctness for the Gaussian REML effective DOF (`edf`).

These tests exercise the `grad_edf` route through the closed-form Gaussian REML
analytic VJP pipeline: forward via the torch wrapper, then check that the
gradient of `out.edf` (and composite linear combinations involving it) matches
finite-difference references on every differentiable input.
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
_ATOL = 5e-7
_FD_STEP = 1e-5


def _make_penalty(p: int, rng: np.random.Generator) -> np.ndarray:
    """Build a positive-definite ``(p, p)`` penalty.

    A small diagonal ridge keeps the smallest eigenvalue well clear of the
    REML core's PSD tolerance so finite-difference perturbations (which break
    exact symmetry on individual entries) do not push the matrix into the
    ill-conditioned rejection band.
    """

    a = rng.standard_normal((p, p - 1))
    return (a @ a.T + 1e-2 * np.eye(p)).astype(np.float64)


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


def _fd_grad(
    base: np.ndarray,
    indices: list[tuple[int, ...]],
    objective,
    h: float = _FD_STEP,
) -> dict[tuple[int, ...], float]:
    grads: dict[tuple[int, ...], float] = {}
    for idx in indices:
        plus = base.copy()
        minus = base.copy()
        plus[idx] += h
        minus[idx] -= h
        grads[idx] = (objective(plus) - objective(minus)) / (2.0 * h)
    return grads


def _rel_err(analytic: float, fd: float) -> float:
    # Absolute tolerance floor: values that round-trip through the analytic
    # backward at the f64-noise floor (e.g. when both the FD reference and
    # the analytic land below ~1e-7) compare as exactly matching, since the
    # relative error of two numerically-zero quantities is meaningless.
    if abs(analytic) < 1e-7 and abs(fd) < 1e-7:
        return 0.0
    scale = max(abs(analytic), abs(fd), 1e-12)
    return abs(analytic - fd) / scale


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
    targets = {
        "x": (x_np, x.grad.detach().numpy(), _sample_indices(x_np.shape, rng, 5)),
        "y": (y_np, y.grad.detach().numpy(), _sample_indices(y_np.shape, rng, 5)),
        "s": (s_np, s.grad.detach().numpy(), _sample_indices(s_np.shape, rng, 5)),
        "w": (w_np, w.grad.detach().numpy(), _sample_indices(w_np.shape, rng, 4)),
    }

    for name, (base, analytic, idxs) in targets.items():
        def obj(arr: np.ndarray, name: str = name) -> float:
            if name == "x":
                return _fit_edf_value(arr, y_np, s_np, w_np)
            if name == "y":
                return _fit_edf_value(x_np, arr, s_np, w_np)
            if name == "s":
                return _fit_edf_value(x_np, y_np, arr, w_np)
            return _fit_edf_value(x_np, y_np, s_np, arr)

        fd = _fd_grad(base, idxs, obj)
        for idx, fd_val in fd.items():
            an_val = float(analytic[idx])
            err = _rel_err(an_val, fd_val)
            assert err < _RTOL, (
                f"{name}{idx}: analytic={an_val:.6e} fd={fd_val:.6e} "
                f"relerr={err:.2e}"
            )


def test_edf_composite_backward_matches_finite_difference() -> None:
    """grad_edf must compose correctly with other upstream gradients."""

    x_np, y_np, s_np, w_np = _problem(seed=7)

    coef_alpha = 1.0  # sum of coefficients
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
        err = _rel_err(an_val, fd_val)
        assert err < _RTOL, (
            f"composite x{idx}: analytic={an_val:.6e} fd={fd_val:.6e} "
            f"relerr={err:.2e}"
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
    offsets_np = np.concatenate(
        [[0], np.cumsum(sizes)]
    ).astype(np.uintp)

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

    # Forward + backward on a chosen batch element's edf.
    target_batch = 1
    x_t = _to_tensor(x_np, requires_grad=True)
    y_t = _to_tensor(y_np, requires_grad=True)
    s_t = _to_tensor(s_np, requires_grad=True)
    w_t = _to_tensor(w_np, requires_grad=True)
    out = fit_batch(x_t.detach().numpy(), y_t.detach().numpy(), s_t.detach().numpy(), w_t.detach().numpy())

    # Re-do forward with requires_grad tensors so autograd graph is alive.
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
    # Sample a few indices that lie inside the target batch's rows (otherwise
    # the FD gradient is exactly zero by construction — boring but still a
    # valid check; we focus on the active region).
    x_idxs = [(int(rng2.integers(block_start, block_stop)), int(rng2.integers(0, p))) for _ in range(4)]
    y_idxs = [(int(rng2.integers(block_start, block_stop)), int(rng2.integers(0, d))) for _ in range(4)]
    s_idxs = _sample_indices(s_np.shape, rng2, 4)
    w_idxs = [(int(rng2.integers(block_start, block_stop)),) for _ in range(3)]

    for name, base, idxs in (
        ("x", x_np, x_idxs),
        ("y", y_np, y_idxs),
        ("s", s_np, s_idxs),
        ("w", w_np, w_idxs),
    ):
        analytic_grad = {
            "x": x_t.grad.detach().numpy(),
            "y": y_t.grad.detach().numpy(),
            "s": s_t.grad.detach().numpy(),
            "w": w_t.grad.detach().numpy(),
        }[name]

        def obj(arr: np.ndarray, name: str = name) -> float:
            if name == "x":
                return edf_scalar(arr, y_np, s_np, w_np)
            if name == "y":
                return edf_scalar(x_np, arr, s_np, w_np)
            if name == "s":
                return edf_scalar(x_np, y_np, arr, w_np)
            return edf_scalar(x_np, y_np, s_np, arr)

        fd = _fd_grad(base, idxs, obj)
        for idx, fd_val in fd.items():
            an_val = float(analytic_grad[idx])
            # When the perturbed entry sits outside the target batch's block,
            # both analytic and FD should be zero — accept either way.
            if abs(fd_val) < _ATOL and abs(an_val) < _ATOL:
                continue
            err = _rel_err(an_val, fd_val)
            assert err < _RTOL, (
                f"batched {name}{idx}: analytic={an_val:.6e} fd={fd_val:.6e} "
                f"relerr={err:.2e}"
            )
