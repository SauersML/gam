"""Regression tests for the position-API Gaussian-REML helpers (gam#580, #581).

#580 — Periodic Duchon was unreachable through ``gaussian_reml_fit_positions``:
  * a phantom ``0.875`` "support range" guard rejected every explicit ``period``
    (it compared the requested period against an internal knot extent, never the
    data range), and
  * ``period=None`` produced a non-PSD penalty (eigenvalue ``≈ −625``).

The root cause of the non-PSD penalty was that the 1D periodic case was routed
through the *mixed-periodicity* chord-embedding polyharmonic kernel
``φ(r) = c·r^{2m−d}``. That kernel is only CONDITIONALLY positive-definite on ℝ
and is genuinely indefinite under the chord metric on the circle, so its
periodised Gram carries large negative eigenvalues. The fix routes the 1D
periodic case to the Bernoulli Green's-function kernel
``φ(r) = (−1)^{m+1}·B_{2m}(r/P)`` — the actual Green's function of ``(d²/dx²)^m``
on the circle — which is PSD by construction (full rank modulo the constants).
These tests lock that the public API now (a) accepts every period, (b) yields a
PSD system, and (c) recovers a periodic truth.

#581 — ``gaussian_reml_fit_positions_batched`` rejected a torch tensor / default
int64 numpy ``row_offsets`` even though ``t``/``y`` accept torch. ``_index_vector``
now coerces any integer-valued input (torch, int64, list, integral float) to the
``uintp`` layout the FFI needs and rejects only genuinely non-integer input.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit
from gamfit._api import _index_vector


# ---------------------------------------------------------------------------
# #580 — periodic Duchon via the position API
# ---------------------------------------------------------------------------


def _circle_truth(n: int = 600, embed_dim: int = 16, seed: int = 1):
    """A clean closed curve: a circle in R^embed_dim sampled on a [0,1) grid."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)  # canonical half-open periodic grid
    theta = 2.0 * np.pi * t
    q, _ = np.linalg.qr(rng.standard_normal((embed_dim, 2)))
    y = np.stack([np.cos(theta), np.sin(theta)], axis=-1) @ q.T
    return t, y


def _min_penalty_eig(period: float | None) -> float:
    """Smallest eigenvalue of the periodic Duchon function-norm penalty."""
    centers = np.linspace(0.0, 1.0, 16, endpoint=False).reshape(-1, 1)
    penalty = gamfit.duchon_function_norm_penalty(
        centers, m=2, periodic_per_axis=(True,), period=period
    )
    penalty = np.asarray(penalty, dtype=float)
    # Symmetric Gram: use the symmetric eigensolver.
    eig = np.linalg.eigvalsh(0.5 * (penalty + penalty.T))
    return float(eig.min())


@pytest.mark.parametrize("period", [0.9983333333333334, 1.0, None])
def test_periodic_duchon_penalty_is_psd(period: float | None) -> None:
    """The periodic Duchon penalty must be PSD at every period (gam#580).

    The old chord-embedding kernel gave a dominant ``−0.70`` eigenvalue pair and
    ``period=None`` produced ``≈ −625``. The Bernoulli Green's function is PSD by
    construction, so the smallest eigenvalue is non-negative up to scale-relative
    floating noise.
    """
    min_eig = _min_penalty_eig(period)
    # Scale-relative tolerance: the penalty's largest eigenvalue sets the scale.
    centers = np.linspace(0.0, 1.0, 16, endpoint=False).reshape(-1, 1)
    penalty = np.asarray(
        gamfit.duchon_function_norm_penalty(
            centers, m=2, periodic_per_axis=(True,), period=period
        ),
        dtype=float,
    )
    scale = float(np.linalg.eigvalsh(0.5 * (penalty + penalty.T)).max())
    assert scale > 0.0, "periodic Duchon penalty must have a positive eigenvalue"
    assert min_eig >= -1e-9 * scale, (
        f"periodic Duchon penalty must be PSD (gam#580); min eig {min_eig:.3e} "
        f"vs scale {scale:.3e} for period={period}"
    )


@pytest.mark.parametrize("period", [0.9983333333333334, 1.0, None])
def test_periodic_duchon_fit_positions_runs_and_is_psd(period: float | None) -> None:
    """Every period must be ACCEPTED by ``gaussian_reml_fit_positions`` (gam#580).

    The old guard rejected every period with a phantom ``0.875`` support range and
    ``period=None`` tripped the solver's PSD check. The fit must now succeed, land
    a finite REML optimum, and report a PSD penalty spectrum in its cache.
    """
    t, y = _circle_truth()
    out = gamfit.gaussian_reml_fit_positions(
        t, y, basis="duchon", basis_order=2, periodic=True, period=period
    )
    assert out.get("status") == "ok", f"fit did not converge for period={period}: {out.get('status')}"
    assert np.all(np.isfinite(np.asarray(out["fitted"]))), "fitted values must be finite"
    eig = np.asarray(out["cache_penalty_eigenvalues"], dtype=float)
    assert eig.size > 0
    scale = float(np.abs(eig).max())
    assert eig.min() >= -1e-8 * scale, (
        f"REML penalty spectrum must be PSD (gam#580); min eig {eig.min():.3e} "
        f"for period={period}"
    )


@pytest.mark.parametrize("period", [1.0, None])
def test_periodic_duchon_recovers_periodic_truth(period: float | None) -> None:
    """A clean closed curve must be recovered with high fidelity (gam#580).

    R² of the fit against the noise-free circle embedding must be near 1; this
    asserts the Bernoulli periodic smoother fits real periodic structure, not just
    that the solver does not crash.
    """
    t, y = _circle_truth(n=600, embed_dim=16, seed=3)
    out = gamfit.gaussian_reml_fit_positions(
        t, y, basis="duchon", basis_order=2, periodic=True, period=period
    )
    assert out.get("status") == "ok"
    fitted = np.asarray(out["fitted"], dtype=float)
    resid = y - fitted
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - y.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.99, f"periodic Duchon must recover the circle (R²={r2:.4f}, period={period})"


def test_periodic_bspline_position_count_matches_cyclic_penalty() -> None:
    """Count-based periodic B-spline position fits use K columns and KxK penalty.

    ``knots_or_centers=K`` is the public position-API basis count. It must agree
    with ``periodic_spline_curve_basis(..., n_knots=K)`` instead of expanding as
    an open B-spline interior-knot count.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 40, endpoint=False)
    y = np.cos(2.0 * np.pi * t)[:, None] + 0.05 * rng.standard_normal((t.size, 1))
    _, helper_penalty = gamfit.periodic_spline_curve_basis(t, n_knots=12, degree=3)

    for penalty in (None, helper_penalty):
        out = gamfit.gaussian_reml_fit_positions(
            t,
            y,
            knots_or_centers=12,
            penalty=penalty,
            periodic=True,
            period=1.0,
        )
        assert out.get("status") == "ok"
        assert np.asarray(out["coefficients"]).shape == (12, 1)
        assert np.asarray(out["fitted"]).shape == y.shape
        assert np.asarray(out["penalty"]).shape == (12, 12)
        assert np.all(np.isfinite(np.asarray(out["fitted"])))


def test_periodic_bspline_position_fit_recovers_truth_issue_878() -> None:
    """gam#878: periodic B-spline position fits must size the internal basis to
    the public count K AND actually recover the periodic truth.

    The bug expanded ``knots_or_centers=K`` through the *open* B-spline auto-knot
    builder, giving an internal basis of size ``K + degree-wrap`` (e.g. 19 for
    K=12) that never matched the ``K x K`` penalty it validated against — so the
    helper rejected every penalty with ``penalty shape mismatch: expected 19x19,
    got 16x16``. A shape-only assertion is not enough to pin the fix: a basis
    whose size merely *happened* to agree could still mis-fit. This locks the
    count alignment (across several K, auto- and explicit-penalty, and against
    ``periodic_spline_curve_basis``) together with the recovered fit quality on a
    clean two-harmonic periodic signal.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 60, endpoint=False)
    truth = np.cos(2.0 * np.pi * t) + 0.5 * np.sin(4.0 * np.pi * t)
    y = truth + 0.05 * rng.standard_normal(t.size)

    for k in (8, 12, 16):
        _, helper_penalty = gamfit.periodic_spline_curve_basis(t, n_knots=k, degree=3)
        assert np.asarray(helper_penalty).shape == (k, k)
        for penalty, name in ((None, "auto"), (np.asarray(helper_penalty), "explicit")):
            out = gamfit.gaussian_reml_fit_positions(
                t,
                y,
                knots_or_centers=k,
                penalty=penalty,
                periodic=True,
                period=1.0,
            )
            assert out.get("status") == "ok"
            # Internal basis count == public K == cyclic penalty size: the
            # phantom open-knot expansion would have desynced these.
            assert np.asarray(out["coefficients"]).shape == (k, 1)
            assert np.asarray(out["penalty"]).shape == (k, k)
            fit = np.asarray(out["fitted"], dtype=float).ravel()
            ss_res = float(np.sum((y - fit) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot
            assert r2 > 0.97, (
                f"k={k}, {name} penalty: periodic B-spline position fit did not "
                f"recover the truth (R2={r2:.3f}); a size-aligned but ill-posed "
                "internal basis would surface here"
            )


def test_periodic_bspline_position_fit_default_count_issue_878() -> None:
    """gam#878 (second repro): the DEFAULT basis count (no ``knots_or_centers``)
    periodic B-spline position fit must also align its internal basis to its
    penalty and recover the truth — the reporter's ``expected 17x17, got 14x14``
    auto case, on irregularly spaced angles.
    """
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 1.0, 50))
    y = np.cos(2.0 * np.pi * t) + 0.05 * rng.standard_normal(t.size)

    out = gamfit.gaussian_reml_fit_positions(t, y, periodic=True, period=1.0)
    assert out.get("status") == "ok"
    penalty = np.asarray(out["penalty"])
    coeffs = np.asarray(out["coefficients"])
    assert penalty.ndim == 2 and penalty.shape[0] == penalty.shape[1]
    assert penalty.shape[0] == coeffs.shape[0], (
        "default-count periodic penalty size must match the internal basis count"
    )
    fit = np.asarray(out["fitted"], dtype=float).ravel()
    r2 = 1.0 - float(np.sum((y - fit) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    assert r2 > 0.97, f"default-count periodic fit did not recover the truth (R2={r2:.3f})"


def test_periodic_duchon_fit_wraps_at_seam() -> None:
    """The fitted periodic smooth must be continuous across the seam.

    Evaluate the fitted curve just inside each end of the period; a genuinely
    periodic basis predicts (near-)identical values, an open basis does not. The
    fit and the seam-probe design share explicit ``centers`` AND the same wrap
    period (the center span, which is what ``duchon_basis`` derives), so the two
    designs are guaranteed consistent.
    """
    t, y = _circle_truth(n=600, embed_dim=16, seed=5)
    centers = np.linspace(0.0, 1.0, 16, endpoint=False)
    # `duchon_basis` derives the wrap from the center span; match the fit to it.
    period = float(centers.max() - centers.min())
    out = gamfit.gaussian_reml_fit_positions(
        t, y, "duchon", centers, basis_order=2, periodic=True, period=period
    )
    assert out.get("status") == "ok"
    coeffs = np.asarray(out["coefficients"], dtype=float)
    left = centers.min()
    probes = np.array([left + 1e-6, left + period - 1e-6])
    design = np.asarray(
        gamfit.duchon_basis(probes, centers, m=2, periodic_per_axis=(True,)),
        dtype=float,
    )
    pred = design @ coeffs
    gap = float(np.max(np.abs(pred[0] - pred[1])))
    span = float(np.max(np.abs(pred)))
    assert gap < 1e-2 * max(span, 1.0), (
        f"periodic Duchon must wrap at the seam; gap={gap:.3e} (span={span:.3e})"
    )


# ---------------------------------------------------------------------------
# #581 — row_offsets coercion for the batched position API
# ---------------------------------------------------------------------------


def test_index_vector_accepts_integral_inputs() -> None:
    """``_index_vector`` coerces every integer-valued form to uintp (gam#581)."""
    n = 600
    forms = {
        "list": [0, n, 2 * n],
        "int64": np.array([0, n, 2 * n], dtype=np.int64),
        "uintp": np.array([0, n, 2 * n], dtype=np.uintp),
        "integral_float": np.array([0.0, float(n), float(2 * n)], dtype=np.float64),
    }
    for label, values in forms.items():
        arr = _index_vector(values, "row_offsets")
        assert arr.dtype == np.dtype(np.uintp), f"{label} must coerce to uintp"
        np.testing.assert_array_equal(arr, np.array([0, n, 2 * n], dtype=np.uintp))


def test_index_vector_accepts_torch_int_tensor() -> None:
    torch = pytest.importorskip("torch")
    n = 600
    arr = _index_vector(torch.tensor([0, n, 2 * n]), "row_offsets")
    assert arr.dtype == np.dtype(np.uintp)
    np.testing.assert_array_equal(arr, np.array([0, n, 2 * n], dtype=np.uintp))


def test_index_vector_rejects_noninteger() -> None:
    """Genuinely non-integer / negative input is rejected with a typed error."""
    with pytest.raises(TypeError):
        _index_vector(np.array([0.0, 1.5, 3.0]), "row_offsets")
    with pytest.raises(ValueError):
        _index_vector(np.array([0, -1, 2], dtype=np.int64), "row_offsets")


def test_batched_positions_accepts_torch_row_offsets() -> None:
    """``gaussian_reml_fit_positions_batched`` accepts torch ``row_offsets`` (gam#581).

    ``t``/``y`` already accept torch; ``row_offsets`` previously required a numpy
    uintp array. After the coercion fix every integer-valued form is accepted and
    produces the same fit as the explicit uintp array.
    """
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    n = 300
    rng = np.random.default_rng(7)
    t_np = np.concatenate([
        np.linspace(0.0, 1.0, n, endpoint=False),
        np.linspace(0.0, 1.0, n, endpoint=False),
    ])
    y_np = np.sin(2.0 * np.pi * t_np)[:, None] + 0.05 * rng.standard_normal((2 * n, 1))
    t = torch.tensor(t_np)
    y = torch.tensor(y_np)

    ref = gamfit.gaussian_reml_fit_positions_batched(
        t, y, np.array([0, n, 2 * n], dtype=np.uintp), basis="duchon", basis_order=2
    )
    for row_offsets in (
        torch.tensor([0, n, 2 * n]),
        np.array([0, n, 2 * n], dtype=np.int64),
        [0, n, 2 * n],
    ):
        out = gamfit.gaussian_reml_fit_positions_batched(
            t, y, row_offsets, basis="duchon", basis_order=2
        )
        assert np.asarray(out["fitted"]).shape == (2 * n, 1)
        np.testing.assert_allclose(
            np.asarray(out["fitted"]), np.asarray(ref["fitted"]), rtol=1e-10, atol=1e-10
        )


def test_batched_positions_rejects_fractional_row_offsets() -> None:
    """A genuinely fractional ``row_offsets`` is still rejected (gam#581)."""
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    n = 100
    t = torch.rand(2 * n)
    y = torch.randn(2 * n, 1)
    with pytest.raises((TypeError, ValueError)):
        gamfit.gaussian_reml_fit_positions_batched(
            t, y, np.array([0.0, 1.5 * n, 2.0 * n]), basis="duchon", basis_order=2
        )
